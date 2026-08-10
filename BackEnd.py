import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Union
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

MODEL_DIR = 'models/'
LABEL_NAMES = ['I/E', 'N/S', 'F/T', 'P/J']
MODEL_NAMES = ['KNN', '逻辑回归', '朴素贝叶斯', '决策树']
TYPE_LETTERS = {'I/E': ('I', 'E'), 'N/S': ('N', 'S'), 'F/T': ('F', 'T'), 'P/J': ('P', 'J')}


def _safe_label(label: str) -> str:
    return label.replace('/', '_')


def train_all_models(X, y_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=4),
        '逻辑回归': LogisticRegression(C=0.01, solver='lbfgs', max_iter=1000, random_state=666),
        '朴素贝叶斯': GaussianNB(),
        '决策树': DecisionTreeClassifier(max_depth=3, random_state=666),
    }
    feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
    joblib.dump(feature_names, f'{save_dir}/feature_names.pkl')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, f'{save_dir}/scaler.pkl')

    cv_results = pd.DataFrame()
    model_info = {}
    tag = os.path.basename(save_dir.rstrip('/'))

    for label, y in y_dict.items():
        print(f"\n{'='*5}\n[{tag}] 训练标签: {label}\n{'='*5}")
        model_info[label] = {}
        for model_name, model in models.items():
            print(f"训练 {model_name}...")
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
            mean_score, std_score = cv_scores.mean(), cv_scores.std()
            print(f"  CV准确率: {mean_score:.4f} (±{std_score:.4f})")
            cv_results.loc[model_name, f'{label}_acc'] = mean_score
            cv_results.loc[model_name, f'{label}_std'] = std_score
            model.fit(X_scaled, y)
            joblib.dump(model, f'{save_dir}/{model_name}_{_safe_label(label)}.pkl')
            model_info[label][model_name] = {'cv_mean': mean_score, 'cv_std': std_score}

    cv_results.to_csv(f'{save_dir}/cv_scores.csv')
    joblib.dump(model_info, f'{save_dir}/model_info.pkl')
    print(f"\n[{tag}] 训练完成\n{cv_results}")
    return cv_results, model_info


class _ModelSet:
    """单个模型集（双亲 or 单亲）"""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.models: Dict[str, Dict] = {}
        self.weights: Dict[str, Dict] = {}
        self.feature_names = None
        self.scaler = None
        self._load()

    def _load(self):
        print(f"加载模型: {self.model_dir}")
        self.feature_names = joblib.load(f'{self.model_dir}/feature_names.pkl')
        self.scaler = joblib.load(f'{self.model_dir}/scaler.pkl')
        cv_scores = pd.read_csv(f'{self.model_dir}/cv_scores.csv', index_col=0)
        for label in LABEL_NAMES:
            self.models[label], self.weights[label] = {}, {}
            for model_name in MODEL_NAMES:
                path = f'{self.model_dir}/{model_name}_{_safe_label(label)}.pkl'
                if os.path.exists(path):
                    self.models[label][model_name] = joblib.load(path)
                    w = float(cv_scores.loc[model_name, f'{label}_acc'])
                    self.weights[label][model_name] = w
                    print(f"  {model_name} - {label}: 权重={w:.4f}")
                else:
                    print(f"  警告: {path} 不存在")
            total = sum(self.weights[label].values())
            if total > 0:
                for m in self.weights[label]:
                    self.weights[label][m] /= total

    def preprocess_input(self, user_data: Union[List, np.ndarray, Dict]) -> np.ndarray:
        if isinstance(user_data, dict):
            data = [user_data.get(f'q{i}', 0) for i in range(1, len(self.feature_names) + 1)]
            X = np.array(data, dtype=float).reshape(1, -1)
        elif isinstance(user_data, list):
            X = np.array(user_data, dtype=float).reshape(1, -1)
        else:
            X = user_data.reshape(1, -1) if len(user_data.shape) == 1 else user_data
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"特征数量不匹配: 期望 {len(self.feature_names)}, 得到 {X.shape[1]}")
        return self.scaler.transform(X)

    def predict_label(self, X_scaled: np.ndarray, label: str) -> Dict:
        probas, weights = [], []
        for model_name, model in self.models[label].items():
            proba = model.predict_proba(X_scaled)
            if proba.shape[1] == 1:
                proba = np.hstack([1 - proba, proba])
            probas.append(float(proba[0, 1]))
            weights.append(self.weights[label][model_name])
        weights = np.array(weights)
        weighted_prob = float(np.sum(np.array(probas) * weights))
        pred = int(weighted_prob >= 0.5)
        names = list(self.models[label].keys())
        return {
            'prediction': pred,
            'probability': weighted_prob,
            'all_probas': {names[i]: probas[i] for i in range(len(names))},
            'weights': {names[i]: float(weights[i]) for i in range(len(names))},
        }

    def predict_all(self, user_data: Union[List, np.ndarray, Dict]) -> Dict:
        X_scaled = self.preprocess_input(user_data)
        return {label: self.predict_label(X_scaled, label) for label in LABEL_NAMES}


class MBTIPredictor:
    """MBTI 评估预测器：按是否单亲路由到双亲/单亲模型"""

    def __init__(self, model_dir=MODEL_DIR):
        self.two_parent = _ModelSet(f'{model_dir}two_parent')
        self.single_parent = _ModelSet(f'{model_dir}single_parent')

    def get_formatted_result(self, user_data, single_parent=False) -> Dict:
        ms = self.single_parent if single_parent else self.two_parent
        results = ms.predict_all(user_data)
        formatted = {'status': 'success', 'predictions': {}, 'detailed': {}}
        for label, res in results.items():
            letters = TYPE_LETTERS[label]
            prob = res['probability']
            formatted['predictions'][label] = {
                'label': res['prediction'],
                'type': letters[res['prediction']],
                'probability': prob,
                'confidence': round(max(prob, 1 - prob), 4),
            }
            formatted['detailed'][label] = {
                'ensemble_prediction': res['prediction'],
                'ensemble_probability': prob,
                'model_predictions': res['all_probas'],
                'model_weights': res['weights'],
            }
        formatted['mbti_type'] = ''.join(formatted['predictions'][l]['type'] for l in LABEL_NAMES)
        formatted['confidence'] = round(
            float(np.mean([formatted['predictions'][l]['confidence'] for l in LABEL_NAMES])), 4)
        formatted['model'] = 'single_parent' if single_parent else 'two_parent'
        return formatted


# 双亲模型: 仅 是否单亲==0 样本, 保留监A+监B, 删除是否单亲列 (19特征)
def build_two_parent_xy(data):
    d = data[data['是否单亲'] == 0].drop(columns=['收入水平', '自我认可度', '是否单亲'])
    X = d.drop(columns=['道德品行', 'I/E', 'N/S', 'F/T', 'P/J'])
    y = d.loc[:, 'I/E':'P/J'].copy()
    return X, y


# 单亲模型: 全部样本, 删除监B, 保留是否单亲列 (18特征)
def build_single_parent_xy(data):
    d = data.drop(columns=['收入水平', '自我认可度', '监B：I/E', '监B：F/T'])
    X = d.drop(columns=['道德品行', 'I/E', 'N/S', 'F/T', 'P/J'])
    y = d.loc[:, 'I/E':'P/J'].copy()
    return X, y


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
    if mode == 'train':
        data = pd.read_csv('MBTI.csv')
        X2, y2 = build_two_parent_xy(data)
        train_all_models(X2, y2, f'{MODEL_DIR}two_parent')
        X1, y1 = build_single_parent_xy(data)
        train_all_models(X1, y1, f'{MODEL_DIR}single_parent')
    elif mode == 'predict':
        p = MBTIPredictor(MODEL_DIR)
        for sp, name in [(False, '双亲'), (True, '单亲')]:
            ms = p.single_parent if sp else p.two_parent
            feat = [int(np.random.randint(0, 4)) for _ in range(len(ms.feature_names))]
            r = p.get_formatted_result(feat, single_parent=sp)
            print(f"\n[{name}] MBTI: {r['mbti_type']}  置信度: {r['confidence']:.4f}  特征数: {len(ms.feature_names)}")
    else:
        print("用法: python BackEnd.py [train|predict]")
