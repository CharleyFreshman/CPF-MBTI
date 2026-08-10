import joblib
import json
import numpy as np
import pandas as pd

MODEL_DIR = 'models/'
SETS = [('two_parent', f'{MODEL_DIR}two_parent'), ('single_parent', f'{MODEL_DIR}single_parent')]
MODEL_NAMES = ['KNN', '逻辑回归', '朴素贝叶斯', '决策树']
LABELS = ['I/E', 'N/S', 'F/T', 'P/J']


def safe(label):
    return label.replace('/', '_')


def tree_to_dict(tree):
    t = tree.tree_
    val = np.asarray(t.value)
    if val.ndim == 3:
        val = val[:, 0, :]
    return {
        'children_left': t.children_left.tolist(),
        'children_right': t.children_right.tolist(),
        'feature': t.feature.tolist(),
        'threshold': t.threshold.tolist(),
        'value': val.tolist(),
    }


out = {}
for set_name, sdir in SETS:
    scaler = joblib.load(f'{sdir}/scaler.pkl')
    feature_names = joblib.load(f'{sdir}/feature_names.pkl')
    cv = pd.read_csv(f'{sdir}/cv_scores.csv', index_col=0)
    sd = {
        'feature_names': list(feature_names),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'labels': LABELS,
        'models': {},
    }
    for label in LABELS:
        w = {m: float(cv.loc[m, f'{label}_acc']) for m in MODEL_NAMES}
        tot = sum(w.values())
        w = {m: w[m] / tot for m in w}
        sd['models'][label] = {'weights': w}
        for m in MODEL_NAMES:
            mdl = joblib.load(f'{sdir}/{m}_{safe(label)}.pkl')
            if m == 'KNN':
                e = {'type': 'knn', 'X': np.asarray(mdl._fit_X).tolist(),
                     'y': [int(v) for v in np.asarray(mdl._y).tolist()], 'k': int(mdl.n_neighbors)}
            elif m == '逻辑回归':
                e = {'type': 'lr', 'coef': mdl.coef_[0].tolist(), 'intercept': float(mdl.intercept_[0])}
            elif m == '朴素贝叶斯':
                var = getattr(mdl, 'var_', None)
                if var is None:
                    var = np.asarray(mdl.sigma_) ** 2
                e = {'type': 'nb', 'theta': mdl.theta_.tolist(),
                     'var': np.asarray(var).tolist(), 'priors': mdl.class_prior_.tolist()}
            else:  # 决策树
                e = {'type': 'tree', 'tree': tree_to_dict(mdl)}
            sd['models'][label][m] = e
    out[set_name] = sd

with open('mbti_models.js', 'w', encoding='utf-8') as f:
    f.write('window.MBTI_MODELS = ')
    json.dump(out, f, ensure_ascii=False)
    f.write(';\n')

print('exported. features:', {k: len(v['feature_names']) for k, v in out.items()})
print('two_parent KNN samples:', len(out['two_parent']['models']['I/E']['KNN']['X']))
print('single_parent KNN samples:', len(out['single_parent']['models']['I/E']['KNN']['X']))
