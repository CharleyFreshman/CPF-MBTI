(function () {
    const TYPE_LETTERS = { 'I/E': ['I', 'E'], 'N/S': ['N', 'S'], 'F/T': ['F', 'T'], 'P/J': ['P', 'J'] };

    function scaleInput(raw, mean, scale) {
        return raw.map((v, i) => (v - mean[i]) / (scale[i] || 1));
    }

    function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

    // KNN (Euclidean, uniform weights): proba(1) = 正类邻居数 / k
    function knnProba(X, y, k, xs) {
        const d = X.map((row, i) => {
            let s = 0;
            for (let j = 0; j < row.length; j++) { const t = row[j] - xs[j]; s += t * t; }
            return [Math.sqrt(s), y[i]];
        });
        d.sort((a, b) => a[0] - b[0]);
        let c1 = 0;
        for (let i = 0; i < k; i++) if (d[i][1] === 1) c1++;
        return c1 / k;
    }

    function lrProba(coef, intercept, xs) {
        let z = intercept;
        for (let j = 0; j < coef.length; j++) z += coef[j] * xs[j];
        return sigmoid(z);
    }

    // GaussianNB: var_ 已含 sklearn 的 var_smoothing
    function nbProba(theta, varr, priors, xs) {
        const logp = [0, 0];
        for (let c = 0; c < 2; c++) {
            let s = Math.log(priors[c]);
            for (let j = 0; j < xs.length; j++) {
                const v = varr[c][j];
                s += -0.5 * Math.log(2 * Math.PI * v) - (xs[j] - theta[c][j]) ** 2 / (2 * v);
            }
            logp[c] = s;
        }
        const m = Math.max(logp[0], logp[1]);
        const e0 = Math.exp(logp[0] - m), e1 = Math.exp(logp[1] - m);
        return e1 / (e0 + e1);
    }

    function treeProba(tree, xs) {
        let node = 0;
        while (tree.children_left[node] !== -1) {
            const f = tree.feature[node], t = tree.threshold[node];
            node = (xs[f] <= t) ? tree.children_left[node] : tree.children_right[node];
        }
        const val = tree.value[node];
        const tot = val[0] + val[1];
        return tot === 0 ? 0.5 : val[1] / tot;
    }

    function modelProba(entry, xs) {
        switch (entry.type) {
            case 'knn': return knnProba(entry.X, entry.y, entry.k, xs);
            case 'lr': return lrProba(entry.coef, entry.intercept, xs);
            case 'nb': return nbProba(entry.theta, entry.var, entry.priors, xs);
            case 'tree': return treeProba(entry.tree, xs);
        }
        return 0;
    }

    function predictLabel(setModels, label, xs) {
        const lm = setModels[label];
        const names = Object.keys(lm.weights);
        let prob = 0;
        const all = {}, wts = {};
        for (const m of names) {
            const p = modelProba(lm[m], xs);
            const w = lm.weights[m];
            prob += p * w;
            all[m] = p; wts[m] = w;
        }
        return { prediction: prob >= 0.5 ? 1 : 0, probability: prob, all_probas: all, weights: wts };
    }

    window.predictMBTI = function (rawFeatures, singleParent) {
        const set = singleParent ? MBTI_MODELS.single_parent : MBTI_MODELS.two_parent;
        const xs = scaleInput(rawFeatures, set.scaler_mean, set.scaler_scale);
        const predictions = {}, detailed = {};
        for (const label of set.labels) {
            const r = predictLabel(set.models, label, xs);
            const letters = TYPE_LETTERS[label];
            const prob = r.probability;
            predictions[label] = {
                label: r.prediction,
                type: letters[r.prediction],
                probability: prob,
                confidence: Math.round(Math.max(prob, 1 - prob) * 10000) / 10000,
            };
            detailed[label] = {
                ensemble_prediction: r.prediction,
                ensemble_probability: prob,
                model_predictions: r.all_probas,
                model_weights: r.weights,
            };
        }
        const mbti_type = set.labels.map(l => predictions[l].type).join('');
        const conf = set.labels.reduce((s, l) => s + predictions[l].confidence, 0) / set.labels.length;
        return {
            status: 'success',
            mbti_type,
            confidence: Math.round(conf * 10000) / 10000,
            model: singleParent ? 'single_parent' : 'two_parent',
            predictions,
            detailed,
        };
    };
})();
