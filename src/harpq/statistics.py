from collections import defaultdict
from math import sqrt
import numpy as np
from scipy.stats import t


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid p values")
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    result = np.empty_like(p)
    result[order] = adjusted
    return result.tolist()


def mean_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or not 0 < confidence < 1:
        raise ValueError("At least two finite independent units are required")
    n = len(values)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    se = sd / sqrt(n)
    half = float(t.ppf((1 + confidence) / 2, n - 1)) * se
    p = float(2 * t.sf(abs(mean / se), n - 1)) if se else (1.0 if mean == 0 else 0.0)
    return {"n": n, "mean": mean, "sd": sd, "se": se, "df": n - 1, "ci": [mean - half, mean + half], "p": p}


def paired_cluster_summary(first, second, outcome, record_key="record", cluster_key="cluster", replicates=10000, seed=20261021):
    def indexed(rows):
        result = {}
        for row in rows:
            key = row[record_key]
            if key in result:
                raise ValueError("Duplicate paired-record key")
            result[key] = row
        return result
    a, b = indexed(first), indexed(second)
    if set(a) != set(b) or not a:
        raise ValueError("Paired record sets must match exactly")
    groups = defaultdict(list)
    for key in a:
        if a[key][cluster_key] != b[key][cluster_key]:
            raise ValueError("Cluster membership differs across conditions")
        value = float(a[key][outcome]) - float(b[key][outcome])
        if not np.isfinite(value):
            raise ValueError("Nonfinite paired outcome")
        groups[a[key][cluster_key]].append(value)
    means = np.asarray([np.mean(v) for _, v in sorted(groups.items(), key=lambda item: str(item[0]))])
    result = mean_interval(means)
    if type(replicates) is not int or replicates < 1:
        raise ValueError("Replicate count must be positive")
    rng = np.random.default_rng(seed)
    sampled = means[rng.integers(0, len(means), (replicates, len(means)))].mean(axis=1)
    return {"n_pairs": len(a), "n_clusters": len(groups), "estimand": "equally weighted mean of cluster-specific paired means",
        "team_statistics": result, "bootstrap_ci": np.quantile(sampled, [0.025, 0.975]).tolist(),
        "bootstrap_replicates": replicates, "seed": seed}
