"""Metrics with the benchmark's definition (sklearn macro AUROC over the 5 superclasses)
plus percentile-bootstrap confidence intervals and paired comparisons between runs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import project_path
from .data import LABELS, label_matrix, load_metadata

PROB_COLS = [f"p_{label}" for label in LABELS]


def load_predictions(run_dir):
    return pd.read_csv(Path(run_dir) / "predictions.csv").set_index("ecg_id")


def truth(meta, ecg_ids):
    return label_matrix(meta.loc[list(ecg_ids)])


def bootstrap_indices(y, n_boot, seed, max_tries_factor=50):
    """Resamples in which every class has at least one positive and one negative."""
    absent = [LABELS[j] for j in range(y.shape[1]) if not 0 < y[:, j].sum() < len(y)]
    if absent:
        raise ValueError(f"AUROC is undefined: class(es) {absent} have no positives or no negatives among "
                         f"these {len(y)} records (use a larger sample)")
    rng = np.random.default_rng(seed)
    out, tries = [], 0
    while len(out) < n_boot:
        tries += 1
        if tries > max_tries_factor * n_boot:
            raise ValueError("too few positives for a stable bootstrap; use a larger sample")
        idx = rng.integers(0, len(y), len(y))
        if all(0 < y[idx, j].sum() < len(idx) for j in range(y.shape[1])):
            out.append(idx)
    return out


def macro_auroc(y, p):
    return float(roc_auc_score(y, p, average="macro"))


def summarize(y, p, boots):
    values = np.array([macro_auroc(y[b], p[b]) for b in boots])
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {
        "n": int(len(y)),
        "macro_auroc": macro_auroc(y, p),
        "macro_auroc_ci95": [float(lo), float(hi)],
        "macro_auprc": float(average_precision_score(y, p, average="macro")),
        "per_class_auroc": {l: float(roc_auc_score(y[:, j], p[:, j])) for j, l in enumerate(LABELS)},
        "positives": {l: int(y[:, j].sum()) for j, l in enumerate(LABELS)},
    }


def evaluate_run(run_dir, cfg, exclude_ids=None, meta=None):
    meta = meta if meta is not None else load_metadata(project_path(cfg["data"]["ptbxl_dir"]))
    pred = load_predictions(run_dir)
    if exclude_ids:
        pred = pred.drop(index=[i for i in exclude_ids if i in pred.index])
    y = truth(meta, pred.index)
    p = pred[PROB_COLS].to_numpy(float)
    boots = bootstrap_indices(y, cfg["eval"]["n_bootstrap"], cfg["eval"]["bootstrap_seed"])
    result = summarize(y, p, boots)
    if "source" in pred:
        result["records_prior_filled"] = int((pred["source"] != "model").sum())
    return result


def paired_difference(run_a, run_b, cfg, meta=None, exclude_ids=None, allow_subset=False):
    """Macro AUROC of run_a minus run_b on the same records, with a paired bootstrap CI.

    Both runs must cover the same records (as ablations built with the same --n and
    --sample-seed do); pass allow_subset=True to compare on the overlap deliberately.
    """
    meta = meta if meta is not None else load_metadata(project_path(cfg["data"]["ptbxl_dir"]))
    a, b = load_predictions(run_a), load_predictions(run_b)
    if set(a.index) != set(b.index) and not allow_subset:
        raise ValueError(f"{run_a} and {run_b} cover different records ({len(a)} vs {len(b)}, "
                         f"{len(set(a.index) & set(b.index))} shared); rebuild with the same sample")
    ids = sorted(set(a.index) & set(b.index) - set(exclude_ids or []))
    y = truth(meta, ids)
    pa, pb = a.loc[ids, PROB_COLS].to_numpy(float), b.loc[ids, PROB_COLS].to_numpy(float)
    boots = bootstrap_indices(y, cfg["eval"]["n_bootstrap"], cfg["eval"]["bootstrap_seed"])
    diffs = np.array([macro_auroc(y[i], pa[i]) - macro_auroc(y[i], pb[i]) for i in boots])
    per_class = {l: float(roc_auc_score(y[:, j], pa[:, j]) - roc_auc_score(y[:, j], pb[:, j]))
                 for j, l in enumerate(LABELS)}
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"n": len(ids), "a": str(run_a), "b": str(run_b),
            "macro_auroc_a": macro_auroc(y, pa), "macro_auroc_b": macro_auroc(y, pb),
            "diff": macro_auroc(y, pa) - macro_auroc(y, pb), "diff_ci95": [float(lo), float(hi)],
            "per_class_diff": per_class}


def repeats_summary(run_dirs, cfg, meta=None):
    """Run-to-run variation across repeats of the same run, plus the probability-averaged ensemble."""
    meta = meta if meta is not None else load_metadata(project_path(cfg["data"]["ptbxl_dir"]))
    preds = [load_predictions(r) for r in run_dirs]
    if len({frozenset(p.index) for p in preds}) != 1:
        raise ValueError("repeat runs cover different records; build them on the same split and sample")
    ids = sorted(preds[0].index)
    y = truth(meta, ids)
    per_run = [macro_auroc(y, p.loc[ids, PROB_COLS].to_numpy(float)) for p in preds]
    # Ensemble: average only real model answers; a record keeps its prior fill only if no run answered it.
    stack = np.stack([p.loc[ids, PROB_COLS].to_numpy(float) for p in preds])
    answered = np.stack([(p.loc[ids, "source"] == "model").to_numpy() if "source" in p else np.ones(len(ids), bool)
                         for p in preds])
    weights = answered[:, :, None].astype(float)
    mean_probs = np.where(weights.sum(0) > 0, (stack * weights).sum(0) / np.maximum(weights.sum(0), 1), stack[0])
    boots = bootstrap_indices(y, cfg["eval"]["n_bootstrap"], cfg["eval"]["bootstrap_seed"])
    return {"runs": [str(r) for r in run_dirs], "n": len(ids), "per_run_macro_auroc": per_run,
            "mean": float(np.mean(per_run)), "sd": float(np.std(per_run, ddof=1)) if len(per_run) > 1 else None,
            "ensemble": summarize(y, mean_probs, boots)}


def load_benchmark_table(path=None):
    path = path or project_path("benchmarks/ptbxl_superdiagnostic_published.csv")
    return pd.read_csv(path, comment="#", dtype={"reported_uncertainty": str})


def write_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2))


def macro_f1_with_dev_thresholds(test_run, dev_run, meta):
    """Macro-F1 on the test run with per-class thresholds chosen to maximize F1 on a dev run
    (needed to compare with work that reports F1 instead of AUROC)."""
    from sklearn.metrics import f1_score, precision_recall_curve

    dev, test = load_predictions(dev_run), load_predictions(test_run)
    if "source" in dev:
        dev = dev[dev["source"] == "model"]  # tune on real answers only
    y_dev, y_test = truth(meta, dev.index), truth(meta, test.index)
    thresholds, per_class = {}, {}
    for j, (label, col) in enumerate(zip(LABELS, PROB_COLS)):
        precision, recall, cuts = precision_recall_curve(y_dev[:, j], dev[col])
        f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
        thresholds[label] = float(cuts[int(np.argmax(f1))])
        per_class[label] = float(f1_score(y_test[:, j], test[col] >= thresholds[label]))
    return {"macro_f1": float(np.mean(list(per_class.values()))), "per_class_f1": per_class,
            "thresholds_from": str(dev_run), "thresholds": thresholds}
