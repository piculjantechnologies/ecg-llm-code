"""Shared helpers for the analysis scripts (offline; no API calls).

The scripts read the run folders (runs/, git-ignored), the age-and-sex control (controls/)
and the PTB-XL metadata, and write LaTeX macros to paper/generated/ and JSON to analysis/.
Evaluation uses the frozen ecgcode.evaluate functions and the bootstrap settings in the frozen
config (2,000 resamples, seed 0).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ecgcode.config import load_config, project_path  # noqa: E402
from ecgcode.data import LABELS, TRAIN_FOLDS, VAL_FOLD, TEST_FOLD, benchmark_split, load_metadata  # noqa: E402
from ecgcode.evaluate import (PROB_COLS, bootstrap_indices, load_predictions, macro_auroc,  # noqa: E402
                              truth)

RUNS = ROOT / "runs"
GEN = ROOT / "paper" / "generated"
OUT_JSON = ROOT / "analysis"

TEST = RUNS / "test_v2"
AGESEX_TEST = ROOT / "controls" / "agesex_test"   # written by analysis/agesex_control.py
AGESEX_DEV = ROOT / "controls" / "agesex_dev"

CFG = load_config()
N_BOOT, BOOT_SEED = CFG["eval"]["n_bootstrap"], CFG["eval"]["bootstrap_seed"]

_META = None


def meta():
    global _META
    if _META is None:
        _META = load_metadata(project_path(CFG["data"]["ptbxl_dir"]))
    return _META


def preds(*run_dirs):
    """Predictions of one run, or of several runs over disjoint records concatenated."""
    frames = [load_predictions(r) for r in run_dirs]
    out = pd.concat(frames)
    if out.index.duplicated().any():
        raise ValueError(f"runs {run_dirs} overlap")
    return out


def results_lines(run_dir):
    with open(Path(run_dir) / "results.jsonl") as f:
        return [json.loads(line) for line in f]


def key(run_dir):
    """anonymous id -> ecg_id"""
    return {k: int(v) for k, v in json.loads((Path(run_dir) / "key.json").read_text()).items()}


def custom_to_anon(run_dir):
    out = {}
    with open(Path(run_dir) / "requests.jsonl") as f:
        for line in f:
            r = json.loads(line)
            out[r["custom_id"]] = r["record_ids"][0]
    return out


def auc(y, p):
    """AUROC by the Mann-Whitney rank statistic (ties count one half), equal to sklearn's
    roc_auc_score for a binary label; used inside bootstrap loops because it is much faster."""
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fast_macro(y, p):
    return float(np.mean([auc(y[:, j], p[:, j]) for j in range(y.shape[1])]))


def boots_for(ids):
    return bootstrap_indices(truth(meta(), ids), N_BOOT, BOOT_SEED)


def ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def summary(pred, ids=None, boots=None):
    """Macro AUROC with CI, macro AUPRC and per-class AUROC with CIs on the given records."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    ids = sorted(pred.index) if ids is None else list(ids)
    y = truth(meta(), ids)
    p = pred.loc[ids, PROB_COLS].to_numpy(float)
    boots = boots if boots is not None else boots_for(ids)
    per = {l: np.array([auc(y[b, j], p[b, j]) for b in boots]) for j, l in enumerate(LABELS)}
    macro = np.mean([per[l] for l in LABELS], axis=0)
    return {
        "n": len(ids), "macro_auroc": macro_auroc(y, p), "macro_auroc_ci95": ci(macro),
        "macro_auprc": float(average_precision_score(y, p, average="macro")),
        "per_class_auroc": {l: float(roc_auc_score(y[:, j], p[:, j])) for j, l in enumerate(LABELS)},
        "per_class_ci95": {l: ci(per[l]) for l in LABELS},
        "per_class_auprc": {l: float(average_precision_score(y[:, j], p[:, j])) for j, l in enumerate(LABELS)},
        "positives": {l: int(y[:, j].sum()) for j, l in enumerate(LABELS)},
    }


def paired(pa, pb, ids, boots=None):
    """Macro and per-class AUROC of a minus b on the same records, paired bootstrap CIs (the same
    computation as ecgcode.evaluate.paired_difference, on prediction frames)."""
    from sklearn.metrics import roc_auc_score
    ids = list(ids)
    y = truth(meta(), ids)
    a, b = pa.loc[ids, PROB_COLS].to_numpy(float), pb.loc[ids, PROB_COLS].to_numpy(float)
    boots = boots if boots is not None else boots_for(ids)
    per, dsum = {}, np.zeros(len(boots))
    for j, l in enumerate(LABELS):
        dj = np.array([auc(y[i, j], a[i, j]) - auc(y[i, j], b[i, j]) for i in boots])
        dsum += dj
        per[l] = {"diff": float(roc_auc_score(y[:, j], a[:, j]) - roc_auc_score(y[:, j], b[:, j])),
                  "ci95": ci(dj)}
    d = dsum / len(LABELS)
    return {"n": len(ids), "a": macro_auroc(y, a), "b": macro_auroc(y, b),
            "diff": macro_auroc(y, a) - macro_auroc(y, b), "ci95": ci(d), "per_class": per}


# --- LaTeX macros ------------------------------------------------------------------

def f3(x):
    return f"{x:.3f}"


def s3(x):
    """Signed, three decimals, for math mode (use as $\\Macro$): 0.2 -> '+0.200', -0.05 -> '-0.050', 0 -> '0.000'."""
    v = round(x, 3)
    return f"{'+' if v > 0 else ('-' if v < 0 else '')}{abs(v):.3f}" if v != 0 else "0.000"


def f2(x):
    return f"{x:.2f}"


def thousands(n):
    return f"{int(n):,}".replace(",", "{,}")


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}"


def long_date(d):
    """27 September 2026"""
    return f"{d.day} {d:%B %Y}"


def write_macros(path, macros, header):
    lines = [f"% {header}", "% Generated; do not edit by hand."]
    for name, value in macros.items():
        if not name.isalpha():
            raise ValueError(f"macro name {name!r} must be letters only")
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def write_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
