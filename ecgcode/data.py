"""PTB-XL loading with the label and split definitions of the public benchmark.

Matches Strodthoff et al. (2021), github.com/helme/ecg_ptbxl_benchmarking:
- superdiagnostic labels: every SCP code in `scp_codes` (any likelihood) whose
  scp_statements row has diagnostic == 1 contributes its diagnostic_class;
- records without any superdiagnostic label are dropped;
- folds 1-8 train, fold 9 validation, fold 10 test.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

LABELS = ("NORM", "MI", "STTC", "CD", "HYP")
TRAIN_FOLDS = tuple(range(1, 9))
VAL_FOLD = 9
TEST_FOLD = 10


def resolve_ptbxl_dir(path):
    """Accept the dataset folder itself or its parent (PhysioNet zips nest a version folder)."""
    path = Path(path)
    if (path / "ptbxl_database.csv").exists():
        return path
    for child in sorted(path.glob("*")):
        if (child / "ptbxl_database.csv").exists():
            return child
    raise FileNotFoundError(f"ptbxl_database.csv not found under {path}")


def load_metadata(ptbxl_dir):
    """DataFrame indexed by ecg_id with age, sex, strat_fold, filenames, labels and y_<LABEL>."""
    root = resolve_ptbxl_dir(ptbxl_dir)
    meta = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval)
    statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
    diagnostic = statements[statements["diagnostic"] == 1.0]["diagnostic_class"].dropna()

    def aggregate(codes):
        return sorted({diagnostic[c] for c in codes if c in diagnostic.index})

    meta["labels"] = meta["scp_codes"].apply(aggregate)
    for label in LABELS:
        meta[f"y_{label}"] = meta["labels"].apply(lambda ls, lab=label: int(lab in ls))
    meta.attrs["root"] = str(root)
    return meta


def benchmark_split(meta, folds):
    """Records in the given folds that have at least one superdiagnostic label."""
    folds = [folds] if isinstance(folds, int) else list(folds)
    return meta[meta["strat_fold"].isin(folds) & (meta["labels"].str.len() > 0)]


def label_matrix(meta):
    return meta[[f"y_{label}" for label in LABELS]].to_numpy(int)


def class_priors(meta):
    """Label prevalence on the benchmark training folds (used to fill failed predictions)."""
    train = benchmark_split(meta, TRAIN_FOLDS)
    return dict(zip(LABELS, label_matrix(train).mean(axis=0).round(4).tolist()))


def signal_path(meta, ecg_id, sampling_rate=500):
    column = "filename_hr" if sampling_rate == 500 else "filename_lr"
    return str(Path(meta.attrs["root"]) / meta.loc[ecg_id, column])


def read_signal(path):
    signal, _ = wfdb.rdsamp(path)
    return np.asarray(signal, dtype=float)


def load_signal(meta, ecg_id, sampling_rate=500):
    return read_signal(signal_path(meta, ecg_id, sampling_rate))
