"""Age-and-sex control (pre-specified comparator): a logistic regression per superclass on age, age
squared and sex, fitted on the benchmark training folds 1-8. It measures how much of the task demographics alone
explain. No API calls.

Run from the repository root:  python analysis/agesex_control.py
Writes controls/agesex_test/predictions.csv (fold 10) and controls/agesex_dev/predictions.csv (fold 9).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import LABELS, ROOT, TEST_FOLD, TRAIN_FOLDS, VAL_FOLD, benchmark_split, meta
from ecgcode.data import label_matrix


def features(df):
    age = pd.to_numeric(df["age"], errors="coerce").clip(upper=90)  # PTB-XL codes ages above 89 as 300
    age = age.fillna(age.median())
    sex = pd.to_numeric(df["sex"], errors="coerce").fillna(0)
    return np.c_[age / 100, (age / 100) ** 2, sex]


def main():
    m = meta()
    train = benchmark_split(m, TRAIN_FOLDS)
    y = label_matrix(train)
    models = [LogisticRegression(max_iter=1000).fit(features(train), y[:, j]) for j in range(len(LABELS))]
    for name, fold in [("agesex_test", TEST_FOLD), ("agesex_dev", VAL_FOLD)]:
        target = benchmark_split(m, fold)
        probs = np.stack([mod.predict_proba(features(target))[:, 1] for mod in models], axis=1)
        out = pd.DataFrame(probs, columns=[f"p_{l}" for l in LABELS], index=target.index.rename("ecg_id"))
        out["source"] = "model"
        path = ROOT / "controls" / name
        path.mkdir(parents=True, exist_ok=True)
        out.reset_index().to_csv(path / "predictions.csv", index=False)
        print(f"wrote {path / 'predictions.csv'} ({len(out)} records)")


if __name__ == "__main__":
    main()
