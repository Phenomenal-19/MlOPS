"""Clean the raw Cleveland data and write data/processed/heart_clean.csv.

Cleaning steps:
  * '?' placeholders -> NaN (they appear in `ca` and `thal`)
  * `num` (0-4 severity) -> binary `target` (0 = no disease, 1 = disease)
  * missing values are kept as NaN here; imputation happens inside the
    model pipeline so it is fit on training folds only (no leakage).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/heart_cleveland_raw.csv")
CLEAN_PATH = Path("data/processed/heart_clean.csv")

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy with a binary target column."""
    df = df.copy().replace("?", np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET] = (df["num"] > 0).astype(int)
    return df.drop(columns=["num"])


def main(src: Path = RAW_PATH, dest: Path = CLEAN_PATH) -> pd.DataFrame:
    df = clean(pd.read_csv(src))
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=RAW_PATH)
    parser.add_argument("--dest", type=Path, default=CLEAN_PATH)
    args = parser.parse_args()
    out = main(args.src, args.dest)
    print(f"Wrote {len(out)} rows to {args.dest}")
    print("Missing values per column:")
    print(out.isna().sum()[out.isna().sum() > 0].to_string())
