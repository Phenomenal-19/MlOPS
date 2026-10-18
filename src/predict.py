"""Inference helper shared by the CLI and the serving API.

CLI:  python -m src.predict --input sample_input.json
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

from src.features import FEATURE_COLUMNS

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
LABELS = {0: "no heart disease", 1: "heart disease"}


def load_model(path: Path = MODEL_PATH):
    import joblib
    return joblib.load(path)


def predict(model, records: list[dict]) -> list[dict]:
    """Score patient records.

    Missing optional values (e.g. `ca`, `thal`) are imputed by the pipeline;
    unknown keys are ignored; missing columns become NaN. `confidence` is the
    probability of the predicted class, `risk_probability` of heart disease.
    """
    X = pd.DataFrame(records).reindex(columns=FEATURE_COLUMNS).astype("float64")
    proba = model.predict_proba(X)[:, 1]
    out = []
    for p in proba:
        label = int(p >= 0.5)
        out.append({
            "prediction": label,
            "label": LABELS[label],
            "confidence": round(float(p if label else 1 - p), 4),
            "risk_probability": round(float(p), 4),
        })
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSON file with one patient object or a list of them")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    records = data if isinstance(data, list) else [data]
    print(json.dumps(predict(load_model(args.model), records), indent=2))
