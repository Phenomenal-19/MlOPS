"""Export the best tracked model as a standalone, reusable package.

Reads the run tagged `best_model=true` in MLflow and writes:
  models/model.joblib     full sklearn Pipeline (preprocessing + estimator),
                          used by the serving API (no MLflow dependency)
  models/mlflow_model/    the same pipeline in MLflow format (signature + env)
  models/metadata.json    provenance: run id, metrics, versions, feature list
"""
import json
import shutil
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature

from src.data.preprocess import CLEAN_PATH
from src.features import FEATURE_COLUMNS
from src.models.train import EXPERIMENT, TRACKING_URI

MODEL_DIR = Path("models")


def best_run() -> pd.Series:
    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT],
                              filter_string="tags.best_model = 'true'",
                              order_by=["attributes.start_time DESC"])
    if runs.empty:
        sys.exit("No run tagged best_model=true. Run `python -m src.models.train` first.")
    return runs.iloc[0]


def save_mlflow_format(pipe, X: pd.DataFrame, path: Path) -> None:
    """Write the MLflow-format model, replacing any previous export.

    `mlflow.sklearn.save_model` refuses to write into a non-empty directory, so
    a re-run (or a CI checkout that already contains models/) would otherwise fail.
    """
    if path.exists():
        shutil.rmtree(path)
    mlflow.sklearn.save_model(
        pipe, str(path),
        signature=infer_signature(X, pipe.predict(X)),
        input_example=X.head(3), serialization_format="cloudpickle")


def main() -> None:
    run = best_run()
    pipe = mlflow.sklearn.load_model(f"runs:/{run.run_id}/model")
    X = pd.read_csv(CLEAN_PATH)[FEATURE_COLUMNS].astype("float64")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "model.joblib")
    save_mlflow_format(pipe, X, MODEL_DIR / "mlflow_model")

    reloaded = joblib.load(MODEL_DIR / "model.joblib")
    assert np.allclose(pipe.predict_proba(X), reloaded.predict_proba(X)), \
        "Exported model does not reproduce the tracked model's predictions"

    metrics = {k.removeprefix("metrics."): round(float(v), 4)
               for k, v in run.items() if k.startswith("metrics.")}
    meta = {
        "model": run["tags.mlflow.runName"],
        "mlflow_run_id": run.run_id,
        "metrics": metrics,
        "features": FEATURE_COLUMNS,
        "target": "target (1 = heart disease present)",
        "library_versions": {"scikit-learn": sklearn.__version__,
                             "pandas": pd.__version__, "numpy": np.__version__},
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Packaged '{meta['model']}' (run {run.run_id}) into {MODEL_DIR}/")


if __name__ == "__main__":
    main()
