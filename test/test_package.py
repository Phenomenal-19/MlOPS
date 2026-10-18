import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.package import save_mlflow_format
from src.models.train import make_pipeline


def test_mlflow_export_can_be_repeated_over_an_existing_directory(tmp_path, synthetic_data):
    X, y = synthetic_data
    X = X.astype("float64")
    pipe = make_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    target = tmp_path / "models" / "mlflow_model"

    save_mlflow_format(pipe, X, target)
    save_mlflow_format(pipe, X, target)  # second export used to raise MlflowException

    reloaded = mlflow.sklearn.load_model(str(target))
    assert np.allclose(reloaded.predict_proba(X), pipe.predict_proba(X))
