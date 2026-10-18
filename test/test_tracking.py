"""Integration test: one candidate is logged to a throwaway MLflow store."""
import mlflow
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.models import train


@pytest.fixture()
def tracking(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    exp_id = mlflow.create_experiment("test-exp", artifact_location=(tmp_path / "art").as_uri())
    mlflow.set_experiment(experiment_id=exp_id)
    yield exp_id
    mlflow.set_tracking_uri("sqlite:///mlflow.db")


def test_log_run_records_params_metrics_plots_and_model(tracking, synthetic_data):
    X, y = synthetic_data
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    res = train.evaluate_candidate(
        "logistic_regression", LogisticRegression(max_iter=500), {"model__C": [1.0]},
        X_tr, X_te, y_tr, y_te)
    run_id = train.log_run(res, X_tr, X_te, y_tr, y_te)

    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    assert run.data.params["model"] == "logistic_regression"
    assert run.data.params["seed"] == str(train.SEED)
    for key in ("cv_roc_auc_mean", "cv_recall_mean", "test_roc_auc", "test_recall"):
        assert key in run.data.metrics
    plots = {a.path for a in client.list_artifacts(run_id, "plots")}
    assert plots == {"plots/roc_curve.png", "plots/confusion_matrix.png",
                     "plots/feature_importance.png"}
    assert client.list_artifacts(run_id, "tuning")

    reloaded = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    assert reloaded.predict(X_te).shape == (len(X_te),)
