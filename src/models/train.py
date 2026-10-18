"""Train, tune and evaluate candidate classifiers.

Protocol
  1. Stratified 80/20 train/test split (test set is never used for tuning).
  2. GridSearchCV (stratified 5-fold, scored on ROC-AUC) on the training set.
  3. Cross-validated accuracy/precision/recall/F1/ROC-AUC of the tuned pipeline.
  4. Final metrics on the held-out test set.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from src.data.preprocess import CLEAN_PATH, TARGET
from src.features import FEATURE_COLUMNS, build_preprocessor
from src.models.plots import confusion_plot, importance_plot, roc_plot

SEED = 42
CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
REPORT_DIR = Path("reports")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT = "heart-disease-classification"

CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, random_state=SEED),
        {"model__C": [0.01, 0.1, 1, 10]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED),
        {"model__n_estimators": [100, 300],
         "model__max_depth": [3, 5, None],
         "model__min_samples_leaf": [1, 3, 5]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=SEED),
        {"model__n_estimators": [50, 100, 200],
         "model__learning_rate": [0.03, 0.1],
         "model__max_depth": [2, 3]},
    ),
}


def load_split(path: Path = CLEAN_PATH):
    df = pd.read_csv(path)
    X, y = df[FEATURE_COLUMNS], df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def make_pipeline(model) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def test_metrics(pipe: Pipeline, X, y) -> dict:
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def evaluate_candidate(name, model, grid, X_train, X_test, y_train, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(make_pipeline(model), grid, scoring="roc_auc",
                          cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    best = search.best_estimator_

    scores = cross_validate(best, X_train, y_train, cv=cv, scoring=CV_SCORING)
    cv_summary = {m: (scores[f"test_{m}"].mean(), scores[f"test_{m}"].std())
                  for m in CV_SCORING}
    return {
        "name": name,
        "best_params": {k.replace("model__", ""): v
                        for k, v in search.best_params_.items()},
        "cv_mean": {m: v[0] for m, v in cv_summary.items()},
        "cv_std": {m: v[1] for m, v in cv_summary.items()},
        "test": test_metrics(best, X_test, y_test),
        "estimator": best,
        "grid_results": pd.DataFrame(search.cv_results_).sort_values("rank_test_score"),
    }


def log_run(res: dict, X_train, X_test, y_train, y_test) -> str:
    """Log one candidate to MLflow as its own run; returns the run id."""
    name, best = res["name"], res["estimator"]
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"model_family": name, "stage": "experiment"})
        mlflow.log_params({
            "model": name,
            **res["best_params"],
            "seed": SEED,
            "cv_folds": 5,
            "tuning_metric": "roc_auc",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(FEATURE_COLUMNS),
            "positive_rate_train": round(float(y_train.mean()), 3),
        })
        for m in CV_SCORING:
            mlflow.log_metric(f"cv_{m}_mean", res["cv_mean"][m])
            mlflow.log_metric(f"cv_{m}_std", res["cv_std"][m])
        for m, v in res["test"].items():
            mlflow.log_metric(f"test_{m}", v)

        for fig_name, fig in (
            ("roc_curve.png", roc_plot(best, X_test, y_test, name)),
            ("confusion_matrix.png", confusion_plot(best, X_test, y_test, name)),
            ("feature_importance.png", importance_plot(best, name)),
        ):
            mlflow.log_figure(fig, f"plots/{fig_name}")
            plt.close(fig)

        mlflow.log_text(res["grid_results"].to_csv(index=False),
                        "tuning/grid_search_results.csv")
        mlflow.sklearn.log_model(best, name="model", input_example=X_train.head(3),
                                 serialization_format="cloudpickle")
        return run.info.run_id


def run() -> list[dict]:
    X_train, X_test, y_train, y_test = load_split()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    results = []
    for name, (model, grid) in CANDIDATES.items():
        res = evaluate_candidate(name, model, grid, X_train, X_test, y_train, y_test)
        res["run_id"] = log_run(res, X_train, X_test, y_train, y_test)
        results.append(res)
        print(f"{name:20s} CV ROC-AUC {res['cv_mean']['roc_auc']:.3f} "
              f"+/- {res['cv_std']['roc_auc']:.3f} | test ROC-AUC "
              f"{res['test']['roc_auc']:.3f} | params {res['best_params']}")
    best = max(results, key=lambda r: r["cv_mean"]["roc_auc"])
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("best_model", "true")
    print(f"Best model by CV ROC-AUC: {best['name']} (run {best['run_id']})")
    return results


def save_report(results: list[dict]) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    rows = []
    for r in results:
        row = {"model": r["name"]}
        row.update({f"cv_{m}": round(v, 3) for m, v in r["cv_mean"].items()})
        row.update({f"test_{m}": round(v, 3) for m, v in r["test"].items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    serialisable = [{k: v for k, v in r.items() if k not in ("estimator", "grid_results")}
                    for r in results]
    (REPORT_DIR / "model_results.json").write_text(json.dumps(serialisable, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    save_report(run())
    print(pd.read_csv(REPORT_DIR / "model_comparison.csv").to_string(index=False))
