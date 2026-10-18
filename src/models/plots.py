"""Diagnostic plots logged as MLflow artifacts."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def roc_plot(pipe, X, y, title: str):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_estimator(pipe, X, y, ax=ax, plot_chance_level=True)
    ax.set_title(f"ROC curve (test) - {title}")
    fig.tight_layout()
    return fig


def confusion_plot(pipe, X, y, title: str):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipe, X, y, ax=ax, cmap="Blues", display_labels=["No disease", "Disease"])
    ax.set_title(f"Confusion matrix (test) - {title}")
    fig.tight_layout()
    return fig


def importance_plot(pipe, title: str, top: int = 15):
    """Coefficient magnitude (linear models) or impurity importance (trees)."""
    model = pipe.named_steps["model"]
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    if hasattr(model, "coef_"):
        values, label = model.coef_.ravel(), "Coefficient"
    else:
        values, label = model.feature_importances_, "Importance"
    order = np.argsort(np.abs(values))[::-1][:top][::-1]
    colors = ["#E45756" if values[i] > 0 else "#4C78A8" for i in order]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh([names[i].split("__")[-1] for i in order], values[order], color=colors)
    ax.set_xlabel(label)
    ax.set_title(f"Top {top} features - {title}")
    fig.tight_layout()
    return fig
