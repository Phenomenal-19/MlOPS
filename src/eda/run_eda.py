"""Generate EDA figures from the cleaned dataset into reports/figures/."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.preprocess import CATEGORICAL_FEATURES, CLEAN_PATH, NUMERIC_FEATURES, TARGET

FIG_DIR = Path("reports/figures")
PALETTE = {0: "#4C78A8", 1: "#E45756"}
LABELS = {0: "No disease", 1: "Disease"}


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)


def class_balance(df: pd.DataFrame) -> None:
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values,
                  color=[PALETTE[i] for i in counts.index])
    for bar, n in zip(bars, counts.values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 2,
                f"{n} ({n / len(df):.1%})", ha="center")
    ax.set_ylabel("Patients")
    ax.set_title("Class balance")
    _save(fig, "class_balance.png")


def histograms(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), NUMERIC_FEATURES, strict=False):
        sns.histplot(data=df, x=col, hue=TARGET, palette=PALETTE, kde=True,
                     bins=20, ax=ax, legend=False)
        ax.set_title(col)
    axes.ravel()[-1].axis("off")
    axes.ravel()[-1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PALETTE[i])
                                     for i in (0, 1)],
                            labels=[LABELS[i] for i in (0, 1)], loc="center")
    fig.suptitle("Numeric feature distributions by outcome")
    _save(fig, "histograms.png")


def categorical_counts(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), CATEGORICAL_FEATURES, strict=True):
        sns.countplot(data=df, x=col, hue=TARGET, palette=PALETTE, ax=ax, legend=False)
        ax.set_title(col)
        ax.set_xlabel("")
    fig.suptitle("Categorical features by outcome")
    _save(fig, "categorical_counts.png")


def correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation heatmap")
    _save(fig, "correlation_heatmap.png")


def boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    for ax, col in zip(axes, NUMERIC_FEATURES, strict=True):
        sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, palette=PALETTE,
                    ax=ax, legend=False)
        ax.set_xticks([0, 1], [LABELS[0], LABELS[1]])
        ax.set_xlabel("")
    fig.suptitle("Outliers and spread by outcome")
    _save(fig, "boxplots.png")


def summary(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR.parent / "eda_summary.txt"
    with out.open("w") as f:
        f.write(f"Shape: {df.shape}\n\nMissing values:\n{df.isna().sum()}\n\n")
        f.write(f"Describe:\n{df.describe().T}\n\n")
        f.write("Correlation with target:\n")
        f.write(df.corr(numeric_only=True)[TARGET].drop(TARGET)
                .sort_values(ascending=False).round(3).to_string())


def main(src: Path = CLEAN_PATH) -> None:
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(src)
    class_balance(df)
    histograms(df)
    categorical_counts(df)
    correlation_heatmap(df)
    boxplots(df)
    summary(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=CLEAN_PATH)
    main(parser.parse_args().src)
    print(f"Figures written to {FIG_DIR}")
