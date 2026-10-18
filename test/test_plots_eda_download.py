import io

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data import download
from src.eda import run_eda
from src.models.plots import confusion_plot, importance_plot, roc_plot
from src.models.train import make_pipeline


@pytest.fixture(scope="module", params=["linear", "tree"])
def fitted(request, synthetic_data):
    X, y = synthetic_data
    model = LogisticRegression(max_iter=500) if request.param == "linear" else (
        RandomForestClassifier(n_estimators=20, random_state=0))
    return make_pipeline(model).fit(X, y), X, y


def test_plot_helpers_return_figures(fitted):
    pipe, X, y = fitted
    for fig in (roc_plot(pipe, X, y, "m"), confusion_plot(pipe, X, y, "m"),
                importance_plot(pipe, "m")):
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_download_writes_header_and_rows(tmp_path, monkeypatch):
    body = b"63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n\n" \
           b"67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,3.0,2\n"

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(download.urllib.request, "urlopen",
                        lambda url, timeout=0: FakeResponse(body))
    dest = download.download(tmp_path / "raw" / "heart.csv")
    df = pd.read_csv(dest)
    assert list(df.columns) == download.COLUMNS
    assert len(df) == 2  # blank line skipped


def test_eda_main_writes_all_figures_and_summary(tmp_path, monkeypatch, synthetic_data):
    X, y = synthetic_data
    src = tmp_path / "clean.csv"
    pd.concat([X, y], axis=1).to_csv(src, index=False)
    monkeypatch.setattr(run_eda, "FIG_DIR", tmp_path / "reports" / "figures")
    run_eda.main(src)
    names = {p.name for p in run_eda.FIG_DIR.iterdir()}
    assert names == {"class_balance.png", "histograms.png", "categorical_counts.png",
                     "correlation_heatmap.png", "boxplots.png"}
    assert (tmp_path / "reports" / "eda_summary.txt").exists()
