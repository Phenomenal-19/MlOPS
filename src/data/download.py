"""Download the UCI Heart Disease (Cleveland) dataset into data/raw/."""
import argparse
import urllib.request
from pathlib import Path

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
RAW_PATH = Path("data/raw/heart_cleveland_raw.csv")

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def download(dest: Path = RAW_PATH, url: str = URL) -> Path:
    """Fetch the raw file and save it with a header row. Returns the saved path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as resp:
        body = resp.read().decode("utf-8")
    lines = [ln for ln in body.splitlines() if ln.strip()]
    dest.write_text(",".join(COLUMNS) + "\n" + "\n".join(lines) + "\n")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=RAW_PATH)
    args = parser.parse_args()
    path = download(args.dest)
    print(f"Saved raw dataset to {path}")
