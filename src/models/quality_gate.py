"""Fail (exit 1) if the packaged model's tracked metrics are below thresholds.

Used by CI so a retrain that silently degrades the model breaks the pipeline.
"""
import argparse
import json
import sys
from pathlib import Path

METADATA_PATH = Path("models/metadata.json")


def check(metrics: dict, min_cv_roc_auc: float, min_test_recall: float) -> list[str]:
    """Return a list of human-readable failures (empty means the gate passed)."""
    failures = []
    for name, floor in (("cv_roc_auc_mean", min_cv_roc_auc), ("test_recall", min_test_recall)):
        value = metrics.get(name)
        if value is None:
            failures.append(f"metric '{name}' is missing from metadata")
        elif value < floor:
            failures.append(f"{name}={value:.3f} is below the required {floor:.3f}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--min-cv-roc-auc", type=float, default=0.85)
    parser.add_argument("--min-test-recall", type=float, default=0.75)
    args = parser.parse_args(argv)

    metrics = json.loads(args.metadata.read_text())["metrics"]
    failures = check(metrics, args.min_cv_roc_auc, args.min_test_recall)
    if failures:
        print("QUALITY GATE FAILED:\n  - " + "\n  - ".join(failures), file=sys.stderr)
        return 1
    print(f"Quality gate passed: cv_roc_auc_mean={metrics['cv_roc_auc_mean']:.3f}, "
          f"test_recall={metrics['test_recall']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
