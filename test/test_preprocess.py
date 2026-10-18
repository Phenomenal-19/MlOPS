import numpy as np
import pandas as pd

from src.data.preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, clean, main


def test_clean_converts_placeholders_to_nan(raw_frame):
    out = clean(raw_frame)
    assert out["ca"].isna().sum() == 1
    assert out["thal"].isna().sum() == 1
    assert pd.api.types.is_numeric_dtype(out["ca"])
    assert pd.api.types.is_numeric_dtype(out["thal"])


def test_clean_builds_binary_target_and_drops_num(raw_frame):
    out = clean(raw_frame)
    assert "num" not in out.columns
    assert set(out[TARGET].unique()) <= {0, 1}
    # severity 0 -> 0, severity > 0 -> 1
    assert out[TARGET].tolist() == [0, 1, 1, 0]


def test_clean_drops_duplicates_and_resets_index(raw_frame):
    out = clean(raw_frame)
    assert len(out) == 4
    assert out.index.tolist() == [0, 1, 2, 3]


def test_clean_does_not_mutate_input(raw_frame):
    before = raw_frame.copy(deep=True)
    clean(raw_frame)
    pd.testing.assert_frame_equal(raw_frame, before)


def test_feature_lists_are_disjoint_and_complete():
    assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)
    assert len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) == 13


def test_main_round_trip(tmp_path, raw_frame):
    src, dest = tmp_path / "raw.csv", tmp_path / "out" / "clean.csv"
    raw_frame.to_csv(src, index=False)
    out = main(src, dest)
    assert dest.exists()
    reread = pd.read_csv(dest)
    assert reread.shape == out.shape
    assert np.isnan(reread["ca"]).sum() == 1
