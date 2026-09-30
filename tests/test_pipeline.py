import pandas as pd
import pytest

from uk_risk.data import FEATURES, make_demo_dataset, validate_dataset
from uk_risk.modeling import explain_row, train_model


def test_demo_data_is_reproducible_and_valid():
    first = make_demo_dataset()
    second = make_demo_dataset()
    pd.testing.assert_frame_equal(first, second)
    validate_dataset(first)
    assert len(first) == 180
    assert first["high_pressure"].mean() == pytest.approx(0.30)


def test_validation_rejects_missing_columns():
    frame = make_demo_dataset().drop(columns=[FEATURES[0]])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset(frame)


def test_model_produces_probabilities_and_explanation():
    result = train_model(make_demo_dataset())
    assert 0 <= result.metrics["roc_auc"] <= 1
    assert result.test_frame["risk_probability"].between(0, 1).all()
    explanation = explain_row(result, result.test_frame.iloc[0])
    assert set(explanation["feature"]) == set(FEATURES)
