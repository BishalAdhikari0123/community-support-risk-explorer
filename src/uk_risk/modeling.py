"""Model training and explainability utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import FEATURES, TARGET


@dataclass
class ModelResult:
    model: Pipeline
    test_frame: pd.DataFrame
    metrics: dict[str, float]
    importance: pd.DataFrame
    calibration: pd.DataFrame


def train_model(frame: pd.DataFrame, seed: int = 42) -> ModelResult:
    """Train a calibrated, interpretable baseline and return evaluation artefacts."""
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame[TARGET], test_size=0.25, random_state=seed, stratify=frame[TARGET]
    )
    transformer = ColumnTransformer([("numeric", StandardScaler(), FEATURES)], remainder="drop")
    model = Pipeline([
        ("features", transformer),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
    ])
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    threshold = 0.50
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "recall_at_50": float(recall_score(y_test, predictions)),
        "positive_rate": float(y_test.mean()),
    }
    permutation = permutation_importance(model, x_test, y_test, n_repeats=20, random_state=seed, scoring="roc_auc")
    importance = pd.DataFrame({"feature": FEATURES, "importance": permutation.importances_mean}).sort_values("importance", ascending=False)
    fraction_true, fraction_pred = calibration_curve(y_test, probabilities, n_bins=5, strategy="quantile")
    calibration = pd.DataFrame({"predicted": fraction_pred, "observed": fraction_true})
    test_frame = x_test.copy()
    test_frame["actual"] = y_test.to_numpy()
    test_frame["risk_probability"] = probabilities
    return ModelResult(model, test_frame, metrics, importance, calibration)


def explain_row(result: ModelResult, row: pd.Series) -> pd.DataFrame:
    """Return signed, standardised linear contributions for one authority."""
    classifier = result.model.named_steps["classifier"]
    scaler = result.model.named_steps["features"].named_transformers_["numeric"]
    values = (row[FEATURES].astype(float).to_numpy() - scaler.mean_) / scaler.scale_
    contributions = values * classifier.coef_[0]
    return pd.DataFrame({"feature": FEATURES, "contribution": contributions}).sort_values("contribution", ascending=False)
