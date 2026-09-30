"""Data generation and validation for the portfolio demo.

The demo data is synthetic by design. A production adapter should return the same
schema after joining versioned official statistics by local authority code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DATA_DICTIONARY = {
    "authority": "Synthetic local authority label",
    "nation": "UK nation",
    "median_rent_income_ratio": "Median monthly rent as a share of monthly household income",
    "energy_inefficiency_pct": "Estimated share of homes below EPC band C",
    "unemployment_pct": "Working-age unemployment percentage",
    "benefit_reliance_pct": "Working-age households receiving means-tested support",
    "overcrowding_pct": "Households with an overcrowding indicator",
    "food_insecurity_pct": "Estimated household food insecurity percentage",
    "high_pressure": "Synthetic modelling target: elevated local cost-of-living pressure",
}

FEATURES = [
    "median_rent_income_ratio",
    "energy_inefficiency_pct",
    "unemployment_pct",
    "benefit_reliance_pct",
    "overcrowding_pct",
    "food_insecurity_pct",
]
TARGET = "high_pressure"


def make_demo_dataset(n: int = 180, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic, plausible-looking aggregate dataset for the demo."""
    rng = np.random.default_rng(seed)
    nations = rng.choice(["England", "Scotland", "Wales", "Northern Ireland"], n, p=[0.72, 0.12, 0.09, 0.07])
    rent = np.clip(rng.normal(0.34, 0.09, n), 0.16, 0.68)
    unemployment = np.clip(rng.normal(4.8, 1.8, n), 1.3, 12.0)
    energy = np.clip(rng.normal(0.38, 0.11, n), 0.12, 0.72)
    benefits = np.clip(2.4 + unemployment * 0.75 + rng.normal(0, 1.3, n), 2.0, 18.0)
    overcrowding = np.clip(rng.normal(3.7, 1.6, n) + (rent - 0.34) * 8, 0.8, 12.0)
    food = np.clip(3.5 + unemployment * 0.55 + rent * 5 + rng.normal(0, 1.1, n), 2.0, 18.0)
    latent_score = 4.2 * rent + 0.10 * unemployment + 0.045 * energy + 0.06 * benefits + 0.07 * overcrowding + 0.055 * food + rng.normal(0, 0.34, n)
    threshold = float(np.quantile(latent_score, 0.70))
    target = (latent_score >= threshold).astype(int)

    return pd.DataFrame({
        "authority": [f"Demo Authority {i:03d}" for i in range(1, n + 1)],
        "nation": nations,
        "median_rent_income_ratio": rent.round(3),
        "energy_inefficiency_pct": energy.round(3),
        "unemployment_pct": unemployment.round(2),
        "benefit_reliance_pct": benefits.round(2),
        "overcrowding_pct": overcrowding.round(2),
        "food_insecurity_pct": food.round(2),
        "high_pressure": target,
    })


def validate_dataset(frame: pd.DataFrame) -> None:
    """Raise a useful error when an ingestion adapter violates the contract."""
    missing = set(DATA_DICTIONARY) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame[FEATURES].isna().any().any():
        raise ValueError("Feature columns cannot contain missing values")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1")
