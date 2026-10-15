"""Feature engineering + preprocessing pipeline."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL, ENGINEERED, NUMERIC


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived HR features (kept stateless so the same code runs in training and serving)."""
    out = df.copy()
    out["income_per_working_year"] = out["monthly_income"] / (out["total_working_years"] + 1)
    out["promotion_gap_ratio"] = out["years_since_last_promotion"] / (out["years_at_company"] + 1)
    out["avg_satisfaction"] = out[["job_satisfaction", "environment_satisfaction", "work_life_balance"]].mean(axis=1)
    return out


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC + ENGINEERED),
        ("cat", categorical, CATEGORICAL),
    ])
