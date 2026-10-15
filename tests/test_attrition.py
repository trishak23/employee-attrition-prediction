import pandas as pd
import pytest

from src.features import add_features
from src.generate_data import generate
from src.predict import risk_band, risk_factors
from src.train import RAW_FEATURES, candidates


@pytest.fixture(scope="module")
def df():
    return generate(n=1200, seed=3)


def test_generated_data_schema_and_rate(df):
    assert set(RAW_FEATURES) <= set(df.columns)
    assert df["employee_id"].is_unique
    assert set(df["attrition"]) == {"Yes", "No"}
    assert 0.10 < (df["attrition"] == "Yes").mean() < 0.22
    assert df.isna().sum().sum() == 0


def test_overtime_raises_attrition(df):
    rates = df.groupby("overtime")["attrition"].apply(lambda s: (s == "Yes").mean())
    assert rates["Yes"] > rates["No"]


def test_feature_engineering(df):
    out = add_features(df)
    assert {"income_per_working_year", "promotion_gap_ratio", "avg_satisfaction"} <= set(out.columns)
    assert out["avg_satisfaction"].between(1, 4).all()
    assert len(out) == len(df)


def test_model_beats_random_and_handles_unseen_category(df):
    X, y = df[RAW_FEATURES], (df["attrition"] == "Yes").astype(int)
    model = candidates()["Logistic Regression"].fit(X.iloc[:900], y.iloc[:900])
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y.iloc[900:], model.predict_proba(X.iloc[900:])[:, 1]) > 0.65
    odd = X.iloc[:1].copy()
    odd["job_role"] = "Chief Happiness Officer"  # unseen category must not crash
    assert 0 <= model.predict_proba(odd)[0, 1] <= 1


def test_risk_band_thresholds():
    assert risk_band(0.05) == "Low"
    assert risk_band(0.2) == "Medium"
    assert risk_band(0.6) == "High"


def test_risk_factors_flags(df):
    row = df.iloc[0].copy()
    row["overtime"], row["job_satisfaction"], row["years_since_last_promotion"] = "Yes", 1, 6
    flags = risk_factors(row)
    assert "Works overtime" in flags and "Low job satisfaction" in flags and "No promotion for 4+ years" in flags
