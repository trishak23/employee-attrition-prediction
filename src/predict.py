"""Score employees and explain risk with simple, HR-friendly flags.

Usage:
    python -m src.predict --example
    python -m src.predict --csv data/employees.csv --out reports/attrition_scores.csv
"""
from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from .config import DATA_PATH, ID_COLUMN, META_PATH, MODEL_PATH, REPORTS
from .train import RAW_FEATURES

RISK_BANDS = [(0.0, "Low"), (0.15, "Medium"), (0.35, "High")]


def risk_band(p: float) -> str:
    band = "Low"
    for lower, name in RISK_BANDS:
        if p >= lower:
            band = name
    return band


def risk_factors(row: pd.Series) -> list[str]:
    """Rule-based, human-readable reasons (complements the model score)."""
    checks = [
        (row["overtime"] == "Yes", "Works overtime"),
        (row["job_satisfaction"] <= 2, "Low job satisfaction"),
        (row["environment_satisfaction"] <= 2, "Low environment satisfaction"),
        (row["work_life_balance"] <= 2, "Poor work-life balance"),
        (row["years_since_last_promotion"] >= 4, "No promotion for 4+ years"),
        (row["business_travel"] == "Travel_Frequently", "Travels frequently"),
        (row["distance_from_home"] >= 20, "Long commute (20+ km)"),
        (row["stock_option_level"] == 0, "No stock options"),
        (row["num_companies_worked"] >= 5, "History of frequent job changes"),
        (row["monthly_income"] < 3500, "Low monthly income"),
    ]
    return [text for flag, text in checks if flag]


class AttritionPredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.meta = json.loads(META_PATH.read_text())
        self.threshold = self.meta["threshold"]

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[RAW_FEATURES]
        out = df.copy()
        out["attrition_probability"] = self.model.predict_proba(X)[:, 1].round(3)
        out["predicted_to_leave"] = out["attrition_probability"] >= self.threshold
        out["risk_band"] = out["attrition_probability"].map(risk_band)
        out["risk_factors"] = X.apply(lambda r: "; ".join(risk_factors(r)), axis=1)
        return out

    def score_one(self, record: dict) -> dict:
        row = self.score(pd.DataFrame([record])).iloc[0]
        return {
            "probability": float(row["attrition_probability"]),
            "risk_band": row["risk_band"],
            "predicted_to_leave": bool(row["predicted_to_leave"]),
            "risk_factors": [f for f in row["risk_factors"].split("; ") if f],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", help="CSV of employees to score")
    parser.add_argument("--out", default=str(REPORTS / "attrition_scores.csv"))
    parser.add_argument("--example", action="store_true", help="Score one sample employee from the dataset")
    args = parser.parse_args()

    predictor = AttritionPredictor()
    if args.example:
        sample = pd.read_csv(DATA_PATH)[RAW_FEATURES].iloc[0].to_dict()
        print(json.dumps({"input": sample, "prediction": predictor.score_one(sample)}, indent=2, default=str))
    elif args.csv:
        scored = predictor.score(pd.read_csv(args.csv)).sort_values("attrition_probability", ascending=False)
        scored.to_csv(args.out, index=False)
        print(f"Scored {len(scored):,} employees -> {args.out}")
        print(scored["risk_band"].value_counts().to_string())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
