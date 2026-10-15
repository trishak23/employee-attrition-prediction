"""Generate a synthetic HR dataset modelled on the well-known IBM HR Analytics data.

Attrition is simulated from a logistic model whose drivers (overtime, low satisfaction,
low pay, long time since promotion, frequent travel, ...) mirror findings from HR research.
The intercept is calibrated so ~16% of employees leave.

To use the real IBM dataset, rename its columns to snake_case (see src/config.py) and save as
data/employees.csv.

Usage:
    python -m src.generate_data --employees 3000
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import DATA_PATH

ROLES = {
    "Sales": ["Sales Executive", "Sales Representative", "Manager"],
    "Research & Development": ["Research Scientist", "Laboratory Technician", "Healthcare Representative", "Manager"],
    "Human Resources": ["HR Executive", "HR Coordinator", "Manager"],
}
DEPT_P = [0.30, 0.60, 0.10]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n: int = 3000, seed: int = 42, target_rate: float = 0.16) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(37, 9, n), 18, 60).round().astype(int)
    total_years = np.clip(age - 18 - rng.integers(0, 8, n), 0, 40)
    job_level = np.clip(1 + (total_years / 8).astype(int) + rng.integers(-1, 2, n), 1, 5)
    years_at_company = np.clip((total_years * rng.uniform(0.2, 0.9, n)).round().astype(int), 0, 40)
    years_in_role = np.clip((years_at_company * rng.uniform(0.2, 1.0, n)).round().astype(int), 0, 18)
    years_since_promo = np.clip((years_at_company * rng.uniform(0.0, 0.8, n)).round().astype(int), 0, 15)
    income = np.clip(2000 + job_level * 2600 + total_years * 120 + rng.normal(0, 1100, n), 1500, 20000).round(-1)

    dept = rng.choice(list(ROLES), size=n, p=DEPT_P)
    role = np.array([rng.choice(ROLES[d]) for d in dept])

    df = pd.DataFrame({
        "employee_id": np.arange(1, n + 1),
        "age": age,
        "gender": rng.choice(["Male", "Female"], n, p=[0.58, 0.42]),
        "marital_status": rng.choice(["Single", "Married", "Divorced"], n, p=[0.32, 0.46, 0.22]),
        "department": dept,
        "job_role": role,
        "education": rng.choice([1, 2, 3, 4, 5], n, p=[0.11, 0.19, 0.39, 0.27, 0.04]),
        "business_travel": rng.choice(["Non-Travel", "Travel_Rarely", "Travel_Frequently"], n, p=[0.10, 0.70, 0.20]),
        "distance_from_home": np.clip(rng.exponential(8, n).round().astype(int) + 1, 1, 29),
        "job_level": job_level,
        "monthly_income": income,
        "total_working_years": total_years,
        "years_at_company": years_at_company,
        "years_in_current_role": years_in_role,
        "years_since_last_promotion": years_since_promo,
        "num_companies_worked": np.clip(rng.poisson(2.2, n), 0, 9),
        "overtime": rng.choice(["Yes", "No"], n, p=[0.28, 0.72]),
        "job_satisfaction": rng.choice([1, 2, 3, 4], n, p=[0.19, 0.19, 0.31, 0.31]),
        "environment_satisfaction": rng.choice([1, 2, 3, 4], n, p=[0.20, 0.19, 0.31, 0.30]),
        "work_life_balance": rng.choice([1, 2, 3, 4], n, p=[0.05, 0.23, 0.61, 0.11]),
        "performance_rating": rng.choice([3, 4], n, p=[0.85, 0.15]),
        "training_times_last_year": np.clip(rng.poisson(2.8, n), 0, 6),
        "percent_salary_hike": np.clip(rng.normal(15, 3.6, n).round().astype(int), 11, 25),
        "stock_option_level": rng.choice([0, 1, 2, 3], n, p=[0.43, 0.40, 0.11, 0.06]),
    })

    income_z = (np.log(df["monthly_income"]) - np.log(df["monthly_income"]).mean()) / np.log(df["monthly_income"]).std()
    logit = (
        1.15 * (df["overtime"] == "Yes")
        + 0.70 * (df["business_travel"] == "Travel_Frequently")
        - 0.30 * (df["business_travel"] == "Non-Travel")
        + 0.60 * (df["marital_status"] == "Single")
        - 0.35 * (df["job_satisfaction"] - 2.8)
        - 0.30 * (df["environment_satisfaction"] - 2.8)
        - 0.25 * (df["work_life_balance"] - 2.8)
        - 0.55 * income_z
        + 0.035 * df["distance_from_home"]
        + 0.12 * df["num_companies_worked"]
        + 0.10 * df["years_since_last_promotion"]
        - 0.35 * df["stock_option_level"]
        - 0.025 * (df["age"] - 37)
        - 0.05 * df["years_in_current_role"]
        + 0.40 * (df["department"] == "Sales")
        + rng.normal(0, 0.5, n)
    )
    # calibrate intercept so the overall attrition rate hits the target
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _sigmoid(logit + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    prob = _sigmoid(logit + (lo + hi) / 2)
    df["attrition"] = np.where(rng.uniform(size=n) < prob, "Yes", "No")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--employees", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = generate(args.employees, args.seed)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} employees -> {DATA_PATH}  (attrition rate {100 * (df.attrition == 'Yes').mean():.1f}%)")


if __name__ == "__main__":
    main()
