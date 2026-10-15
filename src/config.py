"""Shared constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "employees.csv"
MODEL_PATH = ROOT / "models" / "attrition_model.joblib"
META_PATH = ROOT / "models" / "model_meta.json"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

TARGET = "attrition"  # "Yes" / "No"
ID_COLUMN = "employee_id"

CATEGORICAL = ["gender", "marital_status", "department", "job_role", "business_travel", "overtime"]
NUMERIC = [
    "age", "education", "distance_from_home", "job_level", "monthly_income", "total_working_years",
    "years_at_company", "years_in_current_role", "years_since_last_promotion", "num_companies_worked",
    "job_satisfaction", "environment_satisfaction", "work_life_balance", "performance_rating",
    "training_times_last_year", "percent_salary_hike", "stock_option_level",
]
ENGINEERED = ["income_per_working_year", "promotion_gap_ratio", "avg_satisfaction"]
