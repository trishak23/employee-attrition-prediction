"""Exploratory analysis: attrition rate by key drivers.

Usage:
    python -m src.eda
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import DATA_PATH, FIGURES, TARGET


def attrition_rate(df: pd.DataFrame, by) -> pd.Series:
    return (df[TARGET].eq("Yes").groupby(by, observed=True).mean() * 100).round(1)


def run(df: pd.DataFrame) -> dict[str, pd.Series]:
    df = df.copy()
    df["tenure_band"] = pd.cut(df["years_at_company"], [-1, 1, 3, 5, 10, 50], labels=["0-1", "2-3", "4-5", "6-10", "10+"])
    df["income_band"] = pd.qcut(df["monthly_income"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    drivers = {
        "Overtime": "overtime",
        "Job satisfaction (1-4)": "job_satisfaction",
        "Business travel": "business_travel",
        "Marital status": "marital_status",
        "Tenure (years)": "tenure_band",
        "Income quartile": "income_band",
    }
    rates = {label: attrition_rate(df, df[col]) for label, col in drivers.items()}

    FIGURES.mkdir(parents=True, exist_ok=True)
    overall = df[TARGET].eq("Yes").mean() * 100
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (label, series) in zip(axes.ravel(), rates.items()):
        ax.bar(series.index.astype(str), series.values, color="#ef4444")
        ax.axhline(overall, color="grey", ls="--", lw=1)
        ax.set(title=label, ylabel="Attrition rate (%)")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"Attrition rate by driver (dashed line = overall {overall:.1f}%)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_attrition_drivers.png", dpi=150)
    plt.close(fig)
    return rates


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Employees: {len(df):,} | overall attrition: {df[TARGET].eq('Yes').mean() * 100:.1f}%\n")
    for label, series in run(df).items():
        print(f"{label}:\n{series.to_string()}\n")
    print(f"Figure saved to {FIGURES / 'eda_attrition_drivers.png'}")


if __name__ == "__main__":
    main()
