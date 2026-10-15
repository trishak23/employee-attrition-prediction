"""Streamlit app for HR: score one employee or a whole CSV and see risk drivers.

Run:  streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

from src.config import DATA_PATH, FIGURES, META_PATH, MODEL_PATH
from src.predict import AttritionPredictor
from src.train import RAW_FEATURES

st.set_page_config(page_title="Employee Attrition Predictor", page_icon="👥", layout="wide")
st.title("👥 Employee Attrition Predictor")
st.caption("Estimate how likely an employee is to leave and which factors are driving the risk.")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python -m src.generate_data` and `python -m src.train` first.")
    st.stop()


@st.cache_resource
def load_predictor() -> AttritionPredictor:
    return AttritionPredictor()


predictor = load_predictor()
BAND_ICON = {"Low": "🟢", "Medium": "🟠", "High": "🔴"}

tab_one, tab_batch, tab_model = st.tabs(["Single employee", "Batch scoring (CSV)", "Model & drivers"])

with tab_one:
    with st.form("employee"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 60, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
            education = st.slider("Education (1-5)", 1, 5, 3)
            distance = st.slider("Distance from home (km)", 1, 30, 8)
            total_years = st.slider("Total working years", 0, 40, 10)
        with c2:
            dept = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            role = st.text_input("Job role", "Research Scientist")
            level = st.slider("Job level", 1, 5, 2)
            income = st.number_input("Monthly income", 1500, 20000, 6000, step=100)
            travel = st.selectbox("Business travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"], index=1)
            overtime = st.selectbox("Overtime", ["No", "Yes"])
        with c3:
            yac = st.slider("Years at company", 0, 40, 5)
            yir = st.slider("Years in current role", 0, 18, 2)
            ysp = st.slider("Years since last promotion", 0, 15, 1)
            ncw = st.slider("Companies worked at before", 0, 9, 2)
            js = st.slider("Job satisfaction (1-4)", 1, 4, 3)
            es = st.slider("Environment satisfaction (1-4)", 1, 4, 3)
            wlb = st.slider("Work-life balance (1-4)", 1, 4, 3)
            stock = st.slider("Stock option level (0-3)", 0, 3, 1)
        submitted = st.form_submit_button("Predict attrition risk", type="primary")

    if submitted:
        record = dict(
            age=age, education=education, distance_from_home=distance, job_level=level, monthly_income=income,
            total_working_years=total_years, years_at_company=yac, years_in_current_role=yir,
            years_since_last_promotion=ysp, num_companies_worked=ncw, job_satisfaction=js,
            environment_satisfaction=es, work_life_balance=wlb, performance_rating=3,
            training_times_last_year=2, percent_salary_hike=14, stock_option_level=stock, gender=gender,
            marital_status=marital, department=dept, job_role=role, business_travel=travel, overtime=overtime,
        )
        result = predictor.score_one(record)
        a, b = st.columns(2)
        a.metric("Attrition probability", f"{result['probability']:.0%}")
        b.metric("Risk band", f"{BAND_ICON[result['risk_band']]} {result['risk_band']}")
        st.subheader("Risk factors present")
        if result["risk_factors"]:
            for factor in result["risk_factors"]:
                st.write(f"- {factor}")
        else:
            st.success("No major risk factors flagged.")

with tab_batch:
    st.write("Upload a CSV with the columns:")
    st.code(", ".join(RAW_FEATURES), language="text")
    file = st.file_uploader("Employees CSV", type="csv")
    if file is None and DATA_PATH.exists():
        st.caption("No file uploaded — showing a demo using the bundled dataset (first 300 rows).")
        data = pd.read_csv(DATA_PATH).head(300)
    elif file is not None:
        data = pd.read_csv(file)
    else:
        data = None
    if data is not None:
        missing = [c for c in RAW_FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            scored = predictor.score(data).sort_values("attrition_probability", ascending=False)
            st.bar_chart(scored["risk_band"].value_counts().reindex(["High", "Medium", "Low"]))
            cols = [c for c in ["employee_id", "department", "job_role", "attrition_probability", "risk_band", "risk_factors"] if c in scored.columns]
            st.dataframe(scored[cols], use_container_width=True, hide_index=True)
            st.download_button("Download scores", scored.to_csv(index=False), "attrition_scores.csv")

with tab_model:
    meta = json.loads(META_PATH.read_text())
    st.write(f"**Selected model:** {meta['best_model']} | **decision threshold:** {meta['threshold']}")
    cols = st.columns(5)
    for col, key in zip(cols, ["test_roc_auc", "test_pr_auc", "test_precision", "test_recall", "test_f1"]):
        col.metric(key.replace("test_", "").replace("_", " ").upper(), meta[key])
    for name in ["feature_importance.png", "roc_confusion.png", "eda_attrition_drivers.png"]:
        if (FIGURES / name).exists():
            st.image(str(FIGURES / name))
