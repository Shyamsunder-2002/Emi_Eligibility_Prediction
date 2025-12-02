import streamlit as st

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide"
)

st.title("💰 EMIPredict AI")
st.subheader("Intelligent Financial Risk Assessment Platform")

st.markdown("""
### Welcome to EMIPredict AI

This platform leverages advanced Machine Learning models to assess financial risk and predict EMI eligibility.

#### Key Features:
- **EMI Eligibility Prediction**: Determine if a loan applicant is Eligible, High Risk, or Not Eligible.
- **Max EMI Estimation**: Calculate the maximum safe monthly EMI amount for an applicant.
- **Interactive Dashboard**: Explore the underlying financial data and trends.
- **Model Performance**: View the performance metrics of our ML models tracked via MLflow.

#### How to use:
1. Navigate to **Dashboard** to explore the dataset.
2. Go to **Prediction** to assess a new customer.
3. Check **Model Performance** to see technical details.

---
**Built with Python, Streamlit, Scikit-Learn, XGBoost, and MLflow.**
""")

st.sidebar.success("Select a page above.")
