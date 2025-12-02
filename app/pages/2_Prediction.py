import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import load_artifacts, predict_emi_eligibility, predict_max_emi

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")

st.title("🔮 EMI Prediction")

artifacts = load_artifacts()

if artifacts:
    with st.form("prediction_form"):
        st.subheader("Applicant Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
            
        with col2:
            monthly_salary = st.number_input("Monthly Salary (INR)", min_value=0.0, value=50000.0)
            employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
            years_of_employment = st.number_input("Years of Employment", min_value=0.0, value=5.0)
            company_type = st.selectbox("Company Type", ["MNC", "Large Indian", "Mid-size", "Small", "Startup"])
            
        with col3:
            house_type = st.selectbox("House Type", ["Own", "Rented", "Family"])
            monthly_rent = st.number_input("Monthly Rent", min_value=0.0, value=0.0)
            family_size = st.number_input("Family Size", min_value=1, value=3)
            dependents = st.number_input("Dependents", min_value=0, value=1)
            
        st.subheader("Financial Obligations")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            school_fees = st.number_input("School Fees", min_value=0.0, value=0.0)
            college_fees = st.number_input("College Fees", min_value=0.0, value=0.0)
            travel_expenses = st.number_input("Travel Expenses", min_value=0.0, value=2000.0)
            
        with col5:
            groceries_utilities = st.number_input("Groceries & Utilities", min_value=0.0, value=10000.0)
            other_monthly_expenses = st.number_input("Other Expenses", min_value=0.0, value=5000.0)
            
        with col6:
            existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
            current_emi_amount = st.number_input("Current EMI Amount", min_value=0.0, value=0.0)
            
        st.subheader("Financial Status & Loan Request")
        col7, col8, col9 = st.columns(3)
        
        with col7:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=750)
            bank_balance = st.number_input("Bank Balance", min_value=0.0, value=50000.0)
            emergency_fund = st.number_input("Emergency Fund", min_value=0.0, value=20000.0)
            
        with col8:
            emi_scenario = st.selectbox("EMI Scenario", [
                "E-commerce Shopping EMI", "Home Appliances EMI", 
                "Vehicle EMI", "Personal Loan EMI", "Education EMI"
            ])
            requested_amount = st.number_input("Requested Amount", min_value=0.0, value=100000.0)
            requested_tenure = st.number_input("Requested Tenure (Months)", min_value=1, value=12)
            
        submit_button = st.form_submit_button("Predict")
        
    if submit_button:
        # Create DataFrame from input
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'marital_status': [marital_status],
            'education': [education],
            'monthly_salary': [monthly_salary],
            'employment_type': [employment_type],
            'years_of_employment': [years_of_employment],
            'company_type': [company_type],
            'house_type': [house_type],
            'monthly_rent': [monthly_rent],
            'family_size': [family_size],
            'dependents': [dependents],
            'school_fees': [school_fees],
            'college_fees': [college_fees],
            'travel_expenses': [travel_expenses],
            'groceries_utilities': [groceries_utilities],
            'other_monthly_expenses': [other_monthly_expenses],
            'existing_loans': [existing_loans],
            'current_emi_amount': [current_emi_amount],
            'credit_score': [credit_score],
            'bank_balance': [bank_balance],
            'emergency_fund': [emergency_fund],
            'emi_scenario': [emi_scenario],
            'requested_amount': [requested_amount],
            'requested_tenure': [requested_tenure]
        })
        
        with st.spinner("Analyzing..."):
            eligibility, proba = predict_emi_eligibility(input_data, artifacts)
            max_emi = predict_max_emi(input_data, artifacts)
            
        st.markdown("### Results")
        
        r_col1, r_col2 = st.columns(2)
        
        with r_col1:
            color = "green" if eligibility == "Eligible" else "red" if eligibility == "Not_Eligible" else "orange"
            st.markdown(f"#### Eligibility: :{color}[{eligibility}]")
            if proba is not None:
                st.progress(float(max(proba)))
                st.caption(f"Confidence: {max(proba):.2%}")
                
        with r_col2:
            st.metric("Max Safe Monthly EMI", f"₹{max_emi:,.2f}")
            
        if eligibility == "Not_Eligible":
            st.warning("The applicant is not eligible for the requested loan terms.")
        elif eligibility == "High_Risk":
            st.info("The applicant is eligible but considered High Risk. Consider higher interest rates.")
        else:
            st.success("The applicant is eligible for the loan.")
else:
    st.error("Could not load models. Please check if model training was successful.")
