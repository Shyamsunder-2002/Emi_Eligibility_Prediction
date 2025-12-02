import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

@st.cache_resource
def load_artifacts():
    """Load all necessary models and preprocessors."""
    artifacts = {}
    try:
        artifacts['preprocessor'] = joblib.load('models/preprocessors/preprocessor.joblib')
        artifacts['label_encoder'] = joblib.load('models/label_encoder.joblib')
        artifacts['clf_model'] = joblib.load('models/best_classification_model.joblib')
        artifacts['reg_model'] = joblib.load('models/best_regression_model.joblib')
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None
    return artifacts

def preprocess_input(data, preprocessor):
    """Preprocess input dataframe using the loaded preprocessor."""
    # Calculate derived features
    # Total Monthly Expenses
    expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 'travel_expenses', 
                    'groceries_utilities', 'other_monthly_expenses', 'current_emi_amount']
    
    # Ensure all columns exist, fill with 0 if not
    for col in expense_cols:
        if col not in data.columns:
            data[col] = 0
            
    data['total_monthly_expenses'] = data[expense_cols].sum(axis=1)
    
    # Debt to Income Ratio
    data['debt_to_income_ratio'] = data['total_monthly_expenses'] / data['monthly_salary'].replace(0, 1)
    
    # Savings Potential
    data['savings_potential'] = data['monthly_salary'] - data['total_monthly_expenses']
    
    # Loan Amount to Income Ratio
    data['loan_to_income_ratio'] = data['requested_amount'] / (data['monthly_salary'] * 12).replace(0, 1)
    
    # Transform
    # Ensure columns match the types expected by the preprocessor
    # Extract categorical columns from the preprocessor
    try:
        # Access the transformers from the ColumnTransformer
        # transformers_ is a list of (name, transformer, columns)
        # We look for 'ord' and 'nom' transformers
        
        cat_cols = []
        for name, trans, cols in preprocessor.transformers:
            if name in ['ord', 'nom']:
                # cols can be a list of column names
                cat_cols.extend(cols)
        
        # Cast these columns to string
        for col in cat_cols:
            if col in data.columns:
                data[col] = data[col].astype(str)
                
    except Exception as e:
        # Fallback if structure is different
        print(f"Warning: Could not enforce categorical types: {e}")
        # Fallback to previous method
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            data[col] = data[col].astype(str)
    
    return preprocessor.transform(data)

def predict_emi_eligibility(data, artifacts):
    """Predict EMI Eligibility."""
    preprocessor = artifacts['preprocessor']
    model = artifacts['clf_model']
    le = artifacts['label_encoder']
    
    processed_data = preprocess_input(data, preprocessor)
    prediction_idx = model.predict(processed_data)
    prediction_label = le.inverse_transform(prediction_idx)
    
    proba = model.predict_proba(processed_data) if hasattr(model, "predict_proba") else None
    
    return prediction_label[0], proba[0] if proba is not None else None

def predict_max_emi(data, artifacts):
    """Predict Max Monthly EMI."""
    preprocessor = artifacts['preprocessor']
    model = artifacts['reg_model']
    
    processed_data = preprocess_input(data, preprocessor)
    prediction = model.predict(processed_data)
    
    return prediction[0]
