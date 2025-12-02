import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Load dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Perform data cleaning and preprocessing."""
    # 1. Standardize column names
    df.columns = [col.strip().lower() for col in df.columns]
    
    # 2. Handle inconsistencies in Categorical columns
    # Gender: Fix case sensitivity (e.g., 'female', 'Female')
    if 'gender' in df.columns:
        df['gender'] = df['gender'].str.title()
    
    # 3. Handle Missing Values
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Impute Categorical with Mode
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"Imputed missing values in {col} with mode: {mode_val}")
            
    # Impute Numerical with Median
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed missing values in {col} with median: {median_val}")

    # 4. Basic Data Validation/Cleaning
    # Ensure financial columns are numeric
    financial_cols = ['monthly_salary', 'monthly_rent', 'school_fees', 'college_fees', 
                      'travel_expenses', 'groceries_utilities', 'other_monthly_expenses', 
                      'current_emi_amount', 'bank_balance', 'emergency_fund', 'requested_amount']
    
    for col in financial_cols:
        if col in df.columns:
            # Coerce to numeric, turning errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill NaN (resulting from coercion) with median or 0
            df[col] = df[col].fillna(0)
            df[col] = df[col].apply(lambda x: max(0, x)) # Ensure no negative values

    return df

def main():
    raw_data_path = os.path.join('data', 'raw', 'emi_prediction_dataset.csv')
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    df = load_data(raw_data_path)
    
    if df is not None:
        # Clean Data
        df_cleaned = clean_data(df)
        
        # Save Cleaned Data
        cleaned_path = os.path.join(processed_dir, 'emi_dataset_cleaned.csv')
        df_cleaned.to_csv(cleaned_path, index=False)
        print(f"Cleaned data saved to {cleaned_path}")
        
        # Split Data (80% Train, 20% Test)
        train_df, test_df = train_test_split(df_cleaned, test_size=0.2, random_state=42)
        
        train_path = os.path.join(processed_dir, 'train.csv')
        test_path = os.path.join(processed_dir, 'test.csv')
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Train set saved to {train_path} (Shape: {train_df.shape})")
        print(f"Test set saved to {test_path} (Shape: {test_df.shape})")

if __name__ == "__main__":
    main()
