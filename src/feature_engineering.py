import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(filepath):
    return pd.read_csv(filepath)

def create_features(df):
    """Create derived features."""
    df = df.copy()
    
    # Financial Ratios
    # Total Monthly Expenses (Sum of all expense columns + current EMI)
    expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 'travel_expenses', 
                    'groceries_utilities', 'other_monthly_expenses', 'current_emi_amount']
    
    df['total_monthly_expenses'] = df[expense_cols].sum(axis=1)
    
    # Debt to Income Ratio (DTI)
    # Avoid division by zero
    df['debt_to_income_ratio'] = df['total_monthly_expenses'] / df['monthly_salary'].replace(0, 1)
    
    # Savings Potential
    df['savings_potential'] = df['monthly_salary'] - df['total_monthly_expenses']
    
    # Loan Amount to Income Ratio
    df['loan_to_income_ratio'] = df['requested_amount'] / (df['monthly_salary'] * 12).replace(0, 1)
    
    return df

def main():
    processed_dir = os.path.join('data', 'processed')
    models_dir = os.path.join('models', 'preprocessors')
    os.makedirs(models_dir, exist_ok=True)
    
    train_df = load_data(os.path.join(processed_dir, 'train.csv'))
    test_df = load_data(os.path.join(processed_dir, 'test.csv'))
    
    # 1. Create Derived Features
    train_df = create_features(train_df)
    test_df = create_features(test_df)
    
    # 2. Define Columns for Transformations
    # Target variables
    target_cls = 'emi_eligibility'
    target_reg = 'max_monthly_emi'
    
    # Drop targets from features
    X_train = train_df.drop(columns=[target_cls, target_reg])
    y_train_cls = train_df[target_cls]
    y_train_reg = train_df[target_reg]
    
    X_test = test_df.drop(columns=[target_cls, target_reg])
    y_test_cls = test_df[target_cls]
    y_test_reg = test_df[target_reg]
    
    # Identify columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Ensure categorical columns are strings
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)
    
    # Specific handling for Ordinal 'education' if present
    ordinal_cols = ['education'] if 'education' in categorical_cols else []
    nominal_cols = [col for col in categorical_cols if col not in ordinal_cols]
    
    # 3. Build Preprocessing Pipeline
    # Numeric: Standard Scaler
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Ordinal: Ordinal Encoder
    education_order = [['High School', 'Graduate', 'Post Graduate', 'Professional']]
    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(categories=education_order, handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    # Nominal: OneHot Encoder
    nominal_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('ord', ordinal_transformer, ordinal_cols),
            ('nom', nominal_transformer, nominal_cols)
        ])
    
    # 4. Fit and Transform
    print("Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after transformation
    # Numeric
    feature_names = numerical_cols
    # Ordinal
    feature_names += ordinal_cols
    # Nominal
    if nominal_cols:
        nom_feature_names = preprocessor.named_transformers_['nom']['onehot'].get_feature_names_out(nominal_cols)
        feature_names += list(nom_feature_names)
        
    # Convert back to DataFrame
    X_train_final = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_final = pd.DataFrame(X_test_processed, columns=feature_names)
    
    # Add targets back
    X_train_final[target_cls] = y_train_cls.values
    X_train_final[target_reg] = y_train_reg.values
    
    X_test_final[target_cls] = y_test_cls.values
    X_test_final[target_reg] = y_test_reg.values
    
    # 5. Save
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.joblib'))
    print(f"Preprocessor saved to {os.path.join(models_dir, 'preprocessor.joblib')}")
    
    X_train_final.to_csv(os.path.join(processed_dir, 'train_final.csv'), index=False)
    X_test_final.to_csv(os.path.join(processed_dir, 'test_final.csv'), index=False)
    
    print("Feature engineering completed.")
    print(f"Train Final Shape: {X_train_final.shape}")
    print(f"Test Final Shape: {X_test_final.shape}")

if __name__ == "__main__":
    main()
