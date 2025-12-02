"""
Comprehensive Test Suite for EMIPredict AI Platform
Tests data processing, feature engineering, model loading, and predictions
"""


import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.base_path = Path(__file__).parent
        
    def print_header(self, text):
        print(f"\n{BLUE}{'='*70}{RESET}")
        print(f"{BLUE}{text.center(70)}{RESET}")
        print(f"{BLUE}{'='*70}{RESET}\n")
    
    def print_test(self, test_name, status, message=""):
        if status == "PASS":
            self.passed += 1
            print(f"{GREEN}[PASS]{RESET} {test_name}")
            if message:
                print(f"  {message}")
        elif status == "FAIL":
            self.failed += 1
            print(f"{RED}[FAIL]{RESET} {test_name}")
            if message:
                print(f"  {RED}{message}{RESET}")
        elif status == "WARN":
            self.warnings += 1
            print(f"{YELLOW}[WARN]{RESET} {test_name}")
            if message:
                print(f"  {YELLOW}{message}{RESET}")
    
    def print_summary(self):
        total = self.passed + self.failed + self.warnings
        print(f"\n{BLUE}{'='*70}{RESET}")
        print(f"{BLUE}TEST SUMMARY{RESET}")
        print(f"{BLUE}{'='*70}{RESET}")
        print(f"Total Tests: {total}")
        print(f"{GREEN}Passed: {self.passed}{RESET}")
        print(f"{RED}Failed: {self.failed}{RESET}")
        print(f"{YELLOW}Warnings: {self.warnings}{RESET}")
        
        if self.failed == 0:
            print(f"\n{GREEN}{'*** ALL TESTS PASSED! ***'.center(70)}{RESET}")
        else:
            print(f"\n{RED}{'*** SOME TESTS FAILED ***'.center(70)}{RESET}")
        print(f"{BLUE}{'='*70}{RESET}\n")

def test_file_structure(runner):
    """Test if all required files and directories exist"""
    runner.print_header("TEST 1: FILE STRUCTURE")
    
    required_files = [
        "emi_prediction_dataset.csv",
        "requirements.txt",
        "src/preprocessing.py",
        "src/feature_engineering.py",
        "src/model_training.py",
        "app/main.py",
        "app/utils.py",
        "app/pages/1_Dashboard.py",
        "app/pages/2_Prediction.py",
        "app/pages/3_Model_Performance.py",
        "models/best_classification_model.joblib",
        "models/best_regression_model.joblib",
        "models/label_encoder.joblib"
    ]
    
    for file_path in required_files:
        full_path = runner.base_path / file_path
        if full_path.exists():
            runner.print_test(f"File exists: {file_path}", "PASS")
        else:
            runner.print_test(f"File exists: {file_path}", "FAIL", f"Missing: {full_path}")

def test_data_loading(runner):
    """Test if dataset can be loaded and has correct structure"""
    runner.print_header("TEST 2: DATA LOADING")
    
    try:
        data_path = runner.base_path / "emi_prediction_dataset.csv"
        df = pd.read_csv(data_path)
        
        runner.print_test("Dataset loaded successfully", "PASS", f"Shape: {df.shape}")
        
        # Check expected columns
        expected_features = [
            'age', 'gender', 'marital_status', 'education', 'monthly_salary',
            'employment_type', 'years_of_employment', 'company_type', 'house_type',
            'monthly_rent', 'family_size', 'dependents', 'school_fees', 'college_fees',
            'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
            'existing_loans', 'current_emi_amount', 'credit_score', 'bank_balance',
            'emergency_fund', 'emi_scenario', 'requested_amount', 'requested_tenure'
        ]
        
        expected_targets = ['emi_eligibility', 'max_monthly_emi']
        
        missing_features = [col for col in expected_features if col not in df.columns]
        missing_targets = [col for col in expected_targets if col not in df.columns]
        
        if not missing_features:
            runner.print_test("All feature columns present", "PASS", f"{len(expected_features)} features")
        else:
            runner.print_test("All feature columns present", "FAIL", f"Missing: {missing_features}")
        
        if not missing_targets:
            runner.print_test("All target columns present", "PASS", f"{len(expected_targets)} targets")
        else:
            runner.print_test("All target columns present", "FAIL", f"Missing: {missing_targets}")
        
        # Check data quality
        null_counts = df.isnull().sum().sum()
        if null_counts == 0:
            runner.print_test("No missing values", "PASS")
        else:
            runner.print_test("No missing values", "WARN", f"Found {null_counts} null values")
        
        # Check target distribution
        if 'emi_eligibility' in df.columns:
            eligibility_dist = df['emi_eligibility'].value_counts()
            runner.print_test("EMI Eligibility distribution", "PASS", 
                            f"\n  {eligibility_dist.to_dict()}")
        
        return df
        
    except Exception as e:
        runner.print_test("Dataset loading", "FAIL", str(e))
        return None

def test_preprocessing_module(runner, df):
    """Test preprocessing module"""
    runner.print_header("TEST 3: PREPROCESSING MODULE")
    
    try:
        sys.path.insert(0, str(runner.base_path / "src"))
        from preprocessing import preprocess_data
        
        runner.print_test("Import preprocessing module", "PASS")
        
        # Test with sample data
        if df is not None:
            sample_df = df.head(100).copy()
            try:
                processed_df = preprocess_data(sample_df)
                runner.print_test("Preprocess sample data", "PASS", 
                                f"Input: {sample_df.shape}, Output: {processed_df.shape}")
            except Exception as e:
                runner.print_test("Preprocess sample data", "FAIL", str(e))
        
    except ImportError as e:
        runner.print_test("Import preprocessing module", "FAIL", str(e))
    except Exception as e:
        runner.print_test("Preprocessing module test", "FAIL", str(e))

def test_feature_engineering_module(runner, df):
    """Test feature engineering module"""
    runner.print_header("TEST 4: FEATURE ENGINEERING MODULE")
    
    try:
        sys.path.insert(0, str(runner.base_path / "src"))
        from feature_engineering import engineer_features
        
        runner.print_test("Import feature engineering module", "PASS")
        
        # Test with sample data
        if df is not None:
            sample_df = df.head(100).copy()
            try:
                engineered_df = engineer_features(sample_df)
                runner.print_test("Engineer features on sample data", "PASS",
                                f"Input: {sample_df.shape}, Output: {engineered_df.shape}")
                
                # Check if new features were created
                new_features = set(engineered_df.columns) - set(sample_df.columns)
                if new_features:
                    runner.print_test("New features created", "PASS", 
                                    f"{len(new_features)} new features")
                else:
                    runner.print_test("New features created", "WARN", 
                                    "No new features detected")
            except Exception as e:
                runner.print_test("Engineer features on sample data", "FAIL", str(e))
        
    except ImportError as e:
        runner.print_test("Import feature engineering module", "FAIL", str(e))
    except Exception as e:
        runner.print_test("Feature engineering module test", "FAIL", str(e))

def test_model_loading(runner):
    """Test if trained models can be loaded"""
    runner.print_header("TEST 5: MODEL LOADING")
    
    models = {
        "Classification Model": "models/best_classification_model.joblib",
        "Regression Model": "models/best_regression_model.joblib",
        "Label Encoder": "models/label_encoder.joblib"
    }
    
    loaded_models = {}
    
    for model_name, model_path in models.items():
        try:
            full_path = runner.base_path / model_path
            model = joblib.load(full_path)
            loaded_models[model_name] = model
            
            # Get model type
            model_type = type(model).__name__
            runner.print_test(f"Load {model_name}", "PASS", f"Type: {model_type}")
            
        except Exception as e:
            runner.print_test(f"Load {model_name}", "FAIL", str(e))
    
    return loaded_models

def test_model_predictions(runner, df, models):
    """Test if models can make predictions"""
    runner.print_header("TEST 6: MODEL PREDICTIONS")
    
    if df is None or len(models) == 0:
        runner.print_test("Model predictions", "FAIL", "Missing data or models")
        return
    
    try:
        sys.path.insert(0, str(runner.base_path / "src"))
        from preprocessing import preprocess_data
        from feature_engineering import engineer_features
        
        # Prepare sample data
        sample_df = df.head(10).copy()
        
        # Preprocess and engineer features
        processed_df = preprocess_data(sample_df)
        engineered_df = engineer_features(processed_df)
        
        # Prepare features for prediction
        target_cols = ['emi_eligibility', 'max_monthly_emi']
        X = engineered_df.drop(columns=[col for col in target_cols if col in engineered_df.columns], 
                               errors='ignore')
        
        # Test classification model
        if "Classification Model" in models:
            try:
                clf_model = models["Classification Model"]
                predictions = clf_model.predict(X)
                runner.print_test("Classification predictions", "PASS",
                                f"Predicted {len(predictions)} samples")
                
                # Show sample predictions
                unique_preds = np.unique(predictions)
                runner.print_test("Classification output classes", "PASS",
                                f"Classes: {unique_preds}")
            except Exception as e:
                runner.print_test("Classification predictions", "FAIL", str(e))
        
        # Test regression model
        if "Regression Model" in models:
            try:
                reg_model = models["Regression Model"]
                predictions = reg_model.predict(X)
                runner.print_test("Regression predictions", "PASS",
                                f"Predicted {len(predictions)} samples")
                
                # Show statistics
                runner.print_test("Regression output statistics", "PASS",
                                f"Mean: {predictions.mean():.2f}, Std: {predictions.std():.2f}")
            except Exception as e:
                runner.print_test("Regression predictions", "FAIL", str(e))
                
    except Exception as e:
        runner.print_test("Model predictions", "FAIL", str(e))

def test_mlflow_integration(runner):
    """Test MLflow integration"""
    runner.print_header("TEST 7: MLFLOW INTEGRATION")
    
    mlruns_path = runner.base_path / "mlruns"
    
    if mlruns_path.exists():
        runner.print_test("MLflow directory exists", "PASS")
        
        # Count experiments
        experiments = [d for d in mlruns_path.iterdir() if d.is_dir() and d.name != '0']
        runner.print_test("MLflow experiments", "PASS", 
                        f"Found {len(experiments)} experiment(s)")
        
        # Check for runs
        total_runs = 0
        for exp in experiments:
            runs = list(exp.glob("*/"))
            total_runs += len(runs)
        
        if total_runs > 0:
            runner.print_test("MLflow runs", "PASS", f"Found {total_runs} run(s)")
        else:
            runner.print_test("MLflow runs", "WARN", "No runs found")
    else:
        runner.print_test("MLflow directory exists", "FAIL", "mlruns directory not found")

def test_streamlit_app(runner):
    """Test Streamlit application structure"""
    runner.print_header("TEST 8: STREAMLIT APPLICATION")
    
    try:
        sys.path.insert(0, str(runner.base_path / "app"))
        
        # Test main app
        with open(runner.base_path / "app" / "main.py", 'r') as f:
            main_content = f.read()
            if "st.set_page_config" in main_content:
                runner.print_test("Main app has page config", "PASS")
            else:
                runner.print_test("Main app has page config", "WARN")
        
        # Test pages
        pages = [
            "pages/1_Dashboard.py",
            "pages/2_Prediction.py",
            "pages/3_Model_Performance.py"
        ]
        
        for page in pages:
            page_path = runner.base_path / "app" / page
            if page_path.exists():
                with open(page_path, 'r') as f:
                    content = f.read()
                    if "import streamlit" in content or "import st" in content:
                        runner.print_test(f"Page {page} imports Streamlit", "PASS")
                    else:
                        runner.print_test(f"Page {page} imports Streamlit", "WARN")
            else:
                runner.print_test(f"Page {page} exists", "FAIL")
        
        # Test utils
        try:
            import utils
            runner.print_test("Import app utils", "PASS")
        except ImportError as e:
            runner.print_test("Import app utils", "FAIL", str(e))
            
    except Exception as e:
        runner.print_test("Streamlit app test", "FAIL", str(e))

def main():
    """Run all tests"""
    runner = TestRunner()
    
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{'EMIPredict AI - Comprehensive Test Suite'.center(70)}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    
    # Run all tests
    test_file_structure(runner)
    df = test_data_loading(runner)
    test_preprocessing_module(runner, df)
    test_feature_engineering_module(runner, df)
    models = test_model_loading(runner)
    test_model_predictions(runner, df, models)
    test_mlflow_integration(runner)
    test_streamlit_app(runner)
    
    # Print summary
    runner.print_summary()
    
    return runner.failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
