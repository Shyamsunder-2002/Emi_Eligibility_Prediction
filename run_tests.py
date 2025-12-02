"""
Simplified Test Suite for EMIPredict AI Platform
Tests core functionality without complex output formatting
"""

import os
import sys
import pandas as pd
import joblib
from pathlib import Path

def test_project():
    """Run all tests and return results"""
    results = {
        "passed": [],
        "failed": [],
        "warnings": []
    }
    
    base_path = Path(__file__).parent
    print("="*70)
    print("EMIPredict AI - Test Suite")
    print("="*70)
    
    # TEST 1: File Structure
    print("\n[TEST 1] Checking File Structure...")
    required_files = {
        "Dataset": "emi_prediction_dataset.csv",
        "Requirements": "requirements.txt",
        "Preprocessing Script": "src/preprocessing.py",
        "Feature Engineering Script": "src/feature_engineering.py",
        "Model Training Script": "src/model_training.py",
        "Streamlit Main": "app/main.py",
        "Streamlit Utils": "app/utils.py",
        "Dashboard Page": "app/pages/1_Dashboard.py",
        "Prediction Page": "app/pages/2_Prediction.py",
        "Model Performance Page": "app/pages/3_Model_Performance.py",
        "Classification Model": "models/best_classification_model.joblib",
        "Regression Model": "models/best_regression_model.joblib",
        "Label Encoder": "models/label_encoder.joblib"
    }
    
    for name, file_path in required_files.items():
        full_path = base_path / file_path
        if full_path.exists():
            results["passed"].append(f"{name}: EXISTS")
            print(f"  [PASS] {name}")
        else:
            results["failed"].append(f"{name}: MISSING")
            print(f"  [FAIL] {name} - File not found: {file_path}")
    
    # TEST 2: Data Loading
    print("\n[TEST 2] Testing Data Loading...")
    try:
        data_path = base_path / "emi_prediction_dataset.csv"
        df = pd.read_csv(data_path, low_memory=False)
        results["passed"].append(f"Data Loading: SUCCESS (Shape: {df.shape})")
        print(f"  [PASS] Dataset loaded - Shape: {df.shape}")
        
        # Check for required columns
        required_cols = ['emi_eligibility', 'max_monthly_emi', 'monthly_salary', 'credit_score']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if not missing_cols:
            results["passed"].append("Required columns: PRESENT")
            print(f"  [PASS] All required columns present")
        else:
            results["failed"].append(f"Missing columns: {missing_cols}")
            print(f"  [FAIL] Missing columns: {missing_cols}")
            
    except Exception as e:
        results["failed"].append(f"Data Loading: {str(e)}")
        print(f"  [FAIL] Data loading failed: {e}")
        df = None
    
    # TEST 3: Model Loading
    print("\n[TEST 3] Testing Model Loading...")
    models = {}
    
    model_files = {
        "classification": "models/best_classification_model.joblib",
        "regression": "models/best_regression_model.joblib",
        "encoder": "models/label_encoder.joblib"
    }
    
    for model_name, model_path in model_files.items():
        try:
            full_path = base_path / model_path
            model = joblib.load(full_path)
            models[model_name] = model
            model_type = type(model).__name__
            results["passed"].append(f"{model_name} model: LOADED ({model_type})")
            print(f"  [PASS] {model_name.title()} Model - Type: {model_type}")
        except Exception as e:
            results["failed"].append(f"{model_name} model: {str(e)}")
            print(f"  [FAIL] {model_name.title()} Model - {e}")
    
    # TEST 4: Source Modules
    print("\n[TEST 4] Testing Source Modules...")
    sys.path.insert(0, str(base_path / "src"))
    
    try:
        from preprocessing import load_data, clean_data
        results["passed"].append("Preprocessing module: IMPORTED")
        print(f"  [PASS] Preprocessing module imported")
    except Exception as e:
        results["failed"].append(f"Preprocessing module: {str(e)}")
        print(f"  [FAIL] Preprocessing module - {e}")
    
    try:
        from feature_engineering import create_features
        results["passed"].append("Feature engineering module: IMPORTED")
        print(f"  [PASS] Feature engineering module imported")
    except Exception as e:
        results["failed"].append(f"Feature engineering module: {str(e)}")
        print(f"  [FAIL] Feature engineering module - {e}")
    
    # TEST 5: MLflow Integration
    print("\n[TEST 5] Testing MLflow Integration...")
    mlruns_path = base_path / "mlruns"
    
    if mlruns_path.exists():
        experiments = [d for d in mlruns_path.iterdir() if d.is_dir() and d.name != '0']
        total_runs = sum(len(list(exp.glob("*/"))) for exp in experiments)
        
        results["passed"].append(f"MLflow: {len(experiments)} experiments, {total_runs} runs")
        print(f"  [PASS] MLflow directory found")
        print(f"        Experiments: {len(experiments)}")
        print(f"        Total Runs: {total_runs}")
    else:
        results["warnings"].append("MLflow directory not found")
        print(f"  [WARN] MLflow directory not found")
    
    # TEST 6: Streamlit App Structure
    print("\n[TEST 6] Testing Streamlit Application...")
    sys.path.insert(0, str(base_path / "app"))
    
    try:
        with open(base_path / "app" / "main.py", 'r', encoding='utf-8') as f:
            main_content = f.read()
            if "streamlit" in main_content.lower():
                results["passed"].append("Streamlit main app: VALID")
                print(f"  [PASS] Main app imports Streamlit")
            else:
                results["warnings"].append("Streamlit import not found in main.py")
                print(f"  [WARN] Streamlit import not found in main.py")
    except Exception as e:
        results["failed"].append(f"Main app: {str(e)}")
        print(f"  [FAIL] Main app - {e}")
    
    try:
        import utils
        results["passed"].append("App utils: IMPORTED")
        print(f"  [PASS] App utils imported")
    except Exception as e:
        results["failed"].append(f"App utils: {str(e)}")
        print(f"  [FAIL] App utils - {e}")
    
    # Print Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    total = len(results["passed"]) + len(results["failed"]) + len(results["warnings"])
    print(f"Total Tests: {total}")
    print(f"Passed: {len(results['passed'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Warnings: {len(results['warnings'])}")
    
    if results["failed"]:
        print("\nFailed Tests:")
        for fail in results["failed"]:
            print(f"  - {fail}")
    
    if results["warnings"]:
        print("\nWarnings:")
        for warn in results["warnings"]:
            print(f"  - {warn}")
    
    print("="*70)
    
    if len(results["failed"]) == 0:
        print("\n*** ALL CRITICAL TESTS PASSED! ***\n")
        return True
    else:
        print(f"\n*** {len(results['failed'])} TEST(S) FAILED ***\n")
        return False

if __name__ == "__main__":
    success = test_project()
    sys.exit(0 if success else 1)
