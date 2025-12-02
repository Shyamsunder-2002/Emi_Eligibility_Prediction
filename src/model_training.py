import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    return pd.read_csv(filepath)

def train_classification(X_train, y_train, X_test, y_test):
    """Train and evaluate classification models."""
    mlflow.set_experiment("EMI_Eligibility_Classification")
    
    # Encode Target
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Save Label Encoder
    joblib.dump(le, 'models/label_encoder.joblib')
    
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest_Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost_Classifier": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }
    
    best_model = None
    best_accuracy = 0
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}...")
            model.fit(X_train, y_train_enc)
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            
            # Metrics
            acc = accuracy_score(y_test_enc, y_pred)
            prec = precision_score(y_test_enc, y_pred, average='weighted')
            rec = recall_score(y_test_enc, y_pred, average='weighted')
            f1 = f1_score(y_test_enc, y_pred, average='weighted')
            
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            
            if y_pred_proba is not None:
                try:
                    roc_auc = roc_auc_score(y_test_enc, y_pred_proba, multi_class='ovr')
                    mlflow.log_metric("roc_auc", roc_auc)
                except:
                    pass
            
            mlflow.sklearn.log_model(model, name)
            
            print(f"{name} - Accuracy: {acc:.4f}")
            
            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                
    # Save Best Model
    if best_model:
        joblib.dump(best_model, 'models/best_classification_model.joblib')
        print(f"Best Classification Model saved: {best_model}")

def train_regression(X_train, y_train, X_test, y_test):
    """Train and evaluate regression models."""
    mlflow.set_experiment("Max_EMI_Regression")
    
    models = {
        "Linear_Regression": LinearRegression(),
        "Random_Forest_Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost_Regressor": XGBRegressor(random_state=42)
    }
    
    best_model = None
    best_rmse = float('inf')
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            
            mlflow.sklearn.log_model(model, name)
            
            print(f"{name} - RMSE: {rmse:.4f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                
    # Save Best Model
    if best_model:
        joblib.dump(best_model, 'models/best_regression_model.joblib')
        print(f"Best Regression Model saved: {best_model}")

def main():
    processed_dir = os.path.join('data', 'processed')
    os.makedirs('models', exist_ok=True)
    
    train_df = load_data(os.path.join(processed_dir, 'train_final.csv'))
    test_df = load_data(os.path.join(processed_dir, 'test_final.csv'))
    
    target_cls = 'emi_eligibility'
    target_reg = 'max_monthly_emi'
    
    # Prepare Data
    X_train = train_df.drop(columns=[target_cls, target_reg])
    y_train_cls = train_df[target_cls]
    y_train_reg = train_df[target_reg]
    
    X_test = test_df.drop(columns=[target_cls, target_reg])
    y_test_cls = test_df[target_cls]
    y_test_reg = test_df[target_reg]
    
    print("Starting Classification Training...")
    train_classification(X_train, y_train_cls, X_test, y_test_cls)
    
    print("\nStarting Regression Training...")
    train_regression(X_train, y_train_reg, X_test, y_test_reg)

if __name__ == "__main__":
    main()
