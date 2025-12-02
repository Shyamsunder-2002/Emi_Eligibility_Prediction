# 💰 EMIPredict AI - Intelligent Financial Risk Assessment Platform

A comprehensive machine learning platform for predicting EMI eligibility and calculating maximum safe EMI amounts for loan applicants.

## 🎯 Features

- **EMI Eligibility Prediction**: Classify applicants as Eligible, High Risk, or Not Eligible
- **Max EMI Calculation**: Predict the maximum safe monthly EMI amount
- **Interactive Dashboard**: Visualize financial data and trends
- **Model Performance Tracking**: Monitor ML models via MLflow
- **Real-time Predictions**: User-friendly Streamlit interface

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd EMIPredict-AI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the data pipeline:
```bash
python src/preprocessing.py
python src/feature_engineering.py
```

4. Train models:
```bash
python src/model_training.py
```

5. Launch the web app:
```bash
streamlit run app/main.py
```

## 📊 Dataset

The project uses a dataset of 400,000 financial records with:
- 22 input features (demographics, income, expenses, credit history)
- 2 target variables (EMI eligibility, max monthly EMI)
- 5 EMI scenarios (E-commerce, Home Appliances, Vehicle, Personal Loan, Education)

## 🧠 Models

### Classification (EMI Eligibility)
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier ⭐ (Best)

### Regression (Max EMI)
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor ⭐ (Best - RMSE: 737.85)

## 📁 Project Structure

```
EMIPredict-AI/
├── data/                   # Data storage
│   ├── raw/                # Original dataset
│   └── processed/          # Cleaned data
├── src/                    # Source code
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── model_training.py
├── app/                    # Streamlit application
│   ├── main.py
│   ├── utils.py
│   └── pages/
├── models/                 # Saved models
└── requirements.txt
```

## 🛠️ Tech Stack

- **ML/Data**: scikit-learn, XGBoost, pandas, numpy
- **Tracking**: MLflow
- **Web App**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn

## 📈 Performance

- **Classification Accuracy**: >90%
- **Regression RMSE**: <2000 INR
- **Processing Speed**: Real-time predictions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Built with ❤️ for financial risk assessment
