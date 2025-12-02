import streamlit as st
import mlflow
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")

st.title("📈 Model Performance")

def get_mlflow_data(experiment_name):
    """Fetch runs from MLflow experiment."""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            return runs
    except Exception as e:
        st.error(f"Error fetching MLflow data: {e}")
    return None

st.subheader("Classification Models (EMI Eligibility)")
cls_runs = get_mlflow_data("EMI_Eligibility_Classification")

if cls_runs is not None and not cls_runs.empty:
    # Clean up column names
    cls_metrics = ['metrics.accuracy', 'metrics.precision', 'metrics.recall', 'metrics.f1_score', 'metrics.roc_auc']
    cls_cols = ['tags.mlflow.runName'] + cls_metrics
    
    # Filter available columns
    available_cols = [col for col in cls_cols if col in cls_runs.columns]
    cls_df = cls_runs[available_cols].copy()
    
    # Rename for better display
    rename_dict = {col: col.replace('metrics.', '').replace('tags.mlflow.', '') for col in cls_df.columns}
    rename_dict['tags.mlflow.runName'] = 'Model'
    cls_df.rename(columns=rename_dict, inplace=True)
    
    st.dataframe(cls_df)
    
    # Comparison Chart
    metric_to_plot = st.selectbox("Select Metric to Compare (Classification)", 
                                  ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'])
    
    if metric_to_plot in cls_df.columns:
        fig_cls = px.bar(cls_df, x='Model', y=metric_to_plot, color='Model', 
                         title=f'Model Comparison - {metric_to_plot.capitalize()}',
                         labels={'Model': 'Model', metric_to_plot: metric_to_plot.capitalize()})
        st.plotly_chart(fig_cls, use_container_width=True)
else:
    st.info("No classification experiments found. Run model training first.")

st.markdown("---")

st.subheader("Regression Models (Max EMI)")
reg_runs = get_mlflow_data("Max_EMI_Regression")

if reg_runs is not None and not reg_runs.empty:
    # Clean up column names
    reg_metrics = ['metrics.rmse', 'metrics.mae', 'metrics.r2_score']
    reg_cols = ['tags.mlflow.runName'] + reg_metrics
    
    # Filter available columns
    available_cols = [col for col in reg_cols if col in reg_runs.columns]
    reg_df = reg_runs[available_cols].copy()
    
    # Rename
    rename_dict = {col: col.replace('metrics.', '').replace('tags.mlflow.', '') for col in reg_df.columns}
    rename_dict['tags.mlflow.runName'] = 'Model'
    reg_df.rename(columns=rename_dict, inplace=True)
    
    st.dataframe(reg_df)
    
    # Comparison Chart
    metric_to_plot_reg = st.selectbox("Select Metric to Compare (Regression)", 
                                      ['rmse', 'mae', 'r2_score'])
    
    if metric_to_plot_reg in reg_df.columns:
        fig_reg = px.bar(reg_df, x='Model', y=metric_to_plot_reg, color='Model',
                         title=f'Model Comparison - {metric_to_plot_reg.upper()}',
                         labels={'Model': 'Model', metric_to_plot_reg: metric_to_plot_reg.upper()})
        st.plotly_chart(fig_reg, use_container_width=True)
else:
    st.info("No regression experiments found. Run model training first.")
