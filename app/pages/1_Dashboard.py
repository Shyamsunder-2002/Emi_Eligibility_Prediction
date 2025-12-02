import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

st.title("📊 Financial Data Dashboard")

@st.cache_data
def load_data():
    # Load cleaned data (not the one with one-hot encoding, but the readable one)
    # We saved it as emi_dataset_cleaned.csv in preprocessing
    # We converted it to parquet for performance and size
    path_parquet = os.path.join('data', 'processed', 'emi_dataset_cleaned.parquet')
    if os.path.exists(path_parquet):
        return pd.read_parquet(path_parquet)
        
    path_csv = os.path.join('data', 'processed', 'emi_dataset_cleaned.csv')
    if os.path.exists(path_csv):
        return pd.read_csv(path_csv)
    return None

df = load_data()

if df is not None:
    st.markdown(f"**Total Records:** {len(df):,}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("EMI Eligibility Distribution")
        fig_pie = px.pie(df, names='emi_eligibility', title='Eligibility Status', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        st.subheader("EMI Scenario Distribution")
        fig_bar = px.bar(df['emi_scenario'].value_counts().reset_index(), 
                         x='emi_scenario', y='count', 
                         title='Loan Types', labels={'emi_scenario': 'Scenario', 'count': 'Count'})
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Salary vs Max EMI")
        # Sample for scatter plot to avoid performance issues
        sample_df = df.sample(min(5000, len(df)))
        fig_scatter = px.scatter(sample_df, x='monthly_salary', y='max_monthly_emi', 
                                 color='emi_eligibility', title='Monthly Salary vs Max Safe EMI (Sample)')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col4:
        st.subheader("Credit Score Distribution")
        fig_hist = px.histogram(df, x='credit_score', color='emi_eligibility', 
                                title='Credit Score Distribution by Eligibility', nbins=50)
        st.plotly_chart(fig_hist, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
else:
    st.error("Data not found. Please run the preprocessing pipeline first.")
