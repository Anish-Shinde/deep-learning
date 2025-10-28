import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

def show_eda_components(df):
    """Display EDA components like value counts and correlations."""
    if df is None:
        return
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    with st.expander("📉 Value Counts"):
        col = st.selectbox("Select column", all_cols, key="value_counts_col")
        st.write(df[col].value_counts())

    with st.expander("📊 Correlation Matrix"):
        show_correlation_matrix(df, numeric_cols)

    with st.expander("📋 Column Statistics"):
        show_column_statistics(df, all_cols)

def show_correlation_matrix(df, numeric_cols):
    """Display correlation matrix analysis."""
    if len(numeric_cols) > 1:
        corr_method = st.selectbox(
            "Correlation method",
            ["pearson", "kendall", "spearman"],
            key="corr_method"
        )
        
        corr = df[numeric_cols].corr(method=corr_method)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)
        
        if st.checkbox("Show Correlation Pairs"):
            corr_pairs = corr.unstack().sort_values(ascending=False)
            corr_pairs = corr_pairs[corr_pairs != 1]
            st.write(corr_pairs.head(10))
    else:
        st.warning("Need at least 2 numeric columns for correlation analysis")

def show_column_statistics(df, all_cols):
    """Display detailed column statistics with visualizations."""
    col = st.selectbox("Select a column", all_cols, key="col_stats")
    
    if is_numeric_dtype(df[col]):
        show_numeric_column_stats(df, col)
    else:
        show_categorical_column_stats(df, col)

def show_numeric_column_stats(df, col):
    """Display statistics and plots for numeric columns."""
    st.write(df[col].describe())
    tab1, tab2, tab3 = st.tabs(["Histogram", "Box Plot", "Violin Plot"])
    
    with tab1:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)
    
    with tab2:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        st.pyplot(fig)
    
    with tab3:
        fig, ax = plt.subplots()
        sns.violinplot(x=df[col], ax=ax)
        st.pyplot(fig)

def show_categorical_column_stats(df, col):
    """Display statistics and plots for categorical columns."""
    tab1, tab2 = st.tabs(["Bar Chart", "Pie Chart"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 6))
        df[col].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)
    
    with tab2:
        fig, ax = plt.subplots(figsize=(8, 8))
        df[col].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)