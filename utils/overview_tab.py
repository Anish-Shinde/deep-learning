import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

def show_overview_tab(df):
    """Display the Overview tab content."""
    st.subheader("🔍 Dataset Overview")
    if df is None:
        st.info("📁 Please upload a CSV file to get started.")
        return

    st.dataframe(df.head(), width='stretch')
    
    with st.expander("🧾 Data Summary"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Rows:**", df.shape[0])
            st.write("**Columns:**", df.shape[1])
            st.write("**Column Names:**", list(df.columns))
        with col2:
            st.write("**Data Types:**")
            dtypes_df = df.dtypes.reset_index()
            dtypes_df.columns = ["Column", "Type"]
            dtypes_df["Type"] = dtypes_df["Type"].astype(str)
            st.dataframe(dtypes_df, height=300, width='stretch')

    with st.expander("📈 Descriptive Statistics"):
        all_cols = df.columns.tolist()
        selected = st.multiselect("Select columns for statistics", all_cols, key="desc_cols")
        if selected:
            st.dataframe(df[selected].describe(include='all'), height=400)

    with st.expander("⚠️ Missing Values"):
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            st.dataframe(missing)
        else:
            st.success("No missing values detected!")

    st.download_button(
        "⬇️ Download Current Dataset",
        df.to_csv(index=False),
        "cleaned_data.csv",
        "text/csv"
    )