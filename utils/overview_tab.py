import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

def show_overview_tab(df):
    """Display the Overview tab content."""
    st.subheader("📊 Dataset Overview")
    if df is None or df.empty:
        st.info("💡 Please upload a CSV file using the file uploader above to view the dataset overview.")
        return

    st.info(f"💡 Showing first 5 rows of your dataset. Total: **{df.shape[0]}** rows, **{df.shape[1]}** columns")
    st.dataframe(df.head(), width='stretch')
    
    with st.expander("🧾 Data Summary"):
        st.caption("💡 Basic information about your dataset structure")
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
        st.caption("💡 Statistical summary: mean, median, std, min, max, quartiles, etc.")
        all_cols = df.columns.tolist()
        selected = st.multiselect(
            "Select columns for statistics", 
            all_cols, 
            key="desc_cols",
            help="Choose columns to see detailed statistics. Numeric columns show more statistics than categorical ones."
        )
        if selected:
            st.dataframe(df[selected].describe(include='all'), height=400)
        else:
            st.info("ℹ️ Select one or more columns to view their descriptive statistics")

    with st.expander("⚠️ Missing Values"):
        st.caption("💡 Check which columns have missing data and how many")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Count': missing.values,
                'Percentage': (missing.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, height=300)
            st.warning(f"⚠️ {len(missing)} column(s) have missing values. Consider using the Data Transformation tab to fill them.")
        else:
            st.success("✅ No missing values detected! Your dataset is complete.")

    st.markdown("---")
    st.info("💡 Download your current dataset (including any transformations you've made)")
    st.download_button(
        "⬇️ Download Current Dataset",
        df.to_csv(index=False),
        "cleaned_data.csv",
        "text/csv",
        help="Download the dataset in its current state as a CSV file"
    )