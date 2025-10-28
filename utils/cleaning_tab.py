import streamlit as st
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from .ui_utils import safe_rerun

def show_cleaning_tab(df, st_state):
    """Display the Data Cleaning tab content."""
    st.subheader("🧼 Clean & Transform Data")
    if df is None:
        st.info("Upload CSV to use cleaning tools.")
        return
    
    all_cols = df.columns.tolist()

    with st.expander("📍 Drop Columns"):
        cols_to_drop = st.multiselect("Select columns to drop", all_cols, key="drop_cols")
        if st.button("Drop Selected Columns"):
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                st_state.df = df
                st.success(f"Dropped columns: {cols_to_drop}")
                safe_rerun()

    with st.expander("💡 Fill Missing Values"):
        fill_missing_values(df, all_cols, st_state)

    with st.expander("🔢 Convert Data Type"):
        convert_data_types(df, all_cols, st_state)

    with st.expander("🔍 Filter Rows"):
        filter_rows(df, all_cols, st_state)

    if st.button("🔄 Reset to Original Data"):
        st_state.df = st_state.original_df.copy()
        st.success("Data reset to original state!")
        safe_rerun()

def fill_missing_values(df, all_cols, st_state):
    """Handle missing value imputation."""
    col = st.selectbox("Select a column", all_cols, key="fill_col")
    method = st.selectbox("Fill method", ["Mean", "Median", "Mode", "Custom Value"], key="fill_method")
    custom_value = None
    if method == "Custom Value":
        custom_value = st.text_input("Enter custom value", key="custom_fill")
    
    if st.button("Fill Missing"):
        if method == "Mean" and is_numeric_dtype(df[col]):
            df[col].fillna(df[col].mean(), inplace=True)
        elif method == "Median" and is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)
        elif method == "Mode":
            if not df[col].mode().empty:
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                st.warning(f"No mode available to fill {col}.")
        elif method == "Custom Value" and custom_value is not None:
            try:
                if is_numeric_dtype(df[col]):
                    df[col].fillna(float(custom_value), inplace=True)
                else:
                    df[col].fillna(custom_value, inplace=True)
            except ValueError:
                df[col].fillna(custom_value, inplace=True)
        else:
            st.warning("Cannot apply selected fill method for this column.")
        
        st_state.df = df
        st.success(f"Filled missing values in {col} using {method}.")
        safe_rerun()

def convert_data_types(df, all_cols, st_state):
    """Handle data type conversion."""
    col = st.selectbox("Column to convert", all_cols, key="convert_col")
    dtype = st.selectbox("New type", ["int", "float", "str", "category"], key="convert_type")
    if st.button("Convert Type"):
        try:
            df[col] = df[col].astype(dtype)
            st_state.df = df
            st.success(f"Converted {col} to {dtype}")
            safe_rerun()
        except Exception as e:
            st.error(f"Error: {e}")

def filter_rows(df, all_cols, st_state):
    """Handle row filtering."""
    filter_col = st.selectbox("Select column to filter", all_cols, key="filter_col")
    if is_numeric_dtype(df[filter_col]):
        filter_numeric_column(df, filter_col, st_state)
    else:
        filter_categorical_column(df, filter_col, st_state)

def filter_numeric_column(df, filter_col, st_state):
    """Filter numeric column values."""
    min_val, max_val = float(df[filter_col].min()), float(df[filter_col].max())
    selected_range = st.slider("Select range", min_val, max_val, (min_val, max_val), key="num_filter")
    if st.button("Apply Numeric Filter"):
        df_filtered = df[(df[filter_col] >= selected_range[0]) & (df[filter_col] <= selected_range[1])]
        st_state.df = df_filtered
        st.success(f"Filter applied! Rows remaining: {len(df_filtered)}")
        safe_rerun()

def filter_categorical_column(df, filter_col, st_state):
    """Filter categorical column values."""
    unique_values = df[filter_col].unique().tolist()
    selected_values = st.multiselect(
        "Select values to keep",
        unique_values,
        default=unique_values,
        key="cat_filter"
    )
    if st.button("Apply Categorical Filter"):
        df_filtered = df[df[filter_col].isin(selected_values)]
        st_state.df = df_filtered
        st.success(f"Filter applied! Rows remaining: {len(df_filtered)}")
        safe_rerun()