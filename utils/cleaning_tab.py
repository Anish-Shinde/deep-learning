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
        st.info("💡 **Tip:** Select columns you want to remove from the dataset. This action cannot be undone unless you reset to original data.")
        cols_to_drop = st.multiselect("Select columns to drop", all_cols, key="drop_cols", help="Choose one or more columns to remove from your dataset")
        if cols_to_drop:
            st.warning(f"⚠️ You are about to drop {len(cols_to_drop)} column(s): {', '.join(cols_to_drop)}")
        if st.button("Drop Selected Columns"):
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                st_state.df = df
                st.success(f"✅ Successfully dropped {len(cols_to_drop)} column(s): {', '.join(cols_to_drop)}")
                safe_rerun()
            else:
                st.warning("⚠️ Please select at least one column to drop.")

    with st.expander("💡 Fill Missing Values"):
        fill_missing_values(df, all_cols, st_state)

    with st.expander("🔢 Convert Data Type"):
        convert_data_types(df, all_cols, st_state)

    with st.expander("🔍 Filter Rows"):
        filter_rows(df, all_cols, st_state)

    st.markdown("---")
    st.info("💡 **Reset Option:** Restore your dataset to its original state (undo all transformations)")
    if st.button("🔄 Reset to Original Data"):
        if st_state.original_df is not None:
            st_state.df = st_state.original_df.copy()
            st.success("✅ Data reset to original state! All transformations have been undone.")
            safe_rerun()
        else:
            st.error("❌ No original data available to reset to.")

def fill_missing_values(df, all_cols, st_state):
    """Handle missing value imputation."""
    if df is None or df.empty:
        st.warning("No data available for cleaning.")
        return
    
    col = st.selectbox("Select a column", all_cols, key="fill_col", help="Choose the column that contains missing values")
    
    # Show missing value count for selected column
    if col:
        missing_count = df[col].isna().sum()
        total_count = len(df[col])
        if missing_count > 0:
            st.info(f"📊 Column '{col}' has {missing_count} missing value(s) out of {total_count} total ({missing_count/total_count*100:.1f}%)")
        else:
            st.success(f"✅ Column '{col}' has no missing values!")
    
    method = st.selectbox(
        "Fill method", 
        ["Mean", "Median", "Mode", "Custom Value"], 
        key="fill_method",
        help="Mean/Median: For numeric columns only. Mode: Works for any column. Custom: Enter your own value."
    )
    custom_value = None
    if method == "Custom Value":
        custom_value = st.text_input(
            "Enter custom value", 
            key="custom_fill",
            help="Enter the value you want to use to fill missing values. For numeric columns, enter a number."
        )
        if custom_value and custom_value.strip():
            st.info(f"💡 Will fill missing values with: '{custom_value}'")
    
    if st.button("Fill Missing"):
        try:
            if method == "Mean" and is_numeric_dtype(df[col]):
                if df[col].isna().any():
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    st.info(f"No missing values found in {col}.")
                    return
            elif method == "Median" and is_numeric_dtype(df[col]):
                if df[col].isna().any():
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    st.info(f"No missing values found in {col}.")
                    return
            elif method == "Mode":
                if df[col].isna().any():
                    if not df[col].mode().empty:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    else:
                        st.warning(f"No mode available to fill {col}.")
                        return
                else:
                    st.info(f"No missing values found in {col}.")
                    return
            elif method == "Custom Value" and custom_value is not None and custom_value.strip():
                if df[col].isna().any():
                    try:
                        if is_numeric_dtype(df[col]):
                            df[col].fillna(float(custom_value), inplace=True)
                        else:
                            df[col].fillna(custom_value, inplace=True)
                    except ValueError:
                        df[col].fillna(custom_value, inplace=True)
                else:
                    st.info(f"No missing values found in {col}.")
                    return
            else:
                st.warning("Cannot apply selected fill method for this column. Check if column is numeric for Mean/Median.")
                return
            
            st_state.df = df
            st.success(f"Filled missing values in {col} using {method}.")
            safe_rerun()
        except Exception as e:
            st.error(f"Error filling missing values: {str(e)}")

def convert_data_types(df, all_cols, st_state):
    """Handle data type conversion."""
    if df is None or df.empty:
        st.warning("No data available for conversion.")
        return
    
    col = st.selectbox("Column to convert", all_cols, key="convert_col", help="Select the column whose data type you want to change")
    
    # Show current data type
    if col:
        current_dtype = str(df[col].dtype)
        st.info(f"📊 Current data type: **{current_dtype}**")
    
    dtype = st.selectbox(
        "New type", 
        ["int", "float", "str", "category"], 
        key="convert_type",
        help="int: Whole numbers | float: Decimal numbers | str: Text | category: Categorical (memory efficient)"
    )
    
    if col and dtype:
        # Show warning if conversion might fail
        if dtype == "int" and not is_numeric_dtype(df[col]):
            st.warning("⚠️ Converting non-numeric column to int may fail. Consider converting to float first or handling non-numeric values.")
        elif dtype == "float" and not is_numeric_dtype(df[col]):
            st.warning("⚠️ Converting non-numeric column to float may result in NaN values for non-numeric entries.")
    
    if st.button("Convert Type"):
        try:
            # Handle special cases for int conversion (may have NaN)
            if dtype == "int":
                # Try to convert to float first to handle NaN, then to int
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            else:
                df[col] = df[col].astype(dtype)
            st_state.df = df
            st.success(f"Converted {col} to {dtype}")
            safe_rerun()
        except Exception as e:
            st.error(f"Error converting data type: {str(e)}")
            st.info("Hint: For int conversion, ensure the column contains only numeric values or handle missing values first.")

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
    st.info(f"💡 Current range: {min_val:.2f} to {max_val:.2f}. Adjust the slider to filter rows.")
    selected_range = st.slider(
        "Select range", 
        min_val, max_val, 
        (min_val, max_val), 
        key="num_filter",
        help="Drag the sliders to set the minimum and maximum values to keep"
    )
    
    # Show preview of how many rows will remain
    preview_count = len(df[(df[filter_col] >= selected_range[0]) & (df[filter_col] <= selected_range[1])])
    if preview_count < len(df):
        st.warning(f"⚠️ This filter will reduce rows from {len(df)} to {preview_count} ({len(df) - preview_count} rows will be removed)")
    else:
        st.info(f"ℹ️ All {len(df)} rows will be kept (no filtering applied)")
    
    if st.button("Apply Numeric Filter"):
        if selected_range[0] == min_val and selected_range[1] == max_val:
            st.info("ℹ️ No filtering applied - range includes all values.")
        else:
            df_filtered = df[(df[filter_col] >= selected_range[0]) & (df[filter_col] <= selected_range[1])]
            st_state.df = df_filtered
            st.success(f"✅ Filter applied! Rows remaining: {len(df_filtered)} (removed {len(df) - len(df_filtered)} rows)")
            safe_rerun()

def filter_categorical_column(df, filter_col, st_state):
    """Filter categorical column values."""
    unique_values = df[filter_col].unique().tolist()
    st.info(f"💡 Column '{filter_col}' has {len(unique_values)} unique value(s). Select which ones to keep.")
    selected_values = st.multiselect(
        "Select values to keep",
        unique_values,
        default=unique_values,
        key="cat_filter",
        help="Select one or more values. Only rows with these values will be kept."
    )
    
    # Show preview
    if selected_values:
        preview_count = len(df[df[filter_col].isin(selected_values)])
        if preview_count < len(df):
            st.warning(f"⚠️ This filter will reduce rows from {len(df)} to {preview_count} ({len(df) - preview_count} rows will be removed)")
        else:
            st.info(f"ℹ️ All {len(df)} rows will be kept")
    else:
        st.error("❌ No values selected! Please select at least one value to keep.")
    
    if st.button("Apply Categorical Filter"):
        if not selected_values:
            st.error("❌ Please select at least one value to keep.")
        elif len(selected_values) == len(unique_values):
            st.info("ℹ️ No filtering applied - all values selected.")
        else:
            df_filtered = df[df[filter_col].isin(selected_values)]
            st_state.df = df_filtered
            st.success(f"✅ Filter applied! Rows remaining: {len(df_filtered)} (removed {len(df) - len(df_filtered)} rows)")
            safe_rerun()