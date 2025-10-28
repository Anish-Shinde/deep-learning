import streamlit as st
import pandas as pd
from utils.ui_utils import setup_page, safe_rerun
from utils.overview_tab import show_overview_tab
from utils.cleaning_tab import show_cleaning_tab
from utils.eda import show_eda_components
from utils.linear_classifier import linear_classifier_section
from utils.mlp import mlp_section
from utils.dl_concepts import show_dl_concepts

def initialize_session_state():
    """Initialize session state variables."""
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'df' not in st.session_state:
        st.session_state.df = None

def load_data(uploaded_file):
    """Load data from uploaded file and update session state."""
    if uploaded_file:
        try:
            df_local = pd.read_csv(uploaded_file)
            st.session_state.original_df = df_local.copy()
            st.session_state.df = df_local.copy()
            try:
                fname = getattr(uploaded_file, 'name', 'uploaded file')
                st.success(f"Loaded {fname} ({df_local.shape[0]} rows, {df_local.shape[1]} cols)")
            except Exception:
                st.success("Loaded CSV file")
            return df_local
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    return st.session_state.df if st.session_state.df is not None else None

def show_analysis_tab(df):
    """Display Analysis tab content."""
    st.subheader("📈 Exploratory Data Analysis")
    if df is None:
        st.info("Upload CSV to run EDA.")
        return

    all_cols = df.columns.tolist()

    # Linear Classifier
    with st.expander("🤖 Linear Classifier (Logistic/SVM)"):
        linear_classifier_section(df, all_cols)

    # MLP
    with st.expander("🤖 MLP (scikit-learn)"):
        mlp_section(df, all_cols)

    # EDA Components
    show_eda_components(df)

def main():
    # Setup page
    setup_page()
    st.title("Advanced CSV Data Analyzer")

    # File upload and session state
    uploaded_file = st.file_uploader("📁 Upload your CSV file (for tabular and sequence tasks)", type=["csv"])
    initialize_session_state()
    df = load_data(uploaded_file)

    # Main tabs
    tabs = st.tabs(["🔍 Overview", "🧼 Clean Data", "📈 Analysis", "🧠 DL Concepts"])

    # Overview tab
    with tabs[0]:
        show_overview_tab(df)

    # Cleaning tab
    with tabs[1]:
        show_cleaning_tab(df, st.session_state)

    # Analysis tab
    with tabs[2]:
        show_analysis_tab(df)

    # DL Concepts tab
    with tabs[3]:
        show_dl_concepts()

if __name__ == "__main__":
    main()

# End of app
