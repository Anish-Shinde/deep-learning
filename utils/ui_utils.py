import streamlit as st

# CSS Styles for the application
CSS_STYLES = """
    .main {
        background-color: #f0f2f6;
    }
    h1 {
        color: #2e86de;
    }
    .st-bq {
        border-left: 4px solid #2e86de;
        padding-left: 1rem;
    }
    .stButton>button {
        background-color: #2e86de;
        color: white;
        border-radius: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px 4px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e86de;
        color: white;
    }
"""

def setup_page():
    """Configure page settings and apply styles."""
    st.set_page_config(page_title="Advanced CSV Data Analyzer", layout="wide")
    st.markdown(f"<style>{CSS_STYLES}</style>", unsafe_allow_html=True)

def safe_rerun():
    """Safely rerun the Streamlit app."""
    try:
        # Use st.rerun() for newer Streamlit versions, fallback to experimental_rerun() for older versions
        if hasattr(st, 'rerun'):
            st.rerun()
        else:
            st.experimental_rerun()
    except Exception:
        pass