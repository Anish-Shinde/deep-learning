import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pandas.api.types import is_numeric_dtype
from .model_utils import model_pickle_bytes

def linear_classifier_section(df, all_cols):
    """Display and handle Linear Classifier section."""
    if df is None or df.empty:
        st.info("Please upload a CSV file to train a linear classifier.")
        return
    
    st.write("Train a linear classifier (logistic regression or SVM) on your data.")
    st.info("💡 **What is a linear classifier?** A model that finds a linear boundary to separate different classes in your data.")
    
    target_col = st.selectbox(
        "Target column (class label)", 
        all_cols, 
        key="linear_target",
        help="Select the column that contains the classes/labels you want to predict"
    )
    
    # Show info about target column
    if target_col:
        unique_count = df[target_col].nunique()
        total_count = len(df[target_col])
        is_numeric = is_numeric_dtype(df[target_col])
        
        st.info(f"📊 Target column '{target_col}': {unique_count} unique value(s) out of {total_count} rows, {'Numeric' if is_numeric else 'Non-numeric'}")
        
        # Check if target is likely continuous
        if is_numeric and unique_count > 10:
            st.error(f"⚠️ **Warning**: Your target has {unique_count} unique numeric values - this looks like a continuous variable!")
            st.warning("""
            **Linear Classifiers are for classification only** (predicting categories like "Yes/No", "Low/Medium/High").
            
            **If your target is continuous** (prices, temperatures, measurements):
            - ❌ Don't use Linear Classifier
            - ✅ Use **MLP with Regression** instead (in the same tab below)
            
            **If you want to classify**:
            - Convert your continuous values into categories first
            - Use the Data Transformation tab to create bins
            """)
        elif unique_count > 10:
            st.warning(f"⚠️ Warning: {unique_count} classes detected. Linear classifiers work best with fewer classes (< 10). Consider grouping some classes.")
    
    model_type = st.selectbox(
        "Model type", 
        ["Logistic Regression", "Linear SVM (hinge loss)"], 
        key="linear_type",
        help="Logistic Regression: Probabilistic, good for binary/multi-class. Linear SVM: Maximizes margin, good for separable data."
    )
    
    default_features = [c for c in all_cols if c != target_col and is_numeric_dtype(df[c])]
    features = st.multiselect(
        "Feature columns (numeric)", 
        all_cols, 
        default=default_features, 
        key="linear_features",
        help="Select numeric columns to use as features. The target column is automatically excluded."
    )
    
    if features:
        st.success(f"✅ {len(features)} feature(s) selected: {', '.join(features[:3])}{'...' if len(features) > 3 else ''}")
    
    standardize = st.checkbox(
        "Standardize features (recommended)", 
        value=True, 
        key="linear_scale",
        help="Standardizing scales all features to have mean=0 and std=1. This is recommended when features have different scales."
    )
    
    test_size = st.slider(
        "Test size fraction", 
        0.05, 0.5, 0.2, 
        key="linear_test",
        help="Percentage of data to use for testing (rest is used for training). 0.2 = 20% test, 80% train."
    )
    st.info(f"📊 Data split: {int((1-test_size)*100)}% training, {int(test_size*100)}% testing")
    
    st.markdown("---")
    if st.button("🚀 Train Linear Classifier", type="primary"):
        # Pre-validation checks
        validation_errors = []
        
        if len(features) < 1:
            validation_errors.append("❌ Please select at least one feature column.")
        
        if target_col in features:
            validation_errors.append("❌ Target column cannot be used as a feature. Please remove it from feature selection.")
        
        # Check for missing values in features
        if features:
            missing_in_features = df[features].isnull().sum().sum()
            if missing_in_features > 0:
                validation_errors.append(f"⚠️ Warning: {missing_in_features} missing value(s) found in feature columns. They will be automatically removed.")
        
        # Check for missing values in target
        if target_col:
            missing_in_target = df[target_col].isnull().sum()
            if missing_in_target > 0:
                validation_errors.append(f"⚠️ Warning: {missing_in_target} missing value(s) found in target column. Rows with missing targets will be removed.")
        
        if validation_errors:
            for error in validation_errors:
                st.warning(error)
            if "❌" in str(validation_errors):
                return
        
        # Show training info
        with st.spinner("🔄 Training model... This may take a moment."):
            train_linear_classifier(
                df, features, target_col, model_type, standardize, test_size
            )

def train_linear_classifier(df, features, target_col, model_type, standardize, test_size):
    """Train and evaluate a linear classifier."""
    try:
        X = df[features].dropna()
        y = df.loc[X.index, target_col]

        # Validate target variable
        n_unique = y.nunique()
        if is_numeric_dtype(y) and n_unique > 10:
            st.error("❌ **Target Variable Issue Detected**")
            st.warning(f"Your target column '{target_col}' has {n_unique} unique numeric values, which suggests it's a continuous variable (like prices, temperatures, or measurements).")
            st.info("""
            **Linear Classifiers** (Logistic Regression, Linear SVM) are designed for **classification** tasks where you predict discrete categories/classes.
            
            **What to do:**
            1. **For continuous predictions** → Use the **MLP (Regression)** model in the Analytics & Modeling tab instead
            2. **For classification** → Convert your continuous target into discrete categories:
               - Go to **Data Transformation** tab
               - Use filtering or create bins (e.g., "Low", "Medium", "High")
               - Or manually edit your CSV to have categories instead of numbers
            """)
            st.write(f"**Current target values range:** {y.min():.2f} to {y.max():.2f}")
            st.write(f"**Sample values:** {', '.join(map(str, y.head(10).tolist()))}")
            return
        
        # Handle non-numeric target
        label_mapping = None
        if not is_numeric_dtype(y):
            if n_unique > 10:
                st.error("Error: Too many unique classes in target variable (> 10). " + 
                        "Consider grouping some classes together.")
                return
            y = pd.Categorical(y).codes
            label_mapping = dict(enumerate(pd.Categorical(df[target_col]).categories))
            st.info(f"Target classes mapped to: {label_mapping}")
        
        # Standardize features if requested
        if standardize:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        # Check for missing or infinite values
        if not np.isfinite(X.values).all():
            st.error("Error: Features contain missing or infinite values after preprocessing. " + 
                    "Please handle these values before training.")
            return
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X.values, y.values, test_size=test_size, random_state=42
            )
        except Exception as e:
            st.error(f"Error splitting data: {str(e)}")
            return
        
        # Train model
        try:
            model = LogisticRegression(random_state=42) if model_type == "Logistic Regression" else LinearSVC(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Display results
            show_classifier_results(
                model, X, y, X_test, y_test, y_pred, features, model_type
            )
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
            st.info("Hint: Make sure your target variable contains discrete classes suitable for classification.")
            return
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your data and try again.")

def show_classifier_results(model, X, y, X_test, y_test, y_pred, features, model_type):
    """Display classifier evaluation results and visualizations."""
    st.success("Training completed!")
    st.write("Test accuracy:", accuracy_score(y_test, y_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax)
    plt.title("Confusion Matrix")
    st.pyplot(fig)
    
    # Plot decision boundary for 2D features
    if len(features) == 2:
        plot_decision_boundary(model, X, y, features, model_type)
    
    # Provide model download
    model_bytes = model_pickle_bytes(model)
    st.download_button(
        "Download trained model (pickle)",
        model_bytes,
        f"linear_classifier_{model_type.lower().replace(' ', '_')}.pkl",
        "application/octet-stream"
    )

def plot_decision_boundary(model, X, y, features, model_type):
    """Plot decision boundary for 2D feature space."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title(f"Decision Boundary ({model_type})")
    st.pyplot(fig)