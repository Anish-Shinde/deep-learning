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
    st.write("Train a linear classifier (logistic regression or SVM) on your data.")
    target_col = st.selectbox("Target column (class label)", all_cols, key="linear_target")
    model_type = st.selectbox("Model type", ["Logistic Regression", "Linear SVM (hinge loss)"], key="linear_type")
    default_features = [c for c in all_cols if c != target_col and is_numeric_dtype(df[c])]
    features = st.multiselect("Feature columns (numeric)", all_cols, default=default_features, key="linear_features")
    standardize = st.checkbox("Standardize features (recommended)", value=True, key="linear_scale")
    test_size = st.slider("Test size fraction", 0.05, 0.5, 0.2, key="linear_test")
    
    if st.button("Train Linear Classifier"):
        if len(features) < 1:
            st.warning("Select at least one feature.")
            return
        
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
            st.error("Error: Target variable appears to be continuous. Linear classifiers require discrete classes. " + 
                    "Consider binning your target variable or using a regression model instead.")
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