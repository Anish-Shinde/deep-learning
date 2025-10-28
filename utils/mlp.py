import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
from pandas.api.types import is_numeric_dtype
from .model_utils import model_pickle_bytes

def mlp_section(df, all_cols):
    """Display and handle MLP section."""
    st.write("Train a Multi-Layer Perceptron using scikit-learn.")
    target_col = st.selectbox("Target column", all_cols, key="mlp_target")
    
    # Determine task type
    task_type = st.selectbox("Task Type", ["Auto-detect", "Classification", "Regression"], key="mlp_task")
    task = determine_task_type(df, target_col, task_type)
    
    # Get model parameters
    params = get_mlp_parameters(df, all_cols, target_col)
    
    if st.button("Train MLP"):
        if len(params['features']) < 1:
            st.warning("Select at least one feature.")
            return
        
        train_mlp(df, task, params)

def determine_task_type(df, target_col, task_type):
    """Determine whether to perform classification or regression."""
    if task_type == "Auto-detect":
        return "regression" if is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 20 else "classification"
    return "classification" if task_type == "Classification" else "regression"

def get_mlp_parameters(df, all_cols, target_col):
    """Get all MLP model parameters from user input."""
    default_features = [c for c in all_cols if c != target_col and is_numeric_dtype(df[c])]
    
    return {
        'features': st.multiselect("Feature columns (numeric)", all_cols, default=default_features, key="mlp_features"),
        'hidden_layers': st.text_input("Hidden layer sizes (comma-separated)", "(64,32)", key="mlp_hidden"),
        'activation': st.selectbox("Activation function", ["relu", "tanh", "logistic"], key="mlp_activation"),
        'solver': st.selectbox("Solver", ["adam", "sgd", "lbfgs"], key="mlp_solver"),
        'learning_rate': st.number_input("Learning rate (if using adam/sgd)", 0.001, 1.0, 0.001, format="%.3f", key="mlp_lr"),
        'max_iter': st.slider("Max iterations", 100, 1000, 200, key="mlp_iter"),
        'early_stopping': st.checkbox("Early stopping", value=True, key="mlp_early"),
        'test_size': st.slider("Test size fraction", 0.05, 0.5, 0.2, key="mlp_test"),
        'standardize': st.checkbox("Standardize features (recommended)", value=True, key="mlp_scale")
    }

def train_mlp(df, task, params):
    """Train and evaluate MLP model."""
    # Prepare data
    X = df[params['features']].dropna()
    y = df.loc[X.index, params['target_col']]
    
    # Handle non-numeric targets for classification
    label_mapping = None
    if task == "classification" and not is_numeric_dtype(y):
        y = pd.Categorical(y).codes
        label_mapping = dict(enumerate(pd.Categorical(df[params['target_col']]).categories))
    
    # Standardize if requested
    if params['standardize']:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=params['test_size'], random_state=42
    )
    
    # Parse hidden layer sizes
    try:
        hidden_layer_sizes = eval(params['hidden_layers'])
        if not isinstance(hidden_layer_sizes, tuple):
            hidden_layer_sizes = (hidden_layer_sizes,)
    except:
        st.error("Invalid hidden layer sizes. Use comma-separated integers.")
        st.stop()
    
    # Create and train model
    model = create_mlp_model(task, hidden_layer_sizes, params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Show results
    show_mlp_results(model, task, y_test, y_pred)

def create_mlp_model(task, hidden_layer_sizes, params):
    """Create MLPClassifier or MLPRegressor with specified parameters."""
    model_class = MLPClassifier if task == "classification" else MLPRegressor
    return model_class(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=params['activation'],
        solver=params['solver'],
        learning_rate_init=params['learning_rate'] if params['solver'] in ['adam', 'sgd'] else 0.001,
        max_iter=params['max_iter'],
        early_stopping=params['early_stopping'],
        random_state=42
    )

def show_mlp_results(model, task, y_test, y_pred):
    """Display MLP evaluation results and visualizations."""
    st.success("Training completed!")
    
    if task == "classification":
        show_classification_results(y_test, y_pred)
    else:
        show_regression_results(y_test, y_pred)
    
    show_loss_curves(model)
    
    # Provide model download
    model_bytes = model_pickle_bytes(model)
    st.download_button(
        "Download trained model (pickle)",
        model_bytes,
        f"mlp_{'classifier' if task == 'classification' else 'regressor'}.pkl",
        "application/octet-stream"
    )

def show_classification_results(y_test, y_pred):
    """Show classification metrics and plots."""
    st.write("Test accuracy:", accuracy_score(y_test, y_pred))
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax)
    plt.title("Confusion Matrix")
    st.pyplot(fig)

def show_regression_results(y_test, y_pred):
    """Show regression metrics and plots."""
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Test MSE: {mse:.4f}")
    st.write(f"Test RMSE: {np.sqrt(mse):.4f}")
    
    # Plot predictions vs actual
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predictions vs Actual")
    st.pyplot(fig)

def show_loss_curves(model):
    """Show training loss curves if available."""
    if hasattr(model, 'loss_curve_'):
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(model.loss_curve_)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        
        if hasattr(model, 'validation_scores_'):
            plt.plot(model.validation_scores_)
            plt.legend(['Training loss', 'Validation score'])
        
        st.pyplot(fig)