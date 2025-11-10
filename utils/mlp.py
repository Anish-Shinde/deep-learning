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
    if df is None or df.empty:
        st.info("Please upload a CSV file to train an MLP model.")
        return
    
    st.write("Train a Multi-Layer Perceptron using scikit-learn.")
    st.info("💡 **What is an MLP?** A neural network with multiple layers that can learn complex patterns in your data.")
    
    target_col = st.selectbox(
        "Target column", 
        all_cols, 
        key="mlp_target",
        help="Select the column you want to predict (the output variable)"
    )
    
    # Show info about target column
    if target_col:
        unique_count = df[target_col].nunique()
        total_count = len(df[target_col])
        is_numeric = is_numeric_dtype(df[target_col])
        st.info(f"📊 Target column '{target_col}': {unique_count} unique value(s), {'Numeric' if is_numeric else 'Non-numeric'}")
    
    # Determine task type
    task_type = st.selectbox(
        "Task Type", 
        ["Auto-detect", "Classification", "Regression"], 
        key="mlp_task",
        help="Classification: Predict categories/classes. Regression: Predict numeric values. Auto-detect: Let the system decide based on your target column."
    )
    task = determine_task_type(df, target_col, task_type)
    
    if task_type == "Auto-detect":
        st.success(f"✅ Auto-detected task type: **{task.capitalize()}**")
    
    # Get model parameters
    params = get_mlp_parameters(df, all_cols, target_col)
    
    st.markdown("---")
    if st.button("🚀 Train MLP", type="primary"):
        # Pre-validation checks
        validation_errors = []
        
        if len(params['features']) < 1:
            validation_errors.append("❌ Please select at least one feature column.")
        
        if target_col in params['features']:
            validation_errors.append("❌ Target column cannot be used as a feature. Please remove it from feature selection.")
        
        # Validate hidden layer format
        try:
            test_layers = eval(params['hidden_layers'])
            if not isinstance(test_layers, (tuple, list, int)):
                validation_errors.append("❌ Invalid hidden layer format. Use format like (64,32) or (50)")
            elif isinstance(test_layers, (tuple, list)) and not all(isinstance(x, int) and x > 0 for x in test_layers):
                validation_errors.append("❌ All layer sizes must be positive integers.")
        except:
            validation_errors.append("❌ Invalid hidden layer format. Use format like (64,32) or (50)")
        
        # Check for missing values
        if params['features']:
            missing_in_features = df[params['features']].isnull().sum().sum()
            if missing_in_features > 0:
                validation_errors.append(f"⚠️ Warning: {missing_in_features} missing value(s) found in feature columns. They will be automatically removed.")
        
        if target_col:
            missing_in_target = df[target_col].isnull().sum()
            if missing_in_target > 0:
                validation_errors.append(f"⚠️ Warning: {missing_in_target} missing value(s) found in target column. Rows with missing targets will be removed.")
        
        if validation_errors:
            for error in validation_errors:
                st.warning(error)
            if any("❌" in error for error in validation_errors):
                return
        
        # Show training info
        st.info(f"🔄 Training {task} model with {len(params['features'])} feature(s) and hidden layers {params['hidden_layers']}...")
        with st.spinner("Training in progress... This may take a while depending on your data size."):
            train_mlp(df, task, target_col, params)

def determine_task_type(df, target_col, task_type):
    """Determine whether to perform classification or regression."""
    if task_type == "Auto-detect":
        return "regression" if is_numeric_dtype(df[target_col]) and df[target_col].nunique() > 20 else "classification"
    return "classification" if task_type == "Classification" else "regression"

def get_mlp_parameters(df, all_cols, target_col):
    """Get all MLP model parameters from user input."""
    default_features = [c for c in all_cols if c != target_col and is_numeric_dtype(df[c])]
    
    features = st.multiselect(
        "Feature columns (numeric)", 
        all_cols, 
        default=default_features, 
        key="mlp_features",
        help="Select numeric columns to use as input features. More features = more complexity."
    )
    
    if features:
        st.success(f"✅ {len(features)} feature(s) selected")
    
    hidden_layers = st.text_input(
        "Hidden layer sizes (comma-separated)", 
        "(64,32)", 
        key="mlp_hidden",
        help="Enter layer sizes as a tuple, e.g., (64,32) means 2 hidden layers with 64 and 32 neurons. More layers = deeper network."
    )
    st.caption("💡 Example formats: (64,32) for 2 layers, (128,64,32) for 3 layers, (50) for 1 layer")
    
    activation = st.selectbox(
        "Activation function", 
        ["relu", "tanh", "logistic"], 
        key="mlp_activation",
        help="relu: Most common, good for most cases. tanh: Bounded output. logistic: Sigmoid, good for outputs between 0-1."
    )
    
    solver = st.selectbox(
        "Solver", 
        ["adam", "sgd", "lbfgs"], 
        key="mlp_solver",
        help="adam: Adaptive, good for large datasets. sgd: Stochastic gradient descent. lbfgs: Good for small datasets."
    )
    
    learning_rate = st.number_input(
        "Learning rate (if using adam/sgd)", 
        0.001, 1.0, 0.001, 
        format="%.3f", 
        key="mlp_lr",
        help="How fast the model learns. Lower = slower but more stable. Higher = faster but may overshoot."
    )
    if solver in ['adam', 'sgd']:
        st.caption(f"💡 Current learning rate: {learning_rate}")
    
    max_iter = st.slider(
        "Max iterations", 
        100, 1000, 200, 
        key="mlp_iter",
        help="Maximum number of training iterations. More iterations = longer training but potentially better results."
    )
    
    early_stopping = st.checkbox(
        "Early stopping", 
        value=True, 
        key="mlp_early",
        help="Stop training early if model stops improving. Prevents overfitting and saves time."
    )
    
    test_size = st.slider(
        "Test size fraction", 
        0.05, 0.5, 0.2, 
        key="mlp_test",
        help="Percentage of data for testing. 0.2 = 20% test, 80% train."
    )
    st.info(f"📊 Data split: {int((1-test_size)*100)}% training, {int(test_size*100)}% testing")
    
    standardize = st.checkbox(
        "Standardize features (recommended)", 
        value=True, 
        key="mlp_scale",
        help="Scale features to have mean=0 and std=1. Highly recommended for neural networks."
    )
    
    return {
        'features': features,
        'hidden_layers': hidden_layers,
        'activation': activation,
        'solver': solver,
        'learning_rate': learning_rate,
        'max_iter': max_iter,
        'early_stopping': early_stopping,
        'test_size': test_size,
        'standardize': standardize
    }

def train_mlp(df, task, target_col, params):
    """Train and evaluate MLP model."""
    try:
        # Prepare data
        X = df[params['features']].dropna()
        if X.empty:
            st.error("Error: No valid data after dropping missing values. Please check your feature columns.")
            return
        
        y = df.loc[X.index, target_col]
        
        # Validate target variable
        if task == "classification":
            n_unique = y.nunique()
            if is_numeric_dtype(y) and n_unique > 20:
                st.warning(f"Warning: Target has {n_unique} unique values. Consider using regression or binning the target.")
        
        # Handle non-numeric targets for classification
        label_mapping = None
        if task == "classification" and not is_numeric_dtype(y):
            n_unique = y.nunique()
            if n_unique > 10:
                st.error(f"Error: Too many unique classes in target variable ({n_unique}). Consider grouping some classes together.")
                return
            y = pd.Categorical(y).codes
            label_mapping = dict(enumerate(pd.Categorical(df[target_col]).categories))
            st.info(f"Target classes mapped to: {label_mapping}")
        
        # Standardize if requested
        if params['standardize']:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        # Check for missing or infinite values
        if not np.isfinite(X.values).all():
            st.error("Error: Features contain missing or infinite values after preprocessing. Please handle these values before training.")
            return
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X.values, y.values, test_size=params['test_size'], random_state=42
            )
        except Exception as e:
            st.error(f"Error splitting data: {str(e)}")
            return
        
        # Parse hidden layer sizes
        try:
            hidden_layer_sizes = eval(params['hidden_layers'])
            if not isinstance(hidden_layer_sizes, tuple):
                hidden_layer_sizes = (hidden_layer_sizes,)
            # Validate that all are positive integers
            if not all(isinstance(x, int) and x > 0 for x in hidden_layer_sizes):
                raise ValueError("All layer sizes must be positive integers")
        except Exception as e:
            st.error(f"Invalid hidden layer sizes: {str(e)}. Use comma-separated integers like (64,32).")
            return
        
        # Create and train model
        try:
            model = create_mlp_model(task, hidden_layer_sizes, params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Show results
            show_mlp_results(model, task, y_test, y_pred)
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")
            st.info("Hint: Check your data quality, feature selection, and model parameters.")
            return
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your data and try again.")

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