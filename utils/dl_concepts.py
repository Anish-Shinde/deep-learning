import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
import streamlit as st

def show_dl_concepts():
    """Display Deep Learning Concepts visualizations."""
    st.subheader("🧠 Deep Learning Concepts Visualization")
    st.write("""
    This section helps visualize and understand fundamental concepts in Deep Learning 
    without requiring heavy deep learning frameworks.
    """)

    with st.expander("📊 Gradient Descent & Optimization"):
        show_optimization_demo()

    with st.expander("🔄 Backpropagation Visualization"):
        show_backprop_demo()

    with st.expander("🎯 Decision Surfaces & Linear Machines"):
        show_decision_surfaces_demo()

def show_optimization_demo():
    """Show interactive optimization visualization."""
    st.write("""
    Visualize how different optimization techniques work on a 2D loss surface.
    Compare standard gradient descent with momentum and other optimizers.
    """)
    
    # Create a 2D loss surface
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + 2*Y**2
    
    optimizer = st.selectbox(
        "Select Optimizer",
        ["Gradient Descent", "Momentum", "RMSProp", "Adam"],
        key="optim_select"
    )
    learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01, key="lr_slider")
    n_steps = st.slider("Number of Steps", 10, 100, 50, 5, key="steps_slider")
    
    if st.button("Run Optimization"):
        run_optimization(X, Y, Z, optimizer, learning_rate, n_steps)

def run_optimization(X, Y, Z, optimizer, learning_rate, n_steps):
    """Run selected optimization algorithm and show results."""
    point = np.array([1.5, 1.5])
    points_history = [point.copy()]
    
    # Optimization parameters
    momentum = np.zeros(2)
    v = np.zeros(2)
    m = np.zeros(2)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    for _ in range(n_steps):
        grad = np.array([2*point[0], 4*point[1]])
        
        if optimizer == "Gradient Descent":
            point -= learning_rate * grad
        elif optimizer == "Momentum":
            momentum = beta1 * momentum + (1 - beta1) * grad
            point -= learning_rate * momentum
        elif optimizer == "RMSProp":
            v = beta2 * v + (1 - beta2) * grad**2
            point -= learning_rate * grad / (np.sqrt(v) + eps)
        elif optimizer == "Adam":
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1)
            v_hat = v / (1 - beta2)
            point -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        
        points_history.append(point.copy())
    
    plot_optimization_results(X, Y, Z, points_history, optimizer)

def plot_optimization_results(X, Y, Z, points_history, optimizer):
    """Plot optimization path and loss curve."""
    # Plot optimization path
    fig = plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=20)
    points_history = np.array(points_history)
    plt.plot(points_history[:, 0], points_history[:, 1], 'r.-', label='Optimization path')
    plt.colorbar(label='Loss')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{optimizer} Optimization Path')
    plt.legend()
    st.pyplot(fig)
    
    # Plot loss curve
    fig = plt.figure(figsize=(10, 4))
    loss_history = [p[0]**2 + 2*p[1]**2 for p in points_history]
    plt.plot(loss_history)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss vs. Optimization Step')
    st.pyplot(fig)

def show_backprop_demo():
    """Show backpropagation visualization."""
    st.write("""
    Visualize how backpropagation works in a simple neural network.
    See how gradients flow backwards through the network.
    """)
    
    n_samples = 100
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(float)
    
    n_hidden = st.slider("Number of Hidden Neurons", 2, 10, 4, key="hidden_slider")
    activation = st.selectbox(
        "Activation Function",
        ["sigmoid", "tanh", "relu"],
        key="activation_select"
    )
    
    visualize_network(X, y, n_hidden, activation, n_samples)

def show_decision_surfaces_demo():
    """Show decision surfaces visualization."""
    st.write("""
    Explore different types of decision surfaces and how they separate data.
    Visualize linear classifiers and hinge loss.
    """)
    
    n_samples = 200
    np.random.seed(42)
    X1 = np.random.randn(n_samples, 2)
    y1 = (X1[:, 0]**2 + X1[:, 1]**2 < 2).astype(int)
    
    classifier_type = st.selectbox(
        "Classifier Type",
        ["Linear SVM", "Logistic Regression", "Neural Network"],
        key="clf_type"
    )
    
    if st.button("Train Classifier"):
        visualize_decision_surface(X1, y1, classifier_type)

def activation_fn(x, name):
    """Compute activation function output."""
    if name == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif name == "tanh":
        return np.tanh(x)
    else:  # relu
        return np.maximum(0, x)

def visualize_network(X, y, n_hidden, activation_name, n_samples):
    """Visualize neural network architecture and activations."""
    # Initialize weights
    np.random.seed(42)
    W1 = np.random.randn(2, n_hidden) / np.sqrt(2)
    W2 = np.random.randn(n_hidden, 1) / np.sqrt(n_hidden)
    
    # Forward pass
    h = activation_fn(X @ W1, activation_name)
    y_pred = activation_fn(h @ W2, "sigmoid")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Input space and predictions
    xx, yy = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    X_grid = np.column_stack([xx.ravel(), yy.ravel()])
    h_grid = activation_fn(X_grid @ W1, activation_name)
    y_grid = activation_fn(h_grid @ W2, "sigmoid")
    
    ax1.contourf(xx, yy, y_grid.reshape(xx.shape), alpha=0.3)
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
    ax1.set_title('Input Space & Decision Boundary')
    plt.colorbar(scatter, ax=ax1)
    
    # Hidden layer activations
    ax2.set_title('Hidden Layer Activations')
    for i in range(n_hidden):
        ax2.plot([-2, 2], [i, i], 'k-', alpha=0.1)
        ax2.scatter(X[:, 0], [i]*n_samples, c=h[:, i], cmap='coolwarm')
    
    st.pyplot(fig)
    
    st.write("### Network Architecture")
    st.write(f"""
    - Input Layer: 2 neurons
    - Hidden Layer: {n_hidden} neurons with {activation_name} activation
    - Output Layer: 1 neuron with sigmoid activation
    """)

def visualize_decision_surface(X1, y1, classifier_type):
    """Visualize decision surface and loss functions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot data
    ax1.scatter(X1[y1==0, 0], X1[y1==0, 1], label='Class 0')
    ax1.scatter(X1[y1==1, 0], X1[y1==1, 1], label='Class 1')
    
    if classifier_type == "Linear SVM":
        # Train and plot SVM decision boundary
        clf = LinearSVC(random_state=42)
        clf.fit(X1, y1)
        
        xx = np.linspace(-3, 3, 100)
        yy = np.linspace(-3, 3, 100)
        XX, YY = np.meshgrid(xx, yy)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)
        
        ax1.contour(XX, YY, Z, levels=[0], colors='k', linestyles='-')
        ax1.contour(XX, YY, Z, levels=[-1, 1], colors='k', linestyles='--')
    
    # Plot loss functions
    x_loss = np.linspace(-3, 3, 100)
    hinge_loss = np.maximum(0, 1 - x_loss)
    zero_one_loss = (x_loss < 0).astype(float)
    log_loss = np.log(1 + np.exp(-x_loss))
    
    ax2.plot(x_loss, hinge_loss, label='Hinge Loss')
    ax2.plot(x_loss, zero_one_loss, label='0-1 Loss')
    ax2.plot(x_loss, log_loss, label='Log Loss')
    ax2.set_xlabel('Decision Function Value')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Loss Functions')
    
    ax1.legend()
    ax1.set_title('Decision Boundary')
    st.pyplot(fig)