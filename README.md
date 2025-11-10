# Advanced CSV Data Analyzer

## Overview

A comprehensive, user-friendly web application built with Streamlit for analyzing, cleaning, and building machine learning models with CSV data files. This tool provides an interactive, no-code interface for data exploration, transformation, and machine learning model training.

## Key Features

### 📊 Data Explorer Tab
- **Dataset Overview**: Quick inspection of your data with head preview, metadata (rows, columns, data types), and descriptive statistics
- **Missing Values Analysis**: Identify and analyze missing data with percentage breakdowns
- **Data Export**: Download cleaned/transformed datasets as CSV files

### 🔧 Data Transformation Tab
- **Column Management**: Drop unwanted columns with confirmation warnings
- **Missing Value Imputation**: Fill missing values using Mean, Median, Mode, or custom values
- **Data Type Conversion**: Convert columns to int, float, str, or category types
- **Row Filtering**: Filter data by numeric ranges or categorical values with preview
- **Reset Functionality**: Restore original dataset with one click

### 📈 Analytics & Modeling Tab
- **Exploratory Data Analysis (EDA)**:
  - Value counts for categorical columns
  - Correlation matrices with heatmaps (Pearson, Kendall, Spearman)
  - Column-level statistics and visualizations (histograms, box plots, violin plots for numeric; bar/pie charts for categorical)
- **Machine Learning Models**:
  - **MLP (Multi-Layer Perceptron)**: Train neural networks for both classification and regression tasks
    - Auto-detects task type (classification vs regression)
    - Configurable hidden layers, activation functions, solvers
    - Training loss curves and comprehensive evaluation metrics
    - Model export functionality

### 🧠 Deep Learning Concepts Tab
- **Optimization Visualizations**: Interactive demos of Gradient Descent, Momentum, RMSProp, and Adam optimizers
- **Backpropagation Demo**: Visualize neural network forward pass and activations
- **Decision Surfaces**: Educational visualizations of decision boundaries and loss functions
- **No TensorFlow Required**: All visualizations run with NumPy and matplotlib

## Project Structure

```
├── csv_analyzer.py          # Main Streamlit application entry point
├── requirements.txt          # Python dependencies
├── utils/                    # Utility modules
│   ├── __init__.py
│   ├── ui_utils.py          # UI setup and styling helpers
│   ├── overview_tab.py      # Data Explorer tab functionality
│   ├── cleaning_tab.py      # Data Transformation tab functionality
│   ├── eda.py               # Exploratory Data Analysis components
│   ├── mlp.py               # MLP model training and evaluation
│   ├── dl_concepts.py       # Deep Learning educational visualizations
│   └── model_utils.py        # Model serialization utilities
├── CODE_OVERVIEW.txt        # Detailed code documentation
├── RUN_INSTRUCTIONS.txt     # Setup and deployment guide
├── README.md                # This file
└── Sample Data Files:
    ├── patients.csv
    └── earthquake_data_tsunami.csv
```

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Internet connection (for first-time package installation)

### Installation & Running

1. **Navigate to project directory:**
```powershell
cd "C:\Users\Aarav Comp\Desktop\submission\deep learning"
```

2. **Create virtual environment:**
```powershell
python -m venv .venv
```

3. **Activate virtual environment:**
```powershell
.\.venv\Scripts\Activate.ps1
```
*If activation fails, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`*

4. **Install dependencies:**
```powershell
pip install -r requirements.txt
```

5. **Run the application:**
```powershell
streamlit run csv_analyzer.py
```

**Alternative Method (More Reliable):**
```powershell
.\.venv\Scripts\python.exe -m streamlit run csv_analyzer.py
```

The app will open automatically in your browser at `http://localhost:8501`

For detailed instructions, troubleshooting, and deployment guide, see `RUN_INSTRUCTIONS.txt`.

## Usage

### Input
- **CSV Files**: Upload any CSV file through the web interface
- **Sample Data**: Try the included `patients.csv` or `earthquake_data_tsunami.csv` files

### Output
- **Cleaned Datasets**: Download transformed data as `cleaned_data.csv`
- **Trained Models**: Export MLP models as pickle files (`.pkl`) for later use
- **Visualizations**: Interactive charts and plots for data analysis
- **Evaluation Metrics**: Accuracy, confusion matrices, classification reports, MSE/RMSE for regression

## Features & Advantages

### User-Friendly Interface
- **No Coding Required**: All operations are point-and-click
- **Comprehensive Error Handling**: Clear error messages with helpful hints
- **Input Validation**: Proactive warnings and guidance throughout
- **Real-time Feedback**: Preview changes before applying transformations

### Robust Data Processing
- **Smart Validation**: Prevents common mistakes (e.g., using target as feature)
- **Missing Value Handling**: Multiple imputation strategies with validation
- **Data Type Management**: Intelligent type conversion with error handling
- **Filter Preview**: See how many rows will remain before applying filters

### Machine Learning Capabilities
- **Flexible MLP Models**: Supports both classification and regression
- **Auto-detection**: Automatically determines task type or allows manual override
- **Comprehensive Metrics**: Detailed evaluation with visualizations
- **Model Export**: Save trained models for production use

### Educational Value
- **Deep Learning Concepts**: Interactive visualizations without heavy frameworks
- **Optimization Demos**: Learn how different optimizers work
- **Neural Network Visualization**: Understand forward pass and backpropagation

## Technologies Used

### Core Framework
- **Streamlit**: Web application framework for interactive data apps

### Data Processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing and array operations

### Machine Learning
- **Scikit-learn**: MLP models (MLPClassifier, MLPRegressor), preprocessing (StandardScaler), metrics, and train-test split

### Visualization
- **Matplotlib**: Static plotting and charts
- **Seaborn**: Statistical data visualization and heatmaps

### Model Serialization
- **Joblib**: Model saving and loading (pickle format)

## Dependencies

All required packages are listed in `requirements.txt`:
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn

**Note**: TensorFlow is optional and only needed if you plan to implement additional deep learning features beyond the educational visualizations.

## Documentation

- **CODE_OVERVIEW.txt**: Comprehensive documentation of all modules, algorithms, and packages
- **RUN_INSTRUCTIONS.txt**: Detailed setup, running, and deployment instructions
- **HOW_TO_USE_DIAGRAMS.txt**: Guide for generating project diagrams

## Deployment

### Streamlit Cloud
This app can be easily deployed to Streamlit Cloud:
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Deployments happen automatically on every push (1-3 minutes)

See `RUN_INSTRUCTIONS.txt` for detailed deployment information.

## Recent Updates

- ✅ Removed Linear Classifier feature (simplified to MLP only)
- ✅ Enhanced error handling and user guidance throughout
- ✅ Added comprehensive input validation and helpful messages
- ✅ Improved tab names for better clarity
- ✅ Updated all documentation files
- ✅ Added deployment instructions

## License

This project is provided as-is for educational and data analysis purposes.

---

**Last updated**: October 28, 2025
