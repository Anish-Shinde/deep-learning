# Advanced CSV Data Analyzer + Deep Learning Demos

## Overview

This is a single-file Streamlit application (`csv_analyzer.py`) that provides an interactive web UI to explore, clean, visualize, and run small deep-learning demos on CSV tabular data. It's intended for quick, beginner-friendly exploration of datasets and light experimentation with models.

Key features:

- Upload and inspect CSV files (head, data types, descriptive statistics).
- Cleaning & transformation: drop columns, fill missing values (mean/median/mode/custom), convert column types, filter rows, reset to original.
- Exploratory Data Analysis (EDA): value counts, correlation matrix with heatmap, histograms, box/violin plots, bar/pie charts.
- Visualizations: 2D/3D scatter (Plotly), histograms, box plots, scatter/line/bar, and a 3D scatter for numeric triples.
- PCA: dimensionality reduction with explained variance and 2D/3D plotting of principal components.
- Deep Learning demos (optional, require TensorFlow):
  - MLP for tabular classification/regression with training history and basic evaluation.
  - LSTM demo for sequence data (expects comma-separated numeric sequences in a CSV column). Supports supervised and autoencoder-style training.
  - CNN demos: MNIST, CIFAR-10, and a transfer-learning example (MobileNetV2 on CIFAR10).
  - Simple dense Variational Autoencoder (VAE) for representation learning on numeric features.


## Files

- `csv_analyzer.py` — main Streamlit application (single file). Contains all UI and logic described above.
- `patients.csv` — example dataset present in the workspace (if provided).
- `requirements.txt` — dependency list (install with `pip install -r requirements.txt`).


## How to run (Windows PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r .\requirements.txt
```

3. Run the Streamlit app:

```powershell
streamlit run "c:\Users\nages\OneDrive\Desktop\CSV Aalyzer\csv_analyzer.py"
```

Notes:
- If you don't plan to use the Deep Learning demos, TensorFlow is optional. If you do want DL features, install TensorFlow (`pip install tensorflow`).
- Training large models (CIFAR, MobileNet) may be slow on CPU-only machines. Consider reducing epochs or sampling your dataset.


## Inputs / Outputs (contract)

- Inputs: CSV file uploaded via the app. For the LSTM demo, a column should contain comma-separated numeric sequences (e.g., `0.1,0.2,0.3`).
- Outputs: interactive charts, cleaned dataset download (`cleaned_data.csv`), model weights/history download buttons, evaluation metrics and small visualizations (PCA, latent space when applicable).


## Notable implementation details & known issues

1. TensorFlow import detection:
   - The app attempts to import `tensorflow` and falls back to standalone `keras`. Some DL code paths assume `tf` exists (e.g., `tf.keras.datasets`, `tf.image.resize`), so running DL demos without full TensorFlow may fail. Recommendation: install `tensorflow` for full DL support.

2. Model saving/downloads:
   - Some model download code saves weights directly to an in-memory stream or encodes `get_weights()` as text. This is brittle and may not produce a proper weights file. A more reliable approach is to save to a temporary file (e.g., `tempfile`) and offer that file for download.

3. LSTM sequence parsing:
   - The sequence parser tries to convert comma-separated tokens to floats. If no valid sequences are found or sequences vary widely in length, the code may error or produce unexpected padding. The app should validate and inform how many rows were parsed.

4. Heuristic task detection for MLP:
   - The app auto-detects regression vs classification using a simple heuristic on the target column (numeric with >20 unique values = regression). This can misclassify some label types. Consider an explicit user override.

5. Large datasets and performance:
   - Heavy operations (training CNNs, PCA on very large tables) can be slow or memory-intensive. Consider sampling or adding dataset size checks.


## Suggested low-risk improvements

- Make DL gating stricter: require real TensorFlow for TF-specific demos or adapt demos to standalone Keras where applicable.
- Use `tempfile.NamedTemporaryFile` to save models/weights for reliable downloads.
- Add user-facing validation messages for sequence parsing (rows parsed, max length).
- Add encoding/one-hot pipelines for categorical features selected as model inputs.
- Add `st.spinner()` or progress feedback during long-running training.
- Add a short `README.md` (this file) and a simple example CSV for the LSTM demo.


## Next steps (optional tasks you might want me to do)

- Implement a robust TensorFlow detection and guard DL features.
- Fix model save/download to use temporary files.
- Add a small example CSV and unit tests for the sequence parser.

If you want, I can implement any of those changes now — tell me which one to start with.

---

Last updated: October 27, 2025
