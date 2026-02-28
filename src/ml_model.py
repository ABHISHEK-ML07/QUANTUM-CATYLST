# src/ml_model.py
"""
Train a RandomForestRegressor on DFT adsorption energies (small demo).
Saves model, predictions, and a PNG compare plot.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from src.ml_features import load_dft_table, build_features

RESULTS = Path("results")
MODELS_DIR = RESULTS / "models"
TABLES_DIR = RESULTS / "tables"
PLOTS_DIR = RESULTS / "plots"

def ensure_dirs():
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def train_and_evaluate(random_state=42):
    ensure_dirs()
    df = load_dft_table()
    X, y, dffull = build_features(df)
    # train-test split (small dataset: use test_size=0.3)
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, dffull, test_size=0.3, random_state=random_state
    )
    # model
    model = RandomForestRegressor(n_estimators=200, random_state=random_state)
    model.fit(X_train, y_train)
    # predict
    y_pred = model.predict(X_test)
    # metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.4f} eV   R2: {r2:.4f}")
    # save model
    model_path = MODELS_DIR / "ml_regressor.pkl"
    joblib.dump(model, model_path)
    print("Saved model to", model_path)
    # save predictions CSV
    out_df = pd.DataFrame({
        "material": df_test["material"].values,
        "dopant": df_test["dopant"].values,
        "adsorbate": df_test["adsorbate"].values,
        "true_energy": y_test.values,
        "predicted_energy": y_pred
    })
    out_csv = TABLES_DIR / "ml_predictions.csv"
    out_df.to_csv(out_csv, index=False)
    print("Saved predictions to", out_csv)
    # plot true vs pred
    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("DFT adsorption energy (eV)")
    ax.set_ylabel("Predicted adsorption energy (eV)")
    ax.set_title(f"ML: MAE={mae:.3f} eV  R2={r2:.3f}")
    plot_path = PLOTS_DIR / "ml_vs_vqe.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print("Saved plot to", plot_path)
    return {
        "model_path": str(model_path),
        "predictions_csv": str(out_csv),
        "plot_path": str(plot_path),
        "mae": float(mae),
        "r2": float(r2)
    }

if __name__ == "__main__":
    res = train_and_evaluate()
    print(res)