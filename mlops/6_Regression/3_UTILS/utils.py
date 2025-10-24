# utils.py — version safe pour CI / headless (sauvegarde les plots au lieu de plt.show)
import os
import numpy as np
import pandas as pd
import matplotlib
# backend non-interactif pour serveurs sans display
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }

def print_metrics(metrics):
    print("\nPerformances :")
    for k, v in metrics.items():
        try:
            print(f"{k.upper()} : {v:.4f}")
        except Exception:
            print(f"{k.upper()} : {v}")

def plot_residuals(y_true, y_pred, out_path=None):
    residuals = np.array(y_true) - np.array(y_pred)
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    sns.histplot(residuals, kde=True, bins=30, ax=ax1)
    ax1.set_title("Histogramme des résidus")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(y_true, residuals, alpha=0.3)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel("Réel"); ax2.set_ylabel("Résidus")
    ax2.set_title("Résidus vs Réel")

    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] residuals plot saved to {out_path}")
    else:
        # fallback: save to a default path
        default = "reports/residuals.png"
        os.makedirs("reports", exist_ok=True)
        fig.savefig(default, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] residuals plot saved to {default}")

def shap_summary_plot(model, X_df, out_path=None):
    """
    Génère et sauvegarde le SHAP summary plot.
    - model : modèle entraîné (sklearn-like)
    - X_df  : DataFrame des features
    - out_path : chemin de sauvegarde (ex: "reports/shap_summary.png")
    """
    try:
        import shap
    except Exception as e:
        print("[WARN] shap non installé :", e)
        return

    # Explainer adaptatif (TreeExplainer quand possible pour perf)
    try:
        if hasattr(shap, "TreeExplainer"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model)
        shap_values = explainer(X_df)
    except Exception as e:
        print("[WARN] impossible de calculer shap values :", e)
        return

    # summary plot -> sauvegarde dans un fichier
    try:
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_df, show=False)
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            plt.tight_layout()
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            print(f"[INFO] SHAP summary saved to {out_path}")
        else:
            default = "reports/shap_summary.png"
            os.makedirs("reports", exist_ok=True)
            plt.tight_layout()
            plt.savefig(default, bbox_inches="tight")
            plt.close()
            print(f"[INFO] SHAP summary saved to {default}")
    except Exception as e:
        print("[WARN] impossible de dessiner/sauvegarder SHAP plot:", e)
