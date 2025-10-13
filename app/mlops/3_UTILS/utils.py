import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def print_metrics(metrics):
    print(f"\nPerformances :")
    for k, v in metrics.items():
        print(f"{k.upper()} : {v:.4f}")


def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True, bins=30)
    plt.title("Histogramme des résidus")
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, residuals, alpha=0.3)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Réel")
    plt.ylabel("Résidus")
    plt.title("Résidus vs Réel")
    plt.tight_layout()
    plt.show()


def shap_summary_plot(model, X_df, out_path=None):
    import shap

    explainer = shap.Explainer(model)
    shap_values = explainer(X_df)

    shap.summary_plot(shap_values, X_df, show=False)
    if out_path:
        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
