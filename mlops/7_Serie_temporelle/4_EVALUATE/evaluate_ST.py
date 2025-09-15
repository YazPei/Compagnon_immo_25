import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import argparse
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error as mape_sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mape_continu(y_true, y_pred, window=3):
    """MAPE continue = écart-type du MAPE en rolling window."""
    mape_series = np.abs((y_true - y_pred) / y_true)
    return pd.Series(mape_series).rolling(window).mean().std()

def fallback_prophet(df_test, cluster_id, model_path):
    df_prophet = df_test[["date", "prix_m2_vente"]].copy()
    df_prophet.columns = ["ds", "y"]
    model = Prophet()
    model.fit(df_prophet)
    joblib.dump(model, model_path)
    mlflow.set_tag("fallback", "prophet")
    print(f" Cluster {cluster_id} – Prophet utilisé en fallback.")
    return model

def plot_forecast(y_train, y_test, pred_df, cluster_id, output_folder):
    plt.figure(figsize=(15, 5))
    plt.plot(y_train, label="Train réel")
    plt.plot(y_test, label="Test réel")
    plt.plot(pred_df["mean"], "k--", label="Prédiction")
    plt.fill_between(pred_df.index, pred_df["lower"], pred_df["upper"], color="gray", alpha=0.3, label="IC 95%")
    plt.title(f"Cluster {cluster_id} – Forecast pas-à-pas")
    plt.xlabel("Date")
    plt.ylabel("Prix €/m²")
    plt.legend()
    plt.tight_layout()

    path_png = os.path.join(output_folder, f"forecast_cluster_{cluster_id}.png")
    plt.savefig(path_png)
    mlflow.log_artifact(path_png)
    plt.close()



def evaluate_model(cluster_id, df_train, df_test, model_path, output_folder, suffix="", mape_seuil=0.05, mape_cont_seuil=0.08):
    y_train = np.exp(df_train["prix_m2_vente"])
    y_test  = np.exp(df_test["prix_m2_vente"])
    date_index = df_test.index

    try:
        model = joblib.load(model_path)
        exog_test = df_test[model.model.exog_names]

        preds, lowers, uppers = [], [], []
        for t in range(len(exog_test)):
            xt = exog_test.iloc[t:t+1]
            pf = model.get_forecast(steps=1, exog=xt).summary_frame()
            preds.append(np.exp(pf["mean"].iloc[0]))
            lowers.append(np.exp(pf["mean_ci_lower"].iloc[0]))
            uppers.append(np.exp(pf["mean_ci_upper"].iloc[0]))

        df_pred = pd.DataFrame({
            "mean": preds,
            "lower": lowers,
            "upper": uppers
        }, index=date_index[:len(preds)])

        mae_val = mean_absolute_error(y_test[:len(preds)], df_pred["mean"])
        rmse_val = mean_squared_error(y_test[:len(preds)], df_pred["mean"], squared=False)
        mape_val = mape_sklearn(y_test[:len(preds)], df_pred["mean"])
        mape_cont_val = mape_continu(y_test[:len(preds)].values, df_pred["mean"].values)
        forecast_mean_val = df_pred["mean"].mean()

        mlflow.log_param("cluster", cluster_id)
        mlflow.log_metric("MAE", mae_val)
        mlflow.log_metric("RMSE", rmse_val)
        mlflow.log_metric("MAPE", mape_val)
        mlflow.log_metric("MAPE_continue", mape_cont_val)
        mlflow.log_metric("forecast_mean", forecast_mean_val)

        plot_forecast(y_train, y_test, df_pred, cluster_id, output_folder)

        if mape_val > mape_seuil or mape_cont_val > mape_cont_seuil:
            print(f"⚠️ Cluster {cluster_id} – MAPE {mape_val:.2%}, MAPE_CONT {mape_cont_val:.2%} ➤ fallback Prophet")
            model = fallback_prophet(df_test, cluster_id, model_path)
            mlflow.set_tag("fallback", "prophet")
        else:
            print(f"✅ Cluster {cluster_id} – MAE: {mae_val:.2f}€, RMSE: {rmse_val:.2f}€, MAPE: {mape_val:.2%}, MAPE_CONT: {mape_cont_val:.2%}")

        return {
            'cluster': cluster_id,
            'mae': mae_val,
            'rmse': rmse_val,
            'mape': mape_val,
            'mape_cont': mape_cont_val,
            'forecast_mean': forecast_mean_val
        }

    except Exception as e:
        print(f"❌ Cluster {cluster_id} – Erreur : {e}")
        mlflow.set_tag("error", str(e))
        fallback_prophet(df_test, cluster_id, model_path)
        return {
            'cluster': cluster_id,
            'mae': None,
            'rmse': None,
            'mape': None,
            'mape_cont': None,
            'forecast_mean': None
        }


def main(input_folder, output_folder, model_folder, suffix=""):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("ST-SARIMAX-Evaluation")
    results = []

    for cluster_id in range(4):
        run_name = f"cluster_{cluster_id}"
        with mlflow.start_run(run_name=run_name):
            model_path = os.path.join(model_folder, f"cluster_{cluster_id}_sarimax.pkl")
            df_train = pd.read_csv(os.path.join(input_folder, f"train_cluster_{cluster_id}.csv"), sep=";", parse_dates=["date"]).set_index("date")
            df_test  = pd.read_csv(os.path.join(input_folder, f"test_cluster_{cluster_id}.csv"),  sep=";", parse_dates=["date"]).set_index("date")

            res = evaluate_model(cluster_id, df_train, df_test, model_path, output_folder, suffix=suffix)
            results.append(res)

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(output_folder, f"sarimax_global_eval{suffix}.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=str, required=True)
    parser.add_argument("--output-folder", type=str, required=True)
    parser.add_argument("--model-folder", type=str, required=True)
    parser.add_argument("--suffix", type=str, default="", help="Suffixe pour les fichiers de sortie")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    main(args.input_folder, args.output_folder, args.model_folder, suffix=args.suffix)

