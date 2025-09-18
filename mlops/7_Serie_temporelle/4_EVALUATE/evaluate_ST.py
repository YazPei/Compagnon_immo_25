#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from dotenv import load_dotenv


# charge les variables depuis .env.yaz si présent
if Path(".env.yaz").exists():
    load_dotenv(".env.yaz")

# puis, comme avant :
import mlflow

import os
import re
import glob
import argparse
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import mlflow

try:
    import joblib
except Exception:
    joblib = None
import pickle

from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# ========= MLflow utils =========
def setup_mlflow(exp_name: str = "ST-SARIMAX-Evaluation-remote") -> None:
    """
    Utilise MLFLOW_TRACKING_URI si défini, sinon fallback local file:./mlruns
    Crée/sélectionne l'expérience exp_name.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI")
    try:
        if uri:
            mlflow.set_tracking_uri(uri)
        else:
            raise RuntimeError("MLFLOW_TRACKING_URI non défini")
        mlflow.set_experiment(exp_name)
    except Exception as e:
        print(f"[WARN] MLflow indisponible ({e}). Fallback en local file:./mlruns")
        local_dir = Path.cwd() / "mlruns"
        local_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"file://{local_dir}")
        mlflow.set_experiment(exp_name + " (offline)")


# ========= IO utils =========
def resolve_split_paths(input_folder: str, suffix: Optional[str]) -> Tuple[str, str]:
    """
    Trouve automatiquement les chemins train/test.
    Essaie avec suffix, puis *_q12, puis sans suffixe.
    """
    input_folder = str(Path(input_folder))
    suffix = suffix or ""
    candidates_train = [
        os.path.join(input_folder, f"train_periodique{suffix}.csv"),
        os.path.join(input_folder, "train_periodique_q12.csv"),
        os.path.join(input_folder, "train_periodique.csv"),
    ]
    candidates_test = [
        os.path.join(input_folder, f"test_periodique{suffix}.csv"),
        os.path.join(input_folder, "test_periodique_q12.csv"),
        os.path.join(input_folder, "test_periodique.csv"),
    ]
    train_path = next((p for p in candidates_train if os.path.exists(p)), None)
    test_path = next((p for p in candidates_test if os.path.exists(p)), None)
    if not train_path or not test_path:
        raise FileNotFoundError(
            "Impossible de trouver les splits.\n"
            f"Essayé (train): {candidates_train}\n"
            f"Essayé (test) : {candidates_test}"
        )
    return train_path, test_path


def infer_cluster_ids_from_models(model_folder: str) -> List[Optional[int]]:
    """
    Déduit les cluster_ids depuis les noms des fichiers modèles (regex 'cluster_(\\d+)').
    Si rien, retourne [None] → évaluation globale.
    """
    ids = set()
    for f in glob.glob(os.path.join(model_folder, "*.pkl")):
        m = re.search(r"cluster_(\d+)", os.path.basename(f))
        if m:
            ids.add(int(m.group(1)))
    return sorted(ids) if ids else [None]


def pick_model_for_cluster(model_folder: str, cid: Optional[int]) -> Optional[str]:
    """
    Sélectionne un modèle pour le cluster cid. Si cid=None, prend le premier .pkl.
    """
    if cid is None:
        files = sorted(glob.glob(os.path.join(model_folder, "*.pkl")))
        return files[0] if files else None
    # cherche un modèle contenant 'cluster_<cid>'
    pattern = re.compile(rf"cluster_{cid}\b")
    for f in sorted(glob.glob(os.path.join(model_folder, "*.pkl"))):
        if pattern.search(os.path.basename(f)):
            return f
    # fallback: rien trouvé
    return None


def load_model(model_path: str):
    """
    Charge un modèle .pkl via joblib si dispo, sinon pickle.
    """
    if joblib is not None:
        try:
            return joblib.load(model_path)
        except Exception:
            pass
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ========= Forecast utils =========
def forecast_with_model(model, test_index: pd.DatetimeIndex, steps: int) -> np.ndarray:
    """
    Essaie get_forecast(steps) (statsmodels) puis predict, sinon persistance (naïf).
    """
    # statsmodels SARIMAXResults -> get_forecast
    if hasattr(model, "get_forecast"):
        try:
            fc = model.get_forecast(steps=steps)
            if hasattr(fc, "predicted_mean"):
                return np.asarray(fc.predicted_mean)
            # certains objets renvoient directement une array-like
            arr = np.asarray(fc)
            if arr.shape[0] == steps:
                return arr
        except Exception as e:
            print(f"[WARN] get_forecast a échoué: {e}")

    # API .predict (nombreux modèles)
    if hasattr(model, "predict"):
        try:
            # Beaucoup de modèles acceptent start/end index (pandas index)
            yhat = model.predict(start=test_index[0], end=test_index[-1])
            yhat = np.asarray(yhat)
            # si la shape diffère, tente steps
            if yhat.shape[0] != steps and hasattr(model, "predict"):
                yhat = np.asarray(model.predict(steps=steps))
            return yhat[:steps]
        except Exception as e:
            print(f"[WARN] predict(start/end) a échoué: {e}")

    # Fallback naïf (persistance)
    print("[WARN] Fallback naïf (persistance dernière valeur train) appliqué.")
    # On suppose que le modèle possède l'attribut endog / data.endog ou qu'on ne l'a pas → 0
    last_val = None
    for attr in ("endog", "data", "model"):
        obj = getattr(model, attr, None)
        if obj is None:
            continue
        try:
            if hasattr(obj, "endog"):
                last_val = np.asarray(obj.endog)[-1]
                break
            if hasattr(obj, "y"):
                last_val = np.asarray(obj.y)[-1]
                break
        except Exception:
            continue
    if last_val is None:
        last_val = 0.0
    return np.full(steps, float(last_val))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # MAPE safe (évite division par zéro)
    denom = np.where(np.asarray(y_true) == 0, 1e-8, np.asarray(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape": mape}


# ========= Main =========
def main(input_folder: str, output_folder: str, model_folder: str, suffix: Optional[str] = None) -> None:
    setup_mlflow("ST-SARIMAX-Evaluation")

    y_col = "prix_m2_vente"
    suffix = suffix or ""
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Localise les splits (train/test)
    train_path, test_path = resolve_split_paths(input_folder, suffix)
    print(f"[INFO] train={train_path}\n[INFO] test ={test_path}")

    df_train = pd.read_csv(train_path, sep=";", parse_dates=["date"]).set_index("date")
    df_test = pd.read_csv(test_path, sep=";", parse_dates=["date"]).set_index("date")

    # Déduction des clusters
    cluster_ids = infer_cluster_ids_from_models(model_folder)
    if cluster_ids == [None] and "cluster" in df_train.columns:
        try:
            cluster_ids = sorted(df_train["cluster"].dropna().astype(int).unique().tolist())
        except Exception:
            cluster_ids = [None]

    # Boucle d’évaluation
    for cid in (cluster_ids if cluster_ids != [None] else [None]):
        if cid is not None and "cluster" in df_train.columns:
            tr = df_train[df_train["cluster"] == cid]
            te = df_test[df_test["cluster"] == cid]
            if tr.empty or te.empty:
                print(f"[WARN] Pas de données pour cluster={cid}, on saute.")
                continue
            run_name = f"evaluate_cluster_{cid}{suffix}"
        else:
            tr, te = df_train, df_test
            run_name = f"evaluate_global{suffix}"

        if y_col not in tr.columns or y_col not in te.columns:
            raise KeyError(f"Colonne cible '{y_col}' absente du train/test.")

        # Sélection/chargement modèle
        model_path = pick_model_for_cluster(model_folder, cid)
        if model_path is None or not os.path.exists(model_path):
            print(f"[WARN] Aucun modèle trouvé pour cluster={cid}. Fallback naïf.")
            model = object()  # fantôme pour fallback
        else:
            print(f"[INFO] Modèle utilisé (cluster={cid}): {model_path}")
            model = load_model(model_path)

        # Prévisions
        steps = len(te)
        y_pred = forecast_with_model(model, te.index, steps)
        y_pred = np.asarray(y_pred).reshape(-1)
        if y_pred.shape[0] != steps:
            y_pred = y_pred[:steps]

        y_true = np.asarray(te[y_col]).reshape(-1)
        metrics = compute_metrics(y_true, y_pred)

        # Enregistrement résultats
        pred_df = pd.DataFrame(
            {"date": te.index, "y_true": y_true, "y_pred": y_pred}
        )
        outfile = out_dir / (f"predictions_cluster_{cid}{suffix}.csv" if cid is not None else f"predictions_global{suffix}.csv")
        pred_df.to_csv(outfile, index=False, sep=";")

        # Logs MLflow
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("cluster_id", cid if cid is not None else "global")
            mlflow.log_param("suffix", suffix)
            if model_path:
                mlflow.log_param("model_path", model_path)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_artifact(str(outfile), artifact_path="evaluate")

        print(f"✅ {run_name} -> {outfile} | {metrics}")

    print("✅ Evaluation terminée.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--model-folder", required=True)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    main(args.input_folder, args.output_folder, args.model_folder, suffix=args.suffix)

