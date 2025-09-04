# mlops/7_Serie_temporelle/2_Decompose/seasonal_decomp.py
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import mlflow
from typing import List, Dict

def _load_periodique_concat(input_folder: str) -> pd.DataFrame:
    """Charge train_periodique_q12*.csv et test_periodique_q12*.csv, concatène, trie."""
    candidates = []
    for base in ["train_periodique_q12", "test_periodique_q12"]:
        files = sorted(
            f for f in os.listdir(input_folder)
            if f.startswith(base) and f.endswith(".csv")
        )
        candidates.extend(files)

    if not candidates:
        raise FileNotFoundError(
            f"Aucun fichier '*periodique_q12*.csv' trouvé dans {input_folder}. "
            "Exécute d'abord le stage 'splitst'."
        )

    dfs = []
    for f in candidates:
        df = pd.read_csv(os.path.join(input_folder, f), sep=";", parse_dates=["date"])
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)

    required = {"date", "cluster", "prix_m2_vente"}
    missing = required - set(full.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    full = full.sort_values(["cluster", "date"]).reset_index(drop=True)
    return full

def _safe_decompose(y: pd.Series, model: str, period: int = 12):
    """Décomposition si possible, sinon ValueError explicite."""
    y = y.dropna()
    if len(y) < max(24, 2 * period):
        raise ValueError(f"série trop courte pour une décomposition {model} (n={len(y)})")
    if y.var() == 0:
        raise ValueError("variance nulle — impossible de décomposer")
    return seasonal_decompose(y, model=model, period=period)

def _write_placeholders(output_folder: str, cluster_id, model: str, suffix: str, reason: str) -> Dict[str, str]:
    """Crée des fichiers placeholder (PNG + CSVs) pour satisfaire DVC."""
    os.makedirs(output_folder, exist_ok=True)

    # PNG (sans suffixe, attendu par DVC)
    png_path = os.path.join(output_folder, f"decomposition_{model}_cluster_{cluster_id}.png")
    plt.figure(figsize=(8, 3))
    plt.text(0.5, 0.5, f"No decomposition for cluster {cluster_id}\n({model})\nReason: {reason}",
             ha="center", va="center", fontsize=10)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    # CSVs (vides mais avec en-tête index+value)
    def _empty_csv(path: str):
        pd.DataFrame({"date": pd.to_datetime([]), "value": []}).to_csv(path, sep=";", index=False)

    trend_path    = os.path.join(output_folder, f"trend_{model}_cluster_{cluster_id}.csv")
    seasonal_path = os.path.join(output_folder, f"seasonal_{model}_cluster_{cluster_id}.csv")
    resid_path    = os.path.join(output_folder, f"resid_{model}_cluster_{cluster_id}.csv")
    _empty_csv(trend_path)
    _empty_csv(seasonal_path)
    _empty_csv(resid_path)

    # copies suffixées (pour ton usage perso)
    if suffix:
        png_suff = os.path.join(output_folder, f"decomposition_{model}_cluster_{cluster_id}{suffix}.png")
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, f"No decomposition for cluster {cluster_id}\n({model}, {suffix})\nReason: {reason}",
                 ha="center", va="center", fontsize=10)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(png_suff)
        plt.close()

        for base, p in [("trend", trend_path), ("seasonal", seasonal_path), ("resid", resid_path)]:
            pd.read_csv(p, sep=";").to_csv(
                os.path.join(output_folder, f"{base}_{model}_cluster_{cluster_id}{suffix}.csv"),
                sep=";", index=False
            )

    return {"png": png_path, "trend": trend_path, "seasonal": seasonal_path, "resid": resid_path}

def _save_fig_and_components(decomp, cluster_id, model, output_folder, suffix="") -> Dict[str, str]:
    """Sauvegarde PNG/CSV sans suffixe (DVC) + copie suffixée (si demandé)."""
    # --- Figure principale ---
    fig = decomp.plot()
    fig.suptitle(f"Décomposition {model} - Cluster {cluster_id}")
    base_png = os.path.join(output_folder, f"decomposition_{model}_cluster_{cluster_id}.png")
    plt.tight_layout()
    plt.savefig(base_png)
    plt.close()

    # Copie suffixée optionnelle
    if suffix:
        fig2 = decomp.plot()
        fig2.suptitle(f"Décomposition {model} - Cluster {cluster_id} ({suffix})")
        suff_png = os.path.join(output_folder, f"decomposition_{model}_cluster_{cluster_id}{suffix}.png")
        plt.tight_layout()
        plt.savefig(suff_png)
        plt.close()

    # --- Composantes ---
    trend    = decomp.trend.dropna()
    seasonal = decomp.seasonal.dropna()
    resid    = decomp.resid.dropna()

    def _to_df(series: pd.Series) -> pd.DataFrame:
        out = pd.DataFrame({"date": series.index, "value": series.values})
        # au cas où l'index ne soit pas DatetimeIndex (mais normalement oui)
        out["date"] = pd.to_datetime(out["date"])
        return out

    trend_df    = _to_df(trend)
    seasonal_df = _to_df(seasonal)
    resid_df    = _to_df(resid)

    trend_path    = os.path.join(output_folder, f"trend_{model}_cluster_{cluster_id}.csv")
    seasonal_path = os.path.join(output_folder, f"seasonal_{model}_cluster_{cluster_id}.csv")
    resid_path    = os.path.join(output_folder, f"resid_{model}_cluster_{cluster_id}.csv")

    trend_df.to_csv(trend_path, sep=";", index=False)
    seasonal_df.to_csv(seasonal_path, sep=";", index=False)
    resid_df.to_csv(resid_path, sep=";", index=False)

    if suffix:
        trend_df.to_csv(os.path.join(output_folder, f"trend_{model}_cluster_{cluster_id}{suffix}.csv"), sep=";", index=False)
        seasonal_df.to_csv(os.path.join(output_folder, f"seasonal_{model}_cluster_{cluster_id}{suffix}.csv"), sep=";", index=False)
        resid_df.to_csv(os.path.join(output_folder, f"resid_{model}_cluster_{cluster_id}{suffix}.csv"), sep=";", index=False)

    return {"png": base_png, "trend": trend_path, "seasonal": seasonal_path, "resid": resid_path}

def run_decomposition(input_folder: str, output_folder: str, suffix: str = "", expected_clusters: List[int] = None):
    os.makedirs(output_folder, exist_ok=True)
    mlflow.set_experiment("ST-Decomposition")

    full = _load_periodique_concat(input_folder)

    # Si la liste n'est pas fournie : déduire depuis les données, mais
    # garde un fallback sur [0,1,2,3] si vide.
    data_clusters = sorted([int(c) for c in full["cluster"].dropna().unique()])
    if expected_clusters is None or len(expected_clusters) == 0:
        expected_clusters = data_clusters if data_clusters else [0, 1, 2, 3]

    # Pour chaque cluster attendu, produire des fichiers (réels ou placeholders)
    for cluster_id in expected_clusters:
        dfc = full[full["cluster"] == cluster_id].copy()
        dfc = dfc.sort_values("date").set_index("date")
        y = dfc["prix_m2_vente"] if not dfc.empty else pd.Series(dtype=float)

        with mlflow.start_run(run_name=f"decomp_cluster_{cluster_id}{suffix}"):
            for model in ["additive", "multiplicative"]:
                try:
                    if dfc.empty:
                        raise ValueError("aucune donnée pour ce cluster")
                    decomp = _safe_decompose(y, model=model, period=12)
                    paths = _save_fig_and_components(decomp, cluster_id, model, output_folder, suffix=suffix)

                    # Log MLflow (artefacts sans suffixe = ceux que DVC attend)
                    mlflow.log_artifact(paths["png"])
                    mlflow.log_artifact(paths["trend"])
                    mlflow.log_artifact(paths["seasonal"])
                    mlflow.log_artifact(paths["resid"])

                    # Metrics simples
                    mlflow.log_metric(f"{model}_trend_mean", float(decomp.trend.dropna().mean()))
                    mlflow.log_metric(f"{model}_resid_std", float(decomp.resid.dropna().std()))
                    mlflow.log_metric(f"{model}_resid_skew", float(decomp.resid.dropna().skew()))
                except Exception as e:
                    # Placeholder + log clair
                    paths = _write_placeholders(output_folder, cluster_id, model, suffix, reason=str(e))
                    mlflow.set_tag(f"{model}_placeholder", True)
                    mlflow.set_tag(f"{model}_reason", str(e))
                    mlflow.log_artifact(paths["png"])
                    mlflow.log_artifact(paths["trend"])
                    mlflow.log_artifact(paths["seasonal"])
                    mlflow.log_artifact(paths["resid"])
                    continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Décomposition saisonnière par cluster.")
    parser.add_argument("--input-folder", type=str, required=True, help="Dossier des séries (data/split)")
    parser.add_argument("--output-folder", type=str, required=True, help="Dossier de sortie (ex: outputs/decomposition)")
    parser.add_argument("--suffix", type=str, default="", help="Suffixe optionnel (créera des copies)")
    parser.add_argument(
        "--expected-clusters",
        type=int,
        nargs="*",
        default=None,
        help="Liste des clusters à forcer (ex: --expected-clusters 0 1 2 3). Placeholder si manquant."
    )
    args = parser.parse_args()
    run_decomposition(args.input_folder, args.output_folder, suffix=args.suffix, expected_clusters=args.expected_clusters)

