#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import math
import traceback
import warnings
import logging
from pathlib import Path
from typing import Any, Dict, List

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Assurer l'import du package local (mlops.*) quand DVC lance depuis la racine
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../repo/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---- MLflow (tolérant, et minimal ici) ----
try:
    import mlflow  # type: ignore
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False
    class _MLFlowNoOp:
        def set_tracking_uri(self, *_: Any, **__: Any) -> None: ...
        def set_experiment(self, *_: Any, **__: Any) -> None: ...
        def start_run(self, *_: Any, **__: Any) -> None:
            class _R:
                def __enter__(self): return self
                def __exit__(self, *e): return False
            return _R()
        def log_artifact(self, *_: Any, **__: Any) -> None: ...
        def log_dict(self, *_: Any, **__: Any) -> None: ...
    mlflow = _MLFlowNoOp()  # type: ignore

# ---- Matplotlib headless ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# seaborn optionnel
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

import pandas as pd
import click

# ---- utilitaires strictement PRÉPROCESSING ----
from mlops.preprocessing_4.utils import (
    annee_const,
    clean_classe,
    clean_exposition,
    extract_principal,
    get_numeric_cols,
    calculate_bounds,
    compute_medians,
    mark_outliers,
    clean_outliers,
)

# ---------------- helpers ----------------
def _diag_env() -> Dict[str, Any]:
    return {
        "python_exe": sys.executable,
        "python_ver": sys.version,
        "cwd": os.getcwd(),
        "sys_path_head": sys.path[:5],
        "has_mlflow": _HAS_MLFLOW,
        "has_seaborn": _HAS_SEABORN,
        "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI"),
    }

def _ensure_mlflow(experiment_name: str = "Preprocessing"):
    """Minimal: on ne fait que tracer les artefacts; pas de métriques 'régression' ici."""
    if not _HAS_MLFLOW:
        return
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        try:
            mlflow.set_tracking_uri(uri)
        except Exception as e:
            logging.warning(f"[WARN] set_tracking_uri failed: {e}")
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logging.warning(f"[WARN] set_experiment failed: {e}")

def _log_figure_safe(fig, path: Path, artifact_path: str | None = None, close: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    if _HAS_MLFLOW:
        try:
            mlflow.log_artifact(str(path), artifact_path=artifact_path)
        except Exception:
            pass
    if close:
        plt.close(fig)

def _barplot(ax, y_labels, x_values, title: str):
    if _HAS_SEABORN:
        sns.barplot(y=y_labels, x=x_values, ax=ax)
    else:
        ax.barh(range(len(y_labels)), x_values)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(list(y_labels))
    ax.set_title(title)

def _coalesce_str_col(df: pd.DataFrame, candidates: List[str], default_name: str, keep_original: bool=False) -> pd.DataFrame:
    """Créer une colonne `default_name` depuis l’une des `candidates` si dispo; sinon valeur 'UNKNOWN'."""
    df = df.copy()
    for c in candidates:
        if c in df.columns:
            df[default_name] = df[c].astype("string")
            return df
    df[default_name] = pd.Series(["UNKNOWN"] * len(df), dtype="string")
    return df

# ---------------- pipeline ----------------
def run_preprocessing_pipeline(input_path: str, output_path: str):
    logging.info("=== [preprocessing] diagnostics ===")
    for k, v in _diag_env().items():
        logging.info(f"{k}: {v}")

    _ensure_mlflow()

    input_dir = Path(input_path)
    output_dir = Path(output_path)
    figures_dir = output_dir / "reports" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = input_dir / "df_sample.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_path} (attendu par --input {input_path})")

    logging.info(f"Lecture CSV: {csv_path}")
    df = pd.read_csv(csv_path, sep=";", low_memory=False, dtype={"INSEE_COM": "string"})
    before = len(df)
    df = df.drop_duplicates()
    logging.info(f"Drop duplicates: {before} -> {len(df)}")

    # % manquants
    miss_pct = df.isna().sum().mul(100.0 / max(len(df), 1))
    missing_df = (
        pd.DataFrame({"column_name": df.columns, "percent_missing": miss_pct})
        .sort_values("percent_missing", ascending=False)
        .reset_index(drop=True)
    )
    # plot missing
    fig1, ax1 = plt.subplots(figsize=(10, 14))
    _barplot(ax1, missing_df["column_name"], missing_df["percent_missing"], "Valeurs manquantes (%)")
    ax1.axvline(x=75, color="red", linestyle="--", label="75%"); ax1.legend()
    _log_figure_safe(fig1, figures_dir / "Nan_distribution.png", artifact_path="figures/missing", close=True)

    # garder colonnes <= 75% manquants
    cols_keep = miss_pct[miss_pct <= 75].index
    df1 = df[cols_keep].copy()

    # drop colonnes très spécifiques métier si présentes
    to_drop = [
        "idannonce","annonce_exclusive","typedebien_lite","type_annonceur","categorie_annonceur",
        "REG","DEP","IRIS","CODE_IRIS","TYP_IRIS_x","TYP_IRIS_y","nb_logements_copro","GRD_QUART","UU2010","duree_int",
    ]
    df2 = df1.drop(columns=[c for c in to_drop if c in df1.columns], errors="ignore")

    # normalisation booleans
    for c in ("porte_digicode","cave","ascenseur"):
        if c in df2.columns:
            df2[c] = df2[c].astype(str).str.strip().isin({"1","True","true","OUI","Oui","yes"}).astype("boolean")

    # assurer la présence d’une colonne groupe (INSEE_COM sinon fallback code postal; sinon UNKNOWN)
    if "INSEE_COM" not in df2.columns:
        df2 = _coalesce_str_col(df2, ["code_insee","code_insee_commune","codePostal","code_postal","cp"], "INSEE_COM")

    # nettoyer cible si présente
    TARGET_COL = "prix_m2_vente"
    if TARGET_COL in df2.columns:
        df2 = df2.dropna(subset=[TARGET_COL])

    # enrichissements métier (préprocessing)
    try:
        df2 = annee_const(df2)
    except Exception as e:
        logging.error(f"annee_const a échoué: {e}")
        raise

    for c in ("dpeL","ges_class"):
        if c in df2.columns:
            df2[c] = df2[c].apply(clean_classe).astype("string")

    if "chauffage_energie" in df2.columns:
        df2["chauffage_energie_principal"] = df2["chauffage_energie"].apply(extract_principal).astype("string")
        # correction d'encodage éventuelle
        df2["chauffage_energie_principal"] = df2["chauffage_energie_principal"].str.replace("Ã\x89","É", regex=False)

    if "exposition" in df2.columns:
        df2["exposition"] = df2["exposition"].apply(clean_exposition).astype("string")

    # drop colonnes à risque de fuite
    df3 = df2.drop(columns=[c for c in ["prix_bien","mensualiteFinance"] if c in df2.columns], errors="ignore")

    # figures de distribution cible (si cible là)
    if TARGET_COL in df3.columns:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        if _HAS_SEABORN:
            sns.histplot(df3[TARGET_COL], bins=60, ax=ax2)
        else:
            ax2.hist(df3[TARGET_COL].dropna().to_numpy(), bins=60)
        ax2.set_title("Distribution prix_m2_vente")
        _log_figure_safe(fig2, figures_dir / "prix_m2_distribution.png", artifact_path="figures/distributions", close=True)

    # diagnostics numériques + boxplots (si colonnes numériques existent)
    GROUP_COL = "INSEE_COM"
    numeric_cols = get_numeric_cols(df3, GROUP_COL)
    if numeric_cols:
        cols_per_row = 2
        rows = math.ceil(len(numeric_cols) / cols_per_row)
        fig3, axes = plt.subplots(rows, cols_per_row, figsize=(12, 4*rows))
        axes = axes.flatten()
        for i, c in enumerate(numeric_cols):
            df3.boxplot(column=c, ax=axes[i]); axes[i].set_title(f"Boxplot '{c}'")
        for j in range(i+1, len(axes)):
            fig3.delaxes(axes[j])
        fig3.tight_layout()
        _log_figure_safe(fig3, figures_dir / "Boxplot_variables.png", artifact_path="figures/boxplots", close=True)

    # détection d’anomalies logiques (non bloquant)
    try:
        df_logic = df3.copy()
        df_logic["anomalie_logique"] = False
        if {"nb_toilettes","nb_pieces"}.issubset(df_logic.columns):
            df_logic.loc[df_logic["nb_toilettes"] > df_logic["nb_pieces"], "anomalie_logique"] = True
        if "surface" in df_logic.columns:
            df_logic.loc[(df_logic["surface"] < 10) | (df_logic["surface"] > 1000), "anomalie_logique"] = True
        if {"nb_etages","etage"}.issubset(df_logic.columns):
            df_logic.loc[(df_logic["nb_etages"] == 0) & (df_logic["etage"] > 0), "anomalie_logique"] = True
        if {"logement_neuf","annee_construction"}.issubset(df_logic.columns):
            old_bins = {"avant 1948","1948-1974","1975-1977","1978-1982","1983-1988","1989-2000"}
            df_logic.loc[(df_logic["logement_neuf"] == True) & (df_logic["annee_construction"].isin(old_bins)), "anomalie_logique"] = True
        if TARGET_COL in df_logic.columns:
            df_logic.loc[df_logic[TARGET_COL] < 100, "anomalie_logique"] = True

        anomalies = df_logic[df_logic["anomalie_logique"]].head(10)
        anomalies_csv = figures_dir / "anomaly_logic_preview.csv"
        anomalies.to_csv(anomalies_csv, index=False)
        if _HAS_MLFLOW:
            try: mlflow.log_artifact(str(anomalies_csv), artifact_path="extracts/anomaly_logic")
            except Exception: pass
    except Exception as e:
        logging.warning(f"[WARN] logique anomalies skipped: {e}")

    # ---------------- split + outliers (préproc pur) ----------------
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df3, test_size=0.2, random_state=42)

    bounds = calculate_bounds(train_df, numeric_cols, 0.001, 0.999) if numeric_cols else {}
    train_marked = mark_outliers(train_df, bounds) if bounds else train_df
    test_marked  = mark_outliers(test_df , bounds) if bounds else test_df

    # On enlève les outliers uniquement pour la cible si elle existe, sinon on impute plus tard (encodage)
    if TARGET_COL in train_marked:
        keep_tr = (train_marked.get(f"{TARGET_COL}_outlier_flag", 0) == 0) if f"{TARGET_COL}_outlier_flag" in train_marked else True
        keep_te = (test_marked.get (f"{TARGET_COL}_outlier_flag", 0) == 0) if f"{TARGET_COL}_outlier_flag" in test_marked  else True
        if isinstance(keep_tr, pd.Series): train_marked = train_marked[keep_tr]
        if isinstance(keep_te, pd.Series): test_marked  = test_marked[keep_te]

    train_clean = clean_outliers(train_marked, bounds, *compute_medians(train_marked, bounds, GROUP_COL), GROUP_COL) if bounds else train_marked
    test_clean  = clean_outliers(test_marked , bounds, *compute_medians(train_marked, bounds, GROUP_COL), GROUP_COL) if bounds else test_marked

    # nettoyage des flags _outlier_flag
    for c in list(bounds.keys()):
        f = f"{c}_outlier_flag"
        for d in (train_clean, test_clean):
            if f in d.columns:
                d.drop(columns=[f], inplace=True)

    out_train = output_dir / "df_sales_clean_train.csv"
    out_test  = output_dir / "df_sales_clean_test.csv"
    train_clean.to_csv(out_train, sep=";", index=False)
    test_clean.to_csv(out_test , sep=";", index=False)
    logging.info(f"[OK] Écrits: {out_train} ; {out_test}")

    # artefacts uniquement (pas de métriques 'régression' ici)
    if _HAS_MLFLOW:
        try:
            with mlflow.start_run(run_name="preprocessing"):
                mlflow.log_artifact(str(out_train))
                mlflow.log_artifact(str(out_test))
        except Exception:
            pass

    logging.info("✅ Pipeline preprocessing terminée avec succès")

# ---------------- CLI ----------------
@click.command()
@click.option(
    "--input", "--input-dir", "--input-data", "input_path",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Dossier input (contient df_sample.csv)",
)
@click.option(
    "--out-dir", "output_path",
    type=click.Path(file_okay=False),
    required=True,
    help="Dossier de sortie",
)
def main(input_path: str, output_path: str):
    try:
        run_preprocessing_pipeline(input_path=input_path, output_path=output_path)
    except Exception as e:
        logging.error(f"[FATAL] preprocessing failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

