# path: mlops/preprocessing_4/__init__.py
# package marker (vide)

# path: mlops/preprocessing_4/preprocessing.py
from __future__ import annotations

import os
import sys
import math
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict

warnings.filterwarnings("ignore")

# --- Matplotlib en mode headless ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- seaborn (tolérant si absent) ---
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

# --- MLflow tolérant/no-op ---
try:
    import mlflow  # type: ignore
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False

    class _MLFlowNoOp:
        def set_tracking_uri(self, *_: Any, **__: Any) -> None: ...
        def set_experiment(self, *_: Any, **__: Any) -> None: ...
        def start_run(self, *_: Any, **__: Any) -> None: ...
        def end_run(self, *_: Any, **__: Any) -> None: ...
        def active_run(self): return None
        def log_artifact(self, *_: Any, **__: Any) -> None: ...
        def log_dict(self, *_: Any, **__: Any) -> None: ...
        def log_metric(self, *_: Any, **__: Any) -> None: ...
        def log_param(self, *_: Any, **__: Any) -> None: ...

    mlflow = _MLFlowNoOp()  # type: ignore

# --- pandas ---
import pandas as pd

# --- CLI ---
import click

# --- utils (dep garantie par utils.py livré) ---
from compagnon_immo.mlops.preprocessing_4.utils import (
    annee_const, clean_classe, clean_exposition, extract_principal,
    get_numeric_cols, calculate_bounds, compute_medians, mark_outliers, clean_outliers,
)

# ------------- Helpers ---------------------------------------------------------
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

def _ensure_mlflow_run(experiment_name: str = "Preprocessing", run_name: str = "preprocessing"):
    if hasattr(mlflow, "active_run") and mlflow.active_run() is None:  # no-op compatible
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name)

def _log_figure_safe(fig, path: Path, artifact_path: str | None = None, close: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    try:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception:
        # why: ne pas casser le run si MLflow distant indispo
        pass
    finally:
        if close:
            plt.close(fig)

def _barplot(ax, y_labels, x_values, title: str):
    if _HAS_SEABORN:
        import seaborn as sns  # local import si installé
        sns.barplot(y=y_labels, x=x_values, ax=ax)
    else:
        ax.barh(range(len(y_labels)), x_values)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(list(y_labels))
    ax.set_title(title)

# ------------- Pipeline --------------------------------------------------------
def run_preprocessing_pipeline(input_path: str, output_path: str):
    """
    Exécute le préprocess : lecture data/df_sample.csv, nettoyage, plots, splits, sauvegardes.
    Émet des diagnostics clairs en cas d’erreur (file not found, imports, permissions).
    """
    print("=== [preprocessing] diagnostics ===")
    for k, v in _diag_env().items():
        print(f"{k}: {v}")

    # MLflow URI (si présent)
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        try:
            mlflow.set_tracking_uri(uri)  # no-op si MLflow no-op
        except Exception as e:
            print(f"[WARN] set_tracking_uri failed: {e}")

    _ensure_mlflow_run()

    input_dir = Path(input_path)
    output_dir = Path(output_path)
    figures_dir = output_dir / "reports" / "figures"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Impossible de créer {output_dir} / {figures_dir}: {e}")

    csv_path = input_dir / "df_sample.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_path} (attendu par --input {input_path})")

    print(f"[INFO] Lecture CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path, sep=";", dtype={"INSEE_COM": "string"})
    except Exception as e:
        raise RuntimeError(f"Échec lecture CSV {csv_path}: {e}")

    GROUP_COL = "INSEE_COM"
    TARGET_COL = "prix_m2_vente"

    # --- dédup ---
    before = len(df)
    df = df.drop_duplicates()
    print(f"[INFO] Drop duplicates: {before} -> {len(df)}")

    # --- missing ---
    miss_pct = df.isna().sum().mul(100.0 / max(len(df), 1))
    missing_df = (
        pd.DataFrame({"column_name": df.columns, "percent_missing": miss_pct})
        .sort_values("percent_missing", ascending=False)
        .reset_index(drop=True)
    )

    # plot missing
    fig1, ax1 = plt.subplots(figsize=(10, 14))
    _barplot(ax1, missing_df["column_name"], missing_df["percent_missing"], "Valeurs manquantes (%)")
    # trait vertical 75%
    ax1.axvline(x=75, color="red", linestyle="--", label="75%")
    ax1.legend()
    _log_figure_safe(fig1, figures_dir / "Nan_distribution.png", artifact_path="figures/missing", close=True)

    # garder cols <= 75%
    cols_keep = miss_pct[miss_pct <= 75].index
    df1 = df[cols_keep].copy()

    # drop colonnes métier si présentes
    to_drop = [
        "idannonce","annonce_exclusive","typedebien_lite","type_annonceur","categorie_annonceur",
        "REG","DEP","IRIS","CODE_IRIS","TYP_IRIS_x","TYP_IRIS_y","nb_logements_copro","GRD_QUART","UU2010","duree_int",
    ]
    df2 = df1.drop(columns=[c for c in to_drop if c in df1.columns], errors="ignore")

    # booléens
    for c in ["porte_digicode","cave","ascenseur"]:
        if c in df2.columns:
            # certains CSV mettent "0/1", on force en bool raisonnable
            df2[c] = df2[c].astype(str).map(lambda x: x.strip() in {"1","True","true","OUI","Oui","yes"}).astype("boolean")

    # cible non nulle
    if TARGET_COL in df2.columns:
        df2 = df2.dropna(subset=[TARGET_COL])

    # enrichissements
    df2 = annee_const(df2)
    for c in ("dpeL","ges_class"):
        if c in df2.columns:
            df2[c] = df2[c].apply(clean_classe).astype("string")

    if "chauffage_energie" in df2.columns:
        df2["chauffage_energie_principal"] = df2["chauffage_energie"].apply(extract_principal).astype("string")
        df2["chauffage_energie_principal"] = df2["chauffage_energie_principal"].str.replace("Ã\x89","É", regex=False)

    if "exposition" in df2.columns:
        df2["exposition"] = df2["exposition"].apply(clean_exposition).astype("string")

    # drop colonnes spécifiques si présentes
    df3 = df2.drop(columns=[c for c in ["prix_bien","mensualiteFinance"] if c in df2.columns], errors="ignore")

    # distribution cible
    if TARGET_COL in df3.columns:
        fig2, ax2 = plt.subplots(figsize=(8,4))
        if _HAS_SEABORN:
            import seaborn as sns
            sns.histplot(df3[TARGET_COL], bins=60, ax=ax2)
        else:
            ax2.hist(df3[TARGET_COL].dropna().to_numpy(), bins=60)
        ax2.set_title("Distribution prix_m2_vente")
        _log_figure_safe(fig2, figures_dir / "prix_m2_distribution.png", artifact_path="figures/distributions", close=True)

    # colonnes numériques
    numeric_cols = get_numeric_cols(df3, GROUP_COL)
    if numeric_cols:
        cols_per_row = 2
        rows = math.ceil(len(numeric_cols)/cols_per_row)
        fig3, axes = plt.subplots(rows, cols_per_row, figsize=(12, 4*rows))
        axes = axes.flatten()
        for i, c in enumerate(numeric_cols):
            df3.boxplot(column=c, ax=axes[i]); axes[i].set_title(f"Boxplot '{c}'")
        # supprimer axes vides
        for j in range(i+1, len(axes)):
            fig3.delaxes(axes[j])
        fig3.tight_layout()
        _log_figure_safe(fig3, figures_dir / "Boxplot_variables.png", artifact_path="figures/boxplots", close=True)

    # anomalies logiques
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
    try:
        mlflow.log_artifact(str(anomalies_csv), artifact_path="extracts/anomaly_logic")
    except Exception:
        pass

    # split train/test
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df3, test_size=0.2, random_state=42)

    # bornes & outliers
    bounds = calculate_bounds(train_df, numeric_cols, 0.001, 0.999) if numeric_cols else {}
    group_meds, global_meds = compute_medians(train_df, bounds, GROUP_COL)
    train_marked = mark_outliers(train_df, bounds)
    test_marked  = mark_outliers(test_df , bounds)

    if TARGET_COL in train_marked:
        keep_tr = train_marked.get(f"{TARGET_COL}_outlier_flag", 0) == 0
        keep_te = test_marked.get(f"{TARGET_COL}_outlier_flag", 0) == 0
        train_marked = train_marked[keep_tr]; test_marked = test_marked[keep_te]

    # logging outliers (si MLflow dispo)
    try:
        mlflow.log_dict({c: int(train_marked.get(f"{c}_outlier_flag", 0).sum()) for c in bounds}, "metrics/outlier_counts.json")
    except Exception:
        pass

    train_clean = clean_outliers(train_marked, bounds, group_meds, global_meds, GROUP_COL)
    test_clean  = clean_outliers(test_marked , bounds, group_meds, global_meds, GROUP_COL)

    # drop flags
    for c in list(bounds.keys()):
        f = f"{c}_outlier_flag"
        for d in (train_clean, test_clean):
            if f in d.columns:
                d.drop(columns=[f], inplace=True)

    # save
    out_train = output_dir / "df_sales_clean_train.csv"
    out_test  = output_dir / "df_sales_clean_test.csv"
    train_clean.to_csv(out_train, sep=";", index=False)
    test_clean.to_csv(out_test , sep=";", index=False)
    print(f"[OK] Écrits: {out_train}, {out_test}")

    try:
        mlflow.log_artifact(str(out_train))
        mlflow.log_artifact(str(out_test))
    except Exception:
        pass

    print("✅ Pipeline preprocessing terminée avec succès")

@click.command()
@click.option("--input", "input_path", type=click.Path(exists=True, file_okay=False), required=True, help="Dossier input (contient df_sample.csv)")
@click.option("--out-dir", "output_path", type=click.Path(file_okay=False), required=True, help="Dossier de sortie")
def main(input_path: str, output_path: str):
    try:
        run_preprocessing_pipeline(input_path=input_path, output_path=output_path)
    except Exception as e:
        # why: imprimer stacktrace pour DVC
        print("[FATAL] preprocessing failed:", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

