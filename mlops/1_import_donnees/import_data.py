# mlops/1_import_donnees/import_data.py
import os
import json
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple

import click
import pandas as pd
import mlflow
import dvc.api


# ============= MLflow bootstrap =============
def setup_mlflow() -> Optional[str]:
    """
    Configure MLflow à partir de l'env si présent, sinon fallback local (file://mlruns).
    Retourne artifact_location si local, sinon None (serveur distant gère).
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        return None
    artifact_dir = os.path.abspath("mlruns")
    mlflow.set_tracking_uri("file://" + artifact_dir)
    return "file://" + artifact_dir


# ============= Checkpoint I/O =============
def load_checkpoint(path: Path) -> Tuple[set, Optional[str]]:
    """
    Checkpoint = parquet de clés + .json adjacent avec watermark temporel.
    """
    if not path.exists():
        return set(), None

    df = pd.read_parquet(path)
    seen = set(df["key_hash"].astype(str).tolist())

    meta_path = path.with_suffix(".json")
    watermark = None
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            watermark = meta.get("last_watermark")
    return seen, watermark


def save_checkpoint(path: Path, seen_keys: set, last_watermark: Optional[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"key_hash": sorted(seen_keys)}).to_parquet(path, index=False)
    meta_path = path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"last_watermark": last_watermark}, f)


# ============= Utilitaires =============
def make_key_hash(df: pd.DataFrame, key_cols: List[str]) -> pd.Series:
    """
    Hash stable des clés métier (concat '||'), sinon hash du dataframe.
    """
    if not key_cols:
        return pd.util.hash_pandas_object(df, index=False).astype(str)
    arr = df[key_cols].astype(str).fillna("")
    cat = arr.apply(lambda r: "||".join(r.values.tolist()), axis=1)
    return pd.util.hash_pandas_object(cat, index=False).astype(str)


def parse_date(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    return df


def to_str_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


# ============= Extraction incrémentale =============
def incremental_extract(
    dvc_path: str,
    repo: Optional[str],
    delta_folder: Path,
    cumulative_csv: Path,
    checkpoint_path: Path,
    date_col: Optional[str],
    key_cols: List[str],
    sep: str,
    run_ds: Optional[str],
) -> Tuple[Path, Path, int, int]:
    """
    Lit la source par chunks, filtre > watermark (si date), anti-join sur clés,
    écrit le delta (df_new.csv) ET met à jour le cumul (df_sample.csv) sans doublons.

    Retourne: (delta_path, cumul_path, rows_delta, rows_cumul)
    """
    seen_keys, watermark = load_checkpoint(checkpoint_path)

    content = dvc.api.read(dvc_path, repo=repo)
    source_file = StringIO(content)
    chunks = pd.read_csv(
        source_file, sep=sep, chunksize=200_000, on_bad_lines="skip", low_memory=False
    )

    new_rows = []
    max_date_seen = pd.to_datetime(watermark, utc=True, errors="coerce") if watermark else None

    for chunk in chunks:
        chunk = parse_date(chunk, date_col)

        if date_col and watermark:
            wm = pd.to_datetime(watermark, utc=True, errors="coerce")
            if wm is not None:
                chunk = chunk.loc[chunk[date_col] > wm]

        if chunk.empty:
            continue

        kh = make_key_hash(chunk, key_cols)
        mask_new = ~kh.isin(seen_keys)
        delta = chunk.loc[mask_new].copy()
        if delta.empty:
            continue

        delta["__key_hash__"] = kh.loc[mask_new].values

        if date_col in delta.columns:
            cand = delta[date_col].max()
            if pd.notna(cand):
                max_date_seen = cand if max_date_seen is None else max(max_date_seen, cand)

        new_rows.append(delta)

    # Prépare dossiers
    delta_folder.mkdir(parents=True, exist_ok=True)
    delta_path = delta_folder / "df_new.csv"
    cumulative_csv.parent.mkdir(parents=True, exist_ok=True)

    # Concatène delta
    if new_rows:
        df_new = pd.concat(new_rows, ignore_index=True)
        df_new = to_str_cols(df_new, ["code_postal", "INSEE_COM", "departement", "commune"])
        # écrit delta (sans colonne technique)
        df_new.drop(columns=["__key_hash__"], errors="ignore").to_csv(delta_path, sep=";", index=False)

        # met à jour cumul
        if cumulative_csv.exists() and cumulative_csv.stat().st_size > 0:
            df_old = pd.read_csv(cumulative_csv, sep=sep, low_memory=False)
            df_all = pd.concat([df_old, df_new.drop(columns=["__key_hash__"], errors="ignore")], ignore_index=True)
        else:
            df_all = df_new.drop(columns=["__key_hash__"], errors="ignore")

        if key_cols:
            df_all = df_all.drop_duplicates(subset=key_cols, keep="last")
        else:
            df_all = df_all.drop_duplicates(keep="last")

        df_all.to_csv(cumulative_csv, sep=sep, index=False)

        # Update checkpoint (ajoute les nouvelles clés, watermark)
        seen_keys.update(make_key_hash(df_new, key_cols).astype(str).tolist())
        last_wm = max_date_seen.isoformat() if isinstance(max_date_seen, pd.Timestamp) else watermark
        save_checkpoint(checkpoint_path, seen_keys, last_wm)
    else:
        # pas de nouveauté → persiste un delta vide pour cohérence
        delta_path.write_text("", encoding="utf-8")
        # cumul inchangé

    # Métriques
    rows_delta = 0
    if delta_path.exists() and delta_path.stat().st_size > 0:
        with open(delta_path, "r", encoding="utf-8") as f:
            rows_delta = max(sum(1 for _ in f) - 1, 0)

    rows_cumul = 0
    if cumulative_csv.exists() and cumulative_csv.stat().st_size > 0:
        with open(cumulative_csv, "r", encoding="utf-8") as f:
            rows_cumul = max(sum(1 for _ in f) - 1, 0)

    return delta_path, cumulative_csv, rows_delta, rows_cumul


# ============= CLI =============
@click.command()
@click.option("--dvc-path", type=str, required=True, help="Chemin du fichier dans DVC (ex: data/merged_sales_data.csv)")
@click.option("--repo", type=str, default=None, help="URL du repo DVC (optionnel, utilise la config locale si non spécifié)")
@click.option("--output-folder", type=click.Path(), required=True, help="Dossier de sortie du DELTA (df_new.csv)")
@click.option("--cumulative-path", type=click.Path(), default="data/df_sample.csv",
              help="Chemin du CSV cumul (df_sample.csv) — NOM HISTORIQUE CONSERVÉ")
@click.option("--checkpoint-path", type=click.Path(), required=True, help="Chemin du checkpoint (parquet)")
@click.option("--date-column", type=str, default=None, help="Colonne date pour watermark (ex: date_vente)")
@click.option("--key-columns", type=str, default="", help="Colonnes clés séparées par des virgules (ex: id_transaction,lot)")
@click.option("--sep", type=str, default=";", help="Séparateur CSV (défaut ';')")
def main(
    dvc_path,
    repo,
    output_folder,
    cumulative_path,
    checkpoint_path,
    date_column,
    key_columns,
    sep,
):
    artifact_location = setup_mlflow()

    key_cols = [c.strip() for c in key_columns.split(",") if c.strip()]
    delta_folder = Path(output_folder)
    cumulative_csv = Path(cumulative_path)
    checkpoint_path = Path(checkpoint_path)

    # Expériment MLflow (créée seulement en local file://)
    experiment_name = "Import données"
    if artifact_location and mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(name=experiment_name, artifact_location=artifact_location)

    # Airflow fournit 'ds' (YYYY-MM-DD) dans l'env; fallback "manual"
    run_ds = os.getenv("AIRFLOW_CTX_EXECUTION_DATE", os.getenv("ds", "manual"))

    with mlflow.start_run(run_name="Import incrémental"):
        delta_path, cumul_path, rows_delta, rows_cumul = incremental_extract(
            dvc_path=dvc_path,
            repo=repo,
            delta_folder=delta_folder,
            cumulative_csv=cumulative_csv,
            checkpoint_path=checkpoint_path,
            date_col=date_column,
            key_cols=key_cols,
            sep=sep,
            run_ds=run_ds,
        )

        # Params
        mlflow.log_param("source_path", dvc_path)
        mlflow.log_param("delta_folder", str(delta_folder))
        mlflow.log_param("cumulative_path", str(cumul_path))
        mlflow.log_param("checkpoint_path", str(checkpoint_path))
        mlflow.log_param("date_column", date_column or "")
        mlflow.log_param("key_columns", ",".join(key_cols))
        mlflow.log_param("sep", sep)

        # Metrics
        mlflow.log_metric("rows_delta", rows_delta)
        mlflow.log_metric("rows_cumul", rows_cumul)

        # Artifacts (légers ; évite les giga-fichiers selon ta politique)
        if delta_path.exists():
            mlflow.log_artifact(str(delta_path))
        if cumul_path.exists():
            mlflow.log_artifact(str(cumul_path))

        print(
            f"✅ Delta → {delta_path} (rows={rows_delta}) | "
            f"Cumul → {cumul_path} (rows={rows_cumul})"
        )


if __name__ == "__main__":
    # Dépendances nécessaires côté Airflow/requirements:
    #   pandas, pyarrow, click, mlflow
    main()

