# mlops/1_import_donnees/import_data.py
import os
import json
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple, IO, Union

import click
import pandas as pd
import mlflow
import dvc.api


# ============= MLflow bootstrap =============
def setup_mlflow() -> Optional[str]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        return None
    artifact_dir = os.path.abspath("mlruns")
    mlflow.set_tracking_uri("file://" + artifact_dir)
    return "file://" + artifact_dir


# ============= Checkpoint I/O =============
def load_checkpoint(path: Path) -> Tuple[set, Optional[str]]:
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


# ============= Source opening (DVC) =============
def open_source_for_csv(
    *,
    dvc_repo_url: Optional[str],
    dvc_path: Optional[str],
    dvc_rev: Optional[str],
    dvc_remote: Optional[str],
) -> Tuple[Union[IO[str], Path], bool]:
    if not (dvc_repo_url and dvc_path):
        raise ValueError("DVC repo URL et path requis.")

    f = dvc.api.open(
        path=dvc_path,
        repo=dvc_repo_url,
        rev=dvc_rev,
        remote=dvc_remote,
        mode="r",
        encoding="utf-8",
    )
    return f, True


# ============= Extraction incrémentale =============
def incremental_extract(
    source_path: Optional[Path],
    delta_folder: Path,
    cumulative_csv: Path,
    checkpoint_path: Path,
    date_col: Optional[str],
    key_cols: List[str],
    sep: str,
    run_ds: Optional[str],
    *,
    dvc_repo_url: Optional[str],
    dvc_path: Optional[str],
    dvc_rev: Optional[str],
    dvc_remote: Optional[str],
) -> Tuple[Path, Path, int, int]:
    seen_keys, watermark = load_checkpoint(checkpoint_path)
    handle_or_path, is_stream = open_source_for_csv(
        dvc_repo_url=dvc_repo_url,
        dvc_path=dvc_path,
        dvc_rev=dvc_rev,
        dvc_remote=dvc_remote,
    )

    delta_folder.mkdir(parents=True, exist_ok=True)
    delta_path = delta_folder / "df_new.csv"
    cumulative_csv.parent.mkdir(parents=True, exist_ok=True)

    max_date_seen = pd.to_datetime(watermark, utc=True, errors="coerce") if watermark else None
    new_rows: List[pd.DataFrame] = []

    cm = handle_or_path if is_stream else nullcontext(open(handle_or_path, "r", encoding="utf-8"))
    with cm as f:
        chunks = pd.read_csv(
            f,
            sep=sep,
            chunksize=200_000,
            on_bad_lines="skip",
            low_memory=False,
        )
        for chunk in chunks:
            chunk = parse_date(chunk, date_col)

            if date_col and watermark:
                wm = pd.to_datetime(watermark, utc=True, errors="coerce")
                if wm is not None:
                    chunk = chunk.loc[chunk[date_col] > wm]

            if chunk.empty:
                continue

            kh = make_key_hash(chunk, key_cols)
            mask_new = ~kh.astype(str).isin(seen_keys)
            delta = chunk.loc[mask_new].copy()
            if delta.empty:
                continue

            delta["__key_hash__"] = kh.loc[mask_new].astype(str).values

            if date_col in delta.columns:
                cand = delta[date_col].max()
                if pd.notna(cand):
                    max_date_seen = cand if max_date_seen is None else max(max_date_seen, cand)

            new_rows.append(delta)

    if new_rows:
        df_new = pd.concat(new_rows, ignore_index=True)
        df_new = to_str_cols(df_new, ["code_postal", "INSEE_COM", "departement", "commune"])
        df_new.drop(columns=["__key_hash__"], errors="ignore").to_csv(delta_path, sep=sep, index=False)

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

        seen_keys.update(make_key_hash(df_new, key_cols).astype(str).tolist())
        last_wm = max_date_seen.isoformat() if isinstance(max_date_seen, pd.Timestamp) else watermark
        save_checkpoint(checkpoint_path, seen_keys, last_wm)
    else:
        delta_path.write_text("", encoding="utf-8")

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
@click.option("--folder-path", type=click.Path(), required=False,
              help="Dossier source local contenant le CSV (ignoré en mode DVC).")
@click.option("--input-file", type=str, required=False,
              help="Nom du fichier CSV local (ignoré en mode DVC).")
@click.option("--output-folder", type=click.Path(), required=True, help="Dossier de sortie du DELTA (df_new.csv)")
@click.option("--cumulative-path", type=click.Path(), default="data/df_sample.csv",
              help="Chemin du CSV cumul (df_sample.csv)")
@click.option("--checkpoint-path", type=click.Path(), required=True, help="Chemin du checkpoint (parquet)")
@click.option("--date-column", type=str, default=None, help="Colonne date pour watermark")
@click.option("--key-columns", type=str, default="", help="Colonnes clés séparées par des virgules")
@click.option("--sep", type=str, default=";", help="Séparateur CSV")
@click.option("--dvc-repo-url", type=str, required=True, help="URL du repo DVC sur DagsHub")
@click.option("--dvc-path", type=str, required=True, help="Chemin du fichier dans DVC")
@click.option("--dvc-rev", type=str, default="main", help="Révision git")
@click.option("--dvc-remote", type=str, default=None, help="Nom du remote DVC")
def main(
    output_folder,
    cumulative_path,
    checkpoint_path,
    date_column,
    key_columns,
    sep,
    dvc_repo_url,
    dvc_path,
    dvc_rev,
    dvc_remote,
):
    artifact_location = setup_mlflow()
    key_cols = [c.strip() for c in key_columns.split(",") if c.strip()]
    source_path = None
    # Valider le mode choisi
    if dvc_repo_url and dvc_path:
        # mode DVC
        pass
    else:
        # mode local
        if not folder_path or not input_file:
            raise click.UsageError("En mode local, --folder-path et --input-file sont requis.")
        source_path = Path(folder_path) / input_file
        if not source_path.exists():
            raise click.ClickException(f"Fichier local introuvable: {source_path}")

    delta_folder = Path(output_folder)
    cumulative_csv = Path(cumulative_path)
    checkpoint_path = Path(checkpoint_path)

    experiment_name = "Import données"
    if artifact_location and mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(name=experiment_name, artifact_location=artifact_location)

    run_ds = os.getenv("AIRFLOW_CTX_EXECUTION_DATE", os.getenv("ds", "manual"))

    with mlflow.start_run(run_name="Import incrémental"):
        delta_path, cumul_path, rows_delta, rows_cumul = incremental_extract(
            delta_folder=delta_folder,
            cumulative_csv=cumulative_csv,
            checkpoint_path=checkpoint_path,
            date_col=date_column,
            key_cols=key_cols,
            sep=sep,
            run_ds=run_ds,
            dvc_repo_url=dvc_repo_url,
            dvc_path=dvc_path,
            dvc_rev=dvc_rev,
            dvc_remote=dvc_remote,
        )

        # Params
        mlflow.log_param("mode", "dvc" if (dvc_repo_url and dvc_path) else "local")
        mlflow.log_param("source_local_path", str(source_path) if source_path else "")
        mlflow.log_param("dvc_repo_url", dvc_repo_url or "")
        mlflow.log_param("dvc_path", dvc_path or "")
        mlflow.log_param("dvc_rev", dvc_rev or "")
        mlflow.log_param("dvc_remote", dvc_remote or "")
        mlflow.log_param("delta_folder", str(delta_folder))
        mlflow.log_param("cumulative_path", str(cumulative_csv))
        mlflow.log_param("checkpoint_path", str(checkpoint_path))
        mlflow.log_param("date_column", date_column or "")
        mlflow.log_param("key_columns", ",".join(key_cols))
        mlflow.log_param("sep", sep)

        mlflow.log_metric("rows_delta", rows_delta)
        mlflow.log_metric("rows_cumul", rows_cumul)

        if delta_path.exists():
            mlflow.log_artifact(str(delta_path))
        if cumulative_csv.exists():
            mlflow.log_artifact(str(cumulative_csv))

        print(
            f"✅ Delta → {delta_path} (rows={rows_delta}) | Cumul → {cumul_path} (rows={rows_cumul})"
        )


if __name__ == "__main__":
    main()

