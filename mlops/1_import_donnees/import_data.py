#!/usr/bin/env python3
# path: import_data.py


from __future__ import annotations
import os
import io
import json
import time
import math
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple, IO, Union

import click
import pandas as pd
import mlflow

# ============= Optional deps (lazy import) =============
def _lazy_import_boto3():
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError
    return boto3, Config, (ClientError, EndpointConnectionError, ReadTimeoutError)

def _lazy_import_dvc():
    import dvc.api
    return dvc.api

def _lazy_import_requests():
    import requests
    from requests.exceptions import RequestException, Timeout, ConnectionError as ReqConnErr
    return requests, (RequestException, Timeout, ReqConnErr)

# ============= MLflow bootstrap =============
def setup_mlflow() -> Optional[str]:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
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
    wm = None
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            wm = meta.get("last_watermark")
    return seen, wm

def save_checkpoint(path: Path, seen_keys: set, last_watermark: Optional[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"key_hash": sorted(seen_keys)}).to_parquet(path, index=False)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"last_watermark": last_watermark}, f)

# ============= Utils =============
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

def _is_parquet_path(p: Union[str, Path]) -> bool:
    s = str(p).lower()
    return s.endswith(".parquet") or s.endswith(".pq") or s.endswith(".parq")

# ============= Source opening (multi modes) =============
class SourceHandle:
    def __init__(self, handle_or_path: Union[IO[str], IO[bytes], Path], is_stream: bool, cleanup=lambda: None):
        self.handle_or_path = handle_or_path
        self.is_stream = is_stream
        self.cleanup = cleanup

def open_source_dvc(*, dvc_repo_url: str, dvc_path: str, dvc_rev: Optional[str], dvc_remote: Optional[str]) -> SourceHandle:
    dvc = _lazy_import_dvc()
    f = dvc.open(
        path=dvc_path,
        repo=dvc_repo_url,
        rev=dvc_rev,
        remote=dvc_remote,
        mode="r",
        encoding="utf-8",
    )
    return SourceHandle(f, is_stream=True)

def _backoff_sleep(attempt: int, base: float = 1.5, cap: float = 30.0):
    time.sleep(min(cap, base ** attempt))

def open_source_http(*, url: str, timeout: int = 30, retries: int = 5) -> SourceHandle:
    requests, REX = _lazy_import_requests()
    tf = tempfile.NamedTemporaryFile(prefix="http_src_", suffix=Path(url).name or ".tmp", delete=False)
    tf.close()
    last = None
    for i in range(retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tf.name, "wb") as w:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            w.write(chunk)
            return SourceHandle(Path(tf.name), is_stream=False, cleanup=lambda: os.unlink(tf.name))
        except REX as e:
            last = e
            if i == retries:
                break
            _backoff_sleep(i + 1)
    raise RuntimeError(f"HTTP download failed after {retries} retries: {last}")

def open_source_s3(
    *,
    endpoint_url: Optional[str],
    bucket: str,
    key: str,
    region: str = "us-east-1",
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    session_token: Optional[str] = None,
    path_style: bool = True,
    verify_ssl: bool = True,
    timeout: int = 30,
    retries: int = 5,
) -> SourceHandle:
    boto3, Config, BEX = _lazy_import_boto3()
    cfg = Config(
        region_name=region,
        retries={"max_attempts": 10, "mode": "adaptive"},
        signature_version="s3v4",
        s3={"addressing_style": "path" if path_style else "virtual"},
        connect_timeout=10,
        read_timeout=120,
    )
    s = boto3.session.Session()
    c = s.client(
        "s3",
        endpoint_url=endpoint_url or None,
        region_name=region,
        aws_access_key_id=access_key or os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=secret_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=session_token or os.getenv("AWS_SESSION_TOKEN"),
        config=cfg,
        verify=verify_ssl,
    )
    tf = tempfile.NamedTemporaryFile(prefix="s3_src_", suffix=Path(key).name or ".tmp", delete=False)
    tf.close()
    last = None
    for i in range(retries + 1):
        try:
            c.download_file(bucket, key, tf.name)
            return SourceHandle(Path(tf.name), is_stream=False, cleanup=lambda: os.unlink(tf.name))
        except BEX as e:
            last = e
            if i == retries:
                break
            _backoff_sleep(i + 1)
    raise RuntimeError(f"S3 download failed after {retries} retries: {last}")

def open_source_local(path: Path) -> SourceHandle:
    if not path.exists():
        raise FileNotFoundError(path)
    return SourceHandle(path, is_stream=False)

# ============= Chunk reader (CSV/Parquet) =============
def iter_chunks_csv(handle: Union[IO[str], Path], is_stream: bool, sep: str, chunksize: int = 200_000):
    cs_env = os.getenv("IMPORT_CHUNKSIZE")
    cs = chunksize if not (cs_env and cs_env.isdigit()) else int(cs_env)
    if is_stream:
        yield from pd.read_csv(handle, sep=sep, chunksize=cs, on_bad_lines="skip", low_memory=False)
    else:
        yield from pd.read_csv(str(handle), sep=sep, chunksize=cs, on_bad_lines="skip", low_memory=False)

# --- append disque pour éviter la concat RAM ---
def _append_csv(dst: Path, delta: Path, sep: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    has_header = dst.exists() and dst.stat().st_size > 0
    for chunk in pd.read_csv(delta, sep=sep, chunksize=200_000, low_memory=False):
        chunk.to_csv(dst, sep=sep, index=False, mode="a", header=not has_header)
        has_header = True

# --- (optionnel) dedup disque avec DuckDB ---
def _dedup_with_duckdb(csv_path: Path, key_cols: list[str], date_col: Optional[str], sep: str) -> None:
    try:
        import duckdb
    except Exception:
        return
    con = duckdb.connect()
    con.execute("CREATE OR REPLACE TABLE _all AS SELECT * FROM read_csv_auto(?, delim=?, header=True);",
                [str(csv_path), sep])
    if key_cols:
        order = f"ORDER BY {date_col} NULLS LAST" if date_col else "ORDER BY rowid() DESC"
        part = ", ".join(key_cols)
        con.execute(f"""
            CREATE OR REPLACE TABLE _dedup AS
            SELECT * FROM (
              SELECT *, row_number() OVER(PARTITION BY {part} {order}) AS _rn
              FROM _all
            ) WHERE _rn = 1;
        """)
        con.execute("COPY _dedup TO ? (FORMAT CSV, HEADER, DELIMITER ?);", [str(csv_path), sep])

def iter_chunks_parquet(handle: Union[IO[str], Path], is_stream: bool, batch_rows: int = 200_000):
    # pandas ne stream pas les Parquet; on charge puis on “chunk” en mémoire.
    df = pd.read_parquet(handle if is_stream else str(handle))
    n = len(df)
    if n == 0:
        return
    steps = math.ceil(n / batch_rows)
    for i in range(steps):
        yield df.iloc[i * batch_rows : (i + 1) * batch_rows].copy()

# ============= Extraction incrémentale =============
def incremental_extract(
    *,
    source_mode: str,  # "dvc" | "http" | "s3" | "local"
    # DVC
    dvc_repo_url: Optional[str],
    dvc_path: Optional[str],
    dvc_rev: Optional[str],
    dvc_remote: Optional[str],
    # HTTP
    http_url: Optional[str],
    # S3
    s3_endpoint_url: Optional[str],
    s3_bucket: Optional[str],
    s3_key: Optional[str],
    s3_region: str,
    s3_path_style: bool,
    s3_verify_ssl: bool,
    # Local
    local_path: Optional[str],
    # Common
    delta_folder: Path,
    cumulative_csv: Path,
    checkpoint_path: Path,
    date_col: Optional[str],
    key_cols: List[str],
    sep: str,
    run_ds: Optional[str],
) -> Tuple[Path, Path, int, int]:
    seen_keys, watermark = load_checkpoint(checkpoint_path)

    # Ouvre la source
    if source_mode == "dvc":
        if not (dvc_repo_url and dvc_path):
            raise ValueError("DVC: dvc_repo_url et dvc_path requis.")
        src = open_source_dvc(dvc_repo_url=dvc_repo_url, dvc_path=dvc_path, dvc_rev=dvc_rev, dvc_remote=dvc_remote)
        src_path_like = dvc_path
    elif source_mode == "http":
        if not http_url:
            raise ValueError("HTTP: --http-url requis.")
        src = open_source_http(url=http_url)
        src_path_like = http_url
    elif source_mode == "s3":
        if not (s3_bucket and s3_key):
            raise ValueError("S3: --s3-bucket et --s3-key requis.")
        src = open_source_s3(
            endpoint_url=s3_endpoint_url or os.getenv("AWS_S3_ENDPOINT"),
            bucket=s3_bucket,
            key=s3_key,
            region=s3_region or os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            path_style=s3_path_style,
            verify_ssl=s3_verify_ssl,
        )
        src_path_like = f"s3://{s3_bucket}/{s3_key}"
    elif source_mode == "local":
        if not local_path:
            raise ValueError("Local: --local-path requis.")
        src = open_source_local(Path(local_path))
        src_path_like = local_path
    else:
        raise ValueError(f"source_mode inconnu: {source_mode}")

    # Prépare sorties
    delta_folder.mkdir(parents=True, exist_ok=True)
    delta_path = delta_folder / "df_new.csv"
    cumulative_csv.parent.mkdir(parents=True, exist_ok=True)

    max_date_seen = pd.to_datetime(watermark, utc=True, errors="coerce") if watermark else None
    new_rows: List[pd.DataFrame] = []

    try:
        # Choix CSV/Parquet
        is_parquet = _is_parquet_path(src_path_like)
        if is_parquet:
            chunks_iter = iter_chunks_parquet(src.handle_or_path, src.is_stream, batch_rows=200_000)
        else:
            chunks_iter = iter_chunks_csv(src.handle_or_path, src.is_stream, sep=sep, chunksize=200_000)

        for chunk in chunks_iter:
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

            if date_col and date_col in delta.columns:
                cand = delta[date_col].max()
                if pd.notna(cand):
                    max_date_seen = cand if max_date_seen is None else max(max_date_seen, cand)

            new_rows.append(delta)
    finally:
        # cleanup (tempfiles http/s3)
        try:
            src.cleanup()
        except Exception:
            pass

    # Concatène & écrit
    if new_rows:
        df_new = pd.concat(new_rows, ignore_index=True)
        df_new = to_str_cols(df_new, ["code_postal", "INSEE_COM", "departement", "commune"])

        df_new.drop(columns=["__key_hash__"], errors="ignore").to_csv(delta_path, sep=";", index=False)

        if os.getenv("IMP_APPEND_ONLY", "0") == "1":
            # évite la lecture complète du cumul en RAM
            _append_csv(cumulative_csv, delta_path, sep)
            if os.getenv("IMP_DEDUP_DUCKDB", "0") == "1" and key_cols:
                try:
                    _dedup_with_duckdb(cumulative_csv, key_cols, date_col, sep)
                except Exception:
                    pass
        else:
            # mode legacy (RAM)
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

        # checkpoint
        seen_keys.update(make_key_hash(df_new, key_cols).astype(str).tolist())
        last_wm = max_date_seen.isoformat() if isinstance(max_date_seen, pd.Timestamp) else watermark
        save_checkpoint(checkpoint_path, seen_keys, last_wm)
    else:
        delta_path.write_text("", encoding="utf-8")

    # métriques
    def _count_rows(csv_path: Path) -> int:
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            return 0
        with open(csv_path, "r", encoding="utf-8") as f:
            return max(sum(1 for _ in f) - 1, 0)

    rows_delta = _count_rows(delta_path)
    rows_cumul = _count_rows(cumulative_csv)

    return delta_path, cumulative_csv, rows_delta, rows_cumul

# ============= CLI =============
@click.command()
@click.option("--output-folder", type=click.Path(), required=True, help="Dossier de sortie du DELTA (df_new.csv)")
@click.option("--cumulative-path", type=click.Path(), default="data/df_sample.csv", help="CSV cumul (df_sample.csv)")
@click.option("--checkpoint-path", type=click.Path(), required=True, help="Chemin du checkpoint (parquet)")
@click.option("--date-column", type=str, default=None, help="Colonne date pour watermark (ex: date_vente)")
@click.option("--key-columns", type=str, default="", help="Clés séparées par des virgules (ex: id_transaction,lot)")
@click.option("--sep", type=str, default=";", help="Séparateur CSV (défaut ';')")

# Choix de la source
@click.option("--source-mode", type=click.Choice(["dvc", "http", "s3", "local"]), required=True)

# DVC
@click.option("--dvc-repo-url", type=str, default=None, help="https://dagshub.com/<user>/<repo>")
@click.option("--dvc-path", type=str, default=None, help="Chemin dans le repo (ex: data/raw.csv)")
@click.option("--dvc-rev", type=str, default="main")
@click.option("--dvc-remote", type=str, default=None)

# HTTP
@click.option("--http-url", type=str, default=None, help="URL HTTP(S) du fichier (CSV/Parquet)")

# S3 (DagsHub/MinIO/AWS)
@click.option("--s3-endpoint-url", type=str, default=lambda: os.getenv("AWS_S3_ENDPOINT"), help="Endpoint S3 (DagsHub)")
@click.option("--s3-bucket", type=str, default=lambda: os.getenv("DAGSHUB_BUCKET"), help="Bucket/repo")
@click.option("--s3-key", type=str, default=None, help="Key (ex: data/file.csv)")
@click.option("--s3-region", type=str, default=lambda: os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
@click.option("--s3-path-style", is_flag=True, default=True, help="Path-style (recommandé S3-compatibles)")
@click.option("--s3-no-verify-ssl", is_flag=True, default=False, help="Ne pas vérifier le cert TLS (self-signed)")
# Local
@click.option("--local-path", type=click.Path(), default=None, help="Chemin local (CSV/Parquet)")
def main(
    output_folder,
    cumulative_path,
    checkpoint_path,
    date_column,
    key_columns,
    sep,
    source_mode,
    dvc_repo_url,
    dvc_path,
    dvc_rev,
    dvc_remote,
    http_url,
    s3_endpoint_url,
    s3_bucket,
    s3_key,
    s3_region,
    s3_path_style,
    s3_no_verify_ssl,
    local_path,
):
    artifact_location = setup_mlflow()

    key_cols = [c.strip() for c in key_columns.split(",") if c.strip()]
    delta_folder = Path(output_folder)
    cumulative_csv = Path(cumulative_path)
    checkpoint_path = Path(checkpoint_path)

    experiment_name = "Import données"
    if artifact_location and mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(name=experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)

    run_ds = os.getenv("AIRFLOW_CTX_EXECUTION_DATE", os.getenv("ds", "manual"))

    with mlflow.start_run(run_name=f"Import incrémental [{source_mode}]"):
        delta_path, cumul_path, rows_delta, rows_cumul = incremental_extract(
            source_mode=source_mode,
            dvc_repo_url=dvc_repo_url,
            dvc_path=dvc_path,
            dvc_rev=dvc_rev,
            dvc_remote=dvc_remote,
            http_url=http_url,
            s3_endpoint_url=s3_endpoint_url,
            s3_bucket=s3_bucket,
            s3_key=s3_key,
            s3_region=s3_region,
            s3_path_style=s3_path_style,
            s3_verify_ssl=(not s3_no_verify_ssl),
            local_path=local_path,
            delta_folder=delta_folder,
            cumulative_csv=cumulative_csv,
            checkpoint_path=checkpoint_path,
            date_col=date_column,
            key_cols=key_cols,
            sep=sep,
            run_ds=run_ds,
        )

        # Params
        mlflow.log_param("source_mode", source_mode)
        mlflow.log_param("dvc_repo_url", dvc_repo_url or "")
        mlflow.log_param("dvc_path", dvc_path or "")
        mlflow.log_param("dvc_rev", dvc_rev or "")
        mlflow.log_param("http_url", http_url or "")
        mlflow.log_param("s3_endpoint_url", s3_endpoint_url or "")
        mlflow.log_param("s3_bucket", s3_bucket or "")
        mlflow.log_param("s3_key", s3_key or "")
        mlflow.log_param("delta_folder", str(delta_folder))
        mlflow.log_param("cumulative_path", str(cumulative_csv))
        mlflow.log_param("checkpoint_path", str(checkpoint_path))
        mlflow.log_param("date_column", date_column or "")
        mlflow.log_param("key_columns", ",".join(key_cols))
        mlflow.log_param("sep", sep)

        # Metrics
        mlflow.log_metric("rows_delta", rows_delta)
        mlflow.log_metric("rows_cumul", rows_cumul)

        # Artifacts (léger)
        if delta_path.exists():
            mlflow.log_artifact(str(delta_path))
        if cumulative_csv.exists():
            mlflow.log_artifact(str(cumulative_csv))

        print(f"✅ Delta → {delta_path} (rows={rows_delta}) | Cumul → {cumul_path} (rows={rows_cumul})")

if __name__ == "__main__":
    main()

