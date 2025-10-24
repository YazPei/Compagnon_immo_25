#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import logging
import traceback
from pathlib import Path
from typing import Tuple, List, Dict, Any

import click
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# MLflow optional (tolerant)
try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False
    class _NoMlflow:
        def set_tracking_uri(self, *_): pass
        def set_experiment(self, *_): pass
        def start_run(self, *a, **k): return self
        def end_run(self, *a, **k): pass
        def active_run(self): return None
        def log_param(self, *a, **k): pass
        def log_metric(self, *a, **k): pass
        def log_artifact(self, *a, **k): pass
    mlflow = _NoMlflow()  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("encoding")

def robust_read_csv(path: Path, sep: str = ";") -> pd.DataFrame:
    # try common encodings and separators
    attempts = []
    for enc in ("utf-8", "latin-1", "cp1252"):
        for s in (sep, ",", "\t", "|"):
            try:
                df = pd.read_csv(path, sep=s, encoding=enc, low_memory=False)
                logger.info("read_csv success: enc=%s sep=%s shape=%s", enc, s, df.shape)
                return df
            except Exception as e:
                attempts.append((enc, s, str(e)))
    msg = "robust_read_csv failed; attempts:\n" + "\n".join(str(a) for a in attempts)
    raise RuntimeError(msg)

def infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # treat object/string as categorical, numeric types as numeric
    cat = [c for c, t in df.dtypes.items() if t == "object" or t.name.startswith("string")]
    num = [c for c in df.columns if c not in cat]
    # remove target-like columns if present - keep all for encoding step; training script will pick features/target
    return cat, num

def simple_encode(df: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    encoders: Dict[str, LabelEncoder] = {}
    for c in categorical_cols:
        try:
            le = LabelEncoder()
            # fillna before encoding with sentinel
            vals = df[c].fillna("__NA__").astype(str)
            le.fit(vals)
            df[c] = le.transform(vals)
            encoders[c] = le
        except Exception:
            # fallback: map unique -> integer
            mapping = {v: i for i, v in enumerate(df[c].astype(str).fillna("__NA__").unique())}
            df[c] = df[c].astype(str).fillna("__NA__").map(mapping).astype(int)
    return df, encoders

def scale_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols].astype(float))
    return df, scaler

def write_csvs(output_dir: Path, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / "X_train.csv", index=False, sep=";")
    X_test.to_csv(output_dir / "X_test.csv", index=False, sep=";")
    y_train.to_csv(output_dir / "y_train.csv", index=False, sep=";")
    y_test.to_csv(output_dir / "y_test.csv", index=False, sep=";")
    logger.info("Wrote encoded CSVs to %s", output_dir)

@click.command()
@click.option("--data-path", "data_path", required=True, type=click.Path(exists=True), help="Path to input CSV (exports/df_cluster.csv)")
@click.option("--output", "output_dir", required=True, type=click.Path(), help="Output folder for encoded data")
@click.option("--target", "target_col", default="prix_m2_vente", help="Target column name")
@click.option("--test-size", "test_size", default=0.2, type=float, help="Test size fraction")
@click.option("--mlflow-uri", "mlflow_uri", default=None, help="MLflow tracking URI (optional)")
def main(data_path: str, output_dir: str, target_col: str, test_size: float, mlflow_uri: str):
    try:
        data_path_p = Path(data_path)
        out_p = Path(output_dir)
        logger.info("Encoding: data_path=%s output=%s target=%s", data_path_p, out_p, target_col)

        # MLflow Setup
        if mlflow_uri:
            try:
                mlflow.set_tracking_uri(mlflow_uri)
                mlflow.set_experiment("Encoding Experiment")
                logger.info("Connected to MLflow at %s", mlflow_uri)
            except Exception as e:
                logger.warning("mlflow.set_tracking_uri failed: %s", e)

        # Read and process the data
        df = robust_read_csv(data_path_p)

        # Ensure target exists
        if target_col not in df.columns:
            logger.warning("Target '%s' not found in data; attempting to infer last numeric column as target", target_col)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                target_col = numeric_cols[-1]
                logger.info("Inferred target: %s", target_col)
            else:
                raise RuntimeError("No numeric column found to treat as target")

        # Drop rows with missing target
        df = df.dropna(subset=[target_col]).reset_index(drop=True)

        # Infer types and encode
        categorical_cols, numeric_cols = infer_column_types(df.drop(columns=[target_col]))
        logger.info("Detected categorical=%d numeric=%d", len(categorical_cols), len(numeric_cols))
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_enc, encoders = simple_encode(X.copy(), categorical_cols)
        X_enc, scaler = scale_numeric(X_enc, numeric_cols)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=test_size, random_state=42)
        write_csvs(out_p, X_train, X_test, y_train, y_test)

        # MLflow Logging (best-effort)
        try:
            with mlflow.start_run(run_name="encoding"):
                mlflow.log_param("n_rows", int(df.shape[0]))
                mlflow.log_param("n_features", int(X_enc.shape[1]))
                mlflow.log_param("n_categorical", int(len(categorical_cols)))
                mlflow.log_param("n_numeric", int(len(numeric_cols)))
                mlflow.log_metric("train_rows", int(X_train.shape[0]))
                mlflow.log_metric("test_rows", int(X_test.shape[0]))

                # Save a sample for artifact logging
                sample = df.head(100)
                sample_path = out_p / "sample_100.csv"
                sample.to_csv(sample_path, index=False, sep=";")
                mlflow.log_artifact(str(sample_path), artifact_path="encoding_samples")
        except Exception as e:
            logger.warning("MLflow logging failed: %s", e)

        logger.info("✅ Encoding finished successfully")
    except Exception as e:
        logger.error("[FATAL] Encoding failed: %s", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

