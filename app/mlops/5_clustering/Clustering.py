#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def setup_mlflow():
    """
    Tente d'utiliser MLFLOW_TRACKING_URI si défini, sinon fallback en local (file://mlruns).
    Crée/choisit l'expérience "Clustering Données Immo".
    """
    exp_name = "Clustering Données Immo"
    uri = os.getenv("MLFLOW_TRACKING_URI")
    try:
        if uri:
            mlflow.set_tracking_uri(uri)
        else:
            raise RuntimeError("MLFLOW_TRACKING_URI non défini")
        mlflow.set_experiment(exp_name)
    except Exception as e:
        print(f"[WARN] MLflow indisponible ({e}). Fallback en local file://mlruns")
        local_dir = Path.cwd() / "mlruns"
        local_dir.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(f"file://{local_dir}")
        mlflow.set_experiment(exp_name + " (offline)")


def run_clustering_pipeline(input_path: str, output_path: str):
    # ─────────────────────────────────────────────────────────────────────────────
    # MLflow setup (résilient)
    # ─────────────────────────────────────────────────────────────────────────────
    setup_mlflow()

    # ─────────────────────────────────────────────────────────────────────────────
    # Paths d'entrée / sortie
    # ─────────────────────────────────────────────────────────────────────────────
    folder_path = str(Path(input_path).resolve())
    train_file = os.path.join(folder_path, "df_sales_clean_train.csv")
    test_file = os.path.join(folder_path, "df_sales_clean_test.csv")
    geo_file = os.path.join(folder_path, "contours-codes-postaux.geojson")

    output_path = Path(output_path)
    if output_path.suffix.lower() == ".csv":
        output_dir = output_path.parent
        out_cluster_csv = output_path
    else:
        output_dir = output_path
        out_cluster_csv = output_dir / "df_cluster.csv"
    out_st_csv = output_dir / "df_sales_clean_ST.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Vérif fichiers requis
    for f in [train_file, test_file, geo_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Fichier manquant: {f}")

    with mlflow.start_run(run_name="clustering_macro_kpi"):
        # ─────────────────────────────────────────────────────────────────────────
        # Chargement des données (chunk-safe)
        # ─────────────────────────────────────────────────────────────────────────
        def load_data(file_path, chunksize=100_000):
            print(f"fonction load_data appelée depuis: {__file__}")
            try:
                chunks = pd.read_csv(
                    file_path,
                    sep=";",
                    chunksize=chunksize,
                    parse_dates=["date"],
                    on_bad_lines="skip",
                    low_memory=False,
                    encoding="utf-8",
                )
                parts = list(chunks)
                if not parts:
                    print(f"Aucun chunk lu: {file_path}")
                    return None
                df = pd.concat(parts).sort_values(by="date")
                print(f"✅ Fichier lu ({len(df)} lignes): {file_path}")
                return df
            except Exception as e:
                raise RuntimeError(f"Erreur lors du chargement de {file_path}: {e}")

        print("Chargement des données d'entraînement...")
        train_cluster = load_data(train_file)
        if train_cluster is None:
            raise ValueError("Train vide ou invalide")
        train_cluster["date"] = pd.to_datetime(train_cluster["date"], errors="coerce")
        train_cluster = train_cluster.dropna(subset=["date"]).copy()
        train_cluster = train_cluster.set_index("date")
        train_cluster["Year"] = train_cluster.index.year.astype("int16")
        train_cluster["Month"] = train_cluster.index.month.astype("int8")

        # Split temporel simple
        train_cluster_ST = train_cluster[train_cluster["Year"] < 2024].copy()
        test_cluster_ST = train_cluster[train_cluster["Year"] >= 2024].copy()

        print("\nChargement des données de test...")
        test_cluster = load_data(test_file)
        if test_cluster is None:
            raise ValueError("Test vide ou invalide")
        test_cluster["date"] = pd.to_datetime(test_cluster["date"], errors="coerce")

        # ─────────────────────────────────────────────────────────────────────────
        # Enrichissement géo: codePostal via sjoin (points ∈ polygones)
        # ─────────────────────────────────────────────────────────────────────────
        print("\nChargement des polygones de codes postaux...")
        pcodes = gpd.read_file(geo_file)[["codePostal", "geometry"]]
        pcodes = pcodes.set_geometry("geometry").to_crs(epsg=4326)
        _ = pcodes.sindex  # index spatial
        print("Polygones chargés :", pcodes.shape)

        train_cluster_ST = train_cluster_ST.reset_index()
        test_cluster_ST = test_cluster_ST.reset_index()

        train_cluster_ST["split"] = "train"
        test_cluster_ST["split"] = "train_test"
        test_cluster["split"] = "test"

        df_cluster = pd.concat(
            [train_cluster_ST, test_cluster_ST, test_cluster], ignore_index=True
        )

        # Drop lat/lon manquants et prépare points
        df_base = df_cluster.dropna(
            subset=["mapCoordonneesLatitude", "mapCoordonneesLongitude"]
        ).copy()
        df_base["lat"] = df_base["mapCoordonneesLatitude"]
        df_base["lon"] = df_base["mapCoordonneesLongitude"]
        df_base["orig_index"] = df_base.index

        def process_chunk(chunk, pcodes_gdf):
            chunk = chunk.copy()
            chunk["geometry"] = gpd.points_from_xy(chunk["lon"], chunk["lat"])
            gdf = gpd.GeoDataFrame(chunk, geometry="geometry", crs="EPSG:4326")
            gdf = gdf[gdf.is_valid]
            if gdf.crs != pcodes_gdf.crs:
                gdf = gdf.to_crs(pcodes_gdf.crs)
            _ = gdf.sindex
            joined = gpd.sjoin(gdf, pcodes_gdf, how="left", predicate="within")
            return joined[["orig_index", "codePostal"]]

        results = []
        chunksize = 100_000
        for i in range(0, len(df_base), chunksize):
            chunk = df_base.iloc[i : i + chunksize]
            results.append(process_chunk(chunk, pcodes))

        df_joined = pd.concat(results, ignore_index=True).drop_duplicates("orig_index")
        df_base = df_base.merge(df_joined, on="orig_index", how="left")
        df_base = df_base.drop(columns=["orig_index"])

        # Nettoyage date/index + codePostal
        df_base["date"] = pd.to_datetime(df_base["date"], errors="coerce")
        df_base = df_base.sort_values("date").set_index("date")
        df_base["codePostal"] = (
            df_base["codePostal"].astype(str).str.replace(r"\.0$", "", regex=True)
        )

        # ─────────────────────────────────────────────────────────────────────────
        # Variable hybride zone_mixte (CP détaillé si >=10 obs, sinon département)
        # ─────────────────────────────────────────────────────────────────────────
        train_cluster = df_base[df_base["split"] == "train"].copy()
        test_cluster = df_base[df_base["split"].isin(["test", "train_test"])].copy()

        cp_counts = train_cluster["codePostal"].value_counts()
        cp_frequents = set(cp_counts[cp_counts >= 10].index)

        def regroup_code(cp: str, frequents_set):
            cp = str(cp)
            if cp in frequents_set:
                return cp
            if cp.startswith("97") and len(cp) >= 3:
                return cp[:3]  # DROM
            if cp.isdigit() and len(cp) == 5:
                return cp[:2]  # département
            return "inconnu"

        train_cluster["zone_mixte"] = (
            train_cluster["codePostal"]
            .astype(str)
            .apply(lambda x: regroup_code(x, cp_frequents))
        )
        test_cluster["zone_mixte"] = (
            test_cluster["codePostal"]
            .astype(str)
            .apply(lambda x: regroup_code(x, cp_frequents))
        )

        # Lags dans le train
        train_cluster = train_cluster.sort_values(["zone_mixte", "date"])
        train_cluster["prix_lag_1m"] = train_cluster.groupby("zone_mixte")[
            "prix_m2_vente"
        ].shift(1)
        train_cluster["prix_roll_3m"] = (
            train_cluster.groupby("zone_mixte")["prix_m2_vente"]
            .rolling(3, closed="left")
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Agrégations mensuelles
        train_cluster["Year"] = train_cluster.index.year.astype(int)
        train_cluster["Month"] = train_cluster.index.month.astype(int)

        train_mensuel = (
            train_cluster.groupby(["Year", "Month", "zone_mixte"])
            .agg(
                prix_m2_vente=("prix_m2_vente", "mean"),
                volume_ventes=("prix_m2_vente", "count"),
            )
            .reset_index()
        )

        def get_code_postal_final(zone):
            s = str(zone)
            if s.isdigit() and len(s) == 5:
                return s
            if s.isdigit() and len(s) == 2:
                return s + "000"
            if s.startswith("97") and len(s) == 3:
                return s + "00"
            return "inconnu"

        train_mensuel["date"] = pd.to_datetime(
            dict(
                year=train_mensuel["Year"].astype(int),
                month=train_mensuel["Month"].astype(int),
                day=1,
            ),
            errors="raise",
        )
        train_mensuel["codePostal_recons"] = train_mensuel["zone_mixte"].apply(
            get_code_postal_final
        )
        train_mensuel = train_mensuel.sort_values(["codePostal_recons", "date"])
        train_mensuel["ym_ordinal"] = (
            train_mensuel["Year"] * 12 + train_mensuel["Month"]
        )
        train_mensuel["t"] = train_mensuel.groupby("codePostal_recons")[
            "ym_ordinal"
        ].transform(lambda x: x - x.min())

        # TCAM via régression sur log(prix)
        train_mensuel["log_prix"] = np.log(train_mensuel["prix_m2_vente"])

        def compute_tcam(df_):
            if len(df_) < 2 or df_["log_prix"].isna().any():
                return np.nan
            X = df_[["t"]].values.reshape(-1, 1)
            y = df_["log_prix"].values
            coef = LinearRegression().fit(X, y).coef_[0]
            return (np.exp(coef) - 1) * 100 * 12

        tcam_df = (
            train_mensuel.groupby("codePostal_recons")
            .apply(compute_tcam)
            .reset_index(name="tc_am_reg")
        )

        # Dataset features pour clustering
        train_mensuel = (
            train_mensuel.merge(tcam_df, on="codePostal_recons", how="left")
            .rename(columns={"prix_m2_vente": "prix_m2_mean"})
            .dropna(subset=["tc_am_reg"])
            .reset_index(drop=True)
        )

        df_cluster_input = (
            train_mensuel.groupby("codePostal_recons")
            .agg(
                prix_m2_mean=("prix_m2_mean", "mean"),
                prix_m2_std=("prix_m2_mean", "std"),
                prix_m2_max=("prix_m2_mean", "max"),
                prix_m2_min=("prix_m2_mean", "min"),
                avg_lag_1m=("prix_m2_mean", "mean"),  # proxies si besoin
                avg_roll_3m=("prix_m2_mean", "mean"),
            )
            .assign(prix_m2_cv=lambda df: df["prix_m2_std"] / df["prix_m2_mean"])
            .reset_index()
            .merge(tcam_df, on="codePostal_recons", how="left")
        )

        Path("mlflow_outputs").mkdir(exist_ok=True)
        df_cluster_input.to_csv(
            "mlflow_outputs/cluster_input.csv", index=False, sep=";"
        )
        mlflow.log_artifact("mlflow_outputs/cluster_input.csv")

        # ─────────────────────────────────────────────────────────────────────────
        # Clustering KMeans (k choisi avec coude)
        # ─────────────────────────────────────────────────────────────────────────
        features = [
            "prix_m2_std",
            "prix_m2_max",
            "prix_m2_min",
            "tc_am_reg",
            "prix_m2_cv",
            "avg_roll_3m",
            "avg_lag_1m",
        ]
        X = df_cluster_input[features].replace([np.inf, -np.inf], np.nan)
        X_train = X.dropna()
        train_idx = X_train.index

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)

        inertias = []
        ks = list(range(2, 10))
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_train_scaled)
            inertias.append(km.inertia_)

        plt.figure()
        plt.plot(ks, inertias, marker="o")
        plt.title("Méthode du coude – Inertie intra-cluster")
        plt.xlabel("Nombre de clusters")
        plt.ylabel("Inertie")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("mlflow_outputs/elbow_plot.png")
        mlflow.log_artifact("mlflow_outputs/elbow_plot.png")
        plt.close()

        # Fit final (k=4 par défaut)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_train_scaled)
        df_cluster_input.loc[train_idx, "cluster"] = labels.astype(int)

        # Mapping lisible
        cluster_order = (
            df_cluster_input.groupby("cluster")["prix_m2_mean"]
            .mean()
            .sort_values()
            .index.tolist()
        )
        cluster_names = [
            "Zones rurales, petites villes stagnantes",
            "Centres urbains établis, zones résidentielles",
            "Banlieues, zones mixtes",
            "Zones tendues - secteurs spéculatifs",
        ]
        mapping = dict(zip(cluster_order, cluster_names))
        df_cluster_input["cluster_label"] = df_cluster_input["cluster"].map(mapping)

        # Sauvegarde des assignations agrégées (optionnel)
        df_cluster_input.to_csv(
            "mlflow_outputs/cluster_input_labeled.csv", index=False, sep=";"
        )
        mlflow.log_artifact("mlflow_outputs/cluster_input_labeled.csv")

        # ─────────────────────────────────────────────────────────────────────────
        # ⚠️ MERGE DES LABELS DANS LES DONNÉES FINALES
        # ─────────────────────────────────────────────────────────────────────────
        mapping_df = df_cluster_input[
            ["codePostal_recons", "cluster", "cluster_label"]
        ].drop_duplicates(subset=["codePostal_recons"])

        # Reconstitution du full (train + test) en conservant index pour 'date'
        df_cluster_full = pd.concat([train_cluster, test_cluster]).copy()
        if isinstance(df_cluster_full.index, pd.DatetimeIndex):
            df_cluster_full = df_cluster_full.reset_index().rename(
                columns={"index": "date"}
            )

        # Recrée zone_mixte et codePostal_recons avec la même logique que plus haut
        df_cluster_full["zone_mixte"] = (
            df_cluster_full["codePostal"]
            .astype(str)
            .apply(lambda x: regroup_code(x, cp_frequents))
        )
        df_cluster_full["codePostal_recons"] = df_cluster_full["zone_mixte"].apply(
            get_code_postal_final
        )

        # Merge labels
        df_cluster_full = df_cluster_full.merge(
            mapping_df, on="codePostal_recons", how="left"
        )
        df_cluster_full["cluster"] = df_cluster_full["cluster"].astype("Int64")
        df_cluster_full["cluster_label"] = df_cluster_full["cluster_label"].fillna(
            "inconnu"
        )

        # ─────────────────────────────────────────────────────────────────────────
        # Exports finaux pour le pipeline
        # ─────────────────────────────────────────────────────────────────────────
        # Série temporelle (sans 'split'), labels inclus
        df_cluster_ST = df_cluster_full.drop(columns=["split"], errors="ignore")
        df_cluster_ST.to_csv(out_st_csv, sep=";", index=False)
        mlflow.log_artifact(str(out_st_csv))

        # Données pour régression (on force 'train_test' -> 'train'), labels inclus
        df_cluster_reg = df_cluster_full.copy()
        if "split" in df_cluster_reg.columns:
            df_cluster_reg["split"] = df_cluster_reg["split"].replace(
                "train_test", "train"
            )
        df_cluster_reg.to_csv(out_cluster_csv, sep=";", index=False)
        mlflow.log_artifact(str(out_cluster_csv))

        # Duplicatas pour DVC
        dvc_out_dir = Path("data")
        dvc_out_dir.mkdir(parents=True, exist_ok=True)
        (dvc_out_dir / "df_sales_clean_ST.csv").write_text(
            "", encoding="utf-8"
        )  # ensure path exists (optional)
        df_cluster_ST.to_csv(
            dvc_out_dir / "df_sales_clean_ST.csv", sep=";", index=False
        )
        df_cluster_reg.to_csv(dvc_out_dir / "df_cluster.csv", sep=";", index=False)

        print("✅ Exports (exports/ + data/ pour DVC) :")
        print(f"  - exports/df_cluster.csv        → {out_cluster_csv}")
        print(f"  - exports/df_sales_clean_ST.csv → {out_st_csv}")
        print(f"  - data/df_cluster.csv           → {dvc_out_dir/'df_cluster.csv'}")
        print(
            f"  - data/df_sales_clean_ST.csv    → {dvc_out_dir/'df_sales_clean_ST.csv'}"
        )


if __name__ == "__main__":
    run_clustering_pipeline(
        input_path="data/processed/",
        output_path="exports/df_cluster.csv",
    )
