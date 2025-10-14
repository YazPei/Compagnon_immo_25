# mlops/7_Serie_temporelle/1_SPLIT/load_split.py
# Version robuste et corrigée : merge CP, cluster, agrégation mensuelle, standardisation robuste

import os
import logging
from pathlib import Path
import csv
import mlflow

import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# --- MLflow: wrapper no-op si non installé ---
class _NoOpRun:
    def __enter__(self): return self
    def __exit__(self, *a): pass
class _NoOpMLflow:
    def set_experiment(self, *a, **k): pass
    def start_run(self, *a, **k): return _NoOpRun()
    def log_artifact(self, *a, **k): pass
try:
    import mlflow as _mlflow
    MLFLOW = _mlflow
except Exception:
    print("⚠️  MLflow introuvable — exécution sans tracking.")
    MLFLOW = _NoOpMLflow()

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# ========= CHEMINS EN DUR =========
INPUT_PATH     = "data/df_sales_clean_ST.csv"
TAUX_PATH      = "data/clean/taux_immo.xlsx"
GEO_PATH       = "data/processed/contours-codes-postaux.geojson"
OUTPUT_FOLDER  = "data/split"
SUFFIX         = os.getenv("ST_SUFFIX", "")  # laisse vide si tu veux

# Fichiers possibles pour remapper 'cluster'
CLUSTER_MAPPING_CANDIDATES = [
    "exports/df_cluster.csv",
    "data/processed/df_sales_clustered.csv",
    "data/processed/df_cluster.csv",
]
# ==================================

CANDIDATE_CP_COLS = [
    "codePostal", "code_postal", "codepostal", "postal_code",
    "postcode", "cp", "Code_Postal", "CODE_POSTAL", "CODEPOSTAL"
]

def read_csv_autodelim(path: str) -> pd.DataFrame:
    with open(path, "r", newline="", encoding="utf-8") as f:
        sample = f.read(4096); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
            sep = dialect.delimiter
        except Exception:
            sep = ";"
    return pd.read_csv(path, sep=sep, low_memory=False)

def normalize_cp_column(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    found = next((c for c in CANDIDATE_CP_COLS if c in gdf.columns), None)
    if found is None:
        raise KeyError(
            f"Aucune colonne code postal trouvée dans le GeoJSON. Colonnes: {list(gdf.columns)}. "
            f"Attendu l'une de: {CANDIDATE_CP_COLS}"
        )
    if found != "codePostal":
        gdf = gdf.rename(columns={found: "codePostal"})
    gdf["codePostal"] = (
        gdf["codePostal"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
        .str.zfill(5)
    )
    return gdf

def process_chunk(chunk: pd.DataFrame, pcodes_gdf: gpd.GeoDataFrame, debug: bool = False) -> pd.DataFrame:
    """Attribue codePostal via cascades: within → intersects → nearest L93 (5km) → nearest WGS84 (500km)."""
    chunk = chunk.copy()
    for c in list(chunk.columns):
        if c.lower().replace("_", "") == "codepostal":
            chunk.drop(columns=[c], inplace=True, errors="ignore")

    points = gpd.points_from_xy(chunk["lon"], chunk["lat"])
    gdf_local = gpd.GeoDataFrame(chunk, geometry=points, crs="EPSG:4326")

    cp_candidates = [
        "codePostal_pc", "codePostal__pc",
        "code_postal_pc", "code_postal__pc",
        "CODE_POSTAL_pc", "CODE_POSTAL__pc",
        "Code_Postal_pc", "Code_Postal__pc",
        "codePostal_right", "codePostal"
    ]
    def _pick_cp_col(dfj: pd.DataFrame):
        return next((c for c in cp_candidates if c in dfj.columns), None)

    # 1) within
    joined = gpd.sjoin(gdf_local, pcodes_gdf, how="left", predicate="within",
                       lsuffix="_row", rsuffix="_pc")
    cp_col = _pick_cp_col(joined)
    if debug:
        logging.info(f"[debug] within: cp_col={cp_col}, hit={(None if cp_col is None else (~joined[cp_col].isna()).sum())}")
    if cp_col is not None and not joined[cp_col].isna().all():
        if cp_col != "codePostal":
            joined = joined.rename(columns={cp_col: "codePostal"})
        return joined[["orig_index", "codePostal"]]

    # 2) intersects
    joined = gpd.sjoin(gdf_local, pcodes_gdf, how="left", predicate="intersects",
                       lsuffix="_row", rsuffix="_pc")
    cp_col = _pick_cp_col(joined)
    if cp_col is not None and not joined[cp_col].isna().all():
        if cp_col != "codePostal":
            joined = joined.rename(columns={cp_col: "codePostal"})
        return joined[["orig_index", "codePostal"]]

    # 3) nearest (métropole)
    try:
        gdf_local_l93 = gdf_local.to_crs(2154)
        pcodes_l93 = pcodes_gdf.to_crs(2154)
        joined_near = gpd.sjoin_nearest(
            gdf_local_l93, pcodes_l93, how="left",
            max_distance=5000,
            lsuffix="_row", rsuffix="_pc",
        ).to_crs(4326)
        cp_col = _pick_cp_col(joined_near)
        if cp_col is not None and not joined_near[cp_col].isna().all():
            joined = joined_near
            if cp_col != "codePostal":
                joined = joined.rename(columns={cp_col: "codePostal"})
            return joined[["orig_index", "codePostal"]]
    except Exception as e:
        if debug:
            logging.info(f"[debug] nearest L93 failed: {e}")

    # 4) nearest large (DOM/TOM)
    joined = gpd.sjoin_nearest(
        gdf_local, pcodes_gdf, how="left",
        max_distance=5.0,
        lsuffix="_row", rsuffix="_pc"
    )
    cp_col = _pick_cp_col(joined)
    if cp_col is None:
        joined["codePostal"] = pd.NA
        return joined[["orig_index", "codePostal"]]
    if cp_col != "codePostal":
        joined = joined.rename(columns={cp_col: "codePostal"})
    return joined[["orig_index", "codePostal"]]

def enrich_and_split():
    for p in [INPUT_PATH, TAUX_PATH, GEO_PATH]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Fichier introuvable: {Path(p).resolve()}")
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    mlflow.set_experiment("ST-Split-Full")
    with mlflow.start_run(run_name="full_encoding"):

        # -------- Données ventes --------
        df = pd.read_csv(INPUT_PATH, sep=";", parse_dates=["date"], low_memory=False)
        df = df.dropna(subset=["mapCoordonneesLatitude", "mapCoordonneesLongitude"])
        df["lat"] = pd.to_numeric(df["mapCoordonneesLatitude"], errors="coerce")
        df["lon"] = pd.to_numeric(df["mapCoordonneesLongitude"], errors="coerce")
        df["orig_index"] = df.index

        bad = df["lat"].isna() | df["lon"].isna()
        df_valid = df.loc[~bad].copy()

        in_metro_valid = (df_valid["lat"].between(41.0, 51.5)) & (df_valid["lon"].between(-5.5, 9.8))

        # -------- Polygones --------
        pcodes = gpd.read_file(GEO_PATH)
        pcodes = normalize_cp_column(pcodes)
        pcodes = pcodes[["codePostal", "geometry"]].set_geometry("geometry").to_crs(epsg=4326)
        _ = pcodes.sindex

        # -------- Attribution CP --------
        df_metro = df_valid.loc[in_metro_valid].copy()
        df_dom   = df_valid.loc[~in_metro_valid].copy()

        results = []
        chunksize = 100_000
        for local in (df_metro, df_dom):
            n = len(local)
            for i in range(0, n, chunksize):
                part = local.iloc[i:i+chunksize]
                res = process_chunk(part, pcodes, debug=(i == 0))
                results.append(res)

        if results:
            df_joined = pd.concat(results, ignore_index=True).drop_duplicates("orig_index")
        else:
            df_joined = pd.DataFrame(columns=["orig_index", "codePostal"])

        # Merge sécurisé
        cp_like_left = [c for c in df.columns if c.lower().replace("_", "") == "codepostal"]
        if cp_like_left:
            df = df.drop(columns=cp_like_left)
        df = df.merge(df_joined, on="orig_index", how="left", suffixes=("", "_cpjoin"))
        df = df.drop(columns=["orig_index"])
        if "codePostal_cpjoin" in df.columns:
            if "codePostal" in df.columns:
                df["codePostal"] = df["codePostal"].astype("string").fillna(df["codePostal_cpjoin"].astype("string"))
                df = df.drop(columns=["codePostal_cpjoin"])
            else:
                df = df.rename(columns={"codePostal_cpjoin": "codePostal"})

        df["codePostal"] = df["codePostal"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(5)
        df["departement"] = df["codePostal"].str[:2]

        # -------- Cluster --------
        if "cluster" not in df.columns:
            map_path = next((p for p in CLUSTER_MAPPING_CANDIDATES if os.path.exists(p)), None)
            if map_path is None:
                df["cluster"] = 0
            else:
                map_df = read_csv_autodelim(map_path)
                merged = False
                if "INSEE_COM" in df.columns and "INSEE_COM" in map_df.columns and "cluster" in map_df.columns:
                    df["INSEE_COM"] = df["INSEE_COM"].astype(str).str.zfill(5)
                    map_df["INSEE_COM"] = map_df["INSEE_COM"].astype(str).str.zfill(5)
                    df = df.merge(map_df[["INSEE_COM", "cluster"]].drop_duplicates(), on="INSEE_COM", how="left")
                    merged = True
                if not merged and "codePostal" in map_df.columns and "cluster" in map_df.columns:
                    map_df["codePostal"] = map_df["codePostal"].astype(str).str.zfill(5)
                    df = df.merge(map_df[["codePostal", "cluster"]].drop_duplicates(), on="codePostal", how="left")
                    merged = True
                if not merged:
                    df["cluster"] = 0
                df["cluster"] = df["cluster"].fillna(0)

        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").astype("Int64")
        if "cluster_label" in df.columns:
            df["cluster_label"] = df["cluster_label"].fillna("inconnu")
        else:
            df["cluster_label"] = "inconnu"

        # -------- Split train / test ----------
        df["Year"] = df["date"].dt.year
        train = df[(df["Year"] < 2024) & (df["Year"] > 2019)].copy()
        test  = df[df["Year"] >= 2024].copy()

        # -------- Encodage sphérique ----------
        def add_geo_coords(df_):
            lat_rad = np.radians(df_["mapCoordonneesLatitude"].values)
            lon_rad = np.radians(df_["mapCoordonneesLongitude"].values)
            df_["x_geo"] = np.cos(lat_rad) * np.cos(lon_rad)
            df_["y_geo"] = np.cos(lat_rad) * np.sin(lon_rad)
            df_["z_geo"] = np.sin(lat_rad)
            return df_.drop(columns=["mapCoordonneesLatitude", "mapCoordonneesLongitude"])

        train = add_geo_coords(train)
        test  = add_geo_coords(test)

        # -------- Encodage DPE ----------
        for df_temp in (train, test):
            if "dpeL" not in df_temp.columns:
                df_temp["dpeL"] = np.nan
            df_temp["dpeL"] = df_temp["dpeL"].astype(str)
        pipe_dpe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        train["dpeL"] = pipe_dpe.fit_transform(train["dpeL"].values.reshape(-1, 1))
        test["dpeL"]  = pipe_dpe.transform(test["dpeL"].values.reshape(-1, 1))

        # -------- Standardisation ----------
        variables_exp = [
            "taux_rendement_n7", "loyer_m2_median_n7",
            "y_geo", "x_geo", "z_geo",
            "dpeL", "nb_pieces", "IPS_primaire", "rental_yield_pct"
        ]
        for col in variables_exp:
            if col not in train.columns:
                train[col] = np.nan
            if col not in test.columns:
                test[col] = np.nan
        train_med = train[variables_exp].median(numeric_only=True)
        train[variables_exp] = train[variables_exp].fillna(train_med)
        test[variables_exp]  = test[variables_exp].fillna(train_med)
        scaler = StandardScaler()
        train[variables_exp] = scaler.fit_transform(train[variables_exp])
        test[variables_exp]  = scaler.transform(test[variables_exp])

        # -------- Ajout taux d'emprunt ----------
        taux = pd.read_excel(TAUX_PATH)
        taux["date"] = pd.to_datetime(taux["date"])
        taux_col = "10 ans" if "10 ans" in taux.columns else taux.columns[0]
        if taux[taux_col].dtype == "object":
            taux["taux"] = taux[taux_col].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False).astype(float)
        else:
            taux["taux"] = taux[taux_col].astype(float)
        # >> FIX: pas d'alias "MS" ici
        taux["month"] = taux["date"].dt.to_period("M").dt.to_timestamp()
        taux_m = taux.groupby("month", as_index=False)["taux"].mean()

        def monthly_aggregate(df_, split_label, variables_exp):
            # >> FIX: pas d'alias "MS" ici
            df_["month"] = df_["date"].dt.to_period("M").dt.to_timestamp()
            agg = df_.groupby(["cluster", "month"], as_index=False).agg({
                "prix_m2_vente": "mean",
                **{col: "mean" for col in variables_exp}
            })
            agg = agg.rename(columns={"month": "date"})
            agg["split"] = split_label
            return agg

        agg_train = monthly_aggregate(train.copy(), "train", variables_exp)
        agg_test  = monthly_aggregate(test.copy(),  "test",  variables_exp)

        agg_train = agg_train.merge(taux_m, left_on="date", right_on="month", how="left").drop(columns=["month"])
        agg_test  = agg_test.merge(taux_m, left_on="date", right_on="month", how="left").drop(columns=["month"])

        scal = StandardScaler()
        agg_train["taux"] = scal.fit_transform(agg_train[["taux"]])
        agg_test["taux"]  = scal.transform(agg_test[["taux"]])

        agg_train["prix_m2_vente"] = np.log(agg_train["prix_m2_vente"])
        agg_test["prix_m2_vente"]  = np.log(agg_test["prix_m2_vente"])

        # -------- Export ----------
        final_vars = variables_exp + ["taux", "prix_m2_vente", "cluster", "date"]
        train_clean_path = os.path.join(OUTPUT_FOLDER, f"train_clean_ST{SUFFIX}.csv")
        test_clean_path  = os.path.join(OUTPUT_FOLDER, f"test_clean_ST{SUFFIX}.csv")
        train_q12_path   = os.path.join(OUTPUT_FOLDER, f"train_periodique_q12{SUFFIX}.csv")
        test_q12_path    = os.path.join(OUTPUT_FOLDER, f"test_periodique_q12{SUFFIX}.csv")

        train.to_csv(train_clean_path, sep=";", index=False)
        test.to_csv(test_clean_path, sep=";", index=False)
        agg_train[final_vars].to_csv(train_q12_path, sep=";", index=False)
        agg_test[final_vars].to_csv(test_q12_path, sep=";", index=False)

        mlflow.log_artifact(train_clean_path)
        mlflow.log_artifact(test_clean_path)
        mlflow.log_artifact(train_q12_path)
        mlflow.log_artifact(test_q12_path)

        print("✅ Données enrichies et exportées.")

if __name__ == "__main__":
    enrich_and_split()

