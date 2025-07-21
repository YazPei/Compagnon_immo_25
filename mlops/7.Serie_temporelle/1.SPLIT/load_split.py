#1
import os
import argparse
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import mlflow


def enrich_and_split(input_path, taux_path, geo_path, output_folder, suffix=""):
    os.makedirs(output_folder, exist_ok=True)

    mlflow.set_experiment("ST-Split-Full")
    with mlflow.start_run(run_name="full_encoding"):

        df = pd.read_csv(input_path, sep=";", parse_dates=["date"])
        df = df.dropna(subset=["mapCoordonneesLatitude", "mapCoordonneesLongitude"])
        df["lat"] = df["mapCoordonneesLatitude"]
        df["lon"] = df["mapCoordonneesLongitude"]
        df["orig_index"] = df.index

        # -------- Chargement des polygones geojson ----------
        pcodes = gpd.read_file(geo_path)[["codePostal", "geometry"]].set_geometry("geometry").to_crs(epsg=4326)
        pcodes.sindex  # spatial index

        # -------- Attribution codePostal via spatial join ----------
        def process_chunk(chunk, pcodes):
            chunk = chunk.copy()
            chunk["geometry"] = gpd.points_from_xy(chunk["lon"], chunk["lat"])
            gdf = gpd.GeoDataFrame(chunk, geometry="geometry", crs="EPSG:4326")
            gdf = gdf[gdf.is_valid]
            joined = gpd.sjoin(gdf, pcodes, how="left", predicate="within")
            return joined[["orig_index", "codePostal"]]

        results = [process_chunk(df.iloc[i:i+100000], pcodes) for i in range(0, len(df), 100000)]
        df_joined = pd.concat(results, ignore_index=True).drop_duplicates("orig_index")

        df = df.merge(df_joined, on="orig_index", how="left").drop(columns=["orig_index"])
        df["Year"] = df["date"].dt.year
        df["Month"] = df["date"].dt.month
        df["codePostal"] = df["codePostal"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
        df["departement"] = df["codePostal"].str[:2]

        # -------- Split train / test ----------
        train = df[(df["Year"] < 2024) & (df["Year"] > 2019)].copy()
        test = df[df["Year"] >= 2024].copy()

        # -------- Encodage sphérique ----------
        def add_geo_coords(df):
            lat_rad = np.radians(df["mapCoordonneesLatitude"].values)
            lon_rad = np.radians(df["mapCoordonneesLongitude"].values)
            df["x_geo"] = np.cos(lat_rad) * np.cos(lon_rad)
            df["y_geo"] = np.cos(lat_rad) * np.sin(lon_rad)
            df["z_geo"] = np.sin(lat_rad)
            return df.drop(columns=["mapCoordonneesLatitude", "mapCoordonneesLongitude"])

        train = add_geo_coords(train)
        test = add_geo_coords(test)

        # -------- Encodage DPE ----------
        train["dpeL"] = train["dpeL"].astype(str)
        test["dpeL"] = test["dpeL"].astype(str)
        pipe_dpe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        train["dpeL"] = pipe_dpe.fit_transform(train["dpeL"].values.reshape(-1, 1))
        test["dpeL"] = pipe_dpe.transform(test["dpeL"].values.reshape(-1, 1))

        # -------- Standardisation ----------
        variables_exp = ["taux_rendement_n7", "loyer_m2_median_n7", "y_geo", "x_geo", "z_geo", "dpeL", "nb_pieces", "IPS_primaire", "rental_yield_pct"]
        scaler = StandardScaler()
        train[variables_exp] = scaler.fit_transform(train[variables_exp])
        test[variables_exp] = scaler.transform(test[variables_exp])

        # -------- Ajout taux d'emprunt ----------
        taux = pd.read_excel(taux_path)
        taux["date"] = pd.to_datetime(taux["date"])
        taux = taux.set_index("date")
        taux["taux"] = taux["10 ans"].str.replace("%", "").str.replace(",", ".").astype(float)

        for df_temp, label in [(train, "train"), (test, "test")]:
            df_temp["date"] = pd.to_datetime(df_temp["date"])
            df_temp.sort_values("date", inplace=True)

        # Agrégation mensuelle
        def aggregate(df, split_label):
            agg = df.groupby(["cluster", "date"]).agg({
                "prix_m2_vente": "mean",
                **{col: "mean" for col in variables_exp}
            }).reset_index()
            agg["split"] = split_label
            return agg

        agg_train = aggregate(train, "train")
        agg_test = aggregate(test, "test")

        agg_train = agg_train.merge(taux[["taux"]], left_on="date", right_index=True, how="left")
        agg_test = agg_test.merge(taux[["taux"]], left_on="date", right_index=True, how="left")

        # Standardisation du taux
        scal = StandardScaler()
        agg_train["taux"] = scal.fit_transform(agg_train[["taux"]])
        agg_test["taux"] = scal.transform(agg_test[["taux"]])

        # Log transformation
        agg_train["prix_m2_vente"] = np.log(agg_train["prix_m2_vente"])
        agg_test["prix_m2_vente"] = np.log(agg_test["prix_m2_vente"])

        # Export
        final_vars = variables_exp + ["taux", "prix_m2_vente", "cluster", "date"]
        train_clean_path = os.path.join(output_folder, f"train_clean_ST{suffix}.csv")
        test_clean_path = os.path.join(output_folder, f"test_clean_ST{suffix}.csv")
        train_q12_path = os.path.join(output_folder, f"train_periodique_q12{suffix}.csv")
        test_q12_path = os.path.join(output_folder, f"test_periodique_q12{suffix}.csv")

        train.to_csv(train_clean_path, sep=";", index=False)
        test.to_csv(test_clean_path, sep=";", index=False)
        agg_train[final_vars].to_csv(train_q12_path, sep=";", index=False)
        agg_test[final_vars].to_csv(test_q12_path, sep=";", index=False)

        # MLflow logging
        mlflow.log_artifact(train_clean_path)
        mlflow.log_artifact(test_clean_path)
        mlflow.log_artifact(train_q12_path)
        mlflow.log_artifact(test_q12_path)
        
        print("✅ Données enrichies et exportées.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Préparation + split SARIMAX par cluster")
    parser.add_argument("--input-path", type=str, required=True, help="Chemin vers df_sales_clean_ST.csv")
    parser.add_argument("--taux-path", type=str, required=True, help="Chemin vers Taux immo.xlsx")
    parser.add_argument("--geo-path", type=str, required=True, help="Chemin vers contours-codes-postaux.geojson")
    parser.add_argument("--output-folder", type=str, required=True, help="Dossier de sortie")
    parser.add_argument("--suffix", type=str, default="", help="Suffixe à ajouter aux fichiers exportés (ex: _v2)")

    args = parser.parse_args()
    
    enrich_and_split(args.input_path, args.taux_path, args.geo_path, args.output_folder, args.suffix)


