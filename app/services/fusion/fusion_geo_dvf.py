import os
import time
from pathlib import Path

import click
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from sklearn.neighbors import BallTree


def load_and_clean_data(folder_path1, folder_path2):
    input_file1 = os.path.join(folder_path1, "merged_sales_data.csv")
    input_file2 = os.path.join(folder_path2, "DVF_donnees_macroeco.csv")

    chunks1 = pd.read_csv(
        input_file1, sep=";", chunksize=100_000, on_bad_lines="skip", low_memory=False
    )
    df_sales_clean = pd.concat(chunks1)

    chunks2 = pd.read_csv(
        input_file2, sep=",", chunksize=100_000, on_bad_lines="skip", low_memory=False
    )
    df_dvf = pd.concat(chunks2)

    df_sales_clean = df_sales_clean.astype(
        {col: str for col in df_sales_clean.select_dtypes(include="object").columns}
    )
    df_sales_clean["INSEE_COM"] = (
        df_sales_clean["INSEE_COM"].astype(str).str.zfill(5).str.strip()
    )
    df_dvf["INSEE_COM"] = df_dvf["Code INSEE de la commune"].astype(str).str.strip()

    df_dvf = df_dvf[
        (df_dvf["Rentrée scolaire"] == "2023-2024") & (df_dvf["Secteur"] == "public")
    ]
    df_dvf["IPS"] = (
        df_dvf["IPS"]
        .astype(str)
        .replace("NS", np.nan)
        .str.replace(",", ".")
        .astype(float)
    )

    return df_sales_clean, df_dvf


def extract_ips_levels(df_dvf):
    def extract_level(name):
        name = name.upper()
        if "PRIMAIRE" in name:
            return "primaire"
        if "ELEMENTAIRE" in name:
            return "elementaire"
        return np.nan

    df_dvf["level"] = df_dvf["Etablissement"].apply(extract_level)
    df_levels = (
        df_dvf.dropna(subset=["level"])
        .groupby(["INSEE_COM", "level"])["IPS"]
        .mean()
        .reset_index()
    )
    df_ips = (
        df_levels.pivot(index="INSEE_COM", columns="level", values="IPS")
        .rename(columns={"primaire": "IPS_primaire", "elementaire": "IPS_elementaire"})
        .reset_index()
    )
    return df_ips


def merge_macro_indicators(df_sales_clean, df_dvf, df_ips):
    df_dvf_small = df_dvf[
        ["avg_purchase_price_m2", "avg_rent_price_m2", "rental_yield_pct", "INSEE_COM"]
    ].drop_duplicates("INSEE_COM")
    df_ips_small = df_ips[["INSEE_COM", "IPS_primaire"]].drop_duplicates("INSEE_COM")

    for col in [
        "avg_purchase_price_m2",
        "avg_rent_price_m2",
        "rental_yield_pct",
        "IPS_primaire",
    ]:
        source = df_dvf_small if "avg" in col or "yield" in col else df_ips_small
        df_sales_clean[col] = df_sales_clean["INSEE_COM"].map(
            dict(zip(source["INSEE_COM"], source[col]))
        )

    return df_sales_clean


def impute_missing_ips(df_sales_clean):
    communes = (
        df_sales_clean.groupby("INSEE_COM")
        .agg(
            {
                "mapCoordonneesLatitude": "mean",
                "mapCoordonneesLongitude": "mean",
                "IPS_primaire": "mean",
            }
        )
        .reset_index()
    )

    complete = communes[communes["IPS_primaire"].notna()].copy()
    missing = communes[communes["IPS_primaire"].isna()].copy()

    tree = BallTree(
        np.radians(complete[["mapCoordonneesLatitude", "mapCoordonneesLongitude"]]),
        metric="haversine",
    )
    dist, idx = tree.query(
        np.radians(missing[["mapCoordonneesLatitude", "mapCoordonneesLongitude"]]), k=1
    )
    missing["IPS_primaire"] = complete.iloc[idx.flatten()]["IPS_primaire"].values

    communes_filled = pd.concat([complete, missing]).drop_duplicates("INSEE_COM")
    df_sales_clean["IPS_primaire"] = df_sales_clean["INSEE_COM"].map(
        dict(zip(communes_filled["INSEE_COM"], communes_filled["IPS_primaire"]))
    )

    return df_sales_clean, len(missing)


@click.command()
@click.option(
    "--folder-path1",
    type=click.Path(exists=True),
    prompt="📂 Chemin vers les données ventes",
)
@click.option(
    "--folder-path2",
    type=click.Path(exists=True),
    prompt="📂 Chemin vers les données DVF",
)
@click.option("--output-folder", type=click.Path(), prompt="📁 Dossier de sortie")
def main(folder_path1, folder_path2, output_folder):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("Fusion Données IPS")

    with mlflow.start_run(run_name="fusion_geo_dvf"):
        df_sales_clean, df_dvf = load_and_clean_data(folder_path1, folder_path2)
        df_ips = extract_ips_levels(df_dvf)
        df_sales_clean = merge_macro_indicators(df_sales_clean, df_dvf, df_ips)
        df_sales_clean, nb_imputed = impute_missing_ips(df_sales_clean)

        # === Tracking MLflow ===
        mlflow.log_param("nb_rows_sales", len(df_sales_clean))
        mlflow.log_param("nb_rows_dvf", len(df_dvf))
        mlflow.log_metric(
            "nb_missing_ips_initial",
            df_sales_clean["IPS_primaire"].isna().sum() + nb_imputed,
        )
        mlflow.log_metric("nb_imputed_ips", nb_imputed)
        mlflow.log_metric(
            "nb_final_missing_ips", df_sales_clean["IPS_primaire"].isna().sum()
        )
        mlflow.log_param("output_folder", output_folder)

        # === Export final ===
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_folder) / "df_sales_clean_ST.csv"
        pl.from_pandas(df_sales_clean).write_csv(output_path, separator=";")

        # === Log artefact ===
        mlflow.log_artifact(str(output_path))
        print(f"✅ Export CSV terminé : {output_path}")


if __name__ == "__main__":
    main()
