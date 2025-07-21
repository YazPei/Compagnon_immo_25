import os
import math
from pathlib import Path

import click
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mlflow

from mlops.preprocessing.utils import (
    annee_const,
    clean_classe,
    clean_exposition,
    extract_principal,
    get_numeric_cols,
    calculate_bounds,
    compute_medians,
    mark_outliers,
    clean_outliers,
    log_figure,
)


@click.command()
@click.option("--input-path", type=click.Path(exists=True), required=True)
@click.option("--output-path", type=click.Path(), required=True)
def main(input_path: str, output_path: str):
    run_preprocessing_pipeline(input_path, output_path)

def run_preprocessing_pipeline(input_path: str, output_path: str):
    # === BEGIN PIPELINE ===
    run_suffix = os.getenv("RUN_MODE", "default")
    GROUP_COL = "INSEE_COM"

    df = pl.read_csv(input_path, separator=";").to_pandas()
    print("Nombres de lignes en double", df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    print("Nombres de lignes en double après suppression", df.duplicated().sum())
    print("Shape du Dataset après élimination des doublons : ", df.shape)
    
    df = pl.read_csv(input_path, separator=";").to_pandas()
        print("Nombres de lignes en double", df.duplicated().sum())

        df.drop_duplicates(inplace=True)
        print("Nombres de lignes en double après suppression", df.duplicated().sum())
        print("Shape du Dataset après élimination des doublons : ", df.shape)
    ### Proportions des NANs
    missing_data_percentage_sales = df.isna().sum() * 100 / len(df)

    missing_value_percentage_sales = pd.DataFrame(
        {
            "column_name": df.columns,
            "percent_missing": missing_data_percentage_sales,
            "dtypes": df.dtypes,
        }
    ).sort_values(by="percent_missing", ascending=False)

    # Resetting the index to start from 1 for better readability
    # and to match the original DataFrame's index
    missing_value_percentage_sales.index = range(
        1, len(missing_value_percentage_sales) + 1
    )


    ### visualisation des doublons
    # 📊 Création de la figure
    fig, ax = plt.subplots(figsize=(10, 14))
    sns.barplot(
        y=missing_value_percentage_sales.column_name,
        x=missing_value_percentage_sales.percent_missing,
        hue=missing_value_percentage_sales.column_name,
        order=missing_value_percentage_sales.column_name,
        ax=ax
    )

	# Seuil visuel
    ax.axvline(x=75, color="red", linestyle="--", label="Threshold (75%)")

    # Style
    ax.set_title("Répartition des valeurs manquantes dans le dataset", fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    ax.set_ylabel("Features")
    ax.legend()
    
    # Sauvegarde
    output_dir = Path("/app/reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"missing_values_{run_suffix}.png"
    fig_path = os.path.join(output_dir, "Nan_distribution.png")
    fig.savefig(fig_path)
    
    # Log MLflow
    log_figure(
        fig,
        filename=filename,
        artifact_path="figures/missing"
    )
    
    #  Nettoyage mémoire
    plt.close(fig)

    ### Elimination de colonnes (valeurs manquantes supérieures à 75 %)
    # Filtrer les colonnes avec un taux de valeurs manquantes inférieur ou égal à 75%
    columns_to_keep = missing_data_percentage_sales[
        missing_data_percentage_sales <= 75
    ].index

    # Mettre à jour le DataFrame en gardant uniquement les colonnes sélectionnées
    df_1 = df[columns_to_keep]

    print("Colonnes conservées :", list(columns_to_keep))
    print("\nShape du Dataset après élimination des colonnes :", df_1.shape)

    # PRÉSÉLECTION (VARIABLES EXPLICATIVES À ÉLIMINER)
    df_2 = df_1.drop(
        columns=[
            "idannonce",
            "annonce_exclusive",
            "typedebien_lite",
            "type_annonceur",
            "categorie_annonceur",
            "REG",
            "DEP",
            "IRIS",
            "CODE_IRIS",
            "TYP_IRIS_x",
            "TYP_IRIS_y",
            "nb_logements_copro",
            "GRD_QUART",
            "UU2010",
            "duree_int",
        ],
        axis=1,
    )

    df_2.shape

    # VARIABLES EXPLICATIVES À TRAITER
    # les variables porte_digicode, ascenseur et cave sont des variables binaires typées en 'object'
    # nous les transofrmins en type boleeen
    bool_columns = ["porte_digicode", "cave", "ascenseur"]
    for col in bool_columns:
        df_2[col] = df_2[col].astype(bool)

    # vérification des types des colonnes converties
    df_2[bool_columns].dtypes

    ## Dicrétisations
    ### La variable "annee_construction" est transformée en variable catégorielle nominale

    # suppression des doublons
    df_2.dropna(
        subset=["prix_m2_vente", "surface_reelle_bati", "nombre_pieces_principales"],
        inplace=True,
    )
    df_2 = annee_const(df_2)
    # 2. Application des fonctions sur df_2

    # 2.1 : Nettoyage des colonnes DPE et GES
    for col in ("dpeL", "ges_class"):
        if col in df_2.columns:
            df_2[col] = (
                df_2[col]
                .apply(clean_classe)  # Applique clean_classe à chaque valeur
                .astype("object")  # Force le type chaîne pour les résultats
            )

    # 2.2 : Extraction de l'énergie de chauffage principale
    if "chauffage_energie" in df_2.columns:
        df_2["chauffage_energie_principal"] = (
            df_2["chauffage_energie"]
            .apply(extract_principal)  # Garde la première source énergétique
            .astype("object")
        )
    # Correction de l'encodage mal interprété (ex: Ã -> É)
    df_2["chauffage_energie_principal"] = df_2[
        "chauffage_energie_principal"
    ].str.replace("Ã\x89", "É", regex=False)

    # 2.3 : Nettoyage de la colonne exposition
    if "exposition" in df_2.columns:
        df_2["exposition"] = (
            df_2["exposition"]
            .apply(clean_exposition)  # Standardise les orientations
            .astype("object")
        )

        df_3 = df_2.drop(columns=["prix_bien", "mensualiteFinance"], axis=1)
    df_3.shape

    # Visualisation  de la distribution de la variable cible
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df_3, x="prix_m2_vente", bins=30)
    plt.title("Prix m2_vente Distribution")
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    output_dir = "/app/reports/figures"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig_path = os.path.join(output_dir, "prix_m2_distribution.png")
    plt.savefig(fig_path)
    plt.close()
       
        
        
    # visualisation des variables catégorielles
    numeric_cols = get_numeric_cols(df_3, GROUP_COL)

    # Visualisation des boxplots pour les colonnes restantes

    # Nombre de colonnes par ligne
    cols_per_row = 2

    # Calcul du nombre de lignes nécessaires
    num_cols = len(numeric_cols)
    num_rows = math.ceil(num_cols / cols_per_row)

    # Création des sous-graphiques
    fig_o, axes = plt.subplots(num_rows, cols_per_row, figsize=(12, 4 * num_rows))
    axes = axes.flatten()  # Aplatir pour un accès plus simple

    # Boucle pour tracer les boxplots
    for i, col in enumerate(numeric_cols):
        df_3.boxplot(column=col, ax=axes[i])
        axes[i].set_title(f"Boxplot de la colonne '{col}'")

    # Supprimer les axes inutilisés si le nombre de colonnes est impair
    for j in range(i + 1, len(axes)):
        fig_o.delaxes(axes[j])


        plt.tight_layout()

	# 1. Sauvegarde dans un dossier temporaire (compatible Docker)
	output_dir = Path("/app/reports/figures")
	output_dir.mkdir(parents=True, exist_ok=True)
	filename = f"boxplots_outliers_{run_suffix}.png"
	fig_path = os.path.join(output_dir, "Boxplot_variables.png")
	fig_o.savefig(fig_path)
	

	# 2. Log dans MLflow
        log_figure(
        fig_o,
        filename=filename,
        artifact_path="figures/boxplots"
        )

	# 3. Fermeture propre
        plt.close(fig_o)


    ## Anomalies
    ### Détection des anomalies logiques entre colonnes
    # Détection d'anomalies logiques dans les données suite aux boxplots

    # Création d'une colonne 'anomalie_logique' contenant True si une incohérence est détectée

    # Copie pour éviter d'altérer la base brute directement
    df_logic_check = df_3.copy()

    # Initialiser la colonne avec False
    df_logic_check["anomalie_logique"] = False

    # --- Règle 1 : nb_toilettes > nb_pieces (pas logique dans un logement classique)
    mask_toilettes = df_logic_check["nb_toilettes"] > df_logic_check["nb_pieces"]
    df_logic_check.loc[mask_toilettes, "anomalie_logique"] = True

    # --- Règle 2 : surface trop petite (< 10 m²) ou démesurée (> 1000 m²)
    mask_surface = (df_logic_check["surface"] < 10) | (df_logic_check["surface"] > 1000)
    df_logic_check.loc[mask_surface, "anomalie_logique"] = True

    # --- Règle 3 : nb_etages = 0 alors que etage > 0 (impossible sans étage)
    mask_etage = (df_logic_check["nb_etages"] == 0) & (df_logic_check["etage"] > 0)
    df_logic_check.loc[mask_etage, "anomalie_logique"] = True

    # --- Règle 4 : logement neuf mais année de construction ancienne (avant 2000)
    mask_neuf = (df_logic_check["logement_neuf"] == True) & (
        df_logic_check["annee_construction"].isin(
            [
                "avant 1948",
                "1948-1974",
                "1975-1977",
                "1978-1982",
                "1983-1988",
                "1989-2000",
            ]
        )
    )
    df_logic_check.loc[mask_neuf, "anomalie_logique"] = True

    # --- Règle 5 : prix_m2_vente très bas ou nul (hors outlier déjà traité)
    mask_prix = df_logic_check["prix_m2_vente"] < 100
    df_logic_check.loc[mask_prix, "anomalie_logique"] = True

    # Résumé : Nombre total de lignes concernées
    nb_anomalies = df_logic_check["anomalie_logique"].sum()
    print(f"{nb_anomalies} lignes présentent au moins une anomalie logique.")
    # Aperçu des premières anomalies détectées (dans les logs ou stdout)
    anomalies_detected = df_logic_check[df_logic_check["anomalie_logique"]].head(10)
    print("🔎 Aperçu des anomalies logiques détectées :")
    print(anomalies_detected.to_string(index=False))  # ou .to_markdown() si tu veux joli
    
    # Sauvegarde dans un fichier CSV (optionnel)
    output_dir = Path("/app/reports/extracts")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = "anomaly_logic_preview.csv"
    csv_path = os.path.join(output_dir, "anomaly_logic_preview.csv")
    anomalies_detected.to_csv(csv_path, index=False)
    
    # Logging MLflow
    mlflow.log_artifact(str(csv_path), artifact_path="extracts/anomaly_logic")
    
        
    ### Détection 	des anomalies de saisie
    # L'idée ici est de borner les valeurs complètement aberrantes avant d'éliminer les valeurs extrêmes
    # Colonnes à vérifier pour erreurs d’échelle
    cols_suspectes = [
        "etage",
        "surface",
        "surface_terrain",
        "nb_pieces",
        "balcon",
        "bain",
        "dpeC",
        "nb_etages",
        "places_parking",
        "nb_toilettes",
        "charges_copro",
        "loyer_m2_median_n6",
        "nb_log_n6",
        "taux_rendement_n6",
        "loyer_m2_median_n7",
        "nb_log_n7",
        "taux_rendement_n7",
        "prix_m2_vente",
    ]

    # Seuils définis de manière métier ou empirique
    seuils_max = {
        "etage": 60,  # étage > 60
        "surface": 2000,  # surface > 2000 m²
        "surface_terrain": 500_000,  # terrain > 50 hectares
        "nb_pieces": 100,  # plus de 100 pièces
        "balcon": 100,  # plus de 100 balcons
        "bain": 20,  # plus de 20 salles de bain
        "dpeC": 10000,  # plus de 10 000 DPE C
        "nb_etages": 60,  # plus de 60 étages
        "places_parking": 50,  # plus de 50 places de parking
        "nb_toilettes": 50,  # plus de 50 toilettes
        "charges_copro": 10_000,  # charges mensuelles > 10k €
        "loyer_m2_median_n6": 500,  # loyer m2 > 500 €
        "nb_log_n6": 15000,  # plus de 15000 logements
        "taux_rendement_n6": 1,  # taux de rendement > 100%
        "loyer_m2_median_n7": 500,  # loyer m2 > 500 €
        "nb_log_n7": 15000,  # plus de 15000 logements
        "taux_rendement_n7": 1,  # taux de rendement > 100%
        "prix_m2_vente": 100_000,  # prix au m² > 100k €
    }

    seuils_min = {
        "etage": -3,  # étage < -3
        "balcon": -1,  # balcon < -1
        "dpeC": -1,  # DPE C < -1
    }

    # Création d’un DataFrame résumé des cas problématiques
    problemes = {}
    mask_valeurs_improbables = pd.Series(
        False, index=df_3.index
    )  # Initialiser le masque

    for col in cols_suspectes:
        # Détection des valeurs au-dessus du seuil maximum
        if col in seuils_max:
            mask_above = df_3[col] > seuils_max[col]
            mask_valeurs_improbables |= mask_above
            n_anormaux_max = mask_above.sum()
        else:
            n_anormaux_max = 0

        # Détection des valeurs en dessous du seuil minimum
        if col in seuils_min:
            mask_below = df_3[col] < seuils_min[col]
            mask_valeurs_improbables |= mask_below
            n_anormaux_min = mask_below.sum()
        else:
            n_anormaux_min = 0

        # Ajouter au rapport si des valeurs aberrantes sont détectées
        if n_anormaux_max > 0 or n_anormaux_min > 0:
            problemes[col] = {
                "nb_anormaux_max": n_anormaux_max,
                "max_valeur": df_3[col].max(),
                "nb_anormaux_min": n_anormaux_min,
                "min_valeur": df_3[col].min(),
            }

        # Nombre total de lignes identifiées comme improbables
    nb_lignes_improbables = mask_valeurs_improbables.sum()
    print(f"{nb_lignes_improbables} lignes contiennent des valeurs improbables.")

    ### Suppression des lignes concernées
    # --- Création d'un masque combiné pour les anomalies logiques et valeurs improbables ---

    # Masque pour les anomalies logiques
    mask_anomalies_logiques = df_logic_check["anomalie_logique"]

    # Combinaison des deux masques
    mask_combined = mask_anomalies_logiques | mask_valeurs_improbables

    # Nombre total de lignes identifiées comme problématiques
    nb_lignes_problemes = mask_combined.sum()
    print(
        f"{nb_lignes_problemes} lignes contiennent des anomalies logiques ou des valeurs improbables."
    )

    # --- Suppression des lignes identifiées ---

    # Suppression des lignes identifiées
    df_3 = df_3[~mask_combined]


    ## Outliers
    ### Outliers Regression
    #### Imputation par mediane par code insee
    # SPLIT ET PARAMÈTRES

    # SPLIT

    train_data, test_data = train_test_split(df_3, test_size=0.2, random_state=42)

    #  Constantes et paramètres ─────────────
    LOWER_PERC = 0.001  # 1er dixième de percentile
    UPPER_PERC = 0.999  # dernier dixième de percentile
    GROUP_COL = "INSEE_COM"  # colonne de regroupement
    TARGET_COL = "prix_m2_vente"  # variable à prédire
    OUTLIER_TAG = -999  # valeur pour différencier les outliers

    # Application des fonctions de nettoyage ----------------
    ## Bounds
    bounds = calculate_bounds(train_data, numeric_cols, LOWER_PERC, UPPER_PERC)

    ## Médianes
    group_medians, global_medians = compute_medians(train_data, bounds, GROUP_COL)

    ## Marquage des outliers
    train_marked = mark_outliers(train_data, bounds)
    test_marked = mark_outliers(test_data, bounds)

    ### masque de conservation
    mask_train_keep = train_marked[f"{TARGET_COL}_outlier_flag"] == 0
    mask_test_keep = test_marked[f"{TARGET_COL}_outlier_flag"] == 0

    ### application du filtre et suppression des outliers de la target
    train_marked = train_marked[mask_train_keep]
    test_marked = test_marked[mask_test_keep]

    # Calcul du nombre d'outliers identifiés par colonne avant leur remplacement
    outlier_counts = {col: train_marked[f"{col}_outlier_flag"].sum() for col in bounds}

    ## Nettoyage (remplacement des -999) avec les médianes du TRAIN
    train_clean = clean_outliers(
        train_marked, bounds, group_medians, global_medians, GROUP_COL
    )
    test_clean = clean_outliers(
        test_marked, bounds, group_medians, global_medians, GROUP_COL
    )

    # suppression des colonnes de marquage
    train_clean.drop(columns=[f"{col}_outlier_flag" for col in bounds], inplace=True)
    test_clean.drop(columns=[f"{col}_outlier_flag" for col in bounds], inplace=True)

    #### Visualisation après traitement outliers

    # Nombre de colonnes par ligne
    cols_per_row = 2

    # Calcul du nombre de lignes nécessaires
    num_cols = len(bounds)
    num_rows = math.ceil(num_cols / cols_per_row)

    # Création des sous-graphiques
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(12, 4 * num_rows))
    axes = axes.flatten()  # Aplatir pour un accès plus simple

    # Boucle pour tracer les boxplots
    print(" Visualisation des boxplots après traitement des outliers...")
    
    # Création des figures et axes
    fig, axes = plt.subplots(nrows=math.ceil(len(bounds) / 2), ncols=2, figsize=(14, 6 * math.ceil(len(bounds) / 2)))
    axes = axes.flatten()
    
    # Tracer les boxplots
    for i, col in enumerate(bounds):
        train_clean.boxplot(column=col, ax=axes[i])
        axes[i].set_title(f"Boxplot de la colonne '{col}' après traitement des outliers")

    # Supprimer les axes inutiles
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    # Sauvegarde dans dossier Docker-friendly
    output_dir = Path("/app/reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"boxplots_outliers_clean_{run_suffix}.png"
    fig_path = output_dir / filename
    fig.savefig(fig_path)

    # Logging MLflow
    log_figure(
        fig,
        filename=filename,
        artifact_path="figures/boxplots"
    )

    # Fermeture propre
    plt.close(fig)

    ### Outliers Serie temporelle
    df_3["date"] = pd.to_datetime(df_3["date"], errors="coerce")
    df_3["Year"] = df_3["date"].dt.year
    df_3["Month"] = df_3["date"].dt.month

    # SPLIT

    df_3["split"] = df_3["Year"].map(
        lambda x: "train_data_ST" if x < 2024 else "test_data_ST"
    )
    train_data_ST = df_3[df_3["split"] == "train_data_ST"]
    test_data_ST = df_3[df_3["split"] == "test_data_ST"]
    df_3 = df_3.drop(columns="split")

    # Application des fonctions de nettoyage ----------------
    ## Bounds
    bounds_ST = calculate_bounds(train_data_ST, numeric_cols, LOWER_PERC, UPPER_PERC)

    ## Médianes
    group_medians_ST, global_medians_ST = compute_medians(
        train_data_ST, bounds_ST, GROUP_COL
    )

    ## Marquage des outliers
    train_marked_ST = mark_outliers(train_data_ST, bounds_ST)
    test_marked_ST = mark_outliers(test_data_ST, bounds_ST)

    ### masque de conservation
    mask_train_keep = train_marked_ST[f"{TARGET_COL}_outlier_flag"] == 0
    mask_test_keep = test_marked_ST[f"{TARGET_COL}_outlier_flag"] == 0

    ### application du filtre et suppression des outliers de la target
    train_marked_ST = train_marked_ST[mask_train_keep]
    test_marked_ST = test_marked_ST[mask_test_keep]

    # Calcul du nombre d'outliers identifiés par colonne avant leur remplacement
    outlier_counts_ST = {
        col: train_marked_ST[f"{col}_outlier_flag"].sum() for col in bounds_ST
    }

    ## Nettoyage (remplacement des -999) avec les médianes du TRAIN
    train_clean_ST = clean_outliers(
        train_marked_ST, bounds_ST, group_medians_ST, global_medians_ST, GROUP_COL
    )
    test_clean_ST = clean_outliers(
        test_marked_ST, bounds_ST, group_medians_ST, global_medians_ST, GROUP_COL
    )

    # suppression des colonnes de marquage

    test_clean_ST.drop(
        columns=[f"{col}_outlier_flag" for col in bounds_ST], inplace=True
    )

    # Vérification des outliers
    print("Valeurs extrêmes détectées et remplacées :")
    for col, count in outlier_counts_ST.items():
        print(f"Colonne '{col}: {count} outliers détectés et remplacés.")
    # Vérification des valeurs extrêmes restantes
    print("Valeurs extrêmes restantes :")
    for col in bounds_ST:
        print(
            f"Colonne '{col}': {train_clean_ST[col].min()} à {train_clean_ST[col].max()}"
        )

    ## Visualisation de la distribution de la target
    fig_distribution, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=train_clean["prix_m2_vente"], bins=150, kde=True, ax=ax)
    ax.set_title("Distribution Prix m2_vente")
    ax.set_xlim(0, 20000)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    fig_distribution.tight_layout()

    log_figure(
        fig_distribution,
        filename=f"distribution_prix_m2_{run_suffix}.png",
        artifact_path="figures/distributions",
    )
    plt.close(fig_distribution)

    # === Log des stats de distribution de la target ===
    prix_stats = train_clean["prix_m2_vente"].describe()
    for stat in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
        mlflow.log_metric(f"prix_m2_vente_{stat}", prix_stats[stat])

    df_sales_short_ST = pd.concat([train_clean_ST, test_clean_ST], axis=0).reset_index(
        drop=True
    )

    # === Tracking MLflow ===
    mlflow.log_param("input_path", input_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    pl.from_pandas(train_clean).write_csv(
        output_path.replace(".csv", "_train.csv"), separator=";"
    )
    pl.from_pandas(test_clean).write_csv(
        output_path.replace(".csv", "_test.csv"), separator=";"
    )
    pl.from_pandas(df_sales_short_ST).write_csv(
        output_path.replace(".csv", "_series.csv"), separator=";"
    )

    mlflow.log_artifact(output_path.replace(".csv", "_train.csv"))
    mlflow.log_artifact(output_path.replace(".csv", "_test.csv"))
    mlflow.log_artifact(output_path.replace(".csv", "_series.csv"))

    print(f" Données sauvegardées dans : {output_path}")


if __name__ == "__main__":
    main()


    # === END PIPELINE ===
    print("✅ Pipeline preprocessing terminée avec succès")


