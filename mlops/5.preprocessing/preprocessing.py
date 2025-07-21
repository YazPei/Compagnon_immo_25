import re
import os
import click
import pandas as pd
import polars as pl
import mlflow
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from datetime import datetime
import math

run_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")


# === Visualisation du dataset ===


# === Fonctions de nettoyage et utilitaires ===
def annee_const(x):
    bins = [
        float("-inf"),
        1948,
        1974,
        1977,
        1982,
        1988,
        2000,
        2005,
        2012,
        2021,
        float("inf"),
    ]
    labels = [
        "avant 1948",
        "1948-1974",
        "1975-1977",
        "1978-1982",
        "1983-1988",
        "1989-2000",
        "2001-2005",
        "2006-2012",
        "2013-2021",
        "après 2021",
    ]
    if "annee_construction" not in x.columns:
        raise ValueError("La colonne 'annee_construction' est manquante.")
    x["annee_construction_cat"] = pd.cut(
        x["annee_construction"], bins=bins, labels=labels, right=False
    )
    return x


# 1. Fonctions utilitaires


def clean_classe(x):
    """
    Nettoie et standardise les classes DPE/GES :
    - Vide, "Blank" ou "0" → np.nan
    - Conserve A→G, NS, VI
    - Sinon, extrait un code valide en début de chaîne via regex
    """
    # Cas manquant
    if pd.isna(x) or str(x).strip() in ["", "Blank", "0"]:
        return np.nan
    # Mise en majuscule et suppression des espaces
    s = str(x).strip().upper()
    # Acceptation stricte des codes connus
    if s in ["A", "B", "C", "D", "E", "F", "G", "NS", "VI"]:
        return s
    # Tentative de capture d'un code valide en début de chaîne
    m = re.match(r"^(NS|VI|[A-G])", s, re.IGNORECASE)
    return m.group(1).upper() if m else np.nan


def extract_principal(x):
    """
    Extrait la première source énergétique listée.
    Séparateurs gérés : ',', ';', '/', 'et'
    """
    # Cas manquant ou chaîne vide
    if pd.isna(x) or not str(x).strip():
        return np.nan
    # Split sur les séparateurs, on ne garde que la première partie
    parts = re.split(r"\s*(?:,|;|/|et)\s*", str(x).strip(), maxsplit=1)
    return parts[0] if parts else np.nan


# Configuration pour la normalisation de l'exposition
PATTERN_EXPO = r"(?i)\b(?:Nord(?:-Est|-Ouest)?|Sud(?:-Est|-Ouest)?|Est|Ouest|N|S|E|O)\b"
ORDRE_EXPO = ["Nord", "Est", "Sud", "Ouest"]
NORM_DIR = {
    "N": "Nord",
    "S": "Sud",
    "E": "Est",
    "O": "Ouest",
    "NORD": "Nord",
    "SUD": "Sud",
    "EST": "Est",
    "OUEST": "Ouest",
    "NORD-EST": "Nord-Est",
    "NORD-OUEST": "Nord-Ouest",
    "SUD-EST": "Sud-Est",
    "SUD-OUEST": "Sud-Ouest",
}


def clean_exposition(x):
    """
    Nettoie et standardise la colonne exposition :
    - Détecte les mots-clés de multi-exposition → "Multi-exposition"
    - Extrait les points cardinaux via regex
    - Traduit et normalise selon NORM_DIR
    - Trie et déduplique selon ORDRE_EXPO
    """
    # Cas manquant ou valeur vide
    if pd.isna(x) or str(x).strip() in ["", "0"]:
        return np.nan
    s = str(x).strip()
    low = s.lower()
    # 1) Multi-exposition via mots-clés
    if any(
        kw in low
        for kw in ["traversant", "multi", "toutes", "double expo", "triple", "360"]
    ):
        return "Multi-exposition"
    # 2) Extraction des directions
    matches = re.findall(PATTERN_EXPO, s, flags=re.IGNORECASE)
    dirs = [
        NORM_DIR[m.upper().replace(" ", "-")]
        for m in matches
        if m.upper().replace(" ", "-") in NORM_DIR
    ]
    # 3) Tri et déduplication
    uniq = sorted(set(dirs), key=lambda d: ORDRE_EXPO.index(d.split("-")[0]))
    return "-".join(uniq) if uniq else np.nan


######
# clean des variables numeriques
# Future colonne de regroupement pour les outliers
GROUP_COL = "INSEE_COM"  # colonne de regroupement


# Identification des colonnes numériques et exclusion des colonnes de coordonnées géographiques
def get_numeric_cols(x, group_col):
    """
    Retourne les colonnes numériques en excluant la future colonne de regroupement - qui est l'équivalent d'un code postal- pour les outliers à venir plus tard
    et les colonnes de coordonnées géographiques.
    """
    excluded_cols = ["mapCoordonneesLatitude", "mapCoordonneesLongitude"]
    return [
        col
        for col in x.select_dtypes(include="number").columns
        if col != group_col and col not in excluded_cols
    ]


def calculate_bounds(x, numeric_cols, lower_perc, upper_perc):
    """
    Calcule les bornes inférieures et supérieures pour chaque colonne numérique.
    """
    return {
        col: (x[col].quantile(lower_perc), x[col].quantile(upper_perc))
        for col in numeric_cols
    }


def compute_medians(train_data, bounds, group_col):
    """
    Calcule les médianes par groupe (group_col) et globales
    UNIQUEMENT à partir du train).
    """
    group_meds = {col: train_data.groupby(group_col)[col].median() for col in bounds}
    global_meds = train_data[list(bounds)].median()
    return group_meds, global_meds


OUTLIER_TAG = -999


def mark_outliers(x, bounds, outlier_tag=OUTLIER_TAG):
    """
    Pour chaque colonne dans bounds :
      - crée col+'_outlier_flag' = 1 si en dehors des bornes
      - remplace la valeur outlier par outlier_tag
    """
    for col, (low, high) in bounds.items():
        mask = (x[col] < low) | (x[col] > high)
        x[f"{col}_outlier_flag"] = mask.astype(int)
        x.loc[mask, col] = outlier_tag
    return x


def clean_outliers(
    df, bounds, group_meds, global_meds, group_col, outlier_tag=OUTLIER_TAG
):
    """
    Remplace dans df (train ou test) les tags outlier_tag par :
      - la médiane du groupe (via group_meds)
      - sinon la médiane globale (global_meds)
    """
    for col in bounds:
        mask = df[col] == outlier_tag
        # remplace uniquement les outliers taggés
        df.loc[mask, col] = (
            df.loc[mask, group_col]
            .map(group_meds[col])
            .fillna(global_meds[col])
            .astype(df[col].dtype)  # Ajout d'un cast pour éviter les problèmes de type
        )
    return df


def log_figure(fig, filename, artifact_path="figures"):
    """Sauvegarde une figure matplotlib et la loggue dans MLflow."""
    os.makedirs("outputs", exist_ok=True)
    full_path = os.path.join("outputs", filename)
    fig.savefig(full_path)
    mlflow.log_artifact(full_path, artifact_path=artifact_path)
    mlflow.set_tag("figure_logged", filename)


#############


# === Pipeline principale CLI ===


@click.command()
@click.option(
    "--input-path", type=click.Path(exists=True), prompt=" Fichier d’entrée fusionné"
)
@click.option("--output-path", type=click.Path(), prompt=" Fichier de sortie nettoyé")
def main(input_path, output_path):

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment("Preprocessing Données Immo")

    with mlflow.start_run(run_name="preprocessing_pipeline"):
        mlflow.set_tag("phase", "preprocessing")
        mlflow.set_tag("version", "v1.0")

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

    display(missing_value_percentage_sales)
    ### visualisation des doublons

    plt.figure(figsize=(10, 14))

    sns.barplot(
        y=missing_value_percentage_sales.column_name,
        x=missing_value_percentage_sales.percent_missing,
        hue=missing_value_percentage_sales.column_name,
        order=missing_value_percentage_sales.column_name,
    )

    # Add a vertical line at x=50 (adjust as needed)
    plt.axvline(x=75, color="red", linestyle="--", label="Threshold (75%)")

    plt.title("Répartition des valeurs manquantes dans le dataset", fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=9)
    plt.ylabel("Features")
    plt.legend()

    plt.show()
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
    plt.show()

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
    plt.show()
    log_figure(
        fig_o,
        filename=f"boxplots_outliers_{run_suffix}.png",
        artifact_path="figures/boxplots",
    )

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

    # Aperçu des premières anomalies détectées
    display(df_logic_check[df_logic_check["anomalie_logique"]].head(10))

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

    # Résumé : Nombre de lignes supprimées
    print(
        f"{nb_lignes_problemes} lignes ont été supprimées en raison d'anomalies logiques ou de valeurs improbables."
    )

    # Aperçu du DataFrame nettoyé
    display(df_3.head())

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
    print("Visualisation des boxplots après traitement des outliers :")
    for i, col in enumerate(bounds):
        train_clean.boxplot(column=col, ax=axes[i])
        axes[i].set_title(
            f"Boxplot de la colonne '{col}' après traitement des outliers"
        )

    # Supprimer les axes inutilisés si le nombre de colonnes est impair
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    log_figure(
        fig,
        filename=f"boxplots_outliers_clean_{run_suffix}.png",
        artifact_path="figures/boxplots",
    )
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
