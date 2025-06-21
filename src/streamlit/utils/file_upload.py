import pandas as pd
import os
import streamlit as st
import re



@st.cache_data
def try_read_csv(path_or_file, sep=";", chunksize=100000):
    """
    Tente de lire un fichier CSV avec différents encodages.
    """
    encodings_to_try = ['ISO-8859-1', 'latin1', 'utf-8']
    for encoding in encodings_to_try:
        try:
            print(f"⏳ Tentative d'ouverture avec encodage : {encoding}")
            chunks = pd.read_csv(
                path_or_file,
                sep=sep,
                chunksize=chunksize,
                index_col=None,
                on_bad_lines='skip',
                low_memory=False,
                encoding=encoding
            )
            df = pd.concat(chunk for chunk in chunks)
            print(f"✅ Fichier lu avec succès avec encodage : {encoding}")
            return df
        except UnicodeDecodeError as e:
            print(f"⚠️ Échec avec encodage {encoding} : {e}")
        except Exception as e:
            print(f"❌ Autre erreur : {e}")
    raise ValueError("Aucun encodage valide n'a permis d'ouvrir le fichier.")

@st.cache_data
def load_and_prepare_data(uploaded_file=None, local_file_path=None):
    """
    Charge un fichier CSV depuis un fichier uploadé ou un chemin local,
    le nomme `df_sales_clean`, et effectue les premières transformations nécessaires.

    Parameters:
        uploaded_file: Fichier uploadé via Streamlit.
        local_file_path: Chemin local vers le fichier CSV.

    Returns:
        pd.DataFrame: Le DataFrame chargé et transformé.
    """
    if uploaded_file:
        # Charger depuis un fichier uploadé
        df_sales_clean = try_read_csv(uploaded_file)
    elif local_file_path and os.path.exists(local_file_path):
        # Charger depuis un fichier local
        df_sales_clean = try_read_csv(local_file_path)
    else:
        raise ValueError("Aucun fichier valide n'a été fourni.")

    # Transformation de la colonne 'date' en datetime si elle existe
    if 'date' in df_sales_clean.columns:
        df_sales_clean['date'] = pd.to_datetime(df_sales_clean['date'], errors='coerce')

    return df_sales_clean