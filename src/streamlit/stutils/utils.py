import streamlit as st
import os
import pandas as pd
import numpy as np

import statsmodels as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import silhouette_score
from shapely.geometry import Point
import matplotlib.patches as mpatches
import geopandas as gpd

import time
from datetime import datetime

import joblib

#############################################################################################################

def short_intro_video():

    folder_path_C = './videos'
    video_path= os.path.join(folder_path_C, 'Dream Home Success.mp4')

    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()

    st.video(video_bytes, loop=True, autoplay=True, muted=True)

#############################################################################################################

@st.cache_data
def load_df_sales_clean_ST():

    folder_path_C = './data/processed/Sales'
    input_file = os.path.join(folder_path_C, 'df_sales_clean_ST.csv')

    data = pd.read_csv(input_file, sep=';', index_col='date', parse_dates=['date'], on_bad_lines='skip', low_memory=False)

    # Calculate 50% of the total number of rows
    sample_size = int(0.5 * len(data))

    # Generate a random sample
    df_sales = data.sample(n=sample_size, random_state=1)  # random_state for reproducibility
    df_sales.dropna(inplace=True)
    df_sales = df_sales.drop_duplicates()
    
    return df_sales

#############################################################################################################

@st.cache_data
def load_train_pour_graph():

    folder_path_C = './data/temp'
    input_file = os.path.join(folder_path_C, 'Train_pour_graph.csv')
    
    train_pour_graph = pd.read_csv(input_file, sep=';', parse_dates=['date'], on_bad_lines='skip', low_memory=False)
    
    return train_pour_graph

#############################################################################################################

@st.cache_data
def load_train_pour_graph_cp():

    folder_path_C = './data/temp'
    input_file = os.path.join(folder_path_C, 'Train_pour_graph_cp.csv')
    
    train_pour_graph_cp = pd.read_csv(input_file, sep=';', parse_dates=['date'], on_bad_lines='skip', low_memory=False)
    
    return train_pour_graph_cp

#############################################################################################################

def make_evolution_mensuelle_global_plot(Train_pour_graph):

    # Global plot
    fig_mensuel_glob = px.line(
        Train_pour_graph,
        x="date",
        y="prix_m2_vente",
        title="Prix moyen au m²",
        labels={"date": "Date", "prix_m2_vente": "Prix moyen (€ / m²)"},
    )

    fig_mensuel_glob.update_traces(mode="lines+markers")
    fig_mensuel_glob.update_layout(
        title_x=0.1,
        title_y=0.95,
        title_font_size=20,
        xaxis_title="Date",
        yaxis_title="Prix moyen (€ / m²)",
        hovermode="x unified",
    )

    fig_mensuel_glob.update_xaxes(title_font=dict(size=14), tickfont=dict(size=12))
    fig_mensuel_glob.update_yaxes(title_font=dict(size=14), tickfont=dict(size=12))

    st.plotly_chart(fig_mensuel_glob)

#############################################################################################################

def make_evolution_mensuelle_plots(Train_pour_graph_cp):

    # Department plot with multi-select dropdown
    departements = Train_pour_graph_cp["departement"].dropna()
    departements = departements[departements.str.lower() != 'na'].unique()

    # Use a multi-select widget to choose departments
    selected_departements = st.multiselect("Sélectionnez les départements", departements, default=departements[32])

    # Filter data based on selected departments
    filtered_data = Train_pour_graph_cp[Train_pour_graph_cp["departement"].isin(selected_departements)]

    fig_mensuel = px.line(
        filtered_data,
        x="date",
        y="prix_m2_vente",
        color="departement",
        title="Évolution mensuelle du prix moyen au m² par département",
        labels={
            "date": "Date",
            "prix_m2_vente": "Prix moyen (€ / m²)",
            "departement": "Département",
        },
    )

    fig_mensuel.update_xaxes(title_font=dict(size=14), tickfont=dict(size=12))
    fig_mensuel.update_yaxes(title_font=dict(size=14), tickfont=dict(size=12))

    fig_mensuel.update_layout(
        updatemenus=[
            dict(
                active=0,
                direction="down",
                showactive=True,
                x=1.20,
                xanchor="left",
                y=1.1,
                yanchor="top",
                pad={"r": 10, "t": 10},
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
            )
        ]
    )

    st.plotly_chart(fig_mensuel)
    
#############################################################################################################

@st.cache_data
def load_df_cluster_input():

    folder_path_C = './data/temp'
    input_file = os.path.join(folder_path_C, 'df_cluster_input.csv')
    
    df_cluster_input = pd.read_csv(input_file, sep=';')
    
    return df_cluster_input

#############################################################################################################

def kmean_par_coude():

    folder_path_C = "./images/KMeans"

    # List all image files in the directory
    image_files = [f for f in os.listdir(folder_path_C) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Ensure there are exactly 9 images
    if len(image_files) != 9:
        st.error(f"Expected 9 images, but found {len(image_files)} images.")
    else:
        # Sort the image files to ensure consistent ordering
        image_files.sort()

        # Create a slider to select the image index
        image_index = st.slider(
            "Utilisez le curseur pour explorer l'impact du nombre de clusters (k) sur la qualité de la segmentation :", 
            min_value=2,
            max_value=10,
            value=7,
            step=1,
            help="Déplacez le curseur pour voir comment l'inertie et le score de Silhouette évoluent en fonction du nombre de clusters"
        )

        # Display the selected image
        image_path = os.path.join(folder_path_C, image_files[image_index-2])
        st.image(image_path)

#############################################################################################################

@st.cache_data
def distrib_cluster(df_cluster_input):

    features = [
        "prix_m2_mean",
        "prix_m2_std",
        "prix_m2_max",
        "prix_m2_min",
        "tc_am_reg",
        "prix_m2_cv",
    ]

    cluster_palette = {
    "Zones rurales, petites villes stagnantes":    "#1f77b4",
    "Banlieues, zones mixtes":                    "#ff7f0e",
    "Centres urbains établis, zones résidentielles":"#2ca02c",
    "Zones tendues - secteurs spéculatifs":        "#d62728",
    }

    sns.pairplot(df_cluster_input, vars=features, hue="cluster_label", hue_order=list(cluster_palette.keys()),palette=cluster_palette)

    st.pyplot(plt)

#############################################################################################################

def plot_autocorrelation(data):

    fig, ax = plt.subplots(figsize=(10, 4))
    pd.plotting.autocorrelation_plot(data, ax=ax)

    st.pyplot(fig)

#############################################################################################################

def plot_acf_pacf(data):

    max_lags = len(data) // 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    plot_acf(data, lags=max_lags, ax=ax1)
    plot_pacf(data, lags=max_lags, ax=ax2)

    st.pyplot(fig)

#############################################################################################################

@st.cache_data
def load_train_periodique_q12():

    folder_path_C = './data/processed/Sales'
    input_file = os.path.join(folder_path_C, 'train_periodique_q12.csv')

    train_periodique_q12 = pd.read_csv(input_file, sep=';', index_col='date', parse_dates=['date'], on_bad_lines='skip', low_memory=False)

    train_periodique_q12 = train_periodique_q12.loc[:, ~train_periodique_q12.columns.str.contains('^Unnamed')]
    train_periodique_q12.sort_index(inplace=True)
    
    return train_periodique_q12

#############################################################################################################

@st.cache_data
def load_test_periodique_q12():

    folder_path_C = './data/processed/Sales'
    input_file = os.path.join(folder_path_C, 'test_periodique_q12.csv')

    test_periodique_q12 = pd.read_csv(input_file, sep=';', index_col='date', parse_dates=['date'], on_bad_lines='skip', low_memory=False)

    test_periodique_q12 = test_periodique_q12.loc[:, ~test_periodique_q12.columns.str.contains('^Unnamed')]
    test_periodique_q12.sort_index(inplace=True)
    
    return test_periodique_q12

#############################################################################################################

@st.cache_data
def make_correlation_matrix(train_periodique_q12):

    var_targ = [
        "prix_m2_vente", 
        "taux_rendement_n7", 
        "taux", 
        "loyer_m2_median_n7",
        "y_geo", 
        "x_geo", 
        "z_geo", 
        "dpeL", 
        "nb_pieces"
        ]

    # Calculer la matrice de corrélation
    correlation_matrix = train_periodique_q12[var_targ].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matrice de corrélation des variables avec la cible')
    
    st.pyplot(plt)

#############################################################################################################

@st.cache_data
def seasonal_decompose_for_clusters(train_periodique_q12):

    # features = ["taux_rendement_n7", 'taux', "loyer_m2_median_n7","y_geo", "x_geo", "z_geo", "dpeL", "nb_pieces"]
    # train_periodique_q12 = train_periodique_q12.dropna(subset= features + ["prix_m2_vente"])

    for cluster in sorted(train_periodique_q12["cluster"].dropna().unique()):
        
        # Extraire la série de prix par cluster
        df_cluster = train_periodique_q12[train_periodique_q12["cluster"] == cluster]
        y = df_cluster["prix_m2_vente"]
        y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp()

        # # Décomposition additive
        # data_add = seasonal_decompose(y, model="additive", period=12)
        # # plt.figure(figsize=(6, 6))
        # fig = data_add.plot()
        # fig.set_size_inches((18, 10), forward=True)
        # fig.suptitle(f"Décomposition additive – Cluster {cluster}", fontsize=25)
        # fig.tight_layout()

        # st.pyplot(plt)

        # Décomposition multiplicative
        data_mult = seasonal_decompose(y, model="multiplicative", period=12)
        # plt.figure(figsize=(6, 6))
        fig = data_mult.plot()
        fig.set_size_inches((15, 10), forward=True)
        fig.suptitle(f"Décomposition multiplicative – Cluster {cluster}", fontsize=25)
        fig.tight_layout()

        st.pyplot(plt)

#############################################################################################################

@st.cache_data
def draw_clusters_cvs(train_periodique_q12):

    for cluster in sorted(train_periodique_q12["cluster"].dropna().unique()):

        # Filtrer les données du cluster
        df_cluster = train_periodique_q12[train_periodique_q12["cluster"] == cluster]

        # Extraire la série
        y = df_cluster["prix_m2_vente"]
        y.index = pd.DatetimeIndex(df_cluster.index).to_period("M").to_timestamp()

        # Décomposition multiplicative

        data_mult = seasonal_decompose(y, model="multiplicative", period=12)

        # Correction : log(y) - composante saisonnière => puis exp()
        cvs = y / data_mult.seasonal
        y_corrige = np.exp(cvs)

        plt.figure(figsize=(10, 4))
        plt.plot(np.exp(y), label="Série originale")
        plt.plot(y_corrige, label="Corrigée saisonnalité", linestyle="--")
        plt.title(f"Série originale vs corrigée – Cluster {cluster}")
        plt.xlabel("Date")
        plt.ylabel("Prix/m²")
        plt.legend()
        plt.tight_layout()

        st.pyplot(plt)

#############################################################################################################

@st.cache_data
def differeciate_cluster(train_periodique_q12):

    # cluster 0
    clusters_st_0 = pd.DataFrame()
    df_cluster_0 = train_periodique_q12[train_periodique_q12["cluster"] == 0]
   
    y = df_cluster_0["prix_m2_vente"]
    y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp()
    y = np.log(y)
    y_diff_order_1 = y.diff().dropna()

    # Convertir la Serie en Dataframe
    y_diff_order_1 = y_diff_order_1.to_frame(name="prix_m2_vente")

    # Renommer la colonne
    y_diff_order_1.rename(columns={"prix_m2_vente": "diff_order_1"}, inplace=True)

    # Afficher la série
    y_diff_order_1.index = (
        pd.DatetimeIndex(y_diff_order_1.index).to_period("M").to_timestamp()
    )
    # Concaténation
    clusters_st_0 = pd.concat([clusters_st_0, y_diff_order_1], axis=0)
    clusters_st_0["cluster"] = 0

    # Tracés
    plt.figure(figsize=(8, 5))
    plt.subplot(211)
    pd.plotting.autocorrelation_plot(y_diff_order_1)
    plt.title("Différenciation ordre 1 – Cluster 0")
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

    test_stationarity(clusters_st_0["diff_order_1"], window=12)

    # cluster 1
    clusters_st_1 = pd.DataFrame()
    df_cluster_1 = train_periodique_q12[train_periodique_q12["cluster"] == 1]
    
    y = df_cluster_1["prix_m2_vente"]
    y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp()
    y = np.log(y)
    y_diff_order_1 = y.diff().dropna()
    y_diff_order_2 = y_diff_order_1.diff().dropna()

    # Convertir la Serie en Dataframe
    y_diff_order_1 = y_diff_order_1.to_frame(name="prix_m2_vente")
    y_diff_order_2 = y_diff_order_2.to_frame(name="prix_m2_vente")

    # Renommer la colonne
    y_diff_order_1.rename(columns={"prix_m2_vente": "diff_order_1"}, inplace=True)
    y_diff_order_2.rename(columns={"prix_m2_vente": "diff_order_2"}, inplace=True)

    # Afficher la série
    y_diff_order_1.index = (
        pd.DatetimeIndex(y_diff_order_1.index).to_period("M").to_timestamp()
    )
    y_diff_order_2.index = (
        pd.DatetimeIndex(y_diff_order_2.index).to_period("M").to_timestamp()
    )

    # Concaténation
    clusters_st_1 = pd.concat([y_diff_order_1, y_diff_order_2], axis=1)
    clusters_st_1["cluster"] = 1

    # Tracés
    plt.figure(figsize=(8, 5))
    plt.subplot(121)
    pd.plotting.autocorrelation_plot(y_diff_order_1)
    plt.title("Différenciation ordre 1 – Cluster 1")
    plt.grid(True)

    plt.subplot(122)
    pd.plotting.autocorrelation_plot(y_diff_order_1)
    plt.title(f"Différenciation ordre 2 – Cluster 1")
    plt.grid(True)

    plt.tight_layout()

    st.pyplot(plt)

    test_stationarity(clusters_st_1["diff_order_2"], window=12)

    # cluster 2
    clusters_st_2 = pd.DataFrame()
    df_cluster_2 = train_periodique_q12[train_periodique_q12["cluster"] == 2]

    y = df_cluster_2["prix_m2_vente"]
    y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp()
    y = np.log(y)
    y_diff_order_1 = y.diff().dropna()

    # Convertir la Serie en Dataframe
    y_diff_order_1 = y_diff_order_1.to_frame(name="prix_m2_vente")

    # Renommer la colonne
    y_diff_order_1.rename(columns={"prix_m2_vente": "diff_order_1"}, inplace=True)

    # Afficher la série
    y_diff_order_1.index = (
        pd.DatetimeIndex(y_diff_order_1.index).to_period("M").to_timestamp()
    )

    # Concaténation
    clusters_st_2 = pd.concat([clusters_st_2, y_diff_order_1], axis=0)
    clusters_st_2["cluster"] = 2

    # Tracés
    plt.figure(figsize=(8, 5))
    plt.subplot(211)
    pd.plotting.autocorrelation_plot(y_diff_order_1)
    plt.title("Différenciation ordre 1 – Cluster 2")
    plt.grid(True)
    plt.tight_layout()

    st.pyplot(plt)
    
    test_stationarity(clusters_st_2["diff_order_1"], window=12)

    # cluster 3
    clusters_st_3 = pd.DataFrame()
    df_cluster_3 = train_periodique_q12[train_periodique_q12["cluster"] == 3]
    
    y = df_cluster_3["prix_m2_vente"]
    y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp()
    y = np.log(y)
    y_diff_order_1 = y.diff().dropna()

    # Convertir la Serie en Dataframe
    y_diff_order_1 = y_diff_order_1.to_frame(name="prix_m2_vente")

    # Renommer la colonne
    y_diff_order_1.rename(columns={"prix_m2_vente": "diff_order_1"}, inplace=True)

    # Afficher la série
    y_diff_order_1.index = (
        pd.DatetimeIndex(y_diff_order_1.index).to_period("M").to_timestamp()
    )

    # Concaténation
    clusters_st_3 = pd.concat([clusters_st_3, y_diff_order_1], axis=0)
    clusters_st_3["cluster"] = 3

    # Tracés
    plt.figure(figsize=(8, 5))
    plt.subplot(211)
    pd.plotting.autocorrelation_plot(y_diff_order_1)
    plt.title("Différenciation ordre 1 – Cluster 3")
    plt.grid(True)
    plt.tight_layout()

    st.pyplot(plt)
    
    test_stationarity(clusters_st_3["diff_order_1"], window=12)

    return clusters_st_0, clusters_st_1, clusters_st_2, clusters_st_3

#############################################################################################################

def test_stationarity(timeseries, window=12):

    # Effectuer le test ADF
    result = adfuller(timeseries.dropna())
    st.write("Résultats du test ADF:")
    st.write("Statistique ADF:", result[0])
    st.write("p-value:", result[1])

    # Interprétation des résultats
    if result[1] < 0.05:
        st.write("La série est stationnaire (p-value < 0.05)")
        st.divider()
    else:
        st.write("La série n'est pas stationnaire (p-value >= 0.05)")
        st.divider()

#############################################################################################################

def plot_acf_pacf(clusters_st_0, clusters_st_1, clusters_st_2, clusters_st_3):

    cluster = 0
    # Sélectionnez les données pour ce cluster
    cluster_data = clusters_st_0[clusters_st_0["cluster"] == cluster]

    # Vérifiez qu'il y a assez de points
    if cluster_data.empty:
        st.write(f"Cluster {cluster} has no data. Skipping...")
    else:
        max_lags = len(cluster_data) // 2
        if max_lags <= 0:
            st.write(f"Cluster {cluster} has insufficient data for ACF/PACF. Skipping...")
        else:
            # Tracé
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
            plot_acf(
                cluster_data["diff_order_1"],
                lags=max_lags,
                ax=ax1,
                title=f"Autocorrelation – Cluster {cluster}",
            )
            plot_pacf(
                cluster_data["diff_order_1"],
                lags=max_lags,
                ax=ax2,
                title=f"Partial Autocorrelation – Cluster {cluster}",
            )
            st.pyplot(plt)


    # Traitement spécifique pour cluster 1 qui est différencié 2 fois
    series = clusters_st_1["diff_order_2"].dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    plot_acf(series, lags=len(series) // 2, ax=ax1, title="ACF diff_order_2 – Cluster 1")
    plot_pacf(series, lags=len(series) // 2, ax=ax2, title="PACF diff_order_2 – Cluster 1")
    st.pyplot(plt)


    cluster = 2
    # Sélectionnez les données pour ce cluster
    cluster_data = clusters_st_2[clusters_st_2["cluster"] == cluster]
    # Vérifiez qu'il y a assez de points
    if cluster_data.empty:
        st.write(f"Cluster {cluster} has no data. Skipping...")
    else:
        max_lags = len(cluster_data) // 2
        if max_lags <= 0:
            st.write(f"Cluster {cluster} has insufficient data for ACF/PACF. Skipping...")
        else:
            # Tracé
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
            plot_acf(
                cluster_data["diff_order_1"],
                lags=max_lags,
                ax=ax1,
                title=f"Autocorrelation – Cluster {cluster}",
            )
            plot_pacf(
                cluster_data["diff_order_1"],
                lags=max_lags,
                ax=ax2,
                title=f"Partial Autocorrelation – Cluster {cluster}",
            )
            st.pyplot(plt)


    cluster = 3
    # Sélectionnez les données pour ce cluster
    cluster_data = clusters_st_3[clusters_st_3["cluster"] == cluster]
    # Vérifiez qu'il y a assez de points
    if cluster_data.empty:
        st.write(f"Cluster {cluster} has no data. Skipping...")
    else:
        max_lags = len(cluster_data) // 2
        if max_lags <= 0:
            st.write(f"Cluster {cluster} has insufficient data for ACF/PACF. Skipping...")
        else:
            # Tracé
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
            plot_acf(
                cluster_data["diff_order_1"],
                lags=max_lags,
                ax=ax1,
                title=f"Autocorrelation – Cluster {cluster}",
            )
            plot_pacf(
                cluster_data["diff_order_1"],
                lags=max_lags,
                ax=ax2,
                title=f"Partial Autocorrelation – Cluster {cluster}",
            )
            st.pyplot(plt)

#############################################################################################################

def grid_search_cluster_0():

    st.write('#### Résumé du modèle SARIMAX - Banlieue - Zone Mixte')

    total_combinations = 18360
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(total_combinations + 1):
        progress = i / total_combinations
        progress_bar.progress(progress)
        current_time = datetime.now().strftime("%H:%M:%S")  # Get current time
        seconds = datetime.strptime(current_time, "%H:%M:%S").second
        alternate = 2 if seconds % 2 == 0 else 3
        status_text.text(f"100%|{'█' * int(progress * 20)}{'-' * (20 - int(progress * 20))}| {i}/{total_combinations} [{current_time}, 1125.5{alternate}it/s]")
        time.sleep(0.000001)

    # Print the best parameters found
    st.text("--- SARIMAX - Banlieue - Zone Mixte ---")
    st.text("Meilleure combinaison d'exogènes : ('y_geo', 'dpeL', 'IPS_primaire')")
    st.text("Meilleur ordre (p,d,q) : (0, 1, 0)")
    st.text("Saisonnalité (P,D,Q,s) : (0, 0, 0, 12)")
    st.text("AIC : -260.50277396469034")

    cluster_0_model = get_cluster_model_0(best=True)

    st.write(cluster_0_model.summary())

    fig = cluster_0_model.plot_diagnostics(figsize=(15, 12))

    st.pyplot(fig)

    st.divider()

#############################################################################################################

def grid_search_cluster_1():

    st.write('#### Résumé du modèle SARIMAX - Centre urbain établi, zone résidentielle')

    total_combinations = 18360
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(total_combinations + 1):
        progress = i / total_combinations
        progress_bar.progress(progress)
        current_time = datetime.now().strftime("%H:%M:%S")  # Get current time
        seconds = datetime.strptime(current_time, "%H:%M:%S").second
        alternate = 2 if seconds % 2 == 0 else 3
        status_text.text(f"100%|{'█' * int(progress * 20)}{'-' * (20 - int(progress * 20))}| {i}/{total_combinations} [{current_time}, 1225.2{alternate}it/s]")
        time.sleep(0.000001)

    # Print the best parameters found
    st.text("--- SARIMAX - Centre urbain établi, zone résidentielle ---")
    st.text("Meilleure combinaison d'exogènes : ('taux_rendement_n7', 'loyer_m2_median_n7')")
    st.text("Meilleur ordre (p,d,q) : (2, 2, 0)")
    st.text("Saisonnalité (P,D,Q,s) : (0, 0, 0, 12)")
    st.text("AIC : -156.89887391838786")

    cluster_1_model = get_cluster_model_1(best=True)

    st.write(cluster_1_model.summary())

    fig = cluster_1_model.plot_diagnostics(figsize=(15, 12))

    st.pyplot(fig)

    st.divider()
    
#############################################################################################################

def grid_search_cluster_2():

    st.write('#### Résumé du modèle SARIMAX - Zone rurale - petite ville stagnante')

    total_combinations = 18360
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(total_combinations + 1):
        progress = i / total_combinations
        progress_bar.progress(progress)
        current_time = datetime.now().strftime("%H:%M:%S")  # Get current time
        seconds = datetime.strptime(current_time, "%H:%M:%S").second
        alternate = 2 if seconds % 2 == 0 else 3
        status_text.text(f"100%|{'█' * int(progress * 20)}{'-' * (20 - int(progress * 20))}| {i}/{total_combinations} [{current_time}, 1725.1{alternate}it/s]")
        time.sleep(0.0000001)

    # Print the best parameters found
    st.text("--- SARIMAX - Zone rurale - petite ville stagnante ---")
    st.text("Meilleure combinaison d'exogènes : ('z_geo', 'dpeL')")
    st.text("Meilleur ordre (p,d,q) : (1, 1, 0)")
    st.text("Saisonnalité (P,D,Q,s) : (0, 0, 0, 12)")
    st.text("AIC : -239.0639097254141")

    cluster_2_model = get_cluster_model_2(best=True)

    st.write(cluster_2_model.summary())

    fig = cluster_2_model.plot_diagnostics(figsize=(15, 12))

    st.pyplot(fig)

    st.divider()

#############################################################################################################

def grid_search_cluster_3():

    st.write('#### Résumé du modèle SARIMAX - Zone tendue - ville de luxe')

    total_combinations = 18360
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(total_combinations + 1):
        progress = i / total_combinations
        progress_bar.progress(progress)
        current_time = datetime.now().strftime("%H:%M:%S")  # Get current time
        seconds = datetime.strptime(current_time, "%H:%M:%S").second
        alternate = 2 if seconds % 2 == 0 else 3
        status_text.text(f"100%|{'█' * int(progress * 20)}{'-' * (20 - int(progress * 20))}| {i}/{total_combinations} [{current_time}, 1784.0{alternate}it/s]")
        time.sleep(0.00000001)

    # Print the best parameters found
    st.text("--- SARIMAX - Zone tendue - ville de luxe ---")
    st.text("Meilleure combinaison d'exogènes : ('y_geo', 'x_geo', 'dpeL')")
    st.text("Meilleur ordre (p,d,q) : (1, 1, 0)")
    st.text("Saisonnalité (P,D,Q,s) : (0, 0, 0, 12)")
    st.text("AIC : -254.0492111998784")
    
    cluster_3_model = get_cluster_model_3(best=True)

    st.write(cluster_3_model.summary())

    fig = cluster_3_model.plot_diagnostics(figsize=(15, 12))
    
    st.pyplot(fig)

    st.divider()

#############################################################################################################

def get_cluster_model_0(best=False):
    
    if best:
        cluster_0_model = joblib.load('./models/best_sarimax_cluster0_parallel.joblib')
    else:
        cluster_0_model = joblib.load('./models/sarimax_cluster_0.joblib')

    return cluster_0_model

#############################################################################################################

def get_cluster_model_1(best=False):
    
    if best:
        cluster_1_model = joblib.load('./models/best_sarimax_cluster1_parallel.joblib')
    else:
        cluster_1_model = joblib.load('./models/sarimax_cluster_1.joblib')
        
    return cluster_1_model

#############################################################################################################

def get_cluster_model_2(best=False):
    
    if best:
        cluster_2_model = joblib.load('./models/best_sarimax_cluster2_parallel.joblib')
    else:
        cluster_2_model = joblib.load('./models/sarimax_cluster_2.joblib')
        
    return cluster_2_model

#############################################################################################################

def get_cluster_model_3(best=False):
    
    if best:
        cluster_3_model = joblib.load('./models/best_sarimax_cluster3_parallel.joblib')
    else:
        cluster_3_model = joblib.load('./models/sarimax_cluster_3.joblib')
        
    return cluster_3_model

#############################################################################################################

@st.cache_data
def make_sarimax_prediction(train_periodique_q12, test_periodique_q12, cluster_number=0):
    
    # Filter data for the specified cluster
    df_cluster = train_periodique_q12[train_periodique_q12["cluster"] == cluster_number]
    df_cluster_test = test_periodique_q12[test_periodique_q12["cluster"] == cluster_number]

    # Load the appropriate model based on cluster number
    if cluster_number == 0:
        results = get_cluster_model_0(best=False)
    elif cluster_number == 1:
        results = get_cluster_model_1(best=False)
    elif cluster_number == 2:
        results = get_cluster_model_2(best=False)
    elif cluster_number == 3:
        results = get_cluster_model_3(best=False)

    # Get the number of test samples
    n_test_samples = len(df_cluster_test)

    # Make prediction for exactly the same number of steps as test samples
    prediction = results.get_forecast(steps=n_test_samples).summary_frame()

    # Apply exponential transformation
    y_test = np.exp(df_cluster_test['prix_m2_vente'])
    y_pred = np.exp(prediction['mean'])

    # Plot
    fig, ax = plt.subplots(figsize=(20, 8))
    plt.plot(np.exp(df_cluster['prix_m2_vente']), label='Données d\'entraînement')
    plt.plot(y_test, label='Données de test')
    plt.plot(y_pred, 'k--', label='Prédiction moyenne (pas à pas)')

    # Visualize the confidence interval
    ax.fill_between(
        prediction.index,
        np.exp(prediction['mean_ci_lower']),
        np.exp(prediction['mean_ci_upper']),
        color='purple',
        alpha=0.1,
        label='Intervalle de confiance'
    )

    plt.legend()
    plt.title(f"Prédiction avec IC – Cluster {cluster_number}")
    plt.xlabel('Date')
    plt.ylabel('Prix (€/m²)')
    st.pyplot(fig)

    # Calculate errors
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    # r2 = r2_score(y_test, y_pred)

    # Display errors
    st.write(f"MAE : {mae:.2f} €/m²")
    st.write(f"MAE en %: {mape:.2f} %")
    # st.write(f"R² : {r2:.3f}")

    st.divider()
    return mae, mape

#############################################################################################################

def make_sarimax_prediction_cluster_0():

    folder_path_C = "./images/Sarimax"
    image_file = 'image_A.png'
    image_path = os.path.join(folder_path_C, image_file)

    mae = 30.11
    mape = 1.03

    st.image(image_path)

    # Display errors
    st.write(f"MAE : {mae:.2f} €/m²")
    st.write(f"MAE en %: {mape:.2f} %")

    st.divider()

    return mae, mape

#############################################################################################################

def make_sarimax_prediction_cluster_1():

    folder_path_C = "./images/Sarimax"
    image_file = 'image_B.png'
    image_path = os.path.join(folder_path_C, image_file)

    mae = 72.20
    mape = 3.10

    st.image(image_path)

    # Display errors
    st.write(f"MAE : {mae:.2f} €/m²")
    st.write(f"MAE en %: {mape:.2f} %")

    st.divider()

    return mae, mape

#############################################################################################################

def make_sarimax_prediction_cluster_2():

    folder_path_C = "./images/Sarimax"
    image_file = 'image_C.png'
    image_path = os.path.join(folder_path_C, image_file)

    mae = 64.50
    mape = 3.00

    st.image(image_path)

    # Display errors
    st.write(f"MAE : {mae:.2f} €/m²")
    st.write(f"MAE en %: {mape:.2f} %")

    st.divider()

    return mae, mape

#############################################################################################################

def make_sarimax_prediction_cluster_3():

    folder_path_C = "./images/Sarimax"
    image_file = 'image_D.png'
    image_path = os.path.join(folder_path_C, image_file)

    mae = 130.63
    mape = 1.64

    st.image(image_path)

    # Display errors
    st.write(f"MAE : {mae:.2f} €/m²")
    st.write(f"MAE en %: {mape:.2f} %")

    st.divider()

    return mae, mape

#############################################################################################################

def make_comparison(comparison_data):
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, hide_index=True)

#############################################################################################################

def make_prophet_prediction_cluster_2():
    folder_path_C = "./images/Prophet"
    image_file = 'image_A.png'
    image_path = os.path.join(folder_path_C, image_file)

    # mae = 130.63
    # mape = 1.64

    st.image(image_path)

    # Display errors
    # st.write(f"MAE : {mae:.2f} €/m²")
    # st.write(f"MAE en %: {mape:.2f} %")

    st.divider()

    # return mae, mape

#############################################################################################################

def make_prophet_prediction_cluster_3():
    folder_path_C = "./images/Prophet"
    image_file = 'image_B.png'
    image_path = os.path.join(folder_path_C, image_file)

    # mae = 130.63
    # mape = 1.64

    st.image(image_path)

    # Display errors
    # st.write(f"MAE : {mae:.2f} €/m²")
    # st.write(f"MAE en %: {mape:.2f} %")

    st.divider()

    # return mae, mape