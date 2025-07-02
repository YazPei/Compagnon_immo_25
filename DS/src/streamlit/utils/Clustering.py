import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

class ClusteringUtils:
    """
    Classe utilitaire pour le clustering des données immobilières.
    Fournit des fonctions pour préparer les données, effectuer le clustering
    et visualiser les résultats.
    """
    
    @staticmethod
    def regroup_cp(code_postal):
        """
        Regroupe les codes postaux pour réduire la granularité.
        
        Args:
            code_postal: Code postal à regrouper
            
        Returns:
            str: Code postal regroupé
        """
        if pd.isna(code_postal):
            return "inconnu"
        
        cp = str(code_postal)
        
        # DROM-COM
        if cp.startswith("97") and len(cp) >= 3:
            return cp[:3]
        
        # Codes postaux standards
        if cp.isdigit() and len(cp) == 5:
            return cp[:2]
        
        return "inconnu"
    
    @staticmethod
    def get_code_postal_final(row):
        """
        Obtient le code postal final à partir de la zone mixte.
        
        Args:
            row: Ligne du DataFrame
            
        Returns:
            str: Code postal final
        """
        if "codePostal_grouped" in row:
            zone = row["codePostal_grouped"]
        elif "zone_mixte" in row:
            zone = row["zone_mixte"]
        else:
            return "00000"
        
        s = str(zone)
        if s.isdigit() and len(s) == 5:
            return s
        if s.isdigit() and len(s) == 2:
            return s + "000"
        if s.startswith("97") and len(s) == 3:
            return s + "00"
        return "00000"
    
    @staticmethod
    def prepare_data_for_clustering(df, date_col="date", code_postal_col="codePostal", price_col="prix_m2_vente"):
        """
        Prépare les données pour le clustering.
        
        Args:
            df: DataFrame contenant les données
            date_col: Nom de la colonne contenant la date
            code_postal_col: Nom de la colonne contenant le code postal
            price_col: Nom de la colonne contenant le prix au m²
            
        Returns:
            DataFrame: DataFrame préparé pour le clustering
        """
        # Vérification des colonnes nécessaires
        required_cols = [date_col, code_postal_col, price_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"La colonne {col} est requise pour le clustering")
        
        # Copie du DataFrame pour éviter les modifications en place
        df_result = df.copy()
        
        # Conversion de la date en datetime si ce n'est pas déjà fait
        if not pd.api.types.is_datetime64_any_dtype(df_result[date_col]):
            df_result[date_col] = pd.to_datetime(df_result[date_col], errors='coerce')
        
        # Création de la zone mixte
        df_result["zone_mixte"] = df_result[code_postal_col].apply(ClusteringUtils.regroup_cp)
        
        return df_result
    
    @staticmethod
    def calculate_tcam(df, date_col="date", code_postal_col="zone_mixte", price_col="prix_m2_vente"):
        """
        Calcule le Taux de Croissance Annuel Moyen (TCAM) par zone.
        
        Args:
            df: DataFrame contenant les données
            date_col: Nom de la colonne contenant la date
            code_postal_col: Nom de la colonne contenant le code postal ou la zone
            price_col: Nom de la colonne contenant le prix au m²
            
        Returns:
            DataFrame: DataFrame contenant le TCAM par zone
        """
        # Vérification des colonnes nécessaires
        required_cols = [date_col, code_postal_col, price_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"La colonne {col} est requise pour le calcul du TCAM")
        
        # Copie du DataFrame pour éviter les modifications en place
        df_result = df.copy()
        
        # Conversion de la date en datetime si ce n'est pas déjà fait
        if not pd.api.types.is_datetime64_any_dtype(df_result[date_col]):
            df_result[date_col] = pd.to_datetime(df_result[date_col], errors='coerce')
        
        # Fonction pour calculer le TCAM pour une zone
        def calculate_zone_tcam(zone_data):
            if len(zone_data) <= 1:
                return 0  # Pas assez de données pour calculer un TCAM
            
            # Tri par date
            zone_data = zone_data.sort_values(date_col)
            
            # Calcul du TCAM
            first_price = zone_data.iloc[0][price_col]
            last_price = zone_data.iloc[-1][price_col]
            years = (zone_data.iloc[-1][date_col] - zone_data.iloc[0][date_col]).days / 365.25
            
            if years <= 0 or first_price <= 0:
                return 0
            
            tcam = (last_price / first_price) ** (1 / years) - 1
            return tcam
        
        # Calcul du TCAM par zone
        tcam_values = []
        
        for zone, group in df_result.groupby(code_postal_col):
            tcam = calculate_zone_tcam(group)
            tcam_values.append({"zone_mixte": zone, "tcam": tcam})
        
        return pd.DataFrame(tcam_values)
    
    @staticmethod
    def aggregate_monthly_data(df, date_col="date", code_postal_col="zone_mixte", price_col="prix_m2_vente"):
        """
        Agrège les données par mois et par zone.
        
        Args:
            df: DataFrame contenant les données
            date_col: Nom de la colonne contenant la date
            code_postal_col: Nom de la colonne contenant le code postal ou la zone
            price_col: Nom de la colonne contenant le prix au m²
            
        Returns:
            DataFrame: DataFrame contenant les données agrégées
        """
        # Vérification des colonnes nécessaires
        required_cols = [date_col, code_postal_col, price_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"La colonne {col} est requise pour l'agrégation mensuelle")
        
        # Copie du DataFrame pour éviter les modifications en place
        df_result = df.copy()
        
        # Conversion de la date en datetime si ce n'est pas déjà fait
        if not pd.api.types.is_datetime64_any_dtype(df_result[date_col]):
            df_result[date_col] = pd.to_datetime(df_result[date_col], errors='coerce')
        
        # Extraction de l'année et du mois
        df_result["year"] = df_result[date_col].dt.year
        df_result["month"] = df_result[date_col].dt.month
        
        # Agrégation mensuelle par zone
        monthly_data = (
            df_result.groupby(["year", "month", code_postal_col])
            .agg(
                prix_moyen=(price_col, "mean"),
                volume_ventes=(price_col, "count"),
                prix_median=(price_col, "median"),
                prix_min=(price_col, "min"),
                prix_max=(price_col, "max")
            )
            .reset_index()
        )
        
        # Création de la date complète
        monthly_data["date"] = pd.to_datetime(
            monthly_data["year"].astype(str) + "-" + 
            monthly_data["month"].astype(str).str.zfill(2) + "-01"
        )
        
        return monthly_data
    
    @staticmethod
    def create_clustering_features(tcam_df, monthly_df, price_col="prix_moyen"):
        """
        Crée les features pour le clustering.
        
        Args:
            tcam_df: DataFrame contenant le TCAM par zone
            monthly_df: DataFrame contenant les données mensuelles
            price_col: Nom de la colonne contenant le prix moyen
            
        Returns:
            DataFrame: DataFrame contenant les features pour le clustering
        """
        # Vérification des colonnes nécessaires
        if "zone_mixte" not in tcam_df.columns or "tcam" not in tcam_df.columns:
            raise ValueError("Les colonnes 'zone_mixte' et 'tcam' sont requises dans tcam_df")
        
        if "zone_mixte" not in monthly_df.columns or price_col not in monthly_df.columns:
            raise ValueError(f"Les colonnes 'zone_mixte' et '{price_col}' sont requises dans monthly_df")
        
        # Calcul de la volatilité (écart-type des prix) par zone
        volatility_df = monthly_df.groupby("zone_mixte")[price_col].std().reset_index()
        volatility_df.rename(columns={price_col: "volatility"}, inplace=True)
        
        # Calcul du prix moyen par zone
        price_df = monthly_df.groupby("zone_mixte")[price_col].mean().reset_index()
        price_df.rename(columns={price_col: "avg_price"}, inplace=True)
        
        # Calcul du volume moyen de ventes par zone
        volume_df = monthly_df.groupby("zone_mixte")["volume_ventes"].mean().reset_index()
        
        # Fusion des DataFrames
        result = tcam_df.merge(volatility_df, on="zone_mixte", how="left")
        result = result.merge(price_df, on="zone_mixte", how="left")
        result = result.merge(volume_df, on="zone_mixte", how="left")
        
        # Remplacement des valeurs manquantes
        result.fillna({
            "tcam": 0,
            "volatility": result["volatility"].median(),
            "avg_price": result["avg_price"].median(),
            "volume_ventes": result["volume_ventes"].median()
        }, inplace=True)
        
        return result
    
    @staticmethod
    def perform_clustering(data, n_clusters=4, random_state=42):
        """
        Réalise le clustering K-means sur les données.
        
        Args:
            data: DataFrame contenant les features pour le clustering
            n_clusters: Nombre de clusters à créer
            random_state: Graine aléatoire pour la reproductibilité
            
        Returns:
            tuple: (clusters, cluster_centers)
        """
        # Sélection des colonnes numériques pour le clustering
        numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
        
        # Exclusion de la colonne zone_mixte si présente
        if "zone_mixte" in numeric_cols:
            numeric_cols.remove("zone_mixte")
        
        # Normalisation des données
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[numeric_cols])
        
        # Création et entraînement du modèle K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        clusters = kmeans.fit_predict(data_scaled)
        
        # Récupération des centres des clusters
        cluster_centers = kmeans.cluster_centers_
        
        return clusters, cluster_centers
    
    @staticmethod
    def apply_clustering_pipeline(df, n_clusters=4, date_col="date", code_postal_col="codePostal", price_col="prix_m2_vente"):
        """
        Applique le pipeline de clustering complet.
        
        Args:
            df: DataFrame contenant les données
            n_clusters: Nombre de clusters à créer
            date_col: Nom de la colonne contenant la date
            code_postal_col: Nom de la colonne contenant le code postal
            price_col: Nom de la colonne contenant le prix au m²
            
        Returns:
            tuple: (df_clustered, cluster_info)
        """
        # Préparation des données pour le clustering
        df_prepared = ClusteringUtils.prepare_data_for_clustering(
            df,
            date_col=date_col,
            code_postal_col=code_postal_col,
            price_col=price_col
        )
        
        # Calcul du TCAM par zone
        tcam_df = ClusteringUtils.calculate_tcam(
            df_prepared,
            date_col=date_col,
            code_postal_col="zone_mixte",
            price_col=price_col
        )
        
        # Agrégation des données mensuelles
        monthly_df = ClusteringUtils.aggregate_monthly_data(
            df_prepared,
            date_col=date_col,
            code_postal_col="zone_mixte",
            price_col=price_col
        )
        
        # Création des features pour le clustering
        clustering_features = ClusteringUtils.create_clustering_features(
            tcam_df,
            monthly_df,
            price_col="prix_moyen"
        )
        
        # Réalisation du clustering
        clusters, _ = ClusteringUtils.perform_clustering(
            clustering_features,
            n_clusters=n_clusters
        )
        
        # Ajout des clusters aux features
        clustering_features["cluster"] = clusters
        
        # Ajout des clusters au DataFrame original
        zone_cluster_map = dict(zip(clustering_features["zone_mixte"], clustering_features["cluster"]))
        df_clustered = df_prepared.copy()
        df_clustered["cluster"] = df_clustered["zone_mixte"].map(zone_cluster_map)
        
        # Remplissage des valeurs manquantes pour cluster
        df_clustered["cluster"].fillna(-1, inplace=True)
        
        # Conversion du type de cluster en entier
        df_clustered["cluster"] = df_clustered["cluster"].astype(int)
        
        return df_clustered, clustering_features
    
    @staticmethod
    def determine_optimal_clusters(data, max_clusters=10, random_state=42):
        """
        Détermine le nombre optimal de clusters en utilisant la méthode du coude et le score de silhouette.
        
        Args:
            data: DataFrame contenant les données pour le clustering
            max_clusters: Nombre maximum de clusters à tester
            random_state: Graine aléatoire pour la reproductibilité
            
        Returns:
            tuple: (inertia_values, silhouette_values)
        """
        # Sélection des colonnes numériques pour le clustering
        numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
        
        # Exclusion de la colonne zone_mixte si présente
        if "zone_mixte" in numeric_cols:
            numeric_cols.remove("zone_mixte")
        
        # Normalisation des données
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data[numeric_cols])
        
        # Initialisation des listes pour stocker les résultats
        inertia_values = []
        silhouette_values = []
        
        # Calcul de l'inertie et du score de silhouette pour différents nombres de clusters
        for k in range(2, max_clusters + 1):
            # Création et entraînement du modèle K-means
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            kmeans.fit(data_scaled)
            
            # Stockage de l'inertie
            inertia_values.append(kmeans.inertia_)
            
            # Calcul et stockage du score de silhouette
            labels = kmeans.labels_
            silhouette_avg = silhouette_score(data_scaled, labels)
            silhouette_values.append(silhouette_avg)
        
        return inertia_values, silhouette_values
    
    @staticmethod
    def plot_cluster_evaluation(inertia_values, silhouette_values, max_clusters=10):
        """
        Visualise les résultats de la méthode du coude et du score de silhouette.
        
        Args:
            inertia_values: Liste des valeurs d'inertie
            silhouette_values: Liste des scores de silhouette
            max_clusters: Nombre maximum de clusters testés
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Tracé de l'inertie (méthode du coude)
        ax1.set_xlabel('Nombre de clusters')
        ax1.set_ylabel('Inertie', color='tab:blue')
        ax1.plot(range(2, max_clusters + 1), inertia_values, 'o-', color='tab:blue', label='Inertie')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Création d'un second axe y pour le score de silhouette
        ax2 = ax1.twinx()
        ax2.set_ylabel('Score de silhouette', color='tab:red')
        ax2.plot(range(2, max_clusters + 1), silhouette_values, 'o-', color='tab:red', label='Score de silhouette')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        # Titre et légende
        plt.title('Méthode du coude et score de silhouette pour déterminer le nombre optimal de clusters')
        
        # Légende combinée
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_cluster_distributions(data, cluster_col='cluster', price_col='prix_m2_vente'):
        """
        Visualise la distribution des variables par cluster.
        
        Args:
            data: DataFrame contenant les données et les clusters
            cluster_col: Nom de la colonne contenant les clusters
            price_col: Nom de la colonne contenant le prix au m²
            
        Returns:
            list: Liste de figures matplotlib
        """
        # Sélection des variables numériques
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != cluster_col]
        
        # Création des figures
        figures = []
        
        # Distribution des prix par cluster
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        sns.boxplot(x=cluster_col, y=price_col, data=data, ax=ax1)
        ax1.set_title(f'Distribution de {price_col} par cluster')
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel(price_col)
        figures.append(fig1)
        
        # Distribution des autres variables par cluster
        for col in numeric_cols[:5]:  # Limiter à 5 variables pour la lisibilité
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(x=cluster_col, y=col, data=data, ax=ax)
            ax.set_title(f'Distribution de {col} par cluster')
            ax.set_xlabel('Cluster')
            ax.set_ylabel(col)
            figures.append(fig)
        
        return figures
    
    @staticmethod
    def plot_cluster_map(data, lat_col='mapCoordonneesLatitude', lon_col='mapCoordonneesLongitude', cluster_col='cluster'):
        """
        Visualise la répartition géographique des clusters.
        
        Args:
            data: DataFrame contenant les données et les clusters
            lat_col: Nom de la colonne contenant la latitude
            lon_col: Nom de la colonne contenant la longitude
            cluster_col: Nom de la colonne contenant les clusters
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        # Vérification des colonnes nécessaires
        required_cols = [lat_col, lon_col, cluster_col]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"La colonne {col} est requise pour la visualisation géographique")
        
        # Échantillonnage pour la performance
        if len(data) > 5000:
            data_sample = data.sample(5000, random_state=42)
        else:
            data_sample = data
        
        # Création de la figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Tracé des points colorés par cluster
        scatter = ax.scatter(
            data_sample[lon_col],
            data_sample[lat_col],
            c=data_sample[cluster_col],
            cmap='viridis',
            alpha=0.6,
            s=10
        )
        
        # Ajout d'une légende
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        
        # Titre et labels
        ax.set_title('Répartition géographique des clusters')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Ajustement des limites pour la France métropolitaine
        ax.set_xlim(-5, 10)
        ax.set_ylim(41, 52)
        
        plt.tight_layout()
        return fig

# Expose les méthodes statiques comme fonctions de module pour import direct
regroup_cp = ClusteringUtils.regroup_cp
get_code_postal_final = ClusteringUtils.get_code_postal_final
prepare_data_for_clustering = ClusteringUtils.prepare_data_for_clustering
calculate_tcam = ClusteringUtils.calculate_tcam
aggregate_monthly_data = ClusteringUtils.aggregate_monthly_data
create_clustering_features = ClusteringUtils.create_clustering_features
perform_clustering = ClusteringUtils.perform_clustering
apply_clustering_pipeline = ClusteringUtils.apply_clustering_pipeline
determine_optimal_clusters = ClusteringUtils.determine_optimal_clusters
plot_cluster_evaluation = ClusteringUtils.plot_cluster_evaluation
plot_cluster_distributions = ClusteringUtils.plot_cluster_distributions
plot_cluster_map = ClusteringUtils.plot_cluster_map
