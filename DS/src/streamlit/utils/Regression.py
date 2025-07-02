import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import shap
import optuna

class RegressionUtils:
    """
    Classe utilitaire pour la modélisation prédictive des prix immobiliers.
    Fournit des fonctions pour préparer les données, entraîner des modèles,
    évaluer les performances et interpréter les résultats.
    """
    
    @staticmethod
    def prepare_features(train_data, test_data, target_variable="prix_m2_vente", feature_list=None):
        """
        Prépare les features pour la modélisation en séparant X et y.
        
        Args:
            train_data (pd.DataFrame): Données d'entraînement
            test_data (pd.DataFrame): Données de test
            target_variable (str): Nom de la variable cible
            feature_list (list): Liste des features à utiliser (si None, utilise toutes les colonnes sauf la cible)
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names)
        """
        # Vérification de la présence de la variable cible
        if target_variable not in train_data.columns:
            raise ValueError(f"La variable cible '{target_variable}' n'est pas présente dans les données d'entraînement.")
        
        # Sélection des features
        if feature_list is None:
            # Utilisation de toutes les colonnes sauf la cible
            feature_names = [col for col in train_data.columns if col != target_variable]
        else:
            # Vérification des colonnes disponibles
            feature_names = [f for f in feature_list if f in train_data.columns]
        
        # Extraction des features et de la cible
        X_train = train_data[feature_names]
        y_train = train_data[target_variable]
        X_test = test_data[feature_names]
        y_test = test_data[target_variable]
        
        # Gestion des valeurs manquantes
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # Imputation des valeurs manquantes
        imputer = SimpleImputer(strategy="median")
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
        
        return X_train, X_test, y_train, y_test, feature_names
    
    @staticmethod
    def create_lightgbm_model(params=None):
        """
        Crée un modèle LightGBM avec les paramètres spécifiés.
        
        Args:
            params (dict): Paramètres du modèle
            
        Returns:
            LGBMRegressor: Modèle LightGBM
        """
        # Paramètres par défaut
        default_params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 7,
            'num_leaves': 31,
            'random_state': 42
        }
        
        # Utilisation des paramètres spécifiés ou des paramètres par défaut
        if params is not None:
            model_params = {**default_params, **params}
        else:
            model_params = default_params
        
        # Création du modèle
        model = lgb.LGBMRegressor(**model_params)
        
        return model
    
    @staticmethod
    def create_optuna_search_space():
        """
        Crée un espace de recherche pour l'optimisation des hyperparamètres avec Optuna.
        
        Returns:
            dict: Espace de recherche
        """
        return {
            'n_estimators': (50, 1000),
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 15),
            'num_leaves': (10, 255),
            'min_child_samples': (5, 100),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0)
        }
    
    @staticmethod
    def optimize_hyperparameters(X_train, y_train, n_trials=30, random_state=42):
        """
        Optimise les hyperparamètres du modèle LightGBM avec Optuna.
        
        Args:
            X_train: Features d'entraînement
            y_train: Variable cible d'entraînement
            n_trials (int): Nombre d'essais pour l'optimisation
            random_state (int): Graine aléatoire pour la reproductibilité
            
        Returns:
            tuple: (best_params, study)
        """
        # Création de l'espace de recherche
        param_space = RegressionUtils.create_optuna_search_space()
        
        # Fonction objective pour Optuna
        def objective(trial):
            # Paramètres à optimiser
            params = {
                'n_estimators': trial.suggest_int('n_estimators', param_space['n_estimators'][0], param_space['n_estimators'][1]),
                'learning_rate': trial.suggest_float('learning_rate', param_space['learning_rate'][0], param_space['learning_rate'][1], log=True),
                'max_depth': trial.suggest_int('max_depth', param_space['max_depth'][0], param_space['max_depth'][1]),
                'num_leaves': trial.suggest_int('num_leaves', param_space['num_leaves'][0], param_space['num_leaves'][1]),
                'min_child_samples': trial.suggest_int('min_child_samples', param_space['min_child_samples'][0], param_space['min_child_samples'][1]),
                'subsample': trial.suggest_float('subsample', param_space['subsample'][0], param_space['subsample'][1]),
                'colsample_bytree': trial.suggest_float('colsample_bytree', param_space['colsample_bytree'][0], param_space['colsample_bytree'][1]),
                'random_state': random_state
            }
            
            # Création et entraînement du modèle
            model = lgb.LGBMRegressor(**params)
            
            # Validation croisée
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
            
            # Retourne la RMSE moyenne
            return np.sqrt(-np.mean(scores))
        
        # Création et exécution de l'étude Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Récupération des meilleurs paramètres
        best_params = study.best_params
        
        return best_params, study
    
    @staticmethod
    def visualize_optuna_results(study):
        """
        Visualise les résultats de l'optimisation des hyperparamètres avec Optuna.
        
        Args:
            study: Étude Optuna
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        # Création de la figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Récupération des paramètres importants
        param_names = ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves', 'subsample', 'colsample_bytree']
        
        # Tracé des graphiques d'importance des paramètres
        for i, param_name in enumerate(param_names):
            if i < len(axes):
                try:
                    # Récupération des valeurs du paramètre et des scores correspondants
                    param_values = []
                    scores = []
                    
                    for trial in study.trials:
                        if trial.state == optuna.trial.TrialState.COMPLETE and param_name in trial.params:
                            param_values.append(trial.params[param_name])
                            scores.append(trial.value)
                    
                    # Tracé du graphique
                    axes[i].scatter(param_values, scores, alpha=0.5)
                    axes[i].set_xlabel(param_name)
                    axes[i].set_ylabel('RMSE')
                    axes[i].set_title(f'Impact de {param_name} sur la RMSE')
                except Exception as e:
                    print(f"Erreur lors de la visualisation du paramètre {param_name}: {e}")
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """
        Évalue les performances du modèle sur les données de test.
        
        Args:
            model: Modèle entraîné
            X_test: Features de test
            y_test: Variable cible de test
            
        Returns:
            dict: Métriques d'évaluation
        """
        # Prédictions sur les données de test
        y_pred = model.predict(X_test)
        
        # Calcul des métriques
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Création du dictionnaire de métriques
        metrics = {
            'rmse': rmse,
            'r2': r2,
            'mae': mae
        }
        
        return metrics, y_pred
    
    @staticmethod
    def plot_predictions_vs_actual(y_test, y_pred):
        """
        Visualise les prédictions vs les valeurs réelles.
        
        Args:
            y_test: Valeurs réelles
            y_pred: Valeurs prédites
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        # Création de la figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Tracé des points
        ax.scatter(y_test, y_pred, alpha=0.3)
        
        # Tracé de la ligne y = x
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        
        # Titre et labels
        ax.set_xlabel('Prix réel (€/m²)')
        ax.set_ylabel('Prix prédit (€/m²)')
        ax.set_title('Prédictions vs Réalité')
        
        return fig
    
    @staticmethod
    def plot_error_distribution(y_test, y_pred):
        """
        Visualise la distribution des erreurs de prédiction.
        
        Args:
            y_test: Valeurs réelles
            y_pred: Valeurs prédites
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        # Calcul des erreurs
        errors = y_pred - y_test
        
        # Création de la figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Tracé de l'histogramme
        sns.histplot(errors, kde=True, ax=ax)
        
        # Titre et labels
        ax.set_xlabel('Erreur (€/m²)')
        ax.set_ylabel('Fréquence')
        ax.set_title('Distribution des erreurs de prédiction')
        
        return fig
    
    @staticmethod
    def plot_feature_importance(model, feature_names):
        """
        Visualise l'importance des features.
        
        Args:
            model: Modèle entraîné
            feature_names: Noms des features
            
        Returns:
            tuple: (feature_importance_df, matplotlib.figure.Figure)
        """
        # Récupération de l'importance des features
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Création de la figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Tracé du graphique
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20), ax=ax)
        
        # Titre
        ax.set_title('Importance des 20 principales features')
        
        return feature_importance, fig
    
    @staticmethod
    def calculate_shap_values(model, X_sample, feature_names, n_samples=500):
        """
        Calcule les valeurs SHAP pour l'interprétation du modèle.
        
        Args:
            model: Modèle entraîné
            X_sample: Échantillon de données pour le calcul des valeurs SHAP
            feature_names: Noms des features
            n_samples (int): Nombre d'échantillons à utiliser
            
        Returns:
            tuple: (shap_values, X_sample)
        """
        # Échantillonnage pour SHAP
        if len(X_sample) > n_samples:
            X_sample = X_sample.sample(n_samples, random_state=42)
        
        # Calcul des valeurs SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        return shap_values, X_sample
    
    @staticmethod
    def plot_shap_summary(shap_values, X_sample, feature_names):
        """
        Visualise le résumé des valeurs SHAP.
        
        Args:
            shap_values: Valeurs SHAP
            X_sample: Échantillon de données
            feature_names: Noms des features
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        # Création de la figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Tracé du résumé SHAP
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        
        return fig
    
    @staticmethod
    def plot_shap_dependence(shap_values, X_sample, feature_names, feature_idx):
        """
        Visualise la dépendance SHAP pour une feature spécifique.
        
        Args:
            shap_values: Valeurs SHAP
            X_sample: Échantillon de données
            feature_names: Noms des features
            feature_idx: Index de la feature à visualiser
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib
        """
        # Création de la figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Tracé de la dépendance SHAP
        shap.dependence_plot(feature_idx, shap_values, X_sample, feature_names=feature_names, show=False)
        
        return fig
    
    @staticmethod
    def save_model(model, model_path, feature_names=None, feature_path=None):
        """
        Sauvegarde le modèle et les noms des features.
        
        Args:
            model: Modèle à sauvegarder
            model_path (str): Chemin de sauvegarde du modèle
            feature_names (list): Noms des features
            feature_path (str): Chemin de sauvegarde des noms des features
            
        Returns:
            bool: True si la sauvegarde a réussi
        """
        # Création du répertoire si nécessaire
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Sauvegarde du modèle
        joblib.dump(model, model_path)
        
        # Sauvegarde des noms des features si spécifiés
        if feature_names is not None and feature_path is not None:
            joblib.dump(feature_names, feature_path)
        
        return True
    
    @staticmethod
    def load_model(model_path, feature_path=None):
        """
        Charge le modèle et les noms des features.
        
        Args:
            model_path (str): Chemin du modèle
            feature_path (str): Chemin des noms des features
            
        Returns:
            tuple: (model, feature_names)
        """
        # Chargement du modèle
        model = joblib.load(model_path)
        
        # Chargement des noms des features si spécifiés
        feature_names = None
        if feature_path is not None and os.path.exists(feature_path):
            feature_names = joblib.load(feature_path)
        
        return model, feature_names
    
    @staticmethod
    def predict_price(model, new_data, feature_names=None):
        """
        Prédit le prix pour de nouvelles données.
        
        Args:
            model: Modèle entraîné
            new_data: Nouvelles données
            feature_names (list): Noms des features à utiliser
            
        Returns:
            numpy.ndarray: Prédictions
        """
        # Vérification du type de new_data
        if isinstance(new_data, pd.DataFrame):
            # Sélection des features si spécifiées
            if feature_names is not None:
                # Vérification des colonnes disponibles
                available_features = [f for f in feature_names if f in new_data.columns]
                if len(available_features) < len(feature_names):
                    warnings.warn(f"Certaines features sont manquantes : {set(feature_names) - set(available_features)}")
                
                # Sélection des features disponibles
                X = new_data[available_features]
            else:
                X = new_data
        else:
            # Conversion en DataFrame si nécessaire
            X = pd.DataFrame([new_data])
        
        # Gestion des valeurs manquantes
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Imputation des valeurs manquantes
        imputer = SimpleImputer(strategy="median")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Prédiction
        predictions = model.predict(X)
        
        return predictions
