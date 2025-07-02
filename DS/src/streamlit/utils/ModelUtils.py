################################################################################
#                                                                              #
#                          ANALYSE ET MODÉLISATION                             #
#                                                                              #
################################################################################
"""
Module contenant les fonctions d'analyse, d'évaluation, de visualisation et de gestion
des modèles pour un projet d'analyse immobilière basé sur Streamlit, pandas et scikit-learn.

Ce module complète Encoding.py en fournissant des outils pour l'analyse des données,
l'optimisation des modèles, l'interprétation des résultats et la sauvegarde/chargement
des modèles et features.
"""

# Standard library
import os
import warnings
from typing import List, Dict, Tuple, Union, Optional, Any, Callable

# Data manipulation
import numpy as np
import pandas as pd

# Machine learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Model saving/loading
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    warnings.warn("Le module joblib n'est pas installé. "
                 "Les fonctionnalités de sauvegarde/chargement de modèles seront limitées.")

# Hyperparameter optimization
try:
    import optuna
    from optuna.integration import OptunaSearchCV
    import optuna.visualization as vis
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    warnings.warn("Le module optuna n'est pas installé. "
                 "Les fonctionnalités d'optimisation d'hyperparamètres seront limitées.")

# Model interpretation
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("Le module shap n'est pas installé. "
                 "Les fonctionnalités d'interprétation de modèles seront limitées.")

# LightGBM
try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("Le module lightgbm n'est pas installé. "
                 "Les fonctionnalités de modélisation LightGBM seront limitées.")

try:
    from optuna.integration import OptunaSearchCV
except ImportError:
    OptunaSearchCV = None


################################################################################
####################### ÉCHANTILLONNAGE ET PRÉPARATION #########################
################################################################################

def create_stratified_sample(df: pd.DataFrame, 
                           target_col: str = "prix_m2_vente",
                           train_size: float = 0.2,
                           n_bins: int = 10,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Crée un échantillon stratifié à partir d'un DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame source
        target_col (str): Nom de la colonne cible pour la stratification
        train_size (float): Proportion de l'échantillon à extraire
        n_bins (int): Nombre de bins pour la stratification
        random_state (int): Graine aléatoire pour la reproductibilité
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Échantillon stratifié et reste des données
    """
    df = df.copy()
    
    # Création des bins pour la stratification
    df["target_bin"] = pd.qcut(
        df[target_col], 
        q=n_bins, 
        duplicates="drop"
    )
    
    # Échantillonnage stratifié
    sample, rest = train_test_split(
        df,
        train_size=train_size,
        stratify=df["target_bin"],
        random_state=random_state
    )
    
    # Suppression de la colonne temporaire
    sample = sample.drop(columns=["target_bin"])
    rest = rest.drop(columns=["target_bin"])
    
    return sample, rest


def prepare_train_test_data(df: pd.DataFrame, 
                          features: List[str],
                          target_col: str = "prix_m2_vente",
                          test_size: float = 0.2,
                          random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prépare les données d'entraînement et de test à partir d'un DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame source
        features (List[str]): Liste des colonnes à utiliser comme features
        target_col (str): Nom de la colonne cible
        test_size (float): Proportion des données à utiliser pour le test
        random_state (int): Graine aléatoire pour la reproductibilité
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test
    """
    # Création des bins pour la stratification
    df["target_bin"] = pd.qcut(
        df[target_col], 
        q=10, 
        duplicates="drop"
    )
    
    # Split train/test stratifié
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["target_bin"],
        random_state=random_state
    )
    
    # Suppression de la colonne temporaire
    train_df = train_df.drop(columns=["target_bin"])
    test_df = test_df.drop(columns=["target_bin"])
    
    # Extraction des features et de la cible
    X_train = train_df[features]
    y_train = train_df[target_col]
    X_test = test_df[features]
    y_test = test_df[target_col]
    
    return X_train, X_test, y_train, y_test


################################################################################
####################### OPTIMISATION DES HYPERPARAMÈTRES #######################
################################################################################

def create_optuna_search_space() -> Dict[str, Any]:
    """
    Crée un espace de recherche pour l'optimisation des hyperparamètres avec Optuna.
    
    Returns:
        Dict[str, Any]: Dictionnaire des distributions de paramètres
        
    Raises:
        ImportError: Si optuna n'est pas installé
    """
    if not HAS_OPTUNA:
        raise ImportError("Cette fonction nécessite optuna. "
                         "Installez-le avec 'pip install optuna'.")
    
    param_distributions = {
        "model__num_leaves":       optuna.distributions.IntDistribution(16, 64),
        "model__learning_rate":    optuna.distributions.FloatDistribution(1e-3, 0.2, log=True),
        "model__n_estimators":     optuna.distributions.IntDistribution(50, 500),
        "model__max_depth":        optuna.distributions.IntDistribution(3, 8),
        "model__subsample":        optuna.distributions.FloatDistribution(0.5, 1.0),
        "model__colsample_bytree": optuna.distributions.FloatDistribution(0.5, 1.0),
        "model__reg_alpha":        optuna.distributions.FloatDistribution(0.0, 2.0),
        "model__reg_lambda":       optuna.distributions.FloatDistribution(0.0, 2.0),
    }
    
    return param_distributions


def optimize_hyperparameters(pipeline: Pipeline, 
                           X_train: pd.DataFrame, 
                           y_train: pd.Series,
                           param_distributions: Optional[Dict[str, Any]] = None,
                           n_trials: int = 30,
                           cv: int = 5,
                           scoring: str = "r2",
                           n_jobs: int = -1,
                           random_state: int = 42,
                           verbose: int = 1) -> OptunaSearchCV:
    """
    Optimise les hyperparamètres d'un pipeline avec Optuna.
    
    Args:
        pipeline (Pipeline): Pipeline scikit-learn à optimiser
        X_train (pd.DataFrame): Features d'entraînement
        y_train (pd.Series): Cible d'entraînement
        param_distributions (Optional[Dict[str, Any]]): Espace de recherche des hyperparamètres
        n_trials (int): Nombre d'essais pour l'optimisation
        cv (int): Nombre de folds pour la validation croisée
        scoring (str): Métrique d'évaluation
        n_jobs (int): Nombre de jobs parallèles (-1 pour utiliser tous les cœurs)
        random_state (int): Graine aléatoire pour la reproductibilité
        verbose (int): Niveau de verbosité
        
    Returns:
        OptunaSearchCV: Objet OptunaSearchCV ajusté
        
    Raises:
        ImportError: Si optuna n'est pas installé
    """
    if not HAS_OPTUNA:
        raise ImportError("Cette fonction nécessite optuna. "
                         "Installez-le avec 'pip install optuna'.")
    
    # Utiliser l'espace de recherche par défaut si aucun n'est fourni
    if param_distributions is None:
        param_distributions = create_optuna_search_space()
    
    # Configuration de l'optimisation
    optuna_search = OptunaSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        cv=cv,
        n_trials=n_trials,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose
    )
    
    # Lancement de l'optimisation
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        optuna_search.fit(X_train, y_train)
    
    # Affichage des meilleurs résultats
    print(f"🏆 Best {scoring} CV : {optuna_search.best_score_:.4f}")
    print("🔧 Best params:")
    for k, v in optuna_search.best_params_.items():
        print(f" - {k}: {v}")
    
    return optuna_search


def visualize_optuna_results(optuna_search: OptunaSearchCV) -> None:
    """
    Visualise les résultats de l'optimisation Optuna.
    
    Args:
        optuna_search (OptunaSearchCV): Objet OptunaSearchCV ajusté
        
    Raises:
        ImportError: Si optuna n'est pas installé
    """
    if not HAS_OPTUNA:
        raise ImportError("Cette fonction nécessite optuna. "
                         "Installez-le avec 'pip install optuna'.")
    
    study = optuna_search.study_
    
    # Importance des hyperparamètres
    param_importances = vis.plot_param_importances(study)
    param_importances.show()
    
    # Historique de l'optimisation
    optimization_history = vis.plot_optimization_history(study)
    optimization_history.show()
    
    # Diagramme de contours pour deux hyperparamètres
    contour_plot = vis.plot_contour(study, params=['model__learning_rate', 'model__num_leaves'])
    contour_plot.show()
    
    # Diagramme de coordonnées parallèles
    parallel_coordinate = vis.plot_parallel_coordinate(study)
    parallel_coordinate.show()


################################################################################
####################### ÉVALUATION DES MODÈLES #################################
################################################################################

def evaluate_model(model: Any, 
                 X_train: pd.DataFrame, 
                 y_train: pd.Series,
                 X_test: pd.DataFrame, 
                 y_test: pd.Series) -> Dict[str, float]:
    """
    Évalue un modèle sur les données d'entraînement et de test.
    
    Args:
        model (Any): Modèle ajusté
        X_train (pd.DataFrame): Features d'entraînement
        y_train (pd.Series): Cible d'entraînement
        X_test (pd.DataFrame): Features de test
        y_test (pd.Series): Cible de test
        
    Returns:
        Dict[str, float]: Dictionnaire des métriques d'évaluation
    """
    # Prédictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calcul des métriques
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Écart R²
    diff_r2 = r2_train - r2_test
    
    # Affichage des résultats
    print("\n🔍 Vérification du surapprentissage…")
    print(f" - R² train : {r2_train:.4f}")
    print(f" - RMSE train : {rmse_train:.2f}")
    
    print("\n📊 Évaluation sur le jeu de test...")
    print(f" - R² test : {r2_test:.4f}")
    print(f" - RMSE test : {rmse_test:.2f}")
    
    print(f"\n▶ Écart R² (train – test) : {diff_r2:.4f}")
    if diff_r2 > 0.05:
        print("⚠️  Écart R² > 0.05 : possible surapprentissage.")
    else:
        print("✅ Écart R² raisonnable, surapprentissage limité.")
    
    # Retour des métriques
    metrics = {
        "r2_train": r2_train,
        "rmse_train": rmse_train,
        "r2_test": r2_test,
        "rmse_test": rmse_test,
        "diff_r2": diff_r2
    }
    
    return metrics


def perform_stratified_cv(model: Any, 
                        X: pd.DataFrame, 
                        y: pd.Series,
                        n_splits: int = 5,
                        scoring: str = "r2",
                        random_state: int = 42) -> Tuple[np.ndarray, float, float]:
    """
    Effectue une validation croisée stratifiée sur un modèle.
    
    Args:
        model (Any): Modèle à évaluer
        X (pd.DataFrame): Features
        y (pd.Series): Cible
        n_splits (int): Nombre de folds
        scoring (str): Métrique d'évaluation
        random_state (int): Graine aléatoire pour la reproductibilité
        
    Returns:
        Tuple[np.ndarray, float, float]: Scores par fold, score moyen, écart-type
    """
    # Création des bins pour la stratification
    y_strat = pd.qcut(y, q=n_splits, duplicates='drop', labels=False)
    
    # Configuration de la validation croisée
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Calcul des scores
    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv.split(X, y_strat),
        scoring=scoring
    )
    
    # Calcul des statistiques
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    
    # Affichage des résultats
    print(f"Scores {scoring} par fold : {cv_scores}")
    print(f"Score {scoring} moyen (CV) : {mean_score:.4f} ± {std_score:.4f}")
    
    return cv_scores, mean_score, std_score


################################################################################
####################### INTERPRÉTATION DES MODÈLES #############################
################################################################################

def get_feature_names_from_column_transformer(column_transformer: Any) -> List[str]:
    """
    Récupère les noms des features transformées à partir d'un ColumnTransformer.
    
    Args:
        column_transformer (Any): ColumnTransformer ajusté
        
    Returns:
        List[str]: Liste des noms de features transformées
    """
    feature_names = []
    
    for name, trans, cols in column_transformer.transformers_:
        if name != 'remainder':
            if hasattr(trans, 'get_feature_names_out'):
                try:
                    names = trans.get_feature_names_out()
                except TypeError:
                    names = trans.get_feature_names_out(cols)
                except:
                    names = cols
            elif hasattr(trans, 'named_steps'):
                last_step = list(trans.named_steps.values())[-1]
                if hasattr(last_step, 'get_feature_names_out'):
                    try:
                        names = last_step.get_feature_names_out(cols)
                    except:
                        names = cols
                else:
                    names = cols
            else:
                names = cols
            feature_names.extend(names)
        else:
            if trans == 'passthrough':
                feature_names.extend(cols)
    
    return feature_names


def calculate_shap_values(model: Any, 
                        X: pd.DataFrame, 
                        preproc: Optional[Any] = None,
                        sample_size: int = 500,
                        random_state: int = 42) -> Tuple[Any, np.ndarray, List[str]]:
    """
    Calcule les valeurs SHAP pour un modèle.
    
    Args:
        model (Any): Modèle ajusté
        X (pd.DataFrame): Features
        preproc (Optional[Any]): Préprocesseur ajusté
        sample_size (int): Taille de l'échantillon pour le calcul SHAP
        random_state (int): Graine aléatoire pour la reproductibilité
        
    Returns:
        Tuple[Any, np.ndarray, List[str]]: Explainer SHAP, valeurs SHAP, noms des features
        
    Raises:
        ImportError: Si shap n'est pas installé
    """
    if not HAS_SHAP:
        raise ImportError("Cette fonction nécessite shap. "
                         "Installez-le avec 'pip install shap'.")
    
    # Sélection d'un sous-échantillon pour accélérer le calcul
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=random_state)
    
    # Transformation des données si un préprocesseur est fourni
    if preproc is not None:
        X_transformed = preproc.transform(X_sample)
        feature_names = get_feature_names_from_column_transformer(preproc)
    else:
        X_transformed = X_sample
        feature_names = X_sample.columns.tolist()
    
    # Calcul des valeurs SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(X_transformed)
    
    return explainer, shap_values, feature_names


def plot_shap_summary(shap_values: Any, 
                    X_transformed: np.ndarray, 
                    feature_names: List[str]) -> None:
    """
    Affiche un résumé des valeurs SHAP.
    
    Args:
        shap_values (Any): Valeurs SHAP calculées
        X_transformed (np.ndarray): Features transformées
        feature_names (List[str]): Noms des features
        
    Raises:
        ImportError: Si shap n'est pas installé
    """
    if not HAS_SHAP:
        raise ImportError("Cette fonction nécessite shap. "
                         "Installez-le avec 'pip install shap'.")
    
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names)


def select_top_features_by_shap(shap_values: Any, 
                              feature_names: List[str],
                              original_features: List[str],
                              top_n: int = 40) -> List[str]:
    """
    Sélectionne les meilleures features selon leur importance SHAP.
    
    Args:
        shap_values (Any): Valeurs SHAP calculées
        feature_names (List[str]): Noms des features transformées
        original_features (List[str]): Noms des features originales
        top_n (int): Nombre de features à sélectionner
        
    Returns:
        List[str]: Liste des meilleures features
        
    Raises:
        ImportError: Si shap n'est pas installé
    """
    if not HAS_SHAP:
        raise ImportError("Cette fonction nécessite shap. "
                         "Installez-le avec 'pip install shap'.")
    
    # Calcul de l'importance SHAP
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    
    # Création d'un DataFrame d'importance
    df_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": shap_importance
    }).sort_values(by="importance", ascending=False)
    
    # Filtrage pour ne garder que les features métiers
    features_metier = [f for f in original_features if f in df_importance["feature"].values]
    df_importance_metier = df_importance[df_importance["feature"].isin(features_metier)]
    
    # Sélection des meilleures features
    top_features = df_importance_metier["feature"].head(top_n).tolist()
    
    print(f"Top {top_n} features métiers : {top_features}")
    
    return top_features


################################################################################
####################### SAUVEGARDE ET CHARGEMENT ###############################
################################################################################

def save_model_and_features(model: Any, 
                          features: List[str],
                          model_path: str = "model.pkl",
                          features_path: str = "features.pkl") -> None:
    """
    Sauvegarde un modèle et sa liste de features.
    
    Args:
        model (Any): Modèle à sauvegarder
        features (List[str]): Liste des features utilisées par le modèle
        model_path (str): Chemin de sauvegarde du modèle
        features_path (str): Chemin de sauvegarde des features
        
    Raises:
        ImportError: Si joblib n'est pas installé
    """
    if not HAS_JOBLIB:
        raise ImportError("Cette fonction nécessite joblib. "
                         "Installez-le avec 'pip install joblib'.")
    
    # Sauvegarde du modèle et des features
    joblib.dump(model, model_path)
    joblib.dump(features, features_path)
    
    print(f"✅ Modèle sauvegardé dans {model_path}")
    print(f"✅ Features sauvegardées dans {features_path}")
    
    # Contrôle de cohérence
    loaded_features = joblib.load(features_path)
    assert features == loaded_features, "❌ La liste des features sauvegardée ne correspond pas à celle utilisée pour l'entraînement !"
    print("✅ Contrôle de cohérence : la liste des features est identique.")


def load_model_and_features(model_path: str = "model.pkl",
                          features_path: str = "features.pkl") -> Tuple[Any, List[str]]:
    """
    Charge un modèle et sa liste de features.
    
    Args:
        model_path (str): Chemin du modèle sauvegardé
        features_path (str): Chemin des features sauvegardées
        
    Returns:
        Tuple[Any, List[str]]: Modèle chargé et liste des features
        
    Raises:
        ImportError: Si joblib n'est pas installé
        FileNotFoundError: Si les fichiers n'existent pas
    """
    if not HAS_JOBLIB:
        raise ImportError("Cette fonction nécessite joblib. "
                         "Installez-le avec 'pip install joblib'.")
    
    # Vérification de l'existence des fichiers
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier modèle {model_path} n'existe pas.")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Le fichier features {features_path} n'existe pas.")
    
    # Chargement du modèle et des features
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    
    print(f"✅ Modèle chargé depuis {model_path}")
    print(f"✅ Features chargées depuis {features_path} : {len(features)} features")
    
    return model, features


def make_prediction_example(model: Any, 
                          features: List[str],
                          X: pd.DataFrame,
                          index: int = 0) -> float:
    """
    Fait une prédiction sur un exemple pour vérifier le fonctionnement du modèle.
    
    Args:
        model (Any): Modèle ajusté
        features (List[str]): Liste des features utilisées par le modèle
        X (pd.DataFrame): DataFrame contenant les features
        index (int): Index de l'exemple à utiliser
        
    Returns:
        float: Valeur prédite
    """
    # Extraction de l'exemple
    example_dict = {f: X.iloc[index][f] for f in features}
    X_input = pd.DataFrame([example_dict], columns=features)
    
    # Prédiction
    y_pred = model.predict(X_input)
    
    print(f"Prédiction pour l'exemple fourni : {y_pred[0]:.2f} €/m²")
    
    return y_pred[0]


################################################################################
####################### SAUVEGARDE DES DONNÉES #################################
################################################################################

def save_dataframe(df: pd.DataFrame, 
                 file_path: str,
                 sep: str = ';',
                 index: bool = False,
                 verify: bool = True,
                 chunksize: int = 100000) -> Optional[pd.DataFrame]:
    """
    Sauvegarde un DataFrame et vérifie optionnellement le résultat.
    
    Args:
        df (pd.DataFrame): DataFrame à sauvegarder
        file_path (str): Chemin de sauvegarde
        sep (str): Séparateur à utiliser
        index (bool): Si True, sauvegarde l'index
        verify (bool): Si True, recharge et vérifie le DataFrame sauvegardé
        chunksize (int): Taille des chunks pour la vérification
        
    Returns:
        Optional[pd.DataFrame]: DataFrame rechargé si verify=True, None sinon
    """
    # Sauvegarde du DataFrame
    df.to_csv(file_path, sep=sep, index=index)
    print(f"✅ DataFrame sauvegardé dans {file_path}")
    
    # Vérification optionnelle
    if verify:
        # Rechargement par chunks pour optimiser la mémoire
        chunks = pd.read_csv(
            file_path, 
            sep=sep, 
            chunksize=chunksize, 
            index_col=None if not index else 0,
            on_bad_lines='skip', 
            low_memory=False
        )
        df_verify = pd.concat(chunk for chunk in chunks)
        
        # Affichage des informations
        print("\nVérification du DataFrame sauvegardé :")
        print(df_verify.info())
        
        # Si le DataFrame contient une colonne cluster, afficher sa distribution
        if 'cluster' in df_verify.columns:
            print("\nValeurs uniques dans la colonne cluster :")
            print(df_verify['cluster'].value_counts())
        
        return df_verify
    
    return None


def save_train_test_data(X_train: pd.DataFrame, 
                       X_test: pd.DataFrame,
                       y_train: pd.Series,
                       y_test: pd.Series,
                       folder_path: str,
                       prefix: str = "",
                       sep: str = ';') -> None:
    """
    Sauvegarde les données d'entraînement et de test.
    
    Args:
        X_train (pd.DataFrame): Features d'entraînement
        X_test (pd.DataFrame): Features de test
        y_train (pd.Series): Cible d'entraînement
        y_test (pd.Series): Cible de test
        folder_path (str): Dossier de sauvegarde
        prefix (str): Préfixe pour les noms de fichiers
        sep (str): Séparateur à utiliser
    """
    # Création du dossier si nécessaire
    os.makedirs(folder_path, exist_ok=True)
    
    # Construction des chemins
    prefix = prefix + "_" if prefix else ""
    X_train_path = os.path.join(folder_path, f'{prefix}X_train.csv')
    X_test_path = os.path.join(folder_path, f'{prefix}X_test.csv')
    y_train_path = os.path.join(folder_path, f'{prefix}y_train.csv')
    y_test_path = os.path.join(folder_path, f'{prefix}y_test.csv')
    
    # Sauvegarde des données
    X_train.to_csv(X_train_path, sep=sep, index=False)
    X_test.to_csv(X_test_path, sep=sep, index=False)
    y_train.to_csv(y_train_path, sep=sep, index=False)
    y_test.to_csv(y_test_path, sep=sep, index=False)
    
    print(f"✅ Données d'entraînement et de test sauvegardées dans {folder_path}")
    print(f" - X_train : {X_train.shape}")
    print(f" - X_test : {X_test.shape}")
    print(f" - y_train : {y_train.shape}")
    print(f" - y_test : {y_test.shape}")
