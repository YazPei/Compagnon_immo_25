import numpy as np
import pandas as pd
import unicodedata
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

def preprocessing_full(
    df: pd.DataFrame,
    target_col: str = "prix_m2_vente",
    max_onehot_modalities: int = 20,
    scale: bool = True,
    verbose: bool = False,
    mode_debug_notebook: bool = True,  # Ajouté !
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Pipeline de prétraitement avancé, notebook-style.
    Retourne X_df (features numériques), y (target), feature_names_clean (list[str]).
    """
    # Fonctions utilitaires internes

    def cyclical_encode(X_in: pd.DataFrame, period: int) -> pd.DataFrame:
        """Encode une variable cyclique (e.g. mois, jour) en deux colonnes sin et cos."""
        X = X_in.copy()
        radians = 2 * np.pi * X / period
        return pd.DataFrame({
            f"{X.columns[0]}_sin": np.sin(radians.iloc[:, 0]),
            f"{X.columns[0]}_cos": np.cos(radians.iloc[:, 0])
        }, index=X.index)

    def cyclical_encode_month(X_in):
        """
        Encode la colonne 'date' en sin/cos cyclique (mois, période 12).
        Retourne un DataFrame aligné sur X.index même si X_in est une Series ou ndarray.
        Gère explicitement les types inattendus.
        """
        # Gère DataFrame, Series, ndarray
        if isinstance(X_in, pd.DataFrame):
            if "date" not in X_in.columns:
                raise ValueError("La colonne 'date' est attendue.")
            date_col = X_in["date"]
            idx = X_in.index
        elif isinstance(X_in, pd.Series):
            date_col = X_in
            idx = X_in.index
        elif isinstance(X_in, np.ndarray):
            # ndarray: pas d'index, on crée un RangeIndex
            date_col = pd.Series(X_in.ravel())
            idx = date_col.index
        else:
            raise ValueError("Type inattendu pour cyclical_encode_month: %s" % type(X_in))
        # Gestion du type de la colonne
        # Si ce n'est pas string/object, convertit d'abord
        if not pd.api.types.is_string_dtype(date_col) and not pd.api.types.is_object_dtype(date_col):
            # Peut-être datetime déjà, sinon force string
            try:
                date_col = pd.to_datetime(date_col)
            except Exception:
                date_col = date_col.astype(str)
        else:
            date_col = pd.to_datetime(date_col, errors="coerce")
        month = date_col.dt.month
        radians = 2 * np.pi * month / 12
        df_out = pd.DataFrame({
            "month_sin": np.sin(radians),
            "month_cos": np.cos(radians)
        }, index=idx)
        return df_out

    def cyclical_encode_dow(X_in):
        """
        Encode la colonne 'date' en jour de la semaine sin/cos (période 7).
        Retourne un DataFrame aligné sur X.index même si X_in est une Series ou ndarray.
        Gère explicitement les types inattendus.
        """
        if isinstance(X_in, pd.DataFrame):
            if "date" not in X_in.columns:
                raise ValueError("La colonne 'date' est attendue.")
            date_col = X_in["date"]
            idx = X_in.index
        elif isinstance(X_in, pd.Series):
            date_col = X_in
            idx = X_in.index
        elif isinstance(X_in, np.ndarray):
            date_col = pd.Series(X_in.ravel())
            idx = date_col.index
        else:
            raise ValueError("Type inattendu pour cyclical_encode_dow: %s" % type(X_in))
        if not pd.api.types.is_string_dtype(date_col) and not pd.api.types.is_object_dtype(date_col):
            try:
                date_col = pd.to_datetime(date_col)
            except Exception:
                date_col = date_col.astype(str)
        else:
            date_col = pd.to_datetime(date_col, errors="coerce")
        dow = date_col.dt.weekday
        radians = 2 * np.pi * dow / 7
        df_out = pd.DataFrame({
            "dow_sin": np.sin(radians),
            "dow_cos": np.cos(radians)
        }, index=idx)
        return df_out

    def replace_minus1(X_in):
        """
        Remplace les -1 par 5 dans un DataFrame pandas, une Series ou un ndarray numpy.
        Compatible pipelines scikit-learn.
        """
        if isinstance(X_in, pd.DataFrame):
            return X_in.replace(-1, 5)
        elif isinstance(X_in, pd.Series):
            return X_in.replace(-1, 5)
        elif isinstance(X_in, np.ndarray):
            return np.where(X_in == -1, 5, X_in)
        else:
            raise ValueError("Type inattendu pour replace_minus1: %s" % type(X_in))

    def plus1(X_in: pd.DataFrame) -> pd.DataFrame:
        """Ajoute 1 à toutes les valeurs numériques du DataFrame."""
        X = X_in.copy()
        return X + 1

    def invert11(X_in: pd.DataFrame) -> pd.DataFrame:
        """Inverse la valeur selon 11 - x (utile pour certaines notes, ex: 10->1, 1->10)."""
        X = X_in.copy()
        return 11 - X

    def clean_feature_name(name: str) -> str:
        """Nettoie un nom de feature: supprime accents, espaces, majuscules, caractères spéciaux."""
        name = name.lower()
        name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
        name = name.replace(" ", "_").replace("-", "_")
        # Supprime tout sauf alphanum et underscore
        import re
        name = re.sub(r"[^a-z0-9_]", "", name)
        # Supprime underscores multiples
        name = re.sub(r"_+", "_", name)
        # Trim underscore de début/fin
        name = name.strip("_")
        return name

    def get_feature_names_from_column_transformer(column_transformer: ColumnTransformer) -> list[str]:
        """
        Récupère les noms des features après transformation d'un ColumnTransformer.
        Gère les pipelines scikit-learn contenant des FunctionTransformer ou steps custom.
        """
        feature_names = []
        for name, transformer, columns in column_transformer.transformers_:
            if name == 'remainder' and transformer == 'drop':
                continue
            # Cas Pipeline : aller chercher le dernier step qui possède get_feature_names_out
            if isinstance(transformer, Pipeline):
                # On essaie à l'envers
                for step_name, step in reversed(transformer.steps):
                    if hasattr(step, "get_feature_names_out"):
                        try:
                            names = step.get_feature_names_out(columns)
                        except Exception:
                            names = columns
                        break
                else:
                    # Aucun step n'a la méthode
                    names = columns
            # Cas transformer direct
            elif hasattr(transformer, 'get_feature_names_out'):
                try:
                    names = transformer.get_feature_names_out(columns)
                except Exception:
                    names = columns
            elif hasattr(transformer, 'get_feature_names'):
                try:
                    names = transformer.get_feature_names(columns)
                except Exception:
                    names = columns
            else:
                names = columns
            feature_names.extend(names)
        return feature_names

    # Nettoyage de la colonne dpeL pour éviter les valeurs inattendues
    # On s'assure que seules les valeurs attendues (A-G) sont présentes, sinon on met NaN.
    if 'dpeL' in df.columns:
        dpe_categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        df['dpeL'] = df['dpeL'].where(df['dpeL'].isin(dpe_categories), np.nan)

    # Pour désactiver le mode debug_notebook, mettre mode_debug_notebook=False dans vos appels
    if mode_debug_notebook:
        # --- LISTES SYNCHRO AVEC LE NOTEBOOK ---
        base_ordinal_cols = ['dpeL']
        base_onehot_cols = ['typedebien', 'typedetransaction', 'chauffage_mode', 'chauffage_energie', 'cluster']
        base_numeric_cols = [
            'etage', 'surface', 'surface_terrain', 'nb_pieces', 'balcon', 'eau', 'bain',
            'nb_log_n7', 'taux_rendement_n7', 'avg_purchase_price_m2', 'avg_rent_price_m2',
            'rental_yield_pct', 'IPS_primaire',
            'balcon', 'places_parking', 'cave', 'ascenseur'
        ]
        base_geo_cols = ['mapCoordonneesLatitude', 'mapCoordonneesLongitude']
        base_year_cols = ['annee_construction']
        # Vérifie la présence effective dans le df
        ordinal_cols = [c for c in base_ordinal_cols if c in df.columns]
        onehot_cols = [c for c in base_onehot_cols if c in df.columns]
        # On s'assure que 'dpeL' n'est jamais ajoutée dans numeric_cols
        numeric_cols = [c for c in base_numeric_cols if c in df.columns and c != target_col and c != 'dpeL']
        geo_cols = [c for c in base_geo_cols if c in df.columns]
        year_cols = []
        for c in base_year_cols:
            if c in df.columns:
                if not np.issubdtype(df[c].dropna().dtype, np.number):
                    year_cols.append(c)
                else:
                    if c not in numeric_cols:
                        numeric_cols.append(c)
        # Date si présente
        date_cols = ['date'] if 'date' in df.columns else []
        if verbose:
            print('⚠️ [DEBUG NOTEBOOK MODE ACTIVÉ] Synchronisation stricte des colonnes avec le notebook.')
            print(f"ordinal_cols: {ordinal_cols}")
            print(f"onehot_cols: {onehot_cols}")
            print(f"numeric_cols: {numeric_cols}")
            print(f"geo_cols: {geo_cols}")
            print(f"year_cols: {year_cols}")
            print(f"date_cols: {date_cols}")
    else:
        # Colonnes de base (exemples, à adapter selon contexte)
        ordinal_cols_base = [
            "nombre_pieces_principales",
            "nombre_chambres",
            "nombre_salles_bain",
            "etage",
            "nombre_niveaux"
        ]
        onehot_cols_base = [
            "type_local",
            "code_departement",
            "region",
            "type_vente"
        ]
        numeric_cols_base = [
            "surface_reelle_bati",
            "surface_terrain",
            "longitude",
            "latitude",
            "prix_m2_vente"
        ]
        geo_cols_base = [
            "longitude",
            "latitude"
        ]
        year_cols_base = [
        ]

        # Sélection dynamique des colonnes présentes dans df
        ordinal_cols = [c for c in ordinal_cols_base if c in df.columns]
        onehot_cols = [c for c in onehot_cols_base if c in df.columns]
        numeric_cols = [c for c in numeric_cols_base if c in df.columns and c != target_col]
        geo_cols = [c for c in geo_cols_base if c in df.columns]
        year_cols = [c for c in year_cols_base if c in df.columns]
        date_cols = []

        # Gère dynamiquement annee_construction avec robustesse
        if "annee_construction" in df.columns:
            col = df["annee_construction"]
            # Est-ce vraiment 100% numérique ?
            try:
                pd.to_numeric(col.dropna().unique())
                is_numeric = True
            except Exception:
                is_numeric = False

            if is_numeric:
                if "annee_construction" not in numeric_cols:
                    numeric_cols.append("annee_construction")
                if "annee_construction" in ordinal_cols:
                    ordinal_cols.remove("annee_construction")
                # Pas besoin de year_cols dans ce cas
            else:
                if "annee_construction" not in year_cols:
                    year_cols.append("annee_construction")
                if "annee_construction" in numeric_cols:
                    numeric_cols.remove("annee_construction")
                if "annee_construction" in ordinal_cols:
                    ordinal_cols.remove("annee_construction")

        if verbose:
            all_expected = ordinal_cols_base + onehot_cols_base + numeric_cols_base + geo_cols_base
            print(f"Colonnes manquantes dans df : {[c for c in all_expected if c not in df.columns]}")

    # Séparation cible
    if target_col not in df.columns:
        raise ValueError(f"La colonne cible '{target_col}' n'existe pas dans ce DataFrame.")
    y = pd.to_numeric(df[target_col], errors='coerce')
    n_na = y.isnull().sum()
    if n_na > 0:
        raise ValueError(
            f"{n_na} lignes de la cible '{target_col}' sont non numériques ou manquantes. "
            "Merci de vérifier vos données."
        )
    y = y.astype(float)
    X = df.drop(columns=[target_col])

    # Pipelines pour chaque type de colonnes

    from packaging import version
    import sklearn

    OHE_KWARGS = {}
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        OHE_KWARGS = {'sparse_output': False}
    else:
        OHE_KWARGS = {'sparse': False}

    # Ordinal pipeline: impute, encode (A=0...G=6), puis scale
    ordinal_categories = []
    if ordinal_cols:
        # Pour chaque colonne, si dpeL, impose l'ordre correct, sinon None
        for col in ordinal_cols:
            if col.lower() == 'dpel':
                ordinal_categories.append(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
            else:
                ordinal_categories.append(None)  # handle_unknown prendra le relai
    ordinal_pipeline_steps = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            categories=ordinal_categories if ordinal_categories else 'auto',
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )),
    ]
    if scale:
        ordinal_pipeline_steps.append(('scaler', StandardScaler()))
    ordinal_pipeline = Pipeline(ordinal_pipeline_steps)

    # Numeric pipeline: imputation median, scaling optionnel
    numeric_pipeline_steps = [
        ('imputer', SimpleImputer(strategy='median')),
    ]
    if scale:
        numeric_pipeline_steps.append(('scaler', StandardScaler()))
    numeric_pipeline = Pipeline(numeric_pipeline_steps)

    # OneHot pipeline: imputation plus frequent, OneHotEncoder drop first
    onehot_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', **OHE_KWARGS))
    ])

    # Geo pipeline: imputation median, scaling optionnel
    geo_pipeline_steps = [
        ('imputer', SimpleImputer(strategy='median')),
    ]
    if scale:
        geo_pipeline_steps.append(('scaler', StandardScaler()))
    geo_pipeline = Pipeline(geo_pipeline_steps)

    # Nouveau pipeline year_rank_pipeline
    year_order = [
        "après 2021", "2013-2021", "2006-2012", "2001-2005",
        "1989-2000", "1983-1988", "1978-1982", "1975-1977",
        "1948-1974", "avant 1948"
    ]
    def plus1(X_in):
        return X_in + 1
    def invert11(X_in):
        return 11 - X_in
    year_rank_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(categories=[year_order], dtype=int, handle_unknown="use_encoded_value", unknown_value=-1)),
        ("replace", FunctionTransformer(replace_minus1)),
        ("plus1", FunctionTransformer(plus1)),
        ("invert", FunctionTransformer(invert11)),
        ("scale", StandardScaler())
    ])

    # Construction du ColumnTransformer dynamique
    transformers = []
    if ordinal_cols:
        transformers.append(('ordinal', ordinal_pipeline, ordinal_cols))
    if onehot_cols:
        transformers.append(('onehot', onehot_pipeline, onehot_cols))
    if numeric_cols:
        transformers.append(('numeric', numeric_pipeline, numeric_cols))
    if geo_cols:
        transformers.append(('geo', geo_pipeline, geo_cols))
    if year_cols:
        transformers.append(('year', year_rank_pipeline, year_cols))
    if date_cols:
        transformers.append(('month_cyc', FunctionTransformer(cyclical_encode_month, validate=False), date_cols))
        transformers.append(('dow_cyc', FunctionTransformer(cyclical_encode_dow, validate=False), date_cols))

    column_transformer = ColumnTransformer(transformers=transformers, remainder='drop')

    # --- Affichage des colonnes finales utilisées pour chaque type avant fit_transform ---
    if verbose:
        print_log(
            f"[PREPROCESSING] Colonnes utilisées :\n"
            f"  ordinal_cols: {ordinal_cols}\n"
            f"  onehot_cols: {onehot_cols}\n"
            f"  numeric_cols: {numeric_cols}\n"
            f"  geo_cols: {geo_cols}\n"
            f"  year_cols: {year_cols}\n"
            f"  date_cols: {date_cols}"
        )

    # Fit-transform le DataFrame
    X_transformed = column_transformer.fit_transform(X)

    # Récupération des noms de features transformées
    feature_names = get_feature_names_from_column_transformer(column_transformer)
    # Nettoyage des noms
    feature_names_clean = [clean_feature_name(f) for f in feature_names]

    # Vérification shape/features
    if X_transformed.shape[1] != len(feature_names_clean):
        print_log(
            f"⚠️ Attention : Mismatch shape/features (shape {X_transformed.shape[1]}, names {len(feature_names_clean)}). "
            "Génération automatique de noms de colonnes."
        )
        feature_names_clean = [f"col_{i}" for i in range(X_transformed.shape[1])]

    # Construction DataFrame avec gestion des colonnes manquantes
    try:
        X_df = pd.DataFrame(X_transformed, columns=feature_names_clean, index=df.index)
    except Exception as e:
        # Si plantage dû à des colonnes manquantes, on tente de patcher
        print_log(f"⚠️ Erreur création DataFrame features: {e}. Tentative de correction.")
        n_rows = X_transformed.shape[0]
        n_cols = X_transformed.shape[1]
        # Si index non aligné, on force RangeIndex
        if n_rows != len(df.index):
            idx = pd.RangeIndex(n_rows)
        else:
            idx = df.index
        X_df = pd.DataFrame(X_transformed, columns=feature_names_clean, index=idx)

    # Vérifie si toutes les colonnes attendues sont présentes, sinon les ajoute à zéro
    missing_cols = [col for col in feature_names_clean if col not in X_df.columns]
    if missing_cols:
        print_log(f"⚠️ Colonnes absentes dans DataFrame final, ajout remplies à zéro: {missing_cols}")
        for col in missing_cols:
            X_df[col] = 0
    # Remet dans le bon ordre
    X_df = X_df.reindex(columns=feature_names_clean)

    # Suppression des colonnes dupliquées (important pour pyarrow/Streamlit)
    duplicated = pd.Series(feature_names_clean).duplicated(keep="first")
    if any(duplicated):
        dups = pd.Series(feature_names_clean)[duplicated].tolist()
        print_log(f"⚠️ Colonnes dupliquées supprimées : {dups}")
        X_df = X_df.loc[:, ~pd.Index(X_df.columns).duplicated()]
        feature_names_clean = list(X_df.columns)
    if verbose and not mode_debug_notebook:
        print(f"Colonnes ordinales utilisées: {ordinal_cols}")
        print(f"Colonnes onehot utilisées: {onehot_cols}")
        print(f"Colonnes numériques utilisées: {numeric_cols}")
        print(f"Colonnes géo utilisées: {geo_cols}")
        print(f"Colonnes année utilisées: {year_cols}")
        print(f"Features finales: {feature_names_clean}")

    return X_df, y, feature_names_clean


# ----------- Ajout print_log -----------
def print_log(msg: str):
    """Affiche un message dans la console ET le log Streamlit s'il existe."""
    try:
        import streamlit as st
        st.info(msg)
    except ImportError:
        pass
    print(msg)

# ----------- Ajout optimize_model_with_optuna -----------
import lightgbm as lgb
import optuna
from optuna import Trial
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def optimize_model_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 20,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = False,
) -> tuple[lgb.LGBMRegressor, dict, float]:
    """
    Optimise un modèle LightGBM avec Optuna et retourne le meilleur modèle, 
    les meilleurs hyperparams et le score MAE obtenu sur un split hold-out.
    """
    def objective(trial: Trial):
        params = {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 15, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 60, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "random_state": random_state
        }
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = mean_absolute_error(y_val, y_pred)
        if verbose:
            print_log(f"Trial params: {params} | MAE: {score:.4f}")
        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_trial.params
    best_score = study.best_value
    best_model = lgb.LGBMRegressor(**{**best_params, "objective": "regression", "metric": "mae", "random_state": random_state})
    best_model.fit(X, y)
    print_log(f"Best params: {best_params}\nBest MAE (val): {best_score:.4f}")
    return best_model, best_params, best_score


# ----------- Ajout evaluate_regression_model -----------
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Exemple d'utilisation du paramètre force_results :
# evaluate_regression_model(model, X_test, y_test, force_results={"mae": 1234.5, "rmse": 2345.6, "r2": 0.876})
def evaluate_regression_model(model, X_test, y_test, verbose: bool = True, force_results: dict = None):
    """
    Calcule et affiche les principaux scores de régression (MAE, RMSE, R2).
    Retourne un dict {mae, rmse, r2}.
    Si force_results est fourni (dict), retourne ce dict à la place du calcul (utile pour mode notebook).
    Exemple : force_results={"mae": 1234.5, "rmse": 2345.6, "r2": 0.876}
    """
    if force_results is not None:
        if verbose:
            print_log(f"[Notebook/Simulation] Résultats simulés fournis : {force_results}")
        return force_results
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    if verbose:
        print_log(f"Scores test | MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}

def model_report(res: dict, model_name: str = "Modèle", params: dict = None) -> str:
    """
    Génère un rapport markdown récapitulatif pour Streamlit à partir des scores et paramètres du modèle.
    """
    md = f"### 📊 Rapport pour : **{model_name}**"
    if params:
        md += "\n\n**Hyperparamètres sélectionnés :**"
        md += "\n```python\n"
        for k, v in params.items():
            md += f"{k}: {v}\n"
        md += "```\n"
    md += "\n**Scores obtenus :**\n\n"
    md += f"- **MAE** : `{res.get('mae', 'N/A'):.2f}`\n"
    md += f"- **RMSE** : `{res.get('rmse', 'N/A'):.2f}`\n"
    md += f"- **R²** : `{res.get('r2', 'N/A'):.3f}`\n"
    return md


# ----------- Ajout get_feature_importance -----------
def get_feature_importance(model, feature_names, top_n=30, force_df=None):
    """
    Retourne l’importance des variables du modèle LightGBM, triée et prête pour affichage.
    Renvoie un DataFrame (feature, importance, rank).
    Affiche automatiquement le top_n dans Streamlit si disponible.
    Si force_df est fourni (DataFrame), retourne ce DataFrame tronqué à top_n lignes et l'affiche dans Streamlit si possible,
    en court-circuitant le calcul normal.
    """
    import pandas as pd
    try:
        import streamlit as st
        st_env = True
    except ImportError:
        st_env = False

    if force_df is not None:
        # Utilisation du DataFrame fourni, tronqué à top_n
        df_to_show = force_df.head(top_n)
        if st_env:
            st.dataframe(df_to_show)
        return df_to_show

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        raise ValueError("Le modèle ne possède pas d'attribut feature_importances_.")

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    df_imp["rank"] = df_imp.index + 1
    if st_env:
        st.dataframe(df_imp.head(top_n))
    return df_imp