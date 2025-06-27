"""
Service pour gérer les estimations immobilières.
"""
from datetime import datetime
import uuid
from typing import Dict, Any
import pandas as pd
import os

from api_test.app.models.schemas import (
    EstimationRequest,
    EstimationResponse,
    EstimationResultModel,
    MarcheModel,
    MetadataModel
)
from api_test.app.services.ml_service import ml_service

# Chargement du CSV de marché local (une seule fois)
DF_SALES_PATH = os.path.join(os.path.dirname(__file__), '../../df_sales_clean_with_cluster.csv')
if os.path.exists(DF_SALES_PATH):
    df_sales = pd.read_csv(DF_SALES_PATH, sep=';')
else:
    df_sales = pd.DataFrame()

def get_local_market_info(code_postal, ville, cluster, latitude=None, longitude=None):
    # Filtrage intelligent : ville > code postal > cluster > coordonnées géographiques
    df_local = pd.DataFrame()
    if ville and ville in df_sales.get('ville', []):
        df_local = df_sales[df_sales['ville'] == ville]
    # Recherche robuste du code postal
    code_postal_col = None
    for col in df_sales.columns:
        if col.lower().replace('_','').replace(' ','') == 'codepostal':
            code_postal_col = col
            break
    if df_local.empty and code_postal and code_postal_col:
        df_local = df_sales[df_sales[code_postal_col].astype(str) == str(code_postal)]
    # Recherche robuste du cluster
    cluster_col = None
    for col in df_sales.columns:
        if col.lower().replace('_','').replace(' ','') == 'cluster':
            cluster_col = col
            break
    if df_local.empty and cluster is not None and cluster_col:
        df_local = df_sales[df_sales[cluster_col] == cluster]
    # Recherche par coordonnées géographiques si tout est vide
    if df_local.empty and latitude is not None and longitude is not None:
        # On cherche la ligne la plus proche en distance euclidienne
        if 'y' in df_sales.columns and 'x' in df_sales.columns:
            try:
                df_sales['distance'] = ((df_sales['y'].astype(float) - float(latitude))**2 + (df_sales['x'].astype(float) - float(longitude))**2)**0.5
                idx_min = df_sales['distance'].idxmin()
                df_local = df_sales.loc[[idx_min]]
            except Exception:
                pass
    if df_local.empty:
        return {
            "prix_m2_moyen": 0.0,
            "evolution_annuelle": 0.0,
            "delai_vente_moyen": 45
        }
    # Prix moyen au m²
    prix_m2_moyen = float(df_local['prix_m2_vente'].mean()) if 'prix_m2_vente' in df_local else 0.0
    # Évolution annuelle
    if 'date' in df_local and 'prix_m2_vente' in df_local:
        df_local['date'] = pd.to_datetime(df_local['date'], errors='coerce')
        df_local = df_local.sort_values('date')
        df_jan = df_local[df_local['date'].dt.month == 1]
        if len(df_jan) >= 2:
            prix_start = df_jan.iloc[0]['prix_m2_vente']
            prix_end = df_jan.iloc[-1]['prix_m2_vente']
            if prix_start and prix_end and prix_start > 0:
                evolution_annuelle = float(100 * (prix_end - prix_start) / prix_start)
            else:
                evolution_annuelle = 0.0
        else:
            evolution_annuelle = 0.0
    else:
        evolution_annuelle = 0.0
    # Délai de vente moyen (exemple : valeur par défaut ou moyenne cluster)
    delai_vente_moyen = 45
    return {
        "prix_m2_moyen": prix_m2_moyen,
        "evolution_annuelle": evolution_annuelle,
        "delai_vente_moyen": int(delai_vente_moyen)
    }

def generate_estimation_id() -> str:
    """Génère un ID unique pour l'estimation."""
    return str(uuid.uuid4())

def determine_cluster(localisation: Dict[str, Any]) -> int:
    """
    Détermine le cluster SARIMAX à utiliser à partir de la localisation.
    Pour l'instant, mock : cluster 0 si code postal commence par 75, sinon 1.
    """
    code_postal = localisation.get("code_postal", "")
    if str(code_postal).startswith("75"):
        return 0
    return 1

def estimate_property(request: EstimationRequest, enriched_features: dict = None, debug_info: dict = None) -> EstimationResponse:
    """
    Estime un bien immobilier avec enrichissement automatique des features si fourni.
    Args:
        request: Requête d'estimation
        enriched_features: Dictionnaire de features enrichies (optionnel)
        debug_info: Dictionnaire d'informations de debug à inclure dans la réponse (optionnel)
    Returns:
        Réponse d'estimation complète
    """
    estimation_id = generate_estimation_id()
    try:
        # Préparer les données pour le modèle
        input_data = {
            **request.bien.model_dump(),
            **request.localisation.model_dump(),
            **request.transaction.model_dump()
        }
        # Fusionner avec les features enrichies si présentes
        if enriched_features:
            input_data.update({k: v for k, v in enriched_features.items() if v is not None})
        # Prédiction de prix
        price_prediction = ml_service.predict_price(input_data)
        # Correction : extraction stricte des valeurs float
        estimation_val = float(price_prediction["estimation"])
        prix_m2_val = float(price_prediction.get("prix_m2", 0))
        intervalle = price_prediction.get("intervalle_confiance", {"min": estimation_val*0.95, "max": estimation_val*1.05})
        prix_min = float(intervalle["min"]) if isinstance(intervalle, dict) else estimation_val*0.95
        prix_max = float(intervalle["max"]) if isinstance(intervalle, dict) else estimation_val*1.05
        features_utilisees_prix = input_data.copy()
        # Détermination du cluster SARIMAX
        cluster_id = determine_cluster(request.localisation.model_dump())
        # Sélection dynamique des exogènes selon le cluster
        exog_vars_by_cluster = {
            0: ["dpeL"],
            1: ["x_geo", "z_geo"],
            2: ["x_geo", "y_geo", "dpeL"],
            3: ["taux_rendement_n7", "loyer_m2_median_n7", "y_geo", "dpeL"]
        }
        exog_vars = exog_vars_by_cluster.get(cluster_id, [])
        exog = None
        exog_used = {}
        if enriched_features and exog_vars:
            exog_dict = {}
            for var in exog_vars:
                val = enriched_features.get(var, 0)
                if var not in enriched_features or val in [None, '', 'nan', 'NaN']:
                    import warnings
                    warnings.warn(f"Exogène '{var}' absente ou nulle, valeur 0 utilisée.")
                    val = 0
                exog_dict[var] = [val] * 12
                exog_used[var] = val
            exog = pd.DataFrame(exog_dict)
            print("DEBUG exog DataFrame shape:", exog.shape)
        market_trend = ml_service.predict_trend({
            "code_postal": request.localisation.code_postal,
            "type_bien": request.bien.type,
            "start_date": datetime.now().strftime("%Y-%m-%d"),
            "periods": 12,
            "exog": exog
        }, cluster_id=cluster_id)
        print('DEBUG estimation_val:', estimation_val)
        print('DEBUG prix_min:', prix_min)
        print('DEBUG prix_max:', prix_max)
        surface = getattr(request.bien, 'surface', None)
        try:
            prix_m2 = float(estimation_val) / float(surface) if surface else 0.0
        except Exception:
            prix_m2 = 0.0
        # Calcul dynamique de l'indice de confiance basé sur l'intervalle de confiance
        interval_width = prix_max - prix_min
        if estimation_val > 0:
            indice_confiance = max(0, min(100, 100 - (interval_width / estimation_val) * 100))
        else:
            indice_confiance = 0
        print("DEBUG market_trend:", market_trend)
        # Calcul du prix moyen au m² dans le quartier (moyenne sur la période)
        if market_trend["tendance"]:
            predictions = [mois["prediction"] for mois in market_trend["tendance"] if mois.get("prediction") is not None]
            print("DEBUG predictions:", predictions)
            if predictions:
                prix_moyen_quartier = sum(predictions) / len(predictions)
            else:
                prix_moyen_quartier = 0
        else:
            print("DEBUG Pas de tendance retournée par le modèle SARIMAX")
            prix_moyen_quartier = 0
        # Récupération des infos locales dynamiques
        code_postal = getattr(request.localisation, 'code_postal', None)
        ville = getattr(request.localisation, 'ville', None)
        latitude = getattr(request.localisation, 'latitude', None)
        longitude = getattr(request.localisation, 'longitude', None)
        local_info = get_local_market_info(code_postal, ville, cluster_id, latitude, longitude)
        prix_moyen_quartier = local_info["prix_m2_moyen"]
        evolution_annuelle = local_info["evolution_annuelle"]
        delai_vente_moyen = local_info["delai_vente_moyen"]
        estimation = EstimationResultModel(
            prix=float(estimation_val),
            prix_min=float(prix_min),
            prix_max=float(prix_max),
            prix_m2=prix_m2_val,
            indice_confiance=int(indice_confiance)
        )
        marche = MarcheModel(
            prix_moyen_quartier=prix_moyen_quartier,
            evolution_annuelle=evolution_annuelle,
            delai_vente_moyen=delai_vente_moyen
        )
        metadata = MetadataModel(
            id_estimation=estimation_id,
            date_estimation=datetime.now().isoformat(),
            version_modele="1.0.0"
        )
        response = EstimationResponse(
            estimation=estimation,
            marche=marche,
            metadata=metadata,
            explications={
                "prix": "Ceci est le prix estimé de votre bien, calculé à partir des caractéristiques fournies et de notre modèle d'intelligence artificielle.",
                "prix_min": "C'est le prix le plus bas estimé, tenant compte d'une marge d'incertitude. Il est peu probable que votre bien se vende en dessous de ce montant.",
                "prix_max": "C'est le prix le plus haut estimé, tenant compte d'une marge d'incertitude. Il est rare qu'un bien similaire se vende au-dessus de ce montant.",
                "prix_m2": "Il s'agit du prix estimé au mètre carré, utile pour comparer avec d'autres biens dans la même zone.",
                "indice_confiance": "Cet indice (sur 100) reflète la fiabilité de l'estimation : plus il est élevé, plus l'estimation est jugée fiable.",
                "prix_moyen_quartier": "Prix moyen au mètre carré observé dans votre quartier, selon les dernières tendances du marché.",
                "evolution_annuelle": "Variation estimée des prix sur un an dans votre secteur. Une valeur positive indique une hausse, une valeur négative une baisse.",
                "delai_vente_moyen": "Nombre moyen de jours pour vendre un bien similaire dans votre quartier.",
                "exogenes_utilisees": "Liste des variables externes utilisées pour la prévision de tendance du marché, selon votre cluster.",
                "features_utilisees_prix": "Voici toutes les caractéristiques effectivement transmises au modèle de prix. Vérifiez que les valeurs sont cohérentes et réalistes."
            },
            exogenes_utilisees=exog_used,
            features_utilisees_prix=features_utilisees_prix
        )
        if debug_info:
            response.debug_info = debug_info
        return response
    except Exception as e:
        raise ValueError(f"Erreur lors de l'estimation: {str(e)}")

def calculate_market_index(market_trend: Dict[str, Any]) -> float:
    """Calcule l'indice de variation du marché."""
    # TODO: Implémenter le calcul réel
    # Pour l'instant, retourne une valeur mock
    return 0.05  # 5% de variation positive

def calculate_reliability_score(
    price_prediction: Dict[str, Any],
    market_trend: Dict[str, Any]
) -> float:
    """Calcule le score de fiabilité de l'estimation."""
    # TODO: Implémenter le calcul réel
    # Pour l'instant, retourne une valeur mock
    return 0.85  # 85% de fiabilité 