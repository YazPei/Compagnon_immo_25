from fastapi import APIRouter, Depends, HTTPException, Header
import pandas as pd
import numpy as np
import os
from datetime import datetime
from functools import lru_cache
import logging

from app.api.models.schemas import (
    EstimationRequest,
    EstimationResponse,
    ErrorResponse,
    TooManyRequestsResponse,
    QuestionnaireRequest,
    BienModel,
    LocalisationModel,
    TransactionModel,
)
from app.api.services.estimation_logic import compute_estimation
from app.api.services.estimation_service import estimate_property
from app.api.db.database import SessionLocal
from app.api.db.crud import save_estimation, get_estimation_by_id
from app.api.utils.feature_enrichment import enrich_features_from_code_postal
from app.api.security.auth import verify_api_key
from app.api.utils.model_loader import get_model, get_preprocessor
from app.api.utils.geocoding import reverse_geocode

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["estimation"])


def get_db():
    """Générateur de session base de données."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """Récupère et vérifie la clé API."""
    if not verify_api_key(x_api_key):
        raise HTTPException(
            status_code=401, 
            detail="Clé API invalide"
        )
    return x_api_key


@router.post(
    "/estimation", 
    response_model=EstimationResponse, 
    operation_id="create_estimation",
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        429: {"model": TooManyRequestsResponse},
    }
)
async def create_estimation(
    estimation: EstimationRequest,
    api_key: str = Depends(get_api_key),
    db=Depends(get_db)
):
    """Crée une nouvelle estimation immobilière."""
    try:
        logger.info(f"🔄 Début de création d'estimation pour {estimation}")
        response = compute_estimation(estimation)
        save_estimation(db, response.metadata.id_estimation, estimation, response)
        logger.info(f"✅ Estimation créée: {response.metadata.id_estimation}")
        return response
    except Exception as e:
        logger.error(f"❌ Erreur création estimation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/estimation/{id_estimation}", 
    response_model=EstimationResponse, 
    operation_id="get_estimation_by_id",
    responses={
        404: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
    }
)
async def get_estimation(
    id_estimation: str,
    api_key: str = Depends(get_api_key),
    db=Depends(get_db)
):
    """Récupère une estimation par son ID."""
    try:
        logger.info(f"🔄 Récupération de l'estimation {id_estimation}")
        db_obj = get_estimation_by_id(db, id_estimation)
        if not db_obj:
            logger.warning(f"⚠️ Estimation non trouvée: {id_estimation}")
            raise HTTPException(status_code=404, detail="Estimation non trouvée")
        
        logger.info(f"✅ Estimation récupérée: {id_estimation}")
        return EstimationResponse(
            estimation=db_obj.estimation,
            marche=db_obj.marche,
            metadata=db_obj.estimation_metadata
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur récupération estimation {id_estimation}: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne")


@router.post(
    "/questionnaire_estimation", 
    response_model=EstimationResponse,
    operation_id="questionnaire_estimation"
)
async def questionnaire_estimation(
    data: QuestionnaireRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Endpoint pour estimer un bien à partir du questionnaire client simplifié.
    Enrichissement automatique avec les features du CSV.
    """
    try:
        logger.info(f"🔄 Début estimation questionnaire pour CP: {data.code_postal}")
        
        # Enrichissement automatique des features
        enriched = enrich_features_from_code_postal(data.code_postal)
        debug_enriched_before = enriched.copy() if enriched else {}
        
        # Fallback si aucune donnée enrichie
        if not enriched:
            enriched = {}

        # Nettoyage des valeurs vides (hors champs textuels)
        textual_fields = ['dpeL', 'exposition', 'etat_general']
        for k in list(enriched.keys()):
            if k not in textual_fields:
                if enriched[k] in [None, '', 'nan', 'NaN', np.nan]:
                    enriched[k] = 0

        # Gestion des coordonnées géographiques
        enriched = _handle_coordinates(data, enriched)
        
        # Ajout de l'IPS primaire
        enriched["IPS_primaire"] = _get_ips_primaire(data, enriched)
        
        # Ajout de la date si absente
        if 'date' not in enriched or not enriched['date']:
            enriched['date'] = datetime.now().strftime('%Y-%m-%d')

        # Encodage cyclique de la date
        enriched = _encode_date_features(enriched)
        
        # Construction des features pour le pipeline
        features = _build_pipeline_features(data, enriched)
        
        # Préparation de la requête
        req = _build_estimation_request(data, features)
        
        logger.info("🔄 Appel à estimate_property")
        return estimate_property(req, enriched_features=features, debug_info={
            "enriched_before": debug_enriched_before,
            "selected_features": list(features.keys())
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur questionnaire estimation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


# Fonctions utilitaires pour enrichir et préparer les données
def _handle_coordinates(data: QuestionnaireRequest, enriched: dict) -> dict:
    """Gère la logique des coordonnées géographiques."""
    lat_payload = getattr(data, 'latitude', None)
    lon_payload = getattr(data, 'longitude', None)
    
    if lat_payload is not None:
        enriched['mapCoordonneesLatitude'] = lat_payload
    if lon_payload is not None:
        enriched['mapCoordonneesLongitude'] = lon_payload

    if enriched.get('mapCoordonneesLatitude') in [None, '', 'nan', 'NaN', np.nan]:
        enriched['mapCoordonneesLatitude'] = enriched.get('y', 0.0)
    if enriched.get('mapCoordonneesLongitude') in [None, '', 'nan', 'NaN', np.nan]:
        enriched['mapCoordonneesLongitude'] = enriched.get('x', 0.0)
        
    return enriched


def _get_ips_primaire(data: QuestionnaireRequest, enriched: dict) -> float:
    """Récupère l'IPS primaire à partir du code postal ou coordonnées."""
    lat = getattr(data, 'latitude', None) or enriched.get('mapCoordonneesLatitude')
    lon = getattr(data, 'longitude', None) or enriched.get('mapCoordonneesLongitude')
    
    ips_val = enrich_ips_primaire(data.code_postal, lat, lon)
    
    if not ips_val:
        try:
            csv_path = _get_csv_path()
            if os.path.exists(csv_path):
                df_ips = pd.read_csv(csv_path, sep=';', dtype=str)
                df_ips["IPS_primaire"] = pd.to_numeric(df_ips["IPS_primaire"], errors="coerce")
                ips_val = float(df_ips["IPS_primaire"].mean())
            else:
                ips_val = 0.0
        except Exception as e:
            logger.warning(f"⚠️ Erreur calcul moyenne IPS: {str(e)}")
            ips_val = 0.0
            
    return ips_val


def _encode_date_features(enriched: dict) -> dict:
    """Encode la date en features cycliques."""
    try:
        enriched_df = pd.DataFrame([enriched])
        enriched_df["date"] = pd.to_datetime(enriched_df["date"])
        
        enriched_df["month"] = enriched_df["date"].dt.month
        enriched_df["dow"] = enriched_df["date"].dt.weekday
        enriched_df["hour"] = enriched_df["date"].dt.hour
        
        for col, period in [("month", 12), ("dow", 7), ("hour", 24)]:
            enriched_df[f"{col}_sin"] = np.sin(2 * np.pi * enriched_df[col] / period)
            enriched_df[f"{col}_cos"] = np.cos(2 * np.pi * enriched_df[col] / period)
            enriched_df.drop(columns=[col], inplace=True)
        
        enriched_df['date'] = enriched_df['date'].astype('int64') // 10**9
        enriched_df.drop(columns=["date"], inplace=True, errors='ignore')
        
        return enriched_df.iloc[0].to_dict()
        
    except Exception as e:
        logger.error(f"⚠️ Erreur encodage date: {str(e)}")
        return enriched


def _build_pipeline_features(data: QuestionnaireRequest, enriched: dict) -> dict:
    """Construit le dictionnaire des features pour le pipeline."""
    pipeline_features = [
        'date', 'typedebien', 'typedetransaction', 'etage', 'surface', 'surface_terrain', 
        'nb_pieces', 'balcon', 'eau', 'bain', 'dpeL', 'dpeC', 'mapCoordonneesLatitude', 
        'mapCoordonneesLongitude', 'nb_etages', 'places_parking', 'cave', 'exposition',
        'ges_class', 'annee_construction', 'nb_toilettes', 'porte_digicode', 'ascenseur', 
        'charges_copro', 'chauffage_energie', 'chauffage_systeme', 'chauffage_mode', 
        'logement_neuf', 'INSEE_COM', 'loyer_m2_median_n6', 'nb_log_n6', 'taux_rendement_n6',
        'loyer_m2_median_n7', 'nb_log_n7', 'taux_rendement_n7', 'avg_purchase_price_m2', 
        'avg_rent_price_m2', 'rental_yield_pct', 'IPS_primaire', 'split', 'codePostal', 
        'Year', 'Month', 'departement', 'zone_mixte', 'cluster', 'nom_cluster'
    ]
    
    features = {}
    for feat in pipeline_features:
        val = getattr(data, feat, None)
        if val is not None:
            features[feat] = val
        else:
            val = enriched.get(feat, 0)
            if val in [None, '', 'nan', 'NaN', np.nan]:
                val = 0
            features[feat] = val

    dpeL_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    if 'dpeL' in features:
        val = str(features['dpeL']).strip().upper()
        features['dpeL'] = dpeL_mapping.get(val, 0)

    for k in features:
        if features[k] == '':
            features[k] = 0
            
    return features


def _build_estimation_request(data: QuestionnaireRequest, features: dict) -> EstimationRequest:
    """Construit la requête d'estimation."""
    exposition_value = features.get('exposition', 'NC')
    if not exposition_value or str(exposition_value).lower() == 'nan':
        exposition_value = 'NC'
        
    etat_general_value = features.get('etat_general', 'NC')
    if not etat_general_value or str(etat_general_value).lower() == 'nan':
        etat_general_value = 'NC'

    bien = BienModel(
        type=data.type_bien,
        surface=data.surface,
        nb_pieces=data.nb_pieces,
        nb_chambres=getattr(data, 'nb_chambres', 0),
        etage=int(features.get('etage', 0)),
        annee_construction=data.annee_construction or int(features.get('annee_construction', 0)),
        etat_general=etat_general_value,
        exposition=exposition_value,
        ascenseur=getattr(data, 'ascenseur', False),
        balcon=getattr(data, 'balcon', False),
        parking=getattr(data, 'parking', False),
        cave=getattr(data, 'cave', False),
        gardien=getattr(data, 'gardien', False),
        piscine=getattr(data, 'piscine', False),
        terrasse=getattr(data, 'terrasse', False)
    )

    localisation = LocalisationModel(
        code_postal=data.code_postal,
        ville=getattr(data, 'ville', '')
    )

    transaction = TransactionModel(type="vente")

    return EstimationRequest(
        bien=bien,
        localisation=localisation,
        transaction=transaction
    )


def _get_csv_path() -> str:
    """Retourne le chemin vers le CSV des clusters - compatible Docker."""
    from app.api.config.settings import settings
    
    # Utiliser les chemins configurés dans settings
    if hasattr(settings, 'DATA_DIR'):
        return settings.DATA_DIR / 'df_sales_clean_with_cluster.csv'
    
    # Fallback pour développement
    base_dir = Path(__file__).parent.parent.parent.parent
    return str(base_dir / 'data' / 'df_sales_clean_with_cluster.csv')

@lru_cache(maxsize=1)
def get_ips_dataframe():
    """Cache du dataframe IPS - compatible Docker."""
    try:
        from app.api.config.settings import settings
        
        if hasattr(settings, 'DATA_DIR'):
            csv_path = settings.DATA_DIR / 'X_train_encoded_for_api.csv'
        else:
            base_dir = Path(__file__).parent.parent.parent.parent
            csv_path = base_dir / 'data' / 'X_train_encoded_for_api.csv'
            
        df = pd.read_csv(csv_path, sep=';', usecols=["codePostal", "IPS_primaire"])
        df["IPS_primaire"] = pd.to_numeric(df["IPS_primaire"].astype(str), errors="coerce")
        return df
    except Exception as e:
        logger.error(f"Erreur chargement dataframe IPS: {str(e)}")
        return pd.DataFrame()


def enrich_ips_primaire(code_postal=None, latitude=None, longitude=None) -> float:
    """Enrichit l'IPS primaire à partir du code postal ou coordonnées."""
    try:
        # 1. Détermination du code postal
        if code_postal:
            code_postal_str = str(code_postal)
        elif latitude is not None and longitude is not None:
            code_postal_str = reverse_geocode(latitude, longitude)
            if not code_postal_str:
                return 0.0
        else:
            return 0.0

        # 2. Recherche dans le CSV
        csv_path = _get_csv_path()
        if not os.path.exists(csv_path):
            logger.warning(f"Fichier CSV non trouvé: {csv_path}")
            return 0.0

        df = pd.read_csv(csv_path, sep=';', dtype=str)
        row = df[df['codePostal'] == code_postal_str]
        
        if row.empty:
            logger.debug(f"Aucune donnée IPS pour CP: {code_postal_str}")
            return 0.0

        ips = row.iloc[0].get('IPS_primaire', None)
        return float(ips) if ips not in [None, '', 'nan', 'NaN'] else 0.0
        
    except Exception as e:
        logger.error(f"Erreur enrich_ips_primaire: {str(e)}")
        return 0.0


def predict_with_cached_model(input_data):
    """Utilise le modèle en cache pour faire une prédiction."""
    try:
        model = get_model()
        preprocessor = get_preprocessor()
        X = preprocessor.transform([input_data])
        y_pred = model.predict(X)
        return float(y_pred[0])
    except Exception as e:
        logger.error(f"Erreur prédiction: {str(e)}")
        raise