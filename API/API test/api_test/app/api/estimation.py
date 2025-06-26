from fastapi import APIRouter, Depends, HTTPException, status, Header
from api_test.app.models.schemas import EstimationRequest, EstimationResponse, ErrorResponse, TooManyRequestsResponse, QuestionnaireRequest, BienModel, LocalisationModel, TransactionModel
from api_test.app.security.auth import verify_api_key, get_api_key
from api_test.app.services.estimation_logic import compute_estimation
from api_test.app.services.estimation_service import estimate_property
from api_test.app.db.database import SessionLocal
from api_test.app.db.crud import save_estimation, get_estimation_by_id
from api_test.app.utils.feature_enrichment import enrich_features_from_code_postal
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import re
import joblib
from functools import lru_cache
from api_test.app.utils.model_loader import get_model, get_preprocessor, get_model_metadata
import os
from api_test.app.utils.geocoding import reverse_geocode

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/estimation", response_model=EstimationResponse, responses={
    400: {"model": ErrorResponse},
    401: {"model": ErrorResponse},
    429: {"model": TooManyRequestsResponse},
})
def create_estimation(
    estimation: EstimationRequest,
    x_api_key: str = Header(..., alias="X-API-Key"),
    api_key_valid: bool = Depends(verify_api_key),
    db=Depends(get_db)
):
    try:
        response = compute_estimation(estimation)
        save_estimation(db, response.metadata.id_estimation, estimation, response)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/estimation/{id_estimation}", response_model=EstimationResponse, responses={
    404: {"model": ErrorResponse},
    401: {"model": ErrorResponse},
})
def get_estimation(
    id_estimation: str,
    x_api_key: str = Header(..., alias="X-API-Key"),
    api_key_valid: bool = Depends(verify_api_key),
    db=Depends(get_db)
):
    db_obj = get_estimation_by_id(db, id_estimation)
    if not db_obj:
        raise HTTPException(status_code=404, detail="Estimation non trouvée")
    # On recompose la réponse à partir du JSON stocké
    return EstimationResponse(
        estimation=db_obj.estimation,
        marche=db_obj.marche,
        metadata=db_obj.estimation_metadata
    )

@router.post("/questionnaire_estimation", response_model=EstimationResponse)
def questionnaire_estimation(
    data: QuestionnaireRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Endpoint pour estimer un bien à partir du questionnaire client simplifié, enrichi automatiquement avec les features du CSV.
    """
    try:
        # Enrichissement automatique des features à partir du code postal
        enriched = enrich_features_from_code_postal(data.code_postal)
        debug_enriched_before = enriched.copy()
        # Fallback : si aucune donnée enrichie, on garde les valeurs du questionnaire
        if not enriched:
            enriched = {}
        # Nettoyage des valeurs vides ou 'nan' dans enriched (hors textuelles)
        for k in list(enriched.keys()):
            if k not in ['dpeL', 'exposition', 'etat_general']:
                if enriched[k] in [None, '', 'nan', 'NaN']:
                    enriched[k] = 0
        # --- LOGIQUE GÉOGRAPHIQUE ROBUSTE ---
        # 1. Priorité : coordonnées du payload
        lat_payload = getattr(data, 'latitude', None)
        lon_payload = getattr(data, 'longitude', None)
        if lat_payload is not None:
            enriched['mapCoordonneesLatitude'] = lat_payload
        if lon_payload is not None:
            enriched['mapCoordonneesLongitude'] = lon_payload
        # 2. Sinon, coordonnées du CSV d'enrichissement
        if enriched.get('mapCoordonneesLatitude', None) in [None, '', 'nan', 'NaN']:
            enriched['mapCoordonneesLatitude'] = enriched.get('y', None)
        if enriched.get('mapCoordonneesLongitude', None) in [None, '', 'nan', 'NaN']:
            enriched['mapCoordonneesLongitude'] = enriched.get('x', None)
        # 3. Fallback explicite : 0 si toujours rien
        if enriched.get('mapCoordonneesLatitude', None) in [None, '', 'nan', 'NaN']:
            enriched['mapCoordonneesLatitude'] = 0.0
        if enriched.get('mapCoordonneesLongitude', None) in [None, '', 'nan', 'NaN']:
            enriched['mapCoordonneesLongitude'] = 0.0
        # Ajout automatique de IPS_primaire à partir du code postal ou des coordonnées
        lat = getattr(data, 'latitude', None) or enriched.get('mapCoordonneesLatitude', None)
        lon = getattr(data, 'longitude', None) or enriched.get('mapCoordonneesLongitude', None)
        ips_val = enrich_ips_primaire(data.code_postal, lat, lon)
        if not ips_val:
            # Fallback : moyenne IPS si aucune info
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', '..', 'df_sales_clean_with_cluster.csv')
            if os.path.exists(csv_path):
                try:
                    df_ips = pd.read_csv(csv_path, sep=';', dtype=str)
                    df_ips["IPS_primaire"] = pd.to_numeric(df_ips["IPS_primaire"], errors="coerce")
                    ips_val = float(df_ips["IPS_primaire"].mean())
                except Exception:
                    ips_val = 0.0
            else:
                ips_val = 0.0
        enriched["IPS_primaire"] = ips_val
        # Ajout automatique de la date si absente
        if 'date' not in enriched or not enriched['date']:
            enriched['date'] = datetime.now().strftime('%Y-%m-%d')
        # Encodage cyclique de la date comme lors de l'entraînement
        def cyclical_encode_api(df):
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df["month"] = df["date"].dt.month
            df["dow"]   = df["date"].dt.weekday
            df["hour"]  = df["date"].dt.hour
            for col, period in [("month", 12), ("dow", 7), ("hour", 24)]:
                df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
                df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
                df.drop(columns=[col], inplace=True)
            df.drop(columns=["date"], inplace=True)
            return df
        enriched_df = pd.DataFrame([enriched])
        enriched_encoded = cyclical_encode_api(enriched_df)
        enriched = enriched_encoded.iloc[0].to_dict()
        # Ajout de la colonne 'date' sous forme de float (timestamp UNIX)
        enriched['date'] = float(datetime.now().timestamp())
        # --- Construction du dictionnaire de features harmonisé pour le pipeline ---
        pipeline_features = [
            'date', 'typedebien', 'typedetransaction', 'etage', 'surface', 'surface_terrain', 'nb_pieces', 'balcon', 'eau', 'bain',
            'dpeL', 'dpeC', 'mapCoordonneesLatitude', 'mapCoordonneesLongitude', 'nb_etages', 'places_parking', 'cave', 'exposition',
            'ges_class', 'annee_construction', 'nb_toilettes', 'porte_digicode', 'ascenseur', 'charges_copro', 'chauffage_energie',
            'chauffage_systeme', 'chauffage_mode', 'logement_neuf', 'INSEE_COM', 'loyer_m2_median_n6', 'nb_log_n6', 'taux_rendement_n6',
            'loyer_m2_median_n7', 'nb_log_n7', 'taux_rendement_n7', 'avg_purchase_price_m2', 'avg_rent_price_m2', 'rental_yield_pct',
            'IPS_primaire', 'split', 'codePostal', 'Year', 'Month', 'departement', 'zone_mixte', 'cluster', 'nom_cluster'
        ]
        features = {}
        # Les valeurs du questionnaire priment sur l'enrichissement
        for feat in pipeline_features:
            val = getattr(data, feat, None)
            if val is not None:
                features[feat] = val
            else:
                val = enriched.get(feat, 0)
                if val in [None, '', 'nan', 'NaN']:
                    val = 0
                features[feat] = val
        # Correction du type pour dpeL (ordinal)
        dpeL_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        if 'dpeL' in features:
            val = str(features['dpeL']).strip().upper()
            features['dpeL'] = dpeL_mapping.get(val, 0)
        # Gestion des valeurs manquantes ou NaN pour exposition et etat_general
        exposition_value = features.get('exposition')
        if not exposition_value or str(exposition_value).lower() == 'nan':
            exposition_value = 'NC'
        etat_general_value = features.get('etat_general')
        if not etat_general_value or str(etat_general_value).lower() == 'nan':
            etat_general_value = 'NC'
        # Préparation de la requête comme avant
        bien = BienModel(
            type=data.type_bien,
            surface=data.surface,
            nb_pieces=data.nb_pieces,
            nb_chambres=data.nb_chambres,
            etage=int(features.get('etage', 0)),
            annee_construction=data.annee_construction if data.annee_construction else int(features.get('annee_construction', 0)),
            etat_general=etat_general_value,
            exposition=exposition_value,
            ascenseur=data.ascenseur,
            balcon=data.balcon,
            parking=data.parking,
            cave=data.cave,
            gardien=data.gardien,
            piscine=data.piscine,
            terrasse=data.terrasse
        )
        localisation = LocalisationModel(
            code_postal=data.code_postal,
            ville=data.ville
        )
        transaction = TransactionModel(type="vente")  # Valeur par défaut
        req = EstimationRequest(
            bien=bien,
            localisation=localisation,
            transaction=transaction
        )
        # Nettoyage final : toute valeur '' dans features devient 0
        for k in features:
            if features[k] == '':
                features[k] = 0
        # Passage des features harmonisées à estimate_property
        return estimate_property(req, enriched_features=features, debug_info={
            "enriched_before": debug_enriched_before,
            "selected_features": pipeline_features
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Exemple d'utilisation du modèle et du préprocesseur dans l'API (utilitaire)
def predict_with_cached_model(input_data):
    model = get_model()
    preprocessor = get_preprocessor()
    X = preprocessor.transform([input_data])
    y_pred = model.predict(X)
    return float(y_pred[0])

@lru_cache(maxsize=1)
def get_ips_dataframe():
    df = pd.read_csv("X_train_encoded_for_api.csv", sep=';', usecols=["codePostal", "IPS_primaire"])
    # Conversion en string puis en float (tout ce qui n'est pas un nombre devient NaN)
    df["IPS_primaire"] = pd.to_numeric(df["IPS_primaire"].astype(str), errors="coerce")
    return df

def enrich_ips_primaire(code_postal=None, latitude=None, longitude=None):
    # 1. Si code postal fourni, on l'utilise
    if code_postal:
        code_postal_str = str(code_postal)
    # 2. Sinon, on tente de le trouver via lat/lon
    elif latitude is not None and longitude is not None:
        code_postal_str = reverse_geocode(latitude, longitude)
        if not code_postal_str:
            return 0.0
    else:
        return 0.0
    # 3. Mapping IPS dans df_sales_clean_with_cluster.csv
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', '..', 'df_sales_clean_with_cluster.csv')
    if not os.path.exists(csv_path):
        return 0.0
    try:
        df = pd.read_csv(csv_path, sep=';', dtype=str)
        row = df[df['codePostal'] == code_postal_str]
        if row.empty:
            return 0.0
        ips = row.iloc[0].get('IPS_primaire', None)
        return float(ips) if ips not in [None, '', 'nan', 'NaN'] else 0.0
    except Exception as e:
        print('DEBUG enrich_ips_primaire:', e)
        return 0.0
