import logging
import uuid
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd

from app.api.models.schemas import (EstimationRequest, EstimationResponse,
                                    EstimationResultModel, MarcheModel,
                                    MetadataModel)
from app.api.services.ml_service import ml_service
from app.api.utils.feature_enrichment import enrich_features_from_code_postal
from app.api.utils.geocoding import geocode_address
from app.api.utils.model_loader import get_model, get_preprocessor

logger = logging.getLogger(__name__)


def extract_features(request: EstimationRequest) -> Dict[str, Any]:
    """
    Extrait les features nécessaires à partir de la requête d'estimation.
    Si les coordonnées géographiques sont absentes, elles sont enrichies.
    """
    try:
        # Enrichissement géographique si nécessaire
        if (
            request.localisation.latitude is None
            or request.localisation.longitude is None
        ):
            coords = geocode_address(
                request.localisation.code_postal,
                request.localisation.ville,
                request.localisation.quartier,
            )
            request.localisation.latitude = coords["latitude"]
            request.localisation.longitude = coords["longitude"]

        features = {
            "surface": request.bien.surface,
            "nb_pieces": request.bien.nb_pieces,
            "nb_chambres": request.bien.nb_chambres,
            "etage": request.bien.etage or 0,
            "annee_construction": request.bien.annee_construction or 1980,
            # Ajouter d'autres features selon le modèle
        }
        logger.info(f"✅ Features extraites : {features}")
        return features
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'extraction des features : {str(e)}")
        raise


def prepare_features(estimation_request: EstimationRequest) -> pd.DataFrame:
    """
    Prépare les features pour le modèle ML.
    Enrichit les données avec des informations géographiques.
    """
    try:
        # Extraire les données de la requête
        bien = estimation_request.bien
        localisation = estimation_request.localisation
        transaction = estimation_request.transaction

        # Enrichir avec les données géographiques
        geo_features = enrich_features_from_code_postal(localisation.code_postal)

        # Créer un dictionnaire avec toutes les features
        features = {
            "surface": bien.surface,
            "nb_pieces": bien.nb_pieces,
            "nb_chambres": bien.nb_chambres,
            "etage": bien.etage or 0,
            "annee_construction": bien.annee_construction or 1980,
            "ascenseur": int(bien.ascenseur or 0),
            "balcon": int(bien.balcon or 0),
            "terrasse": int(bien.terrasse or 0),
            "parking": int(bien.parking or 0),
            "cave": int(bien.cave or 0),
            "x": geo_features.get("x", 0.0),
            "y": geo_features.get("y", 0.0),
            "cluster": geo_features.get("cluster", 0),
            "dpeL": geo_features.get("dpeL", "D"),
            "type_vente": 1 if transaction.type == "vente" else 0,
            "type_appartement": 1 if bien.type == "appartement" else 0,
            "type_maison": 1 if bien.type == "maison" else 0,
            "type_studio": 1 if bien.type == "studio" else 0,
            "type_loft": 1 if bien.type == "loft" else 0,
        }

        # Convertir en DataFrame
        df = pd.DataFrame([features])
        logger.info(f"✅ Features préparées pour le modèle : {features}")
        return df
    except Exception as e:
        logger.error(f"❌ Erreur lors de la préparation des features : {str(e)}")
        raise


def compute_estimation(estimation_request: EstimationRequest) -> EstimationResponse:
    """
    Calcule l'estimation d'un bien immobilier.
    """
    try:
        logger.info("🔄 Début du calcul de l'estimation...")

        # Préparer les features
        features_df = prepare_features(estimation_request)

        # Charger le modèle et le préprocesseur
        model = get_model()
        preprocessor = get_preprocessor()

        # Transformer les features
        features_processed = preprocessor.transform(features_df)

        # Assurer que c'est un array numpy 2D
        if isinstance(features_processed, list):
            features_processed = np.array(features_processed)
        if features_processed.ndim == 1:
            features_processed = features_processed.reshape(1, -1)

        # Faire la prédiction
        prediction = model.predict(features_processed)

        # Assurer que la prédiction est un float
        prix_estime = (
            float(prediction[0])
            if isinstance(prediction, (list, np.ndarray))
            else float(prediction)
        )

        # Calculer l'indice de confiance
        indice_confiance = 0.85  # Valeur par défaut, peut être ajustée dynamiquement

        # Créer les résultats d'estimation
        estimation_result = EstimationResultModel(
            prix=prix_estime,
            prix_min=prix_estime * 0.9,
            prix_max=prix_estime * 1.1,
            prix_m2=prix_estime / max(1, estimation_request.bien.surface),
            indice_confiance=indice_confiance,
        )

        # Créer les données du marché
        marche = MarcheModel(
            prix_moyen_quartier=prix_estime * 0.98,
            evolution_annuelle=2.5,
            delai_vente_moyen=45,
        )

        # Créer les métadonnées
        metadata = MetadataModel(
            id_estimation=f"est-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}",
            date_estimation=datetime.now().isoformat(),
            version_modele="1.0.0",
        )

        # Créer la réponse avec la structure attendue
        response = EstimationResponse(
            estimation=estimation_result,
            marche=marche,
            metadata=metadata,
        )

        logger.info(f"✅ Estimation calculée avec succès : {response}")
        return response

    except Exception as e:
        logger.error(f"❌ Erreur lors du calcul de l'estimation : {str(e)}")
        raise Exception(f"Erreur lors du calcul de l'estimation : {str(e)}")
