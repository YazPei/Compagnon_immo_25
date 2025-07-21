# from api_test.app.services.estimation_service import model_loader
from app.services.ml_service import ml_service
from app.models.schemas import EstimationRequest, EstimationResponse, EstimationResultModel, MarcheModel, MetadataModel
from datetime import datetime
import uuid
from app.utils.geocoding import geocode_address

# Mock: transformation simple, à adapter selon le modèle réel

def extract_features(request: EstimationRequest):
    # Enrichissement géographique si besoin
    if request.localisation.latitude is None or request.localisation.longitude is None:
        coords = geocode_address(
            request.localisation.code_postal,
            request.localisation.ville,
            request.localisation.quartier
        )
        request.localisation.latitude = coords["latitude"]
        request.localisation.longitude = coords["longitude"]
    features = {
        'surface': request.bien.surface,
        'nb_pieces': request.bien.nb_pieces,
        'nb_chambres': request.bien.nb_chambres,
        'etage': request.bien.etage,
        'annee_construction': request.bien.annee_construction,
        # ... ajouter d'autres features selon le modèle
    }
    return features

def compute_estimation(request: EstimationRequest) -> EstimationResponse:
    features = extract_features(request)
    prix_dict = ml_service.predict_price(features)
    prix = float(prix_dict["estimation"]) if isinstance(prix_dict, dict) else float(prix_dict)
    prix_min = float(prix_dict["intervalle_confiance"]["min"]) if isinstance(prix_dict, dict) else prix * 0.95
    prix_max = float(prix_dict["intervalle_confiance"]["max"]) if isinstance(prix_dict, dict) else prix * 1.05
    prix_m2 = prix / max(1, request.bien.surface)
    indice_confiance = 87  # Mock, à calculer selon la logique réelle

    estimation = EstimationResultModel(
        prix=prix,
        prix_min=prix_min,
        prix_max=prix_max,
        prix_m2=prix_m2,
        indice_confiance=indice_confiance
    )

    marche = MarcheModel(
        prix_moyen_quartier=prix_m2 * 0.98,
        evolution_annuelle=ml_service.predict_evolution(features) if hasattr(ml_service, 'predict_evolution') else 0.0,
        delai_vente_moyen=45
    )

    metadata = MetadataModel(
        id_estimation=f"est-{datetime.now().strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:5]}",
        date_estimation=datetime.now().isoformat(),
        version_modele="v1.0"
    )

    return EstimationResponse(
        estimation=estimation,
        marche=marche,
        metadata=metadata
    ) 
