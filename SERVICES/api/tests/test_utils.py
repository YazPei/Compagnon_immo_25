from app.api.utils.feature_enrichment import validate_input
from app.api.utils.models_loader import get_model, get_preprocessor


def test_get_model():
    """
    Test pour vérifier que le modèle est correctement chargé.
    """
    model = get_model()
    assert model is not None


def test_get_preprocessor():
    """
    Test pour vérifier que le préprocesseur est correctement chargé.
    """
    preprocessor = get_preprocessor()
    assert preprocessor is not None


def test_validate_input():
    """
    Test pour valider les données d'entrée avec les champs corrects.
    """
    data = {
        "code_postal": "75001",
        "ville": "Paris",
        "quartier": "Louvre",
    }
    result = validate_input(data)
    assert result.code_postal == "75001"
    assert result.ville == "Paris"
    assert result.quartier == "Louvre"
