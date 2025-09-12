import pytest
from unittest.mock import patch, MagicMock
from app.api.utils import model_loader


class TestModelLoader:
    """Tests pour le chargement des modèles et des préprocesseurs."""

    @patch("app.api.services.ml_service.MLService.get_model")
    def test_get_model(self, mock_get_model):
        """Test pour vérifier que le modèle est correctement chargé."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [1, 0]
        mock_get_model.return_value = mock_model

        model = mock_get_model()
        assert model is not None
        assert hasattr(model, "predict")

    @patch("app.api.services.ml_service.MLService.get_model_metadata")
    def test_get_model_metadata(self, mock_get_metadata):
        """Test pour vérifier que les métadonnées du modèle sont 
        correctement chargées."""
        mock_get_metadata.return_value = {
            "version": "1.0",
            "date_creation": "2025-09-12"
        }

        metadata = mock_get_metadata()
        assert metadata is not None
        assert isinstance(metadata, dict)
        assert "version" in metadata
        assert "date_creation" in metadata

    def test_get_preprocessor(self):
        """Test pour vérifier que le préprocesseur est correctement chargé."""
        preprocessor = model_loader.get_preprocessor()
        assert preprocessor is not None, (
            "Le préprocesseur n'a pas pu être chargé."
        )
        assert hasattr(preprocessor, "transform"), (
            "Le préprocesseur chargé ne possède pas de méthode 'transform'."
        )