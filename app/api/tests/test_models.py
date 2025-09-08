import pytest
from app.api.utils import model_loader


class TestModelLoader:
    """Tests pour le chargement des modèles et des préprocesseurs."""

    def test_get_model(self):
        """Test pour vérifier que le modèle est correctement chargé."""
        model = model_loader.get_model()
        assert model is not None, "Le modèle n'a pas pu être chargé."
        assert hasattr(model, "predict"), "Le modèle chargé ne possède pas de méthode 'predict'."

    def test_get_preprocessor(self):
        """Test pour vérifier que le préprocesseur est correctement chargé."""
        preprocessor = model_loader.get_preprocessor()
        assert preprocessor is not None, "Le préprocesseur n'a pas pu être chargé."
        assert hasattr(preprocessor, "transform"), "Le préprocesseur chargé ne possède pas de méthode 'transform'."

    def test_get_model_metadata(self):
        """Test pour vérifier que les métadonnées du modèle sont correctement chargées."""
        metadata = model_loader.get_model_metadata()
        assert metadata is not None, "Les métadonnées du modèle n'ont pas pu être chargées."
        assert isinstance(metadata, dict), "Les métadonnées du modèle doivent être un dictionnaire."
        assert "version" in metadata, "Les métadonnées doivent contenir une clé 'version'."
        assert "date_creation" in metadata, "Les métadonnées doivent contenir une clé 'date_creation'."