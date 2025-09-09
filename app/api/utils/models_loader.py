"""Module pour charger les modèles."""
import numpy as np

def get_model():
    """Retourne un modèle mock pour les tests."""
    class MockModel:
        def predict(self, X):
            # S'assurer que X est un array numpy
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            
            # Retourner une prédiction basée sur la surface (première feature)
            if X.ndim == 1:
                surface = X[0] if len(X) > 0 else 50
            else:
                surface = X[0, 0] if X.shape[1] > 0 else 50
            
            # Prix basé sur la surface * 5000€/m² (mock)
            return np.array([float(surface * 5000)])
    
    return MockModel()

def get_preprocessor():
    """Retourne un préprocesseur mock pour les tests."""
    class MockPreprocessor:
        @property
        def feature_names_in_(self):
            return [
                "surface", "nb_pieces", "nb_chambres", "etage", 
                "annee_construction", "ascenseur", "balcon", "terrasse",
                "parking", "cave", "x", "y", "cluster", "dpeL",
                "type_vente", "type_appartement", "type_maison", 
                "type_studio", "type_loft"
            ]
        
        def transform(self, X):
            # Si X est un DataFrame pandas
            if hasattr(X, 'values'):
                X_array = X.values
            else:
                X_array = np.array(X)
            
            # S'assurer que c'est 2D
            if X_array.ndim == 1:
                X_array = X_array.reshape(1, -1)
            
            # Normalisation mock (retourner les mêmes valeurs)
            return X_array
    
    return MockPreprocessor()

def get_model_metadata():
    """Retourne les métadonnées du modèle."""
    return {
        "feature_names": [
            "surface", "nb_pieces", "nb_chambres", "etage",
            "annee_construction", "ascenseur", "balcon", "terrasse",
            "parking", "cave", "x", "y", "cluster", "dpeL",
            "type_vente", "type_appartement", "type_maison",
            "type_studio", "type_loft"
        ],
        "version": "1.0.0",
        "model_type": "RandomForestRegressor"
    }