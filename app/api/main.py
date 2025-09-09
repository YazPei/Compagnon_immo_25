"""
Point d'entrée principal de l'application Compagnon Immobilier.
Ce fichier orchestre les différents services (API, ML, etc.).
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire racine au path pour permettre les imports
sys.path.append(str(Path(__file__).parent.parent))


def main():
    """Point d'entrée principal de l'application."""
    try:
        import uvicorn
        from api.main import app
        from api.config.settings import settings

        # Afficher les informations de démarrage
        print(f"🚀 Démarrage de l'application : {settings.PROJECT_NAME}")
        print(f"📍 URL : http://{settings.API_HOST}:{settings.API_PORT}")
        print(f"📚 Documentation : http://{settings.API_HOST}:{settings.API_PORT}/docs")

        # Lancer le serveur Uvicorn
        uvicorn.run(
            "api.main:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=settings.DEBUG,
            workers=1 if settings.DEBUG else settings.API_WORKERS,
            log_level=settings.LOG_LEVEL.lower()
        )

    except ImportError as e:
        print(f"❌ Erreur d'import : {e}")
        print("🔧 Vérifiez que toutes les dépendances sont installées.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur de démarrage : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()