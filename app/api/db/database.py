from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
import logging
import os
from dotenv import load_dotenv
from app.api.config.settings import settings

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration base de données
try:
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
        echo=settings.DATABASE_ECHO,  # Utiliser DATABASE_ECHO au lieu de DEBUG
        pool_pre_ping=True,  # Vérification automatique des connexions
        pool_recycle=3600 if "sqlite" not in settings.DATABASE_URL else -1
    )
except Exception as e:
    logger.error(f"Erreur lors de la création de l'engine de base de données: {e}")
    raise

# SessionLocal pour les sessions de base de données
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base pour les modèles
Base = declarative_base()

def get_db():
    """Générateur de session de base de données pour FastAPI dependency injection."""
    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Erreur de base de données: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def init_db():
    """Initialise la base de données avec les tables."""
    try:
        # Créer toutes les tables définies dans les modèles
        Base.metadata.create_all(bind=engine)
        logger.info("Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de la base de données: {e}")
        raise

def check_db_connection() -> tuple[bool, str]:
    """Vérifie la connexion à la base de données."""
    try:
        # Tenter une requête simple
        with SessionLocal() as db:
            result = db.execute(text("SELECT 1")).fetchone()
            
            if result and result[0] == 1:
                logger.info("Connexion à la base de données vérifiée avec succès")
                return True, "Connexion réussie"
            else:
                logger.warning("Résultat de requête de vérification invalide")
                return False, "Résultat de requête invalide"
                
    except SQLAlchemyError as e:
        logger.error(f"Erreur de connexion à la base de données: {e}")
        return False, f"Erreur SQLAlchemy: {str(e)}"
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la vérification de la base de données: {e}")
        return False, f"Erreur inattendue: {str(e)}"

def get_db_info() -> dict:
    """Retourne des informations sur la base de données."""
    try:
        with SessionLocal() as db:
            # Obtenir des informations sur la base de données
            if "sqlite" in settings.DATABASE_URL:
                db_type = "SQLite"
                db_version = db.execute(text("SELECT sqlite_version()")).fetchone()[0]
            elif "postgresql" in settings.DATABASE_URL:
                db_type = "PostgreSQL"
                db_version = db.execute(text("SELECT version()")).fetchone()[0]
            else:
                db_type = "Unknown"
                db_version = "Unknown"
            
            return {
                "database_type": db_type,
                "database_version": db_version,
                "database_url": settings.DATABASE_URL.split("://")[0] + "://***",  # Masquer les credentials
                "connection_status": "Connected"
            }
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des informations de la base de données: {e}")
        return {
            "database_type": "Unknown",
            "database_version": "Unknown",
            "database_url": settings.DATABASE_URL.split("://")[0] + "://***",
            "connection_status": f"Error: {str(e)}"
        }

# Test de connexion au démarrage
if __name__ == "__main__":
    success, message = check_db_connection()
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")