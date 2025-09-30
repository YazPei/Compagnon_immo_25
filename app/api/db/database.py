import logging
import os
from contextlib import contextmanager
from typing import Dict, Generator, Tuple

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import declarative_base, sessionmaker

# Charger les variables d'environnement en premier
load_dotenv()

# Importer settings après load_dotenv() pour s'assurer que les variables sont chargées
from app.api.config.settings import settings

# Configuration du logger
logger = logging.getLogger(__name__)

# Variables globales
engine = None
SessionLocal = None
Base = declarative_base()


def create_database_engine():
    """Crée et configure l'engine de base de données."""
    global engine

    if engine is not None:
        return engine

    try:
        # Configuration spécifique selon le type de base de données
        connect_args = {}
        pool_settings = {}

        if "sqlite" in settings.DATABASE_URL:
            connect_args = {"check_same_thread": False}
            pool_settings = {"poolclass": None}  # Pas de pool pour SQLite
        else:
            pool_settings = {
                "pool_pre_ping": True,
                "pool_recycle": 3600,
                "pool_size": 10,
                "max_overflow": 20,
            }

        engine = create_engine(
            settings.DATABASE_URL,
            connect_args=connect_args,
            echo=settings.DATABASE_ECHO,
            **pool_settings,
        )

        logger.info("Engine de base de données créé avec succès")
        return engine

    except Exception as e:
        logger.error(f"Erreur lors de la création de l'engine de base de données: {e}")
        raise RuntimeError(f"Impossible de créer l'engine de base de données: {e}")


def create_session_factory():
    """Crée la factory de sessions."""
    global SessionLocal

    if SessionLocal is not None:
        return SessionLocal

    if engine is None:
        create_database_engine()

    SessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=engine, expire_on_commit=False
    )

    logger.info("Factory de sessions créée avec succès")
    return SessionLocal


# Initialisation des composants
create_database_engine()
create_session_factory()


def get_db() -> Generator:
    """
    Générateur de session de base de données pour FastAPI dependency injection.

    Yields:
        Session: Session de base de données SQLAlchemy

    Raises:
        SQLAlchemyError: En cas d'erreur de base de données
    """
    if SessionLocal is None:
        raise RuntimeError("SessionLocal n'est pas initialisé")

    db = SessionLocal()
    try:
        yield db
    except SQLAlchemyError as e:
        logger.error(f"Erreur de base de données dans get_db: {e}")
        db.rollback()
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue dans get_db: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_context():
    """
    Context manager pour les sessions de base de données.

    Yields:
        Session: Session de base de données SQLAlchemy
    """
    if SessionLocal is None:
        raise RuntimeError("SessionLocal n'est pas initialisé")

    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Erreur dans le context manager de base de données: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> bool:
    """
    Initialise la base de données avec les tables.

    Returns:
        bool: True si l'initialisation a réussi

    Raises:
        RuntimeError: Si l'initialisation échoue
    """
    try:
        if engine is None:
            raise RuntimeError("Engine de base de données non initialisé")

        # Créer toutes les tables définies dans les modèles
        Base.metadata.create_all(bind=engine)
        logger.info("Base de données initialisée avec succès")
        return True

    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de la base de données: {e}")
        raise RuntimeError(f"Impossible d'initialiser la base de données: {e}")


def check_db_connection() -> Tuple[bool, str]:
    """
    Vérifie la connexion à la base de données.

    Returns:
        Tuple[bool, str]: (succès, message)
    """
    try:
        if SessionLocal is None:
            return False, "SessionLocal n'est pas initialisé"

        # Tester la connexion avec une requête simple
        with get_db_context() as db:
            result = db.execute(text("SELECT 1")).fetchone()

            if result and result[0] == 1:
                logger.info("Connexion à la base de données vérifiée avec succès")
                return True, "Connexion réussie"
            else:
                logger.warning("Résultat de requête de vérification invalide")
                return False, "Résultat de requête invalide"

    except SQLAlchemyError as e:
        error_msg = f"Erreur SQLAlchemy: {str(e)}"
        logger.error(f"Erreur de connexion à la base de données: {e}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Erreur inattendue: {str(e)}"
        logger.error(
            f"Erreur inattendue lors de la vérification de la base de données: {e}"
        )
        return False, error_msg


def get_db_info() -> Dict[str, str]:
    """
    Retourne des informations sur la base de données.

    Returns:
        Dict[str, str]: Informations sur la base de données
    """
    try:
        with get_db_context() as db:
            # Déterminer le type de base de données et obtenir sa version
            if "sqlite" in settings.DATABASE_URL.lower():
                db_type = "SQLite"
                version_result = db.execute(text("SELECT sqlite_version()")).fetchone()
                db_version = version_result[0] if version_result else "Unknown"
            elif "postgresql" in settings.DATABASE_URL.lower():
                db_type = "PostgreSQL"
                version_result = db.execute(text("SELECT version()")).fetchone()
                db_version = version_result[0] if version_result else "Unknown"
            elif "mysql" in settings.DATABASE_URL.lower():
                db_type = "MySQL"
                version_result = db.execute(text("SELECT VERSION()")).fetchone()
                db_version = version_result[0] if version_result else "Unknown"
            else:
                db_type = "Unknown"
                db_version = "Unknown"

            # Masquer les credentials dans l'URL
            masked_url = (
                settings.DATABASE_URL.split("://")[0] + "://***"
                if "://" in settings.DATABASE_URL
                else "***"
            )

            return {
                "database_type": db_type,
                "database_version": db_version,
                "database_url": masked_url,
                "connection_status": "Connected",
                "engine_echo": str(settings.DATABASE_ECHO),
            }

    except Exception as e:
        logger.error(
            (
                "Erreur lors de la récupération des informations de la base de données: "
                f"{e}"
            )
        )
        masked_url = (
            settings.DATABASE_URL.split("://")[0] + "://***"
            if "://" in settings.DATABASE_URL
            else "***"
        )

        return {
            "database_type": "Unknown",
            "database_version": "Unknown",
            "database_url": masked_url,
            "connection_status": f"Error: {str(e)}",
            "engine_echo": str(getattr(settings, "DATABASE_ECHO", False)),
        }


def close_db_connections():
    """Ferme toutes les connexions à la base de données."""
    global engine

    try:
        if engine is not None:
            engine.dispose()
            logger.info("Connexions à la base de données fermées")
    except Exception as e:
        logger.error(f"Erreur lors de la fermeture des connexions: {e}")


# Test de connexion au démarrage du module
def _test_connection_on_startup():
    """Test la connexion lors de l'import du module."""
    try:
        success, message = check_db_connection()
        if success:
            logger.info(f"✅ Test de connexion initial: {message}")
        else:
            logger.warning(f"⚠️ Test de connexion initial échoué: {message}")
    except Exception as e:
        logger.error(f"❌ Erreur lors du test de connexion initial: {e}")


# Exécuter le test de connexion seulement si ce fichier est importé
# (pas en __main__)
if __name__ != "__main__":
    _test_connection_on_startup()

# Test de connexion quand le fichier est exécuté directement
if __name__ == "__main__":
    print("=== Test de la configuration de base de données ===")

    # Test de connexion
    success, message = check_db_connection()
    if success:
        print(f"✅ Connexion: {message}")
    else:
        print(f"❌ Connexion: {message}")

    # Affichage des informations de la base de données
    db_info = get_db_info()
    print("\n=== Informations de la base de données ===")
    for key, value in db_info.items():
        print(f"{key}: {value}")

    # Test d'initialisation
    try:
        init_success = init_db()
        if init_success:
            print("✅ Initialisation de la base de données réussie")
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
