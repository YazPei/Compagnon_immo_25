from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, func
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import logging

from app.api.db.models import Property, Estimation, User
try:
    from app.api.db.schemas import PropertyCreate, EstimationCreate
except ImportError:
    # Fallback temporaire si schemas.py n'existe pas
    from pydantic import BaseModel
    
    class PropertyCreate(BaseModel):
        property_type: str
        surface: float
        postal_code: str
        price: Optional[float] = None
    
    class EstimationCreate(BaseModel):
        property_data: Dict[str, Any]
        estimated_price: float
        confidence_score: float
        model_name: Optional[str] = None

# Configuration du logger
logger = logging.getLogger(__name__)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning(
        "MLflow non disponible - "
        "fonctionnalités MLflow désactivées"
    )


class PropertyCRUD:
    """CRUD operations pour les propriétés."""
    
    @staticmethod
    def create(db: Session, property_data: PropertyCreate) -> Property:
        """Crée une nouvelle propriété."""
        db_property = Property(**property_data.dict())
        db.add(db_property)
        db.commit()
        db.refresh(db_property)
        return db_property
    
    @staticmethod
    def get_by_id(db: Session, property_id: int) -> Optional[Property]:
        """Récupère une propriété par son ID."""
        return db.query(Property).filter(Property.id == property_id).first()
    
    @staticmethod
    def get_by_filters(
        db: Session, 
        filters: Dict[str, Any] = None,
        limit: int = 100, 
        offset: int = 0
    ) -> List[Property]:
        """Récupère les propriétés avec filtres."""
        query = db.query(Property)
        
        if filters:
            if filters.get('property_type'):
                query = query.filter(
                    Property.property_type == filters['property_type']
                )
            if filters.get('min_surface'):
                query = query.filter(
                    Property.surface >= filters['min_surface']
                )
            if filters.get('max_surface'):
                query = query.filter(
                    Property.surface <= filters['max_surface']
                )
            if filters.get('postal_code'):
                query = query.filter(
                    Property.postal_code == filters['postal_code']
                )
        
        return query.offset(offset).limit(limit).all()


class EstimationCRUD:
    """CRUD pour les estimations avec intégration MLflow optionnelle."""
    
    @staticmethod
    def create_with_mlflow_tracking(
        db: Session, 
        estimation_data: EstimationCreate,
        model_name: str,
        model_version: str = None
    ) -> Estimation:
        """Crée une estimation et la track dans MLflow si disponible."""
        
        run_id = None
        
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run():
                    mlflow.log_param(
                        "property_type",
                        estimation_data.property_data.get("property_type")
                    )
                    mlflow.log_param(
                        "surface", 
                        estimation_data.property_data.get("surface")
                    )
                    mlflow.log_param(
                        "postal_code",
                        estimation_data.property_data.get("postal_code")
                    )
                    
                    mlflow.log_metric(
                        "estimated_price",
                        estimation_data.estimated_price
                    )
                    mlflow.log_metric(
                        "confidence_score", 
                        estimation_data.confidence_score
                    )
                    
                    mlflow.set_tag("model_name", model_name)
                    mlflow.set_tag("estimation_type", "user_request")
                    
                    # Récupérer l'ID du run MLflow
                    run_id = mlflow.active_run().info.run_id
                    
            except Exception as e:
                logger.warning(f"Erreur MLflow tracking: {e}")
        
        db_estimation = Estimation(
            property_data=json.dumps(estimation_data.property_data),
            estimated_price=estimation_data.estimated_price,
            confidence_score=estimation_data.confidence_score,
            mlflow_run_id=run_id,
            model_name=model_name,
            created_at=datetime.utcnow()
        )
        
        db.add(db_estimation)
        db.commit()
        db.refresh(db_estimation)
        
        logger.info(
            f"Estimation créée: ID={db_estimation.id}, "
            f"Prix={estimation_data.estimated_price}, "
            f"MLflow_ID={run_id}"
        )
        
        return db_estimation
    
    @staticmethod
    def create(db: Session, estimation_data: EstimationCreate) -> Estimation:
        """Crée une estimation simple sans MLflow."""
        db_estimation = Estimation(
            property_data=json.dumps(estimation_data.property_data),
            estimated_price=estimation_data.estimated_price,
            confidence_score=estimation_data.confidence_score,
            model_name=estimation_data.model_name or "default",
            created_at=datetime.utcnow()
        )
        
        db.add(db_estimation)
        db.commit()
        db.refresh(db_estimation)
        return db_estimation
    
    @staticmethod
    def get_by_id(db: Session, estimation_id: int) -> Optional[Estimation]:
        """Récupère une estimation par son ID."""
        return db.query(Estimation).filter(
            Estimation.id == estimation_id
        ).first()
    
    @staticmethod
    def get_with_mlflow_details(
        db: Session, 
        estimation_id: int
    ) -> Optional[Dict[str, Any]]:
        """Récupère une estimation avec ses détails MLflow."""
        estimation = db.query(Estimation).filter(
            Estimation.id == estimation_id
        ).first()
        
        if not estimation:
            return None
        
        mlflow_details = {}
        if MLFLOW_AVAILABLE and estimation.mlflow_run_id:
            try:
                run = mlflow.get_run(estimation.mlflow_run_id)
                mlflow_details = {
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags,
                    "artifacts": mlflow.list_artifacts(
                        estimation.mlflow_run_id
                    )
                }
            except Exception as e:
                logger.warning(f"Erreur récupération MLflow: {e}")
                mlflow_details = {"error": str(e)}
        
        return {
            "estimation": estimation,
            "mlflow_details": mlflow_details
        }
    
    @staticmethod
    def get_estimations(
        db: Session,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Estimation]:
        """Récupère les estimations avec filtres optionnels."""
        query = db.query(Estimation)
        
        if filters:
            if filters.get('date_debut'):
                query = query.filter(
                    Estimation.created_at >= filters['date_debut']
                )
            if filters.get('date_fin'):
                query = query.filter(
                    Estimation.created_at <= filters['date_fin']
                )
            if filters.get('model_name'):
                query = query.filter(
                    Estimation.model_name == filters['model_name']
                )
            if filters.get('min_price'):
                query = query.filter(
                    Estimation.estimated_price >= filters['min_price']
                )
            if filters.get('max_price'):
                query = query.filter(
                    Estimation.estimated_price <= filters['max_price']
                )
        
        return query.order_by(desc(Estimation.created_at))\
            .offset(offset)\
            .limit(limit)\
            .all()
    
    @staticmethod
    def count_estimations(
        db: Session, 
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Compte le nombre d'estimations avec filtres."""
        query = db.query(func.count(Estimation.id))
        
        if filters:
            if filters.get('date_debut'):
                query = query.filter(
                    Estimation.created_at >= filters['date_debut']
                )
            if filters.get('date_fin'):
                query = query.filter(
                    Estimation.created_at >= filters['date_fin']
                )
            if filters.get('model_name'):
                query = query.filter(
                    Estimation.model_name == filters['model_name']
                )
        
        return query.scalar()
    
    @staticmethod
    def get_estimation_stats(
        db: Session, 
        date_limite: datetime
    ) -> Dict[str, Any]:
        """Récupère les statistiques des estimations."""
        try:
            stats = db.query(
                func.count(Estimation.id).label('total'),
                func.avg(Estimation.estimated_price).label('prix_moyen')
            ).filter(
                Estimation.created_at >= date_limite
            ).first()
            
            # Calculer la médiane séparément (plus compatible)
            median_query = db.query(Estimation.estimated_price)\
                .filter(Estimation.created_at >= date_limite)\
                .order_by(Estimation.estimated_price)
            
            prices = [p[0] for p in median_query.all()]
            median_price = 0
            if prices:
                n = len(prices)
                if n % 2 == 0:
                    median_price = (prices[n//2-1] + prices[n//2]) / 2
                else:
                    median_price = prices[n//2]
            
            return {
                'total': stats.total or 0,
                'prix_moyen': float(stats.prix_moyen or 0),
                'prix_median': float(median_price),
                'repartition_types': {},  # À implémenter si nécessaire
                'repartition_villes': {},  # À implémenter si nécessaire
                'evolution_mensuelle': []  # À implémenter si nécessaire
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul statistiques: {e}")
            return {
                'total': 0,
                'prix_moyen': 0,
                'prix_median': 0,
                'repartition_types': {},
                'repartition_villes': {},
                'evolution_mensuelle': []
            }


class MLModelService:
    """Service pour interagir avec MLflow."""
    
    @staticmethod
    def get_active_model(model_name: str):
        """Récupère le modèle actif depuis MLflow."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow non disponible")
            return None
            
        try:
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(
                model_name, 
                stages=["Production"]
            )
            return latest_version[0] if latest_version else None
        except Exception as e:
            logger.error(f"Erreur récupération modèle: {e}")
            return None
    
    @staticmethod
    def get_model_metrics(
        model_name: str, 
        version: str = None
    ) -> Optional[Dict[str, Any]]:
        """Récupère les métriques d'un modèle depuis MLflow."""
        if not MLFLOW_AVAILABLE:
            return None
            
        try:
            client = mlflow.tracking.MlflowClient()
            if version:
                model_version = client.get_model_version(model_name, version)
                run = client.get_run(model_version.run_id)
                return run.data.metrics
            return None
        except Exception as e:
            logger.error(f"Erreur récupération métriques: {e}")
            return None
    
    @staticmethod
    def list_models() -> List[str]:
        """Liste tous les modèles disponibles."""
        if not MLFLOW_AVAILABLE:
            return []
            
        try:
            client = mlflow.tracking.MlflowClient()
            registered_models = client.list_registered_models()
            return [model.name for model in registered_models]
        except Exception as e:
            logger.error(f"Erreur listing modèles: {e}")
            return []


# Fonctions de commodité pour compatibilité avec historique.py
def get_estimations(
    db: Session,
    limit: int = 100,
    offset: int = 0,
    filters: Optional[Dict[str, Any]] = None
) -> List[Estimation]:
    """Fonction de commodité pour get_estimations."""
    return EstimationCRUD.get_estimations(db, limit, offset, filters)


def count_estimations(
    db: Session, 
    filters: Optional[Dict[str, Any]] = None
) -> int:
    """Fonction de commodité pour count_estimations."""
    return EstimationCRUD.count_estimations(db, filters)


def get_estimation_by_id(
    db: Session,
    estimation_id: str
) -> Optional[Estimation]:
    """Fonction de commodité pour get_estimation_by_id."""
    try:
        estimation_id_int = int(estimation_id)
        return EstimationCRUD.get_by_id(db, estimation_id_int)
    except ValueError:
        return None


def get_estimation_stats(
    db: Session, 
    date_limite: datetime
) -> Dict[str, Any]:
    """Fonction de commodité pour get_estimation_stats."""
    return EstimationCRUD.get_estimation_stats(db, date_limite)


# Instances globales
property_crud = PropertyCRUD()
estimation_crud = EstimationCRUD()
ml_model_service = MLModelService()