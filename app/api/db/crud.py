from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from datetime import datetime, timedelta
import json
import mlflow

from app.api.db.models import Property, Estimation 
from app.api.models.schemas import PropertyCreate, EstimationCreate

class EstimationCRUD:
    """CRUD pour les estimations avec intégration MLflow."""
    
    @staticmethod
    def create_with_mlflow_tracking(
        db: Session, 
        estimation_data: EstimationCreate,
        model_name: str,
        model_version: str = None
    ) -> models.Estimation:
        """Crée une estimation et la track dans MLflow."""
        
        # 1. Logger dans MLflow
        with mlflow.start_run():
            mlflow.log_param("property_type", estimation_data.property_data.get("property_type"))
            mlflow.log_param("surface", estimation_data.property_data.get("surface"))
            mlflow.log_param("postal_code", estimation_data.property_data.get("postal_code"))
            
            mlflow.log_metric("estimated_price", estimation_data.estimated_price)
            mlflow.log_metric("confidence_score", estimation_data.confidence_score)
            
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("estimation_type", "user_request")
            
            # Récupérer l'ID du run MLflow
            run_id = mlflow.active_run().info.run_id
        
        # 2. Sauvegarder en DB avec référence MLflow
        db_estimation = models.Estimation(
            property_data=json.dumps(estimation_data.property_data),
            estimated_price=estimation_data.estimated_price,
            confidence_score=estimation_data.confidence_score,
            mlflow_run_id=run_id,  # Référence vers MLflow
            model_name=model_name
        )
        
        db.add(db_estimation)
        db.commit()
        db.refresh(db_estimation)
        return db_estimation
    
    @staticmethod
    def get_with_mlflow_details(db: Session, estimation_id: int) -> dict:
        """Récupère une estimation avec ses détails MLflow."""
        estimation = db.query(models.Estimation).filter(
            models.Estimation.id == estimation_id
        ).first()
        
        if not estimation:
            return None
        
        # Récupérer les détails depuis MLflow
        mlflow_details = {}
        if estimation.mlflow_run_id:
            try:
                run = mlflow.get_run(estimation.mlflow_run_id)
                mlflow_details = {
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags,
                    "artifacts": mlflow.list_artifacts(estimation.mlflow_run_id)
                }
            except Exception as e:
                print(f"Erreur MLflow: {e}")
        
        return {
            "estimation": estimation,
            "mlflow_details": mlflow_details
        }

class MLModelService:
    """Service pour interagir avec MLflow (remplace MLModelCRUD)."""
    
    @staticmethod
    def get_active_model(model_name: str):
        """Récupère le modèle actif depuis MLflow."""
        try:
            client = mlflow.tracking.MlflowClient()
            latest_version = client.get_latest_versions(
                model_name, 
                stages=["Production"]
            )
            return latest_version[0] if latest_version else None
        except Exception as e:
            print(f"Erreur lors de la récupération du modèle: {e}")
            return None
    
    @staticmethod
    def get_model_metrics(model_name: str, version: str = None):
        """Récupère les métriques d'un modèle depuis MLflow."""
        try:
            client = mlflow.tracking.MlflowClient()
            if version:
                model_version = client.get_model_version(model_name, version)
                run = client.get_run(model_version.run_id)
                return run.data.metrics
            return None
        except Exception as e:
            print(f"Erreur lors de la récupération des métriques: {e}")
            return None

# Instances
property_crud = PropertyCRUD()
estimation_crud = EstimationCRUD()
ml_model_service = MLModelService()