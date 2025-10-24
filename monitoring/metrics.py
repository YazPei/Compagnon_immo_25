from prometheus_client import Counter, Histogram, Gauge, Info
import time
import numpy as np
from typing import Dict, Any, List
import os

# Métriques API
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

# Métriques ML
model_predictions_total = Counter(
    'model_predictions_total',
    'Total model predictions',
    ['model_type', 'model_version']
)

model_prediction_duration_seconds = Histogram(
    'model_prediction_duration_seconds',
    'Model prediction latency',
    ['model_type']
)

model_prediction_confidence = Histogram(
    'model_prediction_confidence',
    'Model prediction confidence score',
    ['model_type'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

model_drift_score = Gauge(
    'model_drift_score',
    'Model drift detection score',
    ['model_type']
)

model_accuracy = Gauge(
    'model_accuracy',
    'Current model accuracy',
    ['model_type', 'dataset']
)

# Métriques de données
data_freshness_timestamp = Gauge(
    'data_freshness_timestamp',
    'Timestamp of last data update',
    ['data_source']
)

data_quality_score = Gauge(
    'data_quality_score',
    'Data quality score (0-1)',
    ['data_source', 'quality_check']
)

data_volume = Gauge(
    'data_volume',
    'Number of records processed',
    ['data_source', 'time_period']
)

# Métriques de déploiement
deployment_info = Info(
    'deployment_info',
    'Deployment information'
)

deployment_timestamp = Gauge(
    'deployment_timestamp',
    'Deployment timestamp'
)

# Métriques de sécurité
security_events_total = Counter(
    'security_events_total',
    'Security events',
    ['event_type', 'severity']
)

rate_limit_hits_total = Counter(
    'rate_limit_hits_total',
    'Rate limit hits',
    ['endpoint', 'client_id']
)

class MetricsCollector:
    """Collecteur de métriques centralisé."""
    
    def __init__(self):
        self.setup_deployment_info()
    
    def setup_deployment_info(self):
        """Configurer les informations de déploiement."""
        deployment_info.info({
            'version': os.getenv('GIT_COMMIT', 'unknown'),
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'build_date': os.getenv('BUILD_DATE', 'unknown')
        })
        deployment_timestamp.set(time.time())
    
    def record_prediction(self, model_type: str, confidence: float, 
                         duration: float, model_version: str = 'v1'):
        """Enregistrer une prédiction."""
        model_predictions_total.labels(
            model_type=model_type, 
            model_version=model_version
        ).inc()
        
        model_prediction_duration_seconds.labels(
            model_type=model_type
        ).observe(duration)
        
        model_prediction_confidence.labels(
            model_type=model_type
        ).observe(confidence)
    
    def update_model_drift(self, model_type: str, drift_score: float):
        """Mettre à jour le score de drift."""
        model_drift_score.labels(model_type=model_type).set(drift_score)
    
    def update_data_quality(self, data_source: str, quality_checks: Dict[str, float]):
        """Mettre à jour les scores de qualité des données."""
        for check_name, score in quality_checks.items():
            data_quality_score.labels(
                data_source=data_source,
                quality_check=check_name
            ).set(score)
    
    def update_data_freshness(self, data_source: str):
        """Mettre à jour la fraîcheur des données."""
        data_freshness_timestamp.labels(data_source=data_source).set(time.time())
    
    def record_security_event(self, event_type: str, severity: str = 'medium'):
        """Enregistrer un événement de sécurité."""
        security_events_total.labels(
            event_type=event_type,
            severity=severity
        ).inc()

# Instance globale
metrics = MetricsCollector()

def calculate_drift_score(reference_data: np.ndarray, 
                         current_data: np.ndarray) -> float:
    """Calculer le score de drift entre deux datasets."""
    from scipy.stats import ks_2samp
    
    # Test de Kolmogorov-Smirnov
    statistic, p_value = ks_2samp(reference_data, current_data)
    
    # Score de drift basé sur la statistique KS
    drift_score = min(statistic * 2, 1.0)
    
    return drift_score

def evaluate_data_quality(data: Dict[str, Any]) -> Dict[str, float]:
    """Évaluer la qualité des données."""
    quality_scores = {}
    
    # Complétude
    if 'completeness' in data:
        non_null_ratio = 1 - (data['null_count'] / data['total_count'])
        quality_scores['completeness'] = non_null_ratio
    
    # Cohérence
    if 'consistency_checks' in data:
        quality_scores['consistency'] = data['consistency_score']
    
    # Validité
    if 'validation_results' in data:
        quality_scores['validity'] = data['validation_score']
    
    return quality_scores
