#!/bin/bash
# Script de démarrage rapide pour Airflow (après initialisation)

set -e

echo "Démarrage des services Airflow..."

# Démarrer les services de base
docker compose up -d postgres-airflow redis mlflow

# Attendre un peu
sleep 5

# Démarrer Airflow
docker compose --profile airflow up -d

echo ""
echo "✓ Services Airflow démarrés!"
echo ""
echo "Interface Airflow: http://localhost:8081"
echo "MLflow UI: http://localhost:5050"
echo ""
echo "Voir les logs: docker compose logs -f airflow-webserver"
