#!/bin/bash
# Script d'arrêt pour Airflow

echo "Arrêt des services Airflow..."

docker compose --profile airflow down

echo ""
echo "✓ Services Airflow arrêtés!"
echo ""
echo "Pour redémarrer: ./scripts/start-airflow.sh"
