#!/bin/bash
# Script d'initialisation Airflow pour Compagnon Immo
# Ce script configure les permissions et initialise Airflow

set -e  # Arrêter en cas d'erreur

echo "=========================================="
echo "Initialisation d'Airflow - Compagnon Immo"
echo "=========================================="

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Créer les répertoires nécessaires s'ils n'existent pas
echo -e "${YELLOW}[1/6] Création des répertoires nécessaires...${NC}"
mkdir -p logs/airflow
mkdir -p data/raw
mkdir -p data/incremental
mkdir -p data/state
mkdir -p exports/st
mkdir -p airflow/logs
mkdir -p airflow/dags
mkdir -p airflow/plugins

echo -e "${GREEN}✓ Répertoires créés${NC}"

# 2. Configurer les permissions pour Airflow (UID 50000)
echo -e "${YELLOW}[2/6] Configuration des permissions (UID 50000 pour Airflow)...${NC}"

# Permissions pour les logs
sudo chown -R 50000:0 logs/airflow 2>/dev/null || chown -R 50000:0 logs/airflow 2>/dev/null || echo "Note: Impossible de changer le propriétaire des logs (peut nécessiter sudo)"
chmod -R 755 logs/airflow

# Permissions pour les DAGs
chmod -R 755 dags
find dags -type f -name "*.py" -exec chmod 644 {} \;

# Permissions pour les données
chmod -R 755 data exports 2>/dev/null || echo "Note: Certains répertoires de données peuvent nécessiter des permissions manuelles"

echo -e "${GREEN}✓ Permissions configurées${NC}"

# 3. Nettoyer les caractères CRLF dans les DAGs (Windows -> Unix)
echo -e "${YELLOW}[3/6] Nettoyage des fichiers DAG (CRLF -> LF)...${NC}"
find dags -type f -name "*.py" -exec sed -i 's/\r$//' {} \; 2>/dev/null || echo "Note: sed non disponible ou pas de fichiers à nettoyer"
echo -e "${GREEN}✓ Fichiers DAG nettoyés${NC}"

# 4. Vérifier que Docker est en cours d'exécution
echo -e "${YELLOW}[4/6] Vérification de Docker...${NC}"
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker n'est pas en cours d'exécution. Veuillez démarrer Docker.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker est actif${NC}"

# 5. Arrêter les services Airflow existants (si en cours d'exécution)
echo -e "${YELLOW}[5/6] Arrêt des services Airflow existants...${NC}"
docker compose --profile airflow down 2>/dev/null || echo "Aucun service à arrêter"
echo -e "${GREEN}✓ Services arrêtés${NC}"

# 6. Démarrer les services Airflow
echo -e "${YELLOW}[6/6] Démarrage des services Airflow...${NC}"
echo "Cela peut prendre quelques minutes lors du premier démarrage..."

# Démarrer PostgreSQL et Redis d'abord
docker compose up -d postgres-airflow redis mlflow

# Attendre que PostgreSQL soit prêt
echo "Attente de PostgreSQL..."
sleep 10

# Démarrer les services Airflow
docker compose --profile airflow up -d

echo -e "${GREEN}✓ Services Airflow démarrés${NC}"

# 7. Attendre que les services soient prêts
echo ""
echo "Attente de l'initialisation des services (30 secondes)..."
sleep 30

# 8. Créer un utilisateur admin Airflow
echo ""
echo -e "${YELLOW}Création de l'utilisateur admin Airflow...${NC}"
docker compose exec airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@compagnon-immo.fr \
    --password admin 2>/dev/null || echo "Note: L'utilisateur admin existe peut-être déjà"

echo ""
echo -e "${GREEN}=========================================="
echo "✓ Initialisation terminée avec succès!"
echo "==========================================${NC}"
echo ""
echo "Informations de connexion:"
echo "  - Interface Airflow: http://localhost:8081"
echo "  - Utilisateur: admin"
echo "  - Mot de passe: admin"
echo ""
echo "  - MLflow UI: http://localhost:5050"
echo ""
echo "Commandes utiles:"
echo "  - Voir les logs: docker compose logs -f airflow-webserver"
echo "  - Voir les services: docker compose ps"
echo "  - Arrêter Airflow: docker compose --profile airflow down"
echo "  - Redémarrer Airflow: docker compose --profile airflow restart"
echo ""
echo -e "${YELLOW}Note: Changez le mot de passe admin après la première connexion!${NC}"
