#!/bin/bash

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Configuration du monitoring avancé${NC}"
echo ""

# Fonction de log
log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Vérifier que Docker est installé
if ! command -v docker &> /dev/null; then
    error "Docker n'est pas installé"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    error "Docker Compose n'est pas installé"
    exit 1
fi

log "Docker et Docker Compose sont disponibles"

# Créer les dossiers nécessaires
log "Création des dossiers de configuration..."
mkdir -p infra/monitoring/grafana/{dashboards,provisioning/{dashboards,datasources}}
mkdir -p tests/load
mkdir -p scripts
mkdir -p logs
mkdir -p outputs/{models,plots}

log "Dossiers créés avec succès"

# Vérifier les fichiers de configuration
log "Vérification des fichiers de configuration..."

config_files=(
    "infra/monitoring/prometheus.yml"
    "infra/monitoring/alertmanager.yml"
    "infra/monitoring/alert_rules.yml"
    "infra/monitoring/loki.yml"
    "infra/monitoring/grafana/provisioning/datasources/datasources.yml"
    "tests/load/locustfile.py"
    "docker-compose.yml"
)

missing_files=0
for file in "${config_files[@]}"; do
    if [ ! -f "$file" ]; then
        error "Fichier manquant: $file"
        ((missing_files++))
    fi
done

if [ $missing_files -gt 0 ]; then
    error "$missing_files fichier(s) de configuration manquant(s)"
    exit 1
fi

log "Tous les fichiers de configuration sont présents"

# Arrêter les services existants
log "Arrêt des services existants..."
docker-compose down -v

# Nettoyer les anciens volumes (optionnel)
read -p "Voulez-vous nettoyer les anciens volumes de données ? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Nettoyage des volumes..."
    docker volume prune -f
fi

# Démarrer les services
log "Démarrage des services de monitoring..."
docker-compose up -d

# Attendre que les services démarrent
log "Attente du démarrage des services..."
sleep 45

# Vérifier les services
log "Vérification des services..."
bash scripts/check_services.sh

# Configuration finale
log "Configuration des permissions..."
chmod +x scripts/*.sh

log "🎉 Configuration terminée !"
echo ""
echo "📊 Accès aux interfaces:"
echo "• Application:    http://localhost:8000"
echo "• Prometheus:     http://localhost:9090"
echo "• Grafana:        http://localhost:3000 (admin/admin)"
echo "• Alertmanager:   http://localhost:9093"
echo "• Jaeger:         http://localhost:16686"
echo "• Locust:         http://localhost:8089"
echo ""
echo "📖 Pour plus d'informations, consultez la documentation dans docs/"
