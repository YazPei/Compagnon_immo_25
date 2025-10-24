#!/bin/bash

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🔍 Vérification des services de monitoring..."
echo ""

# Liste des services à vérifier
declare -A services=(
    ["Application principale"]="http://localhost:8000/health"
    ["Prometheus"]="http://localhost:9090/-/healthy"
    ["Grafana"]="http://localhost:3000/api/health"
    ["Alertmanager"]="http://localhost:9093/-/healthy"
    ["Jaeger"]="http://localhost:16686/api/services"
    ["Locust"]="http://localhost:8089/"
    ["Loki"]="http://localhost:3100/ready"
)

# Fonction pour vérifier un service
check_service() {
    local name=$1
    local url=$2
    local timeout=10
    
    printf "%-25s" "$name"
    
    if curl -s --max-time $timeout "$url" > /dev/null 2>&1; then
        echo -e "[${GREEN}✓${NC}] Accessible"
        return 0
    else
        echo -e "[${RED}✗${NC}] Non accessible"
        return 1
    fi
}

# Vérification de Docker Compose
echo "📋 Vérification de Docker Compose..."
if ! docker-compose ps > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker Compose n'est pas lancé${NC}"
    echo "Lancez: docker-compose up -d"
    exit 1
fi

echo -e "${GREEN}✅ Docker Compose est actif${NC}"
echo ""

# Attendre un peu que les services démarrent
echo "⏳ Attente du démarrage des services (30 secondes)..."
sleep 30
echo ""

# Vérification de chaque service
failed_services=0
for service_name in "${!services[@]}"; do
    if ! check_service "$service_name" "${services[$service_name]}"; then
        ((failed_services++))
    fi
done

echo ""

# Résumé
if [ $failed_services -eq 0 ]; then
    echo -e "${GREEN}🎉 Tous les services sont accessibles !${NC}"
    echo ""
    echo "📊 Accès aux interfaces:"
    echo "• Application:    http://localhost:8000"
    echo "• Prometheus:     http://localhost:9090"
    echo "• Grafana:        http://localhost:3000 (admin/admin)"
    echo "• Alertmanager:   http://localhost:9093"
    echo "• Jaeger:         http://localhost:16686"
    echo "• Locust:         http://localhost:8089"
    echo "• Loki:           http://localhost:3100"
else
    echo -e "${RED}❌ $failed_services service(s) ne sont pas accessibles${NC}"
    echo ""
    echo "🔧 Actions de dépannage:"
    echo "1. Vérifiez les logs: docker-compose logs [service_name]"
    echo "2. Redémarrez les services: docker-compose restart"
    echo "3. Vérifiez les ports: netstat -tlnp | grep [port]"
    exit 1
fi
