# ========== Makefile MLOps - Compagnon Immo ==========
# Gestion des pipelines avec Airflow, MLflow, DVC et Docker

# ===============================
# SOMMAIRE
# ===============================
# 1. Aide & vérifications        : help, lint, check-dependencies
# 2. Préparation & installation  : prepare-dirs, install
# 3. Build                      : docker-build, docker-api-build, airflow-build
# 4. Démarrage services         : quick-start, quick-start-airflow, quick-start-test, docker-api-run, mlflow-up, airflow-up, dvc-add-all, dvc-repro-all, dvc-pull-all
# 5. Tests & CI                 : api-test, ci-test
# 6. Arrêt & nettoyage          : api-stop, docker-api-stop, mlflow-down, airflow-down, stop-all, clean
# 7. Utilitaires                : docker-logs, airflow-logs, airflow-init, airflow-smoke, fix-permissions, check-services


# --- Choix du fichier d'env local ---
# Si tu veux garder env.txt en local, mets: ENV_DST ?= env.txt
ENV_DST  ?= .env
ENV_FILE ?= $(ENV_DST)

# Auto-load variables d'environnement (si fichier présent)
ifneq ("$(wildcard $(ENV_FILE))","")
include $(ENV_FILE)
export $(shell sed -n 's/^\([A-Za-z_][A-ZaZ0-9_]*\)=.*/\1/p' $(ENV_FILE))
endif

# ===== Variables =====
IMAGE_PREFIX := compagnon_immo
NETWORK := ml_net
PYTHON_BIN := python3
PIP := pip3
TEST_DIR := app/api/tests
DVC_TOKEN ?= default_token_securise_ou_vide

MLFLOW_IMAGE := $(IMAGE_PREFIX)-mlflow
DVC_IMAGE := $(IMAGE_PREFIX)-dvc
USER_FLAGS := --user $(shell id -u):$(shell id -g)

MLFLOW_PORT := 5050
MLFLOW_HOST := $(IMAGE_PREFIX)-mlflow
MLFLOW_URI_DCK := http://$(MLFLOW_HOST):$(MLFLOW_PORT)

AIRFLOW_SERVICES := postgres-airflow airflow-webserver airflow-scheduler
AIRFLOW_UID ?= 50000
AIRFLOW_URL ?= http://localhost:8081

DOCKER_COMPOSE_CMD := docker compose

# Couleurs
COLOR_RESET := \033[0m
COLOR_GREEN := \033[32m
COLOR_RED := \033[31m
COLOR_YELLOW := \033[33m

.PHONY: \
  help lint check-dependencies \
  prepare-dirs install \
  docker-build docker-api-build airflow-build build-all \
  quick-start quick-start-airflow quick-start-test docker-api-run mlflow-up airflow-up airflow-start dvc-add-all dvc-repro-all dvc-pull-all \
  api-test ci-test \
  api-stop docker-api-stop mlflow-down airflow-down stop-all clean \
  docker-logs airflow-logs airflow-init airflow-smoke fix-permissions check-services \
  dvc-push-all pipeline-reset run-all-docker run_dvc check-ports rebuild

# ===============================
# 1. Aide & vérifications
# ===============================
help: ## Affiche l'aide
	@echo "========== Compagnon Immo - Commandes disponibles =========="
	@grep -E '^[a-zA-Z0-9_.-]+:.*?##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

lint: ## Vérifie quelques pièges courants
	@echo "🔍 Vérification du Makefile…"
	@grep -o '^[a-zA-Z0-9_.-]\+:' Makefile | sort | uniq -d | xargs -r -I{} echo "⚠️  Cible en double: {}" || true

check-dependencies: ## Vérifie que les dépendances nécessaires sont installées
	@command -v docker >/dev/null 2>&1 || { echo "$(COLOR_RED)❌ Docker n'est pas installé.$(COLOR_RESET)"; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { echo "$(COLOR_RED)❌ Python3 n'est pas installé.$(COLOR_RESET)"; exit 1; }
	@command -v dvc >/dev/null 2>&1 || { echo "$(COLOR_RED)❌ DVC n'est pas installé.$(COLOR_RESET)"; exit 1; }
	@echo "$(COLOR_GREEN)✅ Toutes les dépendances sont installées.$(COLOR_RESET)"

# ===============================
# 2. Préparation & installation
# ===============================
prepare-dirs: ## Prépare les répertoires nécessaires
	@mkdir -p data exports mlruns logs/airflow
	@touch data/.gitkeep

install: prepare-dirs ## Installe les dépendances Python
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

# ===============================
# 3. Build
# ===============================
docker-build: prepare-dirs ## Build via compose
	@$(DOCKER_COMPOSE_CMD) build

docker-api-build: ## Build image API
	DOCKER_BUILDKIT=0 docker build -t $(IMAGE_PREFIX)-api .

airflow-build: ## Build images Airflow
	docker compose build airflow-webserver airflow-scheduler

build-all: docker-build ## Build toutes les images

# ===============================
# 4. Démarrage services
# ===============================
quick-start: prepare-dirs build-all airflow-start ## Build + démarrage complet (Airflow, MLflow, API)

quick-start-airflow: build-all airflow-start ## Build + démarrage d'Airflow uniquement

quick-start-test: quick-start dvc-repro-all ## Quick start + exécution complète de DVC

docker-api-run: docker-api-build ## Run image API
	- docker rm -f $(IMAGE_PREFIX)-api 2>/dev/null || true
	docker run -d -p 8000:8000 --name $(IMAGE_PREFIX)-api --env-file .env $(IMAGE_PREFIX)-api

mlflow-up: ## Démarre MLflow
	docker run -d --rm \
		--name $(MLFLOW_HOST) \
		--network $(NETWORK) \
		-v $(PWD)/mlruns:/mlflow/mlruns \
		-p $(MLFLOW_PORT):$(MLFLOW_PORT) \
		$(MLFLOW_IMAGE) \
		mlflow server --host 0.0.0.0 --port $(MLFLOW_PORT) \
		  --backend-store-uri sqlite:////mlflow/mlruns/mlflow.db \
		  --default-artifact-root /mlflow/mlruns

airflow-up: ## Démarre uniquement Airflow
	docker compose up -d $(AIRFLOW_SERVICES)

airflow-start: ## Démarre Airflow et services associés
	docker compose up -d postgres-airflow redis mlflow
	sleep 5
	docker compose --profile airflow up -d

dvc-add-all: ## Ajoute tous les stages DVC
	docker run --rm -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n import_data \
	  -d data/raw/merged_sales_data.csv \
	  -o data/df_sample.csv \
	  python mlops/1_import_donnees/import_data.py

dvc-repro-all: ## dvc repro de tout le pipeline
	$(DOCKER_COMPOSE_CMD) --profile dvc run --rm dvc dvc repro -f

dvc-pull-all: ## dvc pull
	$(DOCKER_COMPOSE_CMD) --profile dvc run --rm dvc dvc pull

# ===============================
# 5. Tests & CI
# ===============================
api-test: ## Lancer les tests de l'API avec démarrage automatique des services
	@echo "🧪 Tests de l'API…"
	@test -d $(TEST_DIR) || { echo "❌ Dossier de tests introuvable: $(TEST_DIR)"; exit 4; }
	@echo "🚀 Démarrage des services pour les tests..."
	@$(DOCKER_COMPOSE_CMD) up -d api mlflow redis
	@echo "⏳ Attente que l'API soit prête..."
	@timeout 60 bash -c 'until curl -f http://localhost:8000/api/v1/health >/dev/null 2>&1; do sleep 2; done' || { echo "❌ L'API n'a pas démarré dans les temps"; $(DOCKER_COMPOSE_CMD) logs api; exit 1; }
	@echo "✅ API prête, lancement des tests..."
	@API_BASE_URL=http://localhost:8000/api/v1 PYTHONPATH=. $(PYTHON_BIN) -m pytest $(TEST_DIR) -v
	@echo "🛑 Arrêt des services de test..."
	@$(DOCKER_COMPOSE_CMD) stop api mlflow redis

api-test-docker: ## Lancer les tests de l'API dans un environnement Docker complet
	@echo "🐳 Tests de l'API avec Docker…"
	@echo "🚀 Démarrage de l'environnement de test complet..."
	@$(DOCKER_COMPOSE_CMD) --profile test up --build --abort-on-container-exit --exit-code-from api-test
	@echo "🛑 Nettoyage de l'environnement de test..."
	@$(DOCKER_COMPOSE_CMD) --profile test down -v

ci-test: install ## Exécute les tests CI localement
	@echo "🔍 Lancement des tests CI..."
	@make check-services
	@echo "$(COLOR_YELLOW)🧪 Exécution des tests unitaires...$(COLOR_RESET)"
	@PYTHONPATH=. $(PYTHON_BIN) -m pytest $(TEST_DIR) -v || { echo "$(COLOR_RED)❌ Tests unitaires échoués$(COLOR_RESET)"; exit 1; }
	@echo "$(COLOR_YELLOW)🔍 Vérification du linting...$(COLOR_RESET)"
	@$(PIP) install flake8 --quiet || true
	@$(PYTHON_BIN) -m flake8 app/ --max-line-length=88 --ignore=E203,W503 || { echo "$(COLOR_RED)❌ Linting échoué$(COLOR_RESET)"; exit 1; }
	@echo "$(COLOR_GREEN)✅ Tous les tests CI ont réussi !$(COLOR_RESET)"

# ===============================
# 6. Arrêt & nettoyage
# ===============================
api-stop: ## Stoppe l'API dev (process uvicorn en arrière-plan) et le conteneur Docker
	@pkill -f "uvicorn app.routes.main:app" 2>/dev/null || echo "Aucun uvicorn local à stopper"
	docker rm -f $(IMAGE_PREFIX)-api 2>/dev/null || echo "Aucun conteneur $(IMAGE_PREFIX)-api à supprimer"

docker-api-stop: ## Stop & rm API container
	docker rm -f $(IMAGE_PREFIX)-api 2>/dev/null || echo "Aucun conteneur $(IMAGE_PREFIX)-api à supprimer"

mlflow-down: ## Stoppe MLflow
	docker stop $(MLFLOW_HOST) || true

airflow-down: ## Stoppe Airflow
	docker compose stop $(AIRFLOW_SERVICES)
	docker compose rm -f $(AIRFLOW_SERVICES) || true

stop-all: ## Stoppe tous les services, conteneurs, réseaux et processus liés au projet
	@echo "🔴 Arrêt de tous les processus uvicorn locaux..."
	-pkill -f "uvicorn app.routes.main:app" 2>/dev/null || echo "Aucun uvicorn local à stopper"
	@echo "🔴 Suppression des conteneurs Docker nommés compagnon_immo-* ..."
	-docker ps -a --filter "name=compagnon_immo" -q | xargs -r docker rm -f || echo "Aucun conteneur compagnon_immo à supprimer"
	@echo "🔴 Arrêt et suppression des services Docker Compose..."
	-$(DOCKER_COMPOSE_CMD) down -v --remove-orphans || echo "Aucun service compose à stopper"
	@echo "🟢 Tous les services et conteneurs liés au projet sont arrêtés."

clean: ## Nettoie les fichiers temporaires
	@rm -rf .pytest_cache .coverage

# ===============================
# 7. Utilitaires & réparation
# ===============================
docker-logs: ## Logs compose (tous services)
	@$(DOCKER_COMPOSE_CMD) logs -f

airflow-logs: ## Logs webserver
	docker compose logs -f airflow-webserver

airflow-init: ## Init DB Airflow + user admin
	mkdir -p logs/airflow
	sudo chown -R $(AIRFLOW_UID):0 logs/airflow || true
	docker compose run --rm airflow-webserver airflow db upgrade
	docker compose run --rm airflow-webserver \
	  airflow users create --username admin --password admin \
	  --firstname Admin --lastname User --role Admin --email admin@example.com || true

airflow-smoke: ## Vérifie Airflow en listant les DAGs
	@$(DOCKER_COMPOSE_CMD) exec airflow-webserver airflow dags list | head -n 10 || true

fix-permissions: ## Corrige les permissions des fichiers du projet
	@echo "$(COLOR_YELLOW)🔧 Correction des permissions...$(COLOR_RESET)"
	@sudo chown -R $(shell whoami):$(shell whoami) . || true
	@chmod -R u+rwx . || true
	@echo "$(COLOR_GREEN)✅ Permissions corrigées !$(COLOR_RESET)"

check-services: ## Vérifie l'état des services Docker
	@echo "🔍 Vérification des services Docker..."
	@docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "api|mlflow|airflow|redis"

# ...autres cibles annexes si besoin (build-all, pipeline-reset, etc.)...