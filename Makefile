# ========== Makefile MLOps - Compagnon Immo ==========
# Gestion des pipelines avec Airflow, MLflow, DVC et Docker

# ===============================
# SOMMAIRE
# ===============================
# 1. Aide & lint                : help, lint, check-dependencies
# 2. Quick start                : quick-start, quick-start-airflow, quick-start-test
# 3. API et Tests               : prepare-dirs, install, api-test, clean, api-stop
# 4. MLflow                     : mlflow-up, mlflow-down
# 5. Docker - orchestrations    : docker-build, docker-api-build, docker-api-run, docker-api-stop, docker-logs
# 6. DVC                        : dvc-add-all, dvc-repro-all, dvc-push-all, dvc-pull-all, pipeline-reset
# 7. Airflow                    : airflow-build, airflow-init, airflow-up, airflow-down, airflow-logs, airflow-smoke
# 8. Nettoyage et réparation    : fix-permissions, check-services, ci-test

SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

# Auto-load .env
ENV_FILE ?= .env
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
  quick-start quick-start-airflow quick-start-test \
  prepare-dirs install api-test clean api-stop fix-permissions \
  airflow-start airflow-build airflow-init airflow-up airflow-down airflow-logs airflow-smoke \
  mlflow-up mlflow-down mlflow-logs \
  docker-build docker-api-build docker-api-run docker-api-stop docker-logs docker-clean \
  build-all run-all-docker run_dvc \
  dvc-add-all dvc-repro-all dvc-push-all dvc-pull-all pipeline-reset \
  check-ports rebuild ci-test

# ===============================
# Aide & lint
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
# 📦 Quick start
# ===============================
quick-start: prepare-dirs build-all airflow-start ## Build + démarrage complet (Airflow, MLflow, API)

quick-start-airflow: build-all airflow-start ## Build + démarrage d'Airflow uniquement

quick-start-test: quick-start dvc-repro-all ## Quick start + exécution complète de DVC

# ===============================
# 🌐 API et Tests
# ===============================
prepare-dirs: ## Prépare les répertoires nécessaires
	@mkdir -p data exports mlruns logs/airflow
	@touch data/.gitkeep

install: prepare-dirs ## Installe les dépendances Python
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

api-test: ## Lancer les tests de l'API
	@echo "🧪 Tests de l'API…"
	@test -d $(TEST_DIR) || { echo "❌ Dossier de tests introuvable: $(TEST_DIR)"; exit 4; }
	@PYTHONPATH=. $(PYTHON_BIN) -m pytest $(TEST_DIR) -v

clean: ## Nettoie les fichiers temporaires
	@rm -rf .pytest_cache .coverage

api-stop: ## Stoppe l'API dev (process uvicorn en arrière-plan)
	@pkill -f "uvicorn app.routes.main:app" || echo "Aucun uvicorn à stopper"

# ===============================
# 📈 MLflow
# ===============================
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

mlflow-down: ## Stoppe MLflow
	docker stop $(MLFLOW_HOST) || true

# ===============================
# 🐳 Docker - orchestrations
# ===============================
docker-build: prepare-dirs ## Build via compose
	@$(DOCKER_COMPOSE_CMD) build

docker-api-build: ## Build image API
	DOCKER_BUILDKIT=0 docker build -t $(IMAGE_PREFIX)-api .

docker-api-run: docker-api-build ## Run image API
	- docker rm -f $(IMAGE_PREFIX)-api 2>/dev/null || true
	docker run -d -p 8000:8000 --name $(IMAGE_PREFIX)-api --env-file .env $(IMAGE_PREFIX)-api

docker-api-stop: ## Stop & rm API container
	docker rm -f $(IMAGE_PREFIX)-api 2>/dev/null || echo "Aucun conteneur $(IMAGE_PREFIX)-api à supprimer"

docker-logs: ## Logs compose (tous services)
	@$(DOCKER_COMPOSE_CMD) logs -f

# ===============================
# 🏗️ DVC: ajout des stages & orchestration
# ===============================
dvc-add-all: ## Ajoute tous les stages DVC
	docker run --rm -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n import_data \
	  -d data/raw/merged_sales_data.csv \
	  -o data/df_sample.csv \
	  python mlops/1_import_donnees/import_data.py

dvc-repro-all: ## dvc repro de tout le pipeline
	docker run --rm $(USER_FLAGS) \
	  --network $(NETWORK) \
	  -e MLFLOW_TRACKING_URI=$(MLFLOW_URI_DCK) \
	  -v $(PWD):/app -w /app $(DVC_IMAGE) dvc repro -f

dvc-push-all: ## dvc push
	docker run --rm $(USER_FLAGS) \
	  --network $(NETWORK) \
	  -e MLFLOW_TRACKING_URI=$(MLFLOW_URI_DCK) \
	  -v $(PWD):/app -w /app \
	  -e DVC_TOKEN=$(DVC_TOKEN) $(DVC_IMAGE) dvc push

dvc-pull-all: ## dvc pull
	docker run --rm $(USER_FLAGS) \
	  --network $(NETWORK) \
	  -e MLFLOW_TRACKING_URI=$(MLFLOW_URI_DCK) \
	  -v $(PWD):/app -w /app $(DVC_IMAGE) dvc pull

pipeline-reset: dvc-pull-all dvc-add-all dvc-repro-all ## Pull + add + repro

# ===============================
# Airflow
# ===============================
airflow-build: ## Build images Airflow
	docker compose build airflow-webserver airflow-scheduler

airflow-init: ## Init DB Airflow + user admin
	mkdir -p logs/airflow
	sudo chown -R $(AIRFLOW_UID):0 logs/airflow || true
	docker compose run --rm airflow-webserver airflow db upgrade
	docker compose run --rm airflow-webserver \
	  airflow users create --username admin --password admin \
	  --firstname Admin --lastname User --role Admin --email admin@example.com || true

airflow-up: ## Démarre uniquement Airflow
	docker compose up -d $(AIRFLOW_SERVICES)

airflow-down: ## Stoppe Airflow
	docker compose stop $(AIRFLOW_SERVICES)
	docker compose rm -f $(AIRFLOW_SERVICES) || true

airflow-logs: ## Logs webserver
	docker compose logs -f airflow-webserver

airflow-smoke: ## Vérifie Airflow en listant les DAGs
	@$(DOCKER_COMPOSE_CMD) exec airflow-webserver airflow dags list | head -n 10 || true

# ===============================
# 🧹 Nettoyage et réparation
# ===============================
fix-permissions: ## Corrige les permissions des fichiers du projet
	@echo "$(COLOR_YELLOW)🔧 Correction des permissions...$(COLOR_RESET)"
	@sudo chown -R $(shell whoami):$(shell whoami) . || true
	@chmod -R u+rwx . || true
	@echo "$(COLOR_GREEN)✅ Permissions corrigées !$(COLOR_RESET)"

check-services: ## Vérifie l'état des services Docker
	@echo "🔍 Vérification des services Docker..."
	@docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "api|mlflow|airflow|redis"

ci-test: install ## Exécute les tests CI localement
	@echo "🔍 Lancement des tests CI..."
	@make check-services
	@echo "$(COLOR_YELLOW)🧪 Exécution des tests unitaires...$(COLOR_RESET)"
	@PYTHONPATH=. $(PYTHON_BIN) -m pytest $(TEST_DIR) -v || { echo "$(COLOR_RED)❌ Tests unitaires échoués$(COLOR_RESET)"; exit 1; }
	@echo "$(COLOR_YELLOW)🔍 Vérification du linting...$(COLOR_RESET)"
	@$(PIP) install flake8 --quiet || true
	@$(PYTHON_BIN) -m flake8 app/ --max-line-length=88 --ignore=E203,W503 || { echo "$(COLOR_RED)❌ Linting échoué$(COLOR_RESET)"; exit 1; }
	@echo "$(COLOR_GREEN)✅ Tous les tests CI ont réussi !$(COLOR_RESET)"