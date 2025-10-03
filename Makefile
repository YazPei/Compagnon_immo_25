# ========== Makefile MLOps - Compagnon Immo ==========
# Gestion des pipelines avec Airflow, MLflow, DVC et Docker

# ===============================
# SOMMAIRE
# ===============================
# 1. Aide & vérifications        : help, lint, check-dependencies
# 2. Préparation & installation  : prepare-dirs, venv, install
# 3. Build                      : docker-build, docker-api-build, airflow-build
# 4. Démarrage services         : quick-start, quick-start-airflow, quick-start-test, docker-api-run, api-dev, mlflow-up, airflow-up, dvc-add-all, dvc-repro-all, dvc-pull-all
# 5. Tests & CI                 : api-test, ci-test, test-ml, test-all
# 6. Arrêt & nettoyage          : api-stop, docker-api-stop, mlflow-down, airflow-down, stop-all, clean, clean_exports, clean_dvc, clean_all
# 7. Utilitaires                : docker-logs, airflow-logs, airflow-init, airflow-smoke, fix-permissions, check-services, status, ports-check
# 8. DVC Avancé                 : dvc-push-all, pipeline-reset, add_stage_import, add_stage_fusion, ... (stages individuels)

# Shell strict pour robustesse (inspiré de Makefile A)
SHELL := /usr/bin/env bash
.SHELLFLAGS := -euo pipefail -c

# --- Choix du fichier d'env local ---
# Si tu veux garder env.txt en local, mets: ENV_DST ?= env.txt
ENV_DST  ?= .env
ENV_FILE ?= $(ENV_DST)

# Auto-load variables d'environnement (si fichier présent)
ifneq ("$(wildcard $(ENV_FILE))","")
include $(ENV_FILE)
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' $(ENV_FILE))
endif

# Branche courante (utilisée par env-from-gh si non passée, inspiré de Makefile A)
REF ?= $(shell git rev-parse --abbrev-ref HEAD)

# ===== Variables =====
IMAGE_PREFIX := compagnon_immo
NETWORK := ml_net
PYTHON_BIN := python3
PIP := pip3
TEST_DIR := app/api/tests
DVC_TOKEN ?= default_token_securise_ou_vide

# Ajouts inspirés de Makefile A : Venv et détection Docker Compose
VENV       := .venv
PYTHON_BIN := $(if $(VENV),$(VENV)/bin/python,python3)
PIP        := $(if $(VENV),$(VENV)/bin/pip,pip3)
DOCKER_COMPOSE := $(shell command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

MLFLOW_IMAGE := $(IMAGE_PREFIX)-mlflow
DVC_IMAGE := $(IMAGE_PREFIX)-dvc
USER_FLAGS := --user $(shell id -u):$(shell id -g)

MLFLOW_PORT := 5050
MLFLOW_HOST := $(IMAGE_PREFIX)-mlflow
MLFLOW_URI_DCK := http://$(MLFLOW_HOST):$(MLFLOW_PORT)

AIRFLOW_SERVICES := postgres-airflow airflow-webserver airflow-scheduler
AIRFLOW_UID ?= 50000
AIRFLOW_URL ?= http://localhost:8081

# Couleurs
COLOR_RESET := \033[0m
COLOR_GREEN := \033[32m
COLOR_RED := \033[31m
COLOR_YELLOW := \033[33m

.PHONY: \
  help lint check-dependencies install-gh\
  prepare-dirs venv install \
  docker-build docker-api-build airflow-build build-all \
  quick-start quick-start-airflow quick-start-test docker-api-run api-dev mlflow-up airflow-up airflow-start dvc-add-all dvc-repro-all dvc-pull-all dvc-push-all pipeline-reset \
  add_stage_import add_stage_fusion add_stage_preprocessing add_stage_clustering add_stage_encoding add_stage_lgbm add_stage_utils add_stage_analyse add_stage_splitst add_stage_decompose add_stage_sarimax add_stage_evaluate \
  api-test ci-test test-ml test-all \
  api-stop docker-api-stop mlflow-down airflow-down stop-all clean clean_exports clean_dvc clean_all \
  docker-logs airflow-logs airflow-init airflow-smoke fix-permissions check-services status ports-check \
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
	@command -v gh >/dev/null 2>&1 || { echo "$(COLOR_RED)❌ 'gh' (GitHub CLI) introuvable.$(COLOR_RESET)"; echo "$(COLOR_YELLOW)💡 Lance 'make install-gh' pour l'installer.$(COLOR_RESET)"; exit 1; }
	@echo "$(COLOR_GREEN)✅ Toutes les dépendances sont installées.$(COLOR_RESET)"

# ===============================
# 2. Préparation & installation
# ===============================
prepare-dirs: ## Prépare les répertoires nécessaires
	@mkdir -p data exports mlruns logs/airflow
	@touch data/.gitkeep

venv: prepare-dirs ## Crée l'environnement virtuel Python (inspiré de Makefile A)
	@test -d $(VENV) || python3 -m venv $(VENV)

install: venv prepare-dirs ## Installe les dépendances Python (amélioré avec venv et __init__.py)
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@touch app/__init__.py
	@touch app/api/__init__.py
	@touch app/api/utils/__init__.py

install-gh: ## Installe GitHub CLI si absent
	@echo "🔧 Vérification/installation de GitHub CLI..."
	@command -v gh >/dev/null 2>&1 && { echo "✅ GitHub CLI déjà installé."; exit 0; } || true
	@echo "📦 Installation de GitHub CLI via Snap..."
	sudo snap install gh
	@echo "✅ GitHub CLI installé. Lance 'gh auth login' pour te connecter."

# ===============================
# 3. Build
# ===============================
docker-build: prepare-dirs ## Build via compose (utilise détection dynamique)
	@$(DOCKER_COMPOSE) build

docker-api-build: ## Build image API
	DOCKER_BUILDKIT=0 docker build -t $(IMAGE_PREFIX)-api .

airflow-build: ## Build images Airflow
	$(DOCKER_COMPOSE) build airflow-webserver airflow-scheduler

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

api-dev: install ## Démarre l'API en mode développement local (inspiré de Makefile A)
	@echo "🚀 Démarrage de l'API… http://localhost:8000"
	@echo "📚 Docs : http://localhost:8000/docs"
	@PYTHONPATH=. nohup $(PYTHON_BIN) -m uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &

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
	$(DOCKER_COMPOSE) up -d $(AIRFLOW_SERVICES)

airflow-start: ## Démarre Airflow et services associés
	$(DOCKER_COMPOSE) up -d postgres-airflow redis mlflow
	sleep 5
	$(DOCKER_COMPOSE) --profile airflow up -d

dvc-add-all: add_stage_import add_stage_fusion add_stage_preprocessing add_stage_clustering add_stage_encoding add_stage_lgbm add_stage_utils add_stage_analyse add_stage_splitst add_stage_decompose add_stage_sarimax add_stage_evaluate ## Ajoute tous les stages DVC (granularité inspirée de Makefile A)
	@echo "✅ Tous les stages DVC ont été ajoutés avec succès !"

dvc-repro-all: ## dvc repro de tout le pipeline
	$(DOCKER_COMPOSE) --profile dvc run --rm dvc dvc repro -f

dvc-pull-all: ## dvc pull
	$(DOCKER_COMPOSE) --profile dvc run --rm dvc dvc pull

dvc-push-all: ## dvc push de tout le pipeline (inspiré de Makefile A)
	$(DOCKER_COMPOSE) --profile dvc run --rm dvc dvc push

pipeline-reset: dvc-pull-all dvc-add-all dvc-repro-all ## Reset complet du pipeline DVC (inspiré de Makefile A)

# ===== DVC stages individuels (inspirés de Makefile A) =====
add_stage_import: ## Ajoute le stage DVC pour l'import des données
	docker run --rm $(USER_FLAGS) -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n import_data \
	  -d data/raw/merged_sales_data.csv \
	  -o data/df_sample.csv \
	  python mlops/1_import_donnees/import_data.py

add_stage_fusion: ## Ajoute le stage DVC pour la fusion des données
	docker run --rm $(USER_FLAGS) -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n fusion \
	  -d data/df_sample.csv \
	  -d data/raw/DVF_donnees_macroeco.csv \
	  -o data/df_sales_clean_polars.csv \
	  python mlops/3_fusion/fusion.py

add_stage_preprocessing: ## Ajoute le stage DVC pour le preprocessing
	docker run --rm $(USER_FLAGS) -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n preprocessing \
	  -d data/df_sales_clean_polars.csv \
	  -o data/train_clean.csv \
	  -o data/test_clean.csv \
	  -o data/df_sales_clean_ST.csv \
	  python mlops/preprocessing_4/preprocessing.py

add_stage_clustering: ## Ajoute le stage DVC pour le clustering
	docker run --rm $(USER_FLAGS) -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n clustering \
	  -d data/train_clean.csv \
	  -d data/test_clean.csv \
	  -d data/df_sales_clean_ST.csv \
	  -o data/df_cluster.csv \
	  python mlops/5_clustering/Clustering.py

add_stage_encoding: ## Ajoute le stage DVC pour l'encoding
	docker run --rm $(USER_FLAGS) -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n encoding \
	  -d data/df_cluster.csv \
	  -o data/X_train.csv \
	  -o data/y_train.csv \
	  -o data/X_test.csv \
	  -o data/y_test.csv \
	  python mlops/6_Regression/1_Encoding/encoding.py

add_stage_lgbm: ## Ajoute le stage DVC pour LGBM
	docker run --rm $(USER_FLAGS) -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n lgbm \
	  -d data/X_train.csv \
	  -d data/y_train.csv \
	  -d data/X_test.csv \
	  -d data/y_test.csv \
	  -o exports/reg/model_lgbm.joblib \
	  python mlops/6_Regression/2_LGBM/train_lgbm.py

add_stage_utils: ## Ajoute le stage DVC pour les utils
	docker run --rm $(USER_FLAGS) -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n utils \
	  python mlops/6_Regression/3_UTILS/utils.py

add_stage_analyse: ## Ajoute le stage DVC pour l'analyse
	docker run --rm $(USER_FLAGS) -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n analyse \
	  -d data/X_test.csv \
	  -d data/y_test.csv \
	  -o exports/reg/shap_summary.png \
	  python mlops/6_Regression/4_Analyse/analyse.py

add_stage_splitst: ## Ajoute le stage DVC pour le split séries temporelles
	docker run --rm $(USER_FLAGS) -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n splitst \
	  -d data/df_sales_clean_ST.csv \
	  -o data/processed/train_periodique_q12.csv \
	  -o data/processed/test_periodique_q12.csv \
	  python mlops/7_Serie_temporelle/1_SPLIT/load_split.py

add_stage_decompose: ## Ajoute le stage DVC pour la décomposition
	docker run --rm $(USER_FLAGS) -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n decompose \
	  -d data/processed/train_periodique_q12.csv \
	  -o exports/st/fig_decompose.png \
	  python mlops/7_Serie_temporelle/2_Decompose/seasonal_decomp.py

add_stage_sarimax: ## Ajoute le stage DVC pour SARIMAX
	docker run --rm $(USER_FLAGS) -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n sarimax \
	  -d data/processed/train_periodique_q12.csv \
	  -o exports/st/best_model.pkl \
	  python mlops/7_Serie_temporelle/3_SARIMAX/sarimax_api.py

add_stage_evaluate: ## Ajoute le stage DVC pour l'évaluation ST
	docker run --rm $(USER_FLAGS) -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n evaluate \
	  -d data/processed/train_periodique_q12.csv \
	  -d data/processed/test_periodique_q12.csv \
	  -o exports/st/eval_metrics.json \
	  python mlops/7_Serie_temporelle/4_EVALUATE/evaluate_ST.py

# ===============================
# ☁️ Secrets depuis GitHub Actions → .env
# ===============================
# Paramètres overridables : make env-from-gh BRANCH=Auto_github WF=permissions ART_NAME=env-artifact ENV_DST=.env
BRANCH   ?= Auto_github
WF       ?= permissions
ART_NAME ?= env-artifact

env-from-gh: ## Déclenche le workflow GH, attend, télécharge env.txt et l'installe en $(ENV_DST)
	@command -v gh >/dev/null || { echo "❌ 'gh' (GitHub CLI) introuvable"; exit 127; }
	@echo "🚀 Déclenche '$(WF)' sur branche '$(BRANCH)'"
	@if gh auth status >/dev/null 2>&1; then \
	  gh workflow run "$(WF)" --ref "$(BRANCH)" >/dev/null; \
	else \
	  : "$${GH_TOKEN:?Set GH_TOKEN (export GH_TOKEN=<PAT>)}"; \
	  GITHUB_TOKEN="$$GH_TOKEN" gh workflow run "$(WF)" --ref "$(BRANCH)" >/dev/null; \
	fi
	@sleep 2
	@echo "⏳ Récupération du dernier run…"
	@RUN_ID=$$(gh run list --workflow="$(WF)" --limit 30 --json databaseId,headBranch \
	  -q '.[] | select(.headBranch=="'$(BRANCH)'") | .databaseId' | head -n1); \
	[ -n "$$RUN_ID" ] || { echo "❌ Aucun run pour '$(WF)' sur '$(BRANCH)'"; exit 1; }; \
	echo "▶ RUN_ID=$$RUN_ID"; \
	gh run watch "$$RUN_ID" || true; \
	CONC=$$(gh run view "$$RUN_ID" --json conclusion -q .conclusion); \
	if [ "$$CONC" != "success" ]; then \
	  echo "❌ Run $$RUN_ID = $$CONC"; gh run view "$$RUN_ID" --web || true; exit 1; \
	fi; \
	echo "📦 Télécharge l'artefact '$(ART_NAME)'…"; \
	rm -rf tmp-$(ART_NAME); \
	gh run download "$$RUN_ID" -n "$(ART_NAME)" -D tmp-$(ART_NAME) \
	  || { echo "❌ Artefact '$(ART_NAME)' introuvable"; exit 1; }; \
	SRC=$$(find tmp-$(ART_NAME) -type f -name "env.txt" -print -quit); \
	[ -n "$$SRC" ] || { echo "❌ 'env.txt' introuvable. Contenu :" ;