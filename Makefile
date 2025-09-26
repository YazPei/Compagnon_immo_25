# ========== Makefile MLOps - Compagnon immo ==========
# Pipelines Régression + Séries temporelles
# Outils : DVC, MLflow, Docker, bash scripts, Airflow

# permission .env

SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c

# Auto-load .env
ENV_FILE ?= .env
ifneq ("$(wildcard $(ENV_FILE))","")
include $(ENV_FILE)
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' $(ENV_FILE))
endif



# ===== Variables =====
SHELL := /usr/bin/env bash
IMAGE_PREFIX   := compagnon_immo
NETWORK        := ml_net
VENV       := .venv
PYTHON_BIN := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
TEST_DIR   := app/api/tests
DVC_TOKEN     ?= default_token_securise_ou_vide

MLFLOW_IMAGE   := $(IMAGE_PREFIX)-mlflow
DVC_IMAGE      := $(IMAGE_PREFIX)-dvc
USER_FLAGS     := --user $(shell id -u):$(shell id -g)

MLFLOW_PORT    := 5050
MLFLOW_HOST    := $(IMAGE_PREFIX)-mlflow
MLFLOW_URI_DCK := http://$(MLFLOW_HOST):$(MLFLOW_PORT)

# Détection docker compose (nouvelle syntaxe)
DOCKER_COMPOSE := $(shell command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1 && echo "docker compose" || echo "docker-compose")

.PHONY: \
  setup_dags \
  prepare-dirs install help lint \
  quick-start quick-start-pipeline quick-start-test \
  api-dev venv install api-test \
  mlflow-ui mlflow-clean mlflow-status mlflow-dockerized create-network build-mlflow mlflow-up mlflow-down \
  docker-build docker-api-build docker-api-run docker-api-stop docker-stack-up docker-stack-down docker-logs \
  build-all run-all-docker run_full \
  run_dvc run_fusion run_preprocessing run_clustering run_encoding run_lgbm run_util run_analyse run_splitst run_decompose run_SARIMAX run_evaluate \
  build-base build-fusion build-preprocessing build-clustering build-encoding build-lgbm build-util build-analyse build-splitST build-decompose build-SARIMAX build-evaluate \
  dvc-add-all dvc-repro-all dvc-push-all dvc-pull-all pipeline-reset \
  add_stage_import add_stage_fusion add_stage_preprocessing add_stage_clustering add_stage_encoding add_stage_lgbm add_stage_utils add_stage_analyse add_stage_splitst add_stage_decompose add_stage_sarimax add_stage_evaluate \
  clean_exports clean_dvc clean_all \
  test-ml test-all \
  status ports-check \
  airflow airflow-up airflow-down airflow-restart airflow-logs airflow-run airflow-ps

# ===============================
# Aide & lint
# ===============================
help: ## Affiche l'aide
	@echo "========== Compagnon Immo - Commandes disponibles =========="
	@echo ""
	@grep -E '^[a-zA-Z0-9_.-]+:.*?##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo ""

lint: ## Vérifie quelques pièges courants
	@echo "🔍 Vérification du Makefile…"
	@grep -o '^[a-zA-Z0-9_.-]\+:' Makefile | sort | uniq -d | xargs -r -I{} echo "⚠️  Cible en double: {}" || true

# ===============================
# 📦 Quick start
# ===============================
quick-start: setup_dags build-all ## Build + run docker


quick-start-test: quick-starts dvc-repro-all ## + DVC repro complet

# ===============================
# ☁️ DagsHub Setup
# ===============================

# --- DagsHub non-interactive setup ---
setup_dags:
	@set -eu
	@ : "$${DAGSHUB_USER:?Missing DAGSHUB_USER in .env}"
	@ : "$${DAGSHUB_TOKEN:?Missing DAGSHUB_TOKEN in .env}"
	@ : "$${DAGSHUB_REPO:?Missing DAGSHUB_REPO in .env}"
	@ : "$${MLFLOW_TRACKING_URI:?Missing MLFLOW_TRACKING_URI in .env}"
	@mkdir -p infra/config
	@printf 'owner: "%s"\nrepo: "%s"\nmlflow_tracking_uri: "%s"\n' \
		"$${DAGSHUB_USER}" \
		"$${DAGSHUB_REPO}" \
		"$${MLFLOW_TRACKING_URI}" \
		> infra/config/dagshub.yaml
	@echo "✅ DagsHub config written to infra/config/dagshub.yaml"




# ===============================
# 🌐 API et Interface Web
# ===============================
api-dev: ## Démarre l'API en mode développement
	@echo "🚀 Démarrage de l'API…"
	@echo "📍 API : http://localhost:8000"
	@echo "📚 Docs : http://localhost:8000/docs"
	@PYTHONPATH=api nohup uvicorn app.routes.main:app --reload --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &

TEST_DIR   := app/api/tests/

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)

install: venv
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@touch app/__init__.py
	@touch app/api/__init__.py
	@touch app/api/utils/__init__.py


api-test: ## Lancer les tests de l'API
	@echo "🧪 Tests de l'API…"
	@test -d $(TEST_DIR) || { echo "❌ Dossier de tests introuvable: $(TEST_DIR)"; exit 4; }
	@PYTHONPATH=. $(PYTHON_BIN) -m pytest $(TEST_DIR) -v

clean:

	@rm -rf .pytest_cache .coverage


api-stop: ## Stoppe l'API dev (process uvicorn en arrière-plan)
	@pkill -f "uvicorn app.routes.main:app" || echo "Aucun uvicorn à stopper"

# ===============================
# 📈 MLflow (local & docker)
# ===============================
mlflow-ui: ## Démarre l'interface MLflow (local)
	@$(PYTHON_BIN) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port $(MLFLOW_PORT)

mlflow-clean: ## Nettoie les runs MLflow
	@rm -rf mlruns/ mlflow.db

mlflow-status: ## Affiche le statut des derniers runs
	@find mlruns/ -name "meta.yaml" 2>/dev/null | xargs grep -H "status" || echo "Aucun run trouvé"

mlflow-dockerized: create-network build-mlflow mlflow-up ## MLflow dans Docker

create-network:
	@docker network create $(NETWORK) 2>/dev/null || echo "✅ Réseau '$(NETWORK)' déjà existant"

build-mlflow: ## Build image MLflow
	docker build -f mlops/mlflow/Dockerfile.mlflow -t $(MLFLOW_IMAGE) .

mlflow-up: ## Run MLflow server (docker, port $(MLFLOW_PORT))
	docker run -d --rm \
		--name $(MLFLOW_HOST) \
		--network $(NETWORK) \
		-v $(PWD)/mlruns:/mlflow/mlruns \
		-p $(MLFLOW_PORT):$(MLFLOW_PORT) \
		$(MLFLOW_IMAGE) \
		mlflow server --host 0.0.0.0 --port $(MLFLOW_PORT) \
		  --backend-store-uri sqlite:////mlflow/mlruns/mlflow.db \
		  --default-artifact-root /mlflow/mlruns

mlflow-down: ## Stoppe MLflow (docker)
	docker stop $(MLFLOW_HOST) || true

# ===============================
# 🐳 Docker - orchestrations
# ===============================
build-all: prepare-dirs docker-build docker-stack-up 
run_permission_airflow:
	@chmod +x ./dags-airflow-permission.sh && ./dags-airflow-permission.sh

prepare-dirs:
	@mkdir -p data exports mlruns
	@touch data/.gitkeep

docker-build: prepare-dirs ## Build via compose
	@$(DOCKER_COMPOSE) build

docker-stack-up: ## Up via compose (détaché)
	@$(DOCKER_COMPOSE) up -d

docker-stack-down: ## Down via compose
	@$(DOCKER_COMPOSE) down

docker-logs: ## Logs compose
	@$(DOCKER_COMPOSE) logs -f

# API image seule
docker-api-build: ## Build image API
	DOCKER_BUILDKIT=0 docker build -t $(IMAGE_PREFIX)-api .

docker-api-run: docker-api-build ## Run image API
	- docker rm -f $(IMAGE_PREFIX)-api 2>/dev/null || true
	docker run -d -p 8000:8000 --name $(IMAGE_PREFIX)-api --env-file .env $(IMAGE_PREFIX)-api

docker-api-stop: ## Stop & rm API container
	docker rm -f $(IMAGE_PREFIX)-api 2>/dev/null || echo "Aucun conteneur $(IMAGE_PREFIX)-api à supprimer"

free-port-8000: ## Libère le port 8000 si occupé (processus local ou docker)
	@echo "🔄 Libération du port 8000…"
	-@fuser -k 8000/tcp || true
	-@docker ps --filter "publish=8000" --format "{{.ID}}" | xargs -r docker stop




# ===============================
# 📈️ Build images "étapes" pipeline
# ===============================
build-all: docker_build build-base build-fusion build-preprocessing fix-perms-clustering build-clustering build-encoding build-lgbm build-util build-analyse build-splitST build-decompose build-SARIMAX build-evaluate ## Build de toutes les images




docker_build:
	docker build -f mlops/1_import_donnees/Dockerfile.run -t $(IMAGE_PREFIX)-run .


build-base:
	docker build -f mlops/2_dvc/Dockerfile -t $(DVC_IMAGE) .

build-fusion:
	docker build -f mlops/3_fusion/Dockerfile.fusion -t $(IMAGE_PREFIX)-fus .

build-preprocessing:
	docker build -f mlops/preprocessing_4/Dockerfile.preprocessing -t $(IMAGE_PREFIX)-preprocessing .
fix-perms-clustering:
	sudo chown -R $(shell id -u):$(shell id -g) exports
	dvc unprotect exports/df_cluster.csv || true
	chmod u+rw exports/df_cluster.csv || true
	
build-clustering:
	docker build -f mlops/5_clustering/Dockerfile.clustering -t $(IMAGE_PREFIX)-clust .

build-encoding:
	docker build -f mlops/6_Regression/1_Encoding/Dockerfile.encoding.REG -t $(IMAGE_PREFIX)-encod .

build-lgbm:
	docker build -f mlops/6_Regression/2_LGBM/Dockerfile.lgbm.REG -t $(IMAGE_PREFIX)-lgbm .

build-util:
	docker build -f mlops/6_Regression/3_UTILS/Dockerfile.util.REG -t $(IMAGE_PREFIX)-util .

build-analyse:
	docker build -f mlops/6_Regression/4_Analyse/Dockerfile.analyse.REG -t $(IMAGE_PREFIX)-analyse .

build-splitST:
	docker build -f mlops/7_Serie_temporelle/1_SPLIT/Dockerfile.split.ST -t $(IMAGE_PREFIX)-splitst .

build-decompose:
	docker build -f mlops/7_Serie_temporelle/2_Decompose/Dockerfile.decompose.ST -t $(IMAGE_PREFIX)-decomp .

build-SARIMAX:
	docker build -f mlops/7_Serie_temporelle/3_SARIMAX/Dockerfile.sarimax.ST -t $(IMAGE_PREFIX)-sarimax .

build-evaluate:
	docker build -f mlops/7_Serie_temporelle/4_EVALUATE/Dockerfile.evaluate.ST -t $(IMAGE_PREFIX)-evalu .

# ===============================
# 🐳 Exécution pipeline (containers éphémères)
# ===============================
run-all-docker: run_full run_dvc run_fusion run_preprocessing run_clustering run_lgbm run_util run_analyse run_splitst run_decompose run_SARIMAX run_evaluate ## Exécute toutes les étapes

run_full:
	docker run --rm $(IMAGE_PREFIX)-run

run_dvc: ## Lance le script DVC dans l'image
	docker run --rm \
		$(USER_FLAGS) \
		--env-file .env.yaz \
		--env MLFLOW_TRACKING_URI=$(MLFLOW_URI_DCK) \
		--network $(NETWORK) \
		-v $(PWD):/app \
		-w /app \
		$(DVC_IMAGE) \
		bash mlops/2_dvc/run_dvc.sh

run_fusion:
	docker run --rm \
		$(USER_FLAGS) \
		--network $(NETWORK) \
		-e MLFLOW_TRACKING_URI=$(MLFLOW_URI_DCK) \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/mlops:/app/mlops \
		-w /app \
		$(IMAGE_PREFIX)-fus \
		bash mlops/3_fusion/run_fusion.sh

run_preprocessing:
	docker run --rm \
		$(USER_FLAGS) \
		--network $(NETWORK) \
		-e MLFLOW_TRACKING_URI=$(MLFLOW_URI_DCK) \
		-v $(PWD):/app \
		-w /app \
		-p 8001:8001 \
		$(IMAGE_PREFIX)-preprocess \
		python mlops/preprocessing_4/preprocessing.py --input-path data/clean/ --output-path data/processed/

run_clustering:
	docker run --rm \
		$(USER_FLAGS) \
		--network $(NETWORK) \
		-e MLFLOW_TRACKING_URI=$(MLFLOW_URI_DCK) \
		-v $(PWD):/app \
		-w /app \
		-p 8002:8002 \
		$(IMAGE_PREFIX)-clust \
		python mlops/5_clustering/Clustering.py --input-path data/processed --output-path exports/df_cluster.csv

run_encoding:
	docker run --rm \
		$(USER_FLAGS) \
		--network $(NETWORK) \
		-e MLFLOW_TRACKING_URI=$(MLFLOW_URI_DCK) \
		-v $(PWD):/app \
		-w /app \
		-p 8003:8003 \
		$(IMAGE_PREFIX)-encod \
		python /app/mlops/6_Regression/1_Encoding/encoding.py

run_lgbm:
	docker run --rm \
		$(USER_FLAGS) \
		--network $(NETWORK) \
		-e MLFLOW_TRACKING_URI=$(MLFLOW_URI_DCK) \
		-v $(PWD):/app \
		-w /app \
		-p 8004:8004 \
		$(IMAGE_PREFIX)-lgbm \
		python /app/mlops/6_Regression/2_LGBM/train_lgbm.py

run_util:
	docker run --rm \
		$(USER_FLAGS) \
		--network $(NETWORK) \
		-e MLFLOW_TRACKING_URI=$(MLFLOW_URI_DCK) \
		-v $(PWD):/app \
		-w /app \
		-p 8009:8009 \
		$(IMAGE_PREFIX)-util python /app/mlops/6_Regression/3_UTILS/utils.py

run_analyse:
	docker run --rm \
		$(USER_FLAGS) \
		--network $(NETWORK) \
		-e MLFLOW_TRACKING_URI=$(MLFLOW_URI_DCK) \
		-v $(PWD):/app \
		-w /app \
		-p 8005:8005 \
		$(IMAGE_PREFIX)-analyse \
		python /app/mlops/6_Regression/4_Analyse/analyse.py

run_splitst:
	docker run --rm \
		$(USER_FLAGS) \
		--network $(NETWORK) \
		-e MLFLOW_TRACKING_URI=$(MLFLOW_URI_DCK) \
		-v $(PWD):/app \
		-w /app \
		-p 8006:8006 \
		$(IMAGE_PREFIX)-splitst \
		python /app/mlops/7_Serie_temporelle/1_SPLIT/load_split.py

run_decompose:
	docker run --rm \
		$(USER_FLAGS) \
		--network $(NETWORK) \
		-e MLFLOW_TRACKING_URI=$(MLFLOW_URI_DCK) \
		-v $(PWD):/app \
		-w /app \
		-p 8007:8007 \
		$(IMAGE_PREFIX)-decomp \
		python /app/mlops/7_Serie_temporelle/2_Decompose/seasonal_decomp.py --input-folder data/split --output-folder outputs/decomposition

run_SARIMAX:
	docker run --rm \
		$(USER_FLAGS) \
		--network $(NETWORK) \
		-e MLFLOW_TRACKING_URI=$(MLFLOW_URI_DCK) \
		-v $(PWD):/app \
		-w /app \
		-p 8011:8007 \
		$(IMAGE_PREFIX)-sarimax \
		python /app/mlops/7_Serie_temporelle/3_SARIMAX/sarimax_api.py --input-folder data/split --output-folder outputs/best

run_evaluate:
	docker run --rm \
		$(USER_FLAGS) \
		--network $(NETWORK) \
		-e MLFLOW_TRACKING_URI=$(MLFLOW_URI_DCK) \
		-v $(PWD):/app \
		-w /app \
		-p 8008:8008 \
		$(IMAGE_PREFIX)-evalu \
		python /app/mlops/7_Serie_temporelle/4_EVALUATE/evaluate_ST.py --input-folder data/split --model-folder outputs/best --output-folder outputs/evaluate

# ===============================
# 🏗️ DVC: ajout des stages & orchestration
# ===============================

dvc-add-all: add_stage_import add_stage_fusion add_stage_preprocessing add_stage_clustering add_stage_encoding add_stage_lgbm add_stage_utils add_stage_analyse add_stage_splitst add_stage_decompose add_stage_sarimax add_stage_evaluate ## Ajoute tous les stages DVC
	@echo "✅ Tous les stages DVC ont été ajoutés avec succès !"

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

# ===== DVC stages (corrigés) =====
add_stage_import:
	docker run --rm -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n import_data \
	  -d data/raw/merged_sales_data.csv \
	  -o data/df_sample.csv \
	  python mlops/1_import_donnees/import_data.py

add_stage_fusion:
	docker run --rm -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n fusion \
	  -d data/df_sample.csv \
	  -d data/raw/DVF_donnees_macroeco.csv \
	  -o data/df_sales_clean_polars.csv \
	  python mlops/3_fusion/fusion.py

add_stage_preprocessing:
	docker run --rm -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n preprocessing \
	  -d data/df_sales_clean_polars.csv \
	  -o data/train_clean.csv \
	  -o data/test_clean.csv \
	  -o data/df_sales_clean_ST.csv \
	  python mlops/preprocessing_4/preprocessing.py

add_stage_clustering:
	docker run --rm -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n clustering \
	  -d data/train_clean.csv \
	  -d data/test_clean.csv \
	  -d data/df_sales_clean_ST.csv \
	  -o data/df_cluster.csv \
	  python mlops/5_clustering/Clustering.py

add_stage_encoding:
	docker run --rm -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n encoding \
	  -d data/df_cluster.csv \
	  -o data/X_train.csv \
	  -o data/y_train.csv \
	  -o data/X_test.csv \
	  -o data/y_test.csv \
	  python mlops/6_Regression/1_Encoding/encoding.py

add_stage_lgbm:
	docker run --rm -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n lgbm \
	  -d data/X_train.csv \
	  -d data/y_train.csv \
	  -d data/X_test.csv \
	  -d data/y_test.csv \
	  -o exports/reg/model_lgbm.joblib \
	  python mlops/6_Regression/2_LGBM/train_lgbm.py

add_stage_utils:
	docker run --rm -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n utils \
	  python mlops/6_Regression/3_UTILS/utils.py

add_stage_analyse:
	docker run --rm -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n analyse \
	  -d data/X_test.csv \
	  -d data/y_test.csv \
	  -o exports/reg/shap_summary.png \
	  python mlops/6_Regression/4_Analyse/analyse.py

add_stage_splitst:
	docker run --rm -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n splitst \
	  -d data/df_sales_clean_ST.csv \
	  -o data/processed/train_periodique_q12.csv \
	  -o data/processed/test_periodique_q12.csv \
	  python mlops/7_Serie_temporelle/1_SPLIT/load_split.py

add_stage_decompose:
	docker run --rm -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n decompose \
	  -d data/processed/train_periodique_q12.csv \
	  -o exports/st/fig_decompose.png \
	  python mlops/7_Serie_temporelle/2_Decompose/seasonal_decomp.py

add_stage_sarimax:
	docker run --rm -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n sarimax \
	  -d data/processed/train_periodique_q12.csv \
	  -o exports/st/best_model.pkl \
	  python mlops/7_Serie_temporelle/3_SARIMAX/sarimax_api.py

add_stage_evaluate:
	docker run --rm -v $(PWD):/app -w /app $(DVC_IMAGE) \
	  dvc stage add -n evaluate \
	  -d data/processed/train_periodique_q12.csv \
	  -d data/processed/test_periodique_q12.csv \
	  -o exports/st/eval_metrics.json \
	  python mlops/7_Serie_temporelle/4_EVALUATE/evaluate_ST.py

# ===============================
# 🧹 Nettoyage
# ===============================
clean_exports: ## Nettoie les artefacts exports
	rm -rf exports/reg/*.csv exports/reg/*.joblib
	rm -rf exports/st/*.csv exports/st/*.pkl exports/st/*.png exports/st/*.json

clean_dvc: ## Garbage collect DVC (workspace)
	docker run --rm $(USER_FLAGS) -v $(PWD):/app -w /app $(DVC_IMAGE) dvc gc -w --force

clean_all: clean_exports clean_dvc ## Tout nettoyer

# ===============================
# 🧪 Tests
# ===============================
test-ml: ## Tests unitaires ML (si présents)
	@if [ -d "mlops/tests/" ]; then \
		echo "🧪 Tests ML…"; \
		$(PYTHON_BIN) -m pytest mlops/tests/ -v; \
	else \
		echo "❌ Dossier mlops/tests/ non trouvé"; \
	fi

test-all: test-ml api-test ## Tests ML + API

# ===============================
# 🔍 Utilitaires
# ===============================
status: ## Affiche un état rapide de l'environnement
	@echo "========== Statut du projet =========="
	@echo "Dossier : $(PWD)"
	@echo "Python : $$(python3 --version 2>/dev/null || echo 'Non installé')"
	@echo "Env virtuel : $$([ -f .venv/bin/activate ] && echo '✅ Présent' || echo '❌ Absent')"
	@echo "Docker : $$(docker --version 2>/dev/null || echo 'Non installé')"
	@echo "DVC : $$(dvc --version 2>/dev/null || echo 'Non installé')"
	@echo "Données : $$([ -d data ] && echo '✅ Présent' || echo '❌ Absent')"

ports-check: ## Vérifie les ports locaux
	@echo "Port 8000 (API) : $$(lsof -ti:8000 >/dev/null && echo 'Occupé' || echo 'Libre')"
	@echo "Port $(MLFLOW_PORT) (MLflow) : $$(lsof -ti:$(MLFLOW_PORT) >/dev/null && echo 'Occupé' || echo 'Libre')"

# ===============================
#  Airflow (via compose)
# ===============================
airflow: airflow-up airflow-run ## Raccourci


airflow-logs: ## Logs du service Airflow
	@docker logs -f airflow

airflow-run: ## Run un service airflow via compose
	@$(DOCKER_COMPOSE) run --rm airflow

airflow-ps: ## ps des services airflow
	@$(DOCKER_COMPOSE)  ps -a

airflow-down: ## Down propre
	@$(DOCKER_COMPOSE) down airflow

airflow-restart: ## Redémarre Airflow
	@$(DOCKER_COMPOSE) restart airflow
