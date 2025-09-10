# ========== Makefile MLOps ==========
# Pipelines Régression + Séries temporelles
# Outils : DVC, MLflow, Docker, bash scripts
.DEFAULT_GOAL := help
IMAGE_PREFIX=compagnon_immo
PYTHON_BIN=.venv/bin/python
DVC_TOKEN ?= default_token_sécurisé_ou_vide
MLFLOW_IMAGE=$(IMAGE_PREFIX)-mlflow
DVC_IMAGE=$(IMAGE_PREFIX)-dvc

.PHONY: prepare-dirs install docker_build help fusion preprocessing clustering regression series \
	ml-pipeline api-dev api-test dev-env mlflow-ui mlflow-clean mlflow-status \
	create-network docker-build docker-api-build docker-api-run docker-stack-up docker-stack-down docker-logs \
	setup_dags docker_auto build-all run-all-docker run_full run_dvc run_fusion run_preprocessing \
	run_clustering run_encoding run_lgbm run_analyse run_splitst run_decompose run_SARIMAX run_evaluate \
	build-dvc-image run-dvc-repro dvc-push dvc-pull dvc-metrics dvc-plots dvc-save \
	clean_exports clean_dvc clean_all mlflow-run mlflow-log-status test-ml test-all \
	full-stack quick-start status ports-check dvc_pull_raw dvc_repro_all dvc_push_all pipeline_reset_all \
	run_dvc_pull_raw run_dvc_repro_all run_dvc_push_all run_pipeline_reset_all

# ===============================
# Aide
# ===============================
lint:
	@echo "🔍 Vérification du Makefile..."
	@grep -o '^[a-zA-Z_-]\+:' Makefile | sort | uniq -d | xargs -r -I{} echo "⚠️ Duplicate target: {}"

help: ## Affiche l'aide
	@echo "========== Compagnon Immo - Commandes disponibles =========="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ===============================
# 📦 Ligne de commande - test
# ===============================
quick-start-pipeline: build-all run-all-docker
quick-start-test: quick-start-pipeline run_dvc_final
# ===============================
# 🌐 API et Interface Web
# ===============================
api-dev: check-env ## Démarre l'API en mode développement
	@echo "🚀 Démarrage de l'API..."
	@echo "📍 API : http://localhost:8000"
	@echo "📚 Docs : http://localhost:8000/docs"
	nohup PYTHONPATH=api_test uvicorn app.routes.main:app --reload --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &

api-test: check-env ## Lance les tests de l'API
	@echo "🧪 Tests de l'API..."
	@cd api_test && ../.venv/bin/python -m pytest app/tests/ -v


# ===============================
# 📈 MLflow
# ===============================
mlflow-ui: check-env ## Démarre l'interface MLflow
	@../.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5050

mlflow-clean: ## Nettoie les runs MLflow
	@rm -rf mlruns/

mlflow-status: ## Affiche le statut des derniers runs
	@find mlruns/ -name "meta.yaml" 2>/dev/null | xargs grep -H "status" || echo "Aucun run trouvé"

# ===============================
# 🐳 Docker - API
# ===============================
prepare-dirs:
	@mkdir -p data exports mlruns
	@touch data/.gitkeep

docker-build: prepare-dirs
	@docker-compose build

docker-api-build:
	@cd api_test && docker build -t compagnon-immo-api .

docker-api-run: docker-api-build
	@docker run -d -p 8000:8000 --name compagnon-api compagnon-immo-api

docker-api-stop:
	@docker rm -f compagnon-api 2>/dev/null || echo "Aucun conteneur compagnon-api à supprimer"

docker-stack-up:
	@docker-compose up -d

docker-stack-down:
	@docker-compose down

docker-logs:
	@docker-compose logs -f

# ===============================
# ☁️ DagsHub Setup
# ===============================
## @ketsia: faut-il garder cette partie?
setup_dags:
	@chmod +x setup_env.sh
	@./setup_env.sh

# ===============================
# DVC Dockerisé
# ===============================


run_pipeline_reset_all: run_dvc run_dvc_final add_stage_all run_dvc_final

# ===============================
# 📈️ MLflow Dockerisé
# ===============================
mlflow-dockerized: create-network build-mlflow mlflow-up
create-network:
	@docker network create ml_net || echo "✅ Réseau 'ml_net' déjà existant"
#@ketsia testé
build-mlflow: 
	docker build -f mlops/mlflow/Dockerfile.mlflow -t $(MLFLOW_IMAGE) .
#@ketsia testé
mlflow-up:
	docker run -d --rm \
		--name compagnon_immo-mlflow \
		--network ml_net \
		-v $(PWD)/mlruns:/mlflow/mlruns \
		-p 5050:5050 \
		$(MLFLOW_IMAGE) \
		mlflow server --host 0.0.0.0 --port 5050 \
		  --backend-store-uri sqlite:////mlflow/mlruns/mlflow.db \
		  --default-artifact-root /mlflow/mlruns


#@ketsia testé
mlflow-down:
	docker stop compagnon_immo-mlflow || true
#@ketsia testé
# ===============================
# 🐳 Exécution pipeline Docker
# ===============================
docker_auto: build-all run-all-docker

build-all:  docker_build build-base build-fusion build-preprocessing build-clustering build-encoding build-lgbm build-util build-analyse build-splitST build-decompose build-SARIMAX build-evaluate


docker_build:
	docker build -f mlops/1_import_donnees/Dockerfile.run -t $(IMAGE_PREFIX)-run .
#@ketsia testé
build-base:
	docker build -f mlops/2_dvc/Dockerfile.dvc -t $(IMAGE_PREFIX)-dvc .
	
#@ketsia testé
build-fusion:
	docker build -f mlops/3_fusion/Dockerfile.fusion -t $(IMAGE_PREFIX)-fus .
#@ketsia testé
build-preprocessing:
	docker build -f mlops/preprocessing_4/Dockerfile.preprocessing -t $(IMAGE_PREFIX)-preprocess .
#@ketsia testé
build-clustering:
	docker build -f mlops/5_clustering/Dockerfile.clustering -t $(IMAGE_PREFIX)-clust .

build-encoding:
	docker build -f mlops/6_Regression/1_Encoding/Dockerfile.encoding.REG -t $(IMAGE_PREFIX)-encod .

build-lgbm:
	docker build -f mlops/6_Regression/2_LGBM/Dockerfile.lgbm.REG -t $(IMAGE_PREFIX)-lgbm .
build-util:
	docker build -f mlops/6_Regression/3_UTILS/Dockerfile.util.REG -t $(IMAGE_PREFIX)-util .
build-analyse:
	docker build -f mlops/6_Regression/4_Analyse/Dockerfile.analyse.REG -t $(IMAGE_PREFIX)-analyse ..

build-splitST:
	docker build -f mlops/7_Serie_temporelle/1_SPLIT/Dockerfile.split.ST -t $(IMAGE_PREFIX)-splitst .

build-decompose:
	docker build -f mlops/7_Serie_temporelle/2_Decompose/Dockerfile.decompose.ST -t $(IMAGE_PREFIX)-decomp .

build-SARIMAX:
	docker build -f mlops/7_Serie_temporelle/3_SARIMAX/Dockerfile.sarimax.ST -t $(IMAGE_PREFIX)-sarimax .

build-evaluate:
	docker build -f mlops/7_Serie_temporelle/4_EVALUATE/Dockerfile.evaluate.ST -t $(IMAGE_PREFIX)-evalu .

run-all-docker: run_full run_dvc run_fusion run_preprocessing run_clustering run_lgbm run_util run_analyse run_splitst run_decompose run_SARIMAX run_evaluate 

run_full:
	docker run --rm $(IMAGE_PREFIX)-run
#@ketsia testé

run_dvc:
	docker run --rm \
		--env-file .env.yaz \
		--env MLFLOW_TRACKING_URI=file:///app/mlruns \
		--network ml_net \
		-v $(PWD):/app \
		-w /app \
		$(IMAGE_PREFIX)-dvc \
		bash mlops/2_dvc/run_dvc.sh


#@ketsia testé

run_fusion:
	docker run \
		--rm \
		--user $(id -u):$(id -g) \  # pour les permissions à runner avec son propre user
		-v $(PWD)/data:/app/data \
		$(IMAGE_PREFIX)-fus \
		bash run_fusion.sh

run_preprocessing:
	export PYTHONPATH=$(PWD) && \
	docker run -p 8001:8001 --rm $(IMAGE_PREFIX)-preprocess

run_clustering:
	docker run -p 8002:8002 --rm $(IMAGE_PREFIX)-clust

run_encoding:
	docker run -p 8003:8003 --rm $(IMAGE_PREFIX)-encod

run_lgbm:
	docker run -p 8004:8004 --rm $(IMAGE_PREFIX)-lgbm

run_util:
	docker run -p 8009:8009 --rm $(IMAGE_PREFIX)-util

run_analyse:
	docker run -p 8005:8005 --rm $(IMAGE_PREFIX)-analyse

run_splitst:
	docker run -p 8006:8006 --rm $(IMAGE_PREFIX)-splitst

run_decompose:
	docker run -p 8007:8007 --rm $(IMAGE_PREFIX)-decomp

run_SARIMAX:
	docker run -p 8011:8007 --rm $(IMAGE_PREFIX)-sarimax

run_evaluate:
	docker run -p 8008:8008 --rm $(IMAGE_PREFIX)-evalu

# ===============================
# 🏗️ Construction des stages DVC
# ===============================
run_dvc_final: add_stage_all run_dvc_final
# Ajoute tous les stages DVC en cascade
add_stage_all: \
	add_stage_import \
	add_stage_fusion \
	add_stage_preprocessing \
	add_stage_clustering \
	add_stage_encoding \
	add_stage_lgbm \
	add_stage_utils \
	add_stage_analyse \
	add_stage_splitst \
	add_stage_decompose \
	add_stage_sarimax \
	add_stage_evaluate
	@echo "✅ Tous les stages DVC ont été ajoutés avec succès !"
	
# ===============================
# DVC Stage
# ===============================

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
	python mlops/3_fusion_geo/fusion.py

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
	-o data/df_sales_clean_ST.csv \
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
	-o exports/reg/model_lgbm.joblib
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
	-o data/exports/reg/shap_summary.png \
	python mlops/6_Regression/4_Analyse/analyse.py

add_stage_splitst:
	docker run --rm -v $(PWD):/app -w /app $(DVC_IMAGE) \
	dvc stage add -n splitst \
	-d data/processed/X_test.csv \
	-d data/processed/y_test.csv \
	-o data/processed/train_periodique_q12.csv \
	-o data/processed/test_periodique_q12.csv \
	train_clean_ST
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
	dvc stage add -n analyse \
	-d data/processed/train_periodique_q12.csv \
	-d data/processed/test_periodique_q12.csv \
	-o exports/st/eval_metrics.json \
	python mlops/7_Serie_temporelle/4_EVALUATE/evaluate_ST.py
	
USER_FLAGS=--user $(shell id -u):$(shell id -g)

run_dvc:
	docker run --rm \
		$(USER_FLAGS) \
		--env-file .env.yaz \
		--network ml_net \
		-v $(PWD):/app \
		-w /app \
		$(IMAGE_PREFIX)-dvc \
		bash mlops/2_dvc/run_dvc.sh


# ===============================
# 🧹 Nettoyage
# ===============================
clean_exports:
	rm -rf exports/reg/*.csv exports/reg/*.joblib
	rm -rf exports/st/*.csv exports/st/*.pkl exports/st/*.png exports/st/*.json

clean_dvc:
	dvc gc -w --force

clean_all: clean_exports clean_dvc

# ===============================
# 🧪 Tests
# ===============================
test-ml:
	@if [ -d "mlops/tests/" ]; then \
		echo "🧪 Tests ML..."; \
		pytest mlops/tests/ -v; \
	else \
		echo "❌ Dossier mlops/tests/ non trouvé"; \
	fi

test-all: test-ml api-test

# ===============================
# 🔍 Utilitaires
# ===============================
status:
	@echo "========== Statut du projet =========="
	@echo "Dossier : $(PWD)"
	@echo "Python : $$(python3 --version 2>/dev/null || echo 'Non installé')"
	@echo "Env virtuel : $$([ -f .venv/bin/activate ] && echo '✅ Présent' || echo '❌ Absent')"
	@echo "Docker : $$(docker --version 2>/dev/null || echo 'Non installé')"
	@echo "DVC : $$(dvc --version 2>/dev/null || echo 'Non installé')"
	@echo "Données : $$([ -d data ] && echo '✅ Présent' || echo '❌ Absent')"

ports-check:
	@echo "Port 8000 (API) : $$(lsof -ti:8000 && echo 'Occupé' || echo 'Libre')"
	@echo "Port 5050 (MLflow) : $$(lsof -ti:5050 && echo 'Occupé' || echo 'Libre')"

