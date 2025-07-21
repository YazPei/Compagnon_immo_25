
# ========== Makefile MLOps ==========
# Pipelines Régression + Séries temporelles
# Outils : DVC, MLflow, Docker, bash scripts
IMAGE_PREFIX=compagnon_immo
PYTHON_BIN=.venv/bin/python
DVC_TOKEN ?= default_token_sécurisé_ou_vide
.PHONY: prepare-dirs install docker_build help fusion preprocessing clustering regression series \
	ml-pipeline api-dev streamlit api-test dev-env mlflow-ui mlflow-clean mlflow-status \
	docker-build docker-api-build docker-api-run docker-stack-up docker-stack-down docker-logs \
	setup_dags docker_auto build-all run-all-docker run_full run_dvc run_fusion run_preprocessing \
	run_clustering run_lgbm run_analyse run_splitst run_decompose run_SARIMAX run_evaluate \
	build-dvc-image run-dvc-repro dvc-push dvc-pull dvc-metrics dvc-plots dvc-save \
	clean_exports clean_dvc clean_all mlflow-run mlflow-log-status test-ml test-all \
	full-stack quick-start status ports-check
	
# ===============================
# Aide
# ===============================

help: ## Affiche l'aide
	@echo "========== Compagnon Immo - Commandes disponibles =========="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo ""


# ===============================
# 📦 ligne de commande - test
# ===============================

quick-start-pipeline: build-all run-all-docker

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

dev-env: ## Environnement de développement complet
	@echo "🛠️ Démarrage de l'environnement complet..."
	@echo "Démarrage en parallèle : MLflow + API + Streamlit"
	@make -j3 mlflow-ui api-dev streamlit


# ===============================
# 📈 MLflow
# ===============================

mlflow-ui: check-env ## Démarre l'interface MLflow
	@echo "📈 Démarrage de l'interface MLflow"
	@echo "📍 MLflow UI disponible sur : http://localhost:5001"
	@../.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5001

mlflow-clean: ## Nettoie les runs MLflow
	@echo "🧹 Suppression du répertoire mlruns/"
	@rm -rf mlruns/

mlflow-status: ## Affiche le statut des derniers runs
	@echo "📜 Derniers runs MLflow"
	@find mlruns/ -name "meta.yaml" 2>/dev/null | xargs grep -H "status" || echo "Aucun run trouvé"


# ===============================
# 🐳 Docker - API
# ===============================
prepare-dirs: ## Crée les dossiers nécessaires au projet
	@echo "📁 Préparation des dossiers requis..."
	@mkdir -p data exports mlruns
	@touch data/.gitkeep

docker-build: prepare-dirs ## Construction des images Docker
	@echo "🔧 Construction des images..."
	@docker-compose build

docker-api-build: ## Construction de l'image Docker API
	@echo "Construction de l'image Docker API"
	@cd api_test && docker build -t compagnon-immo-api .

docker-api-run: docker-api-build ## Lance l'API dans Docker
	@echo "Lancement de l'API dans Docker"
	@echo "API disponible sur : http://localhost:8000"
	@docker run -d -p 8000:8000 --name compagnon-api compagnon-immo-api

docker-api-stop: ## Stoppe et supprime le conteneur API s'il existe
	@docker rm -f compagnon-api 2>/dev/null || echo "Aucun conteneur compagnon-api à supprimer"


docker-stack-up: ## Démarre la stack Docker complète
	@echo "🐳 Démarrage de la stack..."
	@docker-compose up -d
	@echo "✅ Stack démarrée"

docker-stack-down: ## Arrête la stack Docker
	@echo "🛑 Arrêt de la stack..."
	@docker-compose down

docker-logs: ## Affiche les logs Docker
	@docker-compose logs -f

# ===============================
# ☁️ Setup remote DagsHub
# ===============================

setup_dags:  ## Configure le remote DVC vers DagsHub (secure, local only)
	@echo "☁️ Configuration du remote DVC (DagsHub sécurisé)"
	@chmod +x setup_remote.sh
	@./setup_remote.sh

# 
# ===============================
# 🐳 Exécution dans Docker
# ===============================


docker_auto: build-all run-all-docker

## builds ##
build-all: chmod-dvc-sh docker_build build-base build-fusion build-preprocessing build-clustering build-encoding build-lgbm build-analyse build-splitST build-decompose build-SARIMAX build-evaluate
	@echo "📦 Toutes les images Docker ont été construites avec succès !"

chmod-dvc-sh: ## Rend exécutable run_dvc.sh sur l'hôte
	@chmod +x mlops/2_dvc/run_dvc.sh

docker_build:
	@echo "🔧 Construction de l’image Docker..."
	docker build -f mlops/1_import_donnees/Dockerfile.run -t $(IMAGE_PREFIX)-run .

build-base: ## Build de l'image Docker de base (requirements installés)
	docker build -f mlops/2.dvc/Dockerfile.dvc -t $(IMAGE_PREFIX)-dvc .

build-fusion: ## Build de l'image Docker d'enrichissement du dataset
	docker build -f mlops/3.fusion/Dockerfile.fusion -t $(IMAGE_PREFIX)-fus .

build-preprocessing: ## Build de l'image Docker de preprocessing
	docker build -f mlops/4.preprocessing/Dockerfile.preprocessing -t $(IMAGE_PREFIX)-preprocess .

build-clustering: ## Build de l'image Docker de segmentation geographique
	docker build -f mlops/5.clustering/Dockerfile.clustering -t $(IMAGE_PREFIX)-clust .

build-encoding: ## Build de l'image Docker d'encoding
	docker build -f mlops/6.Regression/1.Encoding/Dockerfile.encoding.REG -t $(IMAGE_PREFIX)-encod .

build-lgbm: ## Build de l'image Docker de la modelisation Regression
	docker build -f mlops/6.Regression/2.LGBM/Dockerfile.lgbm.REG -t $(IMAGE_PREFIX)-lgbm .

build-util: ## Build de l'image Docker d'interpretabilité
	docker build -f mlops/6.Regression/3.UTILS/Dockerfile.util.REG -t $(IMAGE_PREFIX)-util .
		
build-analyse: ## Build de l'image Docker d'interpretabilité
	docker build -f mlops/6.Regression/4.Analyse/Dockerfile.analyse.REG -t $(IMAGE_PREFIX)-split-st .
	
build-splitst: ## Build de l'image Docker du Split de la serie temporelle
	docker build -f mlops/7.Serie_temporelle/1.SPLIT/Dockerfile.split.ST -t $(IMAGE_PREFIX)-decomp .

build-decompose: ## Build de l'image Docker de la décomposition des courbes
	docker build -f mlops/7.Serie_temporelle/2.Decompose/Dockerfile.decompose.ST -t $(IMAGE_PREFIX)-sarimax .		

build-SARIMAX: ## Build de l'image Docker de la modelisation SARIMAX 
	docker build -f mlops/7.Serie_temporelle/3.SARIMAX/Dockerfile.sarimax.ST -t $(IMAGE_PREFIX)-sarimax .		
	
build-evaluate: ## Build de l'image Docker de l'évaluation du modèle SARIMAX
	docker build -f mlops/7.Serie_temporelle/4.EVALUATE/Dockerfile.evaluate.ST -t $(IMAGE_PREFIX)-evalu .		

run-all-docker: run_full run_dvc run_fusion run_preprocessing run_clustering run_lgbm run_util run_analyse run_splitst run_decompose run_SARIMAX run_evaluate ## lancement de tous les containers 
	@echo "🚀 Pipeline complet exécuté dans Docker !"

run_full:
	@echo "🚀 Exécution pipeline lancement"
	docker run --rm $(IMAGE_PREFIX)-run


run_dvc: chmod-dvc-sh ## lancement du dvc
	@echo "🧠 Lancement DVC avec script run_dvc.sh (Docker)"
	docker run --rm \
		-e DVC_TOKEN=$(DVC_TOKEN) \
		-v $(PWD):/app \
		-w /app \
		$(IMAGE_PREFIX)-dvc
													
run_fusion: ## Lancement de la fusion des données (Docker)
	@echo "🌐 Fusion des données IPS et géographiques (Docker)"
	docker run --rm $(IMAGE_PREFIX)-fus

run_preprocessing: ## Lancement du preprocessing (Docker)
	@echo "🧼 Exécution preprocessing (Docker)"
	docker run --rm $(IMAGE_PREFIX)-preprocess

run_clustering: ## Lancement du clustering (Docker)
	@echo "📊 Exécution du clustering (Docker)"
	docker run --rm $(IMAGE_PREFIX)-clust

run_encoding:
	docker run --rm $(IMAGE_PREFIX)-encod

	
run_lgbm: ## Lancement de la régression LGBM (Docker)
	@echo "🔁 Exécution pipeline Régression (Docker)"
	docker run --rm $(IMAGE_PREFIX)-lgbm
	
run-util: ## Build de l'image Docker d'interpretabilité
	docker run --rm $(IMAGE_PREFIX)-util
	
run_analyse: ## Build de l'image Docker d'interpretabilité
	docker run --rm $(IMAGE_PREFIX)-shap

run_splitst: ## Split série temporelle
	docker run --rm $(IMAGE_PREFIX)-split-st

				
run_decompose: ## Build de l'image Docker de la décomposition des courbes
	docker run --rm $(IMAGE_PREFIX)-decomp
			
run_SARIMAX: ## Build de l'image Docker de la modelisation SARIMAX 
	docker run --rm $(IMAGE_PREFIX)-sarimax		

run_evaluate: ## Build de l'image Docker de l'évaluation du modèle SARIMAX
	docker run --rm $(IMAGE_PREFIX)-evalu	

		
# ===============================
# 📁 Commandes DVC (via Docker)
# ===============================
dvc-all: build-dvc-image run-dvc-repro dvc-metrics dvc-push dvc-save ## Reproduit, affiche les métriques et push
	@echo "✅ Pipeline DVC complet exécuté et synchronisé"

build-cache: .venv/.pip_installed ## Build dépendances si requirements.txt modifié
	@echo "✅ Build cache terminé (si besoin)"
		
build-dvc-image: ## Build de l'image Docker DVC + DagsHub
	docker build -f Dockerfile.dvc -t $(IMAGE_PREFIX)-dvc .
	
run_dvc: ## lancement du dvc
	@echo "🧠 Lancement DVC avec script run_dvc.sh (Docker)"
	docker run --rm \
		-e DVC_TOKEN=$(DVC_TOKEN) \
		-v $(PWD):/app \
		-w /app \
		$(IMAGE_PREFIX)-dvc
		
run-dvc-repro: ## Exécution du pipeline DVC (repro) dans un conteneur DVC
	docker run --rm \
		-v $(PWD):/app \
		-v ~/.dvc/config.local:/app/.dvc/config.local:ro \
		-w /app \
		$(IMAGE_PREFIX)-dvc \
		dvc repro

	####	
dvc-push: ## Push vers DagsHub en Docker
	docker run --rm \
		-v $(PWD):/app \
		-v ~/.dvc/config.local:/app/.dvc/config.local:ro \
		-w /app \
		$(IMAGE_PREFIX)-dvc \
		push


dvc-pull: ## Pull depuis DagsHub en Docker
	docker run --rm \
		-v $(PWD):/app \
		-v ~/.dvc/config.local:/app/.dvc/config.local:ro \
		-w /app \
		$(IMAGE_PREFIX)-dvc \
		pull

dvc-metrics: ## Affiche les métriques DVC via Docker
	docker run --rm \
		-v $(PWD):/app \
		-v ~/.dvc/config.local:/app/.dvc/config.local:ro \
		-w /app \
		$(IMAGE_PREFIX)-dvc \
		metrics show
		
dvc-plots: ## Génère les graphiques DVC (plots.html) via Docker
	docker run --rm \
		-v $(PWD):/app \
		-v ~/.dvc/config.local:/app/.dvc/config.local:ro \
		-w /app \
		$(IMAGE_PREFIX)-dvc \
		plots show --html > plots.html && echo "Fichier 'plots.html' généré"
		
dvc-save: ## Ajoute, commit et tag un fichier .dvc modifié
	dvc add data/processed/train_clean.csv
	git add data/processed/train_clean.csv.dvc
	git commit -m "DVC update train_clean"
					
# ===============================
# 🧹 Nettoyage
# ===============================

clean_exports:
	@echo "🧹 Suppression des fichiers exports/"
	rm -rf exports/reg/*.csv exports/reg/*.joblib
	rm -rf exports/st/*.csv exports/st/*.pkl exports/st/*.png exports/st/*.json

clean_dvc:
	@echo "🧹 Nettoyage DVC cache non utilisé (local uniquement)"
	dvc gc -w --force

clean_all: clean_exports clean_dvc




# ===============================
# 📈 Tracking MLflow
# ===============================

mlflow-run:
	@echo "🚀 Lancement manuel d’un run MLflow (ex: clustering)"
	python -m src.clustering \
		--input-path data/processed/train_clean.csv \
		--output-path data/interim

mlflow-clean:
	@echo "🧹 Suppression du répertoire mlruns/"
	rm -rf mlruns/

mlflow-log-status:
	@echo "📜 Derniers runs MLflow"
	@find mlruns/ -name "meta.yaml" | xargs grep -H "status"


# ===============================
# 🧪 Tests complets
# ===============================

test-ml: check-env ## Tests des pipelines ML
	@echo "🧪 Tests ML..."
	@pytest mlops/tests/ -v || echo "Dossier mlops/tests/ non trouvé"

test-all: test-ml api-test ## Tous les tests
	@echo "✅ Tous les tests terminés"

# ===============================
# 🚀 Pipelines complets
# ===============================

full-stack: install ml-pipeline api-dev ## Pipeline ML + API
	@echo "🎉 Pipeline complet + API démarrés !"

quick-start: install full-stack streamlit ## Installation et démarrage rapide
	@echo "🚀 Projet prêt à utiliser !"

# ===============================
# 🔍 Utilitaires
# ===============================

status: ## Affiche le statut du projet
	@echo "========== Statut du projet =========="
	@echo "Dossier : $(PWD)"
	@echo "Python : $$(python3 --version 2>/dev/null || echo 'Non installé')"
	@echo "Env virtuel : $$([ -f .venv/bin/activate ] && echo '✅ Présent' || echo '❌ Absent')"
	@echo "Docker : $$(docker --version 2>/dev/null || echo 'Non installé')"
	@echo "DVC : $$(dvc --version 2>/dev/null || echo 'Non installé')"
	@echo "Données : $$([ -d data ] && echo '✅ Présent' || echo '❌ Absent')"

ports-check: ## Vérifie les ports utilisés
	@echo "Vérification des ports..."
	@echo "Port 8000 (API) : $$(lsof -ti:8000 && echo 'Occupé' || echo 'Libre')"
	@echo "Port 8501 (Streamlit) : $$(lsof -ti:8501 && echo 'Occupé' || echo 'Libre')"
	@echo "Port 5001 (MLflow) : $$(lsof -ti:5001 && echo 'Occupé' || echo 'Libre')"
