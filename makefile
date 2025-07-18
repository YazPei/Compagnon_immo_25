
# ========== Makefile MLOps ==========
# Pipelines Régression + Séries temporelles
# Outils : DVC, MLflow, Docker, bash scripts
IMAGE_PREFIX=compagnon_immo

# ===============================
# Aide
# ===============================

help: ## Affiche l'aide
	@echo "========== Compagnon Immo - Commandes disponibles =========="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo ""


# ===============================
# 📦 Setup initial
# ===============================

install:
	@echo "📦 Vérification de l'environnement virtuel..."
	@if [ ! -f ".venv/bin/activate" ]; then \
		echo "⚙️  Création de l'environnement virtuel (.venv)"; \
		python3 -m venv .venv; \
	else \
		echo "✅ Environnement virtuel déjà présent"; \
	fi

	@echo "📦 Vérification des paquets installés..."
	@if [ ! -d ".venv/lib" ] || ! . .venv/bin/activate && pip list | grep -Fq -f requirements.txt; then \
		echo "📦 Installation des dépendances..."; \
		. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt; \
	else \
		echo "✅ Dépendances déjà installées"; \
	fi


# ===========================================================
# 🧪 Pipelines ML (Local) pour rendre les scripts exécutables
# ===========================================================

chmod-scripts: ## Rend les scripts exécutables
	@chmod +x mlops/fusion/run_fusion.sh
	@chmod +x mlops/preprocessing/run_preprocessing.sh
	@chmod +x mlops/clustering/run_clustering.sh
	@chmod +x mlops/Regression/run_all.sh
	@chmod +x mlops/Serie_temporelle/run_all_st.sh
	@chmod +x run_all_full.sh

fusion: chmod-scripts ## Fusion des données via DVC
	@echo "Fusion des données via DVC (local)"
	@bash mlops/fusion/run_fusion.sh

preprocessing: chmod-scripts ## Prétraitement des données
	@echo "Prétraitement des données (local via DVC)"
	@bash mlops/preprocessing/run_preprocessing.sh

clustering: chmod-scripts ## Clustering KMeans
	@echo "Lancement du clustering KMeans (local via DVC)"
	@bash mlops/clustering/run_clustering.sh

regression: chmod-scripts ## Pipeline de régression
	@echo "Lancement pipeline Régression (local)"
	@bash mlops/Regression/run_all.sh

series: chmod-scripts ## Pipeline de séries temporelles
	@echo "⏳ Lancement pipeline Série Temporelle (local)"
	@bash mlops/Serie_temporelle/run_all_st.sh

ml-pipeline: fusion preprocessing clustering regression ## Pipeline ML complet
	@echo "Pipeline ML complet terminé"

# ===============================
# 🌐 API et Interface Web
# ===============================

api-dev: check-env ## Démarre l'API en mode développement
	@echo "🚀 Démarrage de l'API..."
	@echo "📍 API : http://localhost:8000"
	@echo "📚 Docs : http://localhost:8000/docs"
	@cd api_test && ../.venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


streamlit: check-env ## Démarre l'interface Streamlit
	@echo "🎨 Démarrage de Streamlit..."
	@echo "📍 Interface : http://localhost:8501"
	@cd api_test && ../.venv/bin/streamlit run questionnaire_streamlit.py --server.port 8501

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

docker-build: ## Construction des images Docker
	@echo "🔧 Construction des images..."
	@docker-compose build

docker-api-build: ## Construction de l'image Docker API
	@echo "Construction de l'image Docker API"
	@cd api_test && docker build -t compagnon-immo-api .

docker-api-run: docker-api-build ## Lance l'API dans Docker
	@echo "Lancement de l'API dans Docker"
	@echo "API disponible sur : http://localhost:8000"
	@docker run -p 8000:8000 --name compagnon-api compagnon-immo-api

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
build-all: build-base build-fusion build-preprocessing build-clustering build-encoding build-lgbm build-analyse build-splitST build-decompose build-SARIMAX build-evaluate
	@echo "📦 Toutes les images Docker ont été construites avec succès !"

docker_build:
	@echo "🔧 Construction de l’image Docker..."
	docker build -f Dockerfile.run -t $(IMAGE_PREFIX)-run .
	
build-base: ## Build de l'image Docker de base (requirements installés)
	docker build -f Dockerfile.dvc -t $(IMAGE_PREFIX)-dvc .


		
build-fusion: ## Build de l'image Docker d'enrichissement du dataset
	docker build -f Dockerfile.fusion -t $(IMAGE_PREFIX)-fus .
	
build-preprocessing: ## Build de l'image Docker de preprocessing
	docker build -f Dockerfile.preprocessing -t $(IMAGE_PREFIX)-preprocess .
	
build-clustering: ## Build de l'image Docker de segmentation geographique
	docker build -f Dockerfile.clustering -t $(IMAGE_PREFIX)-clust .
	
build-encoding: ## Build de l'image Docker d'encoding
	docker build -f Dockerfile.encoding.REG -t $(IMAGE_PREFIX)-encod .
	
build-lgbm: ## Build de l'image Docker de la modelisation Regression
	docker build -f Dockerfile.lgbm.REG -t $(IMAGE_PREFIX)-lgbm .
	
build-util: ## Build de l'image Docker d'interpretabilité
	docker build -f Dockerfile.analyse.REG -t $(IMAGE_PREFIX)-util .
		
build-analyse: ## Build de l'image Docker d'interpretabilité
	docker build -f Dockerfile.analyse.REG -t $(IMAGE_PREFIX)-shap .
	
build-splitst: ## Build de l'image Docker du Split de la serie temporelle
	docker build -f Dockerfile.split.ST -t $(IMAGE_PREFIX)-split-st .
				
build-decompose: ## Build de l'image Docker de la décomposition des courbes
	docker build -f Dockerfile.decompose.ST -t $(IMAGE_PREFIX)-decomp .
			
build-SARIMAX: ## Build de l'image Docker de la modelisation SARIMAX 
	docker build -f Dockerfile.sarimax.ST -t $(IMAGE_PREFIX)-sarimax .		

build-evaluate: ## Build de l'image Docker de l'évaluation du modèle SARIMAX
	docker build -f Dockerfile.evaluate.ST -t $(IMAGE_PREFIX)-evalu .		

run-all-docker: run_full run_dvc run_fusion run_preprocessing run_clustering run_lgbm run_analyse run_splitst run_decompose run_SARIMAX run_evaluate ## lancement de tous les containers 
	@echo "🚀 Pipeline complet exécuté dans Docker !"

run_full:
	@echo "🚀 Exécution pipeline lancement"
	docker run --rm -f $(IMAGE_PREFIX)-run

run_dvc: ## lancement du dvc
	@echo "dvc..."
	docker run --rm -f $(IMAGE_PREFIX)-dvc
											
run_fusion: ## Lancement de la fusion des données (Docker)
	@echo "🌐 Fusion des données IPS et géographiques (Docker)"
	docker run --rm -f $(IMAGE_PREFIX)-fus

run_preprocessing: ## Lancement du preprocessing (Docker)
	@echo "🧼 Exécution preprocessing (Docker)"
	docker run --rm -f $(IMAGE_PREFIX)-preprocess

run_clustering: ## Lancement du clustering (Docker)
	@echo "📊 Exécution du clustering (Docker)"
	docker run --rm -f $(IMAGE_PREFIX)-clust
	
run_lgbm: ## Lancement de la régression LGBM (Docker)
	@echo "🔁 Exécution pipeline Régression (Docker)"
	docker run --rm -f $(IMAGE_PREFIX)-lgbm
	
build-util: ## Build de l'image Docker d'interpretabilité
	docker run --rm -f $(IMAGE_PREFIX)-util
	
run_analyse: ## Build de l'image Docker d'interpretabilité
	docker run --rm -f $(IMAGE_PREFIX)-shap
	
run_splitst: ## Build de l'image Docker du Split de la serie temporelle
	docker run --rm -f $(IMAGE_PREFIX)--split-st
				
run_decompose: ## Build de l'image Docker de la décomposition des courbes
	docker run --rm -f $(IMAGE_PREFIX)-decomp
			
run_SARIMAX: ## Build de l'image Docker de la modelisation SARIMAX 
	docker run --rm -f $(IMAGE_PREFIX)-sarimax		

run_evaluate: ## Build de l'image Docker de l'évaluation du modèle SARIMAX
	docker run --rm -f $(IMAGE_PREFIX)-evalu	

		
# ===============================
# 📁 Commandes DVC (via Docker)
# ===============================
dvc-all: build-dvc-image run-dvc-repro dvc-metrics dvc-push dvc-save ## Reproduit, affiche les métriques et push
	@echo "✅ Pipeline DVC complet exécuté et synchronisé"

	
build-dvc-image: ## Build de l'image Docker DVC + DagsHub
	docker build -f Dockerfile.dvc -t $(IMAGE_PREFIX)-dvc .
		
run-dvc-repro: ## Exécution du pipeline DVC (repro) dans un conteneur DVC
	docker compose run --rm dvc-runner dvc repro
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

mlflow-ui:
	@echo "📈 Démarrage de l’interface MLflow sur http://localhost:5001"
	mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5001

mlflow-run:
	@echo "🚀 Lancement manuel d’un run MLflow (ex: clustering)"
	python -m src.clustering \
		--input-path data/processed/train_clean.csv \
		--output-path data/interim

mlflow-clean:
	@echo "🧹 Suppression du répertoire mlruns/"
	rm -rf mlruns/s

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
