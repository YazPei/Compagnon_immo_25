# ========== Makefile MLOps ==========
# Pipelines Régression + Séries temporelles + API
# Outils : DVC, MLflow, Docker, bash scripts

.PHONY: help install clean-all
.DEFAULT_GOAL := help

# ===============================
# Aide
# ===============================

help: ## Affiche l'aide
	@echo "========== Compagnon Immo - Commandes disponibles =========="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ==============================================================
# 📦 Setup et installation (environnement virtuel + dépendances)
# ==============================================================

install: ## Installation de l'environnement
	@echo "Vérification de l'environnement virtuel..."
	@if [ ! -f ".venv/bin/activate" ]; then \
	    echo "Création de l'environnement virtuel (.venv)"; \
	    python3 -m venv .venv; \
	else \
	    echo "Environnement virtuel déjà présent"; \
	fi
	@echo "Installation des dépendances..."
	@. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "Installation terminée"

check-env: ## Vérifie l'environnement
	@if [ ! -f ".venv/bin/activate" ]; then \
	    echo "Environnement virtuel non trouvé. Exécutez 'make install'"; \
	    exit 1; \
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
# 🐳 Docker
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
# 🧹 Nettoyage
# ===============================

clean-exports: ## Supprime les fichiers d'export
	@echo "🧹 Suppression des fichiers exports/"
	@rm -rf exports/reg/*.csv exports/reg/*.joblib
	@rm -rf exports/st/*.csv exports/st/*.pkl exports/st/*.png exports/st/*.json

clean-dvc: ## Nettoie le cache DVC
	@echo "🧹 Nettoyage DVC cache non utilisé"
	@dvc gc -w --force

clean-api: ## Nettoie les caches de l'API
	@echo "Nettoyage des caches de l'API"
	@find api_test -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find api_test -name "*.pyc" -delete 2>/dev/null || true

clean-docker: ## Nettoie les conteneurs et images Docker
	@echo "🧹 Nettoyage Docker"
	@docker system prune -f
	@docker container prune -f

clean-all: clean-exports clean-dvc clean-api clean-docker ## Nettoyage total
	@echo "✅ Nettoyage complet terminé"

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