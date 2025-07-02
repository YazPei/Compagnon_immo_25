# ========== Makefile MLOps ==========
# Pipelines Régression + Séries temporelles
# Outils : DVC, MLflow, Docker, bash scripts

# ===============================
# 🧪 Exécution locale (hors Docker)
# ===============================
chmod_all:
	chmod +x mlops/fusion/run_fusion.sh
	chmod +x mlops/preprocessing/run_preprocessing.sh
	chmod +x mlops/clustering/run_clustering.sh
	chmod +x mlops/Regression/run_all.sh
	chmod +x mlops/Serie_temporelle/run_all_st.sh
	chmod +x run_all_full.sh

fusion_dvc:
	@echo "🌐 Fusion des données via DVC (local)"
	chmod +x mlops/fusion/run_fusion.sh
	bash mlops/fusion/run_fusion.sh
	
preprocessing:
	@echo "🧼 Prétraitement des données (local via DVC)"
	chmod +x mlops/preprocessing/run_preprocessing.sh
	bash mlops/preprocessing/run_preprocessing.sh

clustering:
	@echo "📊 Lancement du clustering KMeans (local via DVC)"
	chmod +x mlops/clustering/run_clustering.sh
	bash mlops/clustering/run_clustering.sh

regression:
	@echo "🔁 Lancement pipeline Régression (local)"
	bash mlops/Regression/run_all.sh

series:
	@echo "⏳ Lancement pipeline Série Temporelle (local)"
	bash mlops/Serie_temporelle/run_all_st.sh

full:
	@echo "🧠 Lancement pipeline Complet (local)"
	bash run_all_full.sh

mlflow-ui:
	@echo "📈 Démarrage de l’interface MLflow sur http://localhost:5001"
	mlflow ui --port 5001

# ===============================
# 🐳 Exécution dans Docker
# ===============================

docker_auto: docker_build docker_run_fusion docker_run_full docker_run_clustering docker_run_preprocessing

	
docker_build:
	@echo "🔧 Construction de l’image Docker..."
	docker compose build run_full

docker_run_full:
	@echo "🚀 Exécution pipeline complet (Docker)"
	docker compose run --rm run_full

docker_run_preprocessing:
	@echo "🧼 Exécution preprocessing (Docker)"
	docker compose run --rm preprocessing


docker_run_clustering:
	@echo "📊 Exécution du clustering (Docker)"
	docker compose run --rm clustering

docker_run_regression:
	@echo "🔁 Exécution pipeline Régression (Docker)"
	docker compose run --rm run_full bash mlops/Regression/run_all.sh

docker_run_series:
	@echo "⏳ Exécution pipeline Série Temporelle (Docker)"
	docker compose run --rm run_full bash mlops/Serie_temporelle/run_all_ST.sh


docker_run_fusion:
	@echo "🌐 Fusion des données IPS et géographiques (Docker)"
	docker compose run --rm fusion_geo
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

