# ========== Makefile MLOps ==========
# Pipelines Régression + Séries temporelles
# Outils : DVC, MLflow, Docker, bash scripts

# ===============================
# 🧪 Exécution locale (hors Docker)
# ===============================

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

docker_auto: docker_build docker_run_full

docker_build:
	@echo "🔧 Construction de l’image Docker..."
	docker compose build run_full

docker_run_full:
	@echo "🚀 Exécution pipeline complet (Docker)"
	docker compose run --rm run_full

docker_run_regression:
	@echo "🔁 Exécution pipeline Régression (Docker)"
	docker compose run --rm run_full bash mlops/Regression/run_all.sh

docker_run_series:
	@echo "⏳ Exécution pipeline Série Temporelle (Docker)"
	docker compose run --rm run_full bash mlops/Serie_temporelle/run_all_ST.sh

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



