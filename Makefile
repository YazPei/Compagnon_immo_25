# ========== Makefile MLOps - Compagnon Immo ==========
# Gestion des pipelines avec Airflow, MLflow, DVC et Docker

# Variables
IMAGE_PREFIX := compagnon_immo
NETWORK := ml_net
MLFLOW_PORT := 5050
MLFLOW_HOST := $(IMAGE_PREFIX)-mlflow
MLFLOW_URI_DCK := http://$(MLFLOW_HOST):$(MLFLOW_PORT)
AIRFLOW_SERVICES := postgres-airflow airflow-webserver airflow-scheduler
DOCKER_COMPOSE_CMD := docker compose

# Couleurs
COLOR_RESET := \033[0m
COLOR_GREEN := \033[32m
COLOR_RED := \033[31m
COLOR_YELLOW := \033[33m

.PHONY: help build start stop test clean logs setup

# Commandes principales
help: ## Affiche l'aide
	@echo "========== Compagnon Immo - Commandes disponibles =========="
	@grep -E '^[a-zA-Z0-9_.-]+:.*?##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Installation complète (dépendances + permissions)
	@mkdir -p data exports mlruns logs/airflow
	@pip install -r requirements.txt
	@sudo chown -R $(shell whoami):$(shell whoami) . || true
	@chmod -R u+rwx . || true
	@echo "$(COLOR_GREEN)✅ Setup terminé$(COLOR_RESET)"

build: ## Build toutes les images Docker
	@echo "$(COLOR_YELLOW)🔨 Building images...$(COLOR_RESET)"
	@$(DOCKER_COMPOSE_CMD) build
	@echo "$(COLOR_GREEN)✅ Build terminé$(COLOR_RESET)"

start: ## Démarre tous les services
	@echo "$(COLOR_YELLOW)🚀 Starting services...$(COLOR_RESET)"
	@docker network create $(NETWORK) || true
	@$(DOCKER_COMPOSE_CMD) up -d
	@docker run -d --rm --name $(MLFLOW_HOST) --network $(NETWORK) \
		-v $(PWD)/mlruns:/mlflow/mlruns -p $(MLFLOW_PORT):$(MLFLOW_PORT) \
		ghcr.io/mlflow/mlflow:v2.13.1 mlflow server --host 0.0.0.0 \
		--port $(MLFLOW_PORT) --backend-store-uri sqlite:////mlflow/mlruns/mlflow.db \
		--default-artifact-root /mlflow/mlruns
	@echo "$(COLOR_GREEN)✅ Services démarrés$(COLOR_RESET)"

stop: ## Arrête tous les services
	@echo "$(COLOR_YELLOW)🛑 Stopping services...$(COLOR_RESET)"
	-@$(DOCKER_COMPOSE_CMD) down --remove-orphans || true
	-docker stop $(MLFLOW_HOST) || true
	-docker rm $(MLFLOW_HOST) || true
	@echo "$(COLOR_GREEN)✅ Services arrêtés$(COLOR_RESET)"





# ===============================
# 5. Tests & CI
# ===============================
api-test: ## Lancer les tests de l'API dans un environnement Docker complet
	@echo "🐳 Tests de l'API avec Docker…"
	@echo "🚀 Démarrage de l'environnement de test complet..."
	@$(DOCKER_COMPOSE_CMD) --profile test up --build --abort-on-container-exit --exit-code-from api-test --quiet-pull
	@echo "🛑 Nettoyage de l'environnement de test..."
	@$(DOCKER_COMPOSE_CMD) --profile test down -v

api-test-fast: ## Lancer les tests de l'API rapidement (sans rebuild si images existent)
	@echo "⚡ Tests de l'API rapides avec Docker…"
	@echo "🚀 Démarrage de l'environnement de test (utilise les images existantes)..."
	@$(DOCKER_COMPOSE_CMD) --profile test up --abort-on-container-exit --exit-code-from api-test --quiet-pull
	@echo "🛑 Nettoyage de l'environnement de test..."
	@$(DOCKER_COMPOSE_CMD) --profile test down -v

ci-test: ## Exécute les tests CI dans Docker
	@echo "🔍 Lancement des tests CI dans Docker..."
	@$(DOCKER_COMPOSE_CMD) --profile ci up --build --abort-on-container-exit --exit-code-from ci --quiet-pull
	@echo "🛑 Nettoyage de l'environnement CI..."
	@$(DOCKER_COMPOSE_CMD) --profile ci down -v
	@echo "$(COLOR_GREEN)✅ Tous les tests CI ont réussi !$(COLOR_RESET)"

# ===============================
# 6. Arrêt & nettoyage
# ===============================

mlflow-down: ## Stoppe MLflow
	docker stop $(MLFLOW_HOST) || true

airflow-down: ## Stoppe Airflow
	docker compose stop $(AIRFLOW_SERVICES)
	docker compose rm -f $(AIRFLOW_SERVICES) || true

stop-all: ## Stoppe tous les services, conteneurs, réseaux et processus liés au projet
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
	docker compose --profile airflow run --rm airflow-webserver airflow db upgrade
	docker compose --profile airflow run --rm airflow-webserver \
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

# ===============================
# ☁️ Secrets depuis GitHub Actions → .env
# ===============================
# Paramètres overridables : make env-from-gh BRANCH=Auto_github WF=permissions ART_NAME=env-artifact ENV_DST=.env

# -------- Defaults (écrasables à l'appel: make env-from-gh VAR=val) --------
WF       ?= permissions.yml
BRANCH   ?= main
ART_NAME ?= env-artifact
ENV_DST  ?= .env

export WF
export BRANCH
export ART_NAME
export ENV_DST

env-from-gh.vars: ## Affiche les variables pour env-from-gh
	@printf "WF=%s\nBRANCH=%s\nART_NAME=%s\nENV_DST=%s\n" "$(WF)" "$(BRANCH)" "$(ART_NAME)" "$(ENV_DST)"

env-from-gh: ## Récupère les secrets depuis GitHub Actions
	@set -eu ; \
	if [ -f "$(ENV_DST)" ]; then echo "OK: $(ENV_DST) already exists."; exit 0; fi ; \
	: "$${WF:?Var WF requise (ex: permissions.yml)}" ; \
	: "$${BRANCH:?Var BRANCH requise (ex: main)}" ; \
	: "$${ART_NAME:?Var ART_NAME requise (ex: env-artifact)}" ; \
	: "$${ENV_DST:?Var ENV_DST requise (ex: .env)}" ; \
	command -v gh >/dev/null 2>&1 || { echo "ERR: GitHub CLI 'gh' introuvable."; exit 127; } ; \
	echo "Start: trigger workflow='$(WF)' on branch='$(BRANCH)'." ; \
	if gh auth status >/dev/null 2>&1 ; then echo "Auth: gh auth status OK." ; \
	else : "$${GH_TOKEN:?Set GH_TOKEN (export GH_TOKEN=<PAT>)}" ; echo "Auth: using GH_TOKEN."; fi ; \
	gh workflow run "$(WF)" --ref "$(BRANCH)" >/dev/null ; \
	ATTEMPTS=0 ; MAX_ATTEMPTS=30 ; RUN_ID="" ; \
	while [ $$ATTEMPTS -lt $$MAX_ATTEMPTS ] ; do \
	  RUN_ID=$$(gh run list --workflow "$(WF)" --branch "$(BRANCH)" --limit 1 --json databaseId -q '.[0].databaseId' 2>/dev/null || true) ; \
	  [ -n "$$RUN_ID" ] && break ; ATTEMPTS=$$((ATTEMPTS+1)) ; sleep 1 ; \
	done ; \
	[ -n "$$RUN_ID" ] || { echo "ERR: aucun run trouvé pour workflow='$(WF)' sur branch='$(BRANCH)'."; exit 1; } ; \
	BRANCH_RUN=$$(gh run view "$$RUN_ID" --json headBranch -q .headBranch) ; \
	echo "RUN_ID=$$RUN_ID (branch=$$BRANCH_RUN)" ; \
	gh run watch "$$RUN_ID" || true ; \
	CONC=$$(gh run view "$$RUN_ID" --json conclusion -q .conclusion) ; \
	[ "$$CONC" = "success" ] || { echo "ERR: run $$RUN_ID = $$CONC" ; gh run view "$$RUN_ID" --web || true ; exit 1 ; } ; \
	echo "Download: artifact '$(ART_NAME)'." ; \
	TMPDIR=$$(mktemp -d "tmp-$(ART_NAME)-XXXXXX") ; \
	trap 'rm -rf "$$TMPDIR"' EXIT INT HUP TERM ; \
	if ! gh run download "$$RUN_ID" -n "$(ART_NAME)" -D "$$TMPDIR" ; then \
	  echo "ERR: artifact '$(ART_NAME)' introuvable. Liste:" ; \
	  gh run view "$$RUN_ID" --json artifacts -q '.artifacts[].name' || true ; \
	  exit 1 ; \
	fi ; \
	SRC=$$(find "$$TMPDIR" -type f -name "env.txt" -print -quit) ; \
	[ -n "$$SRC" ] || { echo "ERR: 'env.txt' introuvable. Contenu:" ; find "$$TMPDIR" -maxdepth 3 -type f -print ; exit 1 ; } ; \
	mkdir -p "$$(dirname -- "$(ENV_DST)")" ; \
	mv "$$SRC" "$(ENV_DST)" ; \
	echo "OK: $(ENV_DST) updated (preview, redacted):" ; \
	n=0 ; while IFS='' read -r line && [ $$n -lt 16 ]; do \
	  case $$line in *"="*) key=$${line%%=*}; printf "%s=***redacted***\n" "$$key" ;; *) printf "%s\n" "$$line" ;; esac ; \
	  n=$$((n+1)) ; done <"$(ENV_DST)"

# -------- Raccourci local (écrase tout via valeurs sûres) --------
env-from-gh.local: ## Raccourci pour récupérer .env depuis GitHub Actions
	@$(MAKE) env-from-gh WF=permissions.yml BRANCH=main ART_NAME=env-artifact ENV_DST=.env

check-permissions: ## Vérifie les permissions du profil utilisateur sur les services du projet
	@echo "🔍 Vérification des permissions pour le profil $(shell whoami):"
	@ls -ld . | awk '{print "Répertoire racine:", $$1, $$3, $$4}'
	@ls -ld data/ exports/ mlruns/ logs/ 2>/dev/null || echo "Certains répertoires n'existent pas encore."
	@echo "Permissions détaillées:"
	@find . -maxdepth 2 -type d -exec ls -ld {} \; | head -10

# ...autres cibles annexes si besoin (build-all, pipeline-reset, etc.)...
