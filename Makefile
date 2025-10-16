# ========== Makefile MLOps - Compagnon Immo ==========
# Gestion des pipelines avec Airflow, MLflow, DVC et Docker

# ===============================
# SOMMAIRE
# ===============================
# 1. Aide & vérifications        : help, lint, check-dependencies
# 2. Préparation & installation  : prepare-dirs, install, install-gh
# 3. Build                      : docker-build, airflow-build
# 4. Démarrage services         : permission, docker-start, dvc-all, quick-start-dvc, api-start, mlflow-up, airflow-up, dvc-add-all, dvc-repro-all, dvc-pull-all
# 5. Tests & CI                 : api-test, api-test-docker, ci-test
# 6. Arrêt & nettoyage          : api-stop, mlflow-down, airflow-down, stop-all, clean
# 7. Utilitaires                : docker-logs, airflow-logs, airflow-init, airflow-smoke, fix-permissions, check-services, env-from-gh, check-permissions


# --- Choix du fichier d'env local ---
# Si tu veux garder env.txt en local, mets: ENV_DST ?= env.txt
ENV_DST  ?= .env
ENV_FILE ?= $(ENV_DST)

# Auto-load variables d'environnement (si fichier présent)
ifneq ("$(wildcard $(ENV_FILE))","")
include $(ENV_FILE)
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' $(ENV_FILE))
endif

# ===== Variables =====
IMAGE_PREFIX := compagnon_immo
NETWORK := ml_net
PYTHON_BIN := python3
PIP := pip3
TEST_DIR := app/api/tests
DVC_TOKEN ?= default_token_securise_ou_vide


MLFLOW_IMAGE := ghcr.io/mlflow/mlflow:v2.13.1
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
  prepare-dirs install install-gh env-from-gh check-permissions \
  permission \
  docker-build airflow-build \
  docker-network docker-up docker-start mlflow-up airflow-up api-start \
  dvc-all quick-start-dvc dvc-repro-all dvc-pull-all \
  api-test api-test-docker ci-test \
  api-stop mlflow-down airflow-down stop-all clean \
  docker-logs airflow-logs airflow-init airflow-smoke fix-permissions check-services \
  pipeline-reset build-all run-all-docker run_dvc check-ports rebuild

# ===============================
# 1. Aide & vérifications
# ===============================
help: ## Affiche l'aide
	@echo "========== Compagnon Immo - Commandes disponibles =========="
	@grep -E '^[a-zA-Z0-9_.-]+:.*?##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

lint: ## Vérifie quelques pièges courants
	@echo "🔍 Vérification du Makefile…"
	@if grep -o '^[a-zA-Z0-9_.-]\+:' Makefile | sort | uniq -d | grep -q .; then \
		echo "⚠️  Cibles en double trouvées :"; \
		grep -o '^[a-zA-Z0-9_.-]\+:' Makefile | sort | uniq -d; \
		exit 1; \
	else \
		echo "✅ Aucune cible en double détectée - Makefile propre !"; \
	fi

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

install: prepare-dirs ## Installe les dépendances Python
	@if $(PIP) install --dry-run -r requirements.txt 2>&1 | grep -q "Would install"; then \
		$(PIP) install --upgrade pip; \
		$(PIP) install -r requirements.txt; \
	else \
		echo "Les dépendances sont déjà installées pour lancer le projet"; \
	fi

install-gh: ## Installe GitHub CLI si absent
	@if command -v gh >/dev/null 2>&1; then \
		echo "✅ GitHub CLI déjà installé."; \
	else \
		echo "🔧 Vérification/installation de GitHub CLI..."; \
		echo "📦 Installation automatique de GitHub CLI..."; \
		type -p curl >/dev/null || sudo apt install curl -y; \
		curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg; \
		sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg; \
		echo "deb [arch=$$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null; \
		sudo apt update; \
		sudo apt install gh -y; \
		echo "✅ GitHub CLI installé avec succès."; \
	fi

permission: prepare-dirs install ## Accord permissions rwx au profil utilisateur sur les services du projet
	@if ! command -v gh >/dev/null 2>&1; then \
		$(MAKE) install-gh; \
	fi
	@echo "$(COLOR_YELLOW)🔧 Attribution des permissions rwx au profil $(shell whoami)...$(COLOR_RESET)"
	@sudo chown -R $(shell whoami):$(shell whoami) . || true
	@chmod -R u+rwx . || true
	@echo "$(COLOR_GREEN)✅ Permissions rwx attribuées.$(COLOR_RESET)"

# ===============================
# 3. Build
# ===============================
docker-build: prepare-dirs ## Build via compose
	@$(DOCKER_COMPOSE_CMD) build

airflow-build: ## Build images Airflow
	docker compose build airflow-webserver airflow-scheduler

# ===============================
# API Management
# ===============================
api-build: ## Build image API
	DOCKER_BUILDKIT=0 docker build -t $(IMAGE_PREFIX)-api .

api-start: docker-api-run ## Démarre l'API (build + run)
api-stop: ## Stoppe l'API
	$(DOCKER_COMPOSE_CMD) down api


# ===============================
# 4. Démarrage services
# ===============================
docker-start: docker-network docker-up
mlflow-up: ## Démarre MLflow
	docker ps -q --filter "name=mlflow" | xargs -r docker stop 2>/dev/null || true
	docker ps -a -q --filter "name=mlflow" | xargs -r docker rm 2>/dev/null || true
	docker run -d --rm \
		--name $(MLFLOW_HOST) \
		--network $(NETWORK) \
		-v $(PWD)/mlruns:/mlflow/mlruns \
		-p $(MLFLOW_PORT):$(MLFLOW_PORT) \
		$(MLFLOW_IMAGE) \
		mlflow server --host 0.0.0.0 --port $(MLFLOW_PORT) \
		  --backend-store-uri sqlite:////mlflow/mlruns/mlflow.db \
		  --default-artifact-root /mlflow/mlruns

dvc-all: dvc-pull-all docker-repro-image-all
quick-start-dvc: docker-api-run mlflow-up docker-network docker-up docker-repro-image-all ## Quick start + exécution complète de DVC

docker-api-run: ## Run API via Docker Compose (sans profil)
	$(DOCKER_COMPOSE_CMD) up api --build -d

docker-network:
	docker network create ml_net || echo "Network ml_net already exists"

docker-up:
	@echo "🛑 Arrêt et suppression des conteneurs existants..."
	-$(DOCKER_COMPOSE_CMD) down --remove-orphans || true
	docker compose up -d

dvc-use-data:
	docker run --rm \
	  -v $(pwd):/app \
	  -w /app \
	  compagnon_immo-dvc \
	  python mlops/1_import_donnees/import_data.py \
	    --output-folder data/incremental \
	    --cumulative-path data/df_sample.csv \
	    --checkpoint-path data/checkpoint.parquet \
	    --date-column date_vente \
	    --key-columns id_transaction \
	    --sep ";" \
	    --dvc-repo-url https://dagshub.com/YazPei/Compagnon_immo \
	    --dvc-path data/dvc_data.csv \
	    --dvc-rev main



docker-repro-image-all: docker-dvc-check dvc-repro-all
docker-dvc-check:
	@if [ -z "$$(docker images -q $(DVC_IMAGE):latest)" ]; then \
		echo "🔧 Build de $(DVC_IMAGE):latest..."; \
		DOCKER_BUILDKIT=0 docker build --no-cache -t $(DVC_IMAGE):latest -f mlops/2_dvc/Dockerfile .; \
	else \
		echo "✅ Image $(DVC_IMAGE) déjà disponible."; \
	fi

dvc-repro-all: docker-dvc-check ## dvc repro de tout le pipeline
	@if ! docker ps --format "{{.Names}}" | grep -q "^$(MLFLOW_HOST)$$"; then \
		echo "🔧 MLflow non démarré, lancement..."; \
		$(MAKE) mlflow-up; \
		echo "⏳ Attente que MLflow soit prêt..."; \
		timeout 60 bash -c 'until docker run --rm --network $(NETWORK) curlimages/curl -s http://$(MLFLOW_HOST):$(MLFLOW_PORT)/api/2.0/mlflow/experiments/list >/dev/null 2>&1; do sleep 2; done' || { echo "❌ MLflow n'a pas démarré dans les temps"; exit 1; }; \
		echo "✅ MLflow prêt"; \
	fi
	sudo chmod -R 755 .dvc || true
	docker run --rm --user root \
	  --network $(NETWORK) \
	  -e MLFLOW_TRACKING_URI=$(MLFLOW_URI_DCK) \
	  -v $(PWD):/app:Z -w /app $(DVC_IMAGE) sh -c "chown -R root:root .dvc && rm -f .dvc/tmp/rwlock && dvc repro -f"





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


# DagsHub S3 section — noms isolés pour éviter les collisions avec MLOps

# Vars dédiées S3 (ne pas réutiliser FILE/KEY déjà déclarées plus haut)
S3_VENV := .s3venv
S3_PY   := $(S3_VENV)/bin/python
S3_PIP  := $(S3_VENV)/bin/pip

S3_FILE ?= merged_sales_data.csv         
S3_KEY  ?= merged_sales_data.csv         # clé objet par défaut (racine du bucket)

.PHONY: s3-help s3-venv s3-install s3-env s3-sanity s3-upload s3-upload-mp s3-list s3-clean

s3-help: ## Aide section S3 (DagsHub)
	@echo "S3 targets: s3-venv s3-install s3-env s3-sanity s3-upload s3-upload-mp s3-list s3-clean"

s3-venv:
	python3 -m venv $(S3_VENV)
	$(S3_PIP) -q install --upgrade pip

s3-install: s3-venv
	$(S3_PIP) -q install boto3 botocore

s3-env:
	@test -f $$HOME/.dagshub.env || (echo "Missing $$HOME/.dagshub.env"; exit 1)
	@set -a; source $$HOME/.dagshub.env; set +a; \
	echo "Endpoint: $$AWS_S3_ENDPOINT"; \
	echo "Bucket  : $$DAGSHUB_BUCKET"; \
	echo "Region  : $$AWS_DEFAULT_REGION"

s3-sanity: s3-env
	@set -a; source $$HOME/.dagshub.env; set +a; \
	$(S3_PY) - <<'PY' || { echo "Sanity FAIL"; exit 3; }
	import os, boto3
	s3=boto3.client("s3",
	endpoint_url=os.environ["AWS_S3_ENDPOINT"],
	aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
	aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
	region_name=os.environ.get("AWS_DEFAULT_REGION","us-east-1"))
	b=os.environ["DAGSHUB_BUCKET"]
	resp=s3.list_objects_v2(Bucket=b, MaxKeys=5)
	print("Sanity OK. KeyCount:", resp.get("KeyCount",0))
	PY

# Uploader Python requis: tools/upload_s3_resilient.py
s3-upload: s3-env
	@set -a; source $$HOME/.dagshub.env; set +a; \
	$(S3_PY) tools/upload_s3_resilient.py "$(S3_FILE)" "$$DAGSHUB_BUCKET" "$(S3_KEY)" \
	  --endpoint-url "$$AWS_S3_ENDPOINT" --path-style --force-single --verbose

s3-upload-mp: s3-env
	@set -a; source $$HOME/.dagshub.env; set +a; \
	$(S3_PY) tools/upload_s3_resilient.py "$(S3_FILE)" "$$DAGSHUB_BUCKET" "$(S3_KEY)" \
	  --endpoint-url "$$AWS_S3_ENDPOINT" --path-style \
	  --chunk-size-mb 8 --multipart-threshold-mb 16 --max-concurrency 2 --verbose

s3-list: s3-env
	@set -a; source $$HOME/.dagshub.env; set +a; \
	$(S3_PY) - <<'PY'
	import os, boto3
	s3=boto3.client("s3",
	endpoint_url=os.environ["AWS_S3_ENDPOINT"],
	aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
	aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
	region_name=os.environ.get("AWS_DEFAULT_REGION","us-east-1"))
	b=os.environ["DAGSHUB_BUCKET"]
	resp=s3.list_objects_v2(Bucket=b, MaxKeys=50)
	for o in resp.get("Contents",[]) or []:
		print(o["Key"])
		PY

s3-clean:
	rm -rf $(S3_VENV) __pycache__ .pytest_cache


