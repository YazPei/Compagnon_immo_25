# ===== Makefile — Compagnon Immo (UTF-8, '>' comme préfixe de recette) =====
.RECIPEPREFIX := >

# ---------- Encodage / locale ----------
export LANG := C.UTF-8
export LC_ALL := C.UTF-8
export PYTHONIOENCODING := UTF-8

# ---------- Variables env ----------
ENV_DST  ?= .env
ENV_FILE ?= $(ENV_DST)
ifneq ("$(wildcard $(ENV_FILE))","")
include $(ENV_FILE)
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' $(ENV_FILE))
endif

# ---------- Variables projet ----------
IMAGE_PREFIX := compagnon_immo
NETWORK := ml_net
PYTHON_BIN := python3
PIP := pip3
DVC_IMAGE := $(IMAGE_PREFIX)-dvc
DOCKER_COMPOSE_CMD := docker compose

MLFLOW_IMAGE := ghcr.io/mlflow/mlflow:v2.13.1
MLFLOW_PORT := 5050
MLFLOW_HOST := $(IMAGE_PREFIX)-mlflow
MLFLOW_URI_DCK := http://$(MLFLOW_HOST):$(MLFLOW_PORT)

AIRFLOW_SERVICES := postgres-airflow airflow-webserver airflow-scheduler
AIRFLOW_UID ?= 50000
AIRFLOW_URL ?= http://localhost:8081

COLOR_RESET := \033[0m
COLOR_GREEN := \033[32m
COLOR_RED := \033[31m
COLOR_YELLOW := \033[33m

# ---------- GitHub Actions → .env (config par défaut écrasable) ----------
WF ?= .github/workflows/permissions.yml
BRANCH ?= dvc_stage
ART_NAME ?= env-artifact

# Defaults (override via: make env-from-gh BRANCH=feature-xyz)

# ENV_DST déjà défini plus haut

# ---------- PHONY ----------
.PHONY: help lint check-dependencies prepare-dirs install install-gh \
        permission fix-permissions check-permissions env-from-gh env-from-gh.local \
        docker-build airflow-build docker-network docker-up docker-start \
        mlflow-up mlflow-down airflow-down docker-logs airflow-logs airflow-init \
        dvc-repro-all dvc-repro-import stop-all clean check-services \
        api-build docker-api-run api-start api-stop api-test api-test-fast \
        api-logs api-shell \
        s3_sanity_dagshub import_s3_dagshub s3_import_data \
        import_s3_public import_s3_profile

# ===============================
# 1) AIDE & CHECKS
# ===============================
help: ## Affiche l'aide
> @echo "========== Compagnon Immo - Commandes =========="
> @grep -E '^[a-zA-Z0-9_.-]+:.*?##.*$$' $(MAKEFILE_LIST) | \
>   awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

lint: ## Cibles dupliquées
> @if grep -o '^[a-zA-Z0-9_.-]\+:' Makefile | sort | uniq -d | grep -q .; then \
>   echo "⚠️  Cibles en double :"; \
>   grep -o '^[a-zA-Z0-9_.-]\+:' Makefile | sort | uniq -d; exit 1; \
> else echo "✅ Aucune cible en double."; fi

check-dependencies: ## Vérifie docker/python/dvc/gh
> @command -v docker >/dev/null 2>&1 || { echo "$(COLOR_RED)❌ Docker manquant$(COLOR_RESET)"; exit 1; }
> @command -v $(PYTHON_BIN) >/dev/null 2>&1 || { echo "$(COLOR_RED)❌ Python3 manquant$(COLOR_RESET)"; exit 1; }
> @command -v dvc >/dev/null 2>&1 || { echo "$(COLOR_RED)❌ DVC manquant$(COLOR_RESET)"; exit 1; }
> @command -v gh >/dev/null 2>&1 || { echo "$(COLOR_RED)❌ gh manquant$(COLOR_RESET)"; exit 1; }
> @echo "$(COLOR_GREEN)✅ Dépendances OK$(COLOR_RESET)"

# ===============================
# 2) PREP / INSTALL
# ===============================
prepare-dirs: ## Crée data/ exports/ mlruns/ logs/airflow
> @mkdir -p data exports mlruns logs/airflow
> @touch data/.gitkeep

install: prepare-dirs ## Installe requirements.txt si nécessaire
> @if $(PIP) install --dry-run -r requirements.txt 2>&1 | grep -q "Would install"; then \
>   $(PIP) install --upgrade pip && $(PIP) install -r requirements.txt ; \
> else echo "✅ Dépendances déjà installées"; fi

install-gh: ## Installe GitHub CLI
> @if command -v gh >/dev/null 2>&1; then echo "✅ gh ok"; else \
>   type -p curl >/dev/null || sudo apt install -y curl; \
>   curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg; \
>   sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg; \
>   echo "deb [arch=$$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null; \
>   sudo apt update && sudo apt install -y gh; fi

# ===============================
# 3) PERMISSIONS & .env
# ===============================
permission: prepare-dirs install ## Attribue rwx à l'utilisateur courant sur tout le repo
> @echo "$(COLOR_YELLOW)🔧 Attribution des permissions...$(COLOR_RESET)"
> @sudo chown -R $$(whoami):$$(whoami) . || true
> @chmod -R u+rwx . || true
> @echo "$(COLOR_GREEN)✅ Permissions rwx attribuées$(COLOR_RESET)"

fix-permissions: ## Corrige rapidement propriétaires/droits (utilisateur courant)
> @echo "$(COLOR_YELLOW)🔧 Fix permissions...$(COLOR_RESET)"
> @sudo chown -R $$(whoami):$$(whoami) . 2>/dev/null || true
> @chmod -R u+rwX . || true
> @find . -type d -exec chmod u+rwx {} \; || true
> @echo "$(COLOR_GREEN)✅ OK$(COLOR_RESET)"

check-permissions: ## Affiche un aperçu des permissions réelles sur les dossiers clés
> @echo "🔍 Permissions:"
> @ls -ld . | awk '{print "Repo:", $$1, $$3, $$4}'
> @ls -ld data/ exports/ mlruns/ logs/ 2>/dev/null || echo "⚠️  Certains dossiers n'existent pas."
> @find . -maxdepth 2 -type d -exec ls -ld {} \; | head -50

env-from-gh.vars: ## Affiche vars pour env-from-gh
> @printf "WF=%s\nBRANCH=%s\nART_NAME=%s\nENV_DST=%s\n" "$(WF)" "$(BRANCH)" "$(ART_NAME)" "$(ENV_DST)"

env-from-gh: ## Récupère .env depuis GitHub Actions (artifact)
> @set -eu ; \
> if [ -f "$(ENV_DST)" ]; then echo "OK: $(ENV_DST) already exists."; exit 0; fi ; \
> : "$${WF:?Var WF requise (ex: permissions.yml)}" ; \
> : "$${BRANCH:?Var BRANCH requise (ex: main)}" ; \
> : "$${ART_NAME:?Var ART_NAME requise (ex: env-artifact)}" ; \
> : "$${ENV_DST:?Var ENV_DST requise (ex: .env)}" ; \
> command -v gh >/dev/null 2>&1 || { echo "ERR: GitHub CLI 'gh' introuvable."; exit 127; } ; \
> echo "➡️  Trigger workflow='$(WF)' on branch='$(BRANCH)'" ; \
> if gh auth status >/dev/null 2>&1 ; then echo "Auth: gh ok" ; \
> else : "$${GH_TOKEN:?Set GH_TOKEN (export GH_TOKEN=<PAT>)}" ; echo "Auth: using GH_TOKEN"; fi ; \
> gh workflow run "$(WF)" --ref "$(BRANCH)" >/dev/null ; \
> ATTEMPTS=0 ; MAX_ATTEMPTS=30 ; RUN_ID="" ; \
> while [ $$ATTEMPTS -lt $$MAX_ATTEMPTS ] ; do \
>   RUN_ID=$$(gh run list --workflow "$(WF)" --branch "$(BRANCH)" --limit 1 --json databaseId -q '.[0].databaseId' 2>/dev/null || true) ; \
>   [ -n "$$RUN_ID" ] && break ; ATTEMPTS=$$((ATTEMPTS+1)) ; sleep 1 ; \
> done ; \
> [ -n "$$RUN_ID" ] || { echo "ERR: aucun run trouvé"; exit 1; } ; \
> echo "RUN_ID=$$RUN_ID" ; \
> gh run watch "$$RUN_ID" || true ; \
> CONC=$$(gh run view "$$RUN_ID" --json conclusion -q .conclusion) ; \
> [ "$$CONC" = "success" ] || { echo "ERR: run $$RUN_ID = $$CONC" ; gh run view "$$RUN_ID" --web || true ; exit 1 ; } ; \
> echo "⬇️  Download artifact '$(ART_NAME)'" ; \
> TMPDIR=$$(mktemp -d "tmp-$(ART_NAME)-XXXXXX") ; \
> trap 'rm -rf "$$TMPDIR"' EXIT INT HUP TERM ; \
> gh run download "$$RUN_ID" -n "$(ART_NAME)" -D "$$TMPDIR" ; \
> SRC=$$(find "$$TMPDIR" -type f -name "env.txt" -print -quit) ; \
> [ -n "$$SRC" ] || { echo "ERR: 'env.txt' introuvable"; find "$$TMPDIR" -maxdepth 3 -type f -print ; exit 1 ; } ; \
> mkdir -p "$$(dirname -- "$(ENV_DST)")" ; \
> mv "$$SRC" "$(ENV_DST)" ; \
> echo "OK: $(ENV_DST) updated (preview redacted):" ; \
> n=0 ; while IFS='' read -r line && [ $$n -lt 16 ]; do \
>   case $$line in *"="*) key=$${line%%=*}; printf "%s=***redacted***\n" "$$key" ;; *) printf "%s\n" "$$line" ;; esac ; \
>   n=$$((n+1)) ; done <"$(ENV_DST)"

env-from-gh.local: ## Raccourci standard
> @$(MAKE) env-from-gh WF=permissions.yml BRANCH=main ART_NAME=env-artifact ENV_DST=.env

# ===============================
# 4) API (Docker Compose)
# ===============================
api-build: ## Build image API
> @DOCKER_BUILDKIT=0 $(DOCKER_COMPOSE_CMD) build api

docker-api-run: ## Run API (detached)
> @$(DOCKER_COMPOSE_CMD) up api --build -d

api-start: docker-api-run ## Alias pour démarrer l'API

api-stop: ## Stoppe uniquement l'API
> @$(DOCKER_COMPOSE_CMD) stop api || true
> @$(DOCKER_COMPOSE_CMD) rm -f api || true

api-test: ## Tests API (profile test)
> @$(DOCKER_COMPOSE_CMD) --profile test up --build --abort-on-container-exit --exit-code-from api-test --quiet-pull
> @$(DOCKER_COMPOSE_CMD) --profile test down -v

api-test-fast: ## Tests API rapides
> @$(DOCKER_COMPOSE_CMD) --profile test up --abort-on-container-exit --exit-code-from api-test --quiet-pull
> @$(DOCKER_COMPOSE_CMD) --profile test down -v

api-logs: ## Logs API (suivi)
> @$(DOCKER_COMPOSE_CMD) logs -f api

api-shell: ## Shell dans le conteneur API
> @cid=$$($(DOCKER_COMPOSE_CMD) ps -q api); \
> if [ -z "$$cid" ]; then echo "API non démarrée. Lancez 'make api-start'."; exit 1; fi; \
> docker exec -it "$$cid" /bin/bash

# ===============================
# 5) DOCKER / MLFLOW / AIRFLOW
# ===============================
docker-build: prepare-dirs ## docker compose build
> @$(DOCKER_COMPOSE_CMD) build

airflow-build: ## build images airflow
> @$(DOCKER_COMPOSE_CMD) build airflow-webserver airflow-scheduler

docker-network:
> @docker network create $(NETWORK) >/dev/null 2>&1 || echo "ℹ️ réseau $(NETWORK) ok"

docker-up:
> @echo "♻️  Restart docker compose"
> -@$(DOCKER_COMPOSE_CMD) down --remove-orphans || true
> @$(DOCKER_COMPOSE_CMD) up -d

docker-start: docker-network docker-up ## démarre l'environnement

mlflow-up: ## Démarre MLflow local (file store)
> -@docker stop $(MLFLOW_HOST) >/dev/null 2>&1 || true
> -@docker rm $(MLFLOW_HOST) >/dev/null 2>&1 || true
> docker run -d --rm \
>   --name $(MLFLOW_HOST) \
>   --network $(NETWORK) \
>   -v $(PWD)/mlruns:/mlflow/mlruns \
>   -p $(MLFLOW_PORT):$(MLFLOW_PORT) \
>   $(MLFLOW_IMAGE) \
>   mlflow server --host 0.0.0.0 --port $(MLFLOW_PORT) \
>     --backend-store-uri sqlite:////mlflow/mlruns/mlflow.db \
>     --default-artifact-root /mlflow/mlruns

mlflow-down:
> -@docker stop $(MLFLOW_HOST) || true

airflow-down:
> -@$(DOCKER_COMPOSE_CMD) stop $(AIRFLOW_SERVICES) || true
> -@$(DOCKER_COMPOSE_CMD) rm -f $(AIRFLOW_SERVICES) || true

docker-logs:
> @$(DOCKER_COMPOSE_CMD) logs -f

airflow-logs:
> @$(DOCKER_COMPOSE_CMD) logs -f airflow-webserver

airflow-init: ## Init DB Airflow + admin/admin
> @mkdir -p logs/airflow
> -@sudo chown -R $(AIRFLOW_UID):0 logs/airflow || true
> @$(DOCKER_COMPOSE_CMD) --profile airflow run --rm airflow-webserver airflow db upgrade
> @$(DOCKER_COMPOSE_CMD) --profile airflow run --rm airflow-webserver \
>   airflow users create --username admin --password admin \
>   --firstname Admin --lastname User --role Admin --email admin@example.com || true

# ===============================
# 6) DVC
# ===============================
dvc-repro-import: ## dvc repro du stage 'import_data'
> @mkdir -p data/state data/incremental
> DVC_LOGLEVEL=DEBUG dvc repro -f import_data

dvc-repro-all: ## dvc repro pipeline complet (image DVC)
> @if ! docker ps --format "{{.Names}}" | grep -q "^$(MLFLOW_HOST)$$"; then \
>   echo "🔧 MLflow non démarré, lancement..."; \
>   $(MAKE) mlflow-up; \
>   echo "⏳ Attente de MLflow..."; \
>   timeout 60 bash -c 'until docker run --rm --network $(NETWORK) curlimages/curl -s http://$(MLFLOW_HOST):$(MLFLOW_PORT)/api/2.0/mlflow/experiments/list >/dev/null 2>&1; do sleep 2; done' || { echo "❌ MLflow KO"; exit 1; }; \
> fi
> @sudo chmod -R 755 .dvc || true
> docker run --rm --user root \
>   --network $(NETWORK) \
>   -e MLFLOW_TRACKING_URI=$(MLFLOW_URI_DCK) \
>   -v $(PWD):/app:Z -w /app $(DVC_IMAGE) sh -c "chown -R root:root .dvc && rm -f .dvc/tmp/rwlock && dvc repro -f"

# ===============================
# 7) S3 — DagsHub (auth requise)
# ===============================
# Requis à l'env:

# en-tête ok
.RECIPEPREFIX := >
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.SILENT:

# ---- S3 / projet ----
S3_ENDPOINT ?= https://dagshub.com/api/v1/repo-buckets/s3/YazPei
BUCKET ?= Compagnon_immo_25
KEY ?= merged_sales_data.csv
REGION ?= us-east-1

# Fichier d'env DagsHub (ton chemin réel)
DAGSHUB_ENV ?= /home/vboxuser/Compagnon_new/.dagshub.env

# ---- venv ----
S3_VENV ?= .s3venv
PY   := $(S3_VENV)/bin/python
PIP  := $(S3_VENV)/bin/pip

s3-venv:
> python3 -m venv "$(S3_VENV)"
> "$(PIP)" -q install --upgrade pip setuptools wheel

s3-install: s3-venv
> "$(PIP)" -q install "boto3>=1.34" "botocore==1.40.46" "urllib3>=2" \
>                          "pandas>=2.2" "pyarrow>=15" "fsspec>=2024.3" "s3fs>=2024.3" \
>                          "click>=8.1" "mlflow>=2.10,<3"


# Shell interactif (optionnel)
venv-shell:
> source .venv/bin/activate
> pip install awscli
> exec "$$SHELL" -i

s3-env:
> set -a
> [ -f .env ] && . .env || true
> [ -f "$(DAGSHUB_ENV)" ] && . "$(DAGSHUB_ENV)" || echo "WARN: $(DAGSHUB_ENV) introuvable"
> set +a
> echo "Endpoint: $(S3_ENDPOINT)"
> echo "Bucket  : $(BUCKET)"
> echo "Region  : $(REGION)"
> if [ -n "$${AWS_ACCESS_KEY_ID:-}" ]; then echo "AWS_ACCESS_KEY_ID: OK"; else echo "AWS_ACCESS_KEY_ID: MISSING"; fi
> if [ -n "$${AWS_SECRET_ACCESS_KEY:-}" ]; then echo "AWS_SECRET_ACCESS_KEY: OK"; else echo "AWS_SECRET_ACCESS_KEY: MISSING"; fi
> if [ -n "$${AWS_SESSION_TOKEN:-}" ]; then echo "AWS_SESSION_TOKEN: present"; fi
> echo "Chargé depuis: $(DAGSHUB_ENV)"

# petit test utile
test-s3: s3-install s3-env
> "$(PY)" - <<'PY'
> import os, boto3
> s3 = boto3.client(
>     "s3",
>     endpoint_url=os.environ.get("S3_ENDPOINT","$(S3_ENDPOINT)"),
>     region_name=os.environ.get("AWS_DEFAULT_REGION","$(REGION)"),
> )
> resp = s3.head_object(Bucket="$(BUCKET)", Key="$(KEY)")
> print("OK HEAD:", resp["ResponseMetadata"]["HTTPStatusCode"])
> PY



import: s3-install s3-env
> [ -n "$${AWS_ACCESS_KEY_ID:-}" ] || { echo "ERROR: AWS_ACCESS_KEY_ID manquant"; exit 2; }
> [ -n "$${AWS_SECRET_ACCESS_KEY:-}" ] || { echo "ERROR: AWS_SECRET_ACCESS_KEY manquant"; exit 2; }
> PYTHONIOENCODING=UTF-8 LC_ALL=C.UTF-8 LANG=C.UTF-8 \
> "$(PY)" mlops/1_import_donnees/import_data.py \
>   --source-mode s3 \
>   --s3-endpoint-url "$(S3_ENDPOINT)" \
>   --s3-bucket "$(BUCKET)" \
>   --s3-key "$(KEY)" \
>   --s3-region "$(REGION)" \
>   --output-folder data/incremental \
>   --cumulative-path data/df_sample.csv \
>   --checkpoint-path data/state/checkpoint.parquet \
>   --date-column "date" \
>   --key-columns "idannonce" \
>   --sep ";"

clean-venv:
> rm -rf "$(S3_VENV)"



# ===============================
# 9) STOP / CLEAN
# ===============================
stop-all: ## Stoppe tout l'écosystème
> -@docker ps -a --filter "name=compagnon_immo" -q | xargs -r docker rm -f || true
> -@$(DOCKER_COMPOSE_CMD) down -v --remove-orphans || true

clean: ## Clean fichiers temporaires
> -@rm -rf .pytest_cache .coverage || true

check-services: ## État des services compose
> @docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "api|mlflow|airflow|redis" || true

