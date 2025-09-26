# Dockerfile API - Compagnon Immo
# Multi-stage pour une image légère et sécurisée

FROM python:3.11-slim-bookworm AS base

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Installation des dépendances système
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        selinux-utils \
        policycoreutils \
    && rm -rf /var/lib/apt/lists/*

# Vérification SELinux
RUN sestatus || echo "SELinux non disponible dans ce conteneur"

# Création de l'environnement virtuel
RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /app

# Installation des dépendances Python
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install gunicorn dvc \
    && pip check

# Copie des fichiers de l'application
COPY app ./app
COPY params.yaml ./params.yaml
COPY .dvc ./.dvc
COPY .env /app/.env

# Configuration des permissions
RUN chmod -R 755 /app

# Utilisateur non-root
RUN useradd -m appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Variables d'environnement par défaut
ENV ENVIRONMENT=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    WORKERS=4 \
    LOG_LEVEL=INFO

# Commande par défaut
CMD ["gunicorn", "app.api.main:app", \
     "-w", "4", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-b", "0.0.0.0:8000"]
    "-w", "4", \
    "-k", "uvicorn.workers.UvicornWorker", \
    "-b", "0.0.0.0:8000" \
    ]
