# ========= Étape 1 : Builder (installation dans venv) =========
FROM python:3.11-slim-bookworm AS builder

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:${PATH}"

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Préparation de l'environnement Python
RUN python -m venv /opt/venv
WORKDIR /app

# Installation des dépendances Python
COPY ./mlops/2_dvc/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# ========= Étape 2 : Image finale minimale =========
FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

# Mises à jour de sécurité système
RUN apt-get update && apt-get -y dist-upgrade && rm -rf /var/lib/apt/lists/*

# Labels d'information (OCI)
LABEL org.opencontainers.image.title="Compagnon_immo_25 dvc image" \
    org.opencontainers.image.description="Image optimisée et durcie pour le pipeline DVC / API" \
    org.opencontainers.image.vendor="Compagnon_immo_25" \
    org.opencontainers.image.source="https://github.com/YazPei/Compagnon_immo_25"

# Utilisateur sécurisé non-root
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} app \
    && useradd -m -u ${USER_ID} -g app -s /bin/bash app

# Dossier de travail
WORKDIR /app

# Récupération de l'environnement Python depuis l'étape "builder"
COPY --from=builder /opt/venv /opt/venv

# Copie du code applicatif et scripts DVC
COPY --chown=app:app ./app /app/app
COPY --chown=app:app ./mlops/2_dvc /app/mlops/2_dvc

# Hygiène : droits + nettoyage
RUN chmod +x /app/mlops/2_dvc/run_dvc.sh \
    && find /opt/venv -name "*.pyc" -delete \
    && rm -rf /root/.cache

# Exécution en tant qu'utilisateur app
USER app
ENV HOME=/home/app

# Commande par défaut
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

