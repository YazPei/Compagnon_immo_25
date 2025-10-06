# mlops/2_dvc/Dockerfile
# Multi-stage durci: base bookworm-slim, venv dédié, packages minimaux, non-root
FROM python:3.11-slim-bookworm AS builder

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Paquets système strictement nécessaires (git pour dvc, certs HTTPS)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Créer un venv pour isoler les deps Python
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Installer les dépendances Python dans le venv
COPY mlops/2_dvc/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# Étape 2 : Image finale minimale
FROM python:3.11-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}"

# Mettre à jour les paquets système avec les derniers correctifs de sécurité
RUN apt-get update \
    && apt-get -y dist-upgrade \
    && rm -rf /var/lib/apt/lists/*

# Labels OCI
LABEL org.opencontainers.image.title="Compagnon_immo dvc image" \
    org.opencontainers.image.description="Image optimisée et durcie pour le pipeline DVC / API" \
    org.opencontainers.image.vendor="Compagnon_immo" \
    org.opencontainers.image.source="https://github.com/YazPei/Compagnon_immo"

# Utilisateur non-root
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} app \
    && useradd -m -u ${USER_ID} -g app -s /bin/bash app

WORKDIR /app

# Copier le venv depuis le builder (inclut toutes les libs Python)
COPY --from=builder /opt/venv /opt/venv

# Copier le code nécessaire (API + scripts dvc)
COPY --chown=app:app app /app/app
COPY --chown=app:app mlops/2_dvc /app/mlops/2_dvc

# Droits et hygiène
RUN chmod +x /app/mlops/2_dvc/run_dvc.sh \
    && find /opt/venv -name "*.pyc" -delete \
    && rm -rf /root/.cache

USER app
ENV HOME=/home/app

# Commande par défaut: lancer l'API (uvicorn est fourni par requirements)
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

