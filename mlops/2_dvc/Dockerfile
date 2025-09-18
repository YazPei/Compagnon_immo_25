# mlops/2_dvc/Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- 0) Paquets système utiles ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates libfreetype6 libpng16-16 \
 && rm -rf /var/lib/apt/lists/*

# --- 1) Créer un user "app" avec les UID/GID de l'hôte ---
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} app && \
    useradd -m -u ${USER_ID} -g app -s /bin/bash app

# Dossier de travail + droits
WORKDIR /app
RUN chown -R app:app /app

# --- 2) Dépendances Python (en root) ---
COPY mlops/2_dvc/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# --- 3) Code (copié pour fallback; sera écrasé si tu montes le volume) ---
COPY mlops/2_dvc /app/mlops/2_dvc
RUN chmod +x /app/mlops/2_dvc/run_dvc.sh && chown -R app:app /app


# (tu passeras --env-file au run)

# --- 4) Runtime en user non-root ---
USER app
ENV HOME=/home/app

# --- 5) Entrypoint ---
CMD ["bash", "mlops/2_dvc/run_dvc.sh"]

