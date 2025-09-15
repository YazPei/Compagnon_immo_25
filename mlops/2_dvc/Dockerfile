FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1


# Dépendances système (build + matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libfreetype6 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
    
# 4. Copie du script shell + sources DVC
COPY mlops/2_dvc app/2_dvc
COPY .env.yaz /app/

# 5. Droits d’exécution
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
RUN chmod +x mlops/2_dvc/run_dvc.sh

# 6. Entrypoint = exécution du pipeline
CMD ["bash", "mlops/2_dvc/run_dvc.sh"]

