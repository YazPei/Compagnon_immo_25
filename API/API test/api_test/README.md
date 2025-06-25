# API d'Estimation Immobilière

## Démarrage rapide

### 1. Prérequis
- Python 3.9+
- Docker & Docker Compose
- PostgreSQL (local ou via Docker)
- Redis (local ou via Docker)

### 2. Installation

```bash
# Installation des dépendances
pip install -r requirements.txt
```

### 3. Lancement en local

```bash
uvicorn app.main:app --reload
```

### 4. Lancement avec Docker

```bash
docker build -t estimation-api .
docker run -p 8000:8000 --env-file .env estimation-api
```

### 5. Modèles Machine Learning
Placez vos modèles dans le dossier `models/` :
- `models/estimation_lgbm.pkl` (LightGBM)
- `models/evolution_sarimax.joblib` (SARIMAX)

### 6. Variables d'environnement
Créez un fichier `.env` à la racine avec :
```
API_KEY=test-key-123
DATABASE_URL=postgresql://user:password@localhost:5432/estimation
REDIS_URL=redis://localhost:6379/0
```

### 7. Documentation interactive
Accédez à la doc interactive sur :
- http://localhost:8000/docs

---

## Structure du projet

- `app/` : code source principal
- `models/` : modèles ML
- `requirements.txt` : dépendances Python
- `Dockerfile` : conteneurisation
