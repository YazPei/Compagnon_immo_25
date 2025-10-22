# Pipeline Série Temporelle – Modélisation des prix immobiliers

Cette pipeline modulaire repose sur une série d’étapes orchestrées avec Docker, MLflow, Click CLI et DVC.

# Arborescence
```bash
.
├── Dockerfile.split
├── Dockerfile.decompose
├── Dockerfile.sarimax
├── Dockerfile.evaluate
├── docker-compose.yml
├── run_all_st.sh
├── requirements.txt
├── dvc.yaml
├── README_series_temporality.md
├── mlops/
│   └── time_series/
│       ├── load_split.py
│       ├── seasonal_decomp.py
│       ├── sarimax_train.py
│       ├── metrics.py
│       ├── utils.py
├── data/
│   └── df_sales_clean_ST.csv
├── exports/
│   └── st/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── model_sarimax_[cluster].pkl

```
## Étapes

1. **Split temporel + features** (`load_split.py`)
2. **Décomposition saisonnière** (`seasonal_decomp.py`)
3. **Entraînement SARIMAX par cluster** (`sarimax_train.py`)
4. **Évaluation & résidus** (`metrics.py`)

## Lancement rapide

```bash
bash run_all.sh
```
## lancement dockker
```bash
docker compose run split
docker compose run decompose
docker compose run train_sarimax
docker compose run evaluate
```


## Tracking MLflow
Tracking

    🧪 MLflow : http://localhost:5001

    📦 Artefacts : ./exports/st/


---

