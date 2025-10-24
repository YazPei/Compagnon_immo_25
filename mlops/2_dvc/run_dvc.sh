#!/usr/bin/env bash
set -euo pipefail

# === 🌱 Charger les variables d'environnement ===
load_env() {
  for f in .env .env.yaz; do
    if [ -f "$f" ]; then
      echo "📦 Chargement des variables depuis $f"
      set -o allexport
      # shellcheck disable=SC1090
      source "$f"
      set +o allexport
    fi
  done
}
load_env

# === 🔌 Choisir iMLflow ===
# 1) Si MLFLOW_TRACKING_URI est déjà défini (via docker-compose), on le garde
# 2) Sinon on tente le service 'mlflow:5000' (réseau docker)
# 3) Sinon fallback local file:./mlruns
choose_mlflow() {
  if [ -n "${MLFLOW_TRACKING_URI:-}" ]; then
    echo "🔗 MLFLOW_TRACKING_URI (préconfiguré) = $MLFLOW_TRACKING_URI"
    return
  fi

  if curl -s -X POST -H 'Content-Type: application/json' -d '{}' http://mlflow:5000/api/2.0/mlflow/experiments/list >/dev/null 2>&1; then
    export MLFLOW_TRACKING_URI="http://mlflow:5050"
    echo "🔗 MLFLOW_TRACKING_URI (service docker) = $MLFLOW_TRACKING_URI"
  else
    mkdir -p ./mlruns
    # format robuste: schéma file: + chemin relatif
    if [ -z "${MLFLOW_TRACKING_URI:-}" ]; then
      echo "MLFLOW_TRACKING_URI non défini → fallback local file:./mlruns"
      export MLFLOW_TRACKING_URI="file:./mlruns"
    fi

    echo "🔗 MLFLOW_TRACKING_URI (fallback local) = $MLFLOW_TRACKING_URI"
  fi
}
choose_mlflow

echo "🔍 Vérification : DVC_USER='${DVC_USER:-undefined}', DVC_TOKEN='(masqué)', ST_SUFFIX=${ST_SUFFIX:-undefined}"
echo "🔍 MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}"

# === 🧰 Init DVC si besoin ===
if [ ! -d ".dvc" ]; then
  echo "⚙️ Initialisation du dépôt DVC..."
  dvc init --quiet
fi

# === 🔗 Configurer le remote DVC (DagsHub) ===
# ⚠️ Ajuste DAGSHUB_DVC_URL à la bonne URL de TON repo data (ex: https://dagshub.com/<user>/<repo>.dvc)
DAGSHUB_DVC_URL="${DAGSHUB_DVC_URL:-https://dagshub.com/${DVC_USER:-user}/compagnon_immo_25.dvc}"

echo "🔗 Configuration du remote DVC par défaut -> $DAGSHUB_DVC_URL"
dvc remote add -d origin "$DAGSHUB_DVC_URL" -f || true

dvc remote modify origin --local auth basic || true
[ -n "${DVC_USER:-}" ]   && dvc remote modify origin --local user "$DVC_USER" || true
[ -n "${DVC_TOKEN:-}" ]  && dvc remote modify origin --local password "$DVC_TOKEN" || true

# Cache local explicite (si pas déjà défini)
if ! dvc config --list | grep -q "^cache\.dir"; then
  echo "🗃️  Configuration du cache DVC local -> .dvc/cache"
  dvc config cache.dir .dvc/cache
fi

# === 📝 Mettre ST_SUFFIX sans écraser tout params.yaml (best-effort) ===
# Si yq dispo, on met à jour; sinon on avertit et on écrase seulement si nécessaire.
if command -v yq >/dev/null 2>&1; then
  echo "💾 Mise à jour params.yaml (ST_SUFFIX) via yq"
  touch params.yaml
  yq -i ".ST_SUFFIX = \"${ST_SUFFIX:-q12}\"" params.yaml
else
  echo "⚠️ yq non disponible. Écriture minimale de params.yaml (peut écraser d'autres clés)."
  echo "ST_SUFFIX: ${ST_SUFFIX:-q12}" > params.yaml
fi

# === ⬇️ Récupérer les données, ⛏️ exécuter le pipeline, ⬆️ publier ===
echo "📥 dvc pull"
dvc pull --force

echo "🚀 dvc repro"
dvc repro

echo "📊 dvc metrics show"
dvc metrics show || echo "(aucune métrique DVC pour le moment)"

echo "📈 dvc plots show"
dvc plots show --html > plots.html || echo "(aucun plot DVC pour le moment)"

echo "☁️ dvc push"
dvc push

echo "✅ Pipeline DVC exécuté avec succès !"

