#!/usr/bin/env bash
sudo apt-get update && sudo apt-get install -y gh
gh auth login   # suivre l’assistant

set -euo pipefail

echo "=== Setup .env pour Airflow/DagsHub ==="

# Prérequis optionnels:
# - gh CLI connecté: gh auth status
# - git remote configuré pour ce repo

REPO_URL="$(git remote get-url origin 2>/dev/null || echo "")"
if [[ -z "$REPO_URL" ]]; then
  echo "⚠️  Impossible de détecter le remote git 'origin'."
  echo "    Je te demanderai owner/repo manuellement."
fi

# Parse owner/repo depuis l’URL si possible
parse_repo () {
  local url="$1"
  # formats https://github.com/owner/repo.git ou git@github.com:owner/repo.git
  if [[ "$url" =~ github.com[:/]+([^/]+)/([^/.]+) ]]; then
    echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
  else
    echo "" ""
  fi
}
read -r GITHUB_OWNER GITHUB_REPO <<<"$(parse_repo "$REPO_URL")"

# Demande au besoin
if [[ -z "${GITHUB_OWNER:-}" ]]; then
  read -rp "Owner/organisation GitHub: " GITHUB_OWNER
fi
if [[ -z "${GITHUB_REPO:-}" ]]; then
  read -rp "Nom du repo GitHub: " GITHUB_REPO
fi

# Vérifie gh
if ! command -v gh >/dev/null 2>&1; then
  echo "⚠️  gh CLI non trouvé. J’essaierai sans."
fi

# Helper: lire VARIABLE GitHub Actions (non secret)
get_repo_variable () {
  local var_name="$1"
  if command -v gh >/dev/null 2>&1; then
    gh api -H "Accept: application/vnd.github+json" \
      "/repos/${GITHUB_OWNER}/${GITHUB_REPO}/actions/variables/${var_name}" \
      --jq .value 2>/dev/null || true
  fi
}

# 1) Remplir les champs NON sensibles automatiquement si possible
DEFAULT_DAGSHUB_USER="$(get_repo_variable DAGSHUB_USERNAME)"
DEFAULT_REPO_OWNER="$(get_repo_variable DAGSHUB_REPO_OWNER)"
DEFAULT_REPO_NAME="$(get_repo_variable DAGSHUB_REPO_NAME)"
DEFAULT_TRACKING_URI="$(get_repo_variable DAGSHUB_MLFLOW_TRACKING_URI)"

# Fallbacks malins
[[ -z "$DEFAULT_REPO_OWNER" ]] && DEFAULT_REPO_OWNER="${GITHUB_OWNER}"
[[ -z "$DEFAULT_REPO_NAME"  ]] && DEFAULT_REPO_NAME="${GITHUB_REPO}"
[[ -z "$DEFAULT_TRACKING_URI" ]] && DEFAULT_TRACKING_URI="https://dagshub.com/${DEFAULT_REPO_OWNER}/${DEFAULT_REPO_NAME}.mlflow"

# 2) Secrets: impossible de les lire depuis GitHub → env / password store / prompt
DAGSHUB_USER="${DAGSHUB_USER:-${DEFAULT_DAGSHUB_USER:-}}"
DAGSHUB_REPO_OWNER="${DAGSHUB_REPO_OWNER:-$DEFAULT_REPO_OWNER}"
DAGSHUB_REPO_NAME="${DAGSHUB_REPO_NAME:-$DEFAULT_REPO_NAME}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-$DEFAULT_TRACKING_URI}"

# Essaie de récupérer le token depuis l’environnement
DAGSHUB_TOKEN="${DAGSHUB_TOKEN:-}"

# Optionnel: lire depuis 'pass' (si tu utilises le password store)
if [[ -z "$DAGSHUB_TOKEN" ]] && command -v pass >/dev/null 2>&1; then
  if pass show dagshub/token >/dev/null 2>&1; then
    DAGSHUB_TOKEN="$(pass show dagshub/token | head -n1)"
    echo "🔐 Token récupéré depuis 'pass' (dagshub/token)."
  fi
fi

# Prompts avec défauts pour les non sensibles
read -rp "Ton login DagsHub [${DAGSHUB_USER:-}]: " _in
DAGSHUB_USER="${_in:-${DAGSHUB_USER:-}}"

read -rp "Owner DagsHub [${DAGSHUB_REPO_OWNER:-}]: " _in
DAGSHUB_REPO_OWNER="${_in:-${DAGSHUB_REPO_OWNER:-}}"

read -rp "Nom du repo DagsHub [${DAGSHUB_REPO_NAME:-}]: " _in
DAGSHUB_REPO_NAME="${_in:-${DAGSHUB_REPO_NAME:-}}"

read -rp "MLflow Tracking URI [${MLFLOW_TRACKING_URI:-}]: " _in
MLFLOW_TRACKING_URI="${_in:-${MLFLOW_TRACKING_URI:-}}"

# Prompt secret uniquement si toujours vide
if [[ -z "$DAGSHUB_TOKEN" ]]; then
  read -rs -p "Ton token DagsHub (caché): " DAGSHUB_TOKEN
  echo
fi

# Sanity checks
if [[ -z "$DAGSHUB_USER" || -z "$DAGSHUB_TOKEN" || -z "$DAGSHUB_REPO_OWNER" || -z "$DAGSHUB_REPO_NAME" ]]; then
  echo "❌ Champs requis manquants. Abandon."
  exit 1
fi

# 3) Générer .env
cat > .env <<EOF
DAGSHUB_USER=${DAGSHUB_USER}
DAGSHUB_TOKEN=${DAGSHUB_TOKEN}
DAGSHUB_REPO_OWNER=${DAGSHUB_REPO_OWNER}
DAGSHUB_REPO_NAME=${DAGSHUB_REPO_NAME}

MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
MLFLOW_TRACKING_USERNAME=${DAGSHUB_USER}
MLFLOW_TRACKING_PASSWORD=${DAGSHUB_TOKEN}
EOF

chmod 600 .env || true
echo "✅ .env généré (permissions restreintes)."
echo "👉 Lance: docker compose up -d"

