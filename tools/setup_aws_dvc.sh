# ------------------------------------------------------------
# setup/aws_s3_dvc_min_setup.sh
# But: Setup minimal et sûr pour DVC + S3 (sans commiter de secrets).
# Pourquoi: éviter NoCredentialsError et lier DVC à la bonne auth.
# ------------------------------------------------------------
set -euo pipefail

# ====== PARAMS À ADAPTER ======
REMOTE_NAME="${REMOTE_NAME:-storage}"       # nom du remote DVC (dvc remote list)
AWS_REGION="${AWS_REGION:-eu-west-1}"       # région du bucket
AWS_PROFILE_NAME="${AWS_PROFILE_NAME:-myproj}"  # si tu choisis la voie "profil"
S3_COMPAT_ENDPOINT="${S3_COMPAT_ENDPOINT:-}"    # ex MinIO: https://minio.local:9000
PATH_STYLE="${PATH_STYLE:-false}"                # true pour MinIO
# ==============================

log(){ printf '[%s] %s\n' "$(date +'%F %T')" "$*" >&2; }

# 0) Outils
command -v dvc >/dev/null || { log "Installe DVC"; exit 1; }
if ! command -v aws >/dev/null; then
  log "aws CLI non trouvé: installe-le pour diagnostics (facultatif mais conseillé)."
fi

# 1) ***CHOISIR UNE SEULE MÉTHODE D’AUTH***
# A) ENV VARS (simple) -> exporte ci-dessous AVANT d’exécuter ce script:
# export AWS_ACCESS_KEY_ID=AKIA...
# export AWS_SECRET_ACCESS_KEY=...
# export AWS_SESSION_TOKEN=...    # si temporaire
# export AWS_DEFAULT_REGION=eu-west-1
#
# B) PROFIL CLI (propre) -> décommente et configure:
# aws configure --profile "${AWS_PROFILE_NAME}"
# export AWS_PROFILE="${AWS_PROFILE_NAME}"
#
# C) SSO -> nécessite SSO configuré:
# export AWS_SDK_LOAD_CONFIG=1
# aws configure sso --profile "${AWS_PROFILE_NAME}"
# aws sso login --profile "${AWS_PROFILE_NAME}"
# export AWS_PROFILE="${AWS_PROFILE_NAME}"

# 2) Région par défaut si absente
: "${AWS_DEFAULT_REGION:=${AWS_REGION}}"; export AWS_DEFAULT_REGION

# 3) Check identité si aws CLI dispo
if command -v aws >/dev/null; then
  if aws sts get-caller-identity >/dev/null 2>&1; then
    log "AWS creds OK."
  else
    log "AUCUN identifiant actif (NoCredentialsError). Choisis A/B/C ci-dessus puis relance."
    exit 2
  fi
fi

# 4) Lier DVC au schéma choisi (LOCAL ONLY => pas de secrets versionnés)
dvc remote list || true
if ! dvc remote list | grep -q "^${REMOTE_NAME}\s*="; then
  log "Le remote '${REMOTE_NAME}' n’existe pas. Exemple: dvc remote add -d ${REMOTE_NAME} s3://mon-bucket/chemin"
  exit 3
fi

# Param génériques S3
dvc remote modify --local "${REMOTE_NAME}" region "${AWS_REGION}" || true

# Si tu utilises un profil CLI/SSO
if [ -n "${AWS_PROFILE:-}" ]; then
  dvc remote modify --local "${REMOTE_NAME}" profile "${AWS_PROFILE}"
  dvc remote modify --local "${REMOTE_NAME}" credentialpath "${HOME}/.aws/credentials" || true
fi

# S3-compatible (MinIO, etc.)
if [ -n "${S3_COMPAT_ENDPOINT}" ]; then
  dvc remote modify --local "${REMOTE_NAME}" endpointurl "${S3_COMPAT_ENDPOINT}"
  dvc remote modify --local "${REMOTE_NAME}" use_ssl "$(echo "${S3_COMPAT_ENDPOINT}" | grep -qi '^https' && echo true || echo false)"
  dvc remote modify --local "${REMOTE_NAME}" addressing_style "$([ "${PATH_STYLE}" = "true" ] && echo path || echo auto)" || true
fi

# 5) Sanity quick tests
if command -v aws >/dev/null; then
  log "Test DVC ↔ remote..."
  dvc status -c || true
fi

log "Run: dvc repro"
dvc repro

