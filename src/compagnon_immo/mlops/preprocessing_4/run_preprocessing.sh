# mlops/preprocessing_4/run_preprocessing.sh
#!/usr/bin/env bash
# why: rendre le stage verbeux/robuste pour diagnostiquer (paths, venv, encodage, permissions)

set -Eeuo pipefail
trap 'ec=$?; echo "[ERROR] ${BASH_SOURCE[0]} failed at line $LINENO (exit=$ec)"; exit $ec' ERR

# --- Config (override possible via env/DVC) ---
IN="${IN:-data/df_sample.csv}"
OUT_DIR="${OUT_DIR:-data/processed}"
# Choisis l’un des deux: module Python OU script .py (le script essaie les deux)
PY_MODULE="${PY_MODULE:-mlops.preprocessing_4.main}"
PY_ENTRY="${PY_ENTRY:-scripts/preprocessing.py}"
LOG_DIR="${LOG_DIR:-mlops/preprocessing_4/logs}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/preprocessing_${TIMESTAMP}.log"

# --- Contexte & préchecks ---
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

{
  echo "=== preprocessing start @ ${TIMESTAMP} ==="
  echo "pwd: $(pwd)"
  echo "python: $(command -v python || true)"
  python -V || true
  echo "IN=${IN}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "PY_MODULE=${PY_MODULE}"
  echo "PY_ENTRY=${PY_ENTRY}"
  echo "--- ls inputs ---"
  ls -lah "$(dirname "${IN}")" || true
} | tee -a "${LOG_FILE}"

if [[ ! -f "${IN}" ]]; then
  echo "[FATAL] Input file not found: ${IN}" | tee -a "${LOG_FILE}"
  exit 2
fi

# --- Exécution Python (module ou script) ---
set -x  # trace les commandes exécutées
if python -c "import importlib; importlib.import_module('${PY_MODULE}')" >/dev/null 2>&1; then
  python -m "${PY_MODULE}" --input "${IN}" --out-dir "${OUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"
elif [[ -f "${PY_ENTRY}" ]]; then
  python "${PY_ENTRY}" --input "${IN}" --out-dir "${OUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"
else
  set +x
  echo "[FATAL] Neither module (${PY_MODULE}) nor script (${PY_ENTRY}) is available." | tee -a "${LOG_FILE}"
  exit 3
fi
set +x

# --- Post-checks ---
if [[ ! -d "${OUT_DIR}" ]]; then
  echo "[FATAL] OUT_DIR not created: ${OUT_DIR}" | tee -a "${LOG_FILE}"
  exit 4
fi

echo "=== preprocessing done ===" | tee -a "${LOG_FILE}"
exit 0

