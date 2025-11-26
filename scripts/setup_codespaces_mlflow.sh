#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT}/.venv"
REQUIREMENTS="${ROOT}/requirements-ml.txt"
MLRUNS_DIR="${ROOT}/mlruns"
DB_FILE="${ROOT}/mlflow.sqlite"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
pip install -r "${REQUIREMENTS}"
mkdir -p "${MLRUNS_DIR}"
if [ ! -f "${DB_FILE}" ]; then
  sqlite3 "${DB_FILE}" "VACUUM;" || true
fi

NOHUP_OUT="${ROOT}/mlflow_nohup.out"
NOHUP_ERR="${ROOT}/mlflow_nohup.err"
nohup mlflow server --host 0.0.0.0 --port "${MLFLOW_PORT}" --backend-store-uri "sqlite://${DB_FILE}" --default-artifact-root "${MLRUNS_DIR}" >"${NOHUP_OUT}" 2>"${NOHUP_ERR}" &
echo "MLflow server starting on port ${MLFLOW_PORT} (logs: ${NOHUP_OUT})"
