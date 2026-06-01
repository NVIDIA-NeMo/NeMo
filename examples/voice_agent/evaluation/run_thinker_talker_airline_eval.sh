#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICE_AGENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NEMO_ROOT="$(cd "${VOICE_AGENT_DIR}/../.." && pwd)"

USER_URL="${USER_URL:-ws://localhost:8766}"
AGENT_URL="${AGENT_URL:-ws://localhost:7860/api/ws?pipeline_mode=cascaded/thinker_talker}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/eval_results_thinker_talker}"
PYTHON_BIN="${PYTHON_BIN:-${VOICE_AGENT_DIR}/.venv/bin/python3}"

export PYTHONPATH="${NEMO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

mkdir -p "${OUTPUT_DIR}"

before="$(mktemp)"
after="$(mktemp)"
find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'eval_*' -print | sort > "${before}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_evaluation.py" \
  --user-url "${USER_URL}" \
  --agent-url "${AGENT_URL}" \
  --domain thinker_talker_airline \
  --judge-url "" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'eval_*' -print | sort > "${after}"
session_dir="$(comm -13 "${before}" "${after}" | tail -n 1)"
rm -f "${before}" "${after}"

if [[ -z "${session_dir}" ]]; then
  session_dir="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'eval_*' -print | sort | tail -n 1)"
fi

if [[ -z "${session_dir}" ]]; then
  echo "No evaluation session directory found under ${OUTPUT_DIR}" >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/score_thinker_talker_airline.py" \
  --session-dir "${session_dir}" \
  --write \
  --pretty

echo "Thinker/Talker airline evaluation results: ${session_dir}"
