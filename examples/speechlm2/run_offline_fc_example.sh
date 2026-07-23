#!/usr/bin/env bash
# Example: offline inference with function calling (two-pass).
#
# Required:
#   CHECKPOINT, WAV, OUTPUT_DIR, API_RESPONSE_JSON
#
# Optional:
#   NEMO_DIR, TEMPLATE, TOOLS_JSON, SYSTEM_MESSAGE

set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT to your HF checkpoint directory}"
: "${WAV:?Set WAV to your input wav file}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to your output directory}"
: "${API_RESPONSE_JSON:?Set API_RESPONSE_JSON to your tool response JSON}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${NEMO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${TORCH_HOME:-/tmp/cache}}"
export TORCH_HOME="${TORCH_HOME:-${HF_HOME}}"
export NEMO_CACHE_DIR="${NEMO_CACHE_DIR:-${HF_HOME}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export LOCAL_RANK="${LOCAL_RANK:-0}" RANK="${RANK:-0}" WORLD_SIZE="${WORLD_SIZE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')}"

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  --checkpoint "${CHECKPOINT}"
  --wav "${WAV}"
  --api-response-json "${API_RESPONSE_JSON}"
  --output-dir "${OUTPUT_DIR}"
  --code-dir "${CODE_DIR}"
)
[[ -n "${TEMPLATE:-}" ]] && ARGS+=(--template "${TEMPLATE}")
[[ -n "${TOOLS_JSON:-}" ]] && ARGS+=(--tools-json "${TOOLS_JSON}")
[[ -n "${SYSTEM_MESSAGE:-}" ]] && ARGS+=(--system-message "${SYSTEM_MESSAGE}")

python3 "${CODE_DIR}/examples/speechlm2/offline_voicechat_fc_infer.py" "${ARGS[@]}"
