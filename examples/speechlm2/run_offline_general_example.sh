#!/usr/bin/env bash
# Example: general offline inference. Set env vars below, then run from anywhere.
#
# Required:
#   CHECKPOINT   HF checkpoint directory (config.json + weights)
#   WAV          Input wav (16 kHz mono after load)
#   OUTPUT_DIR   Output directory
#
# Optional:
#   NEMO_DIR, SYSTEM_PROMPT, HF cache vars, device

set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT to your HF checkpoint directory}"
: "${WAV:?Set WAV to your input wav file}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to your output directory}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${NEMO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful AI voice assistant.}"

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

python3 "${CODE_DIR}/examples/speechlm2/offline_voicechat_infer.py" \
  --checkpoint "${CHECKPOINT}" \
  --wav "${WAV}" \
  --system-prompt "${SYSTEM_PROMPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --code-dir "${CODE_DIR}"
