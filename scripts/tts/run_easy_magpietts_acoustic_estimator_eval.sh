#!/usr/bin/env bash
# Run paired EasyMagpie acoustic-estimator validation inside an allocated container.

set -euo pipefail
umask 077

usage() {
    cat <<'EOF'
Usage: run_easy_magpietts_acoustic_estimator_eval.sh \
  HPARAMS_FILE CHECKPOINT_FILE CODEC_FILE MANIFEST_FILE AUDIO_DIR OUTPUT_DIR [flow|aux_projection|both]
EOF
}

if [[ $# -lt 6 || $# -gt 7 ]]; then
    usage >&2
    exit 2
fi

HPARAMS_FILE="$1"
CHECKPOINT_FILE="$2"
CODEC_FILE="$3"
MANIFEST_FILE="$4"
AUDIO_DIR="$5"
OUTPUT_DIR="$6"
REQUESTED_MODE="${7:-both}"

case "${REQUESTED_MODE}" in
    flow) MODES=(flow) ;;
    aux_projection) MODES=(aux_projection) ;;
    both) MODES=(flow aux_projection) ;;
    *)
        echo "Mode must be flow, aux_projection, or both: ${REQUESTED_MODE}" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_SCRIPT="${CODE_DIR}/examples/tts/evaluate_easy_magpietts_acoustic_estimator.py"
SETENV_FILE="${SETENV_FILE:-$(cd "${CODE_DIR}/.." && pwd)/../setenv.sh}"
CACHE_ROOT="${CACHE_ROOT:-/lustre/fsw/portfolios/nemotron/projects/nemotron_speech_tts/data/cache}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
EVAL_DEVICES="${EVAL_DEVICES:-${SLURM_NTASKS:-8}}"
EVAL_SEED="${EVAL_SEED:-9}"

if [[ -r "${SETENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${SETENV_FILE}"
    set +a
fi

for required_file in "${HPARAMS_FILE}" "${CHECKPOINT_FILE}" "${CODEC_FILE}" "${MANIFEST_FILE}" "${EVAL_SCRIPT}"; do
    [[ -r "${required_file}" ]] || {
        echo "Required file is not readable: ${required_file}" >&2
        exit 2
    }
done
[[ -d "${AUDIO_DIR}" ]] || {
    echo "Audio directory is not readable: ${AUDIO_DIR}" >&2
    exit 2
}
command -v python >/dev/null || {
    echo "python is unavailable" >&2
    exit 2
}

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export NEMO_CACHE_DIR="${CACHE_ROOT}/torch/nemo"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

LIMIT_ARGS=()
if [[ -n "${EVAL_LIMIT_VAL_BATCHES:-}" ]]; then
    LIMIT_ARGS=(--limit-val-batches "${EVAL_LIMIT_VAL_BATCHES}")
fi

for mode in "${MODES[@]}"; do
    echo "Starting ${mode} validation on ${SLURM_NTASKS:-1} task(s); output=${OUTPUT_DIR}/${mode}"
    python "${EVAL_SCRIPT}" \
        --hparams-file "${HPARAMS_FILE}" \
        --checkpoint-file "${CHECKPOINT_FILE}" \
        --codecmodel-path "${CODEC_FILE}" \
        --manifest-path "${MANIFEST_FILE}" \
        --audio-dir "${AUDIO_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --acoustic-inference-mode "${mode}" \
        --batch-size "${EVAL_BATCH_SIZE}" \
        --num-workers "${EVAL_NUM_WORKERS}" \
        --devices "${EVAL_DEVICES}" \
        --seed "${EVAL_SEED}" \
        "${LIMIT_ARGS[@]}"
done

echo "Evaluation complete: ${OUTPUT_DIR}"
