#!/usr/bin/env bash
# Allocate one interactive ORD node and run paired eight-GPU validation.

set -euo pipefail
umask 077

usage() {
    cat <<'EOF'
Usage: start_easy_magpietts_acoustic_estimator_interactive.sh \
  HPARAMS_FILE CHECKPOINT_FILE CODEC_FILE MANIFEST_FILE AUDIO_DIR OUTPUT_DIR [flow|aux_projection|both]

Run this launcher in a detached screen session so the allocation survives SSH disconnects.
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_easy_magpietts_acoustic_estimator_eval.sh"
CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/nemotron/projects/nemotron_speech_tts/containers/nemo_26.06_rc3.sqsh}"
EVAL_DATA_HOST="${EVAL_DATA_HOST:-/lustre/fsw/portfolios/nemotron/projects/nemotron_speech_tts/data/raw_audio_data}"
CONTAINER_MOUNTS="/lustre/fsw:/lustre/fsw,/lustre/fs11:/lustre/fs11,${EVAL_DATA_HOST}:/data/TTS"
EVAL_GPUS="${EVAL_GPUS:-8}"
EVAL_CPUS_PER_TASK="${EVAL_CPUS_PER_TASK:-8}"
EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT:-02:00:00}"
EVAL_MEMORY="${EVAL_MEMORY:-0}"
EVAL_EXCLUSIVE="${EVAL_EXCLUSIVE:-true}"

for required_file in "${HPARAMS_FILE}" "${CHECKPOINT_FILE}" "${CODEC_FILE}" "${RUNNER}" "${CONTAINER}"; do
    [[ -r "${required_file}" ]] || {
        echo "Required file is not readable: ${required_file}" >&2
        exit 2
    }
done
[[ -d "${EVAL_DATA_HOST}" ]] || {
    echo "Evaluation data host directory is not readable: ${EVAL_DATA_HOST}" >&2
    exit 2
}
if ! [[ "${EVAL_GPUS}" =~ ^[1-8]$ && "${EVAL_CPUS_PER_TASK}" =~ ^[1-9][0-9]*$ ]]; then
    echo "EVAL_GPUS must be 1-8 and EVAL_CPUS_PER_TASK must be positive" >&2
    exit 2
fi
case "${EVAL_EXCLUSIVE}" in
    true) EXCLUSIVE_ARGS=(--exclusive) ;;
    false) EXCLUSIVE_ARGS=() ;;
    *)
        echo "EVAL_EXCLUSIVE must be true or false: ${EVAL_EXCLUSIVE}" >&2
        exit 2
        ;;
esac

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export EVAL_DEVICES="${EVAL_DEVICES:-${EVAL_GPUS}}"

echo "Speech checkout: ${CODE_DIR}"
echo "Checkpoint: ${CHECKPOINT_FILE}"
echo "Mode: ${REQUESTED_MODE}"
echo "Output: ${OUTPUT_DIR}"
echo "Container mounts: ${CONTAINER_MOUNTS}"
echo "Resources: ${EVAL_GPUS} GPU(s), ${EVAL_CPUS_PER_TASK} CPU(s)/task, memory=${EVAL_MEMORY}, exclusive=${EVAL_EXCLUSIVE}"
echo "Time limit: ${EVAL_TIME_LIMIT}"

exec srun \
    --account=nemotron_speech_tts \
    --partition=interactive \
    --job-name=nemotron_speech_tts_easymagpie_acoustic_eval \
    --nodes=1 \
    --ntasks="${EVAL_GPUS}" \
    --ntasks-per-node="${EVAL_GPUS}" \
    --gpus-per-node="${EVAL_GPUS}" \
    --cpus-per-task="${EVAL_CPUS_PER_TASK}" \
    --time="${EVAL_TIME_LIMIT}" \
    --mem="${EVAL_MEMORY}" \
    "${EXCLUSIVE_ARGS[@]}" \
    --kill-on-bad-exit=1 \
    --export=ALL \
    --no-container-mount-home \
    --container-image="${CONTAINER}" \
    --container-mounts="${CONTAINER_MOUNTS}" \
    --container-workdir="${CODE_DIR}" \
    bash "${RUNNER}" \
        "${HPARAMS_FILE}" \
        "${CHECKPOINT_FILE}" \
        "${CODEC_FILE}" \
        "${MANIFEST_FILE}" \
        "${AUDIO_DIR}" \
        "${OUTPUT_DIR}" \
        "${REQUESTED_MODE}"
