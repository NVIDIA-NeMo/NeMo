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

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

echo "Speech checkout: ${CODE_DIR}"
echo "Checkpoint: ${CHECKPOINT_FILE}"
echo "Mode: ${REQUESTED_MODE}"
echo "Output: ${OUTPUT_DIR}"
echo "Container mounts: ${CONTAINER_MOUNTS}"

exec srun \
    --account=nemotron_speech_tts \
    --partition=interactive \
    --job-name=nemotron_speech_tts_easymagpie_acoustic_eval \
    --nodes=1 \
    --ntasks=8 \
    --ntasks-per-node=8 \
    --gpus-per-node=8 \
    --cpus-per-task=8 \
    --time=02:00:00 \
    --exclusive \
    --mem=0 \
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
