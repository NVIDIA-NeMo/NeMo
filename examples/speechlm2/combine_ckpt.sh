#!/bin/bash
# Create a combined S2S+RNNT HF checkpoint.
#
# Pipeline:
#   Step A : FSDP .ckpt  →  STT HF dir       (to_hf.py)
#   Step B1: STT HF + TTS  →  S2S HF dir     (nemotron_voicechat_infer_voice_lock.py)
#   Step B2: S2S HF + RNNT →  final combined  (combine_s2s_rnnt_checkpoint.py)
#
# Usage:
#   bash combine_ckpt.sh \
#     --stt-fsdp-ckpt /path/to/step-XXXX.ckpt \
#     --stt-ckpt-config /path/to/exp_config.yaml \
#     --tts-ckpt /path/to/tts.ckpt \
#     --rnnt-nemo /path/to/asr.nemo \
#     --speaker-wav /path/to/speaker.wav \
#     --output-root /path/to/output \
#     --docker-image-tar /path/to/image.tar \
#     --docker-image my_image:latest \
#     --tag NAME \
#     --step N \
#     --docker-vols "-v /home:/home"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root (Speech/) — parent of examples/
OSS_CODE="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Combine STT (FSDP) + TTS + RNNT into a voice-locked S2S HF checkpoint (Docker).

Required:
  --stt-fsdp-ckpt PATH       FSDP STT checkpoint (.ckpt file or dir)
  --stt-ckpt-config PATH     Training exp_config.yaml for STT
  --tts-ckpt PATH            TTS checkpoint (.ckpt)
  --rnnt-nemo PATH           RNNT ASR .nemo (Step A encoder bootstrap + Step B2 merge)
  --speaker-wav PATH         Reference wav for voice lock
  --output-root PATH         Output root for S2S / S2S+RNNT exports
  --docker-image-tar PATH    Docker image tar to load if image missing
  --docker-image NAME        Docker image tag (e.g. my_image:latest)
  --tag NAME                 Checkpoint tag used in output names
  --step N                   STT step used in output names
  --docker-vols "ARGS"       Docker -v mounts, e.g. "-v /home:/home"

Optional:
  --speaker-name NAME        Speaker key (default: Aria)
  --config-name NAME         Hydra config name (default: nemotron_voicechat_nano9b)
  --merge-config-path PATH   Hydra config dir (default: <this_script>/conf)
  --cache PATH               HF/Torch/NeMo cache dir (default: <output-root>/cache)
  --results-root PATH        Exp-manager logs (default: <output-root>/combined_ckpt_result_dir)
  --gpu VALUE                GPU id for NVIDIA_VISIBLE_DEVICES (default: 0).
                             Also accepts docker-style "device=0" or "all"
  --seed N                   RNG seed (default: 42)
  -h, --help                 Show this help

Example:
  bash combine_ckpt.sh \
    --stt-fsdp-ckpt /data/stt/step-10000.ckpt \
    --stt-ckpt-config /data/stt/exp_config.yaml \
    --tts-ckpt /data/tts/tts.ckpt \
    --rnnt-nemo /data/rnnt/model.nemo \
    --speaker-wav /data/speaker/speaker.wav \
    --output-root /data/output \
    --docker-image-tar /data/image.tar \
    --docker-image my_image:latest \
    --tag my_exp \
    --step 10000 \
    --docker-vols "-v /data:/data"
EOF
}

# ---------------------------------------------------------------------------
# Defaults (no cluster-specific hardcoding)
# ---------------------------------------------------------------------------
STT_FSDP_CKPT=""
STT_CKPT_CONFIG=""
TTS_CKPT=""
RNNT_NEMO=""
SPK_WAV=""
SPK_NAME="Aria"
OUTPUT_ROOT=""
DOCKER_IMAGE_TAR=""
DOCKER_IMAGE=""
MERGE_CONFIG_PATH="${SCRIPT_DIR}/conf"
TAG=""
STEP=""
DOCKER_VOLS=""
CACHE=""
RESULTS_ROOT=""
GPU="0"
CONFIG_NAME="nemotron_voicechat_nano9b"
SEED=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stt-fsdp-ckpt) STT_FSDP_CKPT="$2"; shift 2 ;;
    --stt-ckpt-config) STT_CKPT_CONFIG="$2"; shift 2 ;;
    --tts-ckpt) TTS_CKPT="$2"; shift 2 ;;
    --rnnt-nemo) RNNT_NEMO="$2"; shift 2 ;;
    --speaker-wav) SPK_WAV="$2"; shift 2 ;;
    --speaker-name) SPK_NAME="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --docker-image-tar) DOCKER_IMAGE_TAR="$2"; shift 2 ;;
    --docker-image) DOCKER_IMAGE="$2"; shift 2 ;;
    --merge-config-path) MERGE_CONFIG_PATH="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --docker-vols) DOCKER_VOLS="$2"; shift 2 ;;
    --cache) CACHE="$2"; shift 2 ;;
    --results-root) RESULTS_ROOT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --config-name) CONFIG_NAME="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

require_arg() {
  local val="$1"
  local name="$2"
  if [[ -z "${val}" ]]; then
    echo "ERROR: missing required argument: ${name}" >&2
    usage >&2
    exit 1
  fi
}

require_path() {
  local p="$1"
  local label="$2"
  if [[ ! -e "${p}" ]]; then
    echo "ERROR: missing ${label}: ${p}" >&2
    exit 1
  fi
}

require_arg "${STT_FSDP_CKPT}" "--stt-fsdp-ckpt"
require_arg "${STT_CKPT_CONFIG}" "--stt-ckpt-config"
require_arg "${TTS_CKPT}" "--tts-ckpt"
require_arg "${RNNT_NEMO}" "--rnnt-nemo"
require_arg "${SPK_WAV}" "--speaker-wav"
require_arg "${OUTPUT_ROOT}" "--output-root"
require_arg "${DOCKER_IMAGE_TAR}" "--docker-image-tar"
require_arg "${DOCKER_IMAGE}" "--docker-image"
require_arg "${TAG}" "--tag"
require_arg "${STEP}" "--step"
require_arg "${DOCKER_VOLS}" "--docker-vols"

CACHE="${CACHE:-${OUTPUT_ROOT}/cache}"
RESULTS_ROOT="${RESULTS_ROOT:-${OUTPUT_ROOT}/combined_ckpt_result_dir}"
MERGE_CONFIG_PATH="${MERGE_CONFIG_PATH:-${SCRIPT_DIR}/conf}"

# --gpus is broken on some hosts after driver reloads (NVML Unknown Error).
# --runtime=nvidia + NVIDIA_VISIBLE_DEVICES works here.
case "${GPU}" in
  device=*) NVIDIA_VISIBLE_DEVICES="${GPU#device=}" ;;
  *) NVIDIA_VISIBLE_DEVICES="${GPU}" ;;
esac

require_path "${DOCKER_IMAGE_TAR}" "docker image tar"
require_path "${STT_FSDP_CKPT}" "STT FSDP ckpt"
require_path "${STT_CKPT_CONFIG}" "STT exp config"
require_path "${TTS_CKPT}" "TTS ckpt"
require_path "${RNNT_NEMO}" "RNNT .nemo"
require_path "${SPK_WAV}" "speaker wav"
require_path "${OSS_CODE}" "Speech codebase"
require_path "${OSS_CODE}/examples/speechlm2/to_hf.py" "to_hf.py"
require_path "${OSS_CODE}/examples/speechlm2/nemotron_voicechat_infer_voice_lock.py" "voice_lock script"
require_path "${OSS_CODE}/examples/speechlm2/combine_s2s_rnnt_checkpoint.py" "combine_rnnt script"
require_path "${MERGE_CONFIG_PATH}/${CONFIG_NAME}.yaml" "merge config"

# ===========================================================================
# DERIVED PATHS
# ===========================================================================
TTS_NAME="$(basename "${TTS_CKPT}" .ckpt)"
STT_HF_CKPT="${STT_FSDP_CKPT%.ckpt}_hf"
HF_CKPT_NAME="${TAG}-step${STEP}-tts-eartts-${TTS_NAME}"
HF_EXPORT_DIR="${OUTPUT_ROOT}/${TAG}/s2s/${HF_CKPT_NAME}"
OUTPUT_DIR="${OUTPUT_ROOT}/${TAG}/s2s_rnnt/${HF_CKPT_NAME}_rnnt_cand10_preproc_enc_att0_vci50"

mkdir -p "${CACHE}" "${RESULTS_ROOT}" "${OUTPUT_ROOT}"

# to_hf.py reads ckpt_config as a file and does not accept model config
# overrides. Create a temporary copy that:
#   1) points model.pretrained_asr at --rnnt-nemo (encoder architecture bootstrap)
#   2) clears model.pretrained_s2s_model so Step A does not reload a training
#      warm-start checkpoint; weights come from the FSDP ckpt instead
# Leave the source training config unchanged.
STT_CKPT_CONFIG_OVERRIDE="$(mktemp "${CACHE}/stt_exp_config.XXXXXX.yaml")"
cleanup() {
  rm -f "${STT_CKPT_CONFIG_OVERRIDE}"
}
trap cleanup EXIT

python3 - "${STT_CKPT_CONFIG}" "${STT_CKPT_CONFIG_OVERRIDE}" "${RNNT_NEMO}" <<'PY'
import pathlib
import re
import sys

source, destination, pretrained_asr = map(pathlib.Path, sys.argv[1:])
lines = source.read_text().splitlines(keepends=True)
asr_matches = 0
s2s_matches = 0
for index, line in enumerate(lines):
    newline = "\n" if line.endswith("\n") else ""
    if re.match(r"^  pretrained_asr\s*:", line):
        lines[index] = f"  pretrained_asr: {pretrained_asr}{newline}"
        asr_matches += 1
    elif re.match(r"^  pretrained_s2s_model\s*:", line):
        # Empty => skip init_from_model_from_ckpt; FSDP weights are loaded next.
        lines[index] = f"  pretrained_s2s_model: null{newline}"
        s2s_matches += 1

if asr_matches != 1:
    raise SystemExit(
        f"Expected exactly one top-level model.pretrained_asr in {source}; found {asr_matches}"
    )
if s2s_matches != 1:
    raise SystemExit(
        f"Expected exactly one top-level model.pretrained_s2s_model in {source}; found {s2s_matches}"
    )

destination.write_text("".join(lines))
PY

# ===========================================================================
# LOAD DOCKER IMAGE (skipped if already present)
# ===========================================================================
if ! docker image inspect "${DOCKER_IMAGE}" > /dev/null 2>&1; then
  echo "Loading Docker image from ${DOCKER_IMAGE_TAR} ..."
  docker load < "${DOCKER_IMAGE_TAR}"
  echo "Image loaded: ${DOCKER_IMAGE}"
else
  echo "Docker image already loaded: ${DOCKER_IMAGE}"
fi

echo "============================================================"
echo " TAG        : ${TAG}  STEP: ${STEP}"
echo " FSDP ckpt  : ${STT_FSDP_CKPT}"
echo " STT HF     : ${STT_HF_CKPT}"
echo " S2S HF     : ${HF_EXPORT_DIR}"
echo " Final out  : ${OUTPUT_DIR}"
echo " Code       : ${OSS_CODE}"
echo " Config     : ${MERGE_CONFIG_PATH}/${CONFIG_NAME}.yaml"
echo " Image      : ${DOCKER_IMAGE}"
echo " GPU        : NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES} (runtime=nvidia)"
echo " Speaker    : ${SPK_NAME} <- ${SPK_WAV}"
echo " Step A ASR : ${RNNT_NEMO} (via --rnnt-nemo)"
echo " Step A note: pretrained_s2s_model cleared (load FSDP ckpt directly)"
echo "============================================================"

# ===========================================================================
# STEP A: FSDP .ckpt → STT HF dir
# ===========================================================================
echo ""
echo "=== Step A: FSDP → STT HF ==="

CONVERSION_DONE="${STT_HF_CKPT}/.conversion_done"
if [[ -f "${CONVERSION_DONE}" ]]; then
  echo "  Already done: ${STT_HF_CKPT}"
else
  mkdir -p "${STT_HF_CKPT}"
  # shellcheck disable=SC2086
  # voicechat image ENTRYPOINT is /bin/bash — pass -c, not "bash -c"
  docker run --rm --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}" \
    ${DOCKER_VOLS} \
    --ipc=host --ulimit memlock=-1 \
    --entrypoint bash \
    "${DOCKER_IMAGE}" \
    -c "
      export PYTHONPATH=${OSS_CODE}:\${PYTHONPATH:-}
      export HF_HOME=${CACHE} TORCH_HOME=${CACHE} NEMO_CACHE_DIR=${CACHE}
      export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
      python3 ${OSS_CODE}/examples/speechlm2/to_hf.py \
          class_path=nemo.collections.speechlm2.models.DuplexSTTModel \
          ckpt_path=${STT_FSDP_CKPT} \
          ckpt_config=${STT_CKPT_CONFIG_OVERRIDE} \
          output_dir=${STT_HF_CKPT} \
      && touch ${CONVERSION_DONE}
    "
  if [[ ! -f "${CONVERSION_DONE}" ]]; then
    echo "ERROR: Step A failed." >&2
    exit 1
  fi
fi

# ===========================================================================
# STEP B1: STT HF + TTS → S2S HF checkpoint (with voice lock)
# ===========================================================================
echo ""
echo "=== Step B1: STT HF + TTS → S2S HF  ==="

HF_CKPT_BASENAME="$(basename "${HF_EXPORT_DIR}")"
RESULTS_DIR="${RESULTS_ROOT}/${HF_CKPT_BASENAME}"
mkdir -p "${RESULTS_DIR}"

# shellcheck disable=SC2086
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}" \
  ${DOCKER_VOLS} \
  --ipc=host --ulimit memlock=-1 \
  --entrypoint bash \
  "${DOCKER_IMAGE}" \
  -c "
    export PYTHONPATH=${OSS_CODE}:\${PYTHONPATH:-}
    export HF_HOME=${CACHE} TORCH_HOME=${CACHE} NEMO_CACHE_DIR=${CACHE}
    export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
    export WANDB_MODE=offline
    export MASTER_ADDR=localhost MASTER_PORT=\$(python3 -c 'import socket; s=socket.socket(); s.bind((\"\",0)); print(s.getsockname()[1]); s.close()')
    export LOCAL_RANK=0 RANK=0 WORLD_SIZE=1
    HYDRA_FULL_ERROR=1 TORCH_CUDNN_V8_API_ENABLED=1 \
    python3 ${OSS_CODE}/examples/speechlm2/nemotron_voicechat_infer_voice_lock.py \
        --config-path=${MERGE_CONFIG_PATH} \
        --config-name=${CONFIG_NAME} \
        exp_manager.name=${HF_CKPT_BASENAME} \
        exp_manager.explicit_log_dir=${RESULTS_DIR} \
        ++model.stt.model.pretrained_s2s_model=${STT_HF_CKPT} \
        ++model.speech_generation.model.pretrained_model=${TTS_CKPT} \
        ++model.inference_speaker_name=${SPK_NAME} \
        ++model.speech_generation.model.tts_config.backbone_config.sliding_window=7500 \
        ++model.speech_generation.model.tts_config.use_audio_prompt_frozen_projection=True \
        ++model.speech_generation.model.inference_guidance_scale=0.2 \
        ++model.speech_generation.model.inference_guidance_enabled=True \
        ++model.speech_generation.model.inference_top_p_or_k=0.95 \
        ++model.speech_generation.model.inference_noise_scale=0.001 \
        ++model.speech_generation.model.use_system_prompt=False \
        ++model.stt.model.eval_text_turn_taking=True \
        trainer.num_nodes=1 \
        data.train_ds.seed=${SEED} \
        data.validation_ds.seed=${SEED} \
        data.validation_ds.batch_size=1 \
        ++trainer.limit_val_batches=0.0 \
        ++trainer.precision=32 \
        ++trainer.max_steps=1 \
        ++trainer.val_check_interval=1 \
        ++hf_export_dir=${HF_EXPORT_DIR} \
        ++model.stt.model.incremental_loading=True \
        ++model.stt.model.use_function_head=True \
        '++model.stt.model.override_tokens.bos_token=\"<s>\"' \
        '++model.stt.model.override_tokens.eos_token=\"</s>\"' \
        '++model.stt.model.override_tokens.pad_token=\"<SPECIAL_12>\"' \
        '++model.stt.model.bos_token=\"<s>\"' \
        '++model.stt.model.eos_token=\"</s>\"' \
        '++model.stt.model.pad_token=\"<SPECIAL_12>\"' \
        \"+register_speaker_dict={${SPK_NAME}: ${SPK_WAV}}\" \
        ++reinit_audio_prompt_frozen_projection=True
  "

if [[ ! -d "${HF_EXPORT_DIR}" ]]; then
  echo "ERROR: Step B1 failed — ${HF_EXPORT_DIR} not found." >&2
  exit 1
fi

# ===========================================================================
# STEP B2: S2S HF + RNNT → final combined checkpoint
# ===========================================================================
echo ""
echo "=== Step B2: S2S HF + RNNT → final checkpoint ==="

# shellcheck disable=SC2086
docker run --rm --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}" \
  ${DOCKER_VOLS} \
  --ipc=host --ulimit memlock=-1 \
  --entrypoint bash \
  "${DOCKER_IMAGE}" \
  -c "
    export PYTHONPATH=${OSS_CODE}:\${PYTHONPATH:-}
    python3 ${OSS_CODE}/examples/speechlm2/combine_s2s_rnnt_checkpoint.py \
        --s2s_checkpoint_dir ${HF_EXPORT_DIR} \
        --rnnt_nemo_path ${RNNT_NEMO} \
        --output_dir ${OUTPUT_DIR} \
        --overwrite
  "

echo ""
echo "============================================================"
echo " Done. Final checkpoint: ${OUTPUT_DIR}"
echo "============================================================"
