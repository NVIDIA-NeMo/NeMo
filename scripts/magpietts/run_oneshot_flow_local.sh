#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${CODEC_PATH:?Set CODEC_PATH to the 12 fps hybrid semantic-residual codec .nemo file}"
: "${IPA_TOKENIZER:?Set IPA_TOKENIZER to the IPA tokenizer JSON file}"
: "${TRAIN_INPUT_CFG:?Set TRAIN_INPUT_CFG to the Lhotse training input YAML}"
: "${VAL_INPUT_CFG:?Set VAL_INPUT_CFG to the Lhotse validation input YAML}"

CONFIG_NAME="${CONFIG_NAME:-easy_magpietts_lhotse_oneshot_flow}"
MAX_STEPS="${MAX_STEPS:-10000}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1000}"
TRAIN_BATCH_DURATION="${TRAIN_BATCH_DURATION:-10}"
VAL_BATCH_DURATION="${VAL_BATCH_DURATION:-10}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"
RUN_VAL_INFERENCE="${RUN_VAL_INFERENCE:-true}"
CREATE_WANDB_LOGGER="${CREATE_WANDB_LOGGER:-true}"
CREATE_CHECKPOINTS="${CREATE_CHECKPOINTS:-false}"
RUN_NAME="${RUN_NAME:-easy-magpie-oneshot-flow-local}"
EXP_DIR="${EXP_DIR:-${REPO_ROOT}/easy_magpie_oneshot_flow_runs}"
SEED_MODEL="${SEED_MODEL:-}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

overrides=(
  "name=${RUN_NAME}"
  "model.codecmodel_path=${CODEC_PATH}"
  "model.phoneme_tokenizer.tokenizer_path=${IPA_TOKENIZER}"
  "model.train_ds.dataset.input_cfg=${TRAIN_INPUT_CFG}"
  "model.train_ds.dataset.batch_duration=${TRAIN_BATCH_DURATION}"
  "model.train_ds.dataset.num_workers=${DATALOADER_WORKERS}"
  "model.validation_ds.dataset.input_cfg=${VAL_INPUT_CFG}"
  "model.validation_ds.dataset.batch_duration=${VAL_BATCH_DURATION}"
  "+model.run_val_inference=${RUN_VAL_INFERENCE}"
  "+model.use_multilingual_asr=false"
  "+model.use_utmos=false"
  "trainer.num_nodes=1"
  "trainer.devices=1"
  "trainer.accelerator=gpu"
  "trainer.strategy=auto"
  "trainer.precision=bf16-mixed"
  "trainer.max_steps=${MAX_STEPS}"
  "trainer.log_every_n_steps=1"
  "trainer.val_check_interval=${VAL_CHECK_INTERVAL}"
  "+trainer.limit_val_batches=1"
  "trainer.num_sanity_val_steps=0"
  "+trainer.enable_progress_bar=true"
  "exp_manager.exp_dir=${EXP_DIR}"
  "exp_manager.create_tensorboard_logger=false"
  "exp_manager.create_wandb_logger=${CREATE_WANDB_LOGGER}"
  "exp_manager.create_checkpoint_callback=${CREATE_CHECKPOINTS}"
  "exp_manager.resume_if_exists=false"
)

if [[ -n "${SEED_MODEL}" ]]; then
  if [[ ! -f "${SEED_MODEL}" ]]; then
    echo "SEED_MODEL does not exist: ${SEED_MODEL}" >&2
    exit 1
  fi
  overrides+=("init_from_nemo_model=${SEED_MODEL}")
fi

if [[ "${CREATE_WANDB_LOGGER}" == "true" ]]; then
  : "${WANDB_PROJECT:?Set WANDB_PROJECT when CREATE_WANDB_LOGGER=true}"
  overrides+=(
    "exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT}"
    "exp_manager.wandb_logger_kwargs.name=${RUN_NAME}"
    "exp_manager.wandb_logger_kwargs.resume=false"
  )
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    overrides+=("exp_manager.wandb_logger_kwargs.entity=${WANDB_ENTITY}")
  fi
fi

cd "${REPO_ROOT}"
exec python examples/tts/easy_magpietts.py \
  --config-name="${CONFIG_NAME}" \
  "${overrides[@]}"
