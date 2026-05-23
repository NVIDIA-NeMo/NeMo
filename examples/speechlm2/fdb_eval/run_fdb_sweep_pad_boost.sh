#!/bin/bash
# Sweep inference_pad_boost over [0, 1, 2, 3, 5] using the 21-sample FDB mini shar.
# Each run takes ~10 min (mini shar) vs 2.5h (full shar).
# Runs FDB scoring after each inference and prints a summary table at the end.
#
# Usage:
#   bash run_fdb_sweep_pad_boost.sh [--results-base DIR] [--pad-boosts "0 1 2 3 5"]
#
# Requires: Docker with --gpus all; same machine setup as run_fdb_local.sh

set -e

# =========================================
# DEFAULTS — override via args
# =========================================
RESULTS_BASE="/mnt/point2/VoiceChat/fdb_eval_row26/results/sweep_pad_boost"
PAD_BOOSTS="0 1 2 3 5"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-base) RESULTS_BASE="$2"; shift 2 ;;
        --pad-boosts)   PAD_BOOSTS="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# =========================================
# FIXED PATHS
# =========================================
HF_CKPT_PATH="/mnt/point2/VoiceChat/GA_FC_model/e70-step28008-tts-eartts-34014_asr_cand3_0.6b-PK_ep0_01988_300"
RNNT_ASR_CKPT="/mnt/point2/VoiceChat/fdb_eval_row26/rnnt_ckpt/cand3_kratos_ep0_01988_300.nemo"
TTS_CKPT="/mnt/point2/VoiceChat/fdb_eval_row26/tts_ckpt/eartts_megan_step34014.ckpt"
INF_SPK_REF="/mnt/point2/VoiceChat/speaker_prompt/Mg_a_00759.wav"
CONFIG_PATH="/mnt/point2/VoiceChat/fdb_eval_row26/config"
MINI_SHAR_BASE="/mnt/point2/VoiceChat/fdb_eval_row26/fdb_shar_mini"
FDB_CODE_DIR="/home/hdubey/Full-Duplex-Bench-NV"
CLIENT_KEY_JSONL="/mnt/point2/VoiceChat/fdb_eval_row26/client_key.jsonl"
ORIGINAL_DATA_BASE="/mnt/point2/VoiceChat/fdb_eval_row26/original_data/v1.0"
REORGANIZE_SCRIPT="${FDB_CODE_DIR}/dataset/reorganize_candor_outputs.py"
DOCKER_IMAGE="nvcr.io/nvidian/tegra-audio/nemo_framework_jhw:2.6.0rc0_torch_25.06_py3_teleaug_v4"

# Inference params (match vtrinh Apr21 baseline)
SEED=42
TEMPERATURE=0.8
TOP_P=0.9
REPETITION_PENALTY=1.2
PRESENCE_PENALTY=0.0
TTS_SLIDING_WINDOW=1500
TTS_GUIDANCE_SCALE=0.2
TTS_GUIDANCE_ENABLED=True
TTS_TOP_P_OR_K=0.95
TTS_NOISE_SCALE=0.001

mkdir -p "${RESULTS_BASE}"
SUMMARY_FILE="${RESULTS_BASE}/sweep_summary.tsv"
echo -e "pad_boost\tph_synthetic_TOR\tph_candor_TOR\ttt_TOR\ttt_latency\tui_TOR\tui_latency" > "${SUMMARY_FILE}"

declare -A EVAL_TASKS
EVAL_TASKS[candor_turn_taking]="smooth_turn_taking"
EVAL_TASKS[candor_pause_handling]="pause_handling"
EVAL_TASKS[synthetic_pause_handling]="pause_handling"
EVAL_TASKS[synthetic_user_interruption]="user_interruption"

# =========================================
# SWEEP LOOP
# =========================================
for PAD_BOOST in ${PAD_BOOSTS}; do
    echo ""
    echo "======================================================"
    echo "  inference_pad_boost = ${PAD_BOOST}"
    echo "======================================================"

    RUN_DIR="${RESULTS_BASE}/pad_boost_${PAD_BOOST}"
    mkdir -p "${RUN_DIR}"

    # ---------- STEP 1: Inference on mini shar ----------
    echo "[1/3] Running inference (21 samples, pad_boost=${PAD_BOOST}) ..."

    docker run --rm --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -v /mnt/point2:/mnt/point2 \
        -v /home/hdubey/NeMo_elena:/NeMo_elena \
        -v /home/hdubey:/home/hdubey \
        -e HF_CKPT_PATH="${HF_CKPT_PATH}" \
        -e RNNT_ASR_CKPT="${RNNT_ASR_CKPT}" \
        -e S2S_PRETRAINED_ASR="${RNNT_ASR_CKPT}" \
        -e TTS_CKPT="${TTS_CKPT}" \
        -e INF_SPK_REF="${INF_SPK_REF}" \
        -e RUN_DIR="${RUN_DIR}" \
        -e MINI_SHAR_BASE="${MINI_SHAR_BASE}" \
        -e PAD_BOOST="${PAD_BOOST}" \
        -e SEED="${SEED}" \
        -e TEMPERATURE="${TEMPERATURE}" \
        -e TOP_P="${TOP_P}" \
        -e REPETITION_PENALTY="${REPETITION_PENALTY}" \
        -e PRESENCE_PENALTY="${PRESENCE_PENALTY}" \
        -e TTS_SLIDING_WINDOW="${TTS_SLIDING_WINDOW}" \
        -e TTS_GUIDANCE_SCALE="${TTS_GUIDANCE_SCALE}" \
        -e TTS_GUIDANCE_ENABLED="${TTS_GUIDANCE_ENABLED}" \
        -e TTS_TOP_P_OR_K="${TTS_TOP_P_OR_K}" \
        -e TTS_NOISE_SCALE="${TTS_NOISE_SCALE}" \
        "${DOCKER_IMAGE}" \
        bash -c '
set -e
NEMO_COLL=/usr/local/lib/python3.12/dist-packages/nemo/collections
cp -r /NeMo_elena/nemo/collections/speechlm2 ${NEMO_COLL}/
for f in aligner_utils chunking_utils multispk_transcribe_utils tokenizer_utils; do
    src=/NeMo_elena/nemo/collections/asr/parts/utils/${f}.py
    [ -f "${src}" ] && cp "${src}" "${NEMO_COLL}/asr/parts/utils/${f}.py"
done
python3 /mnt/point2/VoiceChat/fdb_eval_row26/nemo_patches/apply_patches.py
cp /mnt/point2/VoiceChat/fdb_eval_row26/nemo_patches/nemo/collections/common/data/fallback.py \
   ${NEMO_COLL}/common/data/fallback.py
cp /mnt/point2/VoiceChat/fdb_eval_row26/nemo_patches/nemo/collections/audio/parts/utils/transforms.py \
   ${NEMO_COLL}/audio/parts/utils/transforms.py
pip install seaborn sacrebleu -q 2>/dev/null
python3 -c "from transformers.models.t5gemma import T5GemmaConfig" 2>/dev/null || \
    pip install "transformers==4.57.3" -q 2>&1 | tail -3
WHEEL_DIR=/mnt/point2/VoiceChat/fdb_eval_row26/mamba_wheels
if python3 -c "import mamba_ssm, causal_conv1d" 2>/dev/null; then
    echo "mamba_ssm already importable"
elif ls "${WHEEL_DIR}"/*.whl 2>/dev/null | grep -q .; then
    pip install "${WHEEL_DIR}"/*.whl --no-deps -q 2>&1 | tail -3
else
    mkdir -p "${WHEEL_DIR}"
    pip wheel causal-conv1d --no-build-isolation --no-deps -w "${WHEEL_DIR}" 2>&1 | tail -3
    pip wheel mamba-ssm     --no-build-isolation --no-deps -w "${WHEEL_DIR}" 2>&1 | tail -3
    pip wheel triton --no-deps -w "${WHEEL_DIR}" -q 2>&1 | tail -3
    pip install "${WHEEL_DIR}"/*.whl --no-deps -q 2>&1 | tail -3
fi
export TORCH_COMPILE_DISABLE=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE=3 HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=1 \
    /NeMo_elena/examples/speechlm2/nemotron_voicechat_infer.py \
    --config-path=/mnt/point2/VoiceChat/fdb_eval_row26/config \
    --config-name=infer_fdb_row26 \
    ++model.stt.model.eval_text_turn_taking=True \
    ++model.stt.model.asr_branch_decode=True \
    ++model.stt.model.pretrained_rnnt_asr=${RNNT_ASR_CKPT} \
    ++model.stt.model.pretrained_s2s_model=${HF_CKPT_PATH} \
    "++model.speech_generation.model.pretrained_model=${TTS_CKPT}" \
    "++model.inference_speaker_reference=${INF_SPK_REF}" \
    ++model.speech_generation.model.tts_config.backbone_config.sliding_window=${TTS_SLIDING_WINDOW} \
    trainer.num_nodes=1 \
    exp_manager.explicit_log_dir=${RUN_DIR} \
    ++trainer.limit_val_batches=null \
    ++trainer.precision=bf16 \
    ++trainer.max_steps=1 \
    data.validation_ds.seed=${SEED} \
    data.validation_ds.batch_size=1 \
    ++trainer.val_check_interval=1 \
    "data.validation_ds.datasets.candor_turn_taking.shar_path=${MINI_SHAR_BASE}/candor_turn_taking" \
    "data.validation_ds.datasets.candor_pause_handling.shar_path=${MINI_SHAR_BASE}/candor_pause_handling" \
    "data.validation_ds.datasets.synthetic_pause_handling.shar_path=${MINI_SHAR_BASE}/synthetic_pause_handling" \
    "data.validation_ds.datasets.synthetic_user_interruption.shar_path=${MINI_SHAR_BASE}/synthetic_user_interruption" \
    ++model.use_asr_timestamps=True \
    ++model.predict_user_text=True \
    ++model.stt.model.inference_pad_boost=${PAD_BOOST} \
    ++model.stt.model.inference_bos_boost=0 \
    ++model.stt.model.inference_eos_boost=0 \
    ++model.stt.model.use_function_head=True \
    ++model.stt.model.force_align_user_text=False \
    ++model.stt.model.incremental_loading=True \
    ++model.temperature=${TEMPERATURE} \
    ++model.top_p=${TOP_P} \
    ++model.repetition_penalty=${REPETITION_PENALTY} \
    ++model.presence_penalty=${PRESENCE_PENALTY} \
    ++model.speech_generation.model.inference_guidance_scale=${TTS_GUIDANCE_SCALE} \
    ++model.speech_generation.model.inference_guidance_enabled=${TTS_GUIDANCE_ENABLED} \
    ++model.speech_generation.model.inference_top_p_or_k=${TTS_TOP_P_OR_K} \
    ++model.speech_generation.model.inference_noise_scale=${TTS_NOISE_SCALE} \
    ++model.speech_generation.model.tts_config.use_audio_prompt_frozen_projection=True \
    ++model.speech_generation.model.use_system_prompt=False \
    ++trainer.devices=1 \
    2>&1 | tee "${RUN_DIR}/inference.log"
' 2>&1 | tee "${RUN_DIR}/docker.log"

    echo "[2/3] Running FDB scoring for pad_boost=${PAD_BOOST} ..."
    PRED_WAVS="${RUN_DIR}/validation_logs/pred_wavs"
    FDB_METRIC_DIR="${RUN_DIR}/fdb_metric"
    mkdir -p "${FDB_METRIC_DIR}"

    for DATASET_NAME in candor_turn_taking candor_pause_handling synthetic_pause_handling synthetic_user_interruption; do
        EVAL_TASK="${EVAL_TASKS[$DATASET_NAME]}"
        ORIGINAL_DATA_PATH="${ORIGINAL_DATA_BASE}/${DATASET_NAME}"
        DEST_DATASET_DIR="${FDB_METRIC_DIR}/v1.0/${DATASET_NAME}"

        python3 "${REORGANIZE_SCRIPT}" \
            --output_path "${PRED_WAVS}" \
            --original_data_path "${ORIGINAL_DATA_PATH}" \
            --revised_output_path "${FDB_METRIC_DIR}" \
            --dataset_name "${DATASET_NAME}" \
            --strict_ids \
            --clean_destination \
            --normal_pattern "^(?!.*clean).*${DATASET_NAME}.*${DATASET_NAME}_(\d+)_rank\d+\.wav\$"
        mkdir -p "${DEST_DATASET_DIR}"

        if [[ "$EVAL_TASK" == "user_interruption" ]]; then
            ASR_TASK="user_interruption"
        else
            ASR_TASK="full"
        fi

        docker run --rm --gpus all --ipc=host \
            -v /mnt/point2:/mnt/point2 \
            -v /home/hdubey:/home/hdubey \
            -e DATASET_NAME="${DATASET_NAME}" \
            -e ASR_TASK="${ASR_TASK}" \
            "${DOCKER_IMAGE}" \
            bash -c "
pip install soundfile -q 2>/dev/null
# Skip sample 46 in user_interruption (missing interrupt.json in original data)
[ '\${DATASET_NAME}' = 'synthetic_user_interruption' ] && \
    [ -d '${DEST_DATASET_DIR}/46' ] && \
    mv '${DEST_DATASET_DIR}/46' '${FDB_METRIC_DIR}/v1.0/sample46_skip_ui' 2>/dev/null || true
cd /home/hdubey/Full-Duplex-Bench-NV/get_transcript
python asr.py --root_dir '${DEST_DATASET_DIR}' --task '\${ASR_TASK}' \
    2>&1 | tee '${FDB_METRIC_DIR}/v1.0/asr_${DATASET_NAME}.log'
# Restore sample 46 if we moved it
[ -d '${FDB_METRIC_DIR}/v1.0/sample46_skip_ui' ] && \
    mv '${FDB_METRIC_DIR}/v1.0/sample46_skip_ui' '${DEST_DATASET_DIR}/46' 2>/dev/null || true
" 2>&1 | tail -5

        cd "${FDB_CODE_DIR}/evaluation"
        if [[ "${EVAL_TASK}" == "user_interruption" ]]; then
            python3 -c "
import sys; sys.path.insert(0,'.')
from eval_user_interruption import eval_user_interruption
eval_user_interruption('${DEST_DATASET_DIR}')
" 2>&1 | tee "${FDB_METRIC_DIR}/v1.0/eval_${DATASET_NAME}.log"
        else
            python3 evaluate.py --task "${EVAL_TASK}" --root_dir "${DEST_DATASET_DIR}" \
                2>&1 | tee "${FDB_METRIC_DIR}/v1.0/eval_${DATASET_NAME}.log"
        fi
        cd - > /dev/null
    done

    echo "[3/3] Extracting metrics for pad_boost=${PAD_BOOST} ..."
    PH_SYN=$(grep "Average take turn" "${FDB_METRIC_DIR}/v1.0/eval_synthetic_pause_handling.log" | awk '{print $NF}')
    PH_CAN=$(grep "Average take turn" "${FDB_METRIC_DIR}/v1.0/eval_candor_pause_handling.log"    | awk '{print $NF}')
    TT_TOR=$(grep "Average take turn" "${FDB_METRIC_DIR}/v1.0/eval_candor_turn_taking.log"        | awk '{print $NF}')
    TT_LAT=$(grep "Average latency"   "${FDB_METRIC_DIR}/v1.0/eval_candor_turn_taking.log"        | awk '{print $NF}')
    UI_TOR=$(grep "Average take turn" "${FDB_METRIC_DIR}/v1.0/eval_synthetic_user_interruption.log" | awk '{print $NF}')
    UI_LAT=$(grep "Average latency"   "${FDB_METRIC_DIR}/v1.0/eval_synthetic_user_interruption.log" | awk '{print $NF}')

    echo -e "${PAD_BOOST}\t${PH_SYN}\t${PH_CAN}\t${TT_TOR}\t${TT_LAT}\t${UI_TOR}\t${UI_LAT}" >> "${SUMMARY_FILE}"
    echo ""
    echo "  pad_boost=${PAD_BOOST} | PH_syn=${PH_SYN} PH_can=${PH_CAN} | TT_TOR=${TT_TOR} TT_lat=${TT_LAT} | UI_TOR=${UI_TOR} UI_lat=${UI_LAT}"
done

echo ""
echo "======================================================"
echo "  SWEEP COMPLETE — summary at: ${SUMMARY_FILE}"
echo "======================================================"
cat "${SUMMARY_FILE}"
