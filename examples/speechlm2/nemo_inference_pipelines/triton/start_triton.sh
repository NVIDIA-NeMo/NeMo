#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Start Triton Inference Server for S2S voicechat model.
#
# Shares the same s2s_streaming.yaml config used by s2s_streaming_infer.py.
# Fields marked ??? in the YAML are resolved from environment variables below.
#
# Usage:
#   S2S_MODEL_PATH=/path/to/eartts_ckpt \
#   S2S_LLM_CHECKPOINT_PATH=/path/to/llm_ckpt \
#   S2S_SPEAKER_REFERENCE=/path/to/speaker.wav \
#   ./start_triton.sh
#
# Environment variables (required):
#   S2S_MODEL_PATH              - Path to the EarTTS / S2S checkpoint
#   S2S_LLM_CHECKPOINT_PATH     - Path to the LLM checkpoint
#   S2S_SPEAKER_REFERENCE       - Path to a speaker reference .wav file
#
# Environment variables (optional):
#   S2S_ENGINE_TYPE             - Engine type (default: native)
#   S2S_SYSTEM_PROMPT           - LLM system prompt text (default: none)
#   S2S_TTS_SYSTEM_PROMPT       - TTS system prompt, (default: none)
#   S2S_CHUNK_SIZE_IN_SECS      - Chunk size in seconds, multiple of 0.08 (default: 0.08)
#   S2S_BUFFER_SIZE_IN_SECS     - Audio buffer size in seconds (default: 5.6)
#   S2S_USE_CODEC_CACHE         - "true"/"false": incremental codec decode (default: true)
#   S2S_TRITON_CONFIG_PATH      - Override the YAML config file path
#   MODEL_REPO_DIR              - Override the Triton model repository path

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# All variables below are exported so they are visible to the Triton Python
# backend (infer_streaming.py reads them via os.environ).

# ========================
# Model paths (required)
# ========================
export S2S_MODEL_PATH="${S2S_MODEL_PATH:?Please set S2S_MODEL_PATH to the EarTTS / S2S checkpoint path}"
export S2S_LLM_CHECKPOINT_PATH="${S2S_LLM_CHECKPOINT_PATH:?Please set S2S_LLM_CHECKPOINT_PATH to the LLM checkpoint path}"
export S2S_SPEAKER_REFERENCE="${S2S_SPEAKER_REFERENCE:?Please set S2S_SPEAKER_REFERENCE to a speaker reference .wav file}"

# ========================
# Optional overrides
# ========================
export S2S_ENGINE_TYPE="${S2S_ENGINE_TYPE:-native}"
export S2S_SYSTEM_PROMPT="${S2S_SYSTEM_PROMPT:-}"
export S2S_TTS_SYSTEM_PROMPT="${S2S_TTS_SYSTEM_PROMPT:-}"
export S2S_CHUNK_SIZE_IN_SECS="${S2S_CHUNK_SIZE_IN_SECS:-0.08}"
export S2S_BUFFER_SIZE_IN_SECS="${S2S_BUFFER_SIZE_IN_SECS:-5.6}"
export S2S_USE_CODEC_CACHE="${S2S_USE_CODEC_CACHE:-true}"
export S2S_TRITON_CONFIG_PATH="${S2S_TRITON_CONFIG_PATH:-${SCRIPT_DIR}/../conf/s2s_streaming.yaml}"
export MODEL_REPO_DIR="${MODEL_REPO_DIR:-${SCRIPT_DIR}/model_repo_s2s}"


echo "=== S2S Triton Server ==="
echo "  S2S_MODEL_PATH:          ${S2S_MODEL_PATH}"
echo "  S2S_LLM_CHECKPOINT_PATH: ${S2S_LLM_CHECKPOINT_PATH}"
echo "  S2S_SPEAKER_REFERENCE:   ${S2S_SPEAKER_REFERENCE}"
echo "  S2S_ENGINE_TYPE:         ${S2S_ENGINE_TYPE}"
echo "  S2S_CHUNK_SIZE_IN_SECS:  ${S2S_CHUNK_SIZE_IN_SECS}"
echo "  S2S_BUFFER_SIZE_IN_SECS: ${S2S_BUFFER_SIZE_IN_SECS}"
echo "  S2S_USE_CODEC_CACHE:     ${S2S_USE_CODEC_CACHE}"
echo "  S2S_SYSTEM_PROMPT:       ${S2S_SYSTEM_PROMPT:-<not set>}"
echo "  S2S_TTS_SYSTEM_PROMPT:   ${S2S_TTS_SYSTEM_PROMPT:-<not set>}"
echo "  MODEL_REPO_DIR:          ${MODEL_REPO_DIR}"
echo "  S2S_TRITON_CONFIG_PATH:  ${S2S_TRITON_CONFIG_PATH}"
echo "========================="

TRITON_BIN="${TRITON_BIN:-/opt/tritonserver/bin/tritonserver}"
if [ ! -x "${TRITON_BIN}" ]; then
    echo "ERROR: Triton server not found at ${TRITON_BIN}"
    echo "       Are you running inside a Triton container?"
    exit 1
fi

"${TRITON_BIN}" --model-repository="${MODEL_REPO_DIR}"
