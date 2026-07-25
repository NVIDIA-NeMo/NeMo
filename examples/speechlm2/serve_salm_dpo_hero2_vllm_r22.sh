#!/usr/bin/env bash
# Serve a tracked Hero2 DPO export with the established r22 vLLM NeMo source.
# The source is installed directly and non-editably: this script never copies,
# rewrites, overlays, or patches either source tree or the model directory.
set -euo pipefail

: "${R22_NEMO_SOURCE:?R22_NEMO_SOURCE must point to the readable r22 NeMo source}"
: "${R22_TRANSFORMER_ENCODER_SHA256:?R22_TRANSFORMER_ENCODER_SHA256 must pin r22 TransformerEncoder}"

transformer="$R22_NEMO_SOURCE/nemo/collections/asr/modules/transformer_encoder.py"
test -r "$transformer"
test "$(sha256sum "$transformer" | awk '{print $1}')" = "$R22_TRANSFORMER_ENCODER_SHA256"
test -r "$R22_NEMO_SOURCE/setup.py"
test -r "$R22_NEMO_SOURCE/pyproject.toml"

python3 -m pip install --no-deps "$R22_NEMO_SOURCE"
export VLLM_PLUGINS=nemo_speechlm
export VLLM_NO_USAGE_STATS=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_jit_cache}"
exec python3 -m nemo_skills.inference.server.serve_vllm "$@"
