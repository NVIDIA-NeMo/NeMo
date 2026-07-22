#!/usr/bin/env bash
# Start an interactive GPU container for offline Nemotron VoiceChat evaluation.
#
# Environment (all optional):
#   NEMO_FC_DIR                  NeMo_fc repo root (default: parent of examples/speechlm2)
#   NEMO_VOICECHAT_IMAGE         Docker image tag (default: nemo_containers:triton25.05_s2svllm26.02.12)
#   NEMO_VOICECHAT_CONTAINER_TAR Path to saved image .tar (used if image tag is missing)
#   NEMO_VOICECHAT_WORKSPACE     Host dir mounted read-write (default: $HOME)
#   NEMO_VOICECHAT_USE_SUDO      Set to 1 to prefix docker with sudo
#   NEMO_VOICECHAT_EXTRA_MOUNTS  Extra docker -v flags (e.g. "-v /data:/data")
#   NEMO_VOICECHAT_PORTS         Extra docker -p flags

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${NEMO_FC_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
IMAGE="${NEMO_VOICECHAT_IMAGE:-nemo_containers:triton25.05_s2svllm26.02.12}"
CONTAINER_TAR="${NEMO_VOICECHAT_CONTAINER_TAR:-}"
WORKSPACE="${NEMO_VOICECHAT_WORKSPACE:-${HOME}}"
EXTRA_MOUNTS="${NEMO_VOICECHAT_EXTRA_MOUNTS:-}"
PORTS="${NEMO_VOICECHAT_PORTS:-}"

if [[ "${NEMO_VOICECHAT_USE_SUDO:-0}" == "1" ]]; then
    DOCKER=(sudo docker)
else
    DOCKER=(docker)
fi

MOUNTS="-v ${WORKSPACE}:${WORKSPACE} -v ${CODE_DIR}:${CODE_DIR} ${EXTRA_MOUNTS}"

if ! "${DOCKER[@]}" image inspect "$IMAGE" >/dev/null 2>&1; then
    EXISTING=$("${DOCKER[@]}" images --format '{{.Repository}}:{{.Tag}}' | grep 'triton25.05_s2svllm26.02.12' | head -1 || true)
    if [[ -n "$EXISTING" ]]; then
        "${DOCKER[@]}" tag "$EXISTING" "$IMAGE"
    elif [[ -n "$CONTAINER_TAR" && -f "$CONTAINER_TAR" ]]; then
        LOADED=$("${DOCKER[@]}" load -i "$CONTAINER_TAR" | awk '/Loaded image:/ {print $3}')
        "${DOCKER[@]}" tag "$LOADED" "$IMAGE"
    else
        echo "Docker image not found: ${IMAGE}" >&2
        echo "Set NEMO_VOICECHAT_CONTAINER_TAR to a .tar file or pull/load the image first." >&2
        exit 1
    fi
fi

exec "${DOCKER[@]}" run \
    --gpus all \
    --rm -it \
    --shm-size=8g \
    --privileged \
    --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    ${MOUNTS} ${PORTS} \
    --entrypoint /bin/bash \
    "$IMAGE"
