## Introduction

NVIDIA NemotronLabs VoiceChat is a 12B end-to-end, real-time speech full duplex (FD) model for conversational AI that jointly performs streaming speech understanding and speech generation [1]. Unlike traditional cascaded stacks (ASR → LLM → TTS), this model achieves full duplex, real-time, seamless voice interaction in one unified architecture, eliminating the need for multiple models or API handoffs, thus reducing end-to-end latency. It sets new benchmarks by bringing open, robust, and highly natural conversation capabilities. Moreover, NVIDIA NemotronLabs VoiceChat is the first open full-duplex model to support tool calling while maintaining a natural conversation flow during tool execution. For each tool, a specific “on-hold” message can be defined that will be spoken by the agent as soon as the LLM generates the text that will trigger the tool call and response.

The model operates on audio signals, which are encoded using a fast conformer module. The resulting audio tokens are inputted into a Nemotron Nano V2 9B LLM backbone to predict text tokens, which are fed to a TTS decoder [2] to predict audio codes for generating the agent's speech. A separate output channel is used to predict tool calling scripts. NemotronLabs VoiceChat offers an unprecedented trade-off between intelligence and latency in the space of open-source voice agents.

## Requirements

- NVIDIA GPU 

## Quickstart

### Offline Evaluation (HF Checkpoint)

Run offline speech-to-speech inference from a Hugging Face-format checkpoint. This requires an NVIDIA GPU, Docker, and the Speech source tree.

#### 1. Clone the Speech repository

`/path/to/Speech` in the commands below means the root of a local clone of
[`NVIDIA-NeMo/Speech`](https://github.com/NVIDIA-NeMo/Speech.git) on the
`nemotron-labs-voicechat` branch:

```bash
cd /path/to/parent-directory
git clone https://github.com/NVIDIA-NeMo/Speech.git
cd Speech
git switch nemotron-labs-voicechat

export NEMO_DIR="$(pwd)"
```

#### 2. Create the Docker image (once per machine)

Dockerfiles and patches live under `docker/voicechat/` in this repository.

```bash
cd "$NEMO_DIR/docker/voicechat"

docker build --no-cache -f Dockerfile.voicechat -t voicechat:v1.1 .
```

Optionally save the image as a portable tar (choose any output path):

```bash
docker save voicechat:v1.1 -o /path/to/voicechat-v1.1.tar
```

#### 3. Start an interactive GPU container (on the host)

If you opened a new shell, set `NEMO_DIR` to the absolute path of the cloned
Speech repository before starting the container:

```bash
export NEMO_DIR=/path/to/Speech
export NEMO_VOICECHAT_WORKSPACE=$HOME          # host dir mounted into the container
# export NEMO_VOICECHAT_USE_SUDO=1             # if your user is not in the docker group
# Optional: if the image is not already loaded locally, point to your saved tar
# export NEMO_VOICECHAT_CONTAINER_TAR=/path/to/voicechat-v1.1.tar

bash "$NEMO_DIR/examples/speechlm2/create_interactive_node_local_docker.sh"
```

#### 4. Run offline inference

```bash
# General
WAV="$NEMO_DIR/examples/speechlm2/sample_audio/sample_general.wav"

python3 "$NEMO_DIR/examples/speechlm2/offline_voicechat_infer.py" \
  --checkpoint "$CHECKPOINT" --wav "$WAV" \
  --output-dir "$OUTPUT_DIR" --code-dir "$NEMO_DIR"

# Function calling
WAV="$NEMO_DIR/examples/speechlm2/sample_audio/sample_fc.wav"

python3 "$NEMO_DIR/examples/speechlm2/offline_voicechat_fc_infer.py" \
  --checkpoint "$CHECKPOINT" --wav "$WAV" \
  --api-response-json "$API_RESPONSE_JSON" --output-dir "$OUTPUT_DIR" \
  --code-dir "$NEMO_DIR"
```

### NIM Deployment 

