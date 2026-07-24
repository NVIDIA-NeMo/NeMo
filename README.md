> **Note:** This branch provides resources for working with the [Nemotron Labs VoiceChat model on Hugging Face](https://huggingface.co/nvidia).

## Introduction

NVIDIA NemotronLabs VoiceChat is a 12B end-to-end, real-time speech full duplex (FD) model for conversational AI that jointly performs streaming speech understanding and speech generation [1]. Unlike traditional cascaded stacks (ASR → LLM → TTS), this model achieves full duplex, real-time, seamless voice interaction in one unified architecture, eliminating the need for multiple models or API handoffs, thus reducing end-to-end latency. It sets new benchmarks by bringing open, robust, and highly natural conversation capabilities. Moreover, NVIDIA NemotronLabs VoiceChat is the first open full-duplex model to support tool calling while maintaining a natural conversation flow during tool execution. For each tool, a specific “on-hold” message can be defined that will be spoken by the agent as soon as the LLM generates the text that will trigger the tool call and response.

The model operates on audio signals, which are encoded using a fast conformer module. The resulting audio tokens are inputted into a Nemotron Nano V2 9B LLM backbone to predict text tokens, which are fed to a TTS decoder [2] to predict audio codes for generating the agent's speech. A separate output channel is used to predict tool calling scripts. NemotronLabs VoiceChat offers an unprecedented trade-off between intelligence and latency in the space of open-source voice agents.

## Hardware Requirements

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

## Code Overview

- Training and model implementation: `nemo/collections/speechlm2/models/duplex_stt_model.py`
- Dataset loading and preprocessing: `nemo/collections/speechlm2/data/s2s_dataset.py`
- Default function-calling Jinja template: `examples/speechlm2/function_calling/template.jinja`

### Function-calling system prompt example

The default Jinja template appends the available tools and tool-call protocol to
the supplied system message. For example, the rendered prompt can look like:

```text
You are an AI voice assistant developed by NVIDIA. Your name is NVIDIA Voice Chat. Your job is to be helpful and harmless and have engaging conversations in English. Maintain a warm and friendly tone. Keep the dialogue open and ongoing. Be clear and direct, especially when answering yes or no questions and multiple-choice questions. Avoid long answers unless the user asks you to provide details or context. You must provide diverse responses and rephrase answers if the user asks the same question. DO NOT interrupt the user when they are speaking, let them finish their turn before answering.

When you receive a request, follow this decision process:
1. Does the request match one of your available tools below? If yes, you MUST call that tool - never answer it directly from your own knowledge, even if you think you know the answer.
2. Is it a general knowledge question (history, science, geography, math, facts, etc.)? If yes, answer directly from your own knowledge - do not call any tool.
3. Does it require an external action or live data that none of your tools cover (e.g. ordering food, sending email)? If yes, politely say you don't have that capability.

NEVER say "I don't have a tool for that" for general knowledge questions you can answer yourself.

DO NOT use any tools when not needed to answer the user's requests, under no circumstance.

You are an expert across history, geography, science, math, literature, biographies, languages, recipes, programming, current affairs, and general knowledge. When the user asks about any of these, answer directly and conversationally from your own knowledge — no <TOOLCALL>.

Call a tool ONLY when the user's request matches one of the tools listed in <AVAILABLE_TOOLS> below. For every other request, do not call any tool — just answer from your knowledge. Never invent or call a tool name that is not literally in <AVAILABLE_TOOLS>.

Tool-call arguments must be values the user spoke. If a required argument is missing, ask the user; never guess.

You can use the following tools to assist the user if required:
<AVAILABLE_TOOLS>[{"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city name as the user spoke it"}}, "required": ["city"]}}, {"name": "get_stock_price", "description": "Get the current stock price for a given ticker symbol", "parameters": {"type": "object", "properties": {"symbol": {"type": "string", "description": "The stock ticker symbol as stated by the user"}}, "required": ["symbol"]}}, {"name": "get_top_news", "description": "Get today's top one news headline from Google News", "parameters": {"type": "object", "properties": {"topic": {"type": "string", "description": "Optional topic: business, technology, science, health, sports, entertainment"}}, "required": []}}]</AVAILABLE_TOOLS>

If you decide to call any tool(s), use the following format:
<TOOLCALL>[{"name": "tool_name1", "arguments": "tool_args1"}, {"name": "tool_name2", "arguments": "tool_args2"}]</TOOLCALL>

The user will execute tool-calls and return responses from tool(s) in this format:
<TOOL_RESPONSE>[{"tool_response1"}, {"tool_response2"}]</TOOL_RESPONSE>

Based on the tool responses, you can call additional tools if needed, correct tool calls if any errors are found, or just respond to the user.
```

## Combine STT, TTS, and RNNT Checkpoints

`examples/speechlm2/combine_ckpt.sh` creates a single Hugging Face-format
checkpoint in three stages:

1. Convert the distributed STT checkpoint to Hugging Face format.
2. Combine STT and TTS.
3. Add the RNNT decoder, joint network, and tokenizer.

Required inputs:

- A distributed STT checkpoint (`step-<N>.ckpt`) and its `exp_config.yaml`.
- A TTS `.ckpt`.
- An RNNT `.nemo`. Its encoder config is used during STT conversion, and its decoder/joint weights are used in the final merge.
- A reference speaker WAV.
- The Docker image and a saved tar archive used as a backup.

The default Hydra configuration is
`examples/speechlm2/conf/nemotron_voicechat_nano9b.yaml`.

`--docker-image-tar` specifies the backup archive. The script loads it only
when the image named by `--docker-image` is not already available locally.

Run the following command on the host. Replace the example paths, tag, and
step with values for your checkpoints:

```bash
export NEMO_DIR=/path/to/Speech
export WORKSPACE=/path/to/checkpoint-workspace

bash "$NEMO_DIR/examples/speechlm2/combine_ckpt.sh" \
  --stt-fsdp-ckpt "$WORKSPACE/checkpoints/stt/step-10000.ckpt" \
  --stt-ckpt-config "$WORKSPACE/checkpoints/stt/exp_config.yaml" \
  --tts-ckpt "$WORKSPACE/checkpoints/tts/tts.ckpt" \
  --rnnt-nemo "$WORKSPACE/checkpoints/rnnt/model.nemo" \
  --speaker-wav "$WORKSPACE/speaker/speaker.wav" \
  --speaker-name my_speaker \
  --output-root "$WORKSPACE/output" \
  --docker-image-tar /path/to/voicechat-v1.1.tar \
  --docker-image voicechat:v1.1 \
  --tag my_exp \
  --step 10000 \
  --cache "$WORKSPACE/cache" \
  --results-root "$WORKSPACE/results" \
  --docker-vols "-v $NEMO_DIR:$NEMO_DIR -v $WORKSPACE:$WORKSPACE" \
  --gpu 0
```

The script writes intermediate and final checkpoints under:

```text
<output-root>/<tag>/s2s/
<output-root>/<tag>/s2s_rnnt/
```

Use a local wrapper outside the repository if a machine needs fixed paths.
Keeping such a wrapper outside the Speech source tree avoids committing
cluster-specific paths.
