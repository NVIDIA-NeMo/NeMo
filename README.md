> **Note:** This branch provides resources for working with the [Nemotron Labs VoiceChat model on Hugging Face](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-12B).

## Introduction

NVIDIA NemotronLabs VoiceChat is a 12B end-to-end, real-time speech full duplex (FD) model for conversational AI that jointly performs streaming speech understanding and speech generation [1]. Unlike traditional cascaded stacks (ASR → LLM → TTS), this model achieves full duplex, real-time, seamless voice interaction in one unified architecture, eliminating the need for multiple models or API handoffs, thus reducing end-to-end latency. It sets new benchmarks by bringing open, robust, and highly natural conversation capabilities. Moreover, NVIDIA NemotronLabs VoiceChat is the first open full-duplex model to support tool calling while maintaining a natural conversation flow during tool execution. For each tool, a specific “on-hold” message can be defined that will be spoken by the agent as soon as the LLM generates the text that will trigger the tool call and response.

The model operates on audio signals, which are encoded using a fast conformer module. The resulting audio tokens are inputted into a Nemotron Nano V2 9B LLM backbone to predict text tokens, which are fed to a TTS decoder [2] to predict audio codes for generating the agent's speech. A separate output channel is used to predict tool calling scripts. NemotronLabs VoiceChat offers an unprecedented trade-off between intelligence and latency in the space of open-source voice agents.

## Hardware Requirements

- NVIDIA GPU with at least 80 GB of memory

## Quickstart

This guide explains how to test the NVIDIA Nemotron Labs VoiceChat model using either of the following approaches:

- Load the [Hugging Face (HF) checkpoint](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-12B) for quick, non-interactive testing with offline batch inference using an Anaconda environment.
- Use an optimized NVIDIA inference container for interactive audio testing with the same HF checkpoint.

The available code can also be used for training. The resulting checkpoint can then be converted and used for inference as described above. Details on how to perform this conversion are provided in [Combine STT, TTS, and RNNT Checkpoints](#combine-stt-tts-and-rnnt-checkpoints).

### Offline Evaluation (HF Checkpoint)

Run offline speech-to-speech inference from a Hugging Face-format checkpoint.
This requires an NVIDIA GPU, Anaconda/Miniconda, and the Speech source tree.

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

#### 2. Create the Anaconda environment (once per machine)

```bash
conda create -y -n voicechat python=3.12
conda activate voicechat
# torch 2.10 is deliberate: it is the newest release with prebuilt mamba-ssm /
# causal-conv1d wheels AND a matching torchaudio. Newer torch = 20+ min of nvcc.
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
pip install "nemo_toolkit[all]==2.6.2"
# megatron-core 0.18.2 asserts nvidia-resiliency-ext>=0.6.0 while NeMo pins
# 0.5.0, which crashes on import. It is training-only, so remove it.
pip uninstall -y nvidia-resiliency-ext
# torchcodec 0.10 pairs with torch 2.10 (torchaudio 2.10 delegates wav I/O to
# it). It declares no torch dependency, so an unpinned install breaks silently.
pip install transformers==4.56.0 tokenizers==0.22.0 lhotse==1.32.2 \
            huggingface-hub==0.34.4 hf-xet==1.1.9 torchcodec==0.10.0 \
            torch_audiomentations jinja2
# --no-deps is required: mamba-ssm 2.3.2 hard-requires tilelang + quack-kernels
# (Mamba-3 only, unused here) and resolving them upgrades torch, which orphans
# the prebuilt .so ("undefined symbol: ...materialize_cow_storage").
pip install ninja packaging wheel einops
pip install --no-build-isolation --no-deps causal-conv1d==1.6.2.post1 mamba-ssm==2.3.2.post1
```

Verify the install (expect: `2.10.0+cu128 True True <your GPU>`):

```bash
python -c "import torch, torchcodec; from transformers.utils.import_utils import is_mamba_2_ssm_available as m, is_causal_conv1d_available as c; print(torch.__version__, m(), c(), torch.cuda.get_device_name(0))"
```

#### 3. Download the checkpoint

```bash
export CHECKPOINT="$HOME/NVIDIA-NemotronLabs-VoiceChat-12B"
export OUTPUT_DIR="$HOME/voicechat_output"

hf download nvidia/NVIDIA-NemotronLabs-VoiceChat-12B --local-dir "$CHECKPOINT"
```

#### 4. Run offline inference

Activate the environment if needed (`conda activate voicechat`), then:

The environment provides NeMo's dependencies via the installed `nemo_toolkit`,
but the VoiceChat model code lives in this repository. Prepend `NEMO_DIR` to
`PYTHONPATH` in every new shell so this source tree is used instead of an older
installed SpeechLM package.

```bash
export PYTHONPATH="$NEMO_DIR:${PYTHONPATH:-}"

# General
WAV="$NEMO_DIR/examples/speechlm2/sample_audio/sample_general.wav"

python "$NEMO_DIR/examples/speechlm2/offline_voicechat_infer.py" \
  --checkpoint "$CHECKPOINT" --wav "$WAV" \
  --output-dir "$OUTPUT_DIR"
```

For function calling, `--api-response-json` supplies the tool response injected
on the second pass. Its `tool_name` must match an available tool, and `response`
must be ASCII-only and TTS-friendly:

```bash
# Function calling
WAV="$NEMO_DIR/examples/speechlm2/sample_audio/sample_fc.wav"
API_RESPONSE_JSON="$NEMO_DIR/examples/speechlm2/function_calling/random_number_response.json"

python "$NEMO_DIR/examples/speechlm2/offline_voicechat_fc_infer.py" \
  --checkpoint "$CHECKPOINT" --wav "$WAV" \
  --api-response-json "$API_RESPONSE_JSON" --output-dir "$OUTPUT_DIR"
```

### Optimized NVIDIA inference container for interactive streaming deployment

The [Nemotron Voicechat container](docs/source/nemotron-voicechat/deploy.md) packages the complete model with the NVIDIA inference stack (CUDA, Triton, vLLM) into a single container. It exposes a bidirectional WebSocket interface for real-time, low-latency voice conversations and supports function calling.

- [Prerequisites](docs/source/nemotron-voicechat/prerequisites.md) — hardware, software, and driver requirements
- [Deploy and Run](docs/source/nemotron-voicechat/deploy.md) — launch the container and run voice conversations
- [API Reference](docs/source/nemotron-voicechat/api-reference.md) — WebSocket and HTTP API reference


## Code Overview

- Training and model implementation: `nemo/collections/speechlm2/models/duplex_stt_model.py`
- Dataset loading and preprocessing: `nemo/collections/speechlm2/data/s2s_dataset.py`
- Default function-calling Jinja template: `examples/speechlm2/function_calling/template.jinja`

### Function-calling system prompt example

The default Jinja template appends the available tools and tool-call protocol to
the supplied system message. See
`examples/speechlm2/offline_voicechat_fc_infer.py` for the default
function-calling system prompt and prompt construction logic.

System prompts and API/tool responses must be ASCII-only. Avoid Unicode
punctuation and symbols (for example em dashes, en dashes, degree symbols, and
emoji). Convert tool responses into concise, TTS-friendly ASCII sentences before
passing them to the model.

For example, the rendered prompt can look like:

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

`examples/speechlm2/combine_ckpt_conda.sh` creates a single Hugging Face-format
checkpoint using a local Anaconda/Miniconda environment in three stages:

1. Convert the distributed STT checkpoint to Hugging Face format.
2. Combine STT and TTS.
3. Add the RNNT decoder, joint network, and tokenizer.

Required inputs:

- A distributed STT checkpoint (`step-<N>.ckpt`) and its `exp_config.yaml`.
- A TTS `.ckpt`.
- An RNNT `.nemo`. Its encoder config is used during STT conversion, and its decoder/joint weights are used in the final merge.
- A reference speaker WAV.
- A conda environment with the VoiceChat dependencies installed (default name: `voicechat`).

The default Hydra configuration is
`examples/speechlm2/conf/nemotron_voicechat_nano9b.yaml`.

Run the following command. Replace the example paths, tag, step, and conda
settings with values for your checkpoints:

```bash
export NEMO_DIR=/path/to/Speech
export WORKSPACE=/path/to/checkpoint-workspace
export STEP=<N>   # training step of the STT checkpoint (matches step-<N>.ckpt)

bash "$NEMO_DIR/examples/speechlm2/combine_ckpt_conda.sh" \
  --stt-fsdp-ckpt "$WORKSPACE/checkpoints/stt/step-${STEP}.ckpt" \
  --stt-ckpt-config "$WORKSPACE/checkpoints/stt/exp_config.yaml" \
  --tts-ckpt "$WORKSPACE/checkpoints/tts/tts.ckpt" \
  --rnnt-nemo "$WORKSPACE/checkpoints/rnnt/model.nemo" \
  --speaker-wav "$WORKSPACE/speaker/speaker.wav" \
  --speaker-name my_speaker \
  --output-root "$WORKSPACE/output" \
  --tag my_exp \
  --step "$STEP" \
  --cache "$WORKSPACE/cache" \
  --results-root "$WORKSPACE/results" \
  --conda-env voicechat \
  --gpu 0
```

Override the conda install location if needed with `--conda-root` or
`CONDA_ROOT`.

The script writes intermediate and final checkpoints under:

```text
<output-root>/<tag>/s2s/
<output-root>/<tag>/s2s_rnnt/
```

## Known Limitations

NemotronLabs VoiceChat is trained with audio context windows of up to two minutes. Conversational context beyond this window may not be retained reliably.

NemotronLabs VoiceChat has been optimized to balance general knowledge and natural conversation. Consequently, it may not perform as well as its LLM backbone, NVIDIA Nemotron Nano v2, in terms of knowledge, instruction-following capabilities, and safety.

The model may make reasoning errors or provide incorrect or incomplete information. NemotronLabs VoiceChat was not explicitly trained for reasoning or alignment, so its performance on tasks requiring multi-step reasoning, arithmetic, or safety-aligned behavior may be limited.

The model may not yet be able to handle user backchanneling systematically.

For tool calling, we recommend using no more than five tools per session, as additional tools may degrade performance. The model cannot yet reliably call multiple tools simultaneously. Long tool responses may delay the agent's speech; on-hold messages can help mitigate these delays. Users cannot interrupt the agent while a tool call is being executed. In mixed conversations that combine general chat and tool requests, the model may answer from its own knowledge instead of calling the appropriate tool.

To ensure reliable speech output, system prompts must contain only ASCII characters. Avoid Unicode punctuation and symbols, including but not limited to em dashes (U+2014), en dashes (U+2013), degree symbols, and emoji. For example, write "72 degrees Fahrenheit" instead of using a degree symbol.

External API and tool responses must be converted into concise, ASCII-only, TTS-friendly sentences before being passed to the model.

Finally, the model is not suitable for noisy or highly reverberant environments, especially those with background speech.
