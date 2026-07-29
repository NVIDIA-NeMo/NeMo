# Nemotron Voicechat

The NVIDIA Nemotron Voicechat microservice enables real-time voice conversations. It accepts spoken audio as input and returns synthesized speech as output in a single end-to-end pipeline, without requiring separate ASR, LLM, and TTS components.

The microservice uses a bidirectional WebSocket interface to stream audio in and stream synthesized speech out with low latency. It packages the complete model with the full NVIDIA inference stack (CUDA, Triton, vLLM) into a single container — no orchestration of multiple containers is required.

## Prerequisites

- Completed [prerequisites](prerequisites.md).
- To use a custom NeMo checkpoint instead of the downloaded model, first [generate a Triton model repository](generate-model-repo.md).

## Deploy the Container

Create a local cache directory and launch the container. The cache avoids repeated model downloads on subsequent runs.

```bash
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p $LOCAL_NIM_CACHE
chmod 777 $LOCAL_NIM_CACHE

docker run -it --rm --name=nemotron-voicechat \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --shm-size=8GB \
  -e NIM_HTTP_API_PORT=9000 \
  -p 9000:9000 \
  -v $LOCAL_NIM_CACHE:/opt/nim/.cache \
  nvcr.io/nvidia/nemotron-voicechat:latest
```

On first startup, the container downloads the model, which can take up to 30 minutes depending on network speed. Subsequent runs load the model from the cache.

### Verify Readiness

Wait for the container to finish model setup, then check the health endpoint.

```bash
curl -X 'GET' 'http://localhost:9000/v1/health/ready'
```

Expected response:

```json
{"object":"health.response","message":"ready","status":"ready"}
```

## Run a Voice Conversation

The Nemotron Voicechat container uses a bidirectional WebSocket connection for real-time voice conversations. Audio is streamed to the server and synthesized speech is streamed back.

### Copy the Client Script

Copy the client script from the running container:

```bash
docker cp nemotron-voicechat:/s2s/nemotron-voicechat-client.py .
```

### Install Dependencies

```bash
pip install websockets soundfile numpy
```

`pyaudio` is required for microphone input and audio playback. Skip it if using `--input-file` with `--no-playback`.

#### Ubuntu/Debian

```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

#### macOS

```bash
brew install portaudio
pip install pyaudio
```

### Real-Time Conversation (WebSocket)

The client streams audio to the server at `ws://<host>:<port>/v1/realtime` and plays back or saves the returned speech. Audio playback is enabled by default.

Stream from a microphone and play on speakers:

```bash
python3 nemotron-voicechat-client.py --server ws://localhost:9000
```

> **Note:** For best results, run the client in a quiet environment. The model is sensitive to background noise and may produce degraded output in noisy or reverberant conditions.

Stream from and to a file:

Use a speech recording file, with 16-bit, Mono, 16KHz format as the input. Make sure to append silence (~20 seconds) after the speech when preparing `sample_speech.wav`, to get the correct response. Nemotron Voicechat is a full duplex model, it generates output as long as there is input.

```bash
python3 nemotron-voicechat-client.py --server ws://localhost:9000 \
  --input-file sample_speech.wav \
  --audio-output output.wav \
  --no-playback
```

> **Note:** The Nemotron Voicechat container supports real-time streaming mode only. Offline (batch) synthesis is not supported.


## Function Calling

The Nemotron Voicechat container supports function calling (tool use), allowing the model to pause its spoken response, request an external function result, and seamlessly resume after receiving the result.

### How It Works

```
User speaks
    │
    ▼
Model responds (audio + text)
    │
    ▼  model signals tool needed
    │
Server sends response.function_call_arguments.done → client
    │
    ▼
Client executes function, sends conversation.item.create (function_call_output) → server
    │
    ▼
Model resumes with final answer
```

### Providing Tools

Pass tool definitions to the client with `--tools` as an inline JSON string or a path to a JSON file.

Inline JSON:

```bash
python3 nemotron-voicechat-client.py --server ws://localhost:9000 \
  --input-file sample_speech.wav \
  --tools '[{"name":"get_current_time","description":"Returns the current local time","parameters":{"type":"object","properties":{}}}]'
```

From a file:

```bash
python3 nemotron-voicechat-client.py --server ws://localhost:9000 \
  --input-file sample_speech.wav \
  --tools tools.json
```

### Tool Definition Format

Tools follow the OpenAI Realtime API function tool specification:

```json
[
  {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "ack_messages": ["Sure, let me check the weather for you."],
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "City name"
        }
      },
      "required": ["city"]
    }
  }
]
```

| Field | Required | Description |
| ----- | -------- | ----------- |
| `name` | Yes | Non-empty string identifying the tool. |
| `description` | No | Human-readable description; recommended for model accuracy. |
| `ack_messages` | No (optional) | Array of one or more non-empty strings. The model speaks one of these while waiting for the tool result, keeping the conversation flowing naturally. Example: `["Sure, let me check that.", "One moment please."]` |
| `parameters` | No | JSON Schema object describing the tool's arguments (`type` must be `"object"`). |

### Built-in Demo Tools

The client script includes built-in handlers for the following demo tools. When you pass their definitions via `--tools`, the model can invoke them automatically and the client responds with real (or plausible dummy) results.

| Tool | Description |
| ---- | ----------- |
| `get_current_time` | Returns the current local time |
| `get_current_datetime` | Returns current date, time, and day of week |
| `calculate_bmi` | Calculates BMI given weight (kg) and height (m) |
| `convert_currency` | Converts an amount between currencies using static rates |
| `get_news_headlines` | Returns sample news headlines |

Example — enable all built-in demo tools:

```bash
python3 nemotron-voicechat-client.py --server ws://localhost:9000 \
  --input-file sample_speech.wav \
  --tools '[
    {"name":"get_current_time","description":"Get the current time","ack_messages":["Sure, let me check the time for you."],"parameters":{"type":"object","properties":{}}},
    {"name":"get_current_datetime","description":"Get current date and time","ack_messages":["One moment, fetching the date and time."],"parameters":{"type":"object","properties":{}}},
    {"name":"calculate_bmi","description":"Calculate BMI","ack_messages":["Let me calculate that for you."],"parameters":{"type":"object","properties":{"weight":{"type":"number"},"height":{"type":"number"}},"required":["weight","height"]}},
    {"name":"convert_currency","description":"Convert currency","ack_messages":["Sure, let me convert that for you."],"parameters":{"type":"object","properties":{"amount":{"type":"number"},"from_currency":{"type":"string"},"to_currency":{"type":"string"}},"required":["amount","from_currency","to_currency"]}},
    {"name":"get_news_headlines","description":"Get news headlines","ack_messages":["Let me fetch the latest headlines."],"parameters":{"type":"object","properties":{}}}
  ]'
```

### WebSocket Events

| Event | Direction | Description |
| ----- | --------- | ----------- |
| `session.update` (with `tools` array) | client → server | Registers tools for the session |
| `response.function_call_arguments.done` | server → client | Signals a complete tool call with name, call ID, and JSON arguments |
| `conversation.item.create` (with `function_call_output` item) | client → server | Returns the function result to the server |

The `response.function_call_arguments.done` event carries:

```json
{
  "type": "response.function_call_arguments.done",
  "call_id": "call_<id>",
  "name": "get_current_time",
  "arguments": "{}"
}
```

The client returns the result with:

```json
{
  "type": "conversation.item.create",
  "item": {
    "type": "function_call_output",
    "call_id": "call_<id>",
    "output": "{\"time\": \"14:32:00\"}"
  }
}
```

### Saving Function Call Logs

Use `--function-text-output` to save all tool invocations and their results to a JSONL file (one entry per call):

```bash
python3 nemotron-voicechat-client.py --server localhost:9000 \
  --input-file sample_speech.wav \
  --tools tools.json \
  --function-text-output function_calls.jsonl
```


## Client Parameters Reference

| Parameter | Description | Default |
| --------- | ----------- | ------- |
| `--server` | Server address as `host:port` or full URI `ws://host[:port]`. | required |
| `--input-file` | Path to an audio file to stream (WAV). If omitted, reads from the default microphone. | -- |
| `--audio-output` | Output WAV file path for the received audio. | auto-generated |
| `--user-text-output` | Output file for user ASR transcripts. | auto-generated |
| `--agent-text-output` | Output file for agent response text. | auto-generated |
| `--conversation-output` | Output file for the full conversation log in JSONL format. | auto-generated |
| `--format` | Audio format for input and output. `pcm16` or `opus`. Opus is experimental. | `pcm16` |
| `--instructions` | Instructions for the agent (inline string or path to a text file). | -- |
| `--tools` | Tool definitions for function calling (JSON array as inline string or path to a JSON file). | -- |
| `--function-text-output` | Output file for tool/function call log in JSONL format. | `function_calls.jsonl` |
| `--no-playback` | Disable audio playback of server responses. | playback enabled |
| `--num-streams` | Number of concurrent streams to launch for load testing. Requires `--input-file`. | `1` |
| `--output-dir` | Directory for per-stream output files. Used with `--num-streams` > 1. | -- |
| `-v`, `--verbose` | Enable verbose logging. | `false` |

## Next Steps

- [API Reference](api-reference.md): WebSocket and HTTP API reference.
- [Generate Model Repository](generate-model-repo.md): Build a Triton model repository from a local NeMo checkpoint.

## Troubleshooting

### Container Startup Takes Longer Than 30 Minutes

**Cause:** First-run model download.

**Solution:** The cache directory mounted at `/opt/nim/.cache` persists the model across restarts. Ensure the volume mount is present in your `docker run` command.

### GPU Out of Memory (OOM)

**Cause:** The model requires approximately 66 GB of GPU memory.

**Solution:** Free other GPU processes or select a different device with `--gpus '"device=1"'`.

### Health Check Returns 503

**Cause:** Triton is not yet ready during model loading.

**Solution:**
- Watch logs with `docker logs -f <container>`.
- Poll `curl http://localhost:9000/v1/health/ready` until it returns `ready`.
- Ensure `--shm-size=8GB` is set on the `docker run` command.

---
Next: [Generate Model Repository](generate-model-repo.md)
