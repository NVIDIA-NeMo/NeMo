# API Reference

The Nemotron Voicechat container exposes a WebSocket-based real-time audio API compatible with the OpenAI Realtime API protocol. Clients stream raw audio to the server and receive synthesized speech and transcripts in return. The wire protocol uses JSON messages with base64-encoded audio payloads.

## HTTP Endpoints

### `GET /`

Service discovery. Returns basic metadata about the running server.

**Response**

```json
{
  "service": "nemotron-voicechat",
  "version": "0.2.0",
  "websocket": "/v1/realtime",
  "health": "/v1/realtime/health",
  "loopback_mode": false,
  "triton_url": "localhost:8000",
  "model_name": "nemotron-voicechat"
}
```

---

### `GET /v1/realtime/health`

Liveness and readiness check.

**Response (healthy)**

```json
{
  "status": "ok",
  "service": "nemotron-voicechat-websocket-server",
  "mode": "triton",
  "triton_status": "ready",
  "model_inference_stats": {
    "success_count": 1234,
    "fail_count": 0
  }
}
```

**Response (degraded)**

```json
{
  "status": "error",
  "service": "nemotron-voicechat-websocket-server",
  "mode": "triton",
  "triton_status": "error",
  "error": "<reason>"
}
```

**Status Codes**

| Code | Description |
|------|-------------|
| `200 OK` | Server is healthy and ready to accept connections. |
| `503 Service Unavailable` | Server is not ready to accept connections. |

---

## WebSocket Endpoint

Two paths are registered:

| Path | Notes |
|------|-------|
| `/v1/realtime` | Primary endpoint. |
| `/realtime` | Alias for OpenAI SDK and Pipecat compatibility. |

**Connection URL examples**

```
ws://localhost:9000/v1/realtime
wss://your-host/v1/realtime
```

Every message on the WebSocket is a JSON object with a `type` field and an `event_id` (UUID string).

---

## Session Lifecycle

```
Client                              Server
  │── connect ──────────────────────► │
  │◄─── session.created ─────────────│
  │── session.update ────────────────►│
  │◄─── session.updated ─────────────│
  │                                   │
  │── input_audio_buffer.append ─────►│  (repeats for each audio chunk)
  │                                   │
  │◄─── input_audio_buffer.speech_started ──────────────│
  │◄─── response.created ────────────│
  │◄─── response.output_item.added ──│
  │◄─── response.content_part.added ─│
  │◄─── response.output_audio.delta ─│  (repeats per output chunk)
  │◄─── response.output_audio_transcript.delta ─────────│
  │◄─── conversation.item.input_audio_transcription.delta ──────│
  │◄─── input_audio_buffer.speech_stopped ──────────────│
  │◄─── conversation.item.input_audio_transcription.completed ──│
  │◄─── response.output_audio_transcript.done ──────────│
  │◄─── response.output_audio.done ──│
  │◄─── response.content_part.done ──│
  │◄─── response.output_item.done ───│
  │◄─── response.done ───────────────│
  │                                   │
  │── session.close / disconnect ────►│
  │◄─── session.end ─────────────────│
```

---

## Audio Format

### Supported Formats

| Format | Structured Object |
|--------|-------------------|
| PCM 16-bit | `{"type": "audio/pcm", "rate": 24000}` |

### Audio Parameters

| Parameter | Value |
|-----------|-------|
| Client input sample rate | 24 kHz |
| Model input sample rate | 16 kHz (server resamples internally) |
| Model output sample rate | 22 050 Hz (server resamples to 24 kHz before sending) |
| Client output sample rate | 24 kHz |
| Channels | Mono |
| PCM encoding | 16-bit signed integer, little-endian |
| Recommended chunk duration | 80 ms |

---

## Client Events

Events sent from client to server.

### Summary

| Event Type | Description |
|------------|-------------|
| [`input_audio_buffer.append`](#inputaudiobufferappend) | Sends one chunk of audio to the server. |
| [`session.update`](#sessionupdate) | Configures the session audio format, system prompt, and tools. |
| [`conversation.item.create`](#conversationitemcreate) | Sends a function call result back to the server. |
| [`session.close`](#sessionclose) | Requests an orderly shutdown. |

---

### `input_audio_buffer.append` {#inputaudiobufferappend}

Sends one chunk of audio to the server. Chunks should be approximately 80 ms long (3 840 bytes at 24 kHz PCM16).

```json
{
  "type": "input_audio_buffer.append",
  "event_id": "<uuid>",
  "audio": "<base64-encoded audio>"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Must be `"input_audio_buffer.append"`. |
| `event_id` | string | Client-generated UUID for this event. |
| `audio` | string | Base64-encoded audio in the negotiated input format. |

---

### `session.update` {#sessionupdate}

Configures the session. Send once immediately after connection, before streaming audio. Unknown fields (for example, `turn_detection`, `voice`) are silently ignored.

```json
{
  "type": "session.update",
  "event_id": "<uuid>",
  "session": {
    "audio": {
      "input":  { "format": {"type": "audio/pcm", "rate": 24000} },
      "output": { "format": {"type": "audio/pcm", "rate": 24000} }
    },
    "instructions": "<system prompt string or null>",
    "tools": []
  }
}
```

| Field | Type | Values | Default |
|-------|------|--------|---------|
| `session.audio.input.format` | string or object | `"pcm16"`, `"opus"`, or structured object | `"pcm16"` |
| `session.audio.output.format` | string or object | `"pcm16"`, `"opus"`, or structured object | `"pcm16"` |
| `session.instructions` | string | Arbitrary string | Model's built-in system prompt |
| `session.tools` | array | Array of function tool objects | `[]` |

`instructions` is applied to the first inference call of the session only. `tools` registers function definitions for function calling; see [Function Calling](deploy.md#function-calling).

---

### `conversation.item.create` {#conversationitemcreate}

Sends a function call result back to the server after executing a tool requested by the model.

```json
{
  "type": "conversation.item.create",
  "item": {
    "type": "function_call_output",
    "call_id": "<call-id>",
    "output": "<result string>"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `item.type` | string | Must be `"function_call_output"`. |
| `item.call_id` | string | The `call_id` from the `response.function_call_arguments.done` event. |
| `item.output` | string | The function result as a plain string (may be JSON-serialized). |

The server will return an error with code `tools_not_set` if tools were not configured in `session.update`.

---

### `session.close` {#sessionclose}

Requests an orderly shutdown. The server sends a [`session.end`](#sessionend) message with session statistics and then closes the WebSocket connection.

```json
{
  "type": "session.close",
  "event_id": "<uuid>"
}
```

---

## Server Events

Events sent from server to client.

### Summary

| Event Type | Description |
|------------|-------------|
| [`session.created`](#sessioncreated) | Sent once immediately after the WebSocket is accepted. |
| [`session.updated`](#sessionupdated) | Acknowledgment of a `session.update`. |
| [`input_audio_buffer.speech_started`](#inputaudiobufferspeechstarted) | Emitted on the first ASR token of a user utterance. |
| [`input_audio_buffer.speech_stopped`](#inputaudiobufferspeechstopped) | Emitted when ASR detects end of speech. |
| [`response.created`](#responsecreated) | Emitted when the model produces its first audio output for a turn. |
| [`response.output_item.added`](#responseoutputitemadded) | Emitted at the start of a new response item. |
| [`response.content_part.added`](#responsecontentpartadded) | Emitted at the start of a new content part. |
| [`response.output_audio.delta`](#responseoutputaudiodelta) | A chunk of synthesized speech audio. |
| [`response.output_audio_transcript.delta`](#responseoutputaudiotranscriptdelta) | Incremental transcript of the agent's spoken audio. |
| [`response.output_audio_transcript.done`](#responseoutputaudiotranscriptdone) | Complete transcript of the agent's utterance. |
| [`response.output_audio.done`](#responseoutputaudiodone) | Emitted after all audio for a turn has been sent. |
| [`response.content_part.done`](#responsecontentpartdone) | Emitted when a content part is complete. |
| [`response.output_item.done`](#responseoutputitemdone) | Emitted when a response item is complete. |
| [`response.done`](#responsedone) | Marks the end of a complete response turn. |
| [`response.function_call_arguments.done`](#responsefunctioncallargumentsdone) | Signals a complete function call with name, call ID, and arguments. |
| [`conversation.item.input_audio_transcription.delta`](#conversationiteminputaudiotranscriptiondelta) | Incremental ASR transcript of the user's audio. |
| [`conversation.item.input_audio_transcription.completed`](#conversationiteminputaudiotranscriptioncompleted) | Complete user utterance transcript. |
| [`error`](#error) | Sent when a recoverable or fatal error occurs. |
| [`session.end`](#sessionend) | Sent before closing the connection. Contains session statistics. |

---

### `session.created` {#sessioncreated}

Sent once immediately after the WebSocket is accepted.

```json
{
  "type": "session.created",
  "event_id": "<uuid>",
  "session": {
    "type": "realtime",
    "id": "<client-id>",
    "model": "nemotron-voicechat",
    "modalities": ["audio"],
    "audio": {
      "input":  { "format": {"type": "audio/pcm", "rate": 24000} },
      "output": { "format": {"type": "audio/pcm", "rate": 24000} }
    },
    "instructions": "..."
  }
}
```

---

### `session.updated` {#sessionupdated}

Acknowledgment of a `session.update`, echoing the effective configuration with structured format objects.

```json
{
  "type": "session.updated",
  "event_id": "<uuid>",
  "session": {
    "audio": {
      "input":  { "format": {"type": "audio/pcm", "rate": 24000} },
      "output": { "format": {"type": "audio/pcm", "rate": 24000} }
    },
    "instructions": "...",
    "tools": "[]"
  }
}
```

---

### `input_audio_buffer.speech_started` {#inputaudiobufferspeechstarted}

Emitted when the first ASR token arrives for a new user utterance.

```json
{
  "type": "input_audio_buffer.speech_started",
  "event_id": "<uuid>",
  "audio_start_ms": 0,
  "item_id": "<input-item-id>"
}
```

---

### `input_audio_buffer.speech_stopped` {#inputaudiobufferspeechstopped}

Emitted when the ASR output contains an end-of-speech marker.

```json
{
  "type": "input_audio_buffer.speech_stopped",
  "event_id": "<uuid>",
  "audio_end_ms": 0,
  "item_id": "<input-item-id>"
}
```

---

### `response.created` {#responsecreated}

Emitted when the model produces its first audio output for a turn.

```json
{
  "type": "response.created",
  "event_id": "<uuid>",
  "response": {
    "id": "<response-id>",
    "object": "realtime.response",
    "status": "in_progress",
    "status_details": null,
    "output": []
  }
}
```

---

### `response.output_item.added` {#responseoutputitemadded}

```json
{
  "type": "response.output_item.added",
  "event_id": "<uuid>",
  "response_id": "<response-id>",
  "output_index": 0,
  "item": {
    "id": "<item-id>",
    "object": "realtime.item",
    "type": "message",
    "role": "assistant"
  }
}
```

---

### `response.content_part.added` {#responsecontentpartadded}

```json
{
  "type": "response.content_part.added",
  "event_id": "<uuid>",
  "response_id": "<response-id>",
  "item_id": "<item-id>",
  "output_index": 0,
  "content_index": 0,
  "part": {"type": "audio"}
}
```

---

### `response.output_audio.delta` {#responseoutputaudiodelta}

A chunk of synthesized speech audio.

```json
{
  "type": "response.output_audio.delta",
  "event_id": "<uuid>",
  "response_id": "<response-id>",
  "item_id": "<item-id>",
  "output_index": 0,
  "content_index": 0,
  "delta": "<base64-encoded audio>"
}
```

`delta` is base64-encoded audio in the negotiated output format (default: PCM16, 16-bit little-endian signed, mono, 24 kHz).

---

### `response.output_audio_transcript.delta` {#responseoutputaudiotranscriptdelta}

Incremental transcript of the agent's spoken audio.

```json
{
  "type": "response.output_audio_transcript.delta",
  "event_id": "<uuid>",
  "response_id": "<response-id>",
  "item_id": "<item-id>",
  "output_index": 0,
  "content_index": 0,
  "delta": "Hello, how can I help"
}
```

---

### `response.output_audio_transcript.done` {#responseoutputaudiotranscriptdone}

Complete transcript of the agent's utterance.

```json
{
  "type": "response.output_audio_transcript.done",
  "event_id": "<uuid>",
  "response_id": "<response-id>",
  "item_id": "<item-id>",
  "output_index": 0,
  "content_index": 0,
  "transcript": "Hello, how can I help you today?"
}
```

---

### `response.output_audio.done` {#responseoutputaudiodone}

Emitted after all audio for a turn has been sent (output buffer drained).

```json
{
  "type": "response.output_audio.done",
  "event_id": "<uuid>",
  "response_id": "<response-id>",
  "item_id": "<item-id>",
  "output_index": 0,
  "content_index": 0
}
```

---

### `response.content_part.done` {#responsecontentpartdone}

```json
{
  "type": "response.content_part.done",
  "event_id": "<uuid>",
  "response_id": "<response-id>",
  "item_id": "<item-id>",
  "output_index": 0,
  "content_index": 0,
  "part": {"type": "audio"}
}
```

---

### `response.output_item.done` {#responseoutputitemdone}

```json
{
  "type": "response.output_item.done",
  "event_id": "<uuid>",
  "response_id": "<response-id>",
  "output_index": 0,
  "item": {
    "id": "<item-id>",
    "object": "realtime.item",
    "type": "message",
    "role": "assistant"
  }
}
```

---

### `response.done` {#responsedone}

Marks the end of a complete response turn.

```json
{
  "type": "response.done",
  "event_id": "<uuid>",
  "response": {
    "id": "<response-id>",
    "object": "realtime.response",
    "status": "completed",
    "status_details": null,
    "output": [],
    "usage": {
      "input_tokens": 0,
      "output_tokens": 0,
      "total_tokens": 0,
      "input_token_details": {"cached_tokens": 0},
      "output_token_details": {"text_tokens": 0, "audio_tokens": 0}
    }
  }
}
```

---

### `response.function_call_arguments.done` {#responsefunctioncallargumentsdone}

Sent when the model requests a function call. The client should execute the function and return the result via [`conversation.item.create`](#conversationitemcreate).

```json
{
  "type": "response.function_call_arguments.done",
  "event_id": "<uuid>",
  "response_id": "<response-id>",
  "item_id": "<item-id>",
  "output_index": 0,
  "call_id": "<call-id>",
  "name": "get_current_time",
  "arguments": "{}"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `call_id` | string | Echo this value in the `conversation.item.create` response. |
| `name` | string | Name of the function to call. |
| `arguments` | string | JSON-encoded argument object. |

---

### `conversation.item.input_audio_transcription.delta` {#conversationiteminputaudiotranscriptiondelta}

Incremental ASR transcript of the user's audio.

```json
{
  "type": "conversation.item.input_audio_transcription.delta",
  "event_id": "<uuid>",
  "item_id": "<input-item-id>",
  "content_index": 0,
  "delta": "what is the weath"
}
```

---

### `conversation.item.input_audio_transcription.completed` {#conversationiteminputaudiotranscriptioncompleted}

Complete user utterance transcript.

```json
{
  "type": "conversation.item.input_audio_transcription.completed",
  "event_id": "<uuid>",
  "item_id": "<input-item-id>",
  "content_index": 0,
  "transcript": "What is the weather today?"
}
```

---

### `error` {#error}

Sent when a recoverable or fatal error occurs server-side.

```json
{
  "type": "error",
  "event_id": "<uuid>",
  "error": {
    "code": "inference_timeout",
    "message": "Inference timeout"
  }
}
```

**Known error codes**

| Code | Trigger |
|------|---------|
| `inference_timeout` | Inference call exceeded the configured timeout limit. |
| `inference_error` | Uncaught exception during inference. |
| `session_timeout` | Client sent more than the maximum allowed session duration of audio. |
| `tools_not_set` | `conversation.item.create` received but tools were not configured in `session.update`. |

---

### `session.end` {#sessionend}

Sent by the server before closing the connection. Contains session statistics.

```json
{
  "type": "session.end",
  "event_id": "<uuid>",
  "stats": {
    "chunks_received": 150,
    "chunks_sent": 148,
    "chunks_dropped": 0,
    "triton_inferences": 75,
    "audio_duration_received_s": 12.0
  }
}
```
