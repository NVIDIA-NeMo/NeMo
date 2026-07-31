# Nemotron Voicechat — Inference Container Instructions

This directory contains instructions for deploying the [NVIDIA NemotronLabs VoiceChat 11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B) model as an optimized NVIDIA inference container. The container provides a real-time, interactive voice conversation service through a bidirectional WebSocket interface, with support for function calling.

## Contents

| Document | Description |
|----------|-------------|
| [Prerequisites](prerequisites.md) | Hardware, driver, Docker, and container toolkit requirements |
| [Deploy and Run](deploy.md) | Launch the container, run voice conversations, and use function calling |
| [Generate Model Repository](generate-model-repo.md) | Alternative to NGC download — build a Triton model repository from a HuggingFace or custom checkpoint |
| [API Reference](api-reference.md) | WebSocket and HTTP API, audio format, and client/server events |

If your environment is not yet set up, start with [Prerequisites](prerequisites.md). If you have already met the requirements, go directly to [Deploy and Run](deploy.md).
