# NVIDIA NemotronLabs VoiceChat — Early Access

NVIDIA NemotronLabs VoiceChat is an 11B end-to-end, real-time, full-duplex speech model for conversational AI. Unlike traditional cascaded stacks (ASR → LLM → TTS), it jointly performs streaming speech understanding and speech generation in a single unified architecture, eliminating the need for multiple models or API handoffs and reducing end-to-end latency.

The model is delivered as an optimized NVIDIA inference container that accepts streaming audio input and returns synthesized speech output through a bidirectional WebSocket interface. It supports real-time function calling, allowing the model to invoke external tools mid-conversation while maintaining a natural conversational flow.

## Deployment Instructions

Step-by-step instructions for deploying and running the Nemotron Voicechat container are available at:

**[https://github.com/NVIDIA-NeMo/Speech/tree/nemotron-labs-voicechat/voicechat_realtime_instructions](https://github.com/NVIDIA-NeMo/Speech/tree/nemotron-labs-voicechat/voicechat_realtime_instructions)**

The instructions cover:

- Hardware, driver, and software prerequisites
- Launching the inference container
- Running real-time voice conversations and function calling
- Generating a Triton model repository from a HuggingFace or custom checkpoint
- WebSocket and HTTP API reference
