## EasyMagpieTTS — vLLM-Omni two-stage inference

Streaming TTS for **NemotronTTS** (Nemotron-H backbone + per-codebook local
transformer over a 25 fps spectral codec) via [vLLM-Omni](https://github.com/vllm-project/vllm-omni).

The in-engine **two-stage pipeline** chains:

| Stage | Role |
|-------|------|
| **0 — talker** | Autoregressive Nemotron-H + local transformer → stacked acoustic codes |
| **1 — codec** | Stateful native vLLM codec → 22.05 kHz waveform |

Model definition and pipeline registration live in
[`easymagpie_vllm_omni/`](easymagpie_vllm_omni/) and
[`vllm_plugin_easymagpie_omni/`](vllm_plugin_easymagpie_omni/).
Deployment knobs are in [`deploy/easymagpie.yaml`](deploy/easymagpie.yaml).

### Convert a NeMo checkpoint

This step turns the training-time `.nemo` checkpoints into a self-contained
vLLM-Omni model directory: it converts the talker and causal codec to native
vLLM models, precomputes the text-embedding lookup, and saves the tokenizer and
optional speaker embedding. Run it in the **NeMo environment** from the repository root:

```bash
python examples/tts/easymagpie_vllm_omni/scripts/convert_to_vllm.py \
  --nemo_file /path/to/emptts.nemo \
  --codec_model_path /path/to/25fps_spectral_codec.nemo \
  --phoneme_tokenizer_path /path/to/bpe_ipa_tokenizer.json \
  --outdir examples/tts/easymagpie_vllm_omni/converted_model \
  --context_audio /path/to/reference_voice.wav \
  --speaker_name eng
```

### Setup the serving environment

Serving needs a GPU, matching **vLLM 0.24 / vLLM-Omni 0.24** versions, and this package.
It does not need NeMo after conversion:

```bash
cd examples/tts/easymagpie_vllm_omni
conda create -n easymagpie-vllm python=3.12 -y
conda activate easymagpie-vllm
pip install -r requirements.txt
pip install -e .
# optionally for notebook
pip install ipykernel
python -m ipykernel install --user \
  --name easymagpie-vllm \
  --display-name "Python (easymagpie-vllm)"
```

Mamba's selective-state-update kernel requires shape- and GPU-specific tuning, so an untuned cache can give
suboptimal performance. Reuse the same Triton/vLLM cache directories across launches so repeated runs accumulate
better kernels; for an explicit sweep, run `python scripts/tune_mamba_ssu.py --model converted_model` and restart.

### Quick start — offline synthesis

See [`scripts/offline_demo.ipynb`](scripts/offline_demo.ipynb) to check how `AsyncOmni` is initialized and used.

### Serve over HTTP and WebSocket

```bash
bash ./scripts/run_server.sh ./converted_model 8091
```

This starts `vllm serve` with the EasyMagpie plugin on port 8091. Two serving
APIs are available:

- `POST /v1/audio/speech` with a complete text input.
- `WS /v1/audio/speech/stream` with incremental text/token updates and
  asynchronous PCM audio output.

Query the HTTP endpoint from any OpenAI-compatible client:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"This is a TTS service test.","voice":"eng","response_format":"wav","stream":true,"stream_format":"audio"}' \
  --output out.wav
```

See [`scripts/server_request.ipynb`](scripts/server_request.ipynb) for examples
of both serving APIs.

### Benchmarks

```bash
# Benchmark acoustic token prediction only (no codec).
python scripts/benchmark_model.py --model ./converted_model -n 128 -c 1 32 \
    [--incremental --tokens-per-chunk 5]

# Benchmark the service's HTTP API.
python scripts/benchmark_server.py --text-file vctk_subset.txt -n 128 -c 1 32

# Benchmark the service's incremental synthesis via its WebSocket API.
python scripts/benchmark_incremental_server.py --model ./converted_model \
    --text-file vctk_subset.txt --tokens-per-chunk 5 -n 128 -c 1 32
```

#### RTX A6000 reference (2026-07-21)

Results for 128 requests using VCTK texts. Service rows use the default FP32 codec with 6/6/8-frame cadence:

| Concurrency | Benchmark | Input | Requests/s | RTF | Mean latency | Underruns |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | Model | Whole text | 1.32 | 7.85x | 28.7 ms TTFT | — |
| 1 | Model | 5 tokens/chunk | 1.28 | 7.64x | 29.3 ms TTFT | — |
| 1 | HTTP service | Whole text | 1.27 | 7.73x | 130.8 ms TTFA | 0 / 1,335 |
| 1 | WebSocket service | 5 tokens/chunk | 1.25 | 7.24x | 137.2 ms TTFA | 0 / 1,226 |
| 32 | Model | Whole text | 11.20 | 67.75x | 107.3 ms TTFT | — |
| 32 | Model | 5 tokens/chunk | 10.44 | 63.01x | 105.1 ms TTFT | — |
| 32 | HTTP service | Whole text | 7.47 | 48.43x | 606.0 ms TTFA | 0 / 1,426 |
| 32 | WebSocket service | 5 tokens/chunk | 7.45 | 44.12x | 690.9 ms TTFA | 0 / 1,249 |
