# Generate Model Repository

As an alternative to downloading the prebuilt model via the NGC CLI, you can generate a Triton model repository from a checkpoint and mount it directly. This is useful for:

- **HuggingFace checkpoint** — use the publicly released [NemotronLabs Voicechat model on HuggingFace](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B).
- **Custom checkpoint** — use a checkpoint from your own training run (e.g., fine-tuned or experimental weights). Substitute your checkpoint path wherever `~/nemotron-labs-voicechat/hf-checkpoint` appears below.

In both cases the checkpoint directory must contain `model.safetensors`.

## Generate the Triton Model Repository

Use the `deploy_s2s_model.sh` script bundled in the inference container at `/s2s/deploy_s2s_model.sh`.

**Step 1 — Download the HuggingFace checkpoint:**

```bash
mkdir -p ~/nemotron-labs-voicechat/hf-checkpoint
hf download nvidia/NVIDIA-NemotronLabs-VoiceChat-11B --local-dir ~/nemotron-labs-voicechat/hf-checkpoint
```

**Step 2 — Prepare the output directory and run the conversion:**

```bash
mkdir -p ~/nemotron-labs-voicechat/model-repo
chmod 777 ~/nemotron-labs-voicechat/model-repo

docker run -it --rm \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --shm-size=8GB \
  -v ~/nemotron-labs-voicechat/hf-checkpoint:/checkpoint \
  -v ~/nemotron-labs-voicechat/model-repo:/data/models \
  -e NEMO_CHECKPOINT_PATH=/checkpoint \
  --entrypoint /s2s/deploy_s2s_model.sh \
  nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest
```

- `-v ~/nemotron-labs-voicechat/hf-checkpoint:/checkpoint` — mounts the checkpoint into the container.
- `-e NEMO_CHECKPOINT_PATH=/checkpoint` — tells the script to use the local checkpoint; NGC model download is skipped.
- `-v ~/nemotron-labs-voicechat/model-repo:/data/models` — captures the generated Triton model repository on the host.

> **Note:** Model repository generation can take up to 15 minutes.

When generation completes successfully, the container log will show:

```
INFO:checkpoint_utils.import_utils:Triton model repository generation completed successfully!
INFO:checkpoint_utils.import_utils:Model repository location: /data/models/nemotron-voicechat
```

## Launch the Inference Container

Once complete, `~/nemotron-labs-voicechat/model-repo` contains the Triton model repository. Launch the inference container with it mounted at `/data/models` and the server entrypoint overridden:

```bash
docker run -it --rm --name=nemotron-labs-voicechat \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --shm-size=8GB \
  -e NIM_HTTP_API_PORT=9000 \
  -p 9000:9000 \
  -v ~/nemotron-labs-voicechat/model-repo:/data/models \
  --entrypoint /s2s/run_s2s_server.sh \
  nvcr.io/nim/nvidia/nemotron-labs-voicechat:latest
```

---
For a full description of the WebSocket protocol, client and server events, and audio format details, see the API reference.

Next: [API Reference](api-reference.md)
