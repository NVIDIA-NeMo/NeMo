# Generate Model Repository

Instead of relying on the container's built-in model download, you can generate a Triton model repository from a checkpoint and mount it directly. This is useful for:

- **HuggingFace checkpoint** — use the publicly released [Nemotron Voicechat model on HuggingFace](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-12B).
- **Custom checkpoint** — use a checkpoint from your own training run (e.g., fine-tuned or experimental weights).

In both cases the checkpoint directory must contain `model.safetensors`.

## Generate the Triton Model Repository

Use the `deploy_s2s_model.sh` script bundled in the inference container at `/s2s/deploy_s2s_model.sh`:

```bash
export CHECKPOINT_DIR=/path/to/checkpoint   # HuggingFace or custom checkpoint
export OUTPUT_DIR=/path/to/output/model-repo

docker run -it --rm \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --shm-size=8GB \
  -v $CHECKPOINT_DIR:/checkpoint \
  -v $OUTPUT_DIR:/data/models \
  -e NEMO_CHECKPOINT_PATH=/checkpoint \
  --entrypoint /s2s/deploy_s2s_model.sh \
  nvcr.io/nvidia/nemotron-voicechat:latest
```

- `-v $CHECKPOINT_DIR:/checkpoint` — mounts the checkpoint into the container.
- `-e NEMO_CHECKPOINT_PATH=/checkpoint` — tells the script to use the local checkpoint; NGC model download is skipped.
- `-v $OUTPUT_DIR:/data/models` — captures the generated Triton model repository on the host.

## Launch the Inference Container

Once complete, `$OUTPUT_DIR` contains the Triton model repository. Launch the inference container with it mounted at `/data/models` and the server entrypoint overridden:

```bash
docker run -it --rm --name=nemotron-voicechat \
  --runtime=nvidia \
  --gpus '"device=0"' \
  --shm-size=8GB \
  -e NIM_HTTP_API_PORT=9000 \
  -p 9000:9000 \
  -v $OUTPUT_DIR:/data/models \
  --entrypoint /s2s/run_s2s_server.sh \
  nvcr.io/nvidia/nemotron-voicechat:latest
```
