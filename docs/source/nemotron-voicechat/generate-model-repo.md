# Generate Model Repository from a NeMo Checkpoint

If you have a local NeMo checkpoint, you can generate the Triton model repository yourself using the `deploy_s2s_model.sh` script bundled in the inference container at `/s2s/deploy_s2s_model.sh`. This skips the NGC model download entirely.

The checkpoint directory must contain `model.safetensors`.

```bash
export CHECKPOINT_DIR=/path/to/nemo-checkpoint
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

- `-v $CHECKPOINT_DIR:/checkpoint` — mounts the NeMo checkpoint into the container.
- `-e NEMO_CHECKPOINT_PATH=/checkpoint` — tells the script to use the local checkpoint; NGC download is skipped.
- `-v $OUTPUT_DIR:/data/models` — captures the generated Triton model repository on the host (the script writes to `/data/models` inside the container by default).

Once complete, `$OUTPUT_DIR` contains the Triton model repository. Launch the inference container with the generated repo mounted at `/data/models` and the server entrypoint overridden:

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
