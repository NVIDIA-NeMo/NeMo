# Conflict matrix — option pairs that don't work together

Table format: `A | B | conflict | severity | resolution`.

Severities:
- **fatal** = auto-patch impossible; requires offline manifest pre-processing
  or data ingestion. Skill exits non-zero with explanation.
- **error** = auto-patchable.
- **warning** = patchable but context-dependent; the skill emits a comment
  in the patched YAML and a section in the report.

| A | B | conflict | severity | resolution |
|---|---|---|---|---|
| `data.train_ds.indexed: true` | `extra_fields:` on a `nemo` / `nemo_tarred` / `multimodal_conversation` entry | `LazyNeMoTarredIterator(indexed=True)` raises `RuntimeError` (`nemo_adapters.py:485-487`). Graph-token random access has no stable index. | fatal | Pre-process the manifest offline to materialize the extra fields, drop the `extra_fields` key. |
| `data.train_ds.indexed: true` | `slice_length:` on a `nemo` / `nemo_tarred` entry | Sliced cuts have no stable index — slicing rewrites the cut sequence. | fatal | Re-shard the audio offline to the target slice length, drop the `slice_length` key. |
| `data.train_ds.indexed: true` | Lhotse Shar `cuts.*.jsonl.gz` (compressed cuts) | `lhotse/indexing.py:88-110` rejects compressed paths in `indexed_path_kind`. AMI's stock distribution hits this. | fatal | Drop the corpus from the blend, OR re-export the Shar with `compress_jsonl=False`, OR convert to `nemo_tarred` format. |
| `data.train_ds.indexed: true` | `tarred_audio_filepaths: *.tar.gz` | Compressed tars can't be indexed. | fatal | Re-pack the tars uncompressed. |
| `data.train_ds.indexed: true` | `pipe:cmd \| cmd2` paths | Pipe commands aren't seekable; `validate_indexed_access` raises `ValueError`. | fatal | Materialize the upstream of the pipe to a real file, then point at that. |
| `data.train_ds.force_map_dataset: true` | `data.train_ds.force_iterable_dataset: true` | `dataloader.py:278-280` asserts these are mutually exclusive. | error | Keep only `force_map_dataset: true`. |
| `data.train_ds.force_map_dataset: true` + `data.train_ds.use_stateful_dataloader: true` | `data.train_ds.shard_seed: "randomized"` | Map path doesn't need per-rank seed differentiation; `"randomized"` adds worker-PID-derived seeding that breaks across resume. NeMo's `dataloader.py:556-572` warns + auto-overwrites with `seed`. | error | Set `shard_seed: <int>` (typically equal to `seed`). |
| `data.train_ds.use_stateful_dataloader: true` | per-chunk seed rotation in launcher | Silent corruption: model RNG (dropout, aux-loss, random-init) diverges across chunks even though sampler state restores correctly. | error | Pin a single seed across the entire chain. `train_and_eval.py:925-952` does this when `--enable-indexes-prefetch` is set. For arbitrary launchers, set the same seed in every chunk's command. |
| `data.train_ds.use_stateful_dataloader: true` | `num_workers` change between save and restore | Hard error from `torchdata.StatefulDataLoader`. | error | Document `num_workers` in the YAML / launcher header; never change between chunks. |
| `data.train_ds.use_stateful_dataloader: true` | `world_size` change between save and restore (`num_nodes * devices_per_node`) | Hard error from torchdata. | error | Restart from a converted HuggingFace checkpoint if you need to scale (no resume in that case). |
| AIStore MOSS GetBatch (`USE_AIS_GET_BATCH=true`, `USE_AIS_INDIVIDUAL_GETS` unset) | non-EN-replicated multilingual data on `s3://FLEURS/...`, `s3://MCV/...`, etc. | MOSS returns 200 + empty body for missing objects. Triggers the empty-content retry path which then crashes (§10 / §16 in failure-modes.md). | warning | Set `USE_AIS_INDIVIDUAL_GETS=true` until the data is replicated to AIS, OR replicate the data, OR switch the blend to lustre tar paths if available. |
| `data.validation_ds.force_finite: true` | training (`data.train_ds`) | `force_finite` caps the infinite-mux behavior that training requires. | error | `force_finite: true` is a validation-only flag; don't propagate it to `data.train_ds`. |
| `exp_manager.checkpoint_callback_params.every_n_train_steps: null` | external preemption (`svc-hwinf-cs-sched`, NODE_FAIL, etc.) at < 1 epoch | No mid-epoch save; chunk progress is lost on every preemption. | warning | Add `every_n_train_steps: 50-250` (and/or `train_time_interval: "00:30:00"`). Lightning ORs the triggers. |
| `exp_manager.max_time_per_run` ≥ SLURM walltime | SLURM SIGKILL during teardown | The internal preemption save never fires; teardown is killed mid-write. | error | Set `max_time_per_run` to `<SLURM walltime - 10min>` (e.g. `00:03:50:00` for a 4h walltime). |
| `data.train_ds.indexes_root` | `prefetch_indexes_to_ssd.sh` destination | Mismatch → manifests fail to find their `.idx` neighbors at training time. | error | Keep both in sync. The prefetch script's default is `/tmp/idx`; the YAML's `indexes_root` must match. |
| `submit_build_indexes.py` (no `--bypass-nvidia-hook`) | NRT cpu partition (lacks `nvidia-container-cli`) | enroot's `98-nvidia.sh` hook hard-fails container start. | error | Pass `--bypass-nvidia-hook` for any cluster whose cpu partition lacks `nvidia-container-cli`. |
| Container `aistore` SDK < 1.17 | AIStore in play | `lhotse_resumable/lhotse/ais/batch_loader.py:75` requires `>=1.17.0`. | error | Pin `aistore>=1.17` in the build/training container preamble; `submit_build_indexes.py:227` does this. |
| `data.train_ds.seed` | per-chunk seed rotation in launcher | Same as above — silent model-level divergence. | error | Pin `seed` in YAML AND in launcher; both must be invariant across the chain. |
| `pretrained_llm` change | resume from a chain | `init_from_checkpoint` resharding issues; tokenizer mismatch. | warning | Don't change the LLM mid-chain. Start fresh if you need a different LLM (+ optionally `init_from_checkpoint: <previous_run.ckpt>` for transfer). |
| `model.aux_loss_coeff > 0` | `model.activation_checkpointing_llm: true` | AC + MoE aux-loss recompute dtype flip (debug-cluster-run §6(16)). `CheckpointError: Recomputed values ... different metadata`. Orthogonal to resumable, but a frequent recipe pitfall. | error | Set `aux_loss_coeff: 0`, OR disable `activation_checkpointing_llm` (perception AC alone is fine). |
