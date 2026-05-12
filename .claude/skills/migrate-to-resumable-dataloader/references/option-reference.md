# Option reference — every YAML/launcher field that interacts with the resumable path

Field-by-field exhaustive reference. Required values, rationale, source code
pointer, see-also link to MIGRATION_GUIDE.md and (when relevant) to the
0909-debug docs that motivated the field.

## `data.train_ds` — required for the indexed + resumable path

| field | required value | purpose | see also |
|---|---|---|---|
| `indexed` | `true` | Routes every nested `input_cfg` source to its indexed adapter (`LazyNeMoTarredIterator(indexed=True)`, `IndexedJsonlReader`, etc.). Without this flag, the streaming/replay path is used. Defined in `LhotseDataLoadingConfig` (`NeMo_resumable/nemo/collections/common/data/lhotse/dataloader.py:261`). | MIGRATION_GUIDE.md "Step 2 — Flip two flags" |
| `use_stateful_dataloader` | `true` | Swaps PyTorch `DataLoader` → `torchdata.StatefulDataLoader` so iterator state is checkpointed in `meta.pt` under `DataModule.train_dataloader` (3 keys: `_snapshot`, `_steps_since_snapshot`, `_iterator_finished`). Verified via `inspect_meta.py` against `step=2000.ckpt` / `step=3000.ckpt` / `step=N-last.ckpt` (see `agent-debug-workspace/nano-v3-1node-resumable-tests.md`). | `dataloader.py:272`, MIGRATION_GUIDE.md "Step 2" |
| `force_map_dataset` | `true` | Consume sources map-style (random `__getitem__`) rather than iterable. Required for full per-worker resume state — without it, the iterator-graph state isn't restorable in O(1). On the map path, cross-rank de-dup is via `rank/world_size` slicing in `DynamicBucketingSampler` (`dataloader.py:680-681` constructs the sampler with `rank=global_rank, world_size=world_size`), NOT via per-rank seed differentiation. | `dataloader.py:247-279` |
| `indexes_root` | local SSD path (e.g. `/tmp/idx`) matching `prefetch_indexes.py` destination | Where the prefetched `.idx` mirror is read from at training time. Mirror tree preserves the data-file paths (`<indexes_root>/lustre/...` mirroring the blend's lustre paths). Resolved by `resolve_idx_path` in `NeMo_resumable/nemo/collections/common/data/lhotse/indexed_adapters.py:170`. **Must match the prefetch script's destination**, otherwise manifests fail to find their `.idx` neighbors at training time. | MIGRATION_GUIDE.md "keep indexes on a separate fast disk" |
| `seed` | a fixed integer, **invariant across chunks** | Controls Python/numpy/torch global RNG via `pl.seed_everything(seed)` at chunk start. **MUST NOT change on resume**, otherwise dropout / aux-loss / random-init diverge across chunks even though `StatefulDataLoader.load_state_dict` restores sampler state correctly. The 0909 longform chains (see `agent-debug-workspace/0909-longform-failures.md`) hit this exact silent-corruption bug because `train_and_eval.py` rotated `FIXED_SEEDS[seed_offset+i]` per chunk. Fixed in `train_and_eval.py:925-952` — when `--enable-indexes-prefetch` is set, all chunks use the same seed. | MIGRATION_GUIDE.md "Operational constraints" §1, `0909-longform-failures.md` Cause A |
| `shard_seed` | a fixed integer (NOT `"randomized"`) when `force_map_dataset: true` | Sampler RNG for `DynamicBucketingSampler`. On the map path, cross-rank de-dup is by index slicing (`rank=global_rank, world_size=world_size` at `dataloader.py:680-681`), so per-rank seed differentiation is unneeded. `"randomized"` is iterable-path machinery; on the map path it adds worker-PID-derived seeding that breaks across resume boundaries. **NeMo's dataloader.py auto-overwrites `shard_seed: "randomized"` → `shard_seed: <seed>` with a warning when `force_map_dataset + use_stateful_dataloader` are both true** (`dataloader.py:556-572`). The auto-overwrite is a safety net; pin it explicitly in YAML so the rationale is visible in code review. | `0909-summary.md` R2, `dataloader.py:543-572` |
| `num_workers` | match between save and restore | `StatefulDataLoader` hard requirement: changing `num_workers` between save and restore raises a hard error from torchdata. Document the value in the YAML / launcher header. | MIGRATION_GUIDE.md "Operational constraints" §1 |
| `concurrent_bucketing` | **`false`** when `force_map_dataset + use_stateful_dataloader` are both true | The default (`true`) spawns a `daemon=True` producer thread inside `DynamicBucketingSampler` (`lhotse_resumable/lhotse/dataset/sampling/dynamic_bucketing.py:924-944`) that pre-pulls cuts from the source iterator and fills per-bucket queues. The main thread (which `StatefulDataLoader` checkpoints) and the producer thread BOTH advance `self.cuts_iter`, so the cursor saved at `state_dict` time does NOT reflect the cuts the producer has already pre-fetched. On resume, the next-cut cursor is correct from the main thread's view but the producer's pre-fetched cuts are gone, so the bucketing/order across resume boundaries is nondeterministic. Also breaks single-run bit-exact reproducibility between two runs of the same config because the producer's scheduling is OS-thread-dependent. Set `concurrent_bucketing: false` in `data.train_ds` for any resumable run. | `lhotse_resumable/lhotse/dataset/sampling/dynamic_bucketing.py:924-944`, `failure-modes.md §<new>`, observed in `0909-multiling-*` (2026-05-11) |
| `force_iterable_dataset` | unset (or `false`) | Mutually exclusive with `force_map_dataset: true`. `dataloader.py:278-280` asserts `not (force_map_dataset and force_iterable_dataset)`. | `dataloader.py:278-280` |
| `force_finite` | unset / `false` (training only) | Setting this to `true` would cap the infinite-mux behavior that training requires. Only validation_ds needs `force_finite: true`. | MIGRATION_GUIDE.md "Operational constraints" §4 |
| `extra_fields` (on any nested `nemo` / `nemo_tarred` / `multimodal_conversation`) | unset | `LazyNeMoTarredIterator(indexed=True)` raises `RuntimeError` if `extra_fields` is set (`nemo_adapters.py:485-487`: "LazyNeMoTarredIterator(indexed=True) does not support 'extra_fields'"). Same constraint on `LazyNeMoIterator` at `nemo_adapters.py:148-152`. Pre-process the manifest offline. | `nemo_adapters.py:148-152, 485-487` |
| `slice_length` (on any nested `nemo` / `nemo_tarred`) | unset | Slicing rewrites cuts in a way that has no stable index. The dataloader still threads `slice_length` through (`dataloader.py:253-256`, `cutset.py:413, 436, 662, 678, 693, 717, 1506, 1551`), but the indexed reader does not honor it. Pre-process offline if needed. | MIGRATION_GUIDE.md "Prerequisites" §3 |
| compressed `.jsonl.gz` / `.tar.gz` paths | reject | `lhotse.indexing.indexed_path_kind` returns `None` for any path matching `_is_compressed_path` (`lhotse_resumable/lhotse/indexing.py:88-110`); `validate_indexed_access` raises `ValueError("...requires uncompressed JSONL or tar...")`. Re-extract or re-export with `compress_jsonl=False` for Shar. | MIGRATION_GUIDE.md "Prerequisites" §1, `lhotse/indexing.py:88-110, 130-135` |
| `pipe:` paths (`pipe:cmd \| cmd`) | reject | Pipe commands aren't seekable. `validate_indexed_access` raises `ValueError("...requires seekable data sources...")`. | `lhotse/indexing.py:126-128` |
| `.json` extension on a JSONL manifest | accepted | NeMo ships many ASR/SLM manifests as `*.json` (one JSON object per line). `lhotse/indexing.py:99-107` accepts both `.json` and `.jsonl` since the indexer only relies on newline-separated records. (Pretty-printed multi-line JSON would produce a bogus index, but that's not a supported NeMo manifest layout.) | `lhotse/indexing.py:99-107` |

## `data.validation_ds` — finite map access required

| field | required value | purpose | see also |
|---|---|---|---|
| `indexed` | `true` (inherited from train_ds OR per-validation set) | Same as `data.train_ds.indexed`. | MIGRATION_GUIDE.md Step 2 |
| `force_map_dataset` | `true` | Map-style finite access. | MIGRATION_GUIDE.md Step 2 |
| `force_finite` | `true` | **Caps the infinite-mux behavior that training uses**. Without this, validation loops forever (the multiplexer never raises StopIteration). MIGRATION_GUIDE.md "Operational constraints" §4 calls this out explicitly. | MIGRATION_GUIDE.md "Operational constraints" §4 |
| `use_stateful_dataloader` | `false` (or `true`, doesn't matter) | Validation never resumes from mid-eval; eval is run-to-completion. Either value works. | — |
| `indexes_root` | same path as train_ds | Same — must match the prefetch destination. | — |
| `seed` / `shard_seed` | same fixed integers as train_ds (or any fixed value) | Determinism for eval. Doesn't need to be invariant across chunks the way training does. | — |

## `exp_manager` — Lightning resume contract

| field | required value | purpose | see also |
|---|---|---|---|
| `resume_if_exists` | `true` | Lightning auto-finds the latest `step=N-last.ckpt` and loads model + optimizer + dataloader state from DCP shards + `meta.pt`. Without this, every chunk starts from scratch. | MIGRATION_GUIDE.md "Lightning resume contract" |
| `resume_ignore_no_checkpoint` | `true` | First chunk runs without prior ckpt; without this flag, the first run errors. | — |
| `checkpoint_callback_params.every_n_train_steps` | small int (50–250 recommended) | Mid-chunk saves so external preemption (`svc-hwinf-cs-sched`, NODE_FAIL, etc.) doesn't waste 80–150 step progress. The 0909 longform chains (`0909-longform-failures.md` Cause B) accumulated **0** progress past `step=1000` because the only save trigger was `every_n_epochs: 1` and chunks averaged 75–150 steps after preemption. | `0909-longform-failures.md` Cause B |
| `checkpoint_callback_params.train_time_interval` | `"00:30:00"` (suggested) | Belt-and-braces wall-clock save trigger. Lightning ORs the per-step and per-time triggers, so both can coexist. | best-practices.md §4 |
| `checkpoint_callback_params.every_n_epochs` | `null` or `1` | If you keep `every_n_epochs: 1`, *also* set `every_n_train_steps`; do not rely on epochs alone. | — |
| `checkpoint_callback_params.save_top_k` | `-1` (no pruning) | Prevents Lightning from deleting old checkpoints when `monitor` doesn't fire. With `every_n_train_steps + every_n_epochs` saves you want all of them on disk. | — |
| `max_time_per_run` | `<SLURM walltime - 10min>` | NeMo's `PreemptionCallback` fires here, leaving a 10-minute buffer for the teardown tail. **Does NOT fire on external SIGTERM** (only on its own timer) — external cancels can still lose progress. Mitigated by frequent step/time-based saves. | debug-cluster-run §6(11) |

## `trainer` — Lightning + parallelism

| field | constraint | purpose | see also |
|---|---|---|---|
| `devices` / `num_nodes` | match between save and restore | StatefulDataLoader is sensitive to `world_size`; changing it between save and restore raises a hard error. To scale a chain mid-flight you must restart from a converted HuggingFace checkpoint (no resume). | MIGRATION_GUIDE.md "Operational constraints" §1 |
| `max_steps` | unchanged across chain | Chain semantics: each chunk advances `global_step`; `max_steps` is the chain target. Don't reduce it mid-chain or Lightning will think training is finished. | — |
| `limit_train_batches` | usually `1000` | Defines an "epoch". With `every_n_epochs: 1` this is also the only save trigger if `every_n_train_steps` is unset. See `every_n_train_steps` above. | — |

## Launcher contract — `train_and_eval.py` and equivalents

| concern | requirement | purpose | see also |
|---|---|---|---|
| Per-chunk seed | **invariant across all chunks of a chain** when `use_stateful_dataloader: true` | StatefulDataLoader contract: model RNG must be the same on resume so dropout/aux-loss/random-init are bit-exact across chunks. The 0909 longform chains hit this with `FIXED_SEEDS[0..9]` rotation. Fixed in `train_and_eval.py:925-952`: when `--enable-indexes-prefetch` is set, `seeds = [seed_or_default] * num_runs`. The skill should grep for any `FIXED_SEEDS[i]` / `seed = randint(...)` / `seed=run_idx` patterns in arbitrary launchers and warn. | `train_and_eval.py:925-952`, `0909-longform-failures.md` Cause A |
| Indexes prefetch preamble | every chunk's container startup runs `prefetch_indexes.py` (or the equivalent rsync) onto each node's local SSD, populating `<indexes_root>` before `salm_train.py` starts | `train_and_eval.py:577-578` does this via `prefetch_indexes_to_ssd.sh`; if missing, training reads `.idx` files from lustre on every `__getitem__` call (slow; defeats the purpose). | `train_and_eval.py:577-578` |
| `num_workers`, `world_size` | invariant across chain | Hard requirement of StatefulDataLoader (see above). Launcher should NOT change `--num-nodes` or `--num-workers` between chunks. | MIGRATION_GUIDE.md "Operational constraints" §1 |
| `--bypass-nvidia-hook` for cpu partitions | required on clusters whose `cpu_partition` lacks `nvidia-container-cli` (e.g. NRT) | Without it, enroot's `98-nvidia.sh` hook hard-fails the container start on cpu partitions of those clusters. Sets `--export=ALL,NVIDIA_VISIBLE_DEVICES=void` on the sbatch line. Used by `submit_build_indexes.py:122-129, 240-245` and `train_and_eval.py`. | `submit_build_indexes.py:122-129` |
| PYTHONPATH | must include both `lhotse_resumable/` and `NeMo_resumable/` | Without it, the in-container default `lhotse` / `nemo` are loaded and lack the resumable code. `submit_build_indexes.py:225` does this; arbitrary launchers must too. | `submit_build_indexes.py:225` |

## AIStore env vars

| env var | required when | purpose | see also |
|---|---|---|---|
| `USE_AIS_GET_BATCH` | training data is on `s3://`, `ais://`, or `http(s)://` AND the cluster has `AIS_ENDPOINT` | Skip eager `IndexedTarMemberReader` per shard; defer audio fetch to AIS at sample time via `AISBatchLoader`. Read at `nemo_adapters.py:459`. | aistore-vs-non-aistore.md |
| `USE_AIS_INDIVIDUAL_GETS` | non-EN-replicated multilingual data on AIS, or any time MOSS GetBatch returns empty content | Routes through per-object `Object.get_reader(archive_config=...).read_all()` instead of MOSS GetBatch. Slower but bypasses MOSS-side issues. `lhotse_resumable/lhotse/ais/batch_loader.py:67, 218` (the `prefer_individual` flag). | failure-modes.md §16 |
| `AIS_ENDPOINT` | always when AIStore in play | The AIS proxy URL. IAD: `http://asr.iad.oci.aistore.nvidia.com:51080`. Set in `cluster_configs/<cluster>.yaml` under `env_vars`. | `cluster_configs/iad.yaml:31` |
| `aistore` SDK version | ≥ 1.17 | `lhotse_resumable/lhotse/ais/batch_loader.py:75` requires `aistore>=1.17.0`. As of 2026-05-10, latest is 1.23.0. The `_moss_attrs` normalizer at `batch_loader.py:81` handles both MossIn (≤1.18) and MossOut (≥1.20) attribute namings. | `lhotse_resumable/lhotse/ais/batch_loader.py:75-89` |

## Index building

| concern | requirement | purpose | see also |
|---|---|---|---|
| Uncompressed sources only | `.jsonl` / `.tar` (NOT `.jsonl.gz` / `.tar.gz`); Shar `cuts.*.jsonl` not `cuts.*.jsonl.gz` | See `lhotse/indexing.py:88-110, 130-135`. AMI's stock distribution as `.jsonl.gz` Shar fails — drop AMI from the blend until an uncompressed export is available. | `data_blends/iad/granary1p1-en-resumable.yaml` header comment, MIGRATION_GUIDE.md "Prerequisites" §1 |
| No `extra_fields` | every `nemo` / `nemo_tarred` / `multimodal_conversation` entry must omit `extra_fields` | `LazyNeMoTarredIterator(indexed=True)` raises explicitly. | `nemo_adapters.py:485-487` |
| No `slice_length` | every `nemo` / `nemo_tarred` entry must omit `slice_length` | Sliced cuts have no stable index. | dataloader.py:253-256 |
| Workers | 95 (on 96-cpu node) for 80k–400k files; 48 if OOM | Tar parsing is GIL-bound (process executor required). 96-cpu / 95-worker / `--exclusive` is the sweet spot. ProcessPool OOM signature: `concurrent.futures.process.BrokenProcessPool: A process in the process pool was terminated abruptly`. Drop to 48 workers if 95 OOMs. | failure-modes.md §8 |
| Time | ~90 min for 80k files; ~2-3 h for 360k files | `submit_build_indexes.py:131` defaults `time_min=04:00:00`. | submit_build_indexes.py:131 |
| Mirror destination | lustre under `<workspace>/indexes_mirror/` (writable, fast enough for prefetch source); NOT S3 | `prefetch_indexes.py` then pulls onto each node's local SSD at `/tmp/idx` (or whatever `indexes_root` resolves to). | submit_build_indexes.py:88-92, prefetch_indexes.py |
| `aistore` SDK in builder container | required if any source is `s3://` / `ais://` | `submit_build_indexes.py:227` does `pip install --quiet --disable-pip-version-check aistore`. Without it, lhotse falls back to smart_open's AWS S3 client and fails with `io.UnsupportedOperation: seek`. Pin the SDK version range to match the lhotse code (`aistore>=1.17`). | submit_build_indexes.py:218-227 |
| Reusability | once per blend; reuse across experiments | Already-indexed files are skipped; `--force` to rebuild. Re-runs are safe. | build_indexes.py:386 |

## Cluster info

| concern | requirement | purpose | see also |
|---|---|---|---|
| `cluster_configs/<cluster>.yaml` must exist | always | `submit_build_indexes.py` and `train_and_eval.py` read SSH creds, partition, container, env_vars from it. | TEMPLATE.yaml |
| `nvidia-container-cli` on cpu partitions | NRT lacks it (cpu / cpu_interactive / cpu_datamover); IAD has it | If absent, use `--bypass-nvidia-hook` (sets `--export=ALL,NVIDIA_VISIBLE_DEVICES=void`). | submit_build_indexes.py:122-129 |
| `AIS_ENDPOINT` env var | required when AIStore is the audio backend | Set in `env_vars:` block of cluster config. IAD has it; lustre-only clusters (typically NRT) won't. | cluster_configs/iad.yaml:31 |
