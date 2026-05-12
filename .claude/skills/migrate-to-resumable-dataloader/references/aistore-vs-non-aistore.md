# AIStore vs non-AIStore workflows

The indexed + resumable Lhotse pipeline supports two storage backends for
the audio tar files. Manifests can be on lustre regardless. The choice of
backend changes which env vars / flags / container deps are required.

## Detection

The skill picks the workflow based on the **blend's `tarred_audio_filepaths`
scheme**, NOT the cluster name:

| signal | workflow |
|---|---|
| `tarred_audio_filepaths: s3://...` or `ais://...` or `http(s)://...` | **AIStore** workflow |
| `tarred_audio_filepaths: /lustre/...` (or any local-FS path) | **non-AIStore** workflow |
| Both in the same blend | **AIStore** workflow (the local files are still loadable; AIS path is the strictly-larger superset) |

Cluster `env_vars` containing `AIS_ENDPOINT=...` is a necessary but not
sufficient signal — the blend may still be all-lustre, in which case
`AIS_ENDPOINT` is unused.

## Workflow A — AIStore (s3:// / ais:// audio)

### Required setup

- **`aistore` SDK installed** in the training container. Either pre-baked
  or `pip install aistore` in the preamble (no version pin needed; the
  lhotse_resumable code's `_moss_attrs` normalizer handles both
  pre-/post-MossOut-rename SDKs).
- **`AIS_ENDPOINT` exported** in cluster env_vars (and forwarded into the
  container via `--container-env=AIS_ENDPOINT,...`). Optionally
  `AIS_AUTHN_URL` and `AIS_AUTHN_TOKEN` for authenticated AIStore
  deployments (MOSS GetBatch requires the token).
- **`USE_AIS_GET_BATCH=true`** env var in the training step (set
  automatically by `--enable-indexes-prefetch` via
  `train_and_eval.py`). This short-circuits eager
  `IndexedTarMemberReader` construction: the indexed tar readers would
  otherwise instantiate one per shard at startup, which on a 41k-shard
  blend means 41k AIS HTTP connections opened before training begins.
  With `USE_AIS_GET_BATCH=true`, audio is fetched lazily at sample time
  via AIStore's MOSS GetBatch — one batched HTTP call per minibatch.

### Optional setup

- **`USE_AIS_INDIVIDUAL_GETS=true`** (or
  `--enable-ais-individual-gets`): bypass MOSS GetBatch entirely and
  fetch each object via `Object.get_reader(archive_config=...).read_all()`.
  Slower (one HTTP call per object instead of one per minibatch) but
  works around MOSS-specific server-side issues — e.g. empty-content
  returns for non-replicated multilingual data on iad AIS, which crashes
  the GetBatch path's empty-content retry logic.

### Required code paths

| component | role |
|---|---|
| `lhotse.serialization.AIStoreIOBackend` | turns `s3://` / `ais://` into actual HTTP fetches via aistore SDK; gated on `AIS_ENDPOINT` env var presence |
| `nemo.collections.common.data.lhotse.indexed_adapters._AISRangeReader` | seekable file-like wrapper that translates `seek()` + `read(n)` into AIS byte-range HTTP requests; used by the indexed tar member readers when `data_path` is a URL |
| `nemo.collections.common.data.lhotse.indexed_adapters._open_data_path` | factory that returns either a regular `open(path, "rb")` for local paths or `_AISRangeReader` for URL paths |
| `lhotse.ais.batch_loader.AISBatchLoader` | minibatch-time MOSS GetBatch client; aggregates all URLs from a CutSet into one request and demultiplexes the response back into manifests |
| `lhotse.ais.batch_loader._moss_attrs` | normalizer for AIS SDK MossIn-vs-MossOut attribute differences (older `.bck` / `.provider` / `.obj_name` vs newer `.bucket_name` / `.bucket_provider` / `.object_name`); handles both transparently |

### Index building

- `submit_build_indexes.py` runs once per blend on a CPU SLURM job.
- Build reads tar files via AIS (HTTP GET with byte-range), parses tar
  headers, writes `.idx` sidecars to `<workspace>/indexes_mirror/`
  (lustre) — mirroring the data files' s3 paths.
- Successful index build proves the data IS on AIS. If indexing
  succeeds for a path but training fails to fetch via MOSS GetBatch with
  empty content, the data is replicated for individual GET but not for
  MOSS — switch the run to `USE_AIS_INDIVIDUAL_GETS=true`.

### Prefetch pipeline

1. **Indexes mirror → local SSD**: `prefetch_indexes_to_ssd.sh` copies
   `<workspace>/indexes_mirror/` to `/tmp/idx` on each node. Reads via
   `lhotse.serialization.open_best`, so source can be lustre or remote.
2. **Manifests** (optional): `prefetch_manifests_to_ssd.sh` pulls AIS
   manifests to `/tmp/manifests/` and rewrites blend YAMLs to point at
   the local copies. Only useful when `manifest_filepath` is `s3://`;
   no-op if manifests are already on lustre.
3. **HF cache** (optional): `cache_pretrained_to_ssd.sh` copies the
   pretrained LLM/ASR weights from `$HF_HOME/hub/` to local SSD to avoid
   N-rank concurrent reads from lustre at training start.

All three preambles are now run **in parallel** by `train_and_eval.py`
(each in a backgrounded subshell with PID capture; `wait` propagates any
non-zero exit). Each prefetch script is flock-guarded, so only one rank
per node does the actual work; the other 7 wait for the lock-holder to
finish.

### Container requirements

- `aistore` Python SDK (any version ≥1.18; the `_moss_attrs` normalizer
  handles MossIn↔MossOut renames in 1.19+).
- `nvidia-container-cli` on every node the build/training runs on. Some
  cpu partitions don't have it (NRT cpu partition is a known case);
  workaround is the `--bypass-nvidia-hook` flag in
  `submit_build_indexes.py`, which injects
  `--export=ALL,NVIDIA_VISIBLE_DEVICES=void` so enroot's
  `98-nvidia.sh` hook short-circuits.

### Failure modes specific to AIStore

See `references/failure-modes.md` §3 (`f.tell()` on non-seekable
ObjectFileReader), §4 (`os.path.getsize` on URL paths), §5 (`open()`
builtin on URL paths), §10 (`MossOut.bck` AttributeError), §16 (MOSS
GetBatch returns empty content for non-replicated data).

## Workflow B — non-AIStore (lustre-only)

### Required setup

- **All `tarred_audio_filepaths` resolve to local-FS paths** (typically
  `/lustre/...`).
- **`AIS_ENDPOINT` UNSET** in cluster env_vars — when present and the
  blend has any URL paths, `AISBatchLoader` would otherwise be
  instantiated and try to MOSS-fetch local-FS paths, causing confusing
  errors. Comment out the env var or use a different cluster_config
  variant.
- **`USE_AIS_GET_BATCH=false`** (the default; `--enable-indexes-prefetch`
  sets it to `true` so use a different launcher invocation, OR pass
  `--no-enable-indexes-prefetch` if your launcher exposes that, OR call
  `salm_train.py` directly without the env var set).

### Required code paths

| component | role |
|---|---|
| `lhotse.serialization.BuiltinIOBackend` | trivial `open(path, "rb")` for local files |
| `nemo.collections.common.data.lhotse.indexed_adapters._open_data_path` | falls through to `open()` for paths that don't match `_URL_RE` |
| `nemo.collections.common.data.lhotse.indexed_adapters.IndexedTarMemberReader` | regular seekable random access into local tars |
| **NOT used**: `_AISRangeReader`, `AISBatchLoader`, `aistore` SDK, MOSS GetBatch, archpath-based archive member fetch |

### Index building

- Same `submit_build_indexes.py` invocation.
- Build reads tar files via local `open(path, "rb")` (the
  `_open_data_path` factory's local branch). No HTTP, no AIS.
- Faster than the AIStore workflow per file (no network round-trip),
  but lustre I/O can be the bottleneck with high worker counts.

### Prefetch pipeline

1. **Indexes mirror → local SSD**: same `prefetch_indexes_to_ssd.sh`,
   but the source is the lustre mirror (no AIS to traverse).
2. **Manifests prefetch**: not needed (manifests are already on
   lustre).
3. **HF cache**: same as AIStore workflow.

### Container requirements

- `aistore` SDK NOT required. Container can be slim.
- `nvidia-container-cli` still required for the GPU portion (training
  itself); for the CPU-only index build, the `--bypass-nvidia-hook`
  flag still applies.

### Failure modes specific to non-AIStore

Mostly the local-FS-only failure modes of §1, §2, §6, §7, §8, §11-§15,
§17 from `references/failure-modes.md`. The AIS-specific modes (§3-§5,
§10, §16) don't fire.

## Decision tree

```
                 [is `tarred_audio_filepaths` a URL?]
                          /                 \
                        no                  yes
                        /                     \
              [non-AIStore workflow]   [is AIS_ENDPOINT set?]
                                          /          \
                                         no          yes
                                         /            \
                              [ERROR: blend uses     [AIStore workflow]
                               URLs but cluster                  \
                               doesn't expose AIS]      [does MOSS GetBatch
                                                         work for this data?]
                                                              /         \
                                                         yes              no
                                                          /                \
                                          [USE_AIS_GET_BATCH=true]   [USE_AIS_GET_BATCH=true
                                          (default)                    USE_AIS_INDIVIDUAL_GETS=true]
```

## Common gotchas in mode-switching

- **Same blend across clusters**: a blend with `s3://` paths only works
  on clusters with `AIS_ENDPOINT` configured. Maintain per-cluster
  blend variants (`data_blends/<cluster>/...`) when porting.
- **Lustre mounts identical?** Don't assume — verify with `ls` on the
  cluster login node before assuming a `/lustre/...` path resolves on a
  new cluster. NRT and IAD have similar mount roots but disjoint data
  trees.
- **`indexes_root` is shared across both workflows**. The `.idx` file
  format is identical (uint64 offsets + sentinel); the source-data
  resolution is what differs. You can re-use a mirror across an AIS
  → lustre migration as long as the blend's data file paths are
  identical strings.
