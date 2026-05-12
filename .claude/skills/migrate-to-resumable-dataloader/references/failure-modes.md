# Failure-mode catalog

Every failure mode observed during the speechlm-2026h1 migration to indexed
+ resumable dataloading. Each entry: **signature** (what you grep for in
logs), **trigger** (the YAML/launcher condition that produces it),
**fix**, and **see-also** pointers.

## §1 — `.jsonl.gz` AMI shar in blend

**Signature**: index build fails with
`ValueError: <ctx> requires uncompressed JSONL or tar data, but got a compressed path: <file>.jsonl.gz`
from `lhotse/indexing.py:130-135`.

**Trigger**: blend YAML references AMI's stock distribution (Lhotse Shar
with `cuts.*.jsonl.gz`).

**Fix**: drop AMI from the blend until an uncompressed Shar export (or a
`nemo_tarred` re-export) is available. The repo's
`data_blends/iad/granary1p1-en-resumable.yaml` does exactly this — see its
header comment.

## §2 — `extra_fields` / `slice_length` on `nemo_tarred` entry

**Signature**:
`RuntimeError: LazyNeMoTarredIterator(indexed=True) does not support 'extra_fields' because <ctx>` from `nemo_adapters.py:485-487`,
or
`RuntimeError: LazyNeMoIterator(indexed=True) does not support 'extra_fields'` from `nemo_adapters.py:148-152`.

**Trigger**: blend entry has `extra_fields:` block (typically attaching
text-iter / text-sample / graph-token features to a `nemo` /
`nemo_tarred`) or `slice_length: N`.

**Fix**: pre-process the manifest offline to materialize the extra fields
into the manifest, then drop the `extra_fields` key. For `slice_length`,
re-shard the audio to the target slice and drop the key.

## §3 — `f.tell()` on AIStore `ObjectFileReader`

**Signature**: `io.UnsupportedOperation: seek` on first read of an
`ais://` / `s3://` tar source.

**Trigger**: AIStore SDK's `ObjectFileReader` doesn't implement
`tell()` / `seek()`. The indexer uses `_CountingReader` to accumulate bytes
manually; if your code path bypasses that, this fires.

**Fix**: ensure the `aistore` SDK is installed in the container so lhotse
routes via `AIStoreIOBackend`. The indexer's `create_jsonl_index` /
`create_tar_index` accumulate bytes via `len(line)` and `_CountingReader`
in `lhotse/indexing.py`. `submit_build_indexes.py:227` does the SDK install
preamble.

## §4 — `os.path.getsize(s3://…)`

**Signature**: `FileNotFoundError: [Errno 2] No such file or directory: 's3://...'`

**Trigger**: legacy code path computing index file size from disk for an
`s3://` URL.

**Fix**: `IndexedJsonlReader._load_index` / `IndexedTarMemberReader._load_index`
now read the size sentinel from the `.idx` file itself for URL paths.
Confirmed at `NeMo_resumable/nemo/collections/common/data/lhotse/indexed_adapters.py:269-294`
(uses `np.fromfile` with `<u8` dtype; final entry is the file-size
sentinel).

## §5 — `open(s3://…)` in tar member readers

**Signature**: `FileNotFoundError: [Errno 2] No such file or directory: 's3://...'`
on first audio fetch.

**Trigger**: `IndexedTarMemberReader` calling stdlib `open()` instead of
the AIS-aware reader on a remote tar.

**Fix**: `_open_data_path` at `indexed_adapters.py:159-166` returns
`_AISRangeReader(str(path))` for any path with a `://` scheme. The
`_AISRangeReader` translates `seek + read` into AIStore HTTP range requests.

## §6 — `np.memmap` exhausts `vm.max_map_count`

**Signature**: `OSError: [Errno 12] Cannot allocate memory` during
training startup, with 80k+ shards.

**Trigger**: legacy `np.memmap` per `.idx` file. With
`vm.max_map_count = 65530` (Linux default), 80k shards × 1 mmap each
exceeds the limit.

**Fix**: switched to `np.fromfile` (resident array). Indexes are tiny
(KB-scale per shard), so the memory cost is negligible. Confirmed at
`indexed_adapters.py:288-294` ("Use np.fromfile (resident memory) rather
than np.memmap so that NeMo blends with 80k+ shards don't exhaust
vm.max_map_count").

## §7 — Validation manifest with `.json` extension

**Signature**: `ValueError: <ctx> path is not indexable: <file>.json`
from `validate_indexed_access`.

**Trigger**: NeMo convention to ship some manifests as `.json` (one JSON
object per line) rather than `.jsonl`. The first version of
`indexed_path_kind` rejected `.json`.

**Fix**: `lhotse/indexing.py:99-107` now accepts both `.jsonl` and
`.json` since the indexer only relies on newline-separated records.

## §8 — ProcessPool OOM during indexing

**Signature**: `concurrent.futures.process.BrokenProcessPool: A process in the process pool was terminated abruptly`
in the build-indexes log, often after several minutes of forward progress.

**Trigger**: 95 workers on a 96-cpu node + huge S3 manifests + Granary 1.1
audio tars. The forks each load a manifest + tar header into RAM; with
176 GiB total and 95 workers, peak per-worker RAM crosses ~1.8 GiB and
the kernel OOM-killer fires.

**Fix**: drop to `--workers 48`, or split the blend across multiple
build-indexes invocations. `submit_build_indexes.py:99-104` defaults
`cpus_per_task=96`; the auto-effective worker count is `cpus_per_task - 1`
= 95. Override with `--workers 48`.

## §9 — Container `nvidia-container-cli` missing on cpu partition

**Signature**: enroot's `98-nvidia.sh` hook hard-fails container start;
sbatch.log shows `nvidia-container-cli: command not found` or similar.

**Trigger**: NRT cluster's `cpu` / `cpu_interactive` / `cpu_datamover`
partitions lack `nvidia-container-cli`. IAD's cpu partition has it.

**Fix**: pass `--bypass-nvidia-hook` to `submit_build_indexes.py`
(`:122-129, 240-245`). Sets `--export=ALL,NVIDIA_VISIBLE_DEVICES=void` on
the sbatch line, which makes enroot's hook short-circuit.

## §10 — AIStore SDK `MossOut.bck` AttributeError

**Signature**:
`AttributeError: 'MossOut' object has no attribute 'bck'` in the empty-
content retry path of `AISBatchLoader.__call__`. Cascade:
`Error collating conversations: 'MossOut' object has no attribute 'bck'`,
then `FallbackDataset received None`, then
`TypeError: 'NoneType' object is not subscriptable` in
`salm_automodel.training_step`, then DeepEP's `'unspecified launch failure'`.

**Trigger**: `aistore>=1.20` (we're on 1.23.0) renames the MossIn-shaped
`info.bck/.provider/.obj_name/.archpath` → MossOut-shaped
`info.bucket_name/.bucket_provider/.obj_name/.archpath`. Triggered when
the underlying object is missing on AIS and the SDK returns 200 + empty
body, kicking the retry path that then crashes on attribute access.

**Fix**: `_moss_attrs` normalizer at
`lhotse_resumable/lhotse/ais/batch_loader.py:81` returns a 4-tuple
`(bck, provider, obj_name, archpath)` for both shapes. Every consumer site
must use it; raw `info.bck` references are bugs.

**See also**: `agent-debug-workspace/0909-multiling-failures.md` for the
full causal chain (multilingual Granary 1.1 audio not on iad AIS → empty
content → retry path crash).

## §11 — `shard_seed: "randomized"` + `force_map_dataset: true` + `use_stateful_dataloader: true`

**Signature**: silent — no crash. Each fork re-derives a worker-PID-hashed
seed at `worker_init_fn` time, but `StatefulDataLoader.load_state_dict`
overrides the sampler state from the checkpoint. The mismatch produces
non-bit-exact resume at the data level (within the saved snapshot
window).

**Trigger**: `shard_seed: "randomized"` literal in YAML, paired with
`force_map_dataset: true` + `use_stateful_dataloader: true`.

**Fix**: pin `shard_seed: <int>` (typically equal to `seed`).
**NeMo's `dataloader.py:556-572` now warns + auto-overwrites** with the
`seed` integer, so this is a safety net; explicit pinning in YAML keeps
the rationale visible.

## §12 — Per-chunk seed rotation in launcher

**Signature**: silent (and worse than §11). On each chunk, Lightning
calls `pl.seed_everything(run_seed)`, re-seeding Python/numpy/torch global
RNG with a different value. Dropout, aux-loss, model random-init RNG draws
diverge across chunks. The data-iteration level is correct (StatefulDataLoader
wins the seed race for sampler state); the model level is not.

**Trigger**: `train_and_eval.py`'s `FIXED_SEEDS[seed_offset+i]` rotation
(pre-fix), or any launcher that picks a fresh seed per chunk
(`seed = randint(...)`, `seed = run_idx`, etc.).

**Fix**: pin a single seed for the entire chain. `train_and_eval.py:925-952`
now does this when `--enable-indexes-prefetch` is set:
`invariant_seed = seed if seed is not None else FIXED_SEEDS[seed_offset]`,
and all chunks use `invariant_seed`. For arbitrary launchers, grep for
seed-per-chunk patterns and warn.

**See also**: `agent-debug-workspace/0909-longform-failures.md` Cause A
(the original investigation).

## §13 — `every_n_epochs: 1` only, no `every_n_train_steps`

**Signature**: visible — only `step=N.ckpt` (where N is one
`limit_train_batches`-aligned boundary) on disk after many hours of
compute, with the rest of the chain producing no new checkpoints.

**Trigger**: `checkpoint_callback_params.every_n_train_steps: null` AND
`every_n_epochs: 1`. With `limit_train_batches: 1000`, 1 epoch = 1000
steps. If chunks get preempted at ~1h before reaching the next 1000-step
boundary, NO save happens (the preemption callback's `step=N-last.ckpt` is
the only fallback).

**Fix**: add `every_n_train_steps: 50-250` (and/or
`train_time_interval: "00:30:00"`). Lightning ORs the triggers, so all
three can coexist.

**See also**: `agent-debug-workspace/0909-longform-failures.md` Cause B.

## §14 — `max_time_per_run` doesn't fire on external SIGTERM

**Signature**: SLURM SIGTERM kills the job; no extra `step=N-last.ckpt`
written; chunk progresses through 75–150 steps but loses them all on
restart.

**Trigger**: any external preemption (`svc-hwinf-cs-sched`, NODE_FAIL, QOS
preemption, manual cancel). NeMo's `PreemptionCallback` fires only on its
own internal timer (`max_time_per_run`).

**Fix**: doesn't fix the root issue; mitigated by frequent step/time-based
saves (§13). Set `max_time_per_run` to `<SLURM walltime - 10min>` to keep
the internal-timer save before SLURM SIGKILLs the teardown.

## §15 — `num_workers` mismatch on resume

**Signature**: torchdata `StatefulDataLoader` raises a hard error at
`load_state_dict` time, complaining the snapshot has different
`num_workers`.

**Trigger**: chain config changes `num_workers` between chunks (e.g. saved
under `num_workers: 4`, restored with `num_workers: 8`).

**Fix**: keep `num_workers` invariant across the chain. Same rule for
`world_size` (= `num_nodes * devices_per_node`).

## §16 — AIS MOSS GetBatch returns empty content for non-replicated data

**Signature**: `_inject_data_into_manifest` retries with empty content; on
old SDK shape (`info.bck`) crashes with §10 AttributeError. With the §10
patch in place: `Error collating conversations: <object>/<archpath> from
bucket <provider>://<bck> returned empty content` → `FallbackDataset
received None` → `TypeError: 'NoneType' object is not subscriptable`.

**Trigger**: data path is `s3://FLEURS/tarred/<lang>/...`,
`s3://MCV/MCV4/.../<lang>/...`, etc., and the cluster's AIS doesn't have
that data replicated. Confirmed for non-EN multilingual on IAD AIS as of
2026-05-09.

**Fix (workaround)**: set `USE_AIS_INDIVIDUAL_GETS=true` to bypass MOSS
GetBatch and use per-object `Object.get_reader(archive_config=...).read_all()`
(slower but works). `lhotse_resumable/lhotse/ais/batch_loader.py:67, 218`
implements `prefer_individual=True`.

**Fix (proper)**: replicate the missing data to AIS. Quick check from
inside an iad container:
```python
from aistore import Client
import os
c = Client(os.environ["AIS_ENDPOINT"])
for url in ["s3://FLEURS/tarred/bg/audio_0.tar"]:
    try:
        print(url, c.get_object_from_url(url).head_v2().size)
    except Exception as e:
        print(url, "MISSING:", e)
```

**See also**: `agent-debug-workspace/0909-multiling-failures.md`.

### Open investigation

**Why does MOSS GetBatch return empty content for non-EN multilingual
data?** Most likely answer: data not replicated to AIS. But worth
confirming via the head_v2 probe above before adopting
`USE_AIS_INDIVIDUAL_GETS` as the permanent workaround. If the data IS on
AIS but unreadable for some other reason, a different fix is needed (and
the lhotse fallback would benefit from raising an explicit
`AISBatchLoaderError: object missing on AIS` instead of letting the
empty-content path lead to a `TypeError` 6 frames down).

## §17 — `indexes_root` mismatch between training YAML and prefetch script

**Signature**: training startup fails with
`FileNotFoundError: <path>/<manifest>.idx` or, in the indexed adapter,
`ValueError: ... .idx file not found ...` from `IndexedJsonlReader._load_index`.

**Trigger**: `data.train_ds.indexes_root: /tmp/idx` in YAML but
`prefetch_indexes_to_ssd.sh` writes to `/scratch/idx`, or vice versa.

**Fix**: keep both in sync. In `submit_build_indexes.py` the mirror
defaults to `<workspace>/indexes_mirror/`; the prefetch script then pulls
onto each node's `/tmp/idx` (the default is `/tmp/idx` per
`prefetch_indexes_to_ssd.sh`). The training YAML's `indexes_root` must
match the prefetch destination.

## §19 — `concurrent_bucketing: true` (default) breaks resume bit-exactness

**Signature**: silent. Loss curves and per-sample order across resume
boundaries diverge from a single-run reference; no exception fires.
Spot-check by saving `state_dict` mid-run, restoring in a fresh
process, and asserting batches 0..K are bit-identical (the
`MIGRATION_GUIDE.md` §3 recipe). Without the fix, you'll see byte-level
mismatches starting from the very first restored batch.

**Trigger**: any resumable training run with `force_map_dataset: true`
and `use_stateful_dataloader: true` but `concurrent_bucketing` left at
its default `True`.

**Cause**: `DynamicBucketingSampler` spawns a daemon producer thread
(`lhotse_resumable/lhotse/dataset/sampling/dynamic_bucketing.py:924-944`)
that pre-pulls cuts from `self.cuts_iter` into per-bucket queues. The
main thread is the one `StatefulDataLoader` checkpoints; the producer
operates concurrently. At `state_dict` time, the saved cursor reflects
the main thread's position, NOT the producer's pre-fetched cuts. On
resume the producer is gone; its pre-fetched cuts are lost. The
bucketing decisions and per-step batch composition diverge from the
non-resumed run. As a side effect the same config is also not
bit-reproducible between two fresh runs (producer scheduling is
OS-dependent).

**Fix**: set `concurrent_bucketing: false` in `data.train_ds`. NeMo
falls through to the synchronous `_collect_cuts_in_buckets` path
(same file, `:954-965`) which advances the iterator only from the
main thread. Slight throughput hit during bucket warm-up; negligible
in steady state since the bucket buffer is normally well-stocked.

**Cross-refs**: `option-reference.md` `data.train_ds.concurrent_bucketing`
row; `best-practices.md` Tier 1.

---

## §18 — `prefetch_indexes.py` PYTHONPATH

**Signature**: `ImportError: cannot import name 'create_jsonl_index'` or
`ModuleNotFoundError: No module named 'lhotse.indexing'` — the
container's stock `lhotse` lacks the resumable extensions.

**Trigger**: prefetch / build_indexes preamble doesn't prepend
`lhotse_resumable/` and `NeMo_resumable/` to PYTHONPATH.

**Fix**: `submit_build_indexes.py:225` does
`export PYTHONPATH={lhotse_remote}:{code_dir}:$PYTHONPATH` before invoking
`build_indexes.py`. Arbitrary launchers must do the same.

---

## Cascading symptoms (NOT root causes)

Distributed failures cascade — one bad rank's exception triggers a NCCL
timeout 30 min later that kills the rest. When the loud error is one of:

- `EPException what(): 'unspecified launch failure'` at `deep_ep.cpp:155`
- `DeepEP timeout check failed: rank=X, thread=Y, value=…`
- `Watchdog caught collective operation timeout: WorkNCCL(...)`

…look upstream for the Python traceback that fired first. The DeepEP /
NCCL chatter is cascade. The 0909-multiling chains had this exact
pattern: `TypeError: 'NoneType' object is not subscriptable` (origin) →
DeepEP `'unspecified launch failure'` (cascade).
