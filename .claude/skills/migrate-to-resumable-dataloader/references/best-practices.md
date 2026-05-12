# Best practices — indexed + resumable Lhotse migration

A short, prioritised checklist distilled from the failure-mode catalog and
real-world adoption pain. Apply these before sweeping any new recipe.

## Tier 1 — non-negotiable

1. **Pin BOTH `seed` and `shard_seed` to fixed integers** when
   `force_map_dataset: true` and `use_stateful_dataloader: true`. The
   sampler RNG (`shard_seed`) is checkpointed into `meta.pt` and restored
   verbatim on resume; if `shard_seed: "randomized"`, each new chunk derives
   a fresh worker-PID-hashed seed at init that diverges from the saved
   snapshot. NeMo's auto-overwrite (`dataloader.py`
   `get_lhotse_sampler_from_config`) papers over this with a warning, but
   pinning up front gives reviewers an obvious signal of intent.

2. **Same seed across every chunk of a chain.** Lightning re-seeds
   Python/torch/numpy global RNGs at chunk start using
   `data.train_ds.seed`. If your launcher rotates this per chunk
   (FIXED_SEEDS-style — what `train_and_eval.py` did historically), every
   resume silently breaks bit-exactness in dropout / augmentation /
   aux-loss permutations. The repo's `train_and_eval.py` now pins a single
   seed when `--enable-indexes-prefetch` is set; for other launchers, do
   it manually.

3. **Match `num_workers` AND `world_size` between save and restore.**
   `torchdata.StatefulDataLoader` enforces this as a hard contract. Any
   mismatch raises immediately at load. Document the values in your
   training script header so a re-submission can't accidentally drift.

4. **Build the index mirror once per blend; reuse across experiments.**
   `submit_build_indexes.py` skips already-indexed files (checks for
   non-empty `.idx`), so re-runs are cheap. Pin a stable
   `indexes_root: <workspace>/indexes_mirror` and don't move it.

5. **Set `concurrent_bucketing: false` in `data.train_ds`.** Default is
   `true`, which spawns a daemon producer thread inside
   `DynamicBucketingSampler` that races the main thread on
   `cuts_iter`. The main thread is the one `StatefulDataLoader`
   checkpoints; the producer is invisible to the snapshot. After resume
   the producer's pre-fetched cuts are lost and the per-step batch
   composition silently diverges from the non-resumed run. The
   throughput cost of the synchronous path is negligible at steady
   state; the determinism gain is non-negotiable for resumable
   training. See `failure-modes.md §19`.

## Tier 2 — strongly recommended

5. **Run the bit-exact verification from `MIGRATION_GUIDE.md` §3 before
   sweeping.** ~10 sec, model-free: take 5 batches → `state_dict` →
   take 5 more (ground-truth); fresh process loads `state_dict`, takes 5,
   asserts equal. Catches sampler/bucketer state-dict bugs that schema
   inspection of `meta.pt` (just confirming the keys exist) won't.

6. **Pick exactly ONE checkpoint trigger** in
   `exp_manager.checkpoint_callback_params` — `every_n_train_steps`,
   `every_n_epochs`, OR `train_time_interval`. Lightning's
   `ModelCheckpoint.__validate_init_configuration` raises
   `MisconfigurationException` if more than one is set. External
   preemption (cluster scheduler kills mid-chunk) doesn't go through
   NeMo's `max_time_per_run`-based PreemptionCallback, so progress
   between the last save and the kill is lost — pick whichever trigger
   matches your chunk's reachable progress: `every_n_train_steps: 50`
   if chunks reach only ~80-100 steps; `every_n_epochs: 1` if you
   reliably get full 1000-step epochs; `train_time_interval: "00:30:00"`
   if you prefer wall-clock semantics.

7. **Test 1-node single-chunk first, then 1-node multi-chunk (resume),
   then full N-node.** The 1-node smoke isolates dataloader/IO bugs from
   distributed/EP issues. The multi-chunk-on-1-node test exercises the
   resume path before scale changes. The repo's
   `nano-v3-granary1p1-en-1node-resumable.yaml` is a working template.

8. **For multilingual / non-English data on AIStore that fails MOSS
   GetBatch with "empty content"**, switch to
   `USE_AIS_INDIVIDUAL_GETS=true` (or
   `train_and_eval.py --enable-ais-individual-gets`). Slower per batch
   but bypasses the buggy MOSS path. See `references/aistore-vs-non-aistore.md`.

9. **Keep `.idx` mirror on lustre, prefetch destination on local SSD.**
   Building indexes writes to lustre (cluster-shared, persistent). The
   training preamble copies the mirror to `/tmp/idx` on each node's local
   SSD via `prefetch_indexes_to_ssd.sh` for fast mmap. Don't store the
   mirror on `/tmp` — it would be lost between jobs.

## Tier 3 — nice to have

10. **Use `--bypass-nvidia-hook`** for clusters whose cpu partition
    lacks `nvidia-container-cli` (e.g. NRT). The launcher injects
    `--export=ALL,NVIDIA_VISIBLE_DEVICES=void` so enroot's
    `98-nvidia.sh` short-circuits instead of failing the container start.

11. **`--exclusive --cpus_per_task=96`** for the index build job. The
    container's unsquashfs needs the full memory budget on first
    extraction; without exclusive, the default per-CPU memory allocation
    can OOM-kill the container before `build_indexes.py` even starts.

12. **`--workers $((cpus - 1))`** for the index ProcessPool, leaving one
    core for OS/scheduler. Indexing is I/O bound when manifests are on
    s3, but the tar-header parse is GIL-heavy (so threads serialize) —
    process pool is the right call. If you OOM, drop workers; with 96
    workers chewing big s3 manifests we've seen `BrokenProcessPool` on
    the very large all-asr blend.

13. **Drop AMI from English Granary blends until uncompressed Shar
    exists.** AMI's Lhotse Shar uses `.jsonl.gz` cuts which can't be
    indexed; either re-export with `compress_jsonl=False`, or use the
    `granary1p1-en-resumable.yaml` blend which omits AMI entirely.

14. **Run preambles in parallel.** `train_and_eval.py` now backgrounds
    each preamble (HF SSD cache / manifest prefetch / index prefetch)
    with PID capture and `wait`-with-error-propagation. Each script's
    flock guards cross-rank de-duplication, so backgrounding from each
    rank is safe.

## What NOT to do

- **Don't skip the bit-exact verification** because "schema looks right".
  Schema-only verification (presence of `_snapshot/_steps_since_snapshot/
  _iterator_finished` in `meta.pt`) confirms the StatefulDataLoader is
  being asked to checkpoint, NOT that the snapshot bytes are restored
  correctly.

- **Don't pin `aistore` SDK to an old version** "to avoid the MossOut
  bug" — the lhotse code already handles both shapes via
  `_moss_attrs`. Use the latest SDK; track future SDK churn with the
  same defensive normalizer pattern.

- **Don't combine `every_n_train_steps + every_n_epochs +
  train_time_interval`** in one `checkpoint_callback_params`. Lightning
  raises `MisconfigurationException` at startup. Pick one trigger.

- **Don't enable `concurrent_bucketing=True` with custom samplers** that
  spawn non-daemon threads. The built-in `DynamicBucketingSampler` is
  correct (background thread is `daemon=True`); only matters if you
  forked it.

- **Don't move `indexes_root` between training and prefetch.** If the
  YAML says `indexes_root: /tmp/idx` and the prefetch script writes to
  `/tmp/idx2`, training silently can't find any index, falls back to
  building on first access (slow).
