---
name: migrate-to-resumable-dataloader
description: This skill should be used when the user asks to "migrate to the resumable dataloader", "switch to indexed Lhotse", "adopt the indexed + resumable pipeline", "make my training resumable", "set up StatefulDataLoader for SALM", "use AIStore GetBatch", or "convert this YAML to the resumable path". Walks a NeMo training YAML (and optional launcher / blend / cluster info) through the indexed + resumable Lhotse migration; lints every interacting field, auto-patches the YAML and any blends, emits a migration report and a pre-flight checklist, and produces a one-shot `submit_build_indexes.py` invocation. Static analysis only; never runs jobs.
argument-hint: '<config.yaml> [launcher.py] [blend.yaml] [--cluster=<name>]'
---

# Migrate a NeMo training YAML to the indexed + resumable Lhotse dataloader

The repo's resumable path (replacing the streaming/replay loader with O(1)
checkpoint-restore via `torchdata.StatefulDataLoader` + `.idx` sidecars) has
~20 distinct ways to silently corrupt or hard-fail. This skill runs every
one of those checks against a concrete YAML, auto-patches what it can, and
emits a teaching-style migration report so the user understands every
decision and the user-only steps to run before launching.

**Map-style vs iterable-style for indexed sources.** The resumable path supports
two dedup modes:

1. **`force_map_dataset: true`** (default; safest) — sampler runs in the main
   GPU process and over-samples `world_size` batches per step, discards
   `world_size - 1`. Works for any source type. Costs `W×` redundant
   sampler/manifest I/O per step.
2. **`force_map_dataset: false`** (optimization for indexed-only configs at
   high `world_size`) — sampler runs co-located with the dataset inside CPU
   worker subprocesses; sample indices are partitioned across
   `(DP rank × DataLoader worker)` via `LazyShuffledRange(shard_id, num_shards)`
   so each shard yields a disjoint slice. Resolved at iteration time via the
   `LHOTSE_USE_WORKER_PARTITION` env-var signal that `worker_init_fn` sets.
   Eliminates the `W×` redundant work; near-`W×` step-time improvement at
   scale. **Requires all sources to be indexed** (or use other dedup
   mechanisms — see `references/failure-modes.md` §20-§23).

## When to apply

Trigger phrases listed in the frontmatter. Three common entry modes:

1. **New migration**: user points at an experiment YAML and asks to migrate.
   Walk every field, write patched YAML + report + pre-flight + build-indexes
   command.
2. **Sanity-check existing migration**: user says "audit this YAML, is it
   resumable-correct?". Same workflow but emit only the report (no patched
   files unless errors found).
3. **AIStore-aware variant**: cluster has `AIS_ENDPOINT` and the blend has
   `s3://` / `ais://` / `http(s)://` paths. Skill switches to the AIStore
   workflow (sets `USE_AIS_GET_BATCH=true`, optionally
   `USE_AIS_INDIVIDUAL_GETS=true`, requires `aistore` SDK in container).

## Inputs

| input | required | source | purpose |
|---|---|---|---|
| Training YAML | yes | argument or `--config=` | every `data.train_ds` / `data.validation_ds` / `model` / `trainer` / `exp_manager` field that interacts with the resumable path |
| Launcher script | no | argument or auto-detect (`train_and_eval.py`, `pretrain.sh`, raw `python salm_train.py …`, `torchrun …`) | grep for per-chunk seed rotation, missing prefetch preamble, etc. If absent, skill emits "launcher review SKIPPED — manual review required" with the things to check by hand |
| Data-blend YAML | no | resolved from `data.train_ds.input_cfg` if it references `${data_blend_dir}/...` | walked for unindexable entries (`extra_fields`, `slice_length`, `.jsonl.gz`, `.tar.gz`, AMI Shar) |
| Cluster name | no | `--cluster=<name>` or detected from `data_blend_dir` path | reads `cluster_configs/<cluster>.yaml` env_vars to detect AIS_ENDPOINT and pick the right code paths |

## Outputs

Every output lands in a fresh directory `migrate-resumable/<config-stem>/`
in the repo root (so multiple migrations stay self-contained):

| output | purpose |
|---|---|
| `migration-report.md` | exhaustive walkthrough — every field touched, every option explained, every pitfall checked, severity-classified findings, links into MIGRATION_GUIDE.md and codebase |
| `<config-stem>-resumable.yaml` | the patched config, ready to drop in. Preserves comments where possible; explicit `# NOTE:` block at every changed line citing the rationale |
| `<blend-stem>-resumable.yaml` | patched blend (drops unindexable entries with rationale comments) — only emitted when blend was inspected |
| `pre-flight-checklist.md` | manual steps before launch: build indexes, verify SDK, verify cluster fits the workflow, etc. |
| `build-indexes-cmd.sh` | concrete one-shot shell command invoking `submit_build_indexes.py` (or a generic equivalent if the repo doesn't have it) |

## Workflow

### 1. Discover and parse inputs

1. Resolve the training YAML path. Read it with OmegaConf.
2. If `data.train_ds.input_cfg` references `${data_blend_dir}/<file>.yaml`,
   try to resolve `data_blend_dir` from the YAML's top-level scalar. Locate
   the blend on disk (try `data_blends/<cluster>/<file>.yaml` first, fall
   back to glob across `data_blends/`).
3. If a launcher path was given, read it as text. Otherwise inspect the
   repo root for any of `train_and_eval.py` / `pretrain.sh` / `salm_train.sh`
   and pick the most-likely match.
4. If `--cluster=<name>` was passed, read `cluster_configs/<name>.yaml`.
   Otherwise grep `data_blend_dir` for a known cluster path prefix
   (`/lustre/fsw/portfolios/llmservice/users/...` → iad, `nrt`, `ord`).
5. Detect AIStore presence: cluster `env_vars` contains `AIS_ENDPOINT=...`,
   AND the blend has `s3://` or `ais://` or `http(s)://` paths in
   `tarred_audio_filepaths` / `manifest_filepath` / `cuts_path`.

### 2. Run lint pipeline

Run every check in `references/option-reference.md`,
`references/conflict-matrix.md`, and `references/failure-modes.md` against
the YAML and (when present) the blend and launcher. Each check emits a
finding entry:

```
{
  severity: fatal | error | warning | note,
  field:    "data.train_ds.shard_seed",
  current:  "randomized",
  recommended: 42,
  rationale: <one paragraph explaining the *why*>,
  link:     "MIGRATION_GUIDE.md §… / file:line",
}
```

Severities:
- **fatal** — auto-patch impossible (e.g. blend uses `extra_fields` on a
  `nemo_tarred` entry). Skill exits non-zero with explanation; user must
  pre-process the manifest offline.
- **error** — auto-patch produces a working config (e.g.
  `shard_seed: "randomized"` → fixed integer). Skill applies the patch.
- **warning** — auto-patch optional or context-dependent; emitted as a
  comment in the patched YAML and a section in the report.
- **note** — informational; report-only.

### 3. Emit patched YAML + blend

Apply every `error`-severity patch. For each, leave a `# NOTE:` comment
above the changed line citing the finding. Comment-preserving YAML round-trip
uses `ruamel.yaml`; if that's not available, fall back to `omegaconf`
serialization (loses comments, but the report still documents every change
in detail).

For the blend: drop every entry that fails the indexability checks. Each
dropped entry leaves a `# DROPPED: <rationale>` block in its place. If the
drop empties out the blend or removes a domain entirely, surface that in
the report.

### 4. Generate `migration-report.md`

Use `templates/migration-report.md`. Sections:

1. **Summary** — one paragraph: AIStore vs non-AIStore, count of changes,
   any fatal blockers.
2. **Inputs** — paths to training YAML / launcher / blend / cluster config
   that were inspected.
3. **Findings table** — every finding with severity / field / current /
   recommended / link.
4. **Per-section walkthrough** — `data.train_ds`, `data.validation_ds`,
   `exp_manager`, `trainer`, AIStore-specific section. For each, the table
   from `references/option-reference.md` filtered to fields that exist in
   this YAML, with current vs. recommended values inline.
5. **Pitfalls / failure modes encountered** — link to
   `references/failure-modes.md` for each that fired.
6. **Conflict matrix** — link to `references/conflict-matrix.md` and call
   out any conflicts found.
7. **Best practices reminder** — copy of `references/best-practices.md`.
8. **Verification recipe** — the bit-exact verification snippet from
   `MIGRATION_GUIDE.md` §3, with the user's actual config path filled in.

### 5. Generate `pre-flight-checklist.md`

Use `templates/pre-flight-checklist.md`. Required steps:

- Build indexes via `submit_build_indexes.py` (or generic equivalent);
  print the exact command.
- If AIStore in play: verify `aistore` SDK ≥ 1.17 in the container; verify
  `AIS_ENDPOINT` is set; warn about the MOSS GetBatch issue for
  multilingual / non-EN-replicated data and recommend
  `USE_AIS_INDIVIDUAL_GETS=true` for those.
- If launcher absent or only a stub: list the launcher items the user must
  hand-check (single-seed across chain, prefetch preamble, num_workers /
  world_size invariance).
- Recommend running the bit-exact verification snippet from MIGRATION_GUIDE
  §3 once before sweeping.
- Recommend the 1-node single-chunk → 1-node multi-chunk → 4-node test
  sequence.

### 6. Generate `build-indexes-cmd.sh`

Single executable shell file with the exact `submit_build_indexes.py`
invocation, using:
- `--cluster=<detected or user-supplied cluster>`
- `--blend=<every blend referenced from the training YAML>` (training +
  validation blends)
- `--bypass-nvidia-hook` if cluster is NRT or any cluster whose `cpu_partition`
  is documented to lack `nvidia-container-cli`
- A comment block at the top with the rationale for each flag

If the repo doesn't have `submit_build_indexes.py`, emit a generic
equivalent that does:
```bash
python <NeMo>/scripts/dataloading/build_indexes.py \
    --indexes-root <local-mirror> \
    --workers <effective> \
    <blend>.yaml <validation-blend>.yaml
```
plus a SLURM wrapper sketch and call out that `submit_build_indexes.py` in
the speechlm-2026h1 repo is the canonical version.

### 7. Print final summary to chat

Short recap (under 10 lines): output dir, count of fatal/error/warning/note
findings, link to migration report, the next single command the user
should run (`bash migrate-resumable/<stem>/build-indexes-cmd.sh` then the
launcher).

## Knowledge base — references baked into this skill

- **`references/option-reference.md`** — exhaustive field-by-field table
  (every YAML key that interacts with the resumable path, required value,
  rationale, see-also link). Read this for every finding.
- **`references/failure-modes.md`** — 18 catalogued failure modes with log
  signatures, tracebacks, and fixes. Plus an "Open investigation" section.
- **`references/conflict-matrix.md`** — the option pairs that don't work
  together and what to do about each.
- **`references/best-practices.md`** — distilled checklist (priority-ordered).
- **`references/aistore-vs-non-aistore.md`** — the two parallel workflows.

- **`examples/iad-english-granary/`** — IAD English training (Granary 1.1,
  lustre manifests, S3 tars, AIStore). Before/after pair.
- **`examples/nrt-lustre-only/`** — NRT lustre-only training (no AIStore).
  Before/after pair, includes the `--bypass-nvidia-hook` build-index
  invocation.
- **`examples/multilingual-mixed/`** — multilingual blend with mixed
  S3/lustre. Demonstrates `USE_AIS_INDIVIDUAL_GETS=true` and the
  AMI-Shar-drop pattern.

- **`templates/migration-report.md`** — output template, fill-in-the-blank.
- **`templates/pre-flight-checklist.md`** — output template.

- **`scripts/analyze.py`** — the analysis engine. Reads YAML, runs every
  lint check, emits findings + writes patched YAML. Pure static analysis;
  no cluster calls.

## Constraints

- **Read MIGRATION_GUIDE.md** at `/Users/pzelasko/canary-dev/speechlm-2026h1/MIGRATION_GUIDE.md`
  in full before running. The references in this skill cite specific
  sections of that doc.
- **Cross-check against the actual code** at:
  - `lhotse_resumable/lhotse/serialization.py` (`open_best`, AIStore backend, MSC backend)
  - `lhotse_resumable/lhotse/indexing.py` (`create_jsonl_index`, `create_tar_index`, `indexed_path_kind`, `IndexedJsonlReader`, `read_index`, `LazyShuffledRange` with `(shard_id, num_shards)` partition)
  - `lhotse_resumable/lhotse/lazy.py` (`LazyIndexedManifestIterator.__iter__` defers `LazyShuffledRange` construction to resolve partition at iter time; `LazyIteratorChain._iter_globally_shuffled` partitions the combined range; `LazyIteratorMultiplexer.__iter__` rejects `seed='randomized'` under multi-shard partition)
  - `lhotse_resumable/lhotse/dataset/dataloading.py` (`worker_init_fn` sets the `LHOTSE_USE_WORKER_PARTITION` signal; `get_worker_partition()` returns the trivial `(0, 1)` when that signal is absent — keeps map-style mode unaffected even under torchrun)
  - `lhotse_resumable/lhotse/ais/batch_loader.py` (`AISBatchLoader`, `force_individual`, byte-range `shar_ptr` fallback, `_moss_attrs`)
  - `lhotse_resumable/lhotse/dataset/input_strategies.py` (`AudioSamples`)
  - `NeMo_resumable/nemo/collections/common/data/lhotse/indexed_adapters.py` (`IndexedTarMemberReader`, `_AISRangeReader`, `_CountingReader`, `_open_data_path`, `_load_index`, `resolve_idx_path`)
  - `NeMo_resumable/nemo/collections/common/data/lhotse/dataloader.py` (`get_lhotse_sampler_from_config`, `get_lhotse_dataloader_from_config`, `force_map_dataset` handling, the auto-overwrite of `shard_seed`, `_maybe_init_main_process_for_iterable` for `num_workers=0` eager `worker_init_fn` call)
  - `NeMo_resumable/nemo/collections/common/data/lhotse/nemo_adapters.py` (`LazyNeMoTarredIterator`, `_init_indexed`, `_iter_batch_for_ais_get_batch`, `USE_AIS_GET_BATCH` gate)
  - `NeMo_resumable/scripts/dataloading/build_indexes.py` and `prefetch_indexes.py`
  - `lhotse_resumable/test/test_partition.py` (49 tests pinning every partition edge case: map-style regression, empty/tiny manifests, composition with shuffler/mapper/filter/repeater, multiplexer state-dict roundtrip, chain topology mismatch, etc.)
- **Cross-check against today's debug docs** at:
  - `agent-debug-workspace/0909-summary.md`
  - `agent-debug-workspace/0909-multiling-failures.md`
  - `agent-debug-workspace/0909-longform-failures.md`
  - `agent-debug-workspace/nano-v3-1node-resumable-tests.md`
  These contain the freshest evidence-based knowledge. Cite line:file
  pointers when emitting findings whose rationale traces back to them.
- **Mention but do not duplicate** the existing `submit_build_indexes.py`
  in the speechlm-2026h1 repo; this skill references it as the canonical
  builder for that repo and provides a generic equivalent for users on
  other repos.
- **Don't write code that runs jobs on the cluster.** Static-analysis +
  migration tool, not a job runner.
- **Identify gaps clearly.** If something is unknown (e.g., why MOSS
  GetBatch returns empty for multilingual data), say so explicitly in
  `failure-modes.md` under "Open investigation" and surface that in the
  report when relevant.

## Non-goals

- Do not run `submit_build_indexes.py` automatically; emit the command and
  let the user invoke it.
- Do not modify upstream code (NeMo_resumable / lhotse_resumable). The
  skill works around upstream bugs via YAML / env-var settings.
- Do not invent fields the user didn't ask about. If a value is ambiguous
  (e.g. `seed` was unset and there's no default we can read), prompt with
  one batched `AskUserQuestion`.

## Style

Match the tone of `hyperparam-sweep/SKILL.md` and `debug-cluster-run/SKILL.md`.
Crisp, evidence-based, no fluff. Inline rationale at every decision. The
skill is a teaching tool as well as an automated migrator — every patched
line should land with a citation the user can verify.
