# Migration report — `<config-stem>`

- **Generated**: <YYYY-MM-DD HH:MM>
- **Source YAML**: `<path/to/source-config.yaml>`
- **Patched YAML**: `<path/to/source-config-resumable.yaml>`
- **Source blend** (if inspected): `<path/to/blend.yaml>`
- **Patched blend** (if emitted): `<path/to/blend-resumable.yaml>`
- **Launcher** (if inspected): `<path/to/launcher.py>` (or "skipped — no launcher provided")
- **Cluster**: `<cluster>` (AIStore: yes/no)

## Summary

<One paragraph: what was changed, severity counts (e.g. "1 fatal, 4 errors,
3 warnings, 2 notes"), whether the patched YAML is ready-to-launch or
requires further user action.>

## Findings

### Fatal (must fix; auto-patching not possible)

- _none_  — OR —
- **`<field-path>`** (`<file>:<line>`): <one-paragraph explanation>
  - **Current**: `<value>`
  - **Recommended**: `<value>` (or "manual rewrite")
  - **Why fatal**: <reason auto-patch isn't possible>
  - **References**: [option-reference §X], [failure-modes §Y]

### Errors (auto-patched; review the diff)

- **`data.train_ds.indexed`** (`<file>:<line>`): <description>
  - **Was**: `false` → **now**: `true`
  - **Why**: <one paragraph>
  - **References**: [option-reference §train_ds.indexed]

- _(more)_

### Warnings (auto-patched OR commented inline; verify intent)

- **`data.train_ds.shard_seed`** (`<file>:<line>`): <description>
  - **Was**: `"randomized"` → **now**: `42`
  - **Why**: NeMo `dataloader.py` would auto-overwrite at runtime with a
    `WARNING` log; pinning at the YAML layer makes intent obvious to
    reviewers and avoids the runtime warning.
  - **References**: [conflict-matrix row 3], [failure-modes §11]

- _(more)_

### Notes (informational; no patch)

- **`data.validation_ds.use_stateful_dataloader`** (`<file>:<line>`):
  Not strictly required for validation (eval doesn't checkpoint), but
  setting it `false` matches the working 1-node smoke recipe. No change
  needed.

- _(more)_

## Cross-cuts

### Data blend audit

<Drop in `references/failure-modes.md` §1 / §2 callouts: blend entries with
`.jsonl.gz`, `.tar.gz`, `extra_fields`, `slice_length`. List the entries
that were removed from the patched blend and the per-entry rationale.>

| corpus | reason for exclusion | upstream fix |
|---|---|---|
| ami | `cuts: *.jsonl.gz` (compressed Lhotse Shar) | re-export with `compress_jsonl=False` OR convert to `nemo_tarred` |
| _(more)_ | _(more)_ | _(more)_ |

### Launcher review

<If launcher script provided: list grep findings. Otherwise: "skipped".>

- **Per-chunk seed rotation**: <not detected | DETECTED at `<file>:<line>` —
  the launcher pulls from a FIXED_SEEDS-like array; this MUST be pinned
  to a single value when the resumable path is on. See
  `failure-modes.md §12`. Manual fix required.>
- **Prefetch preamble wired**: <yes / NO — `--enable-indexes-prefetch`
  flag not set; manual addition needed. See `option-reference.md §launcher
  flags`.>
- **`--bypass-nvidia-hook`**: <not needed | needed for `<cluster>` cpu
  partition — see `failure-modes.md §9`>

### AIStore vs lustre

<One paragraph: which workflow this migration follows (per
`aistore-vs-non-aistore.md` decision tree), and any cross-cluster
caveats.>

## Patched output diff

### `<config>.yaml` → `<config>-resumable.yaml`

```diff
-  data.train_ds:
-    indexed: false
-    use_stateful_dataloader: false
-    shard_seed: "randomized"
+  data.train_ds:
+    indexed: true
+    use_stateful_dataloader: true
+    force_map_dataset: true
+    indexes_root: /tmp/idx
+    shard_seed: 42  # NOTE: pinned for StatefulDataLoader resume; see
+                    # MIGRATION_GUIDE.md §"Operational constraints"
```

_(full diff inline)_

### `<blend>.yaml` → `<blend>-resumable.yaml`

```diff
-  - corpus: ami
-    shar_path:
-      cuts: s3://AMI/lhotse_shar/cuts._OP_*_CL_.jsonl.gz
-    type: lhotse_shar
-    weight: 0.2
+  # AMI dropped — Lhotse Shar `cuts.*.jsonl.gz` cannot be indexed
+  # (uncompressed sources only). Re-export with `compress_jsonl=False`
+  # or convert to `nemo_tarred` to re-include.
```

_(full diff inline)_

## Pre-flight checklist

See `pre-flight-checklist.md` next to this report. The TL;DR:

1. Build indexes via the generated `build-indexes-cmd.sh`.
2. Run the `MIGRATION_GUIDE.md §3` bit-exact verification once on this
   recipe.
3. Confirm `aistore` SDK present in the container (AIStore workflow only).
4. 1-node single-chunk → 1-node multi-chunk → full N-node smoke ladder.
5. Submit the real run.

## References

- `MIGRATION_GUIDE.md` (repo root): canonical migration walkthrough.
- `references/option-reference.md`: every YAML field, every flag.
- `references/conflict-matrix.md`: option pairs that conflict.
- `references/failure-modes.md`: 18-entry failure-mode catalog.
- `references/best-practices.md`: prioritised checklist.
- `references/aistore-vs-non-aistore.md`: workflow selection.
