# Finite direct-Lhotse SpeechLM2 DPO

`salm_dpo_train.py` is a normal NeMo-Speech Lightning entrypoint for an
explicitly finite, same-audio ASR preference trajectory.  It is designed to
make the data, model, update, and checkpoint contracts visible in review; it
does not reuse experiment-local launchers, import hooks, runtime patching, or
an adapter/updater outside the owning package.

## Hero2 historical-r5 replay contract

`conf/salm_dpo_hero2_ami_historical_r5.yaml` expresses the historical
trajectory as inputs rather than hard-coded program state:

- strict source DCP: Hero2 step 14400;
- the configured ASR archive is used only to construct the perception schema;
  `init_from_checkpoint` is disabled and the strict Hero2 step-14400 DCP
  immediately replaces all temporary construction weights before reference
  capture or an update;
- one direct finite Lhotse JSONL of 4,350 preference cuts;
- ten ordered source shards of 435 pairs and two explicit passes (20 updates);
- no shuffle and no cycling; every rank reads only its assigned audio slots;
- same-audio `chosen`/`rejected` pairs, bound to the one direct-Lhotse
  recording whose ID and staged filename equal the manifest's audio identity;
- either a legacy string prompt or a structured `{system, user}` prompt; the
  latter is rendered as separate dialog turns and is never flattened;
- standard DPO `-logsigmoid(beta * ((pi_c-ref_c) - (pi_r-ref_r)))`, beta .20;
- grad-enabled initial-policy reference log probabilities, detached once and
  reused for the second ordered pass;
- a named 269-tensor / 1,074,318,016-scalar FP32 mutation surface, no LoRA,
  PEFT, or adapters; and
- AdamW (`lr=2.5e-6`, betas `.9,.95`, eps `1e-8`, zero weight decay) on one
  eight-GPU node.

The loader pads the five short rank-local schedules with zero-weight copies so
that each rank executes 55 forwards per update.  It applies scale `8/435` to
each active pair before Lightning's distributed gradient reduction, producing
the global mean of exactly 435 pairs.  This retains the bounded-memory update
shape of the historical trajectory while relying on Lightning for backward,
optimizer stepping, and global-step tracking. The DPO model invokes the
inherited `SALMAutomodel.configure_gradient_clipping(..., 1.0, "norm")` once
immediately before AdamW; that existing upstream path is mesh-aware for the
LLM/perception DTensor layouts. It writes a data-free per-selected-gradient
mesh/placement receipt before the clip, so the 269-tensor grouping is visible
in every real-update run.

## Launch

Install the tracked NeMo-Speech revision as the normal package in the chosen
container, then invoke the entrypoint with a new output root:

```bash
torchrun --standalone --nproc-per-node=8 examples/speechlm2/salm_dpo_train.py \
  dpo.output_root=/durable/new-hero2-r5-replay
```

Do not use an editable checkout or a `PYTHONPATH` overlay as a substitute for
the tracked source package in an experiment.  The effective resolved config is
written to the fresh output root before model construction.

For a real one-update integration smoke, retain the complete 4,350-row direct
Lhotse input and override only `trainer.max_steps=1`,
`dpo.expected_updates=1`, and `dpo.checkpoint_steps=[1]`.  That run still
captures all initial reference scalars using the production path; it merely
stops after the first ordered 435-pair optimizer update.  The checked-in
historical-r5 config remains the full 20-update trajectory.

## Evidence produced by a run

Before any reference forward, the run writes `MODEL_AUTHORITY.json`: it binds
the source DCP metadata SHA-256, requires a strict no-missing/no-unexpected
model-state load on every rank, and records compact preconstruction/post-DCP
state receipts. It is the proof that temporary ASR construction weights do not
remain an initialization authority. The run then writes `TRAJECTORY.json`, one
`steps/sNN.json` per optimizer update, and `global_compact.json`. A scheduled
checkpoint has both a model DCP and a model-plus-AdamW DCP.
`CHECKPOINT_READY.json` is emitted only after every rank can observe the two
DCP metadata files. Treat a checkpoint without that file as incomplete.

The source data and source checkpoint are substitutable configuration inputs.
An actual replay must record its source revision, effective config, data
receipt, output root, and matched AMI/Full-HF evaluation artifact alongside
these files.

## Standard r22 serving export and evaluation

`salm_dpo_export.py` converts only a durable `CHECKPOINT_READY` model DCP into
a fresh indexed safetensors directory.  It requires the training
`TRAJECTORY.json`, the immutable Hero2 serving baseline, and a fresh output
directory.  The exporter verifies the full tensor namespace and shapes,
requires precisely the trajectory's 269 BF16-to-FP32 selected tensors, and
round-trips each emitted tensor byte-for-byte before it writes
`EVAL_MODEL_READY.json`.  All served weights come from the candidate DCP; the
baseline supplies only `config.json` (with bootstrap state removed) and
tokenizer/generation assets.

```bash
python examples/speechlm2/salm_dpo_export.py \
  --candidate-dcp=/durable/run/checkpoints/s08/model_weights.dcp \
  --trajectory=/durable/run/TRAJECTORY.json \
  --serving-baseline=/immutable/hero2/eval-step-14400 \
  --output=/durable/evaluation/s08/model
```

Use the tracked `serve_salm_dpo_hero2_vllm_r22.sh` as the server entrypoint
for official NeMo-Skills `ns_eval`.  It verifies the immutable r22 source
fingerprint, stages only its normal package inputs into a fresh writable
job-local package artifact (the source mount remains untouched), and
non-editably installs that artifact before invoking the stock NeMo-Skills
vLLM server.  It does not compose an alternate TransformerEncoder, set
`PYTHONPATH`, overlay files into an installed package, or score outputs
itself.  `--verify-r22-package-install` is the focused regression: it proves
that the staged artifact installs and imports from an isolated venv.  The
evaluation workflow records the r22 source path, server fingerprint,
TransformerEncoder SHA-256, and exact OpenASR AMI/Full-HF
prompt/decoder/scorer configuration.
