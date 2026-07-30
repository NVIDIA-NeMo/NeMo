# Finite direct-Lhotse SpeechLM2 DPO

This package provides generic SpeechLM2 DPO mechanics:

- finite same-audio preference ingestion through direct Lhotse;
- standard DPO loss with a detached initial-policy reference cache;
- an explicit partial-acoustic trainable surface;
- native Lightning backward and AdamW updates;
- model and optimizer DCP checkpoints; and
- conversion of a ready model DCP to indexed HuggingFace safetensors.

It does not contain a model checkpoint, dataset path, rollout policy,
experiment hyperparameters, cluster configuration, or evaluation recipe.
Those belong to the experiment repository.

## Configuration boundary

`conf/salm_dpo.yaml` is a required-input schema, not a runnable experiment.
All `???` values must be supplied through a tracked external Hydra config or
command-line overrides. The effective resolved configuration is written to a
fresh output root before model construction.

An external experiment config must declare:

- trainer topology, precision, and update count;
- the source model DCP and matching base experiment config;
- DPO beta, seed, optimizer values, clipping norm, finite-pass accounting,
  and checkpoint schedule; and
- the finite preference manifest, expected row count, update size, shard
  count, and world size.

The loader requires `shuffle=false` and `cycle=false`. Chosen and rejected
completions must use the same audio. The owning experiment is responsible for
the provenance of the source checkpoint, prompt, audio, chosen text, rejected
text, and preference-selection process.

## Launch

Install the tracked NeMo-Speech revision normally, then compose this entrypoint
with the external experiment config:

```bash
torchrun --standalone --nproc-per-node="${WORLD_SIZE}" \
  examples/speechlm2/salm_dpo_train.py \
  --config-path=/absolute/path/to/experiment/config \
  --config-name=experiment_name
```

Hydra overrides may be appended for a new output root or another explicitly
tracked experiment input. Do not use editable installs, `PYTHONPATH`
overlays, runtime source rewriting, import hooks, or experiment-local
updaters.

## Run artifacts

Before reference capture, the trainer writes `MODEL_AUTHORITY.json`, binding
the source DCP metadata digest and strict model-state load. It then writes:

- the resolved `effective_config.yaml`;
- `TRAJECTORY.json`;
- one `steps/sNN.json` record per update;
- `global_compact.json`; and
- `DONE.json`.

A scheduled checkpoint is usable only when both DCPs and the readiness receipt
exist:

```text
model_weights.dcp/.metadata
training_state.dcp/.metadata
CHECKPOINT_READY.json
```

`training_state.dcp` includes model and AdamW state. A model-only DCP is an
export input, not a resumable optimizer checkpoint.

## Export

`salm_dpo_export.py` converts a ready candidate model DCP into a fresh indexed
safetensors directory:

```bash
python examples/speechlm2/salm_dpo_export.py \
  --candidate-dcp=/path/to/run/checkpoints/sNN/model_weights.dcp \
  --trajectory=/path/to/run/TRAJECTORY.json \
  --serving-baseline=/path/to/matching/source-serving-model \
  --output=/path/to/fresh/export
```

The candidate DCP supplies every model tensor. The serving baseline supplies
only configuration, tokenizer, and generation assets. The exporter verifies
the tensor namespace, shapes, selected FP32 surface, and emitted tensor bytes
before writing `EVAL_MODEL_READY.json`.

Evaluation is deliberately out of scope for NeMo-Speech. Use the standard
evaluation workflow owned by the experiment repository, and evaluate the
source baseline and exported candidate with identical settings.
