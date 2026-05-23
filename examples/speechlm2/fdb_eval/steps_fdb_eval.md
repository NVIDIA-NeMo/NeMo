# FDB v1.0 Evaluation — End-to-End Steps

## Repositories

| Repo | Purpose | URL | Branch |
|------|---------|-----|--------|
| **NeMo_fc** (GitHub) | NeMo S2S model code, inference patches | https://github.com/vtrinh-nvidia/NeMo_fc | `hdubey/ga/eou` |
| **niva_fc_demo** (GitLab) | Run scripts, configs, sweep scripts | https://gitlab-master.nvidia.com/vtrinh/niva_fc_demo | `hdubey/s2s/vtrinh-niva-fc-demo` |
| **Full-Duplex-Bench-NV** | FDB evaluation code | https://github.com/DanielLin94144/Full-Duplex-Bench (NVIDIA fork at `/home/hdubey/Full-Duplex-Bench-NV`) | — |

Scripts in this doc live in `niva_fc_demo` under `fdb_eval/`.
NeMo model code (duplex_stt_model.py patches, nemotron_voicechat.py) lives in `NeMo_fc` branch `hdubey/ga/eou`.

Runs the Full-Duplex-Bench v1.0 benchmark against the ROW26 checkpoint
(e70-step28008 HF + RNNT cand3 + EarTTS Megan) and computes:
- **Pause Handling** — Synthetic & Candor TOR ↓
- **Smooth Turn-Taking** — Candor TOR ↑, Latency ↓
- **User Interruption** — TOR ↑, GPT-4o score ↑, Latency ↓

---

## 0. Prerequisites

### Checkpoints
| Role | Path on machine |
|------|----------------|
| S2S HF checkpoint | `/mnt/point2/VoiceChat/GA_FC_model/e70-step28008-tts-eartts-34014_asr_cand3_0.6b-PK_ep0_01988_300` |
| RNNT `.nemo` | `/mnt/point2/VoiceChat/fdb_eval_row26/rnnt_ckpt/cand3_kratos_ep0_01988_300.nemo` |
| EarTTS `.ckpt` | `/mnt/point2/VoiceChat/fdb_eval_row26/tts_ckpt/eartts_megan_step34014.ckpt` |
| Speaker ref wav | `/mnt/point2/VoiceChat/speaker_prompt/Mg_a_00759.wav` |

On **Draco**, copy/symlink these to the same local paths or update the paths at the top of each script.

### Data
| Item | Path |
|------|------|
| FDB shar datasets (full, 672 samples) | `/mnt/point2/VoiceChat/fdb_eval_row26/fdb_shar/{dataset}` |
| FDB shar mini (21 test samples only) | `/mnt/point2/VoiceChat/fdb_eval_row26/fdb_shar_mini/{dataset}` |
| Original data + ground truth | `/mnt/point2/VoiceChat/fdb_eval_row26/original_data/v1.0/{dataset}` |

Four datasets: `candor_turn_taking`, `candor_pause_handling`, `synthetic_pause_handling`, `synthetic_user_interruption`.

### Docker image
```
nvcr.io/nvidian/tegra-audio/nemo_framework_jhw:2.6.0rc0_torch_25.06_py3_teleaug_v4
```

### GPU memory requirement
- Total model: ~30 GB (LLM 9B bf16 18 GB + RNNT 2.4 GB + EarTTS 9.6 GB)
- Requires **2 GPUs** via model parallelism: STT (LLM+RNNT) → GPU 1, EarTTS → GPU 0
- Minimum free memory: ~20 GB on GPU 1, ~10 GB on GPU 0

### NeMo code
Uses the `hdubey/s2s/rnnt-eou-bou` branch of NeMo_elena (mounted at `/NeMo_elena` inside Docker).
The inference script patches are applied automatically by `apply_patches.py` at container startup.

### NVIDIA API key (for GPT-4o user-interruption scoring)
```
/mnt/point2/VoiceChat/fdb_eval_row26/client_key.jsonl
```
Format: `{"client_id": "...", "client_secret": "...", "token_url": "...", "azure_endpoint": "...", "azure_api_version": "..."}`

---

## 1. Run Inference

### Option A — Full shar (672 samples, ~2.5 h)
```bash
bash /mnt/point2/VoiceChat/fdb_eval_row26/scripts/run_fdb_local.sh
```
Outputs land in `/mnt/point2/VoiceChat/fdb_eval_row26/results/inference/validation_logs/pred_wavs/`.

### Option B — Mini shar (21 FDB test samples only, ~10 min)
Use this for parameter sweeps. The mini shar was built from the same 21 samples
that have ground-truth annotations in `original_data/`.
```bash
# Override shar paths via Hydra overrides (see run_fdb_sweep_pad_boost.sh for template):
data.validation_ds.datasets.candor_turn_taking.shar_path=/mnt/point2/VoiceChat/fdb_eval_row26/fdb_shar_mini/candor_turn_taking
data.validation_ds.datasets.candor_pause_handling.shar_path=...
data.validation_ds.datasets.synthetic_pause_handling.shar_path=...
data.validation_ds.datasets.synthetic_user_interruption.shar_path=...
```

### Key inference flags
| Flag | Default | Effect |
|------|---------|--------|
| `++model.stt.model.inference_pad_boost` | 0 | Boost PAD (silence) logit — higher = model stays quiet longer; improves Pause Handling TOR |
| `++model.stt.model.inference_bos_boost` | 0 | Boost BOS (agent start) logit — higher = model responds faster |
| `++model.stt.model.inference_eos_boost` | 0 | Boost EOS (agent stop) logit |
| `++model.stt.model.inference_user_eos_boost` | 0 | Boost user EOS in ASR channel — higher = more sensitive EOU detection |
| `++model.stt.model.force_bos_num_tokens_after_user_eos` | (unset) | Force model to respond within N tokens after user EOS — improves User Interruption TOR |
| `data.validation_ds.batch_size` | 4 | Set to 1 for FDB eval (avoids padding artifacts) |
| `++trainer.precision` | 16 | Use `bf16` locally |
| `++trainer.devices` | -1 | Set to `1` for single-process model parallelism |

---

## 2. Rebuild mini shar (if needed on Draco)

If the mini shar doesn't exist, rebuild it from the full shar:
```python
import json, gzip, os

FDB_IDS = {
    "candor_turn_taking":        [15, 42, 46, 70, 77, 90],
    "candor_pause_handling":     [160, 163, 188, 46, 70, 77],
    "synthetic_pause_handling":  [15, 42, 46, 70, 77],
    "synthetic_user_interruption": [163, 188, 46, 70],
}
SHAR_BASE      = "/mnt/point2/VoiceChat/fdb_eval_row26/fdb_shar"
MINI_SHAR_BASE = "/mnt/point2/VoiceChat/fdb_eval_row26/fdb_shar_mini"

for ds, ids in FDB_IDS.items():
    target_ids = {f"{ds}_{i:04d}" for i in ids}
    src_dir, dst_dir = f"{SHAR_BASE}/{ds}", f"{MINI_SHAR_BASE}/{ds}"
    os.makedirs(dst_dir, exist_ok=True)
    kept = []
    with gzip.open(f"{src_dir}/cuts.000000.jsonl.gz", "rt") as f:
        for line in f:
            if json.loads(line)["id"] in target_ids:
                kept.append(line)
    with gzip.open(f"{dst_dir}/cuts.000000.jsonl.gz", "wt") as f:
        f.writelines(kept)
    for tar in ["recording.000000.tar", "target_audio.000000.tar"]:
        dst = f"{dst_dir}/{tar}"
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(f"{src_dir}/{tar}"), dst)
    print(f"{ds}: {len(kept)} samples")
```

---

## 3. Run FDB Scoring

After inference completes, `validation_logs/pred_wavs/` contains 672 (or 21) WAV files
named `{dataset}_{dataset}_{id:04d}_rank0.wav`.

```bash
bash /mnt/point2/VoiceChat/fdb_eval_row26/scripts/run_fdb_scoring_row26.sh \
    --results-dir /mnt/point2/VoiceChat/fdb_eval_row26/results/inference
```

The script runs for each dataset:
1. **Reorganize** — maps WAV files to the `original_data/` directory structure
2. **ASR** (Parakeet TDT 0.6b-v2 in Docker) — generates `output.json` with time-aligned transcript
3. **Evaluation** — computes TOR, latency, and (for User Interruption) GPT-4o Adherence score

Metrics are written to `results/inference/fdb_metric/v1.0/eval_{dataset}.log`.

### Known issue: synthetic_user_interruption sample 46
Sample 46 in `synthetic_user_interruption` is missing `interrupt.json` in the original data.
The scoring script automatically moves it aside before ASR and restores it after.
The eval covers 3 samples (163, 188, 70) instead of 4.

### GPT-4o scoring (User Interruption only)
The standard `eval_user_interruption.py` in this repo has the LLM judge stripped.
To run GPT-4o scoring manually after ASR:
```python
import sys; sys.path.insert(0, '/home/hdubey/Full-Duplex-Bench-NV/evaluation')
from auth_utils import load_client_config, refresh_azure_client
from eval_user_interruption import eval_user_interruption  # upstream version with GPT-4o

config = load_client_config('/mnt/point2/VoiceChat/fdb_eval_row26/client_key.jsonl')
_, client = refresh_azure_client(config)
# Note: use gpt-4o (not gpt-4-turbo) on this NVIDIA Azure endpoint
# Monkey-patch MODEL_NAME before calling if needed
eval_user_interruption(
    '/path/to/fdb_metric/v1.0/synthetic_user_interruption',
    client=client
)
```

---

## 4. Parameter Sweep (pad_boost)

To find the best `inference_pad_boost` value:
```bash
bash /home/hdubey/NeMo_elena/examples/speechlm2/fdb_eval/run_fdb_sweep_pad_boost.sh \
    --results-base /mnt/point2/VoiceChat/fdb_eval_row26/results/sweep_pad_boost \
    --pad-boosts "0 1 2 3 5"
```

Each sweep point runs inference on the 21-sample mini shar (~10 min) + scoring (~5 min).
Summary table is written to `sweep_pad_boost/sweep_summary.tsv`.

**Known parameter tradeoffs:**

| Change | Pause Handling TOR ↓ | Smooth TT Latency ↓ | User Interruption TOR ↑ |
|--------|---------------------|--------------------|-----------------------|
| `pad_boost` ↑ | ✓ improves | ✗ higher latency | ✗ lower TOR |
| `RNNT_EOS_SILENCE_FRAMES` ↑ | ✓ improves | ✗ higher latency | — |
| `inference_user_eos_boost` ↑ | ✗ worse | ✓ lower latency | ✓ improves |
| `force_bos_num_tokens_after_user_eos` set | neutral | ✓ | ✓ improves |
| `delay_source_text_by` ↓ | ✗ worse | ✓ lower latency | neutral |

Recommended starting point: `pad_boost=2` + `force_bos_num_tokens_after_user_eos=5`.

---

## 5. ROW26 Baseline Results (2026-05-07)

| Task | Dataset | Metric | Value |
|------|---------|--------|-------|
| Pause Handling | Synthetic | TOR ↓ | 1.0 (worst) |
| Pause Handling | Candor | TOR ↓ | 1.0 (worst) |
| Smooth Turn-Taking | Candor | TOR ↑ | 1.0 (best) |
| Smooth Turn-Taking | Candor | Latency ↓ | 0.0 s (clamped; raw values negative = responds before EOU) |
| User Interruption | Synthetic | TOR ↑ | 0.33 |
| User Interruption | Synthetic | GPT-4o ↑ | 0.0 / 5 |
| User Interruption | Synthetic | Latency ↓ | 0.40 s |

**Inference config:** `transformers==4.57.3`, `bf16`, `batch_size=1`, `temperature=0.8`, `top_p=0.9`,
`repetition_penalty=1.2`, `inference_guidance_scale=0.2`, `inference_noise_scale=0.001`,
all boost params = 0.
