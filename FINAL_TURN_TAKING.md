# Turn-Taking Logic — Code Changes vs Elena/Viet Baseline

## Repos
- **Server inference code**: `https://github.com/vtrinh-nvidia/NeMo_fc` branch `hdubey/ga/eou`
- **Client + server launcher**: `https://gitlab-master.nvidia.com/vtrinh/niva_fc_demo` branch `hdubey/s2s/vtrinh-niva-fc-demo`

---

## Files Changed vs Elena/Viet Baseline

### 1. `nemo/collections/speechlm2/inference/model_wrappers/nemotron_voicechat_inference_wrapper.py`
**Biggest change — all RNNT turn-taking logic is new.**

Elena's code had NO RNNT turn-taking. She had only `_maybe_apply_forced_turn_taking` (ASR-token-based timer). Everything below is new:

| What was added | Why |
|---|---|
| `_rnnt_init_state()` | Initializes 6 state counters per stream: `blank_count`, `nonblank_consec`, `nonblank_total`, `speech_confirmed`, `agent_speaking`, `first_turn` |
| `_rnnt_step()` | Runs RNNT joint network on each ASR encoder frame (80ms), returns blank/non-blank. Updates `blank_count`, `nonblank_consec`, `nonblank_total` |
| `_apply_rnnt_turn_taking()` | Reads RNNT output every 80ms, writes BOS (EOU) or EOS (barge-in) into `gen_text`. All turn-taking logic lives here |
| RNNT call in main inference loop | After LLM token prediction, runs RNNT and overwrites `gen_text[b,t]` if BOS/EOS needed |
| `asr_head` / `embed_asr_tokens` aliasing | GA FC checkpoint lacks separate ASR head; aliased to LM head so code runs without error |
| Token clamping (`clamp(0, vocab-1)`) | Prevents index-out-of-bounds when GA FC checkpoint returns LLM vocab tokens in the ASR slot |

---

### 2. `nemo/collections/speechlm2/inference/vllm/streaming_llm_engine.py`
**GPU pinning for vLLM.**

Elena's code: `AsyncLLM` always launched on whatever GPU was visible. Both EarTTS and LLM fought for GPU 0.

Fix: wrap engine creation with `CUDA_VISIBLE_DEVICES=str(device_id)` so vLLM child processes only see one GPU. Restores env after. Parent process CUDA state unaffected, child workers pin to correct device.

---

### 3. `nemo/collections/speechlm2/inference/model_wrappers/model_factory.py`
**Two bug fixes.**

| Fix | Elena/Viet bug | Change |
|---|---|---|
| `device_id` kwarg | Leaked into `SamplingParams` → TypeError at runtime | Added explicit `device_id` param, passed to engine |
| `asr_tokens` access | `custom_outputs["asr_tokens"]` → KeyError when GA FC checkpoint returns no real ASR | Changed to `.get("asr_tokens", torch.zeros(1))` |

---

### 4. `examples/speechlm2/nemo_inference_pipelines/triton/model_repo_s2s/voicechat/1/infer_streaming.py`
**Two bug fixes.**

| Fix | Bug | Change |
|---|---|---|
| Missing env vars | `S2S_FORCE_TURN_TAKING`, `S2S_MAX_LEN` not mapped → YAML default `force_turn_taking=True` fired at t=0, agent spoke before user | Added 8 missing env var mappings to `_resolve_env_overrides` |
| Prefill text baseline | Initial `</s>` token in `gen_text[0]` showed as "Agent: `</s>`" on first real frame | After prefill `generate_step`, snapshot `len(get_output_text())` as baseline for `text_positions` |

---

### 5. `nemotron_h.py` (niva_fc_demo) — vLLM model patch

| Fix | Bug | Change |
|---|---|---|
| Missing ASR weights | vLLM `AutoWeightsLoader` raised "not initialized from checkpoint" because GA FC checkpoint has no `asr_head` weights | Materialize weight list, detect absent ASR keys, mark them as loaded to skip the check |
| NaN ASR logits | GA FC `asr_head` (aliased from LM head) produced NaN/Inf → `argmax` returned garbage token IDs ≥ vocab_size | `nan_to_num` + `clamp(0, vocab-1)` before argmax |

---

## RNNT Turn-Taking Logic Detail

### State variables (per stream, initialized in `_rnnt_init_state`)

| Variable | Type | Increments on | Resets on |
|---|---|---|---|
| `blank_count` | int | blank RNNT frame | non-blank frame |
| `nonblank_consec` | int | non-blank frame | blank frame |
| `nonblank_total` | int | non-blank frame | EOU fired, barge-in fired, or EOS (turn boundary) |
| `speech_confirmed` | bool | when speech threshold met | EOU fired, or EOS (turn boundary) |
| `agent_speaking` | bool | when BOS written into gen_text | when EOS written into gen_text |
| `first_turn` | bool | (starts True) | first BOS written (agent speaks for first time) |

### EOU — End of User turn → Agent BOS

**Fires when**: `blank_count >= effective_eou AND speech_confirmed AND NOT agent_speaking AND no BOS in lookback window`

- Cannot fire while agent is speaking (`NOT agent_speaking` gate)
- Cannot double-fire (checks lookback window for existing BOS)
- Writes `bos_id` into `gen_text[b, t]` → LLM sees it → starts generating agent response

### Barge-in — User interrupts → Agent EOS

**Fires when**: `nonblank_consec >= user_bos_frames AND agent_speaking AND no EOS in lookback window`

- Only fires during agent speech (`agent_speaking` gate)
- Uses `nonblank_consec` (consecutive non-blank frames) — requires unbroken user speech for 160ms (2 frames)
- Writes `eos_id` into `gen_text[b, t]` → LLM sees it → stops generating

### Speech confirmation — setting `speech_confirmed = True`

`speech_confirmed = True` when:
- `nonblank_consec >= effective_min_speech` (consecutive non-blank frames — strict), **OR**
- `nonblank_total >= effective_min_speech` (total non-blank since last reset — catches short words like "hello" where RNNT inserts blanks between phonemes, breaking the consecutive streak)

### First-turn provisions (`first_turn = True`)

Before the agent has spoken once, lower thresholds apply so "hello" triggers a response:

| | First turn | Subsequent turns |
|---|---|---|
| `effective_min_speech` | 2 frames (160ms) | 3 frames (240ms) |
| `effective_eou` | 2 frames (160ms silence) | 4 frames (320ms silence) |

Additionally, a **first-turn fallback timer**: if `first_turn AND t >= 50 frames (4s) AND speech_confirmed AND no BOS` → force-inject BOS. Ensures the agent always responds on the first turn even if RNNT EOU is too conservative. Gated on `speech_confirmed` so it never fires without user speech.

`first_turn` is set to `False` the moment any BOS is detected in the lookback window.

### Turn-boundary reset at EOS

When agent EOS is detected, `speech_confirmed` and `nonblank_total` are reset to 0. This is NOT echo suppression — it is a clean slate for the next user turn. Without it, signal accumulated during agent speech would immediately re-trigger EOU as soon as the agent stops. Echo cancellation is handled by UI-side AEC and headset hardware.

### Noise guard

After `nonblank_reset_after_silence=10` blank frames (800ms) without `speech_confirmed`, reset `nonblank_total=0`. Prevents isolated noise spikes from slowly accumulating to the speech threshold across minutes of silence.

---

## Thresholds (configured via env vars in `run_s2s_triton_server_hdubey.sh`)

| Env var | Config key | Default | Meaning |
|---|---|---|---|
| `S2S_RNNT_EOU_ENABLED` | `rnnt_eou_enabled` | `false` | Must be `true` to enable RNNT turn-taking |
| `S2S_ASR_EOU` | `asr_eou` | `4` | Blank frames (×80ms) of silence → EOU (subsequent turns) |
| `S2S_USER_BOS_FRAMES` | `user_bos_frames` | `2` | Consecutive non-blank frames → barge-in |
| `S2S_ASR_MIN_SPEECH_FRAMES` | `asr_min_speech_frames` | `3` | Total/consec non-blank frames to confirm speech (subsequent turns) |
| `S2S_FORCE_TURN_TAKING` | `force_turn_taking` | `false` | Must be `false` when RNNT EOU enabled |
| — | `asr_min_speech_frames_first_turn` | `2` | Speech threshold for first turn |
| — | `asr_eou_first_turn` | `2` | EOU silence threshold for first turn |
| — | `first_turn_fallback_frames` | `50` | t frames before fallback BOS on first turn |
| — | `nonblank_reset_after_silence` | `10` | Blank frames before noise-guard reset of nonblank_total |
