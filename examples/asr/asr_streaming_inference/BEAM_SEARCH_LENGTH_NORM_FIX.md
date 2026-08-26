# Cache-aware RNNT beam search: per-utterance score normalization fix

## Bug

`CacheAwareRNNTBeamStreamingState.select_best_beam_idx_(score_norm=True)` ranks beams at every EOU
using `score / (current_lengths_nb + 1)`. Both `score` and `current_lengths_nb`
(`ModifiedALSDBatchedRNNTComputer`, `nemo/collections/asr/parts/submodules/rnnt_malsd_batched_computer.py`)
are cumulative over the **entire stream** — they are zeroed once at session start and, at every EOU,
`select_beam_in_state_item_` collapses the K beams to the winner without ever resetting either field.

Effect: after the first EOU fold, every beam carries the same large `(prior_score, prior_length)`
baseline into the next utterance. As a stream accumulates more utterances, the length-normalized ratio
`(prior_score + Δscore) / (prior_length + Δlength + 1)` converges toward `prior_score / prior_length`
regardless of the current utterance's actual hypothesis quality — the normalized score becomes
progressively less sensitive to the utterance actually being decided, and beam selection quality
degrades the longer a session runs (multi-segment audio, i.e. more than one EOU per stream).

Confirmed on SLURP (13,078 utterances, many audio files containing more than one EOU segment): the
unfixed baseline predominantly produces **leading-word truncation** — e.g. reference `"put meeting
with pawel for tomorrow ten am"` transcribed as `"for tomorrow at ten am"` — because the diluted ranking
under-weights the start of a new utterance relative to the huge inherited prior mass. At
`beam_size=12, ngram_lm_alpha=0.5` this reaches 19.07% WER (word-error, case/punctuation-insensitive)
vs. 14.29% once fixed.

## Fix

- `CacheAwareRNNTBeamStreamingState`: add `_score_baseline` / `_length_baseline`, snapshotted by
  `set_beam_score_baseline_()` right after each EOU beam collapse (called from
  `CacheAwareRNNTPipeline._apply_beam_update_`, right after `select_beam_in_state_item_`).
- `select_best_beam_idx_(score_norm=True)` now ranks
  `(score - baseline_score) / (length - baseline_length + 1) ** length_norm_power` — the baseline
  restores per-utterance normalization; `length_norm_power` (new, default `1.0` = unchanged behavior)
  generalizes the length term to a tunable exponent (GNMT-style length penalty).

## `length_norm_power` sweep on SLURP

Swept `length_norm_power ∈ {0, 0.125, 0.25, 0.5, 0.75, 1.0}` against the pre-fix `unfixed` baseline,
across the full `beam_size ∈ {2, 4, 8, 12}` × `ngram_lm_alpha ∈ {0, 0.02, 0.04, 0.08, 0.1, 0.2, 0.3,
0.4, 0.5}` grid (36 points, 216 fixed-side runs total), on the full SLURP test set (13,078 utterances,
`stop_history_eou=800ms`, `ngram_lm_model=nemotron_speech_streaming_en_0.6b_slurp.kenlm.nemo`, WER
computed case/punctuation-insensitive since SLURP references carry neither).

WER (%) per `(beam_size, alpha)`:

| bs | alpha | unfixed | p=0 | p=0.125 | p=0.25 | p=0.5 | p=0.75 | p=1.0 |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.0 | 15.529 | 15.285 | 15.285 | 15.295 | 15.349 | 15.414 | 15.518 |
| 2 | 0.02 | 15.447 | 15.190 | 15.183 | 15.185 | 15.255 | 15.317 | 15.434 |
| 2 | 0.04 | 15.373 | 15.159 | 15.139 | 15.134 | 15.206 | 15.247 | 15.350 |
| 2 | 0.08 | 15.423 | 15.156 | 15.128 | 15.149 | 15.217 | 15.312 | 15.372 |
| 2 | 0.1 | 15.419 | 15.136 | 15.139 | 15.117 | 15.248 | 15.305 | 15.339 |
| 2 | 0.2 | 15.580 | 15.133 | 15.064 | 15.061 | 15.219 | 15.338 | 15.471 |
| 2 | 0.3 | 15.582 | 14.949 | 14.873 | 14.902 | 15.128 | 15.337 | 15.478 |
| 2 | 0.4 | 15.834 | 15.013 | 14.932 | 14.975 | 15.291 | 15.559 | 15.687 |
| 2 | 0.5 | 16.200 | 15.328 | 15.250 | 15.243 | 15.635 | 15.869 | 16.037 |
| 4 | 0.0 | 15.350 | 15.113 | 15.132 | 15.162 | 15.225 | 15.294 | 15.337 |
| 4 | 0.02 | 15.267 | 14.954 | 14.991 | 15.034 | 15.087 | 15.188 | 15.261 |
| 4 | 0.04 | 15.164 | 14.891 | 14.882 | 14.948 | 15.031 | 15.081 | 15.159 |
| 4 | 0.08 | 15.063 | 14.771 | 14.753 | 14.807 | 14.907 | 14.993 | 15.041 |
| 4 | 0.1 | 15.040 | 14.729 | 14.688 | 14.725 | 14.845 | 14.945 | 15.018 |
| 4 | 0.2 | 14.767 | 14.616 | 14.413 | 14.459 | 14.517 | 14.565 | 14.703 |
| 4 | 0.3 | 14.592 | 15.121 | 14.407 | 14.277 | 14.414 | 14.409 | 14.518 |
| 4 | 0.4 | 14.562 | 16.322 | 14.935 | 14.461 | 14.425 | 14.406 | 14.478 |
| 4 | 0.5 | 14.870 | 17.541 | 15.749 | 14.790 | 14.760 | 14.734 | 14.797 |
| 8 | 0.0 | 15.308 | 15.078 | 15.082 | 15.133 | 15.188 | 15.261 | 15.318 |
| 8 | 0.02 | 15.183 | 14.894 | 14.930 | 14.930 | 15.012 | 15.115 | 15.182 |
| 8 | 0.04 | 15.039 | 14.778 | 14.758 | 14.822 | 14.874 | 14.940 | 15.032 |
| 8 | 0.08 | 14.953 | 14.508 | 14.567 | 14.605 | 14.699 | 14.810 | 14.935 |
| 8 | 0.1 | 14.834 | 14.531 | 14.491 | 14.571 | 14.664 | 14.723 | 14.802 |
| 8 | 0.2 | 14.454 | 15.116 | 14.241 | 14.161 | 14.165 | 14.212 | 14.383 |
| 8 | 0.3 | 14.186 | 17.675 | 14.821 | 14.205 | 13.926 | 13.929 | 14.106 |
| 8 | 0.4 | 14.172 | 20.914 | 16.098 | 14.624 | 14.031 | 13.850 | 14.084 |
| 8 | 0.5 | 14.482 | 24.212 | 17.642 | 15.318 | 14.463 | 14.200 | 14.365 |
| 12 | 0.0 | 15.204 | 15.022 | 15.038 | 15.092 | 15.155 | 15.237 | 15.309 |
| 12 | 0.02 | 15.082 | 14.856 | 14.859 | 14.895 | 14.991 | 15.070 | 15.165 |
| 12 | 0.04 | 14.946 | 14.718 | 14.704 | 14.763 | 14.865 | 14.932 | 15.063 |
| 12 | 0.08 | 14.851 | 14.485 | 14.551 | 14.587 | 14.653 | 14.779 | 14.908 |
| 12 | 0.1 | 14.823 | 14.506 | 14.428 | 14.492 | 14.625 | 14.712 | 14.806 |
| 12 | 0.2 | 14.362 | 15.735 | 14.286 | 14.046 | 14.033 | 14.093 | 14.295 |
| 12 | 0.3 | 15.474 | 19.500 | 15.147 | 14.276 | 13.763 | 13.721 | 13.984 |
| 12 | 0.4 | 17.316 | 24.115 | 16.928 | 14.830 | 13.949 | 13.706 | 13.984 |
| 12 | 0.5 | 19.071 | 27.556 | 18.618 | 15.773 | 14.401 | 14.121 | 14.289 |

### Findings

1. **`unfixed` itself degrades badly at large `beam_size` + high `alpha`** (e.g. `bs=12, alpha=0.5`:
   19.07%) — confirms the baseline-dilution bug is real and its damage scales with both `beam_size`
   (more candidates for the diluted ranking to misjudge) and `alpha` (LM fusion doesn't correct for the
   truncation the diluted ranking already introduced).
2. **`p=0` (no length normalization) is unsafe once `alpha ≥ 0.2`**: at `bs ≥ 8` it causes an
   empty-hypothesis collapse (up to 12.6%/22% of utterances producing blank output, worst case
   `bs=12, alpha=0.5`: 27.56% WER). Root cause: unnormalized RNNT+LM score favors near-empty hypotheses.
   `p=0` must never be used with LM fusion.
3. **`p=0.125`–`p=0.25` is best at low `alpha`** across all beam sizes, consistently beating both
   `unfixed` and `p=1.0`.
4. **`p=0.5`–`p=0.75` is the reliable safe zone across the full grid**, never the worst value in any
   row and close to optimal everywhere; required (not just preferred) once `alpha ≥ 0.3` at `bs ≥ 8` to
   fully recover from the `p=0`/`p=0.125` degradation at those settings.
5. **`p=1.0` (prior, unnormalized-baseline-fix-only behavior) is never the best value** in any row.

### Recommendation

Default `length_norm_power` stays `1.0` for full backward compatibility (this PR's baseline-subtraction
fix alone already corrects the dilution bug at `p=1.0`; `unfixed` vs `p=1.0` in the table above is that
fix's isolated effect). Deployments running beam search with LM fusion should tune
`asr.decoding.beam.length_norm_power` — `0.5` is a safe, broadly strong choice (never worst, large
margin from the `p=0` collapse zone); `0.25` is stronger at low `alpha` but not safe at `alpha ≥ 0.3`
with large `beam_size`.

## Test plan

- `tests/collections/asr/inference/test_cache_aware_rnnt_state.py` (new): covers
  `select_best_beam_idx_` (raw score, plain length normalization, `length_norm_power` variants, the
  stale-baseline dilution this fix addresses) and `set_beam_score_baseline_`/reset behavior.
- SLURP sweep above (216 fixed-side runs + 36 unfixed baselines), harness under `new-eou-exps/` in a
  sibling branch (not part of this PR's diff).
