# EasyMagpie performance experiments

This note records the current EasyMagpie serving reference plus performance
experiments that were intentionally not kept in production. Measurements were
taken on an RTX A6000 with vLLM/vLLM-Omni 0.24 in the `easymagpie-vllm`
environment on 2026-07-20, 2026-07-21, and 2026-07-27. Treat the absolute
numbers as hardware-specific; the controlled comparisons are the useful part.

## Latest deployed reference: four-position text prefill

The current reference was measured on 2026-07-27 after folding the first four
known text-led decode positions into causal prefill. For this checkpoint,
`phoneme_delay=3` and `speech_delay=5`: positions 0--2 contain only target-text
input, while position 3 also contains the known phoneme BOS input. Acoustic
prediction remains disabled during these rows; the phoneme predicted from the
last prefill row is reinjected into the first decode step.

The benchmark used an RTX A6000, vLLM/vLLM-Omni 0.24.0, 128 requests per row,
the deployed FP32 native codec, startup cadence 6/6 frames, steady cadence 8
frames, and five text tokens per WebSocket update. It ran from the pending
prefill working tree based on `3e016e5ed`. The model `config.json` SHA-256 was
`a04011f6e2869630009360c800dceb310773f7e68a37b5c769aaa0f293280fd7`;
the deployment YAML SHA-256 was
`67daa189a35acdf074f0f3f97ee24086706481f475993f177095593af761b9ca`.

| Input / concurrency | Request rate | Audio RTF | TTFA mean / p95 | Latency mean / p95 | Audio ITL mean / p95 | Playback |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Whole-text HTTP, c=1 | 1.30 req/s | 8.07x | 103.7 / 106.7 ms | 0.77 / 1.22 s | 68.8 / 76.7 ms | 0/1,364 underruns |
| Incremental WebSocket, c=1 | 1.34 req/s | 7.51x | 110.2 / 115.5 ms | 0.75 / 1.25 s | 72.8 / 79.3 ms | 0/1,188 underruns |
| Whole-text HTTP, c=32 | 8.11 req/s | 47.52x | 504.9 / 616.5 ms | 3.62 / 5.71 s | 343.2 / 463.8 ms | 0/1,291 underruns; 2 deadlines |
| Incremental WebSocket, c=32 | 7.66 req/s | 42.83x | 594.7 / 761.7 ms | 3.76 / 6.66 s | 361.6 / 461.7 ms | 0/1,182 underruns |

All 512 measured requests succeeded. Relative to the earlier eager c=1 rows
below, mean TTFA improved by 24.1 ms (18.9%) for HTTP and 23.8 ms (17.8%) for
WebSocket. The dtype-matched whole-text c=32 comparison is the mean of the two
explicit FP32 side runs below: request rate improved from 7.50 to 8.11 req/s
(8.1%), RTF from 45.11x to 47.52x (5.3%), and TTFA from 608.6 to 504.9 ms
(103.7 ms, or 17.0%). The older eager whole-text c=32 result of 52.79x aligns
with the explicit FP16 side result of 52.71x, so it is not the appropriate
throughput control for the deployed FP32 codec.

A separate 128-request HTTP c=32 run saved 128 valid mono 22.05 kHz waveforms.
It measured 8.16 req/s, 48.97x RTF, 492.3 ms mean TTFA, and zero playback
underruns. Whisper-turbo scoring produced 1.61% WER (17 substitutions, 6
deletions, and 8 insertions over 1,930 reference words), with no utterance at
or above 20% error. The earlier 1.49% WER used a different 128-item cohort:
the benchmark selects samples with unseeded `random.sample`, so the 0.12
percentage-point difference is not a paired quality regression.

## Native vLLM codec migration

The native codec keeps all 26 causal-convolution histories in vLLM state
pages, so each call processes only new acoustic frames. The previous exported
codec replayed the 11-frame left context on every call. In the controlled FP32
standalone A6000 benchmark, the native path was 1.78x faster at batch 1 and
2.67x faster at batch 32 for the deployed steady shape (8 new frames versus a
19-frame stateless replay). Across batch sizes 1–32, the speedup was 1.78–2.67x
for 8 new frames and 1.80–3.73x for 4 new frames.

The immediately adjacent 128-request, c=32 service measurements (pre-native
`efd36a9da`, native `1f2745b16`) recorded the following end-to-end change:

| Input | Exported codec | Native vLLM codec | Change |
| --- | --- | --- | --- |
| Whole text | 8.30 req/s, 44.08x RTF, 630.2 ms TTFA | 8.73 req/s, 51.79x RTF, 512.9 ms TTFA | +5.2% req/s, +17.5% RTF, -117.3 ms TTFA |
| Incremental | 7.03 req/s, 40.47x RTF, 723.8 ms TTFA | 7.67 req/s, 46.98x RTF, 614.5 ms TTFA | +9.1% req/s, +16.1% RTF, -109.3 ms TTFA |

Both before/after runs had zero playback underruns. These service numbers
measure the complete migration rather than codec kernels alone: the native run
also used FP16 instead of the exported codec's FP32 and changed startup cadence
from 6/6/8 to 5/6/8 frames. The isolated FP32 results above are therefore the
better estimate of codec implementation efficiency; the service table captures
the observed deployment-level benefit at the time. The current deployment uses
native FP32 because later listening tests found audible artifacts with FP16.

## Codec CUDA graphs (discarded)

The experiment added a codec-specific vLLM runner, a persistent acoustic-code
input buffer, and exact-shape CUDA graphs. It captured uniform batches for
chunk lengths 5, 6, and 8 frames and batch sizes 1, 2, 4, 8, 16, 24, and 32.
Mixed batches, final tails, and shapes outside that set stayed eager. Requests
were never padded because dummy rows would update the stateful codec cache.

The implementation required about 185 lines in the custom runner and about 291
production lines overall. It captured 21 graphs in roughly one second and used
about 0.56 GiB of extra GPU memory. Temporary dispatch instrumentation found
that only 318 of 592 Stage-1 executions (53.7%) used a graph; the rest were
mixed or otherwise unmatched and ran eager.

### Results

| Input / concurrency | Eager baseline | Codec CUDA graphs |
| --- | --- | --- |
| Whole text, c=1 | 1.20 req/s, 7.33x RTF, 127.8 ms TTFA, 0/1,350 underruns | 1.14 req/s, 7.34x RTF, 127.7 ms TTFA, 0/1,424 underruns |
| Incremental, c=1 | 1.14 req/s, 6.85x RTF, 134.0 ms TTFA, 0/1,286 underruns | 1.18 req/s, 6.86x RTF, 132.8 ms TTFA, 0/1,243 underruns |
| Whole text, c=32 | 8.76 req/s, 52.79x RTF, 514.0 ms TTFA, 0/1,337 underruns | 8.70 req/s, 51.08x RTF, 515.4 ms TTFA, 0/1,307 underruns |
| Incremental, c=32 | 8.02 req/s, 45.90x RTF, 608.3 ms TTFA, 0/1,225 underruns | 7.87 req/s, 44.56x RTF, 613.0 ms TTFA, 2/1,212 underruns |

A second whole-text c=32 graph run produced 8.48 req/s, 53.02x RTF, and
516.5 ms TTFA. The experiment passed 68 tests (one skipped), produced 128 valid
mono 22.05 kHz waveforms, and had 1.49% WER (21 substitutions, 6 deletions, and
2 insertions over 1,941 reference words).

Conclusion: the native codec is already efficient enough that graph launch
savings were offset by dispatch/copy overhead and incomplete shape coverage.
The result was neutral to slightly negative, so the custom runner and graph
machinery were removed. A future H100 experiment could revisit only the steady
8-frame shape, but it should first demonstrate a material gain in a prototype.

### Codec dtype side result

A controlled 128-request, c=32 Stage-1 comparison favored FP16 over FP32:

| Dtype | RTF (two runs) | Mean TTFA (two runs) | Request rate (two runs) |
| --- | --- | --- | --- |
| FP16 | 52.71x / 51.32x | 505.7 / 509.7 ms | 8.77 / 8.51 req/s |
| FP32 | 45.50x / 44.72x | 608.3 / 608.8 ms | 7.56 / 7.44 req/s |

FP16 improved average RTF by about 15.3% and reduced TTFA by about 101 ms, but
listening tests exposed audible codec artifacts. The default deployment therefore
uses FP32; FP16 remains a throughput reference rather than a recommended mode.

## Component no-op ablations

These experiments estimate where end-to-end service time is spent. They are
compute upper bounds, not candidate model implementations.

### Controlled setup

- Stage 0 used vLLM's dummy weight loader in every variant, including the
  baseline. Stage 1 kept real codec weights except in the codec-forward no-op.
- Every request ran exactly 128 Stage-0 decode steps. Audio EOS and all other
  special acoustic tokens were blocked, and engine stop IDs were disabled.
- The local-transformer no-op returned a fixed valid code (`1`) for all 16
  codebooks. The one-pass variant ran the 3-layer local transformer once and
  then ran all 16 output heads/top-k samplers; it removed only the other 15
  transformer forwards. The codec no-op returned correctly sized zero audio.
- Input selection was deterministic (`random.seed(0)`). Concurrency 1 used 32
  measured requests; concurrency 32 used 128. The benchmark's normal warm-up
  ran before each measured level.
- All variants used the same 5/6/8-frame codec streaming cadence. No run had a
  playback underrun.

The dummy loader skips the model's custom `load_weights`, so the benchmark had
to initialize the forbidden acoustic-token mask in the constructor. A first
attempt also used `min_tokens=128`; that exposed the checkpoint's intentionally
out-of-vocabulary backbone EOS ID (`2` for a two-token dummy vocabulary) in
vLLM's min-token mask and caused a CUDA index assertion. The final harness used
`ignore_eos`, no stop IDs, and `max_tokens=128` instead.

### Results

| Variant | c=1: req/s | c=1: RTF | c=1: TTFA | c=32: req/s | c=32: RTF | c=32: TTFA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline: 16 LT forwards + real codec | 0.77 | 7.96x | 127.8 ms | 7.03 / 7.29 | 70.71x / 73.00x | 421.7 / 414.3 ms |
| One LT forward + real codec | 1.71 | 17.60x | 71.3 ms | 12.72 / 13.52 | 129.42x / 133.39x | 574.9 / 388.1 ms |
| LT no-op + real codec | 2.10 | 21.68x | 60.7 ms | 16.51 / 17.00 | 165.27x / 170.95x | 202.9 / 208.3 ms |
| 16 LT forwards + codec no-op | 0.81 | 8.32x | 124.0 ms | 7.94 / 7.96 | 79.75x / 79.63x | 382.1 / 371.7 ms |

Additional mean values for the two c=32 runs:

| Variant | req/s | RTF | TTFA | ITL | Request latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 7.16 | 71.86x | 418.0 ms | 262.0 ms | 4.45 s |
| One LT forward | 13.12 | 131.41x | 481.5 ms | 126.8 ms | 2.43 s |
| LT no-op | 16.76 | 168.11x | 205.6 ms | 109.9 ms | 1.90 s |
| Codec no-op | 7.95 | 79.69x | 376.9 ms | 236.6 ms | 4.02 s |

The one-pass TTFA had one high-variance c=32 run (574.9 ms); the repeat was
388.1 ms. Throughput and end-to-end latency were much more stable and are the
better signals for component attribution.

### Interpretation

At c=32, using average request rates and comparing service time (`1 / req/s`):

- Removing 15 of the 16 local-transformer forwards reduced baseline service
  time by about 45% and increased throughput by about 83%.
- Removing the complete local-transformer/code-prediction path reduced service
  time by about 57% and increased throughput by about 134%.
- The difference between the one-pass and full no-op results is about 12% of
  baseline service time. It includes the remaining transformer forward, all 16
  output projections, top-k/Gumbel sampling, and staging overhead.
- Removing codec compute reduced service time by about 10% and increased
  throughput by about 11%. At c=1, the improvement was only about 5%.

The dominant optimization target is therefore the intra-frame codebook
predictor, especially repeating the full local transformer 16 times. The codec
is measurable but secondary after the native vLLM conversion. The ablations do
not prove that a one-pass predictor preserves quality: it removes autoregressive
conditioning between codebooks and would require retraining or an architecture
change before it could become a real optimization.

All hardcoded ablation changes were reverted after measurement. No ablation
audio is suitable for WER or perceptual evaluation: dummy weights, fixed codes,
and silent codec output deliberately destroy speech quality.
