# Native THD packed sequences for ASR Transformer, MoE, and PEE

## Goal and compatibility contract

Transition ASR Transformer-family encoders from persistent padded BHSD activations to native token-flat THD while keeping
all historical entry points and checkpoint layouts intact. The completed design obeys these invariants:

1. Existing `forward()` methods, constructor defaults, Hydra fields, exports, state-dict names, `.nemo` archives, and
   legacy padded execution remain compatible.
2. Native packing is explicit through `packed_encoder_sequences`; old configs/checkpoints default to legacy behavior and
   can opt in without weight conversion.
3. Persistent encoder-layer states are `[T_total, D]`; attention Q/K/V are `[T_total, H, D]`. No packed Transformer layer
   reconstructs `[B,H,S,D]`.
4. Utterance boundaries, causal semantics, positional reset, PEE expert alignment, MoE routing/metrics, frozen branches,
   activation checkpointing, FSDP2, context parallelism, chunking, and SALM dummy calls are preserved.
5. Valid-token outputs and gradients match BHSD within backend-appropriate tolerances.
6. Public APIs appear before private helpers in touched files.

## Public representation and opt-in APIs

`nemo/collections/asr/parts/packed_sequence.py` defines:

- `PackedEncoderOutput(data, lengths, cu_seqlens, max_seqlen)` with `data [T_total,D]`, int64 `lengths [B]`, contiguous
  int32 offsets `[B+1]`, and a host integer maximum length.
- `pack_encoder_output`, `unpack_encoder_output`, `split_encoder_output`, and reset-per-utterance position IDs.
- `with_data()` for reusing already validated metadata without repeated CUDA synchronization.

Opt-in entry points are collision-free:

- `TransformerEncoder.forward_sequence_packed(...)` (also used by `MoETransformerEncoder`).
- `GGEMMTransformerEncoder.forward_all_sequence_packed(...)`: serial THD oracle.
- `GGEMMTransformerEncoder.forward_grouped_sequence_packed(...)`: layer-synchronous grouped THD.
- `ParallelExpertEncoder.forward_sequence_packed(...)`: layer-synchronous grouped production path; the low-level GGEMM
  container retains the serial numerical/benchmark oracle.
- `AudioPerceptionModule.forward_sequence_packed(...)`.

Capability selection requires `supports_sequence_packed_output=True` plus the exact method. `packed_encoder_sequences`
is independent of LLM input `packed_sequences`; `packed_encoder_cp` independently opts into token-flat CP gathering.

## Transformer and MoE execution

1. Preprocessing/subsampling and positional-input ordering remain historical. Compaction happens immediately after
   pre-encoding/embed normalization.
2. RoPE uses reset-per-utterance position IDs. Q/K norm and partial RoPE preserve projection order and parameter gradients.
3. Fast CUDA uses FlashAttention varlen for supported fp16/bf16 layouts, duplicate offsets for empty rows, zero attention
   probability dropout, and exact causal flags.
4. CPU/no-provider/rel-pos compatibility uses compact per-utterance attention. CPU no-grad uses FlexAttention; CPU with
   gradients uses a differentiable math reference on older supported PyTorch releases. Rel-pos preserves centered slices
   and Transformer-XL score bias. No compatibility path creates a batch-wide BHSD tensor.
5. Dense FFNs, residuals, final norm, and output projection consume arbitrary leading dimensions. MoE routes only valid
   compact tokens, preserving task-loss router gradients, load-balancing loss, counts, probability sums, and empty-token
   autograd anchors.
6. Packed MoE metrics accumulate once per logical forward and are suppressed only during checkpoint recomputation.

The opt-in packed route deliberately makes MoE routing padding-neutral; legacy padded routing continues counting padded
positions for backwards compatibility.

## Layer-synchronous grouped PEE

Serially running three native THD expert stacks removes padding but defeats PEE's parallel-expert purpose and creates many
small launches. Production therefore executes all experts in layer lockstep:

1. Compatible FeatureStacking projections share one grouped `bmm`. Live stacks retain gradients in train and eval-with-
   grad; detached packed weights are cached only under inference/no-grad.
2. Per layer, speech and sound share grouped QKV projection calls. `fused_qkv=True` uses one grouped call; false uses three
   truthful grouped Q/K/V calls. Biasless projections use `bmm` without allocating zero biases. The narrow speaker branch
   remains a projection singleton.
3. Q/K normalization and RoPE remain per expert. Experts with identical head dimension, mode, device/dtype, and complete
   sequence metadata concatenate heads into one THD variable-length attention call. PEE's three experts form one call per
   layer in the standard configuration.
4. Compatible output projections share grouped calls; residual dropout uses each layer's own train/eval state.
5. Dense FFNs group equal token count/width/dropout behavior with portable `bmm`/`baddbmm`. Both independent FFN dropout
   sites and per-expert train/eval states are validated; incompatible or individually toggled units use their native path.
6. Sparse MoE top-k sorts routed token rows by expert. Supported CUDA BF16/aligned shapes use PyTorch ragged
   `grouped_mm` for both projections and its direct grouped backward for input and weight gradients. Other PyTorch/device/
   dtype/alignment combinations fall back to exact capacity-padded `baddbmm`; no token is dropped.
7. A per-forward trace records grouped call counts and actual dense/MoE backend. It resets for empty/mixed-native calls,
   so profiling cannot report stale kernel choices.

The packed MoE policy is backwards-compatible and state-free:

- `sequence_packed_moe_mode='auto'`: dense grouped eval (speed-first), ragged top-k grouped training.
- `'topk'`: memory-first grouped preset in both phases.
- `'dense'` and `'native'`: explicit diagnostic/compatibility choices.
- `sequence_packed_ggemm_backend='grouped_mm'`: use ragged grouped kernel where eligible, safe capacity fallback elsewhere.

Historical padded `moe_mode='dense'` and `ggemm_backend='baddbmm'` defaults do not change. Configs that omit the new packed-
only fields receive defaults during construction/restoration.

## Metadata, memory, and compatibility fallbacks

- PEE experts with statically identical FeatureStacking geometry share one validated `lengths/cu_seqlens/max_seqlen` set
  and one compact mask. This removes repeated CUDA-to-host validation. Post-forward length validation identity-shortcuts
  the common production case.
- Layer state stores metadata separately from initial packed data, and padded pre-encoder tensors are released before the
  Transformer loop.
- Automatic attention bucketing allows old rel-pos, causal, mixed-head-dimension, and mixed-mode checkpoints to use
  the correct number of THD buckets without exposing a diagnostic-only strict mode.
- Unsupported custom experts keep the serial packed capability protocol. Existing legacy
  `GGEMMTransformerEncoder.forward_packed` keeps its older head-packed meaning and behavior.

## PEE fusion, checkpointing, and state dicts

- Speech/sound/speaker Transformer states stay THD through all expert layers. Speaker states unpack only at Sortformer;
  sound states unpack only for a legacy CTC SoundToken boundary. Sound and diarization fusion repack aligned data and then
  operate token-flat.
- Activation checkpointing covers the entire grouped PEE boundary. The checkpoint returns three data tensors and one
  shared validated metadata set; backward recomputation suppresses MoE stat accumulation without suppressing the forward.
- Frozen experts retain their eval/dropout behavior even when the outer PEE trains. Trainable eval-mode branches retain
  gradients; cached detached banks never sever them.
- No new parameter or persistent-buffer keys are introduced. Synthetic legacy PEE archives and old config shapes load
  strictly; Transformer-to-MoE remains an intentional warm start because routers are new parameters.

## Perception, SALM, chunking, FSDP2, and CP

- SALMAutomodel selects packed perception only when `model.packed_encoder_sequences=true`, including PEE calls without
  speaker targets, microbatches, dummy synchronization calls, and audio-free ranks.
- FSDP2 registers the custom packed root forward. A real two-rank test covers one zero-token rank, an all-empty step, every
  trainable parameter gradient, grouped traces, and packed CP with local batch smaller than world size.
- Chunking splits/recombines packed offsets and preserves the public list of unpadded embeddings.
- Packed CP gathers rank token totals and one differentiable tail-padded token buffer per rank, then removes dummy slots;
  all ranks issue identical collectives and preserve FSDP dummy autograd edges.
- Unsupported connector/RoTE/encoder-return combinations fail capability selection explicitly and retain legacy forward.

## Verification matrix and acceptance gates

Representation:

- round-trip, B=1, ragged, zero-length row, all-empty, invalid dtype/device/offset/total, non-contiguous input, metadata
  replacement, and gradients.

Transformer/MoE:

- full/causal x RoPE/abs/no-pos/rel-pos; QK norm, QKV bias, partial RoPE, empty rows; boundary/future-token isolation;
  compact layer hooks; fp16/bf16 Flash varlen and fp32/reference fallback; valid outputs, inputs, all relevant parameters,
  router loss/counts/stats, dead experts, and activation checkpointing.

Grouped PEE:

- grouped versus serial THD and legacy BHSD outputs; CUDA input and every trainable parameter gradient; fused/independent
  grouped QKV; actual grouped attention/projection/FFN call counts; dense and ragged top-k routing; unaligned capacity
  fallback; empty first/middle/last expert; all-empty autograd; mixed frozen/train dropout and independently configured
  dropout sites; qkv bias/qk norm/causal/rel-pos multi-bucket compatibility; metadata validation count; eval-with-grad;
  checkpoint metadata/stat behavior; old capability signatures; state-dict key identity.

Integration:

- PEE fusion modes and thresholds, perception capability selection/projection gradients, SALM/chunk/dummy/audio-free paths,
  real multiprocess CP, and real two-rank FSDP2+CP backward.

Performance acceptance:

- Six-repeat permutation-counterbalanced legacy/serial/grouped benchmark with raw samples and source hashes.
- Numerical preflight `rtol=0.03, atol=0.03` plus dedicated tighter CPU and full CUDA gradient tests.
- Nsight NVTX ranges prove grouped PEE materially reduces kernels/API calls versus serial THD.
- Publish speed-first and memory-first grouped MoE tradeoffs rather than attributing all changes to THD layout alone.

Both requested implementation review rounds are complete. Round one findings covered per-expert dropout, grouped-mm
alignment/backward, metadata synchronization/lifetime, non-strict compatibility, trace truth, and test gaps. Round two
covered eval-with-grad caches, checkpoint metadata, empty/mixed trace reset, independently mutable dropout sites,
PyTorch-version gating, biasless QKV overhead, fused-QKV semantics, and older-PyTorch CPU autograd. All actionable feedback
was addressed before final verification.

## Future optimization

The remaining avoidable overhead is transient stacking of separate checkpoint-compatible weights/activations for portable
equal-shape grouped calls. A pointer-array or heterogeneous CUTLASS/Triton grouped GEMM could remove these copies and fold
the narrow speaker projection into the same launch without altering state dicts or public APIs.
