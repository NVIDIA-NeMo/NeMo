# Native packed ASR encoder benchmark and Nsight Systems report

## Outcome

Native THD improves Transformer and MoE latency and memory in inference and training. For PEE, the production
layer-synchronous grouped THD path restores the parallel-expert execution that the serial THD correctness oracle loses:

- PEE grouped inference is 1.54x faster than serial THD and statistically level with the legacy grouped BHSD path in
  this run, while using 49.6% less incremental memory than legacy.
- PEE grouped training is 1.17x faster than serial THD and 1.56x faster than legacy, while using 15.0% less incremental
  memory than legacy.
- The memory-first grouped top-k inference preset uses 25.8 MiB (72.8% below legacy) and remains 1.24x faster than the
  serial THD oracle, at the cost of lower throughput than speed-first dense grouped eval.

These are complete-implementation comparisons, not layout-only ablations. Attention providers, PEE scheduling, and MoE
execution differ and are reported explicitly.

## Setup and protocol

- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, compute capability 12.0, 96 GiB.
- Software: PyTorch 2.10.0+cu128, BF16, external FlashAttention variable-length provider.
- Seed 17; six fresh identical model/input repetitions. The implementation order cycles through every permutation, so
  each PEE implementation occupies every order position equally.
- Each repetition uses 20 warm-up and 100 measured iterations. Latency is the median over 600 samples; brackets are IQR.
- Training measures forward plus backward over valid output tokens and includes weighted MoE auxiliary loss. It performs
  no optimizer update.
- Memory is the median incremental peak allocated above the post-warm-up resident baseline. The JSON also records
  resident and total peak allocated memory, all per-repeat samples, order positions, backend/provider identities, and
  grouped execution traces.
- Inputs are ragged: Transformer/MoE use 3,328 valid of 8,192 padded input frames; PEE uses 3,840 valid of 8,192.

Exact command:

```bash
env PYTHONPATH=. /home/pzelasko/miniconda3/envs/nemo/bin/python \
  scripts/speech_recognition/benchmark_packed_asr_encoders.py \
  --encoders transformer moe pee --phases inference training \
  --warmup 20 --iterations 100 --repeats 6 \
  --output packed_sequence_asr_encoders_benchmark_grouped_final.json
```

The raw machine-readable outputs are retained locally rather than versioned, avoiding 13,995 lines of generated
JSON in the review. This report records their aggregate results, provenance, commands, and numerical preflight.

## Final implementation comparison

| Encoder | Phase | Implementation | Latency [IQR] | Incremental peak | Speedup vs legacy | Memory reduction vs legacy |
|---|---|---|---:|---:|---:|---:|
| Transformer | inference | legacy BHSD | 4.978 [4.791, 5.005] ms | 88.0 MiB | 1.00x | 0.0% |
| Transformer | inference | native THD | 2.100 [2.091, 2.111] ms | 44.3 MiB | 2.37x | 49.7% |
| Transformer | training | legacy BHSD | 15.686 [15.431, 15.753] ms | 1,040.7 MiB | 1.00x | 0.0% |
| Transformer | training | native THD | 7.378 [7.370, 7.548] ms | 426.8 MiB | 2.13x | 59.0% |
| MoE | inference | legacy BHSD | 7.870 [7.833, 8.093] ms | 88.0 MiB | 1.00x | 0.0% |
| MoE | inference | native THD | 5.570 [5.541, 5.591] ms | 41.1 MiB | 1.41x | 53.3% |
| MoE | training | legacy BHSD | 22.090 [21.962, 22.280] ms | 1,120.4 MiB | 1.00x | 0.0% |
| MoE | training | native THD | 15.869 [14.681, 16.057] ms | 469.3 MiB | 1.39x | 58.1% |
| PEE | inference | legacy grouped BHSD | 4.900 [4.885, 4.923] ms | 95.1 MiB | 1.00x | 0.0% |
| PEE | inference | serial THD oracle | 7.575 [6.921, 7.669] ms | 9.9 MiB | 0.65x | 89.6% |
| PEE | inference | grouped THD | 4.920 [4.373, 4.964] ms | 47.9 MiB | 1.00x | 49.6% |
| PEE | training | legacy BHSD | 24.177 [23.835, 24.343] ms | 237.7 MiB | 1.00x | 0.0% |
| PEE | training | serial THD oracle | 18.135 [17.847, 18.250] ms | 174.6 MiB | 1.33x | 26.5% |
| PEE | training | grouped THD | 15.454 [15.335, 15.541] ms | 201.9 MiB | 1.56x | 15.0% |

The PEE grouped path's trace proves that each of four layers used one attention group containing all three experts,
one grouped QKV and output-projection bucket for the compatible speech/sound pair, dense baddbmm MoE in speed-first eval,
and true ragged `grouped_mm` top-k MoE in training. The 256-wide speaker branch remains a projection singleton but joins
the common attention call because its head dimension and sequence boundaries match.

## PEE speed-first versus memory-first grouped MoE

`sequence_packed_moe_mode='auto'` selects dense grouped MoE in eval for latency and ragged top-k grouped MoE in training
for compute/memory. `sequence_packed_moe_mode='topk'` is the explicit memory-first preset. A separate six-repeat inference
ablation is stored in `packed_sequence_asr_encoders_benchmark_grouped_topk_ablation.json`:

| PEE inference | Latency [IQR] | Incremental peak | Actual MoE kernel |
|---|---:|---:|---|
| serial THD oracle | 6.909 [6.888, 6.934] ms | 9.9 MiB | native per expert |
| grouped THD, top-k | 5.572 [5.554, 5.594] ms | 25.8 MiB | ragged `grouped_mm` |

Relative to the main-run legacy result, grouped top-k uses 72.8% less incremental memory; relative to the same-run serial
oracle it is 1.24x faster. This makes the policy tradeoff explicit instead of implying that dense grouping preserves the
serial oracle's exceptionally low transient memory.

## Numerical preflight

Before timing, the harness compares valid outputs using `rtol=0.03, atol=0.03`; lengths must be identical. Dedicated
tests additionally compare inputs and every trainable parameter gradient.

| Encoder/implementation | Valid tokens | Max abs error | Mean abs error | Relative L2 | Result |
|---|---:|---:|---:|---:|---|
| Transformer native THD | 3,328 | 0.062500 | 0.002732 | 0.004700 | pass |
| MoE native THD | 3,328 | 0.046875 | 0.002299 | 0.004114 | pass |
| PEE serial THD | 480 | 0.062500 | 0.003168 | 0.005196 | pass |
| PEE grouped THD | 480 | 0.062500 | 0.002112 | 0.004054 | pass |
| PEE grouped top-k vs serial THD | 480 | 0.062500 | 0.003155 | 0.005253 | pass |

## Nsight Systems launch analysis

Nsight Systems 2025.5.2 profiled one 100-iteration PEE inference repetition. Profiling overhead makes these latencies
different from the unprofiled table. The aggregate is checked in as `packed_sequence_asr_encoders_nsys_stats.csv`.

Capture/export commands:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true --output=/tmp/packed_asr_grouped_nsys_final \
  env PYTHONPATH=. /home/pzelasko/miniconda3/envs/nemo/bin/python \
  scripts/speech_recognition/benchmark_packed_asr_encoders.py \
  --encoders pee --phases inference \
  --pee-implementations legacy_bhsd serial_thd grouped_thd \
  --warmup 20 --iterations 100 --repeats 1 --profile \
  --output /tmp/packed_asr_grouped_nsys_final_benchmark.json

nsys export --type sqlite --force-overwrite true \
  --output /tmp/packed_asr_grouped_nsys_final.sqlite \
  /tmp/packed_asr_grouped_nsys_final.nsys-rep
```

Aggregation query (kernels are associated with runtime launches by `correlationId`):

```sql
WITH ranges AS (
  SELECT CASE
      WHEN text LIKE '%/legacy_bhsd/%' THEN 'legacy_bhsd'
      WHEN text LIKE '%/serial_thd/%' THEN 'serial_thd'
      WHEN text LIKE '%/grouped_thd/%' THEN 'grouped_thd'
    END AS implementation, start, end
  FROM NVTX_EVENTS
  WHERE text LIKE 'packed_asr/pee/inference/%/repeat_0/iteration_%' AND end IS NOT NULL
), per_range AS (
  SELECT implementation, (end - start) / 1e6 AS nvtx_cpu_submit_ms,
    (SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME rt
      WHERE rt.start >= ranges.start AND rt.start <= ranges.end) AS cuda_api_calls,
    (SELECT COALESCE(SUM(rt.end - rt.start), 0) / 1e6 FROM CUPTI_ACTIVITY_KIND_RUNTIME rt
      WHERE rt.start >= ranges.start AND rt.start <= ranges.end) AS cuda_api_ms,
    (SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k
      JOIN CUPTI_ACTIVITY_KIND_RUNTIME rt ON rt.correlationId = k.correlationId
      WHERE rt.start >= ranges.start AND rt.start <= ranges.end) AS kernels,
    (SELECT COALESCE(SUM(k.end - k.start), 0) / 1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL k
      JOIN CUPTI_ACTIVITY_KIND_RUNTIME rt ON rt.correlationId = k.correlationId
      WHERE rt.start >= ranges.start AND rt.start <= ranges.end) AS kernel_ms
  FROM ranges
)
SELECT implementation, COUNT(*), AVG(nvtx_cpu_submit_ms), AVG(cuda_api_calls),
  AVG(cuda_api_ms), AVG(kernels), AVG(kernel_ms), MIN(kernel_ms), MAX(kernel_ms)
FROM per_range GROUP BY implementation ORDER BY implementation;
```

| Implementation | Profiled median | Mean CPU-submit | CUDA API calls/time | Kernels | Aggregate kernel time |
|---|---:|---:|---:|---:|---:|
| legacy BHSD | 5.466 ms | 5.492 ms | 601 / 1.695 ms | 494 | 3.982 ms |
| serial THD | 9.686 ms | 9.792 ms | 1,409 / 3.245 ms | 1,013 | 2.914 ms |
| grouped THD | 5.832 ms | 5.876 ms | 662 / 1.458 ms | 564 | 2.882 ms |

Grouped THD cuts kernel count by 44.3%, CUDA API calls by 53.0%, and the CPU-submit range by 40.0% versus serial THD.
Its aggregate kernel time is also slightly lower. This directly validates that expert work launches in parallel grouped
operations rather than three serial expert stacks. Compared with legacy, grouped THD keeps substantially lower allocated
activation memory and lower aggregate kernel time, with modestly more launches from packing/metadata/routing operations.

## Limits and next optimization

Transient weight/activation stacking remains in portable equal-shape grouped helpers. A pointer-array or heterogeneous
CUTLASS/Triton grouped GEMM could remove those copies and include the narrow speaker projection bucket. Production
checkpoint dimensions and sequence distributions should be benchmarked before changing the speed-first/memory-first
policy default.
