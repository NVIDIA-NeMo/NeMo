# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import warnings

from scripts.tts_comparison_report.reporting.constants import SEED
from scripts.tts_comparison_report.reporting.models import AudioPair, BucketData, BucketStructure

_RNG = random.Random(SEED)


def _as_bucket_list(
    bucket_baseline: BucketData,
    bucket_candidate: BucketData | list[BucketData],
) -> list[BucketData]:
    candidates = bucket_candidate if isinstance(bucket_candidate, list) else [bucket_candidate]
    return [bucket_baseline, *candidates]


def _collect_audio_pairs(
    benchmark_name: str,
    bucket_baseline: BucketData,
    bucket_candidate: BucketData | list[BucketData],
    bucket_structure: BucketStructure,
) -> list[AudioPair]:
    buckets = _as_bucket_list(bucket_baseline, bucket_candidate)
    audio_paths = [bucket.get_benchmark_audio_paths(benchmark_name) for bucket in buckets]
    sample_meta = [bucket.get_benchmark_sample_meta(benchmark_name, bucket_structure) for bucket in buckets]
    expected_names = set(audio_paths[0])

    for bucket, paths in zip(buckets[1:], audio_paths[1:]):
        if set(paths) != expected_names:
            raise ValueError(f"Audio sample sets differ for benchmark '{benchmark_name}' in system '{bucket.name}'.")

    pairs = []

    for name in sorted(expected_names):
        if any(name not in paths or name not in metadata for paths, metadata in zip(audio_paths, sample_meta)):
            raise ValueError(
                f"Missing matched sample '{name}' in audio paths or metadata for benchmark '{benchmark_name}'."
            )

        expected_sample_id = sample_meta[0][name].sample_id
        for bucket, metadata in zip(buckets[1:], sample_meta[1:]):
            if metadata[name].sample_id != expected_sample_id:
                raise ValueError(
                    f"Sample id mismatch for '{name}' in benchmark '{benchmark_name}' and system '{bucket.name}'. "
                    "Probably you use different versions of buckets."
                )

        system_paths = {bucket.name: paths[name] for bucket, paths in zip(buckets, audio_paths)}
        pairs.append(
            AudioPair(
                context_path=sample_meta[0][name].context_path,
                baseline_path=audio_paths[0][name],
                candidate_path=audio_paths[1][name],
                text=sample_meta[0][name].gt_text,
                system_paths=system_paths,
            )
        )

    return pairs


def prepare_audio_pairs(
    bucket_baseline: BucketData,
    bucket_candidate: BucketData | list[BucketData],
    bucket_structure: BucketStructure,
    used_benchmarks: list[str],
    samples_per_benchmark: int,
) -> dict[str, list[AudioPair]]:
    """Prepare matched audio samples for every system and selected benchmark."""
    pairs = {}

    for benchmark_name in used_benchmarks:
        benchmark_pairs = _collect_audio_pairs(benchmark_name, bucket_baseline, bucket_candidate, bucket_structure)
        sampled_pairs = _RNG.sample(benchmark_pairs, k=min(samples_per_benchmark, len(benchmark_pairs)))

        if len(sampled_pairs) < samples_per_benchmark:
            warnings.warn(
                f"\nBenchmark '{benchmark_name}' contains only {len(sampled_pairs)} available paired samples, "
                f"but {samples_per_benchmark} were requested.",
                stacklevel=2,
            )
        pairs[benchmark_name] = sampled_pairs

    return pairs
