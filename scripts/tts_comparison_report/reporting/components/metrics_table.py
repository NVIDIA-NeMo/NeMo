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
import html

import numpy as np
from scripts.tts_comparison_report.reporting.metrics import MetricSpec, MetricsRegistry
from scripts.tts_comparison_report.reporting.models import BucketData


def _as_bucket_list(
    bucket_baseline: BucketData,
    bucket_candidate: BucketData | list[BucketData],
) -> list[BucketData]:
    candidates = bucket_candidate if isinstance(bucket_candidate, list) else [bucket_candidate]
    return [bucket_baseline, *candidates]


def _format_metric_values(
    values: list[float],
    metric: MetricSpec,
) -> list[str]:
    scaled_values = [metric.multiplier * value for value in values]
    best_value = None

    if metric.lower_is_better is True:
        best_value = min(scaled_values)
    elif metric.lower_is_better is False:
        best_value = max(scaled_values)

    best_indices = set()
    if best_value is not None:
        best_indices = {index for index, value in enumerate(scaled_values) if value == best_value}
        if len(scaled_values) == 2 and len(best_indices) == 2:
            best_indices = {0}

    output = []
    for index, value in enumerate(scaled_values):
        value_str = html.escape(f"{round(value, metric.round_digits)}{metric.units}")
        if index in best_indices:
            value_str = f"<strong>{value_str}</strong>"
        output.append(value_str)

    return output


def prepare_benchmark_metrics_table_rows(
    benchmark_name: str,
    bucket_baseline: BucketData,
    bucket_candidate: BucketData | list[BucketData],
) -> list[list[str]]:
    """Prepare formatted metric rows for one benchmark across all systems."""
    buckets = _as_bucket_list(bucket_baseline, bucket_candidate)
    rows = []

    for metric in MetricsRegistry:
        values = [
            bucket.get_metric_avg_value(
                metric_name=metric.key,
                benchmark_name=benchmark_name,
            )
            for bucket in buckets
        ]

        if any(value is None for value in values):
            if metric.optional:
                continue
            raise ValueError(f"Unknown metric '{metric.key}' for benchmark '{benchmark_name}'.")

        formatted_values = _format_metric_values([float(value) for value in values], metric)
        rows.append([html.escape(metric.report_name), *formatted_values])

    return rows


def prepare_summary_metrics_table_rows(
    bucket_baseline: BucketData,
    bucket_candidate: BucketData | list[BucketData],
) -> list[list[str]]:
    """Prepare macro-averaged metric rows across all benchmarks and systems."""
    buckets = _as_bucket_list(bucket_baseline, bucket_candidate)
    rows = []

    for metric in MetricsRegistry:
        if not metric.include_in_summary:
            continue

        system_values: list[list[float]] = [[] for _ in buckets]
        skip = False

        for benchmark_name in bucket_baseline.benchmarks:
            values = [
                bucket.get_metric_avg_value(
                    metric_name=metric.key,
                    benchmark_name=benchmark_name,
                )
                for bucket in buckets
            ]

            if any(value is None for value in values):
                if metric.optional:
                    skip = True
                    break
                raise ValueError(f"Unknown metric '{metric.key}' for benchmark '{benchmark_name}'.")

            for system_index, value in enumerate(values):
                system_values[system_index].append(float(value))

        if skip:
            continue

        averages = [float(np.mean(values)) for values in system_values]
        rows.append([html.escape(metric.report_name), *_format_metric_values(averages, metric)])

    return rows
