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
import warnings
from enum import Enum
from itertools import combinations
from typing import Optional

from scipy.stats import mannwhitneyu
from scripts.tts_comparison_report.reporting.constants import P_VAL_ROUND_DIGITS
from scripts.tts_comparison_report.reporting.metrics import DistributionMetricsRegistry
from scripts.tts_comparison_report.reporting.models import BucketData, StatTestAnalysisInfo, StatTestResult, Winner


_SIGNIFICANCE_LEVEL: float = 0.05


class _Alternative(str, Enum):
    two_sided = "two-sided"
    greater = "greater"
    less = "less"


def _as_bucket_list(
    bucket_baseline: BucketData,
    bucket_candidate: BucketData | list[BucketData],
) -> list[BucketData]:
    candidates = bucket_candidate if isinstance(bucket_candidate, list) else [bucket_candidate]
    return [bucket_baseline, *candidates]


def _run_single_stat_test(
    baseline: list[float],
    candidate: list[float],
    lower_is_better: bool,
) -> tuple[Winner, _Alternative, float]:
    if not baseline:
        raise ValueError("Baseline sample is empty.")

    if not candidate:
        raise ValueError("Candidate sample is empty.")

    if len(baseline) != len(candidate):
        warnings.warn(
            "\nBaseline and candidate contain different numbers of samples. "
            "This may indicate missing filewise metrics or dataset mismatch.",
            stacklevel=2,
        )

    p_val_two_sided = mannwhitneyu(baseline, candidate, alternative="two-sided", method="auto").pvalue

    if p_val_two_sided >= _SIGNIFICANCE_LEVEL:
        return Winner.tie, _Alternative.two_sided, round(p_val_two_sided, P_VAL_ROUND_DIGITS)

    p_val = mannwhitneyu(baseline, candidate, alternative="less", method="auto").pvalue

    if p_val < _SIGNIFICANCE_LEVEL:
        winner = Winner.baseline if lower_is_better else Winner.candidate
        return winner, _Alternative.less, round(p_val, P_VAL_ROUND_DIGITS)

    p_val = mannwhitneyu(baseline, candidate, alternative="greater", method="auto").pvalue

    if p_val < _SIGNIFICANCE_LEVEL:
        winner = Winner.candidate if lower_is_better else Winner.baseline
        return winner, _Alternative.greater, round(p_val, P_VAL_ROUND_DIGITS)

    return Winner.tie, _Alternative.two_sided, round(p_val_two_sided, P_VAL_ROUND_DIGITS)


def _map_winner_to_name(
    winner: Winner,
    baseline_name: str,
    candidate_name: str,
) -> str:
    if winner == Winner.baseline:
        return baseline_name
    if winner == Winner.candidate:
        return candidate_name
    return winner.value


def run_stat_tests(
    bucket_baseline: BucketData,
    bucket_candidate: BucketData | list[BucketData],
    benchmark_name: Optional[str] = None,
) -> list[StatTestResult]:
    """Run every configured distribution test for all pairs of systems."""
    buckets = _as_bucket_list(bucket_baseline, bucket_candidate)
    results = []

    for first_bucket, second_bucket in combinations(buckets, 2):
        for metric in DistributionMetricsRegistry:
            winner, alternative, p_value = _run_single_stat_test(
                baseline=first_bucket.get_metric_samples(metric.key, benchmark_name),
                candidate=second_bucket.get_metric_samples(metric.key, benchmark_name),
                lower_is_better=metric.lower_is_better,
            )
            results.append(
                StatTestResult(
                    metric_name=metric.report_name,
                    winner=winner,
                    alternative=alternative.value,
                    p_value=p_value,
                    baseline_name=first_bucket.name,
                    candidate_name=second_bucket.name,
                )
            )

    return results


def prepare_stat_tests_table_rows(
    baseline_name: str,
    candidate_name: str,
    stat_test_results: list[StatTestResult],
    include_comparison: bool = False,
) -> list[list[str]]:
    """Prepare statistical-test table rows, optionally identifying each system pair."""
    rows = []

    for result in stat_test_results:
        first_name = result.baseline_name or baseline_name
        second_name = result.candidate_name or candidate_name
        winner = _map_winner_to_name(result.winner, first_name, second_name)
        row = [
            html.escape(result.metric_name),
            html.escape(winner),
            html.escape(result.alternative),
            html.escape(str(result.p_value)),
        ]
        if include_comparison:
            row.insert(1, html.escape(f"{first_name} vs {second_name}"))
        rows.append(row)

    return rows


def prepare_stat_tests_analysis_info(
    baseline_name: str,
    candidate_name: str,
    stat_test_results: list[StatTestResult],
) -> StatTestAnalysisInfo:
    """Summarize significant wins, preserving the original two-system behavior."""
    comparison_pairs = {
        (result.baseline_name or baseline_name, result.candidate_name or candidate_name)
        for result in stat_test_results
    }
    if len(comparison_pairs) <= 1:
        baseline_wins = [result.metric_name for result in stat_test_results if result.winner == Winner.baseline]
        candidate_wins = [result.metric_name for result in stat_test_results if result.winner == Winner.candidate]
        if not baseline_wins and not candidate_wins:
            return StatTestAnalysisInfo(winner=None, advantages=None)

        winner, wins = (
            (baseline_name, baseline_wins)
            if len(baseline_wins) >= len(candidate_wins)
            else (candidate_name, candidate_wins)
        )
        return StatTestAnalysisInfo(winner=winner, advantages=", ".join(wins))

    wins: dict[str, list[str]] = {}
    for result in stat_test_results:
        first_name = result.baseline_name or baseline_name
        second_name = result.candidate_name or candidate_name
        winner = _map_winner_to_name(result.winner, first_name, second_name)
        if result.winner == Winner.tie:
            continue
        loser = second_name if winner == first_name else first_name
        wins.setdefault(winner, []).append(f"{result.metric_name} over {loser}")

    if not wins:
        return StatTestAnalysisInfo(winner=None, advantages=None)

    max_wins = max(len(metrics) for metrics in wins.values())
    leaders = sorted(name for name, metrics in wins.items() if len(metrics) == max_wins)
    winner = ", ".join(leaders)
    advantages = "; ".join(f"{name}: {', '.join(wins[name])}" for name in leaders)
    return StatTestAnalysisInfo(winner=winner, advantages=advantages)
