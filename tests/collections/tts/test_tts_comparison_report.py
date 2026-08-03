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
import json
from pathlib import Path

import pytest

from scripts.tts_comparison_report.generate_report import _create_argparser
from scripts.tts_comparison_report.reporting.components.audio_report import prepare_audio_pairs
from scripts.tts_comparison_report.reporting.components.boxplots import BoxPlotsConfig
from scripts.tts_comparison_report.reporting.components.eval_report import prepare_eval_artifacts
from scripts.tts_comparison_report.reporting.components.metrics_table import prepare_benchmark_metrics_table_rows
from scripts.tts_comparison_report.reporting.components.stat_tests import (
    prepare_stat_tests_analysis_info,
    run_stat_tests,
)
from scripts.tts_comparison_report.reporting.constants import DUMMY_TASK_ID, TEMPLATES_DIR
from scripts.tts_comparison_report.reporting.helpers import generate_s3_prefix
from scripts.tts_comparison_report.reporting.models import (
    AudioPair,
    BenchmarkData,
    BucketData,
    BucketStructure,
    ExpirationInfo,
    StatTestResult,
    TaskInfo,
    Winner,
)
from scripts.tts_comparison_report.reporting.orchestrator import Orchestrator
from scripts.tts_comparison_report.reporting.renderer import Renderer, TemplateName
from scripts.tts_comparison_report.reporting.storage import LocalStorage


pytestmark = pytest.mark.unit


def _parse_args(*candidate_args: str):
    return _create_argparser().parse_args(
        [
            "--baseline_name",
            "baseline",
            "--baseline_path",
            "/eval/baseline",
            *candidate_args,
            "--s3_endpoint",
            "https://s3.example.com",
            "--s3_bucket",
            "reports",
            "--s3_region",
            "local",
        ]
    )


def _make_bucket(name: str, path: str, metric_offset: float = 0.0) -> BucketData:
    filewise_metrics = []

    for index in range(4):
        sample_name = f"predicted_audio_{index}"
        filewise_metrics.append(
            {
                "pred_audio_filepath": f"/generated/{sample_name}.wav",
                "gt_text": f"sample {index}",
                "gt_audio_filepath": f"/reference/gt_{index}.wav",
                "context_audio_filepath": f"/reference/context_{index}.wav",
                "cer": 0.01 * index + metric_offset,
                "utmosv2": 4.0 - metric_offset,
                "pred_context_ssim": 0.8 - metric_offset,
            }
        )

    metrics = {
        "wer_cumulative": 0.10 + metric_offset,
        "cer_cumulative": 0.05 + metric_offset,
        "wer_filewise_avg": 0.11 + metric_offset,
        "cer_filewise_avg": 0.06 + metric_offset,
        "utmosv2_avg": 4.0 - metric_offset,
        "ssim_pred_gt_avg": 0.75 - metric_offset,
        "ssim_pred_context_avg": 0.80 - metric_offset,
        "total_gen_audio_seconds": 12.0,
    }
    benchmark = BenchmarkData(
        name="libritts",
        metrics=metrics,
        filewise_metrics=filewise_metrics,
        generated_audio_paths={
            f"predicted_audio_{index}": Path(f"/{name}/predicted_audio_{index}.wav") for index in range(4)
        },
        context_audio_paths={
            f"context_audio_{index}": Path(f"/reference/context_audio_{index}.wav") for index in range(4)
        },
    )
    return BucketData(
        name=name,
        path=Path(path),
        configuration_str=f"{name}-config",
        benchmarks={"libritts": benchmark},
    )


def test_candidate_flags_remain_backward_compatible():
    args = _parse_args("--candidate_name", "candidate", "--candidate_path", "/eval/candidate")

    assert args.candidate_name == ["candidate"]
    assert args.candidate_path == ["/eval/candidate"]


def test_candidate_flags_are_repeatable_for_multiple_systems():
    args = _parse_args(
        "--candidate_name",
        "candidate-a",
        "--candidate_path",
        "/eval/candidate-a",
        "--candidate_name",
        "candidate-b",
        "--candidate_path",
        "/eval/candidate-b",
    )

    assert args.candidate_name == ["candidate-a", "candidate-b"]
    assert args.candidate_path == ["/eval/candidate-a", "/eval/candidate-b"]


def test_metrics_table_contains_every_system():
    baseline = _make_bucket("baseline", "/eval/baseline", 0.0)
    candidate_a = _make_bucket("candidate-a", "/eval/candidate-a", 0.01)
    candidate_b = _make_bucket("candidate-b", "/eval/candidate-b", -0.01)

    rows = prepare_benchmark_metrics_table_rows("libritts", baseline, [candidate_a, candidate_b])

    assert all(len(row) == 4 for row in rows)
    cer_row = next(row for row in rows if row[0] == "CER (cumulative)")
    assert cer_row == ["CER (cumulative)", "5.0%", "6.0%", "<strong>4.0%</strong>"]


def test_statistics_cover_all_system_pairs():
    buckets = [
        _make_bucket("baseline", "/eval/baseline", 0.0),
        _make_bucket("candidate-a", "/eval/candidate-a", 0.01),
        _make_bucket("candidate-b", "/eval/candidate-b", -0.01),
    ]

    results = run_stat_tests(buckets[0], buckets[1:])

    comparisons = {(result.baseline_name, result.candidate_name) for result in results}
    assert comparisons == {
        ("baseline", "candidate-a"),
        ("baseline", "candidate-b"),
        ("candidate-a", "candidate-b"),
    }


def test_audio_samples_contain_every_system():
    buckets = [
        _make_bucket("baseline", "/eval/baseline", 0.0),
        _make_bucket("candidate-a", "/eval/candidate-a", 0.01),
        _make_bucket("candidate-b", "/eval/candidate-b", -0.01),
    ]

    pairs = prepare_audio_pairs(
        bucket_baseline=buckets[0],
        bucket_candidate=buckets[1:],
        bucket_structure=BucketStructure(),
        used_benchmarks=["libritts"],
        samples_per_benchmark=2,
    )

    assert len(pairs["libritts"]) == 2
    assert all(list(pair.system_paths) == ["baseline", "candidate-a", "candidate-b"] for pair in pairs["libritts"])


def test_eval_artifacts_and_audio_template_render_every_system():
    buckets = [
        _make_bucket("baseline", "/eval/baseline", 0.0),
        _make_bucket("candidate-a", "/eval/candidate-a", 0.01),
        _make_bucket("candidate-b", "/eval/candidate-b", -0.01),
    ]

    artifacts = prepare_eval_artifacts(buckets[0], buckets[1:], BoxPlotsConfig())

    assert list(artifacts.configuration.systems) == ["baseline", "candidate-a", "candidate-b"]
    assert all(len(row) == 5 for row in artifacts.summary.stat_test_table_row)
    assert artifacts.summary.box_plots.getbuffer().nbytes > 0

    renderer = Renderer(TEMPLATES_DIR)
    rendered_pair = renderer.render(
        TemplateName.audio_report_pair,
        context_url="context.wav",
        system_urls={"baseline": "baseline.wav", "candidate-a": "a.wav", "candidate-b": "b.wav"},
        text="hello",
    )
    assert rendered_pair.count("<audio") == 4


def test_two_system_artifacts_keep_original_table_shapes():
    baseline = _make_bucket("baseline", "/eval/baseline", 0.0)
    candidate = _make_bucket("candidate", "/eval/candidate", 0.01)

    artifacts = prepare_eval_artifacts(baseline, candidate, BoxPlotsConfig())

    assert all(len(row) == 3 for row in artifacts.summary.metrics_table_row)
    assert all(len(row) == 4 for row in artifacts.summary.stat_test_table_row)


def test_two_system_metric_ties_keep_original_baseline_highlight():
    baseline = _make_bucket("baseline", "/eval/baseline", 0.0)
    candidate = _make_bucket("candidate", "/eval/candidate", 0.0)

    rows = prepare_benchmark_metrics_table_rows("libritts", baseline, candidate)

    cer_row = next(row for row in rows if row[0] == "CER (cumulative)")
    assert cer_row == ["CER (cumulative)", "<strong>5.0%</strong>", "5.0%"]


def test_two_system_stat_summary_keeps_original_tie_breaking():
    results = [
        StatTestResult("CER", Winner.baseline, "less", 0.01, "baseline", "candidate"),
        StatTestResult("UTMOS v2", Winner.candidate, "greater", 0.01, "baseline", "candidate"),
    ]

    analysis = prepare_stat_tests_analysis_info("baseline", "candidate", results)

    assert analysis.winner == "baseline"
    assert analysis.advantages == "CER"


def test_two_system_audio_artifact_names_remain_backward_compatible(tmp_path: Path, monkeypatch):
    orchestrator = Orchestrator(
        bucket_structure=BucketStructure(),
        storage=LocalStorage(),
        s3_client=None,
        renderer=Renderer(TEMPLATES_DIR),
        local_output_dir=tmp_path,
    )
    uploaded_keys = []

    def record_upload(_path: Path, key: str) -> str:
        uploaded_keys.append(key)
        return key

    monkeypatch.setattr(orchestrator, "_upload_audio_file", record_upload)
    pair = AudioPair(
        context_path=Path("/context.wav"),
        baseline_path=Path("/baseline.wav"),
        candidate_path=Path("/candidate.wav"),
        text="sample",
        system_paths={
            "baseline": Path("/baseline.wav"),
            "candidate": Path("/candidate.wav"),
        },
    )

    orchestrator._upload_audio(["libritts"], {"libritts": [pair]}, "report-prefix")

    assert uploaded_keys == [
        "report-prefix/audio/context_libritts_0.wav",
        "report-prefix/audio/baseline_libritts_0.wav",
        "report-prefix/audio/candidate_libritts_0.wav",
    ]


def test_parser_accepts_local_output_without_s3_arguments():
    args = _create_argparser().parse_args(
        [
            "--baseline_name",
            "baseline",
            "--baseline_path",
            "/eval/baseline",
            "--candidate_name",
            "candidate",
            "--candidate_path",
            "/eval/candidate",
            "--local_output_dir",
            "/tmp/report-preview",
        ]
    )

    assert args.local_output_dir == "/tmp/report-preview"
    assert args.s3_endpoint is None
    assert args.s3_bucket is None
    assert args.s3_region is None


def test_s3_prefix_override_replaces_generated_directory_name():
    task_info = TaskInfo(task_id="NMP-I-123", jira_id="NMP-I-123", jira_url="https://jira/NMP-I-123")
    expiration_info = ExpirationInfo(timestamp=1, path_str="2027-01-01T00-00-00Z", user_str="2027-01-01")

    default_prefix = generate_s3_prefix(
        Path("/eval/baseline"),
        Path("/eval/candidate"),
        task_info,
        expiration_info,
    )
    assert default_prefix == "NMP-I-123-baseline_vs_candidate-2027-01-01T00-00-00Z"

    prefix = generate_s3_prefix(
        Path("/eval/very-long-baseline-name"),
        [Path("/eval/candidate-a"), Path("/eval/candidate-b"), Path("/eval/candidate-c")],
        task_info,
        expiration_info,
        override="/codec-reports/four-way/",
    )

    assert prefix == "codec-reports/four-way"

    with pytest.raises(ValueError, match="must not be empty"):
        generate_s3_prefix(Path("/a"), Path("/b"), task_info, expiration_info, override="///")

    with pytest.raises(ValueError, match="relative path"):
        generate_s3_prefix(Path("/a"), Path("/b"), task_info, expiration_info, override="reports/../run")


def _write_evaluation_bucket(root: Path, name: str, metric_offset: float) -> None:
    benchmark_dir = root / "results" / f"{name}_libritts"
    audio_dir = benchmark_dir / "audio" / "repeat_0"
    audio_dir.mkdir(parents=True)

    metrics = {
        "wer_cumulative": 0.10 + metric_offset,
        "cer_cumulative": 0.05 + metric_offset,
        "wer_filewise_avg": 0.11 + metric_offset,
        "cer_filewise_avg": 0.06 + metric_offset,
        "utmosv2_avg": 4.0 - metric_offset,
        "ssim_pred_gt_avg": 0.75 - metric_offset,
        "ssim_pred_context_avg": 0.80 - metric_offset,
        "total_gen_audio_seconds": 12.0,
    }
    filewise_metrics = []
    for index in range(4):
        filewise_metrics.append(
            {
                "pred_audio_filepath": f"/generated/predicted_audio_{index}.wav",
                "gt_text": f"sample {index}",
                "gt_audio_filepath": f"/reference/gt_{index}.wav",
                "context_audio_filepath": f"/reference/context_{index}.wav",
                "cer": 0.01 * index + metric_offset,
                "utmosv2": 4.0 - 0.01 * index - metric_offset,
                "pred_context_ssim": 0.8 - 0.01 * index - metric_offset,
            }
        )
        (audio_dir / f"predicted_audio_{index}.wav").write_bytes(b"RIFF-generated-" + name.encode())
        (audio_dir / f"context_audio_{index}.wav").write_bytes(b"RIFF-context")

    (benchmark_dir / "libritts_metrics_0.json").write_text(json.dumps(metrics), encoding="utf-8")
    (benchmark_dir / "libritts_filewise_metrics_0.json").write_text(json.dumps(filewise_metrics), encoding="utf-8")


def test_local_preview_writes_complete_report_with_relative_assets(tmp_path: Path):
    systems = [
        ("baseline", 0.0),
        ("candidate-a", 0.01),
        ("candidate-b", -0.01),
    ]
    bucket_paths = []
    for name, offset in systems:
        bucket_path = tmp_path / name
        _write_evaluation_bucket(bucket_path, name, offset)
        bucket_paths.append(bucket_path)

    output_dir = tmp_path / "preview"
    orchestrator = Orchestrator(
        bucket_structure=BucketStructure(),
        storage=LocalStorage(),
        s3_client=None,
        renderer=Renderer(TEMPLATES_DIR),
        local_output_dir=output_dir,
    )
    eval_report_path, audio_report_path = orchestrator.run(
        baseline_name="baseline",
        candidate_name=["candidate-a", "candidate-b"],
        baseline_path=bucket_paths[0],
        candidate_path=bucket_paths[1:],
        benchmarks=["libritts"],
        generate_audio_report=True,
        audio_report_benchmarks=["libritts"],
        samples_per_benchmark=2,
        task_id=DUMMY_TASK_ID,
    )

    assert eval_report_path == "eval_report.html"
    assert audio_report_path == "audio_report.html"
    assert (output_dir / eval_report_path).is_file()
    assert (output_dir / audio_report_path).is_file()
    assert (output_dir / "images" / "box_plot_summary.png").is_file()
    assert (output_dir / "images" / "box_plot_libritts.png").is_file()
    assert len(list((output_dir / "audio").glob("*.wav"))) == 8

    eval_html = (output_dir / eval_report_path).read_text(encoding="utf-8")
    audio_html = (output_dir / audio_report_path).read_text(encoding="utf-8")
    assert 'href="audio_report.html"' in eval_html
    assert 'src="images/box_plot_summary.png"' in eval_html
    assert 'src="audio/context_libritts_' in audio_html
    assert audio_html.count("<audio") == 8
    assert "Local preview" in eval_html
