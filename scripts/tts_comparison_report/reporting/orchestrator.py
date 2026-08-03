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
import shutil
from io import BytesIO
from logging import Logger
from pathlib import Path
from typing import Optional, TypeVar

from scripts.tts_comparison_report.reporting.components import (
    BoxPlotsConfig,
    prepare_audio_pairs,
    prepare_eval_artifacts,
)
from scripts.tts_comparison_report.reporting.constants import (
    BENCHMARK_META,
    S3_AUDIO_DIR,
    S3_IMAGES_DIR,
    S3_LINK_EXPIRES_IN,
    TQDM_NCOLS,
)
from scripts.tts_comparison_report.reporting.helpers import generate_s3_prefix, make_expiration_info, make_task_info
from scripts.tts_comparison_report.reporting.models import (
    AudioPair,
    BucketData,
    BucketStructure,
    EvalArtifacts,
    ExpirationInfo,
    TaskInfo,
    UploadedAudioPairInfo,
    UploadedBoxPlotsInfo,
)
from scripts.tts_comparison_report.reporting.renderer import Renderer, TemplateName
from scripts.tts_comparison_report.reporting.s3_client import S3Client
from scripts.tts_comparison_report.reporting.storage import BaseStorage
from tqdm import tqdm

_T = TypeVar("_T")


class Orchestrator:
    """Coordinate loading, processing, rendering, and publishing comparison reports."""

    def __init__(
        self,
        bucket_structure: BucketStructure,
        storage: BaseStorage,
        s3_client: Optional[S3Client],
        renderer: Renderer,
        logger: Optional[Logger] = None,
        local_output_dir: Optional[Path] = None,
    ) -> None:
        self.bucket_structure = bucket_structure
        self.storage = storage
        self.s3_client = s3_client
        self.renderer = renderer
        self.logger = logger
        self.local_output_dir = local_output_dir.resolve() if local_output_dir is not None else None
        self.show_pbar = logger is not None

        if (self.s3_client is None) == (self.local_output_dir is None):
            raise ValueError("Configure exactly one report destination: S3 or a local output directory.")

    def _log_info(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.info(msg)

    def _expiration_comment(self, expiration_info: ExpirationInfo) -> str:
        if self.local_output_dir is not None:
            return "Local preview; audio and images are served from this directory."
        return f"This report will expire at {expiration_info.user_str}"

    @staticmethod
    def _normalize_candidates(candidate: _T | list[_T]) -> list[_T]:
        return candidate if isinstance(candidate, list) else [candidate]

    def _load_buckets(
        self,
        baseline_name: str,
        candidate_name: str | list[str],
        baseline_path: Path,
        candidate_path: Path | list[Path],
        benchmark_names: tuple[str, ...],
        check_audio: bool,
    ) -> list[BucketData]:
        candidate_names = self._normalize_candidates(candidate_name)
        candidate_paths = self._normalize_candidates(candidate_path)
        if not candidate_names:
            raise ValueError("At least one candidate system is required.")
        if len(candidate_names) != len(candidate_paths):
            raise ValueError("Candidate names and paths must have the same length.")

        names = [baseline_name, *candidate_names]
        paths = [baseline_path, *candidate_paths]
        if len(set(names)) != len(names):
            raise ValueError("System names must be unique.")

        buckets = []
        for name, path in zip(names, paths):
            self._log_info(f"\nLoading metadata for {name}...")
            buckets.append(
                BucketData.from_storage(
                    bucket_name=name,
                    bucket_path=path,
                    bucket_structure=self.bucket_structure,
                    benchmark_names=benchmark_names,
                    check_audio=check_audio,
                    storage=self.storage,
                )
            )

        reference_set = set(buckets[0].benchmarks)
        for bucket in buckets[1:]:
            benchmark_set = set(bucket.benchmarks)
            if benchmark_set != reference_set:
                raise ValueError(
                    f"Benchmark sets differ for '{buckets[0].name}' and '{bucket.name}': "
                    f"'{reference_set}' vs '{benchmark_set}'."
                )

        for bucket in buckets:
            self._log_info(f"\nLoading metric data for {bucket.name}:")
            bucket.load_metrics(storage=self.storage, show_pbar=self.show_pbar)

        return buckets

    @staticmethod
    def _artifact_key(prefix: str, relative_key: str) -> str:
        return f"{prefix}/{relative_key}" if prefix else relative_key

    def _local_artifact_path(self, key: str) -> Path:
        if self.local_output_dir is None:
            raise RuntimeError("Local output directory is not configured.")
        target = (self.local_output_dir / key).resolve()
        if not target.is_relative_to(self.local_output_dir):
            raise ValueError(f"Local artifact path escapes output directory: '{key}'.")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _upload_audio_file(self, path: Path, key: str) -> str:
        with self.storage.open_file(path) as fileobj:
            if self.local_output_dir is not None:
                with self._local_artifact_path(key).open("wb") as output:
                    shutil.copyfileobj(fileobj, output)
                return key

            if self.s3_client is None:
                raise RuntimeError("S3 client is not configured.")
            return self.s3_client.upload_fileobj(
                fileobj=fileobj,
                key=key,
                expires_in=S3_LINK_EXPIRES_IN,
                content_type="audio/wav",
            )

    def _upload_png_image(self, image: BytesIO, key: str) -> str:
        if self.local_output_dir is not None:
            self._local_artifact_path(key).write_bytes(image.getvalue())
            return key

        if self.s3_client is None:
            raise RuntimeError("S3 client is not configured.")
        return self.s3_client.upload_bytes(
            data=image.getvalue(),
            key=key,
            expires_in=S3_LINK_EXPIRES_IN,
            content_type="image/png",
        )

    def _upload_audio(
        self,
        used_benchmarks: list[str],
        audio_pairs: dict[str, list[AudioPair]],
        s3_prefix: str,
    ) -> dict[str, list[UploadedAudioPairInfo]]:
        total = sum(len(values) for values in audio_pairs.values())
        pbar = tqdm(total=total, ncols=TQDM_NCOLS) if self.show_pbar else None
        uploaded_info = {}

        for benchmark_name in used_benchmarks:
            benchmark_info = []

            for sample_index, pair in enumerate(audio_pairs[benchmark_name]):
                context_url = self._upload_audio_file(
                    pair.context_path,
                    self._artifact_key(s3_prefix, f"{S3_AUDIO_DIR}/context_{benchmark_name}_{sample_index}.wav"),
                )
                system_urls = {}
                system_items = list(pair.system_paths.items())
                for system_index, (system_name, system_path) in enumerate(system_items):
                    if len(system_items) == 2:
                        artifact_name = ("baseline", "candidate")[system_index]
                    else:
                        artifact_name = f"system_{system_index}"
                    system_urls[system_name] = self._upload_audio_file(
                        system_path,
                        self._artifact_key(
                            s3_prefix,
                            f"{S3_AUDIO_DIR}/{artifact_name}_{benchmark_name}_{sample_index}.wav",
                        ),
                    )

                urls = list(system_urls.values())
                benchmark_info.append(
                    UploadedAudioPairInfo(
                        context_url=context_url,
                        baseline_url=urls[0],
                        candidate_url=urls[1],
                        text=pair.text,
                        system_urls=system_urls,
                    )
                )
                if pbar:
                    pbar.update(1)

            uploaded_info[benchmark_name] = benchmark_info

        if pbar:
            pbar.close()
        return uploaded_info

    def _upload_boxplots(self, eval_artifacts: EvalArtifacts, s3_prefix: str) -> UploadedBoxPlotsInfo:
        name_prefix = "box_plot"
        pbar = tqdm(total=len(eval_artifacts.benchmarks) + 1, ncols=TQDM_NCOLS) if self.show_pbar else None
        summary_url = self._upload_png_image(
            eval_artifacts.summary.box_plots,
            self._artifact_key(s3_prefix, f"{S3_IMAGES_DIR}/{name_prefix}_summary.png"),
        )
        if pbar:
            pbar.update(1)

        benchmark_urls = {}
        for benchmark_name, benchmark_result in eval_artifacts.benchmarks.items():
            benchmark_urls[benchmark_name] = self._upload_png_image(
                benchmark_result.box_plots,
                self._artifact_key(s3_prefix, f"{S3_IMAGES_DIR}/{name_prefix}_{benchmark_name}.png"),
            )
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()
        return UploadedBoxPlotsInfo(summary_url=summary_url, benchmark_urls=benchmark_urls)

    def _upload_report(self, report: str, s3_prefix: str, report_name: str) -> str:
        key = self._artifact_key(s3_prefix, f"{report_name}.html")
        if self.local_output_dir is not None:
            self._local_artifact_path(key).write_text(report, encoding="utf-8")
            return key

        if self.s3_client is None:
            raise RuntimeError("S3 client is not configured.")
        return self.s3_client.upload_bytes(
            data=report.encode("utf-8"),
            key=key,
            expires_in=S3_LINK_EXPIRES_IN,
            content_type="text/html; charset=utf-8",
        )

    def _render_audio_report(
        self,
        system_names: list[str],
        used_benchmarks: list[str],
        uploaded_audio_info: dict[str, list[UploadedAudioPairInfo]],
        task_info: TaskInfo,
        expiration_info: ExpirationInfo,
    ) -> str:
        header_block = self.renderer.render(
            name=TemplateName.audio_report_header,
            system_names=system_names,
            expiration_comment=self._expiration_comment(expiration_info),
        )
        benchmark_blocks, benchmark_section_info = [], []

        for benchmark_name in used_benchmarks:
            pair_blocks = [
                self.renderer.render(
                    name=TemplateName.audio_report_pair,
                    context_url=pair.context_url,
                    system_urls=pair.system_urls,
                    text=pair.text,
                )
                for pair in uploaded_audio_info[benchmark_name]
            ]
            benchmark_blocks.append(
                self.renderer.render(
                    name=TemplateName.audio_report_block,
                    title=benchmark_name,
                    section_id=benchmark_name,
                    system_names=system_names,
                    column_count=len(system_names) + 1,
                    pair_blocks=pair_blocks,
                )
            )
            benchmark_section_info.append((benchmark_name, f"{benchmark_name} ({BENCHMARK_META[benchmark_name]})"))

        return self.renderer.render(
            name=TemplateName.audio_report,
            jira_id=task_info.jira_id,
            jira_url=task_info.jira_url,
            header_block=header_block,
            benchmark_blocks=benchmark_blocks,
            benchmark_section_info=benchmark_section_info,
        )

    def _render_eval_report(
        self,
        system_names: list[str],
        eval_artifacts: EvalArtifacts,
        uploaded_box_plots_info: UploadedBoxPlotsInfo,
        task_info: TaskInfo,
        expiration_info: ExpirationInfo,
        audio_report_url: Optional[str],
    ) -> str:
        include_comparison = len(system_names) > 2
        stat_headers = ["Metric", "Winner", "Alternative", "p-value"]
        if include_comparison:
            stat_headers.insert(1, "Comparison")

        configuration_block = self.renderer.render(
            name=TemplateName.eval_report_configuration,
            configurations=eval_artifacts.configuration.systems,
        )
        header_block = self.renderer.render(
            name=TemplateName.eval_report_header,
            system_names=system_names,
            expiration_comment=self._expiration_comment(expiration_info),
        )
        metrics_table = self.renderer.render(
            name=TemplateName.eval_report_table,
            title="Metrics (macro-average across benchmarks)",
            headers=["Metric", *system_names],
            rows=eval_artifacts.summary.metrics_table_row,
        )
        stat_tests_table = self.renderer.render(
            name=TemplateName.eval_report_table,
            title="Statistical Tests (pooled filewise across benchmarks)",
            headers=stat_headers,
            rows=eval_artifacts.summary.stat_test_table_row,
        )
        stat_tests_analysis = self.renderer.render(
            name=TemplateName.eval_report_stat_analysis,
            winner=eval_artifacts.summary.stat_tests_analysis_info.winner,
            advantages=eval_artifacts.summary.stat_tests_analysis_info.advantages,
            multiple_systems=include_comparison,
        )
        image_block = self.renderer.render(
            name=TemplateName.eval_report_image,
            image_url=uploaded_box_plots_info.summary_url,
        )
        summary_block = self.renderer.render(
            name=TemplateName.eval_report_block,
            is_summary=True,
            metrics_table=metrics_table,
            stat_tests_table=stat_tests_table,
            stat_tests_analysis=stat_tests_analysis,
            image_block=image_block,
        )
        benchmark_blocks, benchmark_section_info = [], []

        for benchmark_name in sorted(eval_artifacts.benchmarks):
            result = eval_artifacts.benchmarks[benchmark_name]
            metrics_table = self.renderer.render(
                name=TemplateName.eval_report_table,
                title="Metrics",
                headers=["Metric", *system_names],
                rows=result.metrics_table_row,
            )
            stat_tests_table = self.renderer.render(
                name=TemplateName.eval_report_table,
                title="Statistical Tests",
                headers=stat_headers,
                rows=result.stat_test_table_row,
            )
            stat_tests_analysis = self.renderer.render(
                name=TemplateName.eval_report_stat_analysis,
                winner=result.stat_tests_analysis_info.winner,
                advantages=result.stat_tests_analysis_info.advantages,
                multiple_systems=include_comparison,
            )
            image_block = self.renderer.render(
                name=TemplateName.eval_report_image,
                image_url=uploaded_box_plots_info.benchmark_urls[benchmark_name],
            )
            benchmark_blocks.append(
                self.renderer.render(
                    name=TemplateName.eval_report_block,
                    is_summary=False,
                    title=benchmark_name,
                    section_id=benchmark_name,
                    metrics_table=metrics_table,
                    stat_tests_table=stat_tests_table,
                    stat_tests_analysis=stat_tests_analysis,
                    image_block=image_block,
                )
            )
            benchmark_section_info.append((benchmark_name, f"{benchmark_name} ({BENCHMARK_META[benchmark_name]})"))

        return self.renderer.render(
            name=TemplateName.eval_report,
            is_self_comparison=eval_artifacts.is_self_comparison,
            jira_id=task_info.jira_id,
            jira_url=task_info.jira_url,
            audio_report_url=audio_report_url,
            configuration_block=configuration_block,
            header_block=header_block,
            summary_block=summary_block,
            benchmark_blocks=benchmark_blocks,
            benchmark_section_info=benchmark_section_info,
            multiple_systems=include_comparison,
        )

    def run(
        self,
        baseline_name: str,
        candidate_name: str | list[str],
        baseline_path: Path,
        candidate_path: Path | list[Path],
        benchmarks: list[str],
        generate_audio_report: bool,
        audio_report_benchmarks: Optional[list[str]],
        samples_per_benchmark: int,
        task_id: str,
        s3_prefix_override: Optional[str] = None,
    ) -> tuple[str, Optional[str]]:
        """Generate N-system reports, publish all artifacts, and return their locations."""
        benchmark_names = tuple(sorted(benchmarks, key=len, reverse=True))
        candidate_names = self._normalize_candidates(candidate_name)
        candidate_paths = self._normalize_candidates(candidate_path)
        system_names = [baseline_name, *candidate_names]
        buckets = self._load_buckets(
            baseline_name,
            candidate_names,
            baseline_path,
            candidate_paths,
            benchmark_names,
            check_audio=generate_audio_report,
        )

        task_info = make_task_info(task_id)
        expiration_info = make_expiration_info(S3_LINK_EXPIRES_IN)
        if self.local_output_dir is not None:
            self.local_output_dir.mkdir(parents=True, exist_ok=True)
            artifact_prefix = ""
        else:
            artifact_prefix = generate_s3_prefix(
                baseline_path,
                candidate_paths,
                task_info,
                expiration_info,
                override=s3_prefix_override,
            )
        box_plots_cfg = BoxPlotsConfig()
        bucket_baseline, candidate_buckets = buckets[0], buckets[1:]

        self._log_info("\nPreparing evaluation artifacts...")
        eval_artifacts = prepare_eval_artifacts(bucket_baseline, candidate_buckets, box_plots_cfg)
        destination = "local preview" if self.local_output_dir is not None else "S3"
        self._log_info(f"\nPublishing images to {destination}:")
        uploaded_box_plots_info = self._upload_boxplots(eval_artifacts, artifact_prefix)

        audio_report_url = None
        if generate_audio_report:
            if audio_report_benchmarks is None:
                raise ValueError("Audio report benchmarks must be provided when audio report is enabled.")
            audio_pairs = prepare_audio_pairs(
                bucket_baseline,
                candidate_buckets,
                self.bucket_structure,
                audio_report_benchmarks,
                samples_per_benchmark,
            )
            self._log_info(f"\nPublishing audio files to {destination}:")
            uploaded_audio_info = self._upload_audio(audio_report_benchmarks, audio_pairs, artifact_prefix)
            self._log_info("\nPreparing audio report...")
            audio_report = self._render_audio_report(
                system_names,
                audio_report_benchmarks,
                uploaded_audio_info,
                task_info,
                expiration_info,
            )
            audio_report_url = self._upload_report(audio_report, artifact_prefix, "audio_report")

        self._log_info("\nPreparing evaluation report...")
        eval_report = self._render_eval_report(
            system_names,
            eval_artifacts,
            uploaded_box_plots_info,
            task_info,
            expiration_info,
            audio_report_url,
        )
        eval_report_url = self._upload_report(eval_report, artifact_prefix, "eval_report")
        if self.local_output_dir is not None:
            self._log_info(f"\nWrote local preview to '{self.local_output_dir}'.")
        else:
            if self.s3_client is None:
                raise RuntimeError("S3 client is not configured.")
            self._log_info(
                f"\nUploaded artifacts to bucket '{self.s3_client.cfg.bucket}' with prefix '{artifact_prefix}'."
            )
        return eval_report_url, audio_report_url
