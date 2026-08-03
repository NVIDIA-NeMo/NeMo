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
import logging
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from typing import Optional

from paramiko import AutoAddPolicy, SSHClient
from paramiko.sftp_client import SFTPClient
from scripts.tts_comparison_report.reporting import (
    DUMMY_TASK_ID,
    SUPPORTED_BENCHMARK_NAMES,
    TEMPLATES_DIR,
    BaseStorage,
    BucketStructure,
    LocalStorage,
    Orchestrator,
    Renderer,
    S3Client,
    S3Config,
    SFTPStorage,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.captureWarnings(True)


_REMOTE_PASSWORD: str = "REMOTE_PASSWORD"
_S3_ACCESS_KEY_ID: str = "S3_ACCESS_KEY_ID"
_S3_SECRET_ACCESS_KEY: str = "S3_SECRET_ACCESS_KEY"

_DEFAULT_BENCHMARK_NAMES: str = "libritts,riva_en"


def _create_argparser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Script for generating MagpieTTS evaluation comparison reports",
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baseline_name",
        type=str,
        required=True,
        help="Name of the baseline model that will be used in report.",
    )
    parser.add_argument(
        "--baseline_path",
        type=str,
        required=True,
        help="Path to the generated evaluation bucket for the baseline model.",
    )
    parser.add_argument(
        "--candidate_name",
        type=str,
        action="append",
        required=True,
        help="Name of a candidate model used in the report. Repeat with --candidate_path to compare more systems.",
    )
    parser.add_argument(
        "--candidate_path",
        type=str,
        action="append",
        required=True,
        help="Path to a candidate evaluation bucket. Repeat with --candidate_name to compare more systems.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=_DEFAULT_BENCHMARK_NAMES,
        help="Comma-separated list of benchmarks used in the evaluation report.",
    )
    parser.add_argument(
        "--s3_endpoint",
        type=str,
        default=None,
        help="S3 endpoint URL. Required unless --local_output_dir is used.",
    )
    parser.add_argument(
        "--s3_bucket",
        type=str,
        default=None,
        help="S3 bucket for report artifacts. Required unless --local_output_dir is used.",
    )
    parser.add_argument(
        "--s3_region",
        type=str,
        default=None,
        help="AWS region name. Required unless --local_output_dir is used.",
    )
    parser.add_argument(
        "--s3_prefix",
        type=str,
        default=None,
        help="Override the generated S3 directory/key prefix, for example 'codec-comparison-aug-2026'.",
    )
    parser.add_argument(
        "--local_output_dir",
        type=str,
        default=None,
        help="Write a complete local HTML preview and assets here instead of uploading to S3.",
    )
    parser.add_argument(
        "--remote_hostname",
        type=str,
        default=None,
        help="Name of the remote host, if the generated buckets are located there.",
    )
    parser.add_argument(
        "--remote_username",
        type=str,
        default=None,
        help="Name of the user on the remote host.",
    )
    parser.add_argument(
        "--task_id",
        type=str,
        default=DUMMY_TASK_ID,
        help="Jira task number associated with this report.",
    )
    parser.add_argument(
        "--results_subdir",
        type=str,
        default="results",
        help="Subdirectory inside the bucket root that contains evaluation outputs produced by `magpietts_inference`.",
    )
    parser.add_argument(
        "--audio_report",
        action='store_true',
        help="Generate additional report with side-by-side audio comparison.",
    )
    parser.add_argument(
        "--audio_report_benchmarks",
        type=str,
        default=_DEFAULT_BENCHMARK_NAMES,
        help="Comma-separated list of benchmarks to include in the audio report.",
    )
    parser.add_argument(
        "--samples_per_benchmark",
        type=int,
        default=30,
        help="Number of samples per benchmark in the audio report.",
    )
    return parser


def _get_benchmarks_list(benchmarks: str) -> list[str]:
    return [x.strip() for x in benchmarks.split(",") if x.strip()]


def _validate_benchmarks(benchmarks: list[str]) -> None:
    if not benchmarks:
        raise ValueError("Empty list of benchmark names was provided.")

    supported_set = set(SUPPORTED_BENCHMARK_NAMES)

    for name in benchmarks:
        if name not in supported_set:
            raise ValueError(f"Unknown benchmark name: '{name}'.")


def _validate_audio_report_benchmarks(
    benchmarks: list[str],
    audio_report_benchmarks: list[str],
) -> None:
    if not audio_report_benchmarks:
        raise ValueError("Empty list of benchmark names was provided for the audio report.")

    supported_set = set(benchmarks)

    for name in audio_report_benchmarks:
        if name not in supported_set:
            raise ValueError(f"Benchmark name for audio report '{name}' is not included in evaluation benchmarks.")


def _validate_candidate_args(candidate_names: list[str], candidate_paths: list[str]) -> None:
    if len(candidate_names) != len(candidate_paths):
        raise ValueError(
            "Each '--candidate_name' must have a matching '--candidate_path' "
            f"({len(candidate_names)} names and {len(candidate_paths)} paths were provided)."
        )

    if len(set(candidate_names)) != len(candidate_names):
        raise ValueError("Candidate names must be unique.")


def main() -> None:
    """Parse CLI arguments, generate comparison reports, and publish their artifacts.

    This function serves as the command-line entry point for the report
    generation workflow. It validates user input, initializes storage and the
    selected destination, runs the report orchestrator, and logs the results.

    Raises:
        ValueError: If required environment variables are missing or CLI
            arguments are invalid.
        RuntimeError: If report generation or upload does not complete
            successfully.
    """
    logger = logging.getLogger(__name__)

    parser = _create_argparser()
    args = parser.parse_args()

    bucket_structure = BucketStructure()
    bucket_structure.eval_output_subdir = args.results_subdir
    baseline_path = Path(args.baseline_path).resolve()
    _validate_candidate_args(args.candidate_name, args.candidate_path)
    candidate_paths = [Path(path).resolve() for path in args.candidate_path]
    local_output_dir = Path(args.local_output_dir).resolve() if args.local_output_dir is not None else None
    task_id = args.task_id

    storage: BaseStorage
    s3_client: Optional[S3Client] = None
    ssh_client: Optional[SSHClient] = None
    sftp: Optional[SFTPClient] = None
    eval_report_url: Optional[str] = None
    audio_report_url: Optional[str] = None
    audio_report_benchmarks: Optional[list[str]] = None

    if local_output_dir is not None:
        if args.s3_prefix is not None:
            raise ValueError("'--s3_prefix' cannot be used with '--local_output_dir'.")
    else:
        s3_args = {
            "--s3_endpoint": args.s3_endpoint,
            "--s3_bucket": args.s3_bucket,
            "--s3_region": args.s3_region,
        }
        missing_s3_args = [name for name, value in s3_args.items() if value is None]
        if missing_s3_args:
            raise ValueError(f"Missing required S3 arguments: {', '.join(missing_s3_args)}.")

        assert args.s3_endpoint is not None
        assert args.s3_bucket is not None
        assert args.s3_region is not None

        s3_key_id = os.getenv(_S3_ACCESS_KEY_ID)
        s3_secret_key = os.getenv(_S3_SECRET_ACCESS_KEY)
        if s3_key_id is None or s3_secret_key is None:
            raise ValueError(
                f"Environment variables '{_S3_ACCESS_KEY_ID}' and '{_S3_SECRET_ACCESS_KEY}' "
                "must be set for uploading reports to S3."
            )

        s3_cfg = S3Config(
            bucket=args.s3_bucket,
            endpoint_url=args.s3_endpoint,
            region_name=args.s3_region,
        )
        s3_client = S3Client(
            cfg=s3_cfg,
            aws_access_key_id=s3_key_id,
            aws_secret_access_key=s3_secret_key,
        )

    benchmarks = _get_benchmarks_list(args.benchmarks)
    _validate_benchmarks(benchmarks)

    if args.audio_report:
        audio_report_benchmarks = _get_benchmarks_list(args.audio_report_benchmarks)
        _validate_audio_report_benchmarks(benchmarks, audio_report_benchmarks)

        if args.samples_per_benchmark <= 0:
            raise ValueError("Number of samples per benchmark for the audio report must be greater than 0.")

    if task_id == DUMMY_TASK_ID:
        logger.warning("\nWARNING: It is recommended to assign the evaluation report to a specific ticket!")

    if args.baseline_name in args.candidate_name:
        raise ValueError("Baseline and candidate names must be unique.")

    for candidate_path in candidate_paths:
        if baseline_path == candidate_path:
            logger.warning(
                "\nWARNING: Baseline and candidate paths are identical. "
                "Comparison report is not meaningful in this case!"
            )

    candidates_str = ", ".join(f"'{name}'" for name in args.candidate_name)
    logger.info(f"\nComparing baseline '{args.baseline_name}' against candidates {candidates_str}")

    try:
        if args.remote_hostname is not None or args.remote_username is not None:
            if args.remote_username is None:
                raise ValueError("'remote_username' must be provided when using remote access.")

            if args.remote_hostname is None:
                raise ValueError("'remote_hostname' must be provided when using remote access.")

            remote_password = os.getenv(_REMOTE_PASSWORD)

            if remote_password is None:
                raise ValueError(f"Environment variable '{_REMOTE_PASSWORD}' is not set.")

            logger.info(f"\nSetting remote connection with host: {args.remote_hostname}")

            ssh_client = SSHClient()
            ssh_client.set_missing_host_key_policy(policy=AutoAddPolicy())
            ssh_client.connect(
                hostname=args.remote_hostname,
                username=args.remote_username,
                password=remote_password,
            )
            sftp = ssh_client.open_sftp()
            storage = SFTPStorage(sftp)

        else:
            storage = LocalStorage()

        renderer = Renderer(templates_dir=TEMPLATES_DIR)

        orchestrator = Orchestrator(
            bucket_structure=bucket_structure,
            storage=storage,
            s3_client=s3_client,
            renderer=renderer,
            logger=logger,
            local_output_dir=local_output_dir,
        )
        eval_report_url, audio_report_url = orchestrator.run(
            baseline_name=args.baseline_name,
            candidate_name=args.candidate_name,
            baseline_path=baseline_path,
            candidate_path=candidate_paths,
            benchmarks=benchmarks,
            generate_audio_report=args.audio_report,
            audio_report_benchmarks=audio_report_benchmarks,
            samples_per_benchmark=args.samples_per_benchmark,
            task_id=task_id,
            s3_prefix_override=args.s3_prefix,
        )

    finally:
        if sftp is not None:
            sftp.close()

        if ssh_client is not None:
            ssh_client.close()

        if s3_client is not None:
            s3_client.close()

    if eval_report_url is None:
        raise RuntimeError("Failed to generate evaluation report.")

    if args.audio_report and audio_report_url is None:
        raise RuntimeError("Failed to generate audio report.")

    if local_output_dir is not None:
        if audio_report_url is not None:
            logger.info(f"\nLocal audio report:\n{local_output_dir / audio_report_url}")
        logger.info(f"\nLocal evaluation report:\n{local_output_dir / eval_report_url}")
        logger.info("\nServe the output directory over HTTP to inspect the complete preview.\n")
    else:
        if audio_report_url is not None:
            logger.info(f"\nAudio report is available at:\n{audio_report_url}")
        logger.info(f"\nEvaluation report is available at:\n{eval_report_url}")
        logger.info("\nSave the links and open in your browser!\n")


if __name__ == "__main__":
    main()
