# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Export a durable SpeechLM2 DPO model checkpoint for standard vLLM evaluation."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from nemo.collections.speechlm2.dpo.export import DEFAULT_SHARD_BYTES, export_dcp_to_hf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dcp", required=True, type=Path)
    parser.add_argument("--serving-baseline", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard-bytes", default=DEFAULT_SHARD_BYTES, type=int)
    parser.add_argument("--failure-report", type=Path)
    args = parser.parse_args()
    try:
        report = export_dcp_to_hf(
            candidate_dcp=args.candidate_dcp,
            serving_baseline=args.serving_baseline,
            trajectory=args.trajectory,
            output=args.output,
            shard_bytes=args.shard_bytes,
        )
    except BaseException as error:
        if args.failure_report is not None:
            args.failure_report.parent.mkdir(parents=True, exist_ok=True)
            args.failure_report.write_text(
                json.dumps(
                    {
                        "schema": "speechlm2.dpo.full-dcp-serving-export-failure.v1",
                        "status": "failed",
                        "candidate_dcp": str(args.candidate_dcp),
                        "output": str(args.output),
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
