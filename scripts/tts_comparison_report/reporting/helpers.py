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
from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.tts_comparison_report.reporting.constants import TICKET_URL_PREFIX
from scripts.tts_comparison_report.reporting.models import ExpirationInfo, TaskInfo


def make_expiration_info(expires_in: int) -> ExpirationInfo:
    """Create formatted expiration metadata for reports and S3 artifact paths.

    Args:
        expires_in: Link lifetime in seconds.

    Returns:
        Expiration information with Unix timestamp and formatted string values.
    """
    expires_at = datetime.now(UTC) + timedelta(seconds=expires_in)

    return ExpirationInfo(
        timestamp=int(expires_at.timestamp()),
        path_str=expires_at.strftime("%Y-%m-%dT%H-%M-%SZ"),
        user_str=expires_at.strftime("%Y-%m-%d %H:%M UTC"),
    )


def make_task_info(task_id: str) -> TaskInfo:
    """Create task metadata and the corresponding link information.

    Args:
        task_id: Task identifier used for the report.

    Returns:
        Task information with the original task ID and task URL.
    """
    task_url = f"{TICKET_URL_PREFIX}/{task_id}"

    return TaskInfo(
        task_id=task_id,
        task_url=task_url,
    )


def generate_s3_prefix(
    baseline_path: Path,
    candidate_path: Path,
    task_info: TaskInfo,
    expiration_info: ExpirationInfo,
) -> str:
    """Generate the S3 prefix used to store report artifacts.

    Args:
        baseline_path: Path to the baseline bucket root.
        candidate_path: Path to the candidate bucket root.
        task_info: Task metadata.
        expiration_info: Expiration metadata.

    Returns:
        S3 key prefix for uploaded report artifacts.
    """
    parts = [
        task_info.task_id,
        f"{baseline_path.stem}_vs_{candidate_path.stem}",
        expiration_info.path_str,
    ]
    return "-".join(parts)
