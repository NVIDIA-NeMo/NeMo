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

from scripts.tts_comparison_report.reporting.constants import DUMMY_TASK_ID, JIRA_TICKET_URL_PREFIX
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
    """Create task metadata and the corresponding Jira link information.

    Args:
        task_id: Jira task identifier used for the report.

    Returns:
        Task information with the original task ID, derived Jira ID, and Jira URL.
    """
    jira_id = task_id if task_id != DUMMY_TASK_ID else task_id.split("-")[0]
    jira_url = f"{JIRA_TICKET_URL_PREFIX}/{jira_id}"

    return TaskInfo(
        task_id=task_id,
        jira_id=jira_id,
        jira_url=jira_url,
    )


def generate_s3_prefix(
    baseline_path: Path,
    candidate_path: Path | list[Path],
    task_info: TaskInfo,
    expiration_info: ExpirationInfo,
    override: str | None = None,
) -> str:
    """Generate the S3 prefix used to store report artifacts.

    Args:
        baseline_path: Path to the baseline bucket root.
        candidate_path: Path or paths to candidate bucket roots.
        task_info: Task metadata.
        expiration_info: Expiration metadata.
        override: Optional explicit S3 key prefix.

    Returns:
        S3 key prefix for uploaded report artifacts.
    """
    if override is not None:
        prefix = override.strip("/")
        if not prefix:
            raise ValueError("S3 prefix override must not be empty.")
        if any(part in {".", ".."} for part in prefix.split("/")):
            raise ValueError("S3 prefix override must not contain relative path components.")
        return prefix

    candidate_paths = candidate_path if isinstance(candidate_path, list) else [candidate_path]
    comparison_name = "_vs_".join([baseline_path.stem, *(path.stem for path in candidate_paths)])
    parts = [task_info.task_id, comparison_name, expiration_info.path_str]
    return "-".join(parts)
