# Copyright (c) 2025, NVIDIA CORPORATION.
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
import os

import requests
from github import Github


def send_slack_notification():
    # Get environment variables
    gh_token = os.environ.get('GH_TOKEN')
    webhook_url = os.environ.get('SLACK_WEBHOOK')
    repository = os.environ.get('REPOSITORY')
    run_id = os.environ.get('RUN_ID')
    server_url = os.environ.get('SERVER_URL', 'https://github.com')
    pr_number = int(os.environ.get('PR_NUMBER'))

    # Get failure info from GitHub API
    gh = Github(gh_token)
    repo = gh.get_repo(repository)
    pr = repo.get_pull(pr_number)

    # Get failed jobs
    failed_jobs = [job.name for job in repo.get_workflow_run(int(run_id)).jobs() if job.conclusion == 'failure']

    # Build message blocks
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*<{server_url}/{repository}/pull/{pr_number}|PR#{pr_number}: {pr.title.replace('`', '')}>*\n"
                    f"• Author: <{server_url}/{pr.user.login}|{pr.user.login}>\n"
                    f"• Branch: <{server_url}/{pr.head.repo.full_name}/tree/{pr.head.ref}|{pr.head.ref}>\n"
                    f"• Pipeline: <{server_url}/{repository}/actions/runs/{run_id}|View Run>\n"
                    f"• Failed Jobs:\n"
                    + "\n".join(
                        [
                            f"    • <{server_url}/{repository}/actions/runs/{run_id}|{job.split('/')[-1]}>"
                            for job in failed_jobs
                            if job.split('/')[-1] != 'Nemo_CICD_Test'
                        ]
                    )
                ),
            },
        }
    ]

    print({"blocks": blocks})

    # Send to Slack
    response = requests.post(webhook_url, json={"blocks": blocks})
    response.raise_for_status()


if __name__ == "__main__":
    send_slack_notification()
