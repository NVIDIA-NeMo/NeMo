# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""
Utilities for using NeMo manifest files with simulstream evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

from nemo.utils import logging


def load_manifest_audio_paths(manifest_path: str | Path) -> list[str]:
    """
    Load audio file paths from a NeMo manifest file.

    Args:
        manifest_path: Path to NeMo manifest JSONL file

    Returns:
        List of audio file paths
    """
    audio_paths = []
    manifest_dir = Path(manifest_path).parent

    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse line {line_num} in manifest: {e}")
                continue
            audio_path = data.get('audio_filepath', data.get('audio_file'))
            if audio_path:
                audio_path = Path(audio_path)
                if not audio_path.is_absolute():
                    audio_path = manifest_dir / audio_path
                audio_paths.append(str(audio_path.resolve()))

    logging.info(f"Loaded {len(audio_paths)} audio files from manifest: {manifest_path}")
    return audio_paths
