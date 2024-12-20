# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import sys
from datetime import datetime
from pathlib import Path

import fiddle as fdl
from fiddle._src.experimental import serialization

from nemo.lightning.ckpt_utils import ckpt_to_context_subdir
from nemo.lightning.io import drop_unexpected_params, load
from nemo.utils import logging

IO_FILE = "io.json"

"""
Script to update NeMo 2.0 model context (stored in io.json) for unexpected
keword arguments for compatibility with the currently running environment.

It accepts path to a NeMo 2.0 checkpoint and optional flag for building
the updated configuration. It performs the following steps:

1. Loads config from the model context directory.
2. Checks the config for unexpected (e.g. deprecated) arguments and drops them.
3. Attempts to build the updated configuration if --build flag is on.
4. Backs up the existing context file and saves the updated configuration.
"""


def get_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Script to drop unexpected arguments from NeMo 2.0 io.json model context."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to a NeMo 2.0 checkpoint.")
    parser.add_argument("--build", action="store_true", help="Whether to test building the updated config.")
    return parser.parse_args()


def save_io(config: fdl.Config, path: str):
    """
    Saves the given configuration object to a specified file path in JSON format.

    Args:
        config (fdl.Config): The configuration object to be saved.
        path (str): The file path where the configuration will be saved.
    """
    config_json = serialization.dump_json(config)
    with open(path, "w") as f:
        f.write(config_json)


if __name__ == "__main__":
    args = get_args()

    model_path = Path(args.model_path)
    context_path = ckpt_to_context_subdir(model_path)
    logging.info(f"Path to model context: {context_path}.")

    config = load(context_path, build=False)
    updated = drop_unexpected_params(config)

    if not updated:
        logging.info("Config does not need any updates.")
        sys.exit(0)

    if args.build:
        try:
            fdl.build(config)
        except Exception as e:
            logging.error("Build for the updated config failed.")
            raise
        else:
            logging.info("Build for the updated config successful.")

    # Backup the existing context file and save the updated config
    io_path = context_path / IO_FILE
    io_path_backup = context_path / f"BACKUP_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_{IO_FILE}"
    io_path.rename(io_path_backup)
    save_io(config, io_path)
    logging.info(f"Config saved to {io_path}.")
