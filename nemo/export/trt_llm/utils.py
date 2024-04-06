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


import logging
import os
import pathlib
import tarfile
import tempfile
import typing
import numpy as np
import torch
import yaml

log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
logging.basicConfig(format=log_format)
LOGGER = logging.getLogger("NeMo")

# numpy doesn't know bfloat16, define abstract binary type instead
np_bfloat16 = np.dtype('V2', metadata={"dtype": "bfloat16"})


def unpack_nemo_ckpt(
    nemo_archive_path: typing.Union[str, pathlib.Path], out_dir_path: typing.Union[str, pathlib.Path],
):
    nemo_archive_path = pathlib.Path(nemo_archive_path)
    if not nemo_archive_path.exists():
        raise FileNotFoundError(f"{nemo_archive_path} does not exist")

    for tar_mode in ["r:", "r:gz"]:
        try:
            with tarfile.open(nemo_archive_path, mode=tar_mode) as tar_file:

                def is_within_directory(directory, target):

                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)

                    prefix = os.path.commonprefix([abs_directory, abs_target])

                    return prefix == abs_directory

                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):

                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")

                    tar.extractall(path, members, numeric_owner=numeric_owner)

                safe_extract(tar_file, path=out_dir_path)
            return out_dir_path
        except tarfile.ReadError:
            pass

    raise RuntimeError(f"Could not unpack {nemo_archive_path}")


def prompt_convert(prompt_config, prompt_weights):
    if "task_templates" in prompt_config:
        prompt_templates = prompt_config["task_templates"]
        actual_task_id = 0
        vtokens_embeddings = []
        vtokens_len = []
        for task_name_id, prompt_task in enumerate(prompt_templates):
            prompt_task_name = prompt_task["taskname"]
            LOGGER.info(f"Task {actual_task_id}: {prompt_task['taskname']}")
            prompt_task_weights = prompt_weights["prompt_table"].get(
                f"prompt_table.{prompt_task_name}.prompt_embeddings.weight"
            )
            if prompt_task_weights is None:
                continue
            vtokens_embeddings.append(prompt_task_weights)
            vtokens_len.append(prompt_task_weights.shape[0])
            actual_task_id += 1

        max_vtoken_len = max(vtokens_len)
        embedding_dim = vtokens_embeddings[0].shape[1]

        # pad tasks to longest task embedding table
        for i, vtoken_emb_table in enumerate(vtokens_embeddings):
            padded_table = torch.zeros((max_vtoken_len, embedding_dim))
            padded_table[: vtoken_emb_table.shape[0], :] = vtoken_emb_table
            vtokens_embeddings[i] = padded_table

        vtokens_embeddings = torch.stack(vtokens_embeddings)
    else:
        vtokens_embeddings = prompt_weights["prompt_embeddings_weights"]

    return vtokens_embeddings


def cpu_map_location(storage, loc):
    return storage.cpu()


def is_nemo_file(path):
    flag = False

    if path is not None:
        if len(path) > 5:
            pc = Path(path)
            if pc.exists():
                if pc.is_file():
                    if path[-5 : len(path)] == ".nemo":
                        flag = True

    return flag


def get_prompt_embedding_table(prompt_checkpoint_path):

    with tempfile.TemporaryDirectory() as prompt_out_dir:
        prompt_out_dir = Path(prompt_out_dir)
        unpack_nemo_ckpt(prompt_checkpoint_path, prompt_out_dir)

        model_weights_ckpt = "model_weights.ckpt"
        with open(prompt_out_dir / "model_config.yaml") as f:
            prompt_config = yaml.full_load(f)
        LOGGER.debug(prompt_config)

        weight_path = prompt_out_dir / model_weights_ckpt
        if not weight_path.exists():
            weight_path = prompt_out_dir / "mp_rank_00" / model_weights_ckpt

        prompt_weights = torch.load(weight_path, map_location=cpu_map_location,)

    return prompt_convert(prompt_config, prompt_weights)
