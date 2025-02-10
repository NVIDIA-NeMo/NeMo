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

import json
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Optional

import tqdm
import zstandard

SOURCES_LIST = [
    "RedPajamaCommonCrawl",
    "RedPajamaC4",
    "RedPajamaGithub",
    "RedPajamaBook",
    "RedPajamaArXiv",
    "RedPajamaWikipedia",
    "RedPajamaStackExchange",
]

DEFAULT_APPROVED_SOURCES = [
    "RedPajamaCommonCrawl",
    "RedPajamaC4",
    "RedPajamaGithub",
    "RedPajamaArXiv",
    "RedPajamaWikipedia",
    "RedPajamaStackExchange",
]


def approve_source(filename: str, source_list: list):
    """
    Function to remove data from non approved sources.
    Books data is removed by default due to copyright issues

    Arguments:
        filename: path to jsonl file with the data
        source_list: list of sources that are allowed to be included in the dataset
    """

    with open(filename, "r") as i:
        with open(filename + ".tmp", "w") as o:
            for line in i.read().splitlines():
                j = json.loads(line)
                if j["meta"]["redpajama_set_name"] in source_list:
                    json.dump(j, o)
                    o.write("\n")
    os.rename(filename + ".tmp", filename)
    return


def _split_shards(dataset: list[str], w_size: int) -> list:
    shards = []
    for shard in range(w_size):
        idx_start = (shard * len(dataset)) // w_size
        idx_end = ((shard + 1) * len(dataset)) // w_size
        shards.append(dataset[idx_start:idx_end])
    return shards


def _get_shard_list(data_dir: str, w_size: int, extension: str = "*zst") -> list:
    files = Path(data_dir).rglob(extension)
    files = sorted([str(f) for f in files])
    return _split_shards(files, w_size)


def _extract_single_zst_file(input_path: str, save_dir: str, file_name: str, rm_input: bool = False):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    if os.path.exists(save_path):
        print(f"File {save_path} already exists, skipping extraction.")
        return save_path

    total_length = os.stat(input_path).st_size
    with tqdm.tqdm(
        total=total_length,
        unit="B",
        unit_scale=True,
        desc=file_name,
    ) as pbar:
        dctx = zstandard.ZstdDecompressor()
        read_size = 131075
        write_size = int(read_size * 4)
        save_path = os.path.join(save_dir, file_name)
        update_len = 0
        with open(input_path, "rb") as in_f, open(save_path, "wb") as out_f:
            for chunk in dctx.read_to_iter(in_f, read_size=read_size, write_size=write_size):
                out_f.write(chunk)
                update_len += read_size
                if update_len >= 3000000:
                    pbar.update(update_len)
                    update_len = 0
    if rm_input:
        os.remove(input_path)


def _extract_single_shard(shard_tuple: tuple):
    data_dir, shard, source_list, rm_downloaded = shard_tuple
    file_path = os.path.join(data_dir, shard)
    _extract_single_zst_file(file_path, data_dir, shard[:-4], rm_downloaded)
    shard_path = os.path.join(data_dir, shard[:-4])
    approve_source(shard_path, source_list)


def _run_extraction_on_shard(
    data_dir: str,
    shards_to_extract: list,
    shard_index: int,
    approved_sources: list,
    rm_downloaded: bool = False,
) -> int:
    source_list = []
    if not approved_sources:
        approved_sources = DEFAULT_APPROVED_SOURCES

    for source in approved_sources:
        if source in SOURCES_LIST:
            source_list.append(source)
        else:
            logging.warning(f"Source: {source} is not recognized, should be one of {SOURCES_LIST}")

    print(f"Task :{shard_index} is extracting shards {shards_to_extract[shard_index]}")

    shards_to_process = [(data_dir, shard, source_list, rm_downloaded) for shard in shards_to_extract[shard_index]]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.map(_extract_single_shard, shards_to_process)


def run_extraction(
    data_dir: str,
    rm_downloaded: bool = False,
    approved_sources: Optional[list] = None,
    num_tasks: Optional[int] = None,
    task_id: Optional[int] = None,
):
    """
    Function to download the pile dataset files on Slurm.

    Arguments:
        cfg: main config file.
    conf variables being used:
        data_dir
    """
    if not num_tasks:
        if "SLURM_ARRAY_TASK_COUNT" in os.environ:
            num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
            task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        else:
            num_tasks = 1
            task_id = 0

    shards_to_extract = _get_shard_list(data_dir, num_tasks)
    _run_extraction_on_shard(data_dir, shards_to_extract, task_id, approved_sources, rm_downloaded)
    print(f"Extracted {len(shards_to_extract[task_id])} files")
