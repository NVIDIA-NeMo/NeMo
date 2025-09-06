# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import glob
import logging
import os
import re
from functools import partial

from lhotse import CutSet
from rich import print
from tqdm import tqdm


def batch_replace_and_write(cut_filepath, new_cut_filepath, dataset_name):
    print(f"    Processing {dataset_name}: {cut_filepath} --> {new_cut_filepath}")
    cuts = CutSet.from_file(cut_filepath)
    cuts_with_validation = cuts.map(partial(replace_audio_context_with_text_context, dataset_name=dataset_name))
    cuts_with_validation.to_file(new_cut_filepath)


def replace_audio_context_with_text_context(cut, dataset_name):
    speaker = cut.supervisions[0].speaker
    seg_id = cut.supervisions[0].id
    items = seg_id.split("-")

    if dataset_name == "rivaLindyRodney":
        speaker_suffix = items[4]
    elif dataset_name == "rivaEmmaMeganSeanTom":
        speaker_suffix = "_".join(items[4].split("_")[1:-1])
    elif dataset_name == "jhsdGtc20Amp20Keynote":
        speaker_suffix = items[3]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    text_context = f"Speaker and Emotion: {speaker.rstrip('| ')}_{speaker_suffix} |"
    new_custom = {"context_text": text_context}

    # keep original emotion state if any.
    if cut.supervisions[0].has_custom("emotion"):
        new_custom.update({"emotion": cut.supervisions[0].emotion})

    cut.supervisions[0].custom = new_custom

    return cut


def find_and_verify_shards(cuts_dir: str):
    cuts_shard_pattern = os.path.join(cuts_dir, "cuts.*.jsonl.gz")
    all_cuts_shard_paths = sorted(glob.glob(cuts_shard_pattern))

    if not all_cuts_shard_paths:
        msg = f"No input cut shards found matching pattern: {cuts_shard_pattern}. Cannot proceed."
        logging.error(msg)
        raise FileNotFoundError(msg)

    num_total_shards = len(all_cuts_shard_paths)

    # Verify shard indices are contiguous and start from 0 based on filenames (globally)
    first_idx_str = re.search(r"cuts\.(\d+)\.jsonl\.gz$", all_cuts_shard_paths[0]).group(1)
    last_idx_str = re.search(r"cuts\.(\d+)\.jsonl\.gz$", all_cuts_shard_paths[-1]).group(1)
    first_idx = int(first_idx_str)
    last_idx = int(last_idx_str)
    expected_last_idx = num_total_shards - 1
    if first_idx != 0:
        raise ValueError(f"Expected first shard index to be 0, but found {first_idx} in {all_cuts_shard_paths[0]}")
    if last_idx != expected_last_idx:
        raise ValueError(
            f"Expected last shard index to be {expected_last_idx}, but found {last_idx} in {all_cuts_shard_paths[-1]}"
        )
    logging.info(
        f"Verified {num_total_shards} total shard files globally, with indices from {first_idx} to {last_idx}."
    )
    return all_cuts_shard_paths


if __name__ == "__main__":
    datasets = ["rivaLindyRodney", "rivaEmmaMeganSeanTom", "jhsdGtc20Amp20Keynote"]
    for dataset in datasets:
        cut_dir = f"./model_release_2505/lhotse_shar/{dataset}/lhotse_shar_shuffle_shardSize256/cuts"
        all_cuts_shard_paths = find_and_verify_shards(cut_dir)
        cut_dir_tc = cut_dir + "_textContext"
        os.makedirs(cut_dir_tc, exist_ok=True)

        for cut_filepath in tqdm(all_cuts_shard_paths, total=len(all_cuts_shard_paths)):
            cut_basename = os.path.basename(cut_filepath)
            cut_filepath_tc = os.path.join(cut_dir_tc, cut_basename)
            batch_replace_and_write(cut_filepath, cut_filepath_tc, dataset_name=dataset)

        # validate
        cuts = CutSet.from_file(cut_filepath_tc)
        cuts_list = list()
        for cut in cuts:
            cuts_list.append(cut)
        print(cuts_list[-1])
