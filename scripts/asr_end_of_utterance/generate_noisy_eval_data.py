# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
This script is used to generate noisy evaluation data for ASR and end of utterance detection.

Example usage:
python generate_noisy_eval_data.py \
    --config-path conf/ \
    --config-name data \
    output_dir=/path/to/output \
    data.manifest_filepath=/path/to/manifest.json \
    data.seed=42 \
    data.noise.manifest_path /path/to/noise_manifest.json

"""

from pathlib import Path
from shutil import rmtree

import lightning.pytorch as pl
import numpy as np
import soundfile as sf
import torch
import yaml
from lhotse.cut import MixedCut
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from nemo.collections.asr.data.audio_to_eou_label_lhotse import LhotseSpeechToTextBpeEOUDataset
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.parts.preprocessing import parsers
from nemo.core.config import hydra_runner
from nemo.utils import logging

# Dummy labels for the tokenizer
labels = [
    " ",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "'",
]


@hydra_runner(config_path="conf/", config_name="data")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    # Seed everything for reproducibility
    seed = cfg.data.get('seed', 42)
    logging.info(f'Setting random seed to {seed}')
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Make output directory
    output_dir = Path(cfg.output_dir)
    if output_dir.exists():
        logging.info(f'Removing existing output directory: {output_dir}')
        rmtree(output_dir)
    logging.info(f'Creating output directory: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dump the config to the output directory
    config = OmegaConf.to_container(cfg, resolve=True)
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    logging.info(f'Config dumped to {output_dir / "config.yaml"}')

    # Load the input manifest
    input_manifest = read_manifest(cfg.data.manifest_filepath)
    logging.info(f'Found {len(input_manifest)} items in input manifest: {cfg.data.manifest_filepath}')
    manifest_parent_dir = Path(cfg.data.manifest_filepath).parent
    if Path(input_manifest[0]["audio_filepath"]).is_absolute():
        output_audio_dir = output_dir / 'wav'
        flatten_audio_path = True
    else:
        output_audio_dir = output_dir
        flatten_audio_path = False

    # Load the dataset
    tokenizer = parsers.make_parser(labels)  # dummy tokenizer
    dataset = LhotseSpeechToTextBpeEOUDataset(
        cfg=cfg.data, tokenizer=tokenizer, return_eou_labels=False, return_cuts=True
    )

    with open_dict(cfg.data):
        cfg.data.force_finite = True
        cfg.data.force_map_dataset = True
        cfg.data.shuffle = False

    dataloader = get_lhotse_dataloader_from_config(
        config=cfg.data,
        global_rank=0,
        world_size=1,
        dataset=dataset,
        tokenizer=tokenizer,
    )

    # Generate noisy evaluation data
    manifest = []
    for i, batch in enumerate(tqdm(dataloader, desc="Generating noisy evaluation data")):
        audio_batch, audio_len_batch, cuts_batch = batch
        audio_batch = audio_batch.cpu().numpy()
        audio_len_batch = audio_len_batch.cpu().numpy()

        for j in range(len(cuts_batch)):
            cut = cuts_batch[j]
            if isinstance(cut, MixedCut):
                cut = cut.first_non_padding_cut

            manifest_item = {}
            for k, v in cut.custom.items():
                if k == "dataloading_info":
                    continue
                manifest_item[k] = v

            audio = audio_batch[j][: audio_len_batch[j]]
            audio_file = cut.recording.sources[0].source

            if flatten_audio_path:
                output_audio_file = output_audio_dir / str(audio_file).replace('/', '_')
            else:
                output_audio_file = output_audio_dir / Path(audio_file).relative_to(manifest_parent_dir)

            output_audio_file.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_audio_file, audio, dataset.sample_rate)

            manifest_item["audio_filepath"] = str(output_audio_file)
            manifest_item["offset"] = 0
            manifest_item["duration"] = audio.shape[0] / dataset.sample_rate

            manifest.append(manifest_item)

    # Write the output manifest
    output_manifest_file = output_dir / Path(cfg.data.manifest_filepath).name
    write_manifest(output_manifest_file, manifest)
    logging.info(f'Output manifest written to {output_manifest_file}')


if __name__ == "__main__":
    main()
