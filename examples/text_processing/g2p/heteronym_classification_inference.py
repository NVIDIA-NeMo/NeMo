# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import os
from dataclasses import dataclass, is_dataclass
from typing import Optional

import pytorch_lightning as pl
import torch
from nemo_text_processing.g2p.models.heteronym_classification import HeteronymClassificationModel
from omegaconf import OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging


"""
This script runs inference with HeteronymClassificationModel
If the input manifest contains target "word_id", evaluation will be also performed.

To prepare dataset, see NeMo/scripts/dataset_processing/g2p/export_wikihomograph_data_to_manifest.py

python heteronym_classification_inference.py \
    manifest="<Path to .json manifest>" \
    pretrained_model="<Path to .nemo file or pretrained model name from list_available_models()>" \
    output_file="<Path to .json manifest to save prediction>"
"""


@dataclass
class TranscriptionConfig:
    # Required configs
    pretrained_model: str  # Path to a .nemo file or Name of a pretrained model
    output_file: str  # Path to .json manifest to save prediction, will be saved in "pred_text" field
    manifest: str  # Path to .json manifest
    grapheme_field: str = "text_graphemes"  # name of the field in .json manifest for input grapheme text

    # if "word_id" targets are present in the manifest, evaluation will be performed and errors will be saved in errors_file
    errors_file: Optional[str] = None  # path to a file to save prediction errors
    batch_size: int = 32
    num_workers: int = 0


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if not cfg.pretrained_model:
        raise ValueError(
            'To run evaluation and inference script a pre-trained model or .nemo file must be provided.'
            f'Choose from {HeteronymClassificationModel.list_available_models()} or "pretrained_model"="your_model.nemo"'
        )

    logging.info(
        'During evaluation/testing, it is currently advisable to construct a new Trainer with single GPU and \
			no DDP to obtain accurate results'
    )

    # setup GPU
    if torch.cuda.is_available():
        device = [0]  # use 0th CUDA device
        accelerator = 'gpu'
    else:
        device = 1
        accelerator = 'cpu'

    map_location = torch.device('cuda:{}'.format(device[0]) if accelerator == 'gpu' else 'cpu')
    trainer = pl.Trainer(devices=device, accelerator=accelerator, logger=False, enable_checkpointing=False)

    if os.path.exists(cfg.pretrained_model):
        model = HeteronymClassificationModel.restore_from(cfg.pretrained_model, map_location=map_location)
    elif cfg.pretrained_model in HeteronymClassificationModel.get_available_model_names():
        model = HeteronymClassificationModel.from_pretrained(cfg.pretrained_model, map_location=map_location)
    else:
        raise ValueError(
            f'Provide path to the pre-trained .nemo checkpoint or choose from {HeteronymClassificationModel.list_available_models()}'
        )
    model.set_trainer(trainer)
    model = model.eval()

    logging.info(f'Config Params: {model._cfg}')

    if not os.path.exists(cfg.manifest):
        raise ValueError(f"{cfg.manifest} is not found")

    with torch.no_grad():
        preds = model.disambiguate(
            manifest=cfg.manifest,
            grapheme_field=cfg.grapheme_field,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

    # save predictions to a file
    if cfg.errors_file is None:
        cfg.errors_file = cfg.output_file.replace(".json", "_errors.txt")

    save_errors = True
    correct = 0
    with open(cfg.manifest, "r", encoding="utf-8") as f_in, open(
        cfg.output_file, "w", encoding="utf-8"
    ) as f_preds, open(cfg.errors_file, "w", encoding="utf-8") as f_errors:
        for idx, line in enumerate(f_in):
            line = json.loads(line)
            current_pred = preds[idx]
            line["pred_text"] = current_pred
            f_preds.write(json.dumps(line, ensure_ascii=False) + '\n')

            # run evaluation if target word_id is available in the input manifest
            if "word_id" in line:
                target = line["word_id"]
                if target != current_pred:
                    f_errors.write(f"INPUT: {line[cfg.grapheme_field]}\n")
                    f_errors.write(f"PRED : {current_pred} -- GT: {target}\n")
                    f_errors.write("===========================\n")
                else:
                    correct += 1
            else:
                save_errors = False
    if save_errors:
        logging.info(
            f"Accuracy: {round(correct / len(preds) * 100, 2)}% ({len(preds) - correct} errors out of {len(preds)})"
        )
        logging.info(f"Errors saved at {cfg.errors_file}")
    else:
        logging.info("No 'word_id' values found, skipping evaluation.")
        if os.path.exists(cfg.errors_file):
            os.remove(cfg.errors_file)

    logging.info(f"Predictions saved at {cfg.output_file}")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
