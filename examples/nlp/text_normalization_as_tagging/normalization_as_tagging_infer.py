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


"""
This script contains an example on how to run itn inference with the ThutmoseTaggerModel.

The inference works on a raw file (no labels required).
Each line of the input file represents a single example for inference.
    Specify inference.from_file and inference.batch_size parameters.

USAGE Example:
1. Train a model, or use a pretrained checkpoint.
2. Run:
    export TOKENIZERS_PARALLELISM=false
    python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_infer.py \
      pretrained_model=./training.nemo \
      inference.from_file=./input.txt \
      inference.out_file=./output.txt \
      model.max_sequence_len=1024 #\
      inference.batch_size=128

This script uses the `/examples/nlp/text_normalization_as_tagging/conf/thutmose_tagger_itn_config.yaml`
config file by default. The other option is to set another config file via command
line arguments by `--config-name=CONFIG_FILE_PATH'.
"""


import os

from helpers import ITN_MODEL, instantiate_model_and_trainer
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.data.text_normalization_as_tagging.utils import spoken_preprocessing
from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="conf", config_name="thutmose_tagger_itn_config")
def main(cfg: DictConfig) -> None:
    logging.debug(f'Config Params: {OmegaConf.to_yaml(cfg)}')

    if cfg.pretrained_model is None:
        raise ValueError("A pre-trained model should be provided.")
    _, model = instantiate_model_and_trainer(cfg, ITN_MODEL, False)

    text_file = cfg.inference.from_file
    logging.info(f"Running inference on {text_file}...")
    if not os.path.exists(text_file):
        raise ValueError(f"{text_file} not found.")

    with open(text_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    batch_size = cfg.inference.get("batch_size", 8)

    batch, all_preds = [], []
    for i, line in enumerate(lines):
        s = spoken_preprocessing(line)  # this is the same input transformation as in corpus preparation
        batch.append(s.strip())
        if len(batch) == batch_size or i == len(lines) - 1:
            outputs = model._infer(batch)
            for x in outputs:
                all_preds.append(x)
            batch = []
    if len(all_preds) != len(lines):
        raise ValueError(
            "number of input lines and predictions is different: predictions="
            + str(len(all_preds))
            + "; lines="
            + str(len(lines))
        )
    out_file = cfg.inference.out_file
    with open(f"{out_file}", "w", encoding="utf-8") as f_out:
        f_out.write("\n".join(all_preds))
    logging.info(f"Predictions saved to {out_file}.")


if __name__ == "__main__":
    main()
