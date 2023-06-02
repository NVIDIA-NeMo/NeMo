# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
This script contains an example on how to run inference with the SpellcheckingAsrCustomizationModel.

An input line should consist of 4 tab-separated columns:
    1. text of ASR-hypothesis
    2. texts of 10 candidates separated by semicolon
    3. 1-based ids of non-dummy candidates
    4. approximate start/end coordinates of non-dummy candidates (correspond to ids in third column)

Example input (in one line):
    t h e _ t a r a s i c _ o o r d a _ i s _ a _ p a r t _ o f _ t h e _ a o r t a _ l o c a t e d _ i n _ t h e _ t h o r a x	
    h e p a t i c _ c i r r h o s i s;u r a c i l;c a r d i a c _ a r r e s t;w e a n;a p g a r;p s y c h o m o t o r;t h o r a x;t h o r a c i c _ a o r t a;a v f;b l o c k a d e d
    1 2 6 7 8 9 10
    CUSTOM 6 23;CUSTOM 4 10;CUSTOM 4 15;CUSTOM 56 62;CUSTOM 5 19;CUSTOM 28 31;CUSTOM 39 48

Each line in SpellMapper output is tab-separated and consists of 4 columns:
    1. ASR-hypothesis (same as in input)
    2. 10 candidates separated with semicolon (same as in input)
    3. fragment predictions, separated with semicolon, each prediction is a tuple (start, end, candidate_id, probability)
    4. letter predictions - candidate_id predicted for each letter (this is only for debug purposes)

Example output (in one line):
    t h e _ t a r a s i c _ o o r d a _ i s _ a _ p a r t _ o f _ t h e _ a o r t a _ l o c a t e d _ i n _ t h e _ t h o r a x
    h e p a t i c _ c i r r h o s i s;u r a c i l;c a r d i a c _ a r r e s t;w e a n;a p g a r;p s y c h o m o t o r;t h o r a x;t h o r a c i c _ a o r t a;a v f;b l o c k a d e d
    56 62 7 0.99998;4 20 8 0.95181;12 20 8 0.44829;4 17 8 0.99464;12 17 8 0.97645
    8 8 8 0 8 8 8 8 8 8 8 8 8 8 8 8 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 7 7 7 7 7 7    
   

USAGE Example:
1. Train a model, or use a pretrained checkpoint.
2. Run on a single file:
    python nemo/examples/nlp/spellchecking_asr_customization/spellchecking_asr_customization_infer.py \
        pretrained_model=${PRETRAINED_NEMO_CHECKPOINT} \
        model.max_sequence_len=512 \
        inference.from_file=input.txt \
        inference.out_file=output.txt \
        inference.batch_size=16 \
        lang=en
or on multiple files:
    python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/spellchecking_asr_customization_infer.py \
        pretrained_model=${PRETRAINED_NEMO_CHECKPOINT} \
        model.max_sequence_len=512 \
        +inference.from_filelist=filelist.txt \
        +inference.output_folder=output_folder \
        inference.batch_size=16 \
        lang=en

This script uses the `/examples/nlp/spellchecking_asr_customization/conf/spellchecking_asr_customization_config.yaml`
config file by default. The other option is to set another config file via command
line arguments by `--config-name=CONFIG_FILE_PATH'.
"""


import os

from helpers import MODEL, instantiate_model_and_trainer
from omegaconf import DictConfig, OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="conf", config_name="spellchecking_asr_customization_config")
def main(cfg: DictConfig) -> None:
    logging.debug(f'Config Params: {OmegaConf.to_yaml(cfg)}')

    if cfg.pretrained_model is None:
        raise ValueError("A pre-trained model should be provided.")
    _, model = instantiate_model_and_trainer(cfg, MODEL, False)

    if cfg.model.max_sequence_len != model.max_sequence_len:
        model.max_sequence_len = cfg.model.max_sequence_len
        model.builder._max_seq_length = cfg.model.max_sequence_len
    input_filenames = []
    output_filenames = []

    if "from_filelist" in cfg.inference and "output_folder" in cfg.inference:
        filelist_file = cfg.inference.from_filelist
        output_folder = cfg.inference.output_folder
        with open(filelist_file, "r", encoding="utf-8") as f:
            for line in f:
                path = line.strip()
                input_filenames.append(path)
                folder, name = os.path.split(path)
                output_filenames.append(os.path.join(output_folder, name))
    else:
        text_file = cfg.inference.from_file
        logging.info(f"Running inference on {text_file}...")
        if not os.path.exists(text_file):
            raise ValueError(f"{text_file} not found.")
        input_filenames.append(text_file)
        output_filenames.append(cfg.inference.out_file)

    dataloader_cfg = {
        "batch_size": cfg.inference.get("batch_size", 8),
        "num_workers": cfg.inference.get("num_workers", 4),
        "pin_memory": cfg.inference.get("num_workers", False),
    }
    for input_filename, output_filename in zip(input_filenames, output_filenames):
        if not os.path.exists(input_filename):
            logging.info(f"Skip non-existing {input_filename}.")
            continue
        model.infer(dataloader_cfg, input_filename, output_filename)
        logging.info(f"Predictions saved to {output_filename}.")


if __name__ == "__main__":
    main()
