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
This script is used to create a tarred dataset for SpellcheckingAsrCustomizationModel.

This script uses the `/examples/nlp/spellchecking_asr_customization/conf/spellchecking_asr_customization_config.yaml`
config file by default. The other option is to set another config file via command
line arguments by `--config-name=CONFIG_FILE_PATH'. Probably it is worth looking
at the example config file to see the list of parameters used for training.

USAGE Example:
1. Obtain a processed dataset
2. Run:
    python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/create_tarred_dataset.py \
      lang=${LANG} \
      data.train_ds.data_path=${DATA_PATH}/train.tsv \
      model.language_model.pretrained_model_name=${LANGUAGE_MODEL} \
      model.label_map=${DATA_PATH}/label_map.txt \
      +output_tar_file=tarred/part1.tar \
      +take_first_n_lines=100000

"""
import pickle
import tarfile
from io import BytesIO

from helpers import MODEL, instantiate_model_and_trainer
from omegaconf import DictConfig, OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging


@hydra_runner(config_path="conf", config_name="spellchecking_asr_customization_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config Params: {OmegaConf.to_yaml(cfg)}')
    logging.info("Start creating tar file from " + cfg.data.train_ds.data_path + " ...")
    _, model = instantiate_model_and_trainer(
        cfg, MODEL, True
    )  # instantiate model like for training because we may not have pretrained model
    dataset = model._train_dl.dataset
    archive = tarfile.open(cfg.output_tar_file, mode="w")
    max_lines = int(cfg.take_first_n_lines)
    for i in range(len(dataset)):
        if i >= max_lines:
            logging.info("Reached " + str(max_lines) + " examples")
            break
        (
            input_ids,
            input_mask,
            segment_ids,
            input_ids_for_subwords,
            input_mask_for_subwords,
            segment_ids_for_subwords,
            character_pos_to_subword_pos,
            labels_mask,
            labels,
            spans,
        ) = dataset[i]

        # do not store masks as they are just arrays of 1
        content = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "input_ids_for_subwords": input_ids_for_subwords,
            "input_mask_for_subwords": input_mask_for_subwords,
            "segment_ids_for_subwords": segment_ids_for_subwords,
            "character_pos_to_subword_pos": character_pos_to_subword_pos,
            "labels_mask": labels_mask,
            "labels": labels,
            "spans": spans,
        }
        b = BytesIO()
        pickle.dump(content, b)
        b.seek(0)
        tarinfo = tarfile.TarInfo(name="example_" + str(i) + ".pkl")
        tarinfo.size = b.getbuffer().nbytes
        archive.addfile(tarinfo=tarinfo, fileobj=b)

    archive.close()
    logging.info("Tar file " + cfg.output_tar_file + " created!")


if __name__ == '__main__':
    main()
