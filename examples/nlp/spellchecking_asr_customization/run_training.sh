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

## TRAIN WITH NON-TARRED DATA 

# Path to NeMo repository
NEMO_PATH=NeMo

## Download repo with training data (very small example)
## If clone doesn't work, run "git lfs install" and try again
git clone https://huggingface.co/datasets/bene-ges/spellmapper_en_train_micro

DATA_PATH=spellmapper_en_train_micro

## Example of all files needed to run training with non-tarred data:
## spellmapper_en_train_micro
##   ├── config.json
##   ├── label_map.txt
##   ├── semiotic_classes.txt
##   ├── test.tsv
##   └── train.tsv

## To generate files config.json, label_map.txt, semiotic_classes.txt - run generate_configs.sh
## Files "train.tsv" and "test.tsv" contain training examples. 
## For data preparation see https://github.com/bene-ges/nemo_compatible/blob/main/scripts/nlp/en_spellmapper/dataset_preparation/build_training_data.sh

## Note that training with non-tarred data only works on single gpu. It makes sense if you use 1-2 million examples or less.

python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/spellchecking_asr_customization_train.py \
  lang="en" \
  data.validation_ds.data_path=${DATA_PATH}/test.tsv \
  data.train_ds.data_path=${DATA_PATH}/train.tsv \
  data.train_ds.batch_size=32 \
  data.train_ds.num_workers=8 \
  model.max_sequence_len=512 \
  model.language_model.pretrained_model_name=huawei-noah/TinyBERT_General_6L_768D \
  model.language_model.config_file=${DATA_PATH}/config.json \
  model.label_map=${DATA_PATH}/label_map.txt \
  model.semiotic_classes=${DATA_PATH}/semiotic_classes.txt \
  model.optim.lr=3e-5 \
  trainer.devices=[1] \
  trainer.num_nodes=1 \
  trainer.accelerator=gpu \
  trainer.strategy=ddp \
  trainer.max_epochs=5
