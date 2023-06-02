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

## TRAIN WITH TARRED DATA

# Path to NeMo repository
NEMO_PATH=NeMo

DATA_PATH=data_folder

## data_folder_example
##   ├── train_tarred
##   |   ├── part1.tar
##   |   ├── ...
##   |   └── part200.tar
##   ├── config.json
##   ├── label_map.txt
##   ├── semiotic_classes.txt
##   └── test.tsv
## To generate files config.json, label_map.txt, semiotic_classes.txt, run generate_configs.sh
## To prepare data, see ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/build_training_data.sh
## To convert data to tarred format, split all.tsv to pieces of 110'000 examples (except for validation part) and use ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/dataset_preparation/convert_data_to_tarred.sh
## To run training with tarred data, use ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/run_training_tarred.sh

## ATTENTION: How to calculate model.optim.sched.max_steps:
##   Suppose, you have 2'000'000 training examples, and want to train for 5 epochs on 4 gpus with batch size 32.
##   5 (epochs) * 32 (bs) * 4 (gpus)
##   1 step consumes 128 examples (32(bs) * 4(gpus))
##   1 epoch makes 2000000/128=15625 steps (updates)
##   5 epochs make 5*15625=78125 steps

python ${NEMO_PATH}/examples/nlp/spellchecking_asr_customization/spellchecking_asr_customization_train.py \
  lang="en" \
  data.validation_ds.data_path=${DATA_PATH}/test.tsv \
  data.train_ds.data_path=${DATA_PATH}/train_tarred/part_OP_1..100_CL_.tar \
  data.train_ds.batch_size=32 \
  data.train_ds.num_workers=16 \
  +data.train_ds.use_tarred_dataset=true \
  data.train_ds.shuffle=false \
  data.validation_ds.batch_size=16 \
  model.max_sequence_len=512 \
  model.language_model.pretrained_model_name=huawei-noah/TinyBERT_General_6L_768D \
  model.language_model.config_file=${DATA_PATH}/config.json \
  model.label_map=${DATA_PATH}/label_map.txt \
  model.semiotic_classes=${DATA_PATH}/semiotic_classes.txt \
  model.optim.sched.name=CosineAnnealing \
  +model.optim.sched.max_steps=195313 \
  trainer.devices=8 \
  trainer.num_nodes=1 \
  trainer.accelerator=gpu \
  trainer.strategy=ddp \
  trainer.max_epochs=5
