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
This script contains an example on how to train, evaluate and perform inference with the question answering model.
The QAModel in NeMo supports extractive question answering problems for data in the SQuAD (https://rajpurkar.github.io/SQuAD-explorer/) format.

***Data format***
The QAModel requires a JSON file for each dataset split. 
In the following we will show example for a training file. Each title has one or multiple paragraph entries, each consisting of the text - "context", and question-answer entries. Each question-answer entry has:
* a question
* a globally unique id
* a boolean flag "is_impossible" which shows if the question is answerable or not
* in case the question is answerable one answer entry, which contains the text span and its starting character index in the context. If not answerable, the "answers" list is empty

The evaluation file follows the above format except for it can provide more than one answers to the same question. 
The inference file follows the above format except for it does not require the "answers" and "is_impossible" keywords.


***Downloading the dataset***
Run ./NeMo/examples/nlp/question_answering/get_squad.py to download the SQuAD dataset:

#   python get_squad.py --destDir=<PATH_TO_DATA>

***Setting the configs***
The model and the PT trainer are defined in a config file which declares multiple important sections.
The most important ones are:
    model: All arguments that are related to the Model - language model, tokenizer, token classifier, optimizer,
            schedulers, and datasets/data loaders.
    trainer: Any argument to be passed to PyTorch Lightning including number of epochs, number of GPUs,
            precision level, etc.

This script uses the `/examples/nlp/question_answering/conf/question_answering_squad_config.yaml` config file
by default. You may update the config file from the file directly. The other option is to set another config file via command line arguments by `--config-name=CONFIG_FILE_PATH'.


***Model Training***
# python question_answering_squad.py
    model.train_ds.file=<TRAIN_JSON_FILE>
    model.validation_ds=<VAL_JSON_FILE>
    trainer.max_epochs=<NUM_EPOCHS>
    trainer.devices=[<CHANGE_TO_GPU_YOU_WANT_TO_USE>]


***Model Evaluation***
Set `do_training=False` in the script and run:

#   python question_answering_squad.py
    model.test_file=<TEST_JSON_FILE>

To load a pretrained checkpoint from cloud prior to training (e.g. for fine-tuning) or evaluation you can set cfg.from_pretrained=<MODEL_NAME>,
e.g. MODEL_NAME='BERTBaseUncasedSQuADv1.1'. You can find all pretrained model names by using 
QAModel.list_available_models(). To load a local checkpoint use qa_model.restore_from(<PATH_TO_CHECKPOINT>)


***Model Inference***
For inference use 
    qa_model.inference(
        file=<INFERENCE_JSON_FILE>,
        batch_size=<BATCH_SIZE>,
        output_nbest_file=<OPTIONAL_OUTPUT_FILE_FOR_NBEST_LIST>,
        output_prediction_file=<OPTIONAL_OUTPUT_FILE_FOR_PREDICTION>
    )

More details on how to use this script can be found in
./NeMo/tutorials/nlp/Question_Answering_Squad.ipynb
"""

import os

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models.question_answering.qa_model import QAModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="question_answering_squad_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(**cfg.trainer)
    exp_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    if cfg.model.nemo_path is not None and os.path.isfile(cfg.model.nemo_path):
        model = QAModel.restore_from(cfg.model.nemo_path, override_config_path=cfg, trainer=trainer)
    elif not cfg.pretrained_model:
        logging.info(f'Config: {OmegaConf.to_yaml(cfg)}')
        model = QAModel(cfg.model, trainer=trainer)
    else:
        logging.info(f'Loading pretrained model {cfg.pretrained_model}')
        model = QAModel.from_pretrained(cfg.pretrained_model)
        if cfg.do_training:
            model.setup_training_data(train_data_config=cfg.model.train_ds)
            model.setup_validation_data(val_data_config=cfg.model.validation_ds)

    if cfg.do_training:
        trainer.fit(model)
        if cfg.model.nemo_path:
            model.save_to(cfg.model.nemo_path)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.file is not None:
        model.setup_test_data(test_data_config=cfg.model.test_ds)
        trainer.test(model)

    # change to path if you want results to be written to file e.g. os.path.join(exp_dir, "output_nbest_file.txt")
    output_nbest_file = None
    # change to path if you want results to be written to file e.g.  os.path.join(exp_dir, "output_prediction_file.txt")
    output_prediction_file = None
    inference_samples = 5  # for test purposes. To use entire inference dataset set to -1
    all_preds, all_nbests = model.inference(
        file=cfg.model.validation_ds.file,
        batch_size=1,
        num_samples=inference_samples,
        output_nbest_file=output_nbest_file,
        output_prediction_file=output_prediction_file,
    )

    for _, item in all_preds.items():
        print(f"question: {item[0]} answer: {item[1]}")


if __name__ == '__main__':
    main()
