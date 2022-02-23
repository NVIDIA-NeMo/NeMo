# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
This script contains an example on how to train, evaluate and perform inference with the PTuneTextClassificationModel.
PTuneTextClassificationModel in NeMo supports text classification problems such as sentiment analysis or
domain/intent detection for dialogue systems, as long as the data follows the format specified below.

***Data format***
PTuneTextClassificationModel requires the data to be stored in loose json format with two keys of sentence and
label in each line, i.e. 
{"sentence": "sentence string", "label": "label string"}

For example:

{"sentence": "The output of the contracts totals 72 MWe. ", "label": "neutral"}
{"sentence": "Pretax profit totaled EUR 9.0 mn , down from EUR 36.3 mn in 2007 .", "label": "negative"}
...

If your dataset is stored in another format, you need to convert it to this format to use the PTuneTextClassificationModel.


***Setting the configs***
The model and the PT trainer are defined in a config file which declares multiple important sections.
The most important ones are:
    model: All arguments that are related to the Model - language model, tokenizer, head classifier, optimizer,
            schedulers, and datasets/data loaders.
    trainer: Any argument to be passed to PyTorch Lightning including number of epochs, number of GPUs,
            precision level, etc.

This script uses the `/examples/nlp/text_classification/conf/ptune_text_classification_config.yaml` default config file
by default. You may update the config file from the file directly or by using the command line arguments.
Other option is to set another config file via command line arguments by `--config-name=CONFIG_FILE_PATH'.

You first need to set the classes in the config file which specifies the class types in the dataset.
Notice that some config lines, including `model.dataset.classes`, have `???` as their value, this means that values
for these fields are required to be specified by the user. We need to specify and set the `model.train_ds.file_name`,
`model.validation_ds.file_name`, and `model.test_ds.file_name` in the config file to the paths of the train, validation,
 and test files if they exist. We may do it by updating the config file or by setting them from the command line.


***How to run the script?***
For example the following would train a model for 50 epochs in 2 GPUs on a classification task with 2 classes:

# python ptune_text_classification.py
        model.dataset.classes=[Label1, Label2]
        model.train_ds=PATH_TO_TRAIN_FILE
        model.validation_ds=PATH_TO_VAL_FILE
        trainer.max_epochs=50
        trainer.devices=2

This script would also reload the last checkpoint after the training is done and does evaluation on the dev set,
then performs inference on some sample queries.

By default, this script uses examples/nlp/text_classification/conf/ptune_text_classifciation_config.py config file, and
you may update all the params in the config file from the command line. You may also use another config file like this:

# python ptune_text_classification.py --config-name==PATH_TO_CONFIG_FILE
        model.dataset.num_classes=2
        model.train_ds=PATH_TO_TRAIN_FILE
        model.validation_ds=PATH_TO_VAL_FILE
        trainer.max_epochs=50
        trainer.devices=2

***Load a saved model***
This script would save the model after training into '.nemo' checkpoint file specified by nemo_path of the model config.
You may restore the saved model like this:
    model = PTuneTextClassificationModel.restore_from(restore_path=NEMO_FILE_PATH)

***Evaluation a saved model on another dataset***
# If you wanted to evaluate the saved model on another dataset, you may restore the model and create a new data loader:
    eval_model = TextClassificationModel.restore_from(restore_path=checkpoint_path)

# Then, you may create a dataloader config for evaluation:
    eval_config = OmegaConf.create(
        {'file_path': cfg.model.test_ds.file_path, 'batch_size': 64, 'shuffle': False, 'num_workers': 3}
    )
    eval_model.setup_test_data(test_data_config=eval_config)

# You need to create a new trainer:
    eval_trainer = pl.Trainer(devices=1)
    eval_model.set_trainer(eval_trainer)
    eval_trainer.test(model=eval_model, verbose=False)
"""
import os
import pathlib

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from nemo.collections.nlp.models.text_classification.ptune_text_classification_model import (
    PTuneTextClassificationModel,
)
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="conf", config_name="ptune_text_classification_config")
def main(cfg: DictConfig) -> None:
    logging.info(f'\nConfig Params:\n{OmegaConf.to_yaml(cfg)}')
    trainer = pl.Trainer(plugins=[NLPDDPPlugin()], **cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    if not cfg.model.train_ds.file_path:
        raise ValueError("'train_ds.file_path' need to be set for the training!")

    model = PTuneTextClassificationModel(cfg.model, trainer=trainer)
    logging.info("===========================================================================================")
    logging.info('Starting training...')
    trainer.fit(model)
    logging.info('Training finished!')
    logging.info("===========================================================================================")

    # We evaluate the trained model on the test set if test_ds is set in the config file
    if cfg.model.test_ds.file_path:
        logging.info("===========================================================================================")
        logging.info("Starting the testing of the trained model on test set...")
        trainer.test(model=model, ckpt_path=None, verbose=False)
        logging.info("Testing finished!")
        logging.info("===========================================================================================")

        # extract the path of the best checkpoint from the training, you may update it to any checkpoint
        checkpoint_path = trainer.checkpoint_callback.best_model_path
        tensor_parallel_size = cfg.model.tensor_model_parallel_size
        pathobj = pathlib.Path(checkpoint_path)
        checkpoint_folder = str(pathobj.parent)
        checkpoint_name = str(pathobj.name)

        rank = trainer.accelerator.training_type_plugin.local_rank
        if tensor_parallel_size > 1:
            # inject model parallel rank
            checkpoint_path = os.path.join(checkpoint_folder, f'mp_rank_{rank:02d}', checkpoint_name)
        else:
            checkpoint_path = os.path.join(checkpoint_folder, checkpoint_name)

        # Load the checkpoint
        best_eval_model = PTuneTextClassificationModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path, strict=False, trainer=trainer
        )
        logging.info(f'best checkpoint path: {checkpoint_path}')
        logging.info("Running Test with best EVAL checkpoint!")
        # setup the test dataset
        best_eval_model.setup_test_data(test_data_config=cfg.model.test_ds)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        trainer.test(model=best_eval_model, ckpt_path=None, verbose=False)
        logging.info("Beset EVAL Testing finished!")
        logging.info("===========================================================================================")

    if cfg.model.nemo_path:
        # '.nemo' file contains the last checkpoint and the params to initialize the model
        best_eval_model.save_to(cfg.model.nemo_path)
        logging.info(f'Model is saved into `.nemo` file: {cfg.model.nemo_path}')

    # perform inference on a list of queries.
    if "infer_samples" in cfg.model and cfg.model.infer_samples:
        logging.info("===========================================================================================")
        logging.info("Starting the inference on some sample queries...")

        # max_seq_length=512 is the maximum length BERT supports.
        results = best_eval_model.cuda().classifytext(
            queries=cfg.model.infer_samples, batch_size=1, prompt='Sentiment'
        )
        logging.info('The prediction results of some sample queries with the trained model:')
        for query, result in zip(cfg.model.infer_samples, results):
            logging.info(f'Query : {query}')
            logging.info(f'Predicted label: {result}')

        logging.info("Inference finished!")
        logging.info("===========================================================================================")


if __name__ == '__main__':
    main()
