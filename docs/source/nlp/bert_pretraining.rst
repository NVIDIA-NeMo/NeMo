.. _bert_pretraining:

BERT
====

BERT is an autoencoding language model with a final loss composed of:

- masked language model loss
- next sentence prediction

The model architecture is published in `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`__ :cite:`nlp-bert-devlin2018bert`.
The model is originally trained on English Wikipedia and BookCorpus. BERT is often used as a language model encoder for downstream tasks, for example, :ref:`token_classification`, :ref:`text_classification`, :ref:`question_answering`, etc.
Domain-specific BERT models can be advantageous for a wide range of applications. One notable application is the domain-specific BERT in a biomedical setting,
e.g. BioBERT :cite:`nlp-bert-lee2019biobert` or its improved derivative BioMegatron :cite:`nlp-bert-shin2020biomegatron`. For the latter, refer to :ref:`megatron_finetuning`.

Quick Start Guide
-----------------

.. code-block:: python

    from nemo.collections.nlp.models import BERTLMModel

    # to get the list of pre-trained models
    BERTLMModel.list_available_models()

    # Download and load the pre-trained BERT-based model
    model = BERTLMModel.from_pretrained("bertbaseuncased")

Available Models
^^^^^^^^^^^^^^^^

.. list-table:: *Pretrained Models*
   :widths: 5 10
   :header-rows: 1

   * - Model
     - Pretrained Checkpoint
   * - BERT-base uncased
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:bertbaseuncased
   * - BERT-large uncased
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:bertlargeuncased

.. _dataset_bert_pretraining:

Data Input for the BERT model
-----------------------------

Data preprocessing can be either done on-the-fly during training or offline before training. The latter is optimized and recommended 
for large text corpora. This was also used in the original paper to train the model on Wikipedia and BookCorpus. For on-the-fly data 
processing, provide text files with sentences for training and validation, where words are separated by spaces, i.e.: ``[WORD] [SPACE] [WORD] [SPACE] [WORD]``. 
To use this pipeline in training, use the dedicated configuration file `NeMo/examples/nlp/language_modeling/conf/bert_pretraining_from_preprocessed_config.yaml`.

To process data offline in advance, refer to the `BERT Quick Start Guide <https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#quick-start-guide>`__.
To recreate the original Wikipedia and BookCorpus datasets, follow steps 1-5 in the Quick Start Guide and run the script ``./data/create_datasets_from_start.sh`` inside the Docker container.
The ``downloaded`` folder should include two sub folders ``lower_case_[0,1]_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5``
and ``lower_case_[0,1]_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5``, containing sequences of length 128 with a maximum of 20 masked tokens
and sequences of length 512 with a maximum of 80 masked tokens respectively. To use this pipeline in training, use the dedicated configuration file ``NeMo/examples/nlp/language_modeling/conf/bert_pretraining_from_text_config.yaml`` 
and specify the path to the created hd5f files.


Training the BERT model
-----------------------

Example of model configuration for on-the-fly data preprocessing: `NeMo/examples/nlp/language_modeling/conf/bert_pretraining_from_text_config.yaml <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/language_modeling/conf/bert_pretraining_from_text_config.yaml>`__.
Example of model configuration for offline data preprocessing: `NeMo/examples/nlp/language_modeling/conf/bert_pretraining_from_preprocessed_config.yaml <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/language_modeling/conf/bert_pretraining_from_preprocessed_config.yaml>`__.

The specification can be grouped into three categories:

- Parameters that describe the training process: **trainer**
- Parameters that describe the datasets: **model.train_ds**, **model.validation_ds**
- Parameters that describe the model: **model**, **model.tokenizer**, **model.language_model**

More details about parameters in the config file can be found below:

+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **Parameter**                             | **Data Type**   | **Description**                                                                                              |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.only_mlm_loss**                   | bool            | Only uses masked language model without next sentence prediction.                                            |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **train_ds.data_file**                    | string          | Name of the text file or hdf5 data directory.                                                                |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **train_ds.num_samples**                  | integer         | Number of samples to use from the training dataset, ``-1`` - to use all.                                     |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+

More details about parameters for offline data preprocessing can be found below:

+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **Parameter**                             | **Data Type**   | **Description**                                                                                              |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **train_ds.max_predictions_per_seq**      | integer         | Maximum number of masked tokens in a sequence in the preprocessed data.                                      |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+

More details about parameters for online data preprocessing can be found below:

+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **Parameter**                             | **Data Type**   | **Description**                                                                                              |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.max_seq_length**                  | integer         | The maximum total input sequence length after tokenization.                                                  |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.mask_prob**                       | float           | Probability of masking a token in the input text during data processing.                                     |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.short_seq_prob**                  | float           | Probability of having a sequence shorter than the maximum sequence length.                                   |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+

.. note::

    For offline data preprocessing, **model.tokenizer** is null. For downstream task, use the same tokenizer that was used for 
    offline preprocessing. For online data preprocessing, **model.tokenizer** needs to be specified. See also :ref:`nlp_model` for 
    details.

Example of the command for training the model:

.. code::

    python bert_pretraining.py \
           model.train_ds.data_file=<PATH_TO_DATA>  \
           trainer.max_epochs=<NUM_EPOCHS> \
           trainer.devices=[<CHANGE_TO_GPU(s)_YOU_WANT_TO_USE>] \
           trainer.accelerator='gpu'


Fine-tuning on Downstream Tasks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use a trained BERT model checkpoint on a NeMo NLP downstream task, e.g. :ref:`question_answering`, specify 
:code:`model.language_model.lm_checkpoint=<PATH_TO_CHECKPOINT>`.

References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-BERT
    :keyprefix: nlp-bert-
