.. _megatron_finetuning:

NeMo Megatron
=============

Megatron-LM :cite:`nlp-megatron-shoeybi2019megatron` is a large, powerful transformer developed by the Applied Deep Learning Research 
team at NVIDIA. Currently NeMo Megatron supports 3 types of models:

* GPT-style models (decoder only)
* T5/BART-style models (encoder-decoder)
* BERT-style models (encoder only)

.. note::
    We recommend using `NeMo Megatron containers <https://developer.nvidia.com/nemo-megatron-early-access>`_ for pre-training, tuning and running inference with large (1B and above) Megatrons.


Model Parallelism
-----------------

`Megatron-LM <https://github.com/NVIDIA/Megatron-LM>`_ is a highly optimized and efficient library for training large language models.
With Megatron model parallelism, language models can be trained with billions of weights and then used in NeMo for downstream tasks.

NeMo handles pretrained model parallel checkpoints from Megatron-LM automatically and model parallel models in NeMo have the all 
the same features as other NeMo Models.

.. note::

    Currently, NeMo only supports tensor model parallelism.

Training
^^^^^^^^

All of the necessary logic to train model parallel models in NeMo with PyTorch Lightning is contained in the ``NLPDDPPlugin``. 
The ``NLPDDPPlugin`` subclasses the PyTorch Lightning training type plugin ``DDPPlugin``.
See `plugins <https://pytorch-lightning.readthedocs.io/en/latest/extensions/plugins.html>`_ for more information on PyTorch Lightning Plugins.

To enable model parallel training in NeMo:

.. code-block:: python

    trainer = Trainer(plugins=[NLPDDPPlugin()], **cfg.trainer)

Megatron-LM checkpoints have a specific format. One checkpoint is saved for each model parallel rank:

.. code-block:: bash

    iter_0080000/
    ├── mp_rank_00
    │   └── model_optim_rng.pt
    └── mp_rank_01
        └── model_optim_rng.pt


To start fine-tuning from a Megatron-LM checkpoint, simply pass the path to the Megatron-LM checkpoint 
via the language model config:

.. code-block:: bash 

    model.language_model.lm_checkpoint=/raid/megatron/bert/iter_0080000 \

We also need to input the model configuration. This can be done via json:

.. code-block:: json

    {
    "hidden-size": 1024, 
    "num-attention-heads": 16, 
    "num-layers": 24, 
    "max-seq-length": 512
    }

And input via command line:

.. code-block:: bash

    model.language_model.config_file=/raid/data/megatron/bert/config.json \

Or the model configuration can be input via YAML:

.. code-block:: YAML

    model:
        language_model:
            config:
                hidden_size: 1024
                num_attention_heads: 16
                num_layers: 24
                max_position_embeddings: 512

Additionally, Megatron-LM requires a vocab file:

.. code-block:: bash

    model.tokenizer.vocab_file=/path/to/vocab.txt

If using the Megatron-LM default tokenizer for training BERT the vocab file can be omitted:

.. code-block:: bash

    # uncased model
    model.tokenizer.tokenizer_name=megatron-bert-uncased

.. code-block:: bash

    # cased model 
    model.tokenizer.tokenizer_name=megatron-bert-uncased

Auto-Resume
^^^^^^^^^^^

Resuming training with NeMo experiment manager and PyTorch Lightning works exactly the same as other NeMo models.
While training with PTL, model parallel checkpoint will be saved and loaded properly.

.. code-block:: bash

    checkpoints/
    ├── mp_rank_00
    │   ├── mp_autoresume-last.ckpt
    │   ├── mp_autoresume---val_loss=0.35-epoch=0.ckpt
    │   ├── mp_autoresume---val_loss=0.38-epoch=1.ckpt
    │   └── mp_autoresume---val_loss=0.39-epoch=2.ckpt
    └── mp_rank_01
        ├── mp_autoresume-last.ckpt
        ├── mp_autoresume---val_loss=0.35-epoch=0.ckpt
        ├── mp_autoresume---val_loss=0.38-epoch=1.ckpt
        └── mp_autoresume---val_loss=0.39-epoch=2.ckpt

Save and Restore
^^^^^^^^^^^^^^^^

Model parallel .nemo files behave the same as all other .nemo files. Calling ``.save_to`` will save 
a checkpoint for each model parallel rank inside the .nemo file:

.. code-block:: bash

    text_class_350m
    ├── megatron-bert-uncased_encoder_config.json
    ├── megatron_checkpoint_version.json
    ├── model_config.yaml
    ├── mp_rank_00
    │   └── model_weights.ckpt
    ├── mp_rank_01
    │   └── model_weights.ckpt
    ├── tokenizer_vocab_dict.json
    └── tokenizer.vocab_file

When restoring a model parallel .nemo file, we must pass in the ``Trainer`` as model parallel requires DDP:

.. code-block:: python

    model = TokenClassificationModel.restore_from(cfg.pretrained_model, trainer=trainer)

Evaluation
^^^^^^^^^^

Since model parallel models always require more than one GPU, the ``Trainer`` is needed for evaluation:

.. code-block:: python

    trainer = pl.Trainer(plugins=[NLPDDPPlugin()], **cfg.trainer)

    model = TextClassificationModel.restore_from(cfg.model.nemo_path, trainer=trainer)
    model.setup_test_data(test_data_config=cfg.model.test_ds)

    trainer.test(model=model, ckpt_path=None)


Prompt Tuning
-------------

Prompt tuning is a continuous or soft prompt approach to finding the optimal prompt for a specific prompting-based tasks. Instead of selecting discrete text prompts in a manual or automated fashion, prompt tuning utilizes continuous prompt tokens that can be optimized via gradient decent. In addition to increased task performance compared to discrete prompting methods, prompt tuning has been shown to yield performance competitive with finetuning all of a model’s parameters for T5 style models greater than 10B parameters. This is particularly exciting because prompt tuning typically involves tuning parameters amounting to less then 1% of the original model’s size. A model can also be prompt tuned for multiple tasks simultaneously without the risk of over fitting on any one task leading to a degradation in performance on other tasks. With these benefits, prompt tuning can be used as a lighter weight and more flexible alternative to full model finetuning. Prompt tuning can also be used additively with other discrete prompt selection methods.

Implementation Overview
^^^^^^^^^^

Our current prompt tuning implementation adapt’s Lester et. al’s EMNLP 2021 "`The Power of Scale for Parameter-Efficient Prompt Tuning <https://arxiv.org/abs/2104.08691>`_" to prompt tuning for GPT style models. In this implementation, a number of soft tokens specified by the user are prepended to the beginning of the discrete token input embeddings during the forward pass. During training, all model parameters are frozen except for those corresponding to the soft tokens. Only the soft prompt parameters are updated via gradient decent in the backward pass. Each soft token has the same dimensionality as a regular token embedding from the model’s vocabulary corresponding to the ``hidden_size`` hyperparameter. Soft token embeddings can be initialized randomly or with selected existing embeddings from the pretrained model. 

As of NeMo 1.7 prompt tuning now works with tensor parallel > 1. 

Data Formatting
^^^^^^^^^^

The dataset should be a .jsonl file where each json object has 3 fields: ``prompt_tag``, ``text``, and ``answer``.

.. code::

  {"prompt_tag": [tag1], "text": [text1], "answer": [answer1]}
  {"prompt_tag": [tag1], "text": [text2], "answer": [answer2]}
  {"prompt_tag": [tag1], "text": [text3], "answer": [answer3]}
  
.. _data-example-label:

Prompt Tuning Specific Config Values
^^^^^^^^^^
.. list-table:: Prompt Tuning Config Parameters
   :widths: 15 15 25
   :header-rows: 1
   
   * - **Parameter**
     - **Data type**
     - **Description**
   * - **restore_from_path**
     - string
     - Path to a .nemo file for a pretrained GPT model
   * - **model.use_soft_prompts**
     - bool
     - Flag indicating whether to use prompt tags. Must be set to true if doing prompt tuning or if you want to existing prompt tags during inference. 
   * - **model.num_prompt_tokens**
     - int
     - The number of soft prompt tokens that will be initialized and prepended to all model inputs. Must be consistent across all prompt tuning tasks.
   * - **model.new_prompt_tags**
     - list of strings
     - A name associated with the task for which you're currently prompt tuning the model. This is used to prepend the correct soft prompt to a corresponding model input and must match the prompt tag associated with the text inputs for that task. See `Data Formatting`_ for an example. Currently prompt tuning on only one task at a time is                supported, but inference can be performed on multiple tasks at once. 
   * - **model.existing_prompt_tags**
     - list of strings
     - list of existing, already tuned soft prompt tags. Only needs to be set when a model has been prompt tuned on a task previously and you want to tune it on another task.
   * - **model.new_prompt_init_methods**
     - list of strings
     - Either ``['text']`` or ``['random']`` corresponding to initializing soft prompt embeddings from existing token embeddings or randomly. ``['text']`` is recommended. 
   * - **model.new_prompt_init_text**
     - list of strings
     - The text you want to use for soft prompt initalization if ``model.new_prompt_init_methods`` is set to ['text']. The text is tokenized and clipped or tiled to match ``model.num_prompt_tokens``. The vocab embeddings associated with each token are copied and use to initialize the soft prompts.
   * - **model.calc_loss_on_answer_only**
     - bool
     - Whether to calculate cross entropy loss on the full text input or only the answer portion of the input during prompt tuning. 
   * - **model.data.train_ds**
     - string
     - path to training dataset .json or .jsonl file. See `Data Formatting`_ for an example
   * - **model.data.valid_ds**
     - string
     - path to validation dataset .json or .jsonl file. See `Data Formatting`_ for an example
   

Example Prompt Tuning Command for the First Task
^^^^^^^^^^

.. code::
  
  EXPR_NAME='winogrande_prompt_tuning'
  RESTORE_PATH='megatron_gpt.nemo'
  GPUS=1
  MAX_STEPS=1000
  PROMPT_LENGTH=150
  
  echo "Prompt tuning starting"
  python megatron_gpt_prompt_tuning.py \
          --config-name=megatron_gpt_config \
          trainer.devices=$GPUS \
          trainer.accelerator='gpu' \
          trainer.max_steps=$MAX_STEPS \
          restore_from_path=$RESTORE_PATH \
          exp_manager.name=$EXPR_NAME \
          exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
          +model.use_soft_prompts=True \
          +model.num_prompt_tokens=$PROMPT_LENGTH \
          +model.new_prompt_tags=['Winogrande'] \
          +model.new_prompt_init_text=['disambiguate pronoun noun names pick correct name fill blank'] \
          +model.new_prompt_init_methods=['text'] \
          model.data.data_prefix=None \
          +model.data.train_ds='winogrande_prompt_tuning_train.jsonl' \
          +model.data.valid_ds='winogrande_prompt_tuning_val.jsonl' \
          +model.data.batch_size=32 \
          model.optim.lr=2e-3 \
          model.optim.sched.min_lr=2e-6 \
          model.optim.sched.warmup_steps=320 \
          model.optim.sched.constant_steps=2240 \
          model.encoder_seq_length=2048

Example Prompt Tuning Command for the Second Task
^^^^^^^^^^

Be sure to update ``model.existing_prompt_tags`` with tags from previous prompt tuning run
and to use the .nemo file saved at the end of the last prompt tuning run.

.. code::

  EXPR_NAME='rte_prompt_tuning'
  RESTORE_PATH='winogrande_prompt_tuning.nemo'
  GPUS=1
  MAX_STEPS=780
  PROMPT_LENGTH=150
  VAL_CHECK_INTERVAL=50

  echo "Prompt tuning starting"
  python megatron_gpt_prompt_tuning.py \
          --config-name=megatron_gpt_config \
          trainer.devices=$GPUS \
          trainer.accelerator='gpu' \
          trainer.max_steps=$MAX_STEPS \
          trainer.val_check_interval=$VAL_CHECK_INTERVAL \
          restore_from_path=$RESTORE_PATH \
          exp_manager.name=$EXPR_NAME \
          exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True \
          +model.use_soft_prompts=True \
          +model.num_prompt_tokens=$PROMPT_LENGTH \
          +model.existing_prompt_tags=['Winogrande'] \
          +model.new_prompt_tags=['RTE'] \
          +model.new_prompt_init_text=['entailment cause relationship imply label text'] \
          +model.new_prompt_init_methods=['text'] \
          model.data.data_prefix=None \
          +model.data.train_ds='RTE_prompt_tuning_train.jsonl' \
          +model.data.valid_ds='RTE_prompt_tuning_val.jsonl' \
          +model.data.batch_size=32 \
          model.optim.lr=2e-4 \
          model.optim.sched.min_lr=2e-6 \
          model.optim.sched.warmup_steps=78 \
          model.optim.sched.constant_steps=545 \
          model.encoder_seq_length=2048


Example Prompt Tuned Inference
^^^^^^^^^^
The inference file can contain a mix of prompts from all the tasks the model has been prompt tuned on. 

.. code::

    python megatron_gpt_eval.py \
            --use_soft_prompts \
            --model_file=PATH_TO_MODEL \
            --path_to_file=PATH_TO_FILE \
            --tokens_to_generate=32 \
            --batch_size=16 \


Example prompt tuning script: `NeMo/examples/nlp/language_modeling/megatron_gpt_prompt_tuning.py <https://github.com/NVIDIA/NeMo/tree/main/examples/nlp/language_modeling/megatron_gpt_prompt_tuning.py>`__.

Example prompt tuned inference script: `NeMo/examples/nlp/language_modeling/megatron_gpt_eval.py <https://github.com/NVIDIA/NeMo/tree/main/examples/nlp/language_modeling/megatron_gpt_eval.py>`__.

BioMegatron
-----------

BioMegatron has the same network architecture as the Megatron-LM, but is pretrained on a different dataset - `PubMed <https://catalog.data.gov/dataset/pubmed>`_, 
a large biomedical text corpus, which achieves better performance in biomedical downstream tasks than the original Megatron-LM.

Examples of using BioMegatron on biomedical downstream tasks can be found at (can be executed with `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_): 
`NeMo/tutorials/nlp/Relation_Extraction-BioMegatron.ipynb <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/nlp/Relation_Extraction-BioMegatron.ipynb>`__ and `NeMo/tutorials/nlp/Token_Classification-BioMegatron.ipynb <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/nlp/Token_Classification-BioMegatron.ipynb>`__.


References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-MEGATRON
    :keyprefix: nlp-megatron-
