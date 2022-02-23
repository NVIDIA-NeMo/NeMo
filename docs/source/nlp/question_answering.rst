.. _question_answering:

Question Answering model
========================

With Question Answering, or Reading Comprehension, given a question and a passage of content (context) that may contain an answer for 
the question, the model predicts the span within the text with a start and end position indicating the answer to the question. For 
datasets like SQuAD 2.0, this model supports cases when the answer is not contained in the content.

For every word in the context of a given question, the model is trained to predict:

- The likelihood this word is the start of the span.
- The likelihood this word is the end of the span.

The model chooses the start and end words with maximal probabilities. When the content does not contain the answer, we would like the 
start and end span to be set for the first token.

A pretrained BERT encoder with two span prediction heads is used for the prediction start and the end position of the answer. The span 
predictions are token classifiers consisting of a single linear layer.

Quick Start Guide
-----------------

.. code-block:: python

    from nemo.collections.nlp.models import QAModel

    # to get the list of pre-trained models
    QAModel.list_available_models()

    # Download and load the pre-trained BERT-based model
    model = QAModel.from_pretrained("qa_squadv1.1_bertbase")

    # try the model on a few examples
    model.inference(test_file)

  
.. note::

    We recommend you try Question Answering model in a Jupyter notebook (can run on `Google's Colab <https://colab.research.google.com/notebooks/intro.ipynb>`_.): 
    `NeMo/tutorials/nlp/Question_Answering_Squad.ipynb <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/nlp/Question_Answering_Squad.ipynb>`__.

    Connect to an instance with a GPU (**Runtime** -> **Change runtime type** -> select **GPU** for the hardware accelerator).

    An example script on how to train and evaluate the model can be found here: `NeMo/examples/nlp/question_answering/question_answering_squad.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/question_answering/question_answering_squad.py>`__.

    The default configuration file for the model can be found at: `NeMo/examples/nlp/question_answering/conf/question_answering_squad.yaml <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/question_answering/conf/question_answering_squad_config.yaml>`__.



Available models
^^^^^^^^^^^^^^^^

.. list-table:: *Pretrained Models*
   :widths: 5 10
   :header-rows: 1

   * - Model
     - Pretrained Checkpoint
   * - qa_squadv1.1_bertbase
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:qa_squadv1_1_bertbase
   * - qa_squadv2.0_bertbase
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:qa_squadv2_0_bertbase
   * - qa_squadv1.1_bertlarge
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:qa_squadv1_1_bertlarge
   * - qa_squadv2.0_bertlarge
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:qa_squadv2_0_bertlarge
   * - qa_squadv1.1_megatron_cased
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:qa_squadv1_1_megatron_cased
   * - qa_squadv2.0_megatron_cased
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:qa_squadv2_0_megatron_cased
   * - qa_squadv1.1_megatron_uncased
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:qa_squadv1_1_megatron_uncased
   * - qa_squadv2.0_megatron_uncased
     - https://ngc.nvidia.com/catalog/models/nvidia:nemo:qa_squadv2_0_megatron_uncased


.. _dataset_question_answering:

Data Format
-----------

This model expects the dataset in `SQuAD format`_ (i.e., a JSON file for each dataset split). The code snippet below shows an example 
of the training file. Each title has one or multiple paragraph entries, each consisting of the "context" and question-answer entries. 
Each question-answer entry has:

- A question
- A globally unique id
- The Boolean flag ``is_impossible``, which shows whether a question is answerable or not:
   - if the question is answerable, one answer entry contains the text span and its starting character index in the context
   - if the question is not answerable, an empty ``answers`` list is provided

.. _SQuAD format: https://rajpurkar.github.io/SQuAD-explorer/

The evaluation files (for validation and testing) follow the above format, except that it can provide more than one answer to the 
same question. The inference file also follows the above format, except that it does not require the ``answers`` and ``is_impossible`` 
keywords.

The following is an example of the data format (JSON file):

.. code::

    {
        "data": [
            {
                "title": "Super_Bowl_50",
                "paragraphs": [
                    {
                        "context": "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.",
                        "qas": [
                            {
                                "question": "Where did Super Bowl 50 take place?",
                                "is_impossible": "false",
                                "id": "56be4db0acb8001400a502ee",
                                "answers": [
                                    {
                                        "answer_start": "403",
                                        "text": "Santa Clara, California"
                                    }
                                ]
                            },
                            {
                                "question": "What was the winning score of the Super Bowl 50?",
                                "is_impossible": "true",
                                "id": "56be4db0acb8001400a502ez",
                                "answers": [
                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }


Dataset Download
----------------

To perform training of the Question Answering model on the SQuAD dataset, you must first download it from `here
<https://rajpurkar.github.io/SQuAD-explorer/>`_ or run:

.. code::

    python get_squad.py 

There are two versions: 

- SQuAD version 1.1, which does not contain questions without the answer and has 100,000+ question-answer pairs on 500+ articles.
- SQuAD version 2.0, which combines the 100,000 questions from SQuAD 1.1 with over 50,000 unanswerable questions. To do well with 
  SQuAD 2.0, a system must not only answer questions when possible, but also determine when no answer is supported by the paragraph 
  and abstain from answering.

After downloading the files, you should have a :code:`squad` data folder that contains the following four files for training and
evaluation:

.. code::
    
    .
    |--squad
         |-- v1.1/train-v1.1.json
         |-- v1.1/dev-v1.1.json
         |-- v2.0/train-v2.0.json
         |-- v2.0/dev-v2.0.json


.. _model_training_question_answering:

Model Training
--------------

In the Question Answering Model, we are training a span prediction head on top of a pre-trained language model, such as 
`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`__ :cite:`nlp-qa-devlin2018bert`.
Unless the user provides a pre-trained checkpoint for the language model, the language model is initialized with the pre-trained model 
from `HuggingFace Transformers <https://github.com/huggingface/transformers>`__.

Example of model configuration file for training the model can be found at: `NeMo/examples/nlp/question_answering/conf/question_answering_squad_config.yaml <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/question_answering/conf/question_answering_squad_config.yaml>`__.

The specification can be grouped into three categories:

- Parameters that describe the training process: **trainer**
- Parameters that describe the datasets: **model.dataset**, **model.train_ds**, **model.validation_ds**, **model.test_ds**
- Parameters that describe the model: **model**

More details about parameters in the spec file can be found below:

+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **Parameter**                             | **Data Type**   | **Description**                                                                                              |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **pretrained_model**                      | string          | Pretrained QA model model from ``list_available_models()`` or path to a ``.nemo`` file.                      |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **do_training**                           | bool            | If ``true``, starts training, otherwise, skips training and continues with evaluation/inference.             |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.dataset.version_2_with_negative** | bool            | Set to ``true`` to allow examples without an answer, e.g. for SQuAD v2.0.                                    |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.dataset.do_lower_case**           | bool            | If ``true``, converts text to lower case, only import for inference/evaluation.                              |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **model.dataset.use_cache**               | bool            | If ``true``, either loads all preprocessed data from cache or saves preprocessed data for future use.        |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **training_ds.file**                      | string          | The training file path.                                                                                      |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **training_ds.num_samples**               | integer         | The number of samples to use from the training dataset (use ``-1`` to specify all samples).                  |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **validation_ds.file**                    | string          | The validation file path.                                                                                    |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **validation_ds.num_samples**             | integer         | The number of samples to use from the validation dataset (use ``-1`` to specify all samples).                |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **test_ds.file**                          | string          | The test file path (optional).                                                                               |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+
| **test_ds.num_samples**                   | integer         | The number of samples to use from the test dataset (use ``-1`` to specify all samples).                      |
+-------------------------------------------+-----------------+--------------------------------------------------------------------------------------------------------------+

Example of the command for training the model:

.. code::

    python question_answering_squad.py \
           model.train_ds.file=<PATH_TO_TRAIN_DATA_FILE>  \
           model.validation_ds.file=<PATH_TO_VALIDATION_DATA_FILE>  \
           model.dataset.version_2_with_negative=<ALLOW_UNANSWERABLE_SAMPLES>  \
           model.dataset.do_lower_case=<DO_LOWER_CASE> \
           trainer.max_epochs=<NUM_EPOCHS> \
           trainer.devices=[<CHANGE_TO_GPU(s)_YOU_WANT_TO_USE>] \
           trainer.accelerator='gpu'

.. Note:: 
  
   The first time you train, it will take an extra 5-10 minutes to process the dataset. For future training runs, it will use the 
   processed dataset if :code:`model.dataset.use_cache=true`, which is automatically cached in the files in the same directory as
   the data.

Required Arguments for Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :code:`model.train_ds.file`: path to the training file in JSON format
- :code:`model.validation_ds.file`: path to the validation file in JSON format

Fine-tuning Procedure
^^^^^^^^^^^^^^^^^^^^^

Fine-tuning procedure and logs look similar to what's described in the Model Training section, with the addition of the model
that is initially loaded from a previously trained checkpoint, e.g. by specifying :code:`pretrained_model=<PRETRAINED_MODEL_NAME>`.

Inference
---------

An example script on how to run inference can be found at `examples/nlp/question_answering/question_answering_squad.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/question_answering/question_answering_squad.py>`_.

To run inference with the pre-trained model, run:

.. code::

    python question_answering_squad.py \
           pretrained_model=<PRETRAINED_MODEL> \
           model.dataset.version_2_with_negative=<ALLOW_UNANSWERABLE_SAMPLES>  \
           model.dataset.do_lower_case=<DO_LOWER_CASE>  \
           do_training=false \
           model.validation_ds.file=<PATH_TO_INFERENCE_DATA_FILE>

Required Arguments for inference:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :code:`pretrained_model`: pretrained QA Model model from ``list_available_models()`` or path to a ``.nemo`` file

Model Evaluation
----------------

An example script on how to evaluate the pre-trained model, can be found at `examples/nlp/question_answering/question_answering_squad.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/nlp/question_answering/question_answering_squad.py>`_.

To run evaluation of the pre-trained model, run:

.. code::

    python question_answering_squad.py \
           pretrained_model=<PRETRAINED_MODEL> \
           model.dataset.version_2_with_negative=<ALLOW_UNANSWERABLE_SAMPLES>  \
           model.dataset.do_lower_case=<DO_LOWER_CASE>  \
           do_training=false \
           model.test_ds.file=<PATH_TO_TEST_DATA_FILE>


Required Arguments:
^^^^^^^^^^^^^^^^^^^

- :code:`pretrained_model`: pretrained QA model from `list_available_models()`` or path to a ``.nemo`` file
- :code:`model.test_ds.file`: path to test file

During evaluation of the :code:`test_ds`, the script generates the following metrics:

- :code:`Exact Match (EM)`
- :code:`F1`

More details about these metrics can be found `here <https://en.wikipedia.org/wiki/F-score>`__.

References
----------

.. bibliography:: nlp_all.bib
    :style: plain
    :labelprefix: NLP-QA
    :keyprefix: nlp-qa-
