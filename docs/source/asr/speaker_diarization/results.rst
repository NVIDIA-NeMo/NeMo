Checkpoints
===========

There are two main ways to load pretrained checkpoints in NeMo as introduced in `loading ASR checkpoints <../results.html#checkpoints>`__.
In speaker diarization, the diarizer loads checkpoints that are passed through the config file. For example: 

Loading Local Checkpoints
---------------------------

Load VAD models

.. code-block:: bash

  pretrained_vad_model='/path/to/vad_multilingual_marblenet.nemo' # local .nemo or pretrained vad model name
  ...
  # pass with hydra config
  config.diarizer.vad.model_path=pretrained_vad_model


Load speaker embedding models

.. code-block:: bash

  pretrained_speaker_model='/path/to/titanet-l.nemo' # local .nemo or pretrained speaker embedding model name
  ...
  # pass with hydra config
  config.diarizer.speaker_embeddings.model_path=pretrained_speaker_model

Load neural diarizer models

.. code-block:: bash

  pretrained_neural_diarizer_model='/path/to/diarizer_msdd_telephonic.nemo' # local .nemo or pretrained neural diarizer model name
  ...
  # pass with hydra config
  config.diarizer.msdd_model.model_path=pretrained_neural_diarizer_model


NeMo will automatically save checkpoints of a model you are training in a `.nemo` format.
You can also manually save your models at any point using :code:`model.save_to(<checkpoint_path>.nemo)`.


Inference
---------

.. note::
  For details and deep understanding, please refer to ``<NeMo_git_root>/tutorials/speaker_tasks/Speaker_Diarization_Inference.ipynb``.

Check out :doc:`Datasets <./datasets>` for preparing audio files and optional label files.

Run and evaluate speaker diarizer with below command:

.. code-block:: bash

  # Have a look at the instruction inside the script and pass the arguments you might need. 
  python <NeMo_git_root>/examples/speaker_tasks/diarization/offline_diarization.py 


NGC Pretrained Checkpoints
--------------------------

The ASR collection has checkpoints of several models trained on various datasets for a variety of tasks.
These checkpoints are obtainable via NGC `NeMo Automatic Speech Recognition collection <https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels>`_.
The model cards on NGC contain more information about each of the checkpoints available.

In general, you can load models with model name in the following format, 

.. code-block:: python

  pretrained_vad_model='vad_multilingual_marblenet'
  pretrained_speaker_model='titanet_large'
  pretrained_neural_diarizer_model='diar_msdd_telephonic'
  ...
  config.diarizer.vad.model_path=retrained_vad_model \
  config.diarizer.speaker_embeddings.model_path=pretrained_speaker_model \
  config.diarizer.msdd_model.model_path=pretrained_neural_diarizer_model

where the model name is the value under "Model Name" entry in the tables below.

Models for Speaker Diarization Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. csv-table::
   :file: data/diarization_results.csv
   :align: left
   :widths: 30, 30, 40
   :header-rows: 1
