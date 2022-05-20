Datasets
========

This page is about preparing input dataset for diarization inference. To train or fine-tune the speaker diarization system, the speaker embedding extractor should be trained or fine-tuned separately. Check out page :doc:`Speech Classification Datasets <../speech_classification/datasets>` and :doc:`Speaker Recognition Datasets <../speaker_recognition/datasets>` 
for preparing datasets for training and validating VAD and speaker embedding models respectively. 


Preparation of Input Data
-------------------------

Diarization inference is based on Hydra configurations which are fulfilled by ``.yaml`` files. See `NeMo Speaker Diarization Configuration Files <../configs>`_ for setting up the input Hydra configuration file for speaker diarization. Input data should be provided in line delimited JSON format as below:
	
.. code-block:: bash

  {"audio_filepath": "/path/to/abcd.wav", "offset": 0, "duration": null, "label": "infer", "text": "-", "num_speakers": null, "rttm_filepath": "/path/to/rttm/abcd.rttm", "uem_filepath": "/path/to/uem/abcd.uem"}

In each line of the input manifest file, ``audio_filepath`` item is mandatory while the rest of the items are optional and can be passed for desired diarization setting. We refer to this file as a manifest file. This manifest file can be created by using the script in ``<NeMo_git_root>/scripts/speaker_tasks/pathfiles_to_diarize_manifest.py``. The following example shows how to run ``pathfiles_to_diarize_manifest.py`` by providing path list files.

.. code-block:: bash
   
    python pathfiles_to_diarize_manifest.py --paths2audio_files /path/to/audio_file_path_list.txt \
                                     --paths2txt_files /path/to/transcript_file_path_list.txt \
                                     --paths2rttm_files /path/to/rttm_file_path_list.txt \
                                     --paths2uem_files /path/to/uem_file_path_list.txt \
                                     --paths2ctm_files /path/to/ctm_file_path_list.txt \
                                     --manifest_filepath /path/to/manifest_output/input_manifest.json 

The ``--paths2audio_files`` and ``--manifest_filepath`` are required arguments. Note that we need to maintain consistency on unique filenames for every field (key) by only changing the filename extensions. For example, if there is an audio file named ``abcd.wav``, the rttm file should be named as ``abcd.rttm`` and the transcription file should be named as ``abcd.txt``. 

- Example audio file path list ``audio_file_path_list.txt``
.. code-block:: bash

  /path/to/abcd01.wav
  /path/to/abcd02.wav

- Example RTTM file path list ``rttm_file_path_list.txt``
.. code-block:: bash
  
  /path/to/abcd01.rttm
  /path/to/abcd02.rttm
   

The path list files containing the absolute paths to these WAV, RTTM, TXT, CTM and UEM files should be provided as in the above example. ``pathsfiles_to_diarize_manifest.py`` script will match each file using the unique filename (e.g. ``abcd``). Finally, the absolute path of the created manifest file should be provided through Hydra configuration as shown below:

.. code-block:: yaml
   
	diarizer.manifest_filepath="path/to/manifest/input_manifest.json"

The following are descriptions about each field in an input manifest JSON file.

.. note::
	We expect all the provided files (e.g. audio, rttm, text) to have the same base name and the name should be unique (uniq-id).

``audio_filepath`` (Required):
  
  a string containing absolute path to the audio file.

``num_speakers`` (Optional):
  
  If the number of speakers is known, provide the integer number or assign null if not known. 
	
``rttm_filepath`` (Optional):
  
  To evaluate a diarization system with known rttm files, one needs to provide Rich Transcription Time Marked (RTTM) files as ground truth label files. If RTTM files are provided, the diarization evaluation will be initiated. Here is one line from a RTTM file as an example:

.. code-block:: bash

  SPEAKER TS3012d.Mix-Headset 1 331.573 0.671 <NA> <NA> MTD046ID <NA> <NA>

``text`` (Optional):

  Ground truth transcription for diarization with ASR inference. Provide the ground truth transcription of the given audio file in string format

.. code-block:: bash

  {"text": "this is an example transcript"}

``uem_filepath`` (Optional):

  The UEM file is used for specifying the scoring regions to be evaluated in the given audio file.
  UEMfile follows the following convention: ``<uniq-id> <channel ID> <start time> <end time>``. ``<channel ID>`` is set to 1.

  Example lines of UEM file:

.. code-block:: bash
  
    TS3012d.Mix-Headset 1 12.31 108.98
    TS3012d.Mix-Headset 1 214.00 857.09

``ctm_filepath`` (Optional):
    
  CTM file is used for the evaluation of word-level diarization results and word-timestamp alignment. CTM file follows the following convention: ``<uniq-id> <speaker ID> <word start time> <word end time> <word> <confidence>`` Since confidence is not required for evaluating diarization results, it can have any value. Note that the ``<speaker_id>`` should be exactly matched with speaker IDs in RTTM. 

  Example lines of CTM file:

.. code-block:: bash
  
   TS3012d.Mix-Headset MTD046ID 12.879 0.32 okay 0
   TS3012d.Mix-Headset MTD046ID 13.203 0.24 yeah 0


Evaluation on Benchmark Datasets
--------------------------------

The following instructions can help one to reproduce the expected diarization performance on two benchmark English dialogue datasets. The following results are evaluations based on 0.25 second collar without evaluating overlapped speech. The evaluation is based on oracle VAD results from RTTM files. Therefore, diarization error rate (DER) is equal to confusion error rate since oracle VAD has no miss detection or false alarm.

AMI Meeting Corpus
~~~~~~~~~~~~~~~~~~

The following are the suggested parameters for reproducing the diarization performance for AMI test set.

.. code-block:: bash

  diarizer.manifest_filepath="/path/to/AMItest_input_manifest.json"
  diarizer.oracle_num_speakers=null # Performing unknown number of speaker case 
  diarizer.oracle_vad=True # Use oracle VAD extracted from RTTM files.
  diarizer.collar=0.25
  diarizer.ignore_overlap=True 
  diarizer.speaker_embeddings.model_path="titanet_large"
  diarizer.speaker_embeddings.window_length_in_sec=[3,1.5,1.0,0.5] # Multiscale setting
  diarizer.speaker_embeddings.shift_length_in_sec=[1.5,0.75,0.5,0.25] # Multiscale setting 
  diarizer.speaker_embeddings.parameters.multiscale_weights=[0.4,0.3,0.2,0.1] # More weights on the longer scales

This setup is expected to reproduce a confusion error rate of 1.17% on AMI test set.

To evaluate the performance on AMI Meeting Corpus, the following instructions can help.
  - Download AMI Meeting Corpus from `AMI website <https://groups.inf.ed.ac.uk/ami/corpus/>`_. Choose ``Headset mix`` which has a mono wav file for each session.
  - Download the test set (whitelist) from `Pyannotate AMI test set whitelist <https://raw.githubusercontent.com/pyannote/pyannote-audio/master/tutorials/data_preparation/AMI/MixHeadset.test.lst>`_.
  - The merged RTTM file for AMI test set can be downloaded from `Pyannotate AMI test set RTTM file <https://raw.githubusercontent.com/pyannote/pyannote-audio/master/tutorials/data_preparation/AMI/MixHeadset.test.rttm>`_. Note that this file should be split into individual rttm files. Download split rttm files for AMI test set from `AMI test set split RTTM files <https://raw.githubusercontent.com/tango4j/diarization_annotation/main/AMI_corpus/test/split_rttms.tar.gz>`_.
  - Generate an input manifest file using ``<NeMo_git_root>/scripts/speaker_tasks/pathfiles_to_diarize_manifest.py``


CallHome American English Speech (CHAES), LDC97S42
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use the CH109 set which is a subset of the CHAES dataset which has only two speakers in one session. 
The following are the suggested parameters for reproducing the diarization performance for the CH109 set.

.. code-block:: bash

  diarizer.manifest_filepath="/path/to/ch109_input_manifest.json"
  diarizer.oracle_num_speakers=2 # Since there are exactly 2 speakers per each CH109 session
  diarizer.oracle_vad=True # Use oracle VAD extracted from RTTM files.
  diarizer.collar=0.25
  diarizer.ignore_overlap=True 
  diarizer.speaker_embeddings.model_path="titanet_large"
  diarizer.speaker_embeddings.window_length_in_sec=[1.5,1.0,0.5] # Multiscale setting
  diarizer.speaker_embeddings.shift_length_in_sec=[0.75,0.5,0.25] # Multiscale setting
  diarizer.speaker_embeddings.parameters.multiscale_weights=[0.33,0.33,0.33] # Equal weights

This setup is expected to reproduce a confusion error rate of 0.94% on CH109 set.

To evaluate the performance on AMI Meeting Corpus, the following instructions can help.
  - Download CHAES Meeting Corpus at LDC website `LDC97S42 <https://catalog.ldc.upenn.edu/LDC97S42>`_ (CHAES is not publicly available).
  - Download the CH109 filename list (whitelist) from `CH109 whitelist <https://raw.githubusercontent.com/tango4j/diarization_annotation/main/CH109/ch109_whitelist.txt>`_.
  - Download RTTM files for CH109 set from `CH109 RTTM files <https://raw.githubusercontent.com/tango4j/diarization_annotation/main/CH109/split_rttms.tar.gz>`_.
  - Generate an input manifest file using ``<NeMo_git_root>/scripts/speaker_tasks/pathfiles_to_diarize_manifest.py``

