Speaker Diarization
==================================

Speaker Diarization (SD) is the task of segmenting audio recordings by speaker labels, which answers the question "Who spoke when?".

.. image:: images/sd_pipeline.png
        :align: center
        :width: 800px
        :alt: Speaker diarization pipeline- VAD, segmentation, speaker embedding extraction, clustering


A speaker diarization system consists of a **Voice Activity Detection (VAD)** model to get the timestamps of audio where speech is being spoken while ignoring the background noise and 
a **Speaker Embedding extractor** model to get speaker embeddings on speech segments obtained from VAD time stamps. 
These speaker embeddings would then be clustered into clusters based on the number of speakers present in the audio recording.

In NeMo we support both **oracle VAD** and **non-oracle VAD** diarization. 


The full documentation tree is as follows:

.. toctree::
   :maxdepth: 8

   models
   datasets
   results
   configs
   api
   resources

.. include:: resources.rst
