Speaker Diarization
==================================

Speaker Diarization (SD) is the task of segmenting audio recordings by speaker labels, which is figuring out "Who spoke when?".

.. image:: images/sd_pipeline.png
        :align: center
        :scale: 75%
        :alt: Speaker Diarization pipeline [todo]


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
