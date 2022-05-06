.. _wfst_tn:

Text (Inverse) Normalization
============================

The `nemo_text_processing` Python package :cite:`textprocessing-norm-zhang2021nemo` is based on WFST grammars :cite:`textprocessing-norm-mohri2005weighted` and supports:

1. Text Normalization (TN) converts text from written form into its verbalized form. It is used as a preprocessing step before Text to Speech (TTS). For example,

.. code-block:: bash

     "123" -> "one hundred twenty three"

2. Inverse text normalization (ITN) is a part of the Automatic Speech Recognition (ASR) post-processing pipeline and can be used to convert normalized ASR model outputs into written form to improve text readability. For example,
   
.. code-block:: bash
    
     "one hundred twenty three" -> "123"

3. Audio-based provides multiple normalization options. For example,

.. code-block:: bash
    
     "123" -> "one hundred twenty three", "one hundred and twenty three", "one two three", "one twenty three" ...  

The normalization which best reflects what is actually said in an audio is then picked. 
Audio-based TN can be used to normalize ASR training data.

    .. image:: images/task_overview.png
        :align: center
        :alt: Text TN and ITN
        :scale: 50%


Installation
------------

`nemo_text_processing` is automatically installed with `NeMo <https://github.com/NVIDIA/NeMo>`_.

Quick Start Guide
-----------------


Text Normalization 
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd NeMo/nemo_text_processing/text_normalization/
    python normalize.py --text="123" --language=en


Inverse Text Normalization 
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd NeMo/nemo_text_processing/inverse_text_normalization/
    python inverse_normalize.py --text="one hundred twenty three" --language=en


Arguments:

* ``text`` - Input text.
* ``input_file`` - Input file with lines of input text. Only one of ``text`` or ``input_file`` is accepted.
* ``output_file`` - Output file to save normalizations. Needed if ``input_file`` is specified.
* ``language`` - language id.
* ``input_case`` - Only for text normalization. ``lower_cased`` or ``cased``.
* ``verbose`` - Outputs intermediate information.
* ``cache_dir`` - Specifies a cache directory for compiled grammars. If grammars exist, this significantly improves speed. 
* ``overwrite_cache`` - Updates grammars in cache.
* ``whitelist`` - TSV file with custom mappings of written text to spoken form.



Audio-based TN 
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd NeMo/nemo_text_processing/text_normalization/
    python normalize_with_audio.py --text="123" --language="en" --n_tagged=10 --cache_dir="cache_dir" --audio_data="example.wav" --model="stt_en_conformer_ctc_large" 

Additional Arguments:

* ``text`` - Input text or `JSON manifest file <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/datasets.html#preparing-custom-asr-data>`_ with multiple audio paths.
* ``audio_data`` - (Optional) Input audio.
* ``model`` - `Off-shelf NeMo CTC ASR model name <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/results.html#speech-recognition-languages>`_ or path to local NeMo model checkpoint ending on .nemo
* ``n_tagged`` - number of normalization options to output.


.. note::

    More details can be found in `NeMo/tutorials/text_processing/Text_(Inverse)_Normalization.ipynb <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/text_processing/Text_(Inverse)_Normalization.ipynb>`__ in `Google's Colab <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/text_processing/Text_(Inverse)_Normalization.ipynb>`_.

Language Support Matrix
------------------------

+------------------+----------+----------+----------+--------------------+
| **Language**     | **ID**   | **TN**   | **ITN**  | **Audio-based TN** |
+------------------+----------+----------+----------+--------------------+
| English          | en       | x        | x        | x                  |
+------------------+----------+----------+----------+--------------------+
| Spanish          | es       | x        | x        | x                  |
+------------------+----------+----------+----------+--------------------+
| German           | de       | x        | x        | x                  |
+------------------+----------+----------+----------+--------------------+
| French           | fr       |          | x        |                    |
+------------------+----------+----------+----------+--------------------+
| Russian          | ru       |          | x        | x                  |
+------------------+----------+----------+----------+--------------------+
| Vietnamese       | vi       |          | x        |                    |
+------------------+----------+----------+----------+--------------------+

Grammar customization
---------------------

.. note::

    In-depth walk through `NeMo/tutorials/text_processing/WFST_tutorial.ipynb <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/text_processing/WFST_Tutorial.ipynb>`__ in `Google's Colab <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/text_processing/WFST_Tutorial.ipynb>`_.


Deploy to C++
-----------------
See :doc:`Text Procesing Deployment <wfst_text_processing_deployment>` for details.



References
----------

.. bibliography:: ../tn_itn_all.bib
    :style: plain
    :labelprefix: TEXTPROCESSING-NORM
    :keyprefix: textprocessing-norm-