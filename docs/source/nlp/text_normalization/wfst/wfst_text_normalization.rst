.. _wfst_tn:

Text Normalization
==================

NeMo Text Normalization converts text from written form into its verbalized form. It is used as a preprocessing step before Text to Speech (TTS). It could also be used for preprocessing Automatic Speech Recognition (ASR) training transcripts.


Quick Start Guide
-----------------

Integrate TN to a text processing pipeline:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # import WFST-based TN module
    from nemo_text_processing.text_normalization.normalize import Normalizer

    # initialize normalizer
    normalizer = Normalizer(input_case="cased", lang="en")

    # try normalizer on a few examples
    print(normalizer.normalize("123"))
    # >>> one hundred twenty three
    print(normalizer.normalize_list(["at 10:00", "it weights 10kg."], punct_post_process=True))
    # >>> ["at ten o'clock", 'it weights ten kilograms.']


Run prediction:
^^^^^^^^^^^^^^^

.. code::

    # run prediction on <INPUT_TEXT_FILE>
    python run_predict.py  --input=<INPUT_TEXT_FILE> --output=<OUTPUT_PATH> --lang=<LANGUAGE> \
        [--input_case=<INPUT_CASE>]

    # single input prediction
    python normalize.py --lang=<LANGUAGE> <INPUT_TEXT> \
        [--verbose] [--overwrite_cache] [--cache_dir=<CACHE_DIR>] [--input_case=<INPUT_CASE>]


``INPUT_CASE`` specifies whether to treat the input as lower-cased or case sensitive. By default treat the input as cased since this is more informative, especially for abbreviations. Punctuation are outputted with separating spaces after semiotic tokens, e.g. `"I see, it is 10:00..."` -> `"I see, it is ten o'clock  .  .  ."`.
Inner-sentence white-space characters in the input are not maintained.


NeMo Text Normalization :cite:`textprocessing-norm-zhang2021nemo` is based on WFST-grammars :cite:`textprocessing-norm-mohri2005weighted` and :cite:`textprocessing-norm-mohri2009weighted`. \
We also provide a deployment route to C++ using `Sparrowhawk <https://github.com/google/sparrowhawk>`_ :cite:`textprocessing-norm-sparrowhawk` -- an open-source version of Google Kestrel :cite:`textprocessing-norm-ebden2015kestrel`.
See :doc:`Text Procesing Deployment <wfst_text_processing_deployment>` for details.


.. note::

    For more details, see the tutorial `NeMo/tutorials/text_processing/Text_Normalization.ipynb <https://github.com/NVIDIA/NeMo/blob/stable/tutorials/text_processing/Text_Normalization.ipynb>`__ in `Google's Colab <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/text_processing/Text_Normalization.ipynb>`_.


Evaluation
----------

Example evaluation run on `Google's text normalization dataset <https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish>`__ :cite:`textprocessing-norm-sproat2016rnn`:

.. code::

    python run_evaluate.py  --input=./en_with_types/output-00001-of-00100 --lang=en \
        [--cat CLASS_CATEGORY] [--input_case INPUT_CASE]
 

Classes
-------

The base class for every grammar is :class:`GraphFst<nemo_text_processing.text_normalization.en.GraphFst>`.
This tool is designed as a two-stage application: 1. `classification` of the input into semiotic tokens and 2. `verbalization` into written form.
For every stage and every semiotic token class there is a corresponding grammar, e.g. :class:`taggers.CardinalFst<nemo_text_processing.text_normalization.en.taggers.cardinal.CardinalFst>`
and :class:`verbalizers.CardinalFst<nemo_text_processing.text_normalization.en.verbalizers.cardinal.CardinalFst>`.
Together, they compose the final grammars :class:`ClassifyFst<nemo_text_processing.text_normalization.en.ClassifyFst>` and
:class:`VerbalizeFinalFst<nemo_text_processing.text_normalization.en.VerbalizeFinalFst>` that are compiled into WFST and used for inference.


.. autoclass:: nemo_text_processing.text_normalization.en.ClassifyFst
    :show-inheritance:
    :members:

.. autoclass:: nemo_text_processing.text_normalization.en.VerbalizeFinalFst
    :show-inheritance:
    :members:

Audio-based Text Normalization
==============================

Quick Start Guide
-----------------

To normalize text that has corresponding audio recording, it is recommened to use `nemo_text_processing/text_normalization/normalize_with_audio.py <https://github.com/NVIDIA/NeMo/blob/stable/nemo_text_processing/text_normalization/normalize_with_audio.py>`__ script \
that provides multiple normalization options and chooses the one that minimizes character error rate (CER) of the automatic speech recognition (ASR) output.
The main difference between the default normalization and the audio-based one, is that most of the semiotic classes use deterministic=False flag.

.. code-block:: python

    # import WFST-based non-deterministic TN module
    from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

    # initialize normalizer
    normalizer = NormalizerWithAudio(
            lang="en",
            input_case="cased",
            overwrite_cache=False,
            cache_dir="cache_dir",
        )
    # try normalizer on a few examples
    print(normalizer.normalize("123", n_tagged=10, punct_post_process=True))
    # >>> {'one hundred twenty three', 'one hundred and twenty three', 'one twenty three', 'one two three'}



To run this script with a .json manifest file, the manifest file should contain the following fields:
Parameters to run audio-based normalization (more details could be found in `nemo_text_processing/text_normalization/normalize_with_audio.py <https://github.com/NVIDIA/NeMo/blob/stable/nemo_text_processing/text_normalization/normalize_with_audio.py>`__)

.. list-table:: Parameters to run audio-based normalization
   :widths: 10 10
   :header-rows: 1

   * - **Parameter**
     - **Description**
   * - **audio_data**
     - path to the audio file
   * - **text**
     - raw text
   * - **pred_text**
     - ASR model prediction
   * - **n_tagged**
     - Number of tagged options to return, -1 - return all possible tagged options


See `examples/asr/transcribe_speech.py <https://github.com/NVIDIA/NeMo/blob/stable/examples/asr/transcribe_speech.py>`__ on how to add ASR predictions.

When the manifest is ready, run:

.. code-block:: python

    python normalize_with_audio.py \
           --audio_data PATH/TO/MANIFEST.JSON \
           --language en


To run with a single audio file, specify path to audio and text with:

    .. code-block:: python

        python normalize_with_audio.py \
               --audio_data PATH/TO/AUDIO.WAV \
               --language en \
               --text raw text OR PATH/TO/.TXT/FILE
               --model QuartzNet15x5Base-En \
               --verbose

To see possible normalization options for a text input without an audio file (could be used for debugging), run:

    .. code-block:: python

        python python normalize_with_audio.py --text "RAW TEXT" --cache_dir "<PATH_TO_CACHE_DIR_TO_STORE_GRAMMARS>"

Specify `--cache_dir` to generate .far grammars once and re-used them for faster inference.

See `nemo_text_processing/text_normalization/normalize_with_audio.py <https://github.com/NVIDIA/NeMo/blob/stable/nemo_text_processing/text_normalization/normalize_with_audio.py>`__ for more arguments.


Supported Languages
-------------------

Deterministic TN supports: English, German and Spanish languages.
Non-deterministic (audio-based) TN supports: English, German, Spanish, and Russian languages.

Installation
------------

`nemo_text_processing` is installed with the `nemo_toolkit`.

See :doc:`NeMo Introduction <../starthere/intro>` for installation details.

Additional requirements can be found in `setup.sh <https://github.com/NVIDIA/NeMo/blob/stable/nemo_text_processing/setup.sh>`_.

References
----------

.. bibliography:: ../tn_itn_all.bib
    :style: plain
    :labelprefix: TEXTPROCESSING-NORM
    :keyprefix: textprocessing-norm-