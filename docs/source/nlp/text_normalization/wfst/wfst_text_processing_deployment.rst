.. _wfst_deployment:

NeMo Text Processing Deployment
===============================

NeMo provides a tool for deployment of :doc:`NeMo Inverse Text Normalization (ITN) <wfst_inverse_text_normalization>` and :doc:`NeMo Text Normalization (TN) <wfst_text_normalization>` for production :cite:`textprocessing-deployment-zhang2021nemo`.
It uses `Sparrowhawk <https://github.com/google/sparrowhawk>`_ :cite:`textprocessing-deployment-sparrowhawk` -- an open-source version of Google Kestrel :cite:`textprocessing-deployment-ebden2015kestrel`.
The scripts for deployment could be found at `NeMo/tools/text_processing_deployment <https://github.com/NVIDIA/NeMo/tree/main/tools/text_processing_deployment>`_.

Requirements
------------

:doc:`nemo_text_processing <intro>` package


Usage
-----

Starts docker container with production backend with plugged in grammars. This is entry point script.

Arguments:
^^^^^^^^^
* ``GRAMMARS`` - ``tn_grammars`` or ``itn_grammars`` to export either TN or ITN grammars from :doc:`WFST ITN <wfst_inverse_text_normalization>` or :doc:`WFST TN <wfst_text_normalization>`.
* ``LANGUAGE`` - `en` for English
* ``INPUT_CASE`` - ``cased`` or ``lower_cased`` (lower_cased is supported only in TN grammars).
* ``MODE`` - choose ``test`` to run test on the grammars inside the container.

For example:


.. code-block:: bash

    # to export ITN grammars
    cd NeMo/tools/text_processing_deployment
    bash export_grammar.sh --GRAMMARS=itn_grammars --LANGUAGE=en

    # to export and test TN grammars
    bash export_grammar.sh --GRAMMARS=itn_grammars --INPUT_CASE=cased --MODE=test --LANGUAGE=en

This script runs the following steps in sequence:

Exports grammar `ClassifyFst` and `VerbalizeFst` from :doc:`nemo_text_processing <intro>` to `OUTPUT_DIR/classify/tokenize_and_classify.far` and `OUTPUT_DIR/verbalize/verbalize.far` respectively.

.. code-block:: bash

    cd NeMo/tools/text_processing_deployment
    python pynini_export.py <--output_dir OUTPUT_DIR> <--grammars GRAMMARS> <--input_case INPUT_CASE> <--language LANGUAGE>

Builds C++ production backend docker

.. code-block:: bash

    cd NeMo/tools/text_processing_deployment
    bash docker/build.sh


Plugs in grammars into production backend by mounting grammar directory `classify/` and `verbalize/` with sparrowhawk grammar directory inside docker. Returns docker prompt

.. code-block:: bash

    cd NeMo/tools/text_processing_deployment
    # to launch container with the exported grammars
    bash docker/launch.sh

    # to launch container with the exported grammars and run tests on TN grammars
    bash docker/launch.sh test_tn_grammars

    # to launch container with the exported grammars and run tests on ITN grammars
    bash docker/launch.sh test_itn_grammars


Runs TN or ITN in docker container:

.. code-block:: bash

    echo "two dollars fifty" | ../../src/bin/normalizer_main --config=sparrowhawk_configuration.ascii_proto

This returns $2.50 for ITN.

References
----------

.. bibliography:: ../tn_itn_all.bib
    :style: plain
    :labelprefix: TEXTPROCESSING-DEPLOYMENT
    :keyprefix: textprocessing-deployment-