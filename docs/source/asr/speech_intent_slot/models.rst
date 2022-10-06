Models
======

There are mainly two approaches in Speech Intent Classification and Slot Filling, where we can either use an End-to-End (E2E) model that directly predicts sematics from audio, or use a cascading model composed of an ASR model followed by an NLU model. E2E methods are preferred over cascading models, since it avoids error propagation from ASR to NLU and thus have better performance.

Our E2E model in NeMo is based on an **Encoder-Decoder** framework, where a Conformer-large module is used as the encoder to extract features, and a Transformer Decoder is applied on top of the features to predict the semantics.

.. image:: images/framework.png
        :align: center
        :scale: 70%
        :alt: sis_framework

The output is a Python dictionary object flattened as a string representation, so that the problem can be formulated as a sequence-to-sequence (audio-to-text) problem.

The model is trained by Negative Log-Likelihood (NLL) Loss with teacher forcing.
