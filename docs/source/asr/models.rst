Models
======

This section gives a brief overview of the models that NeMo's ASR collection currently supports.

Each of these models can be used with the example ASR scripts (in the ``<NeMo_git_root>/examples/asr`` directory) by
specifying the model architecture in the config file used. Examples of config files for each model can be found in 
the ``<NeMo_git_root>/examples/asr/conf`` directory.

For more information about the config files and how they should be structured, refer to the :doc:`./configs` section.

Pretrained checkpoints for all of these models, as well as instructions on how to load them, can be found in the :doc:`./results` 
section. You can use the available checkpoints for immediate inference, or fine-tune them on your own datasets. The checkpoints section 
also contains benchmark results for the available ASR models.

.. _Jasper_model:

Jasper
------

Jasper ("Just Another Speech Recognizer") :cite:`asr-models-li2019jasper` is a deep time delay neural network (TDNN) comprising of 
blocks of 1D-convolutional layers. The Jasper family of models are denoted as ``Jasper_[BxR]`` where ``B`` is the number of blocks 
and ``R`` is the number of convolutional sub-blocks within a block. Each sub-block contains a 1-D convolution, batch normalization, 
ReLU, and dropout:

    .. image:: images/jasper_vertical.png
        :align: center
        :alt: jasper model
        :scale: 50%

Jasper models can be instantiated using the :class:`~nemo.collections.asr.models.EncDecCTCModel` class.

QuartzNet
---------

QuartzNet :cite:`asr-models-kriman2019quartznet` is a version of Jasper :cite:`asr-models-li2019jasper` model with separable
convolutions and larger filters. It can achieve performance similar to Jasper but with an order of magnitude fewer parameters. 
Similarly to Jasper, the QuartzNet family of models are denoted as ``QuartzNet_[BxR]`` where ``B`` is the number of blocks and ``R`` 
is the number of convolutional sub-blocks within a block. Each sub-block contains a 1-D *separable* convolution, batch normalization, 
ReLU, and dropout:

    .. image:: images/quartz_vertical.png
        :align: center
        :alt: quartznet model
        :scale: 40%

QuartzNet models can be instantiated using the :class:`~nemo.collections.asr.models.EncDecCTCModel` class.

.. _Citrinet_model:

Citrinet
--------

Citrinet is a version of QuartzNet :cite:`asr-models-kriman2019quartznet` that extends ContextNet :cite:`asr-models-han2020contextnet`,
utilizing subword encoding (via Word Piece tokenization) and Squeeze-and-Excitation mechanism :cite:`asr-models-hu2018squeeze` to
obtain highly accurate audio transcripts while utilizing a non-autoregressive CTC based decoding scheme for efficient inference.

    .. image:: images/citrinet_vertical.png
        :align: center
        :alt: citrinet model
        :scale: 50%

Citrinet models can be instantiated using the :class:`~nemo.collections.asr.models.EncDecCTCModelBPE` class.

.. _ContextNet_model:

ContextNet
----------

ContextNet is a model uses Transducer/RNNT loss/decoder and is introduced in :cite:`asr-models-han2020contextnet`.
It uses Squeeze-and-Excitation mechanism :cite:`asr-models-hu2018squeeze` to model larger context.
Unlike Citrinet, it has an autoregressive decoding scheme.

ContextNet models can be instantiated using the :class:`~nemo.collections.asr.models.EncDecRNNTBPEModel` class for a
model with sub-word encoding and :class:`~nemo.collections.asr.models.EncDecRNNTModel` for char-based encoding.

You may find the example config files of ContextNet model with character-based encoding at
``<NeMo_git_root>/examples/asr/conf/contextnet_rnnt/contextnet_rnnt_char.yaml`` and
with sub-word encoding at ``<NeMo_git_root>/examples/asr/conf/contextnet_rnnt/contextnet_rnnt.yaml``.

.. _Conformer-CTC_model:

Conformer-CTC
-------------

Conformer-CTC is a CTC-based variant of the Conformer model introduced in :cite:`asr-models-gulati2020conformer`. Conformer-CTC has a
similar encoder as the original Conformer but uses CTC loss and decoding instead of RNNT/Transducer loss, which makes it a non-autoregressive model.
We also drop the LSTM decoder and instead use a linear decoder on the top of the encoder. This model uses the combination of 
self-attention and convolution modules to achieve the best of the two approaches, the self-attention layers can learn the global 
interaction while the convolutions efficiently capture the local correlations. The self-attention modules support both regular 
self-attention with absolute positional encoding, and also Transformer-XL's self-attention with relative positional encodings.

Here is the overall architecture of the encoder of Conformer-CTC:

    .. image:: images/conformer_ctc.png
        :align: center
        :alt: Conformer-CTC Model
        :scale: 50%

This model supports both the sub-word level and character level encodings. You can find more details on the config files for the
Conformer-CTC models at `Conformer-CTC <./configs.html#conformer-ctc>`. The variant with sub-word encoding is a BPE-based model
which can be instantiated using the :class:`~nemo.collections.asr.models.EncDecCTCModelBPE` class, while the
character-based variant is based on :class:`~nemo.collections.asr.models.EncDecCTCModel`.

You may find the example config files of Conformer-CTC model with character-based encoding at
``<NeMo_git_root>/examples/asr/conf/conformer/conformer_ctc_char.yaml`` and
with sub-word encoding at ``<NeMo_git_root>/examples/asr/conf/conformer/conformer_ctc_bpe.yaml``.

.. _Conformer-Transducer_model:

Conformer-Transducer
--------------------

Conformer-Transducer is the Conformer model introduced in :cite:`asr-models-gulati2020conformer` and uses RNNT/Transducer loss/decoder.
It has the same encoder as Conformer-CTC but utilizes RNNT/Transducer loss/decoder which makes it an autoregressive model.

Most of the config file for Conformer-Transducer models are similar to Conformer-CTC except the sections related to the decoder and loss: decoder, loss, joint, decoding.
You may take a look at our `tutorials page <../starthere/tutorials.html>` on Transducer models to become familiar with their configs:
`Introduction to Transducers <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/Intro_to_Transducers.ipynb>` and `ASR with Transducers <https://colab.research.google.com/github/NVIDIA/NeMo/blob/stable/tutorials/asr/ASR_with_Transducers.ipynb>`
You can find more details on the config files for the Conformer-Transducer models at `Conformer-CTC <./configs.html#conformer-ctc>`.

This model supports both the sub-word level and character level encodings. The variant with sub-word encoding is a BPE-based model
which can be instantiated using the :class:`~nemo.collections.asr.models.EncDecRNNTBPEModel` class, while the
character-based variant is based on :class:`~nemo.collections.asr.models.EncDecRNNTModel`.

You may find the example config files of Conformer-Transducer model with character-based encoding at
``<NeMo_git_root>/examples/asr/conf/conformer/conformer_transducer_char.yaml`` and
with sub-word encoding at ``<NeMo_git_root>/examples/asr/conf/conformer/conformer_transducer_bpe.yaml``.

LSTM-Transducer
---------------

LSTM-Transducer is a model which uses RNNs (eg. LSTM) in the encoder. The architecture of this model is followed from suggestions in :cite:`asr-models-he2019streaming`.
It uses RNNT/Transducer loss/decoder. The encoder consists of RNN layers (LSTM as default) with lower projection size to increase the efficiency.
Layer norm is added between the layers to stabilize the training.
It can be trained/used in unidirectional or bidirectional mode. The unidirectional mode is fully causal and can be used easily for simple and efficient frame-wise streaming. However the accuracy of this model is generally lower than other models like Conformer and Citrinet.

This model supports both the sub-word level and character level encodings. You may find the example config file of RNNT model with wordpiece encoding at ``<NeMo_git_root>/examples/asr/conf/lstm/lstm_transducer_bpe.yaml``.
You can find more details on the config files for the RNNT models at ``LSTM-Transducer <./configs.html#lstm-transducer>``.

LSTM-CTC
-------

LSTM-CTC model is a CTC-variant of the LSTM-Transducer model which uses CTC loss/decoding instead of Transducer.
You may find the example config file of LSTM-CTC model with wordpiece encoding at ``<NeMo_git_root>/examples/asr/conf/lstm/lstm_ctc_bpe.yaml``.

References
----------

.. bibliography:: asr_all.bib
    :style: plain
    :labelprefix: ASR-MODELS
    :keyprefix: asr-models-
