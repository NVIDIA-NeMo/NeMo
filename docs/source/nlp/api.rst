NeMo NLP collection API
=======================

Model Classes
-------------

.. autoclass:: nemo.collections.nlp.models.TextClassificationModel
    :show-inheritance:
    :members: setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact, classifytext

.. autoclass:: nemo.collections.nlp.models.GLUEModel
    :show-inheritance:
    :members: setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact

.. autoclass:: nemo.collections.nlp.models.PunctuationCapitalizationModel
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.nlp.models.TokenClassificationModel
    :show-inheritance:
    :members: setup_training_data, setup_optimization, setup_validation_data, setup_test_data, register_artifact
    
.. autoclass:: nemo.collections.nlp.models.QAModel
    :show-inheritance:
    :members: setup_training_data, setup_optimization, setup_validation_data, setup_test_data, inference, validation_epoch_end, test_epoch_end

.. autoclass:: nemo.collections.nlp.models.DuplexTaggerModel
    :show-inheritance:
    :members: setup_training_data, setup_optimization, setup_validation_data, setup_test_data, inference, validation_epoch_end, test_epoch_end

.. autoclass:: nemo.collections.nlp.models.DuplexDecoderModel
    :show-inheritance:
    :members: setup_training_data, setup_optimization, setup_validation_data, setup_test_data, inference, validation_epoch_end, test_epoch_end

.. autoclass:: nemo.collections.nlp.models.BERTLMModel
    :show-inheritance:
    :members: setup_training_data, setup_optimization

Modules
-------

.. autoclass:: nemo.collections.nlp.modules.BertModule
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.nlp.modules.AlbertEncoder
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.nlp.modules.BertEncoder
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.nlp.modules.DistilBertEncoder
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.nlp.modules.RobertaEncoder
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.nlp.modules.SequenceClassifier
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.nlp.modules.SequenceRegression
    :show-inheritance:
    :members:

.. autoclass:: nemo.collections.nlp.modules.SequenceTokenClassifier
    :show-inheritance:
    :members:

.. autofunction::  nemo.collections.nlp.modules.get_lm_model

.. autofunction::  nemo.collections.nlp.modules.get_pretrained_lm_models_list

.. autofunction::  nemo.collections.nlp.modules.common.megatron.get_megatron_lm_models_list

Datasets
--------

.. autoclass:: nemo.collections.nlp.data.token_classification.punctuation_capitalization_dataset.BertPunctuationCapitalizationDataset
    :show-inheritance:
    :members:
    :special-members: __getitem__

.. autofunction:: nemo.collections.nlp.data.token_classification.punctuation_capitalization_tarred_dataset.create_tarred_dataset

.. autoclass:: nemo.collections.nlp.data.token_classification.punctuation_capitalization_tarred_dataset.BertPunctuationCapitalizationTarredDataset
    :show-inheritance:
    :members:
    :special-members: __iter__
    :exclude-members: reinforce_type

.. autoclass:: nemo.collections.nlp.data.token_classification.punctuation_capitalization_infer_dataset.BertPunctuationCapitalizationInferDataset
    :show-inheritance:
    :members:
    :special-members: __getitem__
