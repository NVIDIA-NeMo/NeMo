NeMo Speaker Diarization API
=============================


Model Classes
-------------
.. autoclass:: nemo.collections.asr.models.ClusteringDiarizer
    :show-inheritance:
    :members:  

.. autoclass:: nemo.collections.asr.models.EncDecDiarLabelModel
    :show-inheritance:
    :members: add_speaker_model_config, _init_segmentation_info, _init_speaker_model, setup_training_data, setup_validation_data, setup_test_data, get_ms_emb_seq, get_cluster_avg_embs_model, get_ms_mel_feat, forward, forward_infer, training_step, validation_step, compute_accuracies

Mixins
------
.. autoclass:: nemo.collections.asr.parts.mixins.mixins.DiarizationMixin
    :show-inheritance:
    :members:

