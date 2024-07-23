Checkpoints
===========

In this section, we present four key functionalities of NVIDIA NeMo related to checkpoint management:

1. **Checkpoint Loading**: Use the :code:`restore_from()` method to load local ``.nemo`` checkpoint files.
2. **Partial Checkpoint Conversion**: Convert partially-trained ``.ckpt`` checkpoints to the ``.nemo`` format.
3. **Community Checkpoint Conversion**: Transition checkpoints from community sources, like HuggingFace, into the ``.nemo`` format.
4. **Model Parallelism Adjustment**: Modify model parallelism to efficiently train models that exceed the memory of a single GPU. NeMo employs both tensor (intra-layer) and pipeline (inter-layer) model parallelisms. Dive deeper with `"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" <https://arxiv.org/pdf/2104.04473.pdf>`_. This tool aids in adjusting model parallelism, accommodating users who need to deploy on larger GPU arrays due to memory constraints.

Understanding Checkpoint Formats
--------------------------------

A ``.nemo`` checkpoint is fundamentally a tar file that bundles the model configurations (given as a YAML file), model weights, and other pertinent artifacts like tokenizer models or vocabulary files. This consolidated design streamlines sharing, loading, tuning, evaluating, and inference.

Contrarily, the ``.ckpt`` file, created during PyTorch Lightning training, encompasses both the model weights and the optimizer states, usually employed to pick up training from a pause.

The subsequent sections elucidate instructions for the functionalities above, specifically tailored for deploying fully trained checkpoints for assessment or additional fine-tuning.

Loading Local Checkpoints
-------------------------

By default, NeMo saves checkpoints of trained models in the ``.nemo`` format. To save a model manually during training, use:

.. code-block:: python

   model.save_to(<checkpoint_path>.nemo)

To load a local ``.nemo`` checkpoint:

.. code-block:: python

   import nemo.collections.multimodal as nemo_multimodal
   model = nemo_multimodal.models.<MODEL_BASE_CLASS>.restore_from(restore_path="<path/to/checkpoint/file.nemo>")

Replace `<MODEL_BASE_CLASS>` with the appropriate MM model class.

Converting Community Checkpoints
--------------------------------

CLIP Checkpoints
^^^^^^^^^^^^^^^^


To migrate community checkpoints, use the following command:

.. code-block:: bash

    torchrun --nproc-per-node=1 /opt/NeMo/scripts/checkpoint_converters/convert_clip_hf_to_nemo.py \
        --input_name_or_path=openai/clip-vit-large-patch14 \
        --output_path=openai_clip.nemo \
        --hparams_file=/opt/NeMo/examples/multimodal/vision_language_foundation/clip/conf/megatron_clip_VIT-L-14.yaml

Ensure the NeMo hparams file has the correct model architectural parameters, placed at `path/to/saved.yaml`. An example can be found in `examples/multimodal/foundation/clip/conf/megatron_clip_config.yaml`.

After conversion, you can verify the model with the following command:

.. code-block:: bash

    wget https://upload.wikimedia.org/wikipedia/commons/0/0f/1665_Girl_with_a_Pearl_Earring.jpg
    torchrun --nproc-per-node=1 /opt/NeMo/examples/multimodal/vision_language_foundation/clip/megatron_clip_infer.py \
        model.restore_from_path=./openai_clip.nemo \
        image_path=./1665_Girl_with_a_Pearl_Earring.jpg \
        texts='["a dog", "a boy", "a girl"]'

It should generate a high probability for the "a girl" tag. For example:

.. code-block:: text

    Given image's CLIP text probability:  [('a dog', 0.0049710185), ('a boy', 0.002258187), ('a girl', 0.99277073)]
