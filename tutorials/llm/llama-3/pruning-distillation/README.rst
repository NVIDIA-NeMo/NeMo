Llama 3.1 WikiText Pruning and Distillation with NeMo Framework
=======================================================================================

`Llama 3.1 <https://blogs.nvidia.com/blog/meta-llama3-inference-acceleration/>`_ are open-source large language models by Meta that deliver state-of-the-art performance on popular industry benchmarks. They have been pretrained on over 15 trillion tokens, and support a 128K token context length. They are available in three sizes, 8B, 70B, and 405B, and each size has two variants—base pretrained and instruction tuned.

`NVIDIA NeMo Framework <https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html>`_ provides tools to perform teacher finetuning, pruning and distillation on Llama 3.1 to fit your use case.

`LLM Pruning and Distillation in Practice: The Minitron Approach <https://arxiv.org/abs/2408.11796>`_ provides tools to perform teacher finetuning, pruning and distillation on Llama 3.1 as described in the `tech report <https://arxiv.org/abs/2408.11796>`_.

Objectives
----------

This tutorial shows how to perform depth-pruning, teacher finetuning and distillation on **Llama 3.1 8B Instruct** using the `WikiText-103-v1 <https://huggingface.co/datasets/Salesforce/wikitext/viewer/wikitext-103-v1>`_ dataset with NeMo Framework. The `WikiText-103-v1 <https://huggingface.co/datasets/Salesforce/wikitext/viewer/wikitext-103-v1>`_ language modeling dataset is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. For this demonstration, we will perform a light finetuning procedure on the ``Meta Llama 3.1 8B Instruct`` teacher model to generate a finetuned teacher model ``megatron_llama_ft.nemo`` needed for optimal distillation. This finetuned teacher model is then depth-pruned to create a trimmed model ``4b_trimmed_model.nemo``. These models will serve as a starting point for distillation to create a final distilled 4B model.
We are using models utilizing the ``meta-llama/Meta-Llama-3.1-8B`` tokenizer for this demonstration.

Requirements
-------------

* System Configuration
    * Access to at least 8 NVIDIA GPU with an individual memory of at least 80GB, for example: 8 x H100-80GB or 8 x A100-80GB.
    * A Docker-enabled environment, with `NVIDIA Container Runtime <https://developer.nvidia.com/container-runtime>`_ installed, which will make the container GPU-aware.

* `Authenticate with NVIDIA NGC <https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#ngc-authentication>`_, and download `NGC CLI Tool <https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#ngc-cli-tool>`_. You will use this tool to download the model and customize it with NeMo Framework.

* Get your Hugging Face `access token <https://huggingface.co/docs/hub/en/security-tokens>`_, which will be used to obtain the tokenizer required during training.

``NOTE:`` The default configuration in the notebook runs on 8 x 80GB NVIDIA GPUs but you can potentially reduce Tensor Parallel size ``(TENSOR_PARALLEL_SIZE)`` along with the Micro-Batchsize ``(MICRO_BATCH_SIZE)`` in the teacher finetuning and distillation scripts to accommodate lower resource availability.

Create a pruned and distilled model with NeMo Framework
------------------------------------------------------------------------------

For pruning and distilling the model, you will use the NeMo Framework which is available as a `docker container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo>`_.


1. Download the `Llama 3.1 8B Instruct .nemo <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/llama-3_1-8b-instruct-nemo>`_ from NVIDIA NGC using the `NGC CLI <https://org.ngc.nvidia.com/setup/installers/cli>`_. Generate the ``NGC_API_KEY`` following these `instructions <https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#option-2-from-ngc>`_. The following command saves the ``.nemo`` format model in a folder named ``llama-3_1-8b-instruct-nemo_v1.0`` in the current directory. You can specify another path using the ``-d`` option in the CLI tool.

.. code:: bash

   ngc registry model download-version "nvidia/nemo/llama-3_1-8b-instruct-nemo:1.0"

2. Run the container using the following command. It is assumed that you have the dataset, notebook(s), and the ``llama-3.1-8b-instruct`` model available in the current directory. If not, mount the appropriate folder to ``/workspace``.

.. code:: bash

   export FW_VERSION=dev


.. code:: bash

   docker run \
     --gpus all \
     --shm-size=16GB \
     --net=host \
     --ulimit memlock=-1 \
     --rm -it \
     -v ${PWD}:/workspace \
     -w /workspace \
     nvcr.io/nvidia/nemo:$FW_VERSION bash

3. From within the container, start the Jupyter lab:

.. code:: bash

   jupyter lab --ip 0.0.0.0 --port=8888 --allow-root

4. Then, navigate to `this notebook <./llama3-pruning-distillation-nemofw.ipynb>`_.

Results
------------------------------------------------------------------------------
``NOTE:`` This notebook demonstrates the use of the teacher finetuning, pruning and the distillation script. These scripts should ideally be run on a multi-node cluster with a larger ``GLOBAL_BATCH_SIZE`` and ``STEPS`` to see improvement in the validation loss.

Here is the validation loss over 30 steps of running the training step in the distillation script (at the end of the `notebook <./llama3-pruning-distillation-nemofw.ipynb>`_).

.. figure:: https://github.com/NVIDIA/NeMo/releases/download/r2.0.0rc1/val_loss_distillation.png
  :width: 400px
  :alt: Diagram showing the validation loss over 30 steps of running the training step in the distillation script
  :align: center

  Figure 1: Validation Loss Plot