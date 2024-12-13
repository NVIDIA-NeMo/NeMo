.. _batching:

Batching
--------

Batch size is one of the first parameters you should adjust. For efficiency and convergence, we recommend first maximizing your batch size per GPU to fully utilize your GPU RAM.

NeMo Framework uses the following parameters:

===========================  ==================================================================================================================================
Parameter                    Description
===========================  ==================================================================================================================================
Micro Batch Size             The number of examples per data parallel rank.
Global Batch Size            The global batch size is calculated as: ``global batch size = micro_batch_size * data_parallel_size * gradient_accumulation_steps``. For details on ``data_parallel_size`` see `this page <https://github.com/NVIDIA/NeMo/blob/main/docs/source/features/parallelisms.rst>`_.
Gradient Accumulation        This parameter supports training with large batch sizes while maintaining a fixed memory footprint, though it requires additional compute. The ``accumulate_grad_batches`` is automatically managed by PyTorch Lightning.
===========================  ==================================================================================================================================

Set the Batching Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example shows how to set up a pretraining recipe and batching parameters for a LLaMA-3 8B model:

       .. code-block:: python

              from nemo.collections import llm
              from functools import partial

              # Load train recipe
              recipe = partial(llm.llama3_8b.pretrain_recipe)()
              
              # Set micro and global batch size
              recipe.data.micro_batch_size = 4
              recipe.data.global_batch_size = 16
              
              # Set accumulate_grad_batches
              recipe.trainer.accumulate_grad_batches = 1

Set batching parameters directly from CLI:

       .. code-block:: bash

              nemo llm pretrain --factory llama3_8b data.micro_batch_size=4 data.global_batch_size=16 trainer.accumulate_grad_batches=1

