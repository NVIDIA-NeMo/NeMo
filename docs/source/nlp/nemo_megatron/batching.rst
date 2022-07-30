.. _batching:

Batching
--------

Batch size is one of the first parameters you should play with. For efficiency and convergence reasons we recommend you first try maximizing your batch size per GPU so that your GPU RAM usage is maximized.

NeMo Megatron uses the following concepts.

*Micro batch size* is the number of examples per data parallel rank. It is controlled by ``model.micro_batch_size`` parameter.

*Global batch size* = micro_batch_size * data_parallel_size * gradient_accumulation_steps. For details on ``data_parallel_size`` see :ref:`parallelisms` section, but typically it is equal to the number of GPUs being used.
Global batch size is controlled by ``model.global_batch_size`` parameter. 


*Gradient Accumulation*

    * Idea: Train with large batch sizes with fixed memory footprint at the cost of additional compute.
    * Do k forward and backward passes through the network with different batches, do not perform parameter updates until after k passes.
    * Update paramters

