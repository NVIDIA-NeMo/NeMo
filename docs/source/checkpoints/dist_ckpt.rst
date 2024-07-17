Distributed Checkpoints
=======================

This guide provides details about the distributed checkpoints format from Megatron Core.


Introduction
------------

Model parallel training requires parallelism-aware checkpointing.
Megatron Core provides a checkpointing library capable of handling all types of parallelisms used in LLM training.
Although the distributed checkpointing library is targeted at the Megatron Core model, it can also be used with other models, as long as proper integration is implemented.

The library provides two main entrypoints: ``dist_checkpointing.save`` and ``dist_checkpointing.load`` which are meant to replace the ``torch.save`` and ``torch.load`` in the regular checkpointing flow.
Apart from that, it provides a mechanism to define how different types of local tensors should be combined and split in the global checkpoint.


Basic Sharding
--------------

The main way to define the relationship of a plain, local PyTorch tensor to tensors on other ranks is by wrapping it in a ``ShardedTensor`` class.
This allows to express the fact that a given local tensor is part of a larger *grid* of tensors of a given shape at a given offset.
Instead of saving a simple state dict with ``torch.Tensor``, we save a *sharded* state dict with ``dist_checkpointing.ShardedTensor``.

Example: assume we have a tensor (composed of 128 elements) divided equally across the whole workload which we want to save and load with different number of ranks.

.. code-block:: python

    from pathlib import Path

    import torch

    from megatron.core import dist_checkpointing

    # Setup
    ckpt_root = Path('/tmp/checkpoints')
    native_ckpt_root = ckpt_root / 'native'
    native_ckpt_root.mkdir(exist_ok=True, parents=True)
    dist_ckpt_root = ckpt_root / 'dist_ckpt'
    dist_ckpt_root.mkdir(exist_ok=True, parents=True)

    torch.distributed.init_process_group()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # Local tensor to save
    assert 128 % world_size == 0
    num_elems_per_rank = 128 // world_size
    local_ten = torch.arange(start=num_elems_per_rank * rank,
                             end=num_elems_per_rank * (rank + 1))

    # Native checkpoint save
    state_dict = {
        'weight': local_ten
    }
    torch.save(state_dict, native_ckpt_root / f'ckpt_{rank}.pt')

    # Distributed checkpoint save
    # `(0, rank, world_size)` describes that `weight` ShardedTensor is sharded into `world_size` pieces
    # along the 0th dimension and `local_ten` is the shard at position `rank`.
    # Together, all shards implicitly form a "global" `torch.arange(128)` tensor.
    sharded_state_dict = {
        'weight': dist_checkpointing.ShardedTensor.from_rank_offsets('weight', local_ten, (0, rank, world_size))
    }
    dist_checkpointing.save(sharded_state_dict, dist_ckpt_root)

During load, the distributed checkpoint can be easily read even if the job size changes (contrary to native checkpoints that require the same number of ranks).
The main difference with wrt. ``torch.load`` is that the user has to provide the definition of the sharded state dict that needs to be loaded.

.. code-block:: python

    from pathlib import Path

    import torch

    from megatron.core import dist_checkpointing

    ckpt_root = Path('/tmp/checkpoints')
    dist_ckpt_root = ckpt_root / 'dist_ckpt'

    torch.distributed.init_process_group()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    assert 128 % world_size == 0
    num_elems_per_rank = 128 // world_size

    # Local tensor to load
    local_ten = torch.empty(num_elems_per_rank)
    sharded_state_dict = {
        'weight': dist_checkpointing.ShardedTensor.from_rank_offsets('weight', local_ten, (0, rank, world_size))
    }
    loaded_state_dict = dist_checkpointing.load(sharded_state_dict, dist_ckpt_root)
    expected_local_ten = torch.arange(start=num_elems_per_rank * rank, end=num_elems_per_rank * (rank + 1))
    assert torch.all(loaded_state_dict['weight'] == expected_local_ten)

    # With torch.save and torch.load, we would have to load all files that contain
    # parts of the desired tensor in new configuration and concatenate appropriate fragments.
    # For some distributed checkpoint backends this is actually what happens underneath.


Supported Entities
------------------
The distributed checkpointing library supports saving and loading of different objects in different configurations.

A sharded state dict is a (possibly nested) Python dictionary or list with the following elements:

1. ShardedBase
    a. ShardedTensor
    b. ShardedObject
    c. ShardedTensorFactory
2. LocalNonpersitentObject
3. Arbitrary object


ShardedBase
^^^^^^^^^^^
ShardedBase is the base class for expressing any kind of sharding.
Each sharded entity must be uniquely identified by its ``key``, carry some ``data`` to be saved or loaded, and define ``replica_id`` which helps identify data redundancy.

Note that the ``key`` doesn't have to (and usually doesn't) correspond to the key in the state dict.
The key in the state dict is ephemeral, while the ``ShardedTensor.key`` is used to identify the tensor in the checkpoint.

In the following example, the state dict to be loaded contains different keys than the saved one.
What matters is that the ``ShardedTensor.key`` are equivalent (``tensor-A``):

.. code-block:: python

    import torch

    from megatron.core import dist_checkpointing

    # Checkpoint saved with some key in the state dict that is eventually ignored
    model = ...
    ckpt_dir = ...
    sharded_state_dict = {
        'ignored': dist_checkpointing.ShardedTensor('tensor-A', ...)
    }
    dist_checkpointing.save(sharded_state_dict, ckpt_dir)

    # During loading, all that matters is the ShardedTensor.key.
    sharded_state_dict = {
        'different-key': dist_checkpointing.ShardedTensor('tensor-A', ...)
    }
    loaded_state_dict = dist_checkpointing.load(sharded_state_dict, ckpt_dir)
    assert 'ignored' not in loaded_state_dict
    assert 'tensor-A' not in loaded_state_dict
    assert isinstance(loaded_state_dict['different-key'], torch.Tensor)

    # The key in the state dict is important only from the subsequent `model.load_state_dict`
    # that usually happens after `dist_checkpointing.load` - the state dict must have
    # the structure and keys corresponding to the model structure and submodule names.
    model.load_state_dict(loaded_state_dict)

ShardedTensor
^^^^^^^^^^^^^
``ShardedTensor`` is the primary use case for distributed checkpointing - tensor sharding.
It defines how PyTorch tensors are distributed across the workload.
See the `Tensors transformations`_ section for more details on ShardedTensors.

ShardedObject
^^^^^^^^^^^^^
Sometimes there is a need to save arbitrary objects across the ranks.
ShardedObject allows to structure those objects into arrays of objects with a fixed ``global_shape`` and save/load parts of the arrays on specific ranks.

ShardedTensorFactory
^^^^^^^^^^^^^^^^^^^^
The ShardedTensorFactory class defers tensors transformations until they are actually saved.
A factory can expand a tensor into an arbitrary sub state dict (including all supported entities listed above).
The need for such deferral will be explained in the `Tensors transformations`_ section.

LocalNonpersistentObject
^^^^^^^^^^^^^^^^^^^^^^^^
LocalNonpersistentObject is a simple wrapper indicating that the object wrapped with this class should end up in the final loaded state dict during loading.
During saving such objects are ignored.

Arbitrary Object
^^^^^^^^^^^^^^^^
All objects different than dicts, lists, and the instances of the classes listed above are treated as "common" objects.

During saving, all such objects in the sharded state dict passed to ``dist_checkpointing.save`` are assumed to be duplicated across ranks. Therefore, they are saved only by a single coordinator rank (rank 0).

During loading, all such objects in the sharded state dict passed to ``dist_checkpointing.load`` are simply ignored - the loaded state dict contains only "common" objects that are were actually saved in the checkpoint.




Entry Points
------------
There are several useful user entry points for checkpoint saving and loading.

dist_checkpointing.save
^^^^^^^^^^^^^^^^^^^^^^^
The ``dist_checkpointing.save`` function is the only entry point for checkpoint saving.
It requires providing a sharded state dict to save and saving strategies for handling different entities (see `Save and load strategies`_ for detailed explanation).
The sharded state dict is processed in the following way (see also ``save`` function `documentation <https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/dist_checkpointing.html#module-core.dist_checkpointing.serialization>`_):

1. The ShardedTensorFactories are applied.
2. The LocalNonPersistentObjects are extracted from the sharded state dict and ignored.
3. The ShardedBase objects are extracted.
4. All other objects are treated as "common" and saved according to a sharded strategy (see `Save and load strategies`_).
5. All ShardedObjects are extracted from point (3) objects and saved with a common strategy (see `Save and load strategies`_).
6. All ShardedTensors are saved.
7. The ``metadata.json`` file with backend and version metadata is saved to the checkpoint directory.

dist_checkpointing.load
^^^^^^^^^^^^^^^^^^^^^^^
The ``dist_checkpointing.load`` function is the main entry point for checkpoint loading.
It requires providing a sharded state dict (in order to implicitly define mappings between local tensors and checkpoint tensors) and loading strategies.
In practice, the same sharded state dict can be usually used for both saving and loading (the sharded state dict for loading will just contain tensors with uninitialized data).

When the sharded state dict is provided as input, it is processed in the following way (see also ``load`` function `documentation <https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/dist_checkpointing.html#module-core.dist_checkpointing.serialization>`_):

1. The "common" state dict is loaded from the checkpoint. This forms the base of the resulting state dict.
2. The ShardedTensorFactories from the input sharded state dict are applied.
3. The LocalNonPersistentObjects are extracted from the input sharded state dict, unwrapped and added to the resulting state dict.
4. The ShardedObjects are extracted and loaded from the checkpoint into the resulting state dict.
5. The ShardedTensors are extracted and loaded from the checkpoint into the resulting state dict.
6. Factory merges are applied (see `Optimizers`_ for explanation).

This results in a *regular* state dict with plain tensors that can be further processed by the application (which usually means running ``model.load_state_dict(state_dict)``).


dist_checkpointing.load_common_state_dict
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``dist_checkpointing.load_common_state_dict`` function is an entry point that allows loading only the “common” part of the checkpoints.
Most of the checkpoint config and metadata can be loaded with this method, which allows skipping data loading in order to take decisions regarding checkpoint config, version, etc.

dist_checkpointing.load_tensors_metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``dist_checkpointing.load_tensors_metadata`` function is an entry point that allows reading all ShardedTensors metadata from the checkpoint without loading any data.
The result is a sharded state dict with trivial sharding (every tensor is sharded into one big shard).

dist_checkpointing.load_plain_tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``dist_checkpointing.load_plain_tensors`` function is an entry point that allows reading sharded tensors stored in the checkpoint without any sharding (as plain tensors).
This function is simply a composition of ``load_tensors_metadata`` and ``save``.

Save and Load Strategies
------------------------
There are multiple ways to save a sharded state dict into a serialized checkpoint. They can be provided by the user as saving and loading strategies (e.g. ``TorchDistLoadShardedStrategy`` and ``TorchDistSaveShardedStrategy`` as shown below).

There are four types of strategies:

1. Saving strategy for ShardedTensors
2. Saving strategy for "common" data
3. Loading strategy for ShardedTensors
4. Loading strategy for "common" data

Additionally, ShardedObjects are handled with either "sharded" or "common" strategy depending on its capabilities (``can_handle_sharded_objects`` property).

Each saving strategy is associated with a ``backend`` and a ``version``.
Each loading strategy can be associated with multiple values of ``backend`` and ``version`` it can load.
For a given backend and version, the composition of every saving and loading strategy **must be functionally equivalent**.
Strategies are the main way to introduce optimizations to the saving and loading algorithm without altering the checkpoint format.

In the following example, the "fully parallel" wrappers modify the saving and loading *algorithm*, but the underlying checkpoint *format* (and ``backend`` in consequence) stays the same.
It makes the ``basic_save_load`` and ``fully_parallel_save_load`` functions equivalent:

.. code-block:: python

    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.strategies.torch import (
        TorchDistLoadShardedStrategy,
        TorchDistSaveShardedStrategy
    )
    from megatron.core.dist_checkpointing.strategies.fully_parallel import (
        FullyParallelLoadStrategyWrapper,
        FullyParallelSaveStrategyWrapper
    )

    # Base save and load strategies defining a regular (non-parallel) save
    base_save_strategy = TorchDistSaveShardedStrategy('torch_dist', 1)
    base_load_strategy = TorchDistLoadShardedStrategy()

    def basic_save_load(sharded_state_dict, ckpt_dir):
        """ Save and load using some basic strategies. """
        dist_checkpointing.save(sharded_state_dict, ckpt_dir, base_save_strategy)
        return dist_checkpointing.load(sharded_state_dict, ckpt_dir, base_load_strategy)


    def fully_parallel_save_load(sharded_state_dict):
        """ Save and load using basic strategies wrapped with parallelization strategies. """
        fully_parallel_save_strategy = FullyParallelSaveStrategyWrapper(base_save_strategy)
        # "fully parallel" wrapper modifies the saving strategy, but not the underlying format
        assert fully_parallel_save_strategy.backend == base_save_strategy.backend == 'torch_dist'
        fully_parallel_load_strategy = FullyParallelLoadStrategyWrapper(base_load_strategy)
        dist_checkpointing.save(sharded_state_dict, ckpt_dir, fully_parallel_save_strategy)
        return dist_checkpointing.load(sharded_state_dict, ckpt_dir, fully_parallel_load_strategy)


The ``dist_checkpointing`` package provides default strategies for some sharded backends, so it's enough to specify a tuple ``(backend, version)`` as a saving strategy.
Backends and versions are stored in a ``metadata.json`` file inside the checkpoint so that the loading strategy can be determined automatically (provided that there exists a default loading strategy for a given backend and version).

For "sharded" strategies, currently the backends supported by default are based on `PyTorch Distributed`_ format (``torch_dist`` backend) and `Zarr`_ format (``zarr`` backend).
Additionally, as shown in the example above, some wrappers are provided that enable it to parallelize the save and load across the whole workload (assuming some data duplication).

For "common" strategies, currently the only supported one is ``torch`` which saves "common" data into a ``common.pt`` file.

PyTorch Distributed
^^^^^^^^^^^^^^^^^^^
The PyTorch Distributed based checkpoint format uses the ``torch.distributed.checkpoint`` package in order to serialize the checkpoints to storage.
The Megatron Core sharded state dicts are translated into ``torch.distributed.SharedTensor`` and then ``torch.distributed.checkpoint`` primitives are used to serialize such state dicts.
Even though Megatron Core provides several saving optimizations, the underlying checkpoint can still be read with native `PyTorch loading methods <https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.state_dict_loader.load>`_.
Note that the checkpoint still follows the ``dist_checkpointing`` package format by providing additional ``common.pt`` and ``metadata.json`` files described above.

PyTorch Distributed is a recommended checkpoint format.

Zarr
^^^^
The Zarr based checkpoint format uses the `Zarr <https://zarr.readthedocs.io/en/stable/>`__ library in order to serialize the checkpoints to storage.
This format is deprecated and it's recommended to transition to the ``torch_dist`` format (using this `converter script <https://github.com/NVIDIA/NeMo/blob/main/scripts/checkpoint_converters/convert_zarr_to_torch_dist.py>`_).

Optimizers
----------
The Optimizers module provides helper tools to the user to simplify constructing ShardedTensors for optimizer states.
The ShardedTensors that define local-to-sharded tensors mapping for model parameters should be reused for optimizer states to avoid code duplication.

To this end, the ``dist_checkpointing.optimizers.get_param_id_to_sharded_param_map`` function can build a mapping between optimizer params ids and model ShardedTensors.
This mapping can be used by the ``dist_checkpointing.optimizers.optim_state_to_sharding_state`` function or application code (for non-standard use cases) to construct optimizer sharded state dict with ShardedTensors.
This should support most optimizer cases, but some of them might require custom sharded state dict creation.
A good example is a Distributed Optimizer which flattens the parameters - see `Tensors transformations`_ section for more details.

Note: In order to reuse model SharderTensors to create optimizer ShardedTensors, the model **SharderTensors must wrap model parameters**, not just tensors
(obtaining a state dict with model parameters can be achieved by passing ``keep_vars=True`` to the model ``state_dict`` function).
Otherwise the correspondence between model ShardedTensors and optimizer states is impossible to recreate.
This is the reason for introducing ShardedTensorFactories - we have to register the original model parameter as ``ShardedTensorFactories.data`` and apply any subsequent transformations as a factory function in order to make sure that the same transformation can be applied to the optimizer states.
Even if the model parameters transformations are complex, in most cases the optimizer state dict is easy to recreate based on the model ShardedTensors and ShardedTensorFactories,
e.g. `FP32Optimizer.sharded_state_dict <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/optimizer.py#L793>`_ is just a matter of two generic ``get_param_id_to_sharded_param_map`` and ``optim_state_to_sharding_state`` function calls regardless of the model parameters complexity.


Tensors Transformations
-----------------------
The ShardedTensor API enables the declaration of basic transformations that should be performed during saving and loading.

Shape Mismatch
^^^^^^^^^^^^^^
The ``allow_shape_mismatch`` flag relaxes the requirement of matching global tensor shapes during loading.
Extra padding is filled with zeros or stripped depending on the mismatch kind.
This is useful for layers like embedding which might be padded according to parallelism for performance reasons.

Flattening
^^^^^^^^^^
The ``flattened_range`` attribute declares that ``ShardedTensor.data`` represents a slice of a flattened model parameter.
This corresponds to a transformation used in Distributed Optimizers which flattens the data and shards it along the data-parallel domain.

Extra flattening comes with an efficiency challenge during checkpoint resharding.
Since flattening is applied after the global tensors is sharded into the grid of local chunks, loading after resharding requires accessing incontiguous data fragments.
An example solution for that is implemented in the `resharding <https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/dist_checkpointing/strategies/resharding.py>`_ module and involves saving the flattened tensor with a different global shape than the original one.

Example: For a global tensor ``[[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]]`` with sharding by TP (tensor-parallel) over the second axis, here are the local shards if TP=2:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Rank
     - Local shards
   * - 0
     - ``[[0, 1, 2], [6, 7, 8]]``
   * - 1
     - ``[[3, 4, 5], [9, 10, 11]]``

After flattening and sharding by DP=3 (which would happen in the Megatron Core Distributed Optimizer), the resulting local shards are as follows:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Rank
     - Local shards
   * - 0
     - ``[0, 1]``
   * - 2
     - ``[2, 6]``
   * - 4
     - ``[7, 8]``
   * - 1
     - ``[3, 4]``
   * - 3
     - ``[5, 9]``
   * - 5
     - ``[10, 11]``

After sharding by TP=6 and flattening and sharding by DP=1, the resulting local shards are as follows:


.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Rank
     - Local shards
   * - 0
     - ``[0, 6]``
   * - 1
     - ``[1, 7]``
   * - 2
     - ``[2, 8]``
   * - 3
     - ``[3, 9]``
   * - 4
     - ``[4, 10]``
   * - 5
     - ``[5, 11]``


Arbitrary Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^
The way to apply arbitrary transformations to the tensors during saving and loading is with ShardedTensorFactory.
It defines such a transformation as a function that can be reapplied to any ShardedTensor (in particular, a ShardedTensor representing optimizer states).
Such "build" function is also tied to a "merge" function that can apply an inverse transformation during loading.

If handling an optimizer state is not required, such a transformation could be also applied directly during sharded state dict creation.
In order to apply such transformation both to model and optimizer parameters in a consistent manner, it's necessary to encode them as factory functions (with original model parameter as the ``data`` input so that the optimizer params can be properly mapped to model ShardedTensors).

Note that implementing some transformations might be challenging or impossible while supporting flattening for a Distributed Optimizer case.
For example, if the model weights are supposed to be transposed in the checkpoint, it's almost impossible to implement a performant factory function that is capable of transposing a flattened and sliced tensor. This is because the flattening and slicing should happen in the transposed dimension.

Application Integration
-----------------------
The ``dist_checkpointing`` package provides all general mechanisms for saving arbitrary distributed checkpoints.
The only thing required from the application side is preparing a sharded state dict with ShardedTensors, ShardedObjects, etc. (representing the sharding of the data employed by the application)
and using the ``dist_checkpointing.save`` and ``dist_checkpointing.load`` entrypoints as replacements for ``torch.save`` and ``torch.load``.

In Megatron Core, the sharded state dictionary preparation is already implemented in a ``sharded_state_dict`` method which creates the sharded state dicts in a composable way.
For other applications (e.g. with simpler types of supported parallelisms) it might be possible to apply a straightforward conversion from a regular model state dict into a sharded state dict.

