
Core APIs
=========

Base class for all NeMo models
------------------------------

.. autoclass:: nemo.core.ModelPT
    :show-inheritance:
    :members:
    :member-order: bysource
    :undoc-members: cfg, num_weights
    :exclude-members: set_eff_save, use_eff_save, teardown

Base Neural Module class
------------------------

.. autoclass:: nemo.core.NeuralModule
    :show-inheritance:
    :members:
    :member-order: bysource

Neural Type classes
-------------------

.. autoclass:: nemo.core.neural_types.NeuralType
    :show-inheritance:
    :members:
    :member-order: bysource

.. autoclass:: nemo.core.neural_types.axes.AxisType
    :show-inheritance:
    :members:
    :member-order: bysource

.. autoclass:: nemo.core.neural_types.elements.ElementType
    :show-inheritance:
    :members:
    :member-order: bysource

.. autoclass:: nemo.core.neural_types.comparison.NeuralTypeComparisonResult
    :show-inheritance:
    :members:
    :member-order: bysource

Experiment manager
------------------

.. autoclass:: nemo.utils.exp_manager.exp_manager
    :show-inheritance:
    :members:
    :member-order: bysource
