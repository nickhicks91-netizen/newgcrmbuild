API Reference
=============

This page contains the complete API documentation for all GRCM modules.

Core Module
-----------

.. automodule:: grcm.core
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.core.ModularGRCM
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __call__

Configuration
-------------

.. automodule:: grcm.config
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.config.GRCMConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: grcm.config.AttentionConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.config.PhiConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.config.QualiaConfig
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.config.EthicalConfig
   :members:
   :undoc-members:
   :show-inheritance:

Modules
-------

Grounding Module
~~~~~~~~~~~~~~~~

.. automodule:: grcm.modules.grounding
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.modules.grounding.GroundingModule
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Embedding Module
~~~~~~~~~~~~~~~~

.. automodule:: grcm.modules.embedding
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.modules.embedding.EmbeddingModule
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Attention Module
~~~~~~~~~~~~~~~~

.. automodule:: grcm.modules.attention
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.modules.attention.AttentionModule
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Desire Module
~~~~~~~~~~~~~

.. automodule:: grcm.modules.desire
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.modules.desire.DesireModule
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Memory Module
~~~~~~~~~~~~~

.. automodule:: grcm.modules.memory
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.modules.memory.MemoryModule
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Reflection Module
~~~~~~~~~~~~~~~~~

.. automodule:: grcm.modules.reflection
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.modules.reflection.ReflectionModule
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Qualia Module
~~~~~~~~~~~~~

.. automodule:: grcm.modules.qualia
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.modules.qualia.QualiaModule
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Threading Module
~~~~~~~~~~~~~~~~

.. automodule:: grcm.modules.threading
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.modules.threading.ThreadingModule
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Phi Module
~~~~~~~~~~

.. automodule:: grcm.modules.phi
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.modules.phi.PhiModule
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Body Module
~~~~~~~~~~~

.. automodule:: grcm.modules.body
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.modules.body.BodyModule
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Optimization
------------

.. automodule:: grcm.optimization
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.optimization.GRCMOptimizer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Benchmarking
------------

.. automodule:: grcm.benchmark
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.benchmark.GRCMBenchmark
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

MLflow Logging
--------------

.. automodule:: grcm.logging
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.logging.MLflowLogger
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

User Interface
--------------

.. automodule:: grcm.ui
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: grcm.ui.create_gradio_ui

Training
--------

.. automodule:: grcm.trainer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: grcm.trainer.EchoMirrorTrainer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Utilities
---------

Data Structures
~~~~~~~~~~~~~~~

.. autoclass:: grcm.core.GRCMState
   :members:
   :undoc-members:

Helper Functions
~~~~~~~~~~~~~~~~

.. autofunction:: grcm.modules.attention.compute_coherence
.. autofunction:: grcm.modules.qualia.compute_qualia_distribution
.. autofunction:: grcm.modules.phi.compute_integrated_information
