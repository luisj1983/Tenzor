Optimization Module (tenzor.optim)
===================================

The ``tenzor.optim`` module contains optimization algorithms and learning rate schedulers
for training neural networks.

.. currentmodule:: tenzor.optim

Optimizer Base Class
--------------------

.. autoclass:: Optimizer
   :members:
   :undoc-members:
   :show-inheritance:

Optimizers
----------

Stochastic Gradient Descent
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: SGD
   :members:
   :undoc-members:
   :show-inheritance:

Adam Variants
^^^^^^^^^^^^^

.. autoclass:: Adam
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AdamW
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AdamaxOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: NAdam
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: RAdam
   :members:
   :undoc-members:
   :show-inheritance:

Adaptive Learning Rate Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RMSprop
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Adagrad
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Adadelta
   :members:
   :undoc-members:
   :show-inheritance:

Quasi-Newton Methods
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: LBFGS
   :members:
   :undoc-members:
   :show-inheritance:

Learning Rate Schedulers
-------------------------

Base Scheduler
^^^^^^^^^^^^^^

.. autoclass:: LRScheduler
   :members:
   :undoc-members:
   :show-inheritance:

Step-based Schedulers
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: StepLR
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MultiStepLR
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ExponentialLR
   :members:
   :undoc-members:
   :show-inheritance:

Cosine Annealing Schedulers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: CosineAnnealingLR
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: CosineAnnealingWarmRestarts
   :members:
   :undoc-members:
   :show-inheritance:

Adaptive Schedulers
^^^^^^^^^^^^^^^^^^^

.. autoclass:: ReduceLROnPlateau
   :members:
   :undoc-members:
   :show-inheritance:

Cyclic Schedulers
^^^^^^^^^^^^^^^^^

.. autoclass:: CyclicLR
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: OneCycleLR
   :members:
   :undoc-members:
   :show-inheritance:

Lambda Schedulers
^^^^^^^^^^^^^^^^^

.. autoclass:: LambdaLR
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: MultiplicativeLR
   :members:
   :undoc-members:
   :show-inheritance:

Gradient Clipping
-----------------

.. autofunction:: clip_grad_norm_
.. autofunction:: clip_grad_value_

Usage Examples
--------------

Basic Optimizer Usage
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import tenzor as tz

   # Create model and optimizer
   model = tz.nn.Linear(10, 5)
   optimizer = tz.optim.Adam(model.parameters(), lr=0.001)

   # Training loop
   for epoch in range(100):
       # Forward pass
       output = model(input)
       loss = criterion(output, target)

       # Backward pass
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

Using Learning Rate Schedulers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import tenzor as tz

   # Create optimizer and scheduler
   optimizer = tz.optim.Adam(model.parameters(), lr=0.001)
   scheduler = tz.optim.StepLR(optimizer, step_size=30, gamma=0.1)

   # Training loop
   for epoch in range(100):
       train(...)
       validate(...)
       scheduler.step()

Gradient Clipping
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import tenzor as tz

   # Clip gradients by norm
   tz.optim.clip_grad_norm_(model.parameters(), max_norm=1.0)

   # Clip gradients by value
   tz.optim.clip_grad_value_(model.parameters(), clip_value=0.5)

Optimizer Comparison
--------------------

+----------------+-------------------+------------------------+-------------------+
| Optimizer      | Adaptive LR       | Memory Overhead        | Best For          |
+================+===================+========================+===================+
| SGD            | No                | Low                    | General purpose   |
+----------------+-------------------+------------------------+-------------------+
| Adam           | Yes               | Medium (2x params)     | General purpose   |
+----------------+-------------------+------------------------+-------------------+
| AdamW          | Yes               | Medium (2x params)     | Transformers      |
+----------------+-------------------+------------------------+-------------------+
| RMSprop        | Yes               | Medium (1x params)     | RNNs              |
+----------------+-------------------+------------------------+-------------------+
| Adagrad        | Yes               | Medium (1x params)     | Sparse gradients  |
+----------------+-------------------+------------------------+-------------------+
| LBFGS          | No                | High (history buffer)  | Small batches     |
+----------------+-------------------+------------------------+-------------------+

Scheduler Comparison
--------------------

+---------------------------+---------------------+---------------------------+
| Scheduler                 | Adjustment Pattern  | Best For                  |
+===========================+=====================+===========================+
| StepLR                    | Step decay          | Fixed schedule            |
+---------------------------+---------------------+---------------------------+
| MultiStepLR               | Multi-step decay    | Milestone-based           |
+---------------------------+---------------------+---------------------------+
| ExponentialLR             | Exponential decay   | Gradual reduction         |
+---------------------------+---------------------+---------------------------+
| CosineAnnealingLR         | Cosine curve        | Fixed epochs              |
+---------------------------+---------------------+---------------------------+
| CosineAnnealingWarmRestarts | Cosine w/ restarts | Long training             |
+---------------------------+---------------------+---------------------------+
| ReduceLROnPlateau         | Metric-based        | Adaptive training         |
+---------------------------+---------------------+---------------------------+
| OneCycleLR                | One cycle policy    | Super-convergence         |
+---------------------------+---------------------+---------------------------+
