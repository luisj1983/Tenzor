Optimizers
==========

Optimization algorithms and learning rate schedulers.

Optimizers
----------

- ``tenzor.optim.SGD(params, lr, momentum=0, weight_decay=0, nesterov=False)``
- ``tenzor.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)``
- ``tenzor.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)``
- ``tenzor.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-8)``
- ``tenzor.optim.Adagrad(params, lr=0.01)``
- ``tenzor.optim.Adadelta(params, lr=1.0, rho=0.9)``
- ``tenzor.optim.LAMB(params, lr=0.001)``
- ``tenzor.optim.RAdam(params, lr=0.001)``
- ``tenzor.optim.SparseAdam(params, lr=0.001)``

Learning Rate Schedulers
------------------------

- ``tenzor.optim.StepLR(optimizer, step_size, gamma=0.1)``
- ``tenzor.optim.ExponentialLR(optimizer, gamma)``
- ``tenzor.optim.CosineAnnealingLR(optimizer, T_max, eta_min=0)``
- ``tenzor.optim.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)``
- ``tenzor.optim.CyclicLR(optimizer, base_lr, max_lr)``
- ``tenzor.optim.OneCycleLR(optimizer, max_lr, total_steps)``
