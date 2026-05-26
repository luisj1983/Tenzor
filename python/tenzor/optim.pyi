"""Type stubs for tenzor.optim module (optimizers and schedulers)."""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Callable, Tuple, Iterable
from tenzor import Tensor, Variable

class ParamGroup:
    """Parameter group with per-group hyperparameter overrides.

    Mirrors the C++ ``tenzor::optim::ParamGroup`` struct exposed by
    ``bindings_optim.cpp``. Every hyperparameter (except ``params``, ``lr``,
    and ``weight_decay``) is optional; unset fields fall back to the owning
    optimiser's default at ``step()`` time (see ``ParamGroup::or_else`` in
    ``optimizer.hpp``).
    """

    params: List[Tensor]
    lr: float
    weight_decay: float
    momentum: Optional[float]
    dampening: Optional[float]
    nesterov: Optional[bool]
    beta1: Optional[float]
    beta2: Optional[float]
    eps: Optional[float]
    amsgrad: Optional[bool]
    centered: Optional[bool]
    alpha: Optional[float]
    rho: Optional[float]
    lr_decay: Optional[float]
    initial_accumulator_value: Optional[float]

    def __init__(
        self,
        params: List[Tensor],
        lr: float,
        weight_decay: float = 0.0,
        momentum: Optional[float] = None,
        dampening: Optional[float] = None,
        nesterov: Optional[bool] = None,
        beta1: Optional[float] = None,
        beta2: Optional[float] = None,
        eps: Optional[float] = None,
        amsgrad: Optional[bool] = None,
        centered: Optional[bool] = None,
        alpha: Optional[float] = None,
        rho: Optional[float] = None,
        lr_decay: Optional[float] = None,
        initial_accumulator_value: Optional[float] = None,
    ) -> None: ...

class Optimizer:
    """Base class for all optimizers."""

    param_groups: List[ParamGroup]
    defaults: Dict[str, Any]

    def __init__(self, params: Iterable[Variable], defaults: Dict[str, Any]) -> None: ...

    def zero_grad(self, set_to_none: bool = False) -> None: ...
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]: ...
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...
    def add_param_group(self, param_group: ParamGroup) -> None: ...

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(
        self,
        params: Iterable[Variable],
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ) -> None: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]: ...

class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        params: Iterable[Variable],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False
    ) -> None: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]: ...

class AdamW(Optimizer):
    """AdamW optimizer (Adam with decoupled weight decay)."""

    def __init__(
        self,
        params: Iterable[Variable],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False
    ) -> None: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]: ...

class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(
        self,
        params: Iterable[Variable],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False
    ) -> None: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]: ...

class Adagrad(Optimizer):
    """Adagrad optimizer."""

    def __init__(
        self,
        params: Iterable[Variable],
        lr: float = 1e-2,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10
    ) -> None: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]: ...

class Adadelta(Optimizer):
    """Adadelta optimizer."""

    def __init__(
        self,
        params: Iterable[Variable],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0
    ) -> None: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]: ...

class LBFGS(Optimizer):
    """L-BFGS quasi-Newton optimizer. Requires a closure for line search."""

    def __init__(
        self,
        params: Iterable[Variable],
        lr: float = 1.0,
        max_iter: int = 20,
        max_eval: int = -1,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100
    ) -> None: ...

    def step(self, closure: Callable[[], float]) -> float: ...

class Adamax(Optimizer):
    """Adamax optimizer (variant of Adam based on infinity norm)."""

    def __init__(
        self,
        params: Iterable[Variable],
        lr: float = 2e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ) -> None: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]: ...

class NAdam(Optimizer):
    """NAdam optimizer (Nesterov-accelerated Adam)."""

    def __init__(
        self,
        params: Iterable[Variable],
        lr: float = 2e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_decay: float = 4e-3
    ) -> None: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]: ...

class RAdam(Optimizer):
    """RAdam optimizer (Rectified Adam)."""

    def __init__(
        self,
        params: Iterable[Variable],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ) -> None: ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]: ...

# Learning rate schedulers
class LRScheduler:
    """Base class for learning rate schedulers."""

    optimizer: Optimizer
    last_epoch: int

    def __init__(
        self,
        optimizer: Optimizer,
        last_epoch: int = -1,
        verbose: bool = False
    ) -> None: ...

    def step(self, epoch: Optional[int] = None) -> None: ...
    def get_last_lr(self) -> List[float]: ...
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...

class StepLR(LRScheduler):
    """Step learning rate scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False
    ) -> None: ...

class MultiStepLR(LRScheduler):
    """Multi-step learning rate scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False
    ) -> None: ...

class ExponentialLR(LRScheduler):
    """Exponential learning rate scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        last_epoch: int = -1,
        verbose: bool = False
    ) -> None: ...

class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False
    ) -> None: ...

class CosineAnnealingWarmRestarts(LRScheduler):
    """Cosine annealing with warm restarts."""

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False
    ) -> None: ...

class ReduceLROnPlateau:
    """Reduce learning rate when a metric has stopped improving."""

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = 'rel',
        cooldown: int = 0,
        min_lr: float = 0.0,
        eps: float = 1e-8,
        verbose: bool = False
    ) -> None: ...

    def step(self, metrics: float, epoch: Optional[int] = None) -> None: ...
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...

class CyclicLR(LRScheduler):
    """Cyclic learning rate scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float,
        max_lr: float,
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = 'triangular',
        gamma: float = 1.0,
        scale_fn: Optional[Callable[[int], float]] = None,
        scale_mode: str = 'cycle',
        cycle_momentum: bool = True,
        base_momentum: float = 0.8,
        max_momentum: float = 0.9,
        last_epoch: int = -1,
        verbose: bool = False
    ) -> None: ...

class OneCycleLR(LRScheduler):
    """One cycle learning rate scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        cycle_momentum: bool = True,
        base_momentum: float = 0.85,
        max_momentum: float = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        three_phase: bool = False,
        last_epoch: int = -1,
        verbose: bool = False
    ) -> None: ...

class LambdaLR(LRScheduler):
    """Lambda learning rate scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float] | List[Callable[[int], float]],
        last_epoch: int = -1,
        verbose: bool = False
    ) -> None: ...

class MultiplicativeLR(LRScheduler):
    """Multiplicative learning rate scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable[[int], float] | List[Callable[[int], float]],
        last_epoch: int = -1,
        verbose: bool = False
    ) -> None: ...

# Audit E.10: clip_grad_norm_ / clip_grad_value_ live in tenzor.nn.functional
# (and are re-exported through tenzor.nn). The previous declarations here
# never existed in `tenzor.optim` at runtime and drove typecheckers to
# "undefined attribute" on legitimate `tz.nn.functional.clip_grad_norm_`
# calls. The canonical declarations are in functional.pyi / nn.pyi.
