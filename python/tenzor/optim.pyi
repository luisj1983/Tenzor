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

    def zero_grad(self, set_to_none: bool = True) -> None: ...
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

# Stream S10: extra schedulers, additional optimizers, and config types.
# All of these are registered by ``python/bindings/bindings_optim.cpp`` but
# were previously absent from the stub, so static type-checkers reported
# "undefined attribute" on legitimate ``tz.optim.SAM(...)`` calls etc.
# Constructor signatures are extracted from the binding's ``py::init`` /
# ``py::class_`` declarations where straightforward; opaque or heavily
# overloaded classes use the minimal ``def __init__(self, *args, **kwargs)``
# escape hatch to surface the name without lying about parameters.


class ConstantLR(LRScheduler):
    """Multiplies the learning rate by a constant ``factor`` until ``total_iters``."""

    def __init__(
        self,
        optimizer: Optimizer,
        factor: float = 1.0 / 3.0,
        total_iters: int = 5,
    ) -> None: ...


class LinearLR(LRScheduler):
    """Linearly interpolates the learning rate from ``start_factor`` to ``end_factor``."""

    def __init__(
        self,
        optimizer: Optimizer,
        start_factor: float = 1.0 / 3.0,
        end_factor: float = 1.0,
        total_iters: int = 5,
    ) -> None: ...


class SequentialLR(LRScheduler):
    """Switches between a list of schedulers at the given milestone steps."""

    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: List[LRScheduler],
        milestones: List[int],
    ) -> None: ...


class ChainedScheduler(LRScheduler):
    """Composes multiple schedulers, applying each in turn at every step."""

    def __init__(self, schedulers: List[LRScheduler]) -> None: ...


class SWALR(LRScheduler):
    """SWA learning-rate schedule (constant after the swa_lr ramp)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


# Additional optimizers exposed by the C++ bindings but previously missing
# from the stub.

class ASGD(Optimizer):
    """Averaged Stochastic Gradient Descent."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class LAMB(Optimizer):
    """Layer-wise Adaptive Moments (LAMB) optimizer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class SparseAdam(Optimizer):
    """Adam optimizer specialised for sparse gradients."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class Rprop(Optimizer):
    """Resilient backpropagation optimizer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class SAM(Optimizer):
    """Sharpness-Aware Minimization wrapper around a base optimizer."""

    def __init__(
        self,
        base_optimizer: Optimizer,
        rho: float = 0.05,
    ) -> None: ...

    def first_step(self) -> None: ...
    def second_step(self) -> None: ...
    def set_lr(self, lr: float) -> None: ...
    def get_lr(self) -> float: ...
    def get_rho(self) -> float: ...
    def set_rho(self, rho: float) -> None: ...
    def base_optimizer(self) -> Optimizer: ...


class AveragedModel:
    """Maintains a running average of model parameters (SWA/EMA)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class AdamAtan2(Optimizer):
    """Adam variant using atan2 normalisation (epsilon-free)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


# Gradient-clipping configuration types.

class ClipMode:
    """Enum: gradient-clipping mode (NONE / NORM / VALUE)."""

    NONE: ClipMode
    NORM: ClipMode
    VALUE: ClipMode

    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...


class ClipConfig:
    """Configuration for per-step gradient clipping inside an Optimizer."""

    def __init__(
        self,
        mode: ClipMode = ...,
        max_norm: float = 1.0,
        norm_type: float = 2.0,
    ) -> None: ...


class LBFGSLineSearch:
    """Enum: line-search strategy for LBFGS (Armijo / StrongWolfe)."""

    Armijo: LBFGSLineSearch
    StrongWolfe: LBFGSLineSearch


# ZeRO partitioned-optimizer configs and optimizers.

class ZeROStage1Config:
    """Configuration for ZeROStage1Optimizer (optimizer-state partitioning)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class ZeROStage2Config(ZeROStage1Config):
    """Configuration for ZeROStage2Optimizer (Stage1 + gradient partitioning)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class ZeROStage3Config(ZeROStage2Config):
    """Configuration for ZeROStage3Optimizer (Stage2 + parameter partitioning)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


class ZeROStage1Optimizer:
    """ZeRO Stage 1: partition optimizer state across ranks."""

    def __init__(
        self,
        base_optimizer: Optimizer,
        config: ZeROStage1Config,
    ) -> None: ...

    def step(self) -> None: ...
    def zero_grad(self, set_to_none: bool = True) -> None: ...
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state: Dict[str, Any]) -> None: ...


class ZeROStage2Optimizer(ZeROStage1Optimizer):
    """ZeRO Stage 2: Stage 1 plus gradient partitioning."""

    def __init__(
        self,
        base_optimizer: Optimizer,
        config: ZeROStage2Config,
    ) -> None: ...


class ZeROStage3Optimizer(ZeROStage2Optimizer):
    """ZeRO Stage 3: Stage 2 plus parameter partitioning."""

    def __init__(
        self,
        base_optimizer: Optimizer,
        config: ZeROStage3Config,
    ) -> None: ...


# ---------------------------------------------------------------------------
# Submodule re-export: ``tenzor.optim.lr_scheduler``.
#
# C++ bindings register every LR scheduler under ``tenzor.optim.lr_scheduler``
# for PyTorch parity (``torch.optim.lr_scheduler.StepLR`` etc.). ``__init__.py``
# also hoists those classes to the top-level ``tenzor.optim`` namespace so the
# top-level declarations above remain accurate. The submodule alias here lets
# ``from tenzor.optim.lr_scheduler import StepLR`` typecheck cleanly.
# ---------------------------------------------------------------------------


class lr_scheduler:
    """``tenzor.optim.lr_scheduler`` submodule (PyTorch-style alias).

    Every scheduler registered on this submodule is also re-exported at the
    top of ``tenzor.optim``; the two names resolve to the same class object.
    """

    LRScheduler = LRScheduler
    StepLR = StepLR
    MultiStepLR = MultiStepLR
    ExponentialLR = ExponentialLR
    CosineAnnealingLR = CosineAnnealingLR
    CosineAnnealingWarmRestarts = CosineAnnealingWarmRestarts
    ReduceLROnPlateau = ReduceLROnPlateau
    CyclicLR = CyclicLR
    OneCycleLR = OneCycleLR
    LambdaLR = LambdaLR
    MultiplicativeLR = MultiplicativeLR
    ConstantLR = ConstantLR
    LinearLR = LinearLR
    SequentialLR = SequentialLR
    ChainedScheduler = ChainedScheduler
    SWALR = SWALR


# Audit E.10: clip_grad_norm_ / clip_grad_value_ live in tenzor.nn.functional
# (and are re-exported through tenzor.nn). The previous declarations here
# never existed in `tenzor.optim` at runtime and drove typecheckers to
# "undefined attribute" on legitimate `tz.nn.functional.clip_grad_norm_`
# calls. The canonical declarations are in functional.pyi / nn.pyi.

# Stream S23: pybind11 hoists ``ClipMode`` and ``LBFGSLineSearch`` enum
# members to the parent ``tenzor.optim`` namespace (so users can write
# ``tz.optim.NORM`` as shorthand for ``tz.optim.ClipMode.NORM``). Declare
# them at module scope so static checkers see the shortcuts.
NONE: ClipMode
NORM: ClipMode
VALUE: ClipMode
Armijo: LBFGSLineSearch
StrongWolfe: LBFGSLineSearch
