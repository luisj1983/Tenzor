"""Stream S10: LR-scheduler / optim namespace parity.

The C++ bindings (``python/bindings/bindings_optim.cpp``) register every
learning-rate scheduler under the submodule ``tenzor.optim.lr_scheduler``
for PyTorch parity. Historically the ``.pyi`` stub and a fair amount of
external code expected those same names on the top-level ``tenzor.optim``
namespace as well, so ``from tenzor.optim import StepLR`` would
``ImportError`` even though ``tenzor.optim.lr_scheduler.StepLR`` was fine.

S10 hoists every public name from ``lr_scheduler`` to its parent module
so BOTH import paths resolve to the SAME class object. These tests pin
that contract.

In addition, S10 surfaces a batch of optimiser / config classes that were
registered in the bindings but missing from the stub (ASGD, LAMB,
SparseAdam, Rprop, SAM, AveragedModel, ZeRO* configs/optimisers,
AdamAtan2, ClipMode, ClipConfig, LBFGSLineSearch). The audit assertions
below pin the existence of each name on the runtime module.
"""

from __future__ import annotations

import importlib

import pytest

import tenzor  # noqa: F401 — ensures the pure-Python wrapper module runs.
import tenzor.optim as tz_optim


# ---------------------------------------------------------------------------
# 1. Top-level imports work for every scheduler.
# ---------------------------------------------------------------------------


def test_top_level_lr_scheduler_imports() -> None:
    """``from tenzor.optim import StepLR, ...`` must resolve every scheduler."""

    # Names with explicit C++ bindings in ``bindings_optim.cpp``.  Kept in
    # sync with the grep in the S10 plan; if a future binding adds a new
    # scheduler, add it here.
    expected_names = [
        "LRScheduler",
        "StepLR",
        "MultiStepLR",
        "ExponentialLR",
        "CosineAnnealingLR",
        "CosineAnnealingWarmRestarts",
        "ReduceLROnPlateau",
        "CyclicLR",
        "OneCycleLR",
        "LambdaLR",
        "MultiplicativeLR",
        "ConstantLR",
        "LinearLR",
        "SequentialLR",
        "ChainedScheduler",
        "SWALR",
    ]

    # Filter to whatever the C++ submodule actually publishes — the build
    # under test may legitimately disable a scheduler (e.g. SWALR depends
    # on AveragedModel).  Anything that *is* in the submodule must be
    # hoisted; anything missing from the submodule is allowed to be missing
    # at the top level too.
    submod = getattr(tz_optim, "lr_scheduler", None)
    assert submod is not None, "tenzor.optim.lr_scheduler submodule missing"
    actually_published = {n for n in expected_names if hasattr(submod, n)}

    missing = [n for n in actually_published if not hasattr(tz_optim, n)]
    assert not missing, (
        "Scheduler names registered on tenzor.optim.lr_scheduler but NOT "
        f"hoisted to top-level tenzor.optim: {missing!r}. "
        "Fix in python/tenzor/__init__.py (S10)."
    )


def test_explicit_top_level_imports_smoke() -> None:
    """Belt-and-braces: do the actual ``from ... import`` the user would write."""

    from tenzor.optim import (  # noqa: F401 — import-time check is the test
        StepLR,
        MultiStepLR,
        ExponentialLR,
        CosineAnnealingLR,
        CosineAnnealingWarmRestarts,
        ReduceLROnPlateau,
        CyclicLR,
        OneCycleLR,
        LambdaLR,
        MultiplicativeLR,
        ConstantLR,
        LinearLR,
        SequentialLR,
        ChainedScheduler,
    )


# ---------------------------------------------------------------------------
# 2. Submodule path keeps working.
# ---------------------------------------------------------------------------


def test_lr_scheduler_submodule_import() -> None:
    """``from tenzor.optim.lr_scheduler import StepLR`` must keep working."""

    from tenzor.optim.lr_scheduler import StepLR  # noqa: F401

    # The submodule itself should also be importable as a module.
    mod = importlib.import_module("tenzor.optim.lr_scheduler")
    assert hasattr(mod, "StepLR")


# ---------------------------------------------------------------------------
# 3. Identity: top-level alias and submodule resolve to the same class.
# ---------------------------------------------------------------------------


def test_top_level_and_submodule_class_identity() -> None:
    """``tenzor.optim.StepLR is tenzor.optim.lr_scheduler.StepLR``.

    Two distinct copies would silently break ``isinstance`` checks and
    pickled state-dicts moving between code paths that used different
    import styles.
    """

    submod = tz_optim.lr_scheduler
    for name in dir(submod):
        if name.startswith("_"):
            continue
        sub_obj = getattr(submod, name)
        top_obj = getattr(tz_optim, name, None)
        assert top_obj is sub_obj, (
            f"tenzor.optim.{name} is not the same object as "
            f"tenzor.optim.lr_scheduler.{name}: "
            f"{top_obj!r} vs {sub_obj!r}"
        )


# ---------------------------------------------------------------------------
# 4. Audit names: optimiser / config classes flagged missing from the stub.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        # Optimizers
        "ASGD",
        "LAMB",
        "SparseAdam",
        "Rprop",
        "SAM",
        "AdamAtan2",
        # Helpers
        "AveragedModel",
        # Gradient-clip config
        "ClipMode",
        "ClipConfig",
        # LBFGS line-search enum
        "LBFGSLineSearch",
        # ZeRO partitioned optimisers + configs
        "ZeROStage1Config",
        "ZeROStage2Config",
        "ZeROStage3Config",
        "ZeROStage1Optimizer",
        "ZeROStage2Optimizer",
        "ZeROStage3Optimizer",
    ],
)
def test_optim_audit_name_exists(name: str) -> None:
    """Each audit-flagged class is reachable as an attribute of ``tenzor.optim``."""

    assert hasattr(tz_optim, name), (
        f"tenzor.optim.{name} missing at runtime; S10 .pyi audit declared it "
        "but the C++ binding registration appears to have regressed."
    )
