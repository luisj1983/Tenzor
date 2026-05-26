"""
Tenzor: High-performance neural network and tensor library

A PyTorch-compatible deep learning framework with C++ backend for performance.

Basic usage:
    import tenzor as tz

    # Create tensors
    x = tz.randn([3, 4])
    y = tz.zeros([4, 2])

    # Create a neural network
    class MyNet(tz.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = tz.nn.Linear(10, 20)
            self.fc2 = tz.nn.Linear(20, 5)

        def forward_impl(self, x):
            x = tz.nn.relu(self.fc1(x))
            return self.fc2(x)

    model = MyNet()
    model.cuda()  # Move to GPU if available

Thread safety:
    The GIL is released during most C++ operations (matmul, conv2d, etc.),
    so Python threads can overlap computation with I/O. However:
    - Variables (autograd tensors) are NOT thread-safe by default.
    - Call var.make_thread_safe() before concurrent gradient accumulation.
    - NoGradGuard is thread-local and does NOT propagate to spawned threads.
    - For data-parallel training, use separate Variable instances per thread
      or call make_thread_safe() on shared parameters.
"""

# Import sys and importlib for module registration
import sys as _sys
import importlib.util as _importlib_util
import os as _os

# Configure optimal thread count before loading C++ library
# This prevents OpenMP from initializing with too many threads
if 'OMP_NUM_THREADS' not in _os.environ:
    try:
        import multiprocessing as _mp
        physical_cores = _mp.cpu_count() or 1
        # On Linux, try to detect hyperthreading to get physical core count
        try:
            with open('/sys/devices/system/cpu/cpu0/topology/thread_siblings_list') as f:
                siblings = len(f.read().strip().split(','))
                if siblings > 1:
                    physical_cores = max(1, physical_cores // siblings)
        except (OSError, IOError):
            pass  # Non-Linux: use cpu_count() directly
        _os.environ['OMP_NUM_THREADS'] = str(max(1, physical_cores))
    except Exception as _omp_detect_exc:
        # Audit item H.1: replace bare `except Exception: pass`.  CPU-count
        # detection isn't critical — the user's OMP_NUM_THREADS / system
        # default still applies — but a bare swallow hid real bugs in the
        # past.  Print a one-line breadcrumb to stderr at WARN level
        # equivalent (Python doesn't have stdlib logging configured yet at
        # this point in module init).
        import sys as _sys_for_warn
        _sys_for_warn.stderr.write(
            f"[tenzor] OMP_NUM_THREADS detection failed: "
            f"{type(_omp_detect_exc).__name__}: {_omp_detect_exc}\n"
        )

# Import C++ core module first
# NOTE: This will also register tenzor.nn in sys.modules pointing to C++ nn
from .tenzor_core import *

# _foreach_* names start with '_' so are not pulled in by import *; import explicitly.
from .tenzor_core import (
    _foreach_add, _foreach_add_,
    _foreach_sub, _foreach_sub_,
    _foreach_mul, _foreach_mul_,
    _foreach_div, _foreach_div_,
    _foreach_neg, _foreach_neg_,
    _foreach_abs, _foreach_abs_,
    _foreach_sqrt, _foreach_sqrt_,
    _foreach_zero_, _foreach_copy,
    _foreach_addcdiv_, _foreach_addcmul_,
    _foreach_lerp_, _foreach_norm,
)

# Audit E.10: module-level `from_numpy` convenience. Until now the .pyi
# advertised `tz.from_numpy(array, requires_grad=False)` but the runtime
# only exposed `tz.Tensor.from_numpy(...)`. Expose a thin top-level alias
# so the documented surface matches reality.
def from_numpy(array, requires_grad: bool = False):
    """Create a Tensor (or Variable, if ``requires_grad=True``) from a NumPy array.

    Mirrors ``torch.from_numpy``: zero-copy when the dtype and stride layout
    allow it, otherwise materialises a host copy. The returned tensor lives
    on CPU; move it explicitly with ``.to(device)``.
    """
    from .tenzor_core import Tensor as _Tensor, Variable as _Variable
    t = _Tensor.from_numpy(array)
    if requires_grad:
        return _Variable(t, True)
    return t


# Import the nn submodule from C++ (keep reference for internal use)
from .tenzor_core import nn as _cpp_nn

# CRITICAL: Manually load tenzor/nn.py to bypass sys.modules cache
# pybind11 registers 'tenzor.nn' pointing to C++ module, so normal import won't work
_nn_path = _os.path.join(_os.path.dirname(__file__), 'nn.py')
_spec = _importlib_util.spec_from_file_location('tenzor.nn', _nn_path)
_nn_module = _importlib_util.module_from_spec(_spec)

# Execute the module to define its contents
_spec.loader.exec_module(_nn_module)

# Override sys.modules with our Python wrapper
_sys.modules['tenzor.nn'] = _nn_module

# Make nn point to the wrapper module which has the enhanced Module class
nn = _nn_module

# Load nn.functional submodule
_func_path = _os.path.join(_os.path.dirname(__file__), 'functional.py')
_func_spec = _importlib_util.spec_from_file_location('tenzor.nn.functional', _func_path)
_func_module = _importlib_util.module_from_spec(_func_spec)
_func_spec.loader.exec_module(_func_module)
_sys.modules['tenzor.nn.functional'] = _func_module
_nn_module.functional = _func_module

# Load data submodule (Dataset, DataLoader, DistributedSampler, etc.)
_data_path = _os.path.join(_os.path.dirname(__file__), 'data.py')
if _os.path.exists(_data_path):
    _data_spec = _importlib_util.spec_from_file_location('tenzor.data', _data_path)
    _data_module = _importlib_util.module_from_spec(_data_spec)
    _data_spec.loader.exec_module(_data_module)
    # Re-attach the C++ data submodules (transforms, datasets) which are
    # registered under tenzor_core.data and would otherwise be shadowed
    # by the pure-Python tenzor.data wrapper that replaces sys.modules
    # ['tenzor.data'] below. Same pattern as tenzor.nn / tenzor.nn.functional.
    from .tenzor_core import data as _cpp_data
    if hasattr(_cpp_data, 'transforms'):
        _data_module.transforms = _cpp_data.transforms
        _sys.modules['tenzor.data.transforms'] = _cpp_data.transforms
    if hasattr(_cpp_data, 'datasets'):
        _data_module.datasets = _cpp_data.datasets
        _sys.modules['tenzor.data.datasets'] = _cpp_data.datasets
    _sys.modules['tenzor.data'] = _data_module
    data = _data_module

# Load autograd submodule with Python Function base class
_autograd_path = _os.path.join(_os.path.dirname(__file__), 'autograd.py')
if _os.path.exists(_autograd_path):
    _autograd_spec = _importlib_util.spec_from_file_location(
        'tenzor.autograd', _autograd_path)
    _autograd_module = _importlib_util.module_from_spec(_autograd_spec)
    _autograd_spec.loader.exec_module(_autograd_module)
    _sys.modules['tenzor.autograd'] = _autograd_module
    autograd = _autograd_module

# Expose linalg submodule from C++ bindings (det, inv, solve, svd, qr, etc.)
from .tenzor_core import linalg
_sys.modules['tenzor.linalg'] = linalg

# Audit-8 II.12: expose ``tenzor.exceptions`` as a typed-namespace alias to
# the C++ exception classes registered on ``tenzor_core``. This gives users a
# stable ``except tenzor.exceptions.ValueError`` surface that doesn't depend
# on the wildcard ``from .tenzor_core import *`` import order and doesn't
# shadow Python builtins when consumers write ``from tenzor.exceptions
# import ...`` explicitly. The original ``tz.MemoryError``-style attribute
# access continues to work (the typed exception names are still attributes of
# the ``tenzor`` namespace via the wildcard import above).
from . import exceptions as exceptions  # noqa: E402,F401

# Load special functions submodule (Bessel, Erf, Gamma, Ndtr, ...)
# Manually load to bypass the wildcard-import binding of `special` if
# tenzor_core ever grows a `special` submodule (currently it does not, so
# `from . import special` works, but the manual load is robust either way).
_special_path = _os.path.join(_os.path.dirname(__file__), 'special', '__init__.py')
if _os.path.exists(_special_path):
    _special_spec = _importlib_util.spec_from_file_location(
        'tenzor.special', _special_path,
        submodule_search_locations=[_os.path.dirname(_special_path)])
    _special_module = _importlib_util.module_from_spec(_special_spec)
    _sys.modules['tenzor.special'] = _special_module
    _special_spec.loader.exec_module(_special_module)
    special = _special_module
else:  # pragma: no cover — defensive only.
    from . import special as special  # type: ignore
    _sys.modules['tenzor.special'] = special

# Phase 13: MLIR/StableHLO/IREE JIT pipeline.
#
# `from .tenzor_core import *` above bound `tenzor.jit` to the C++ submodule
# `tenzor_core.jit` (a pybind11 module exposing show_graph/show_mlir/... and
# compile_function). The Python user-facing API is the sibling
# `python/tenzor/jit.py` module, which wraps `tenzor_core.jit.compile_function`
# in a @tz.jit decorator plus show_* helpers that accept either a raw fn or
# a wrapped fn. We load the Python module explicitly and re-bind:
#   tz.jit  -> the @tz.jit decorator (callable)
# while `tenzor.jit` as a sys.modules entry points at the Python wrapper so
# `tz.jit.show_graph(fn)` etc. resolve through it.
_jit_path = _os.path.join(_os.path.dirname(__file__), 'jit.py')
if _os.path.exists(_jit_path):
    _jit_spec = _importlib_util.spec_from_file_location('tenzor.jit', _jit_path)
    _jit_mod = _importlib_util.module_from_spec(_jit_spec)
    _sys.modules['tenzor.jit'] = _jit_mod
    _jit_spec.loader.exec_module(_jit_mod)
    jit = _jit_mod.jit  # the decorator (callable)
    # Expose show_* / cache_stats helpers as attributes of the decorator so
    # `tz.jit.show_graph(fn)` works (in addition to `tz._jit_mod.show_graph`).
    for _attr in ('show_graph', 'show_mlir', 'show_stablehlo', 'show_iree',
                  'cache_stats', 'reset_cache_stats', 'JitNotEnabledError'):
        if hasattr(_jit_mod, _attr):
            setattr(jit, _attr, getattr(_jit_mod, _attr))
    # Pre-Phase-13 tests use `tz.jit.trace`, `tz.jit.Compiler`,
    # `tz.jit.export_graph_text`, etc. — these are pybind11 attributes on
    # the C++ submodule `tenzor_core.jit`. To keep both APIs working we
    # also copy every non-private attribute from the C++ submodule onto
    # the Python decorator.
    try:
        from .tenzor_core import jit as _cpp_jit_mod  # type: ignore[import]
        for _attr in dir(_cpp_jit_mod):
            if not _attr.startswith('_') and not hasattr(jit, _attr):
                setattr(jit, _attr, getattr(_cpp_jit_mod, _attr))
    except (ImportError, AttributeError):
        pass
else:  # pragma: no cover — defensive only; jit.py is always shipped.
    from . import jit as _jit_mod  # type: ignore
    jit = _jit_mod.jit
    _sys.modules['tenzor.jit'] = _jit_mod

# Load distributions submodule (pure Python — numpy/scipy backend).
# Manually load because `from . import distributions` would resolve through
# the wildcard-import binding of the C++ submodule `tenzor_core.distributions`
# rather than the sibling Python package `python/tenzor/distributions/`.
_dist_path = _os.path.join(_os.path.dirname(__file__), 'distributions',
                           '__init__.py')
if _os.path.exists(_dist_path):
    _dist_spec = _importlib_util.spec_from_file_location(
        'tenzor.distributions', _dist_path,
        submodule_search_locations=[_os.path.dirname(_dist_path)])
    _dist_module = _importlib_util.module_from_spec(_dist_spec)
    _sys.modules['tenzor.distributions'] = _dist_module
    _dist_spec.loader.exec_module(_dist_module)
    distributions = _dist_module
else:  # pragma: no cover — defensive only.
    from . import distributions as distributions  # type: ignore
    _sys.modules['tenzor.distributions'] = distributions

# ----------------------------------------------------------------
# Phase 2.6 — __tensor_function__ subclass override protocol.
#
# Load the override helper module and wrap a curated set of the most
# commonly used public ops so Tensor subclasses that define
# __tensor_function__ intercept them. Covers:
#   add/sub/mul/div/matmul/bmm/sum/mean/max/min
#
# Users can extend the wrapped set via tz.overrides.implements; see
# tenzor/overrides.py for the full protocol semantics.
# ----------------------------------------------------------------
_overrides_path = _os.path.join(_os.path.dirname(__file__), 'overrides.py')
_overrides_spec = _importlib_util.spec_from_file_location(
    'tenzor.overrides', _overrides_path)
_overrides_module = _importlib_util.module_from_spec(_overrides_spec)
_overrides_spec.loader.exec_module(_overrides_module)
_sys.modules['tenzor.overrides'] = _overrides_module
overrides = _overrides_module

# X.8: `tenzor.optim` was bound as an attribute via `from .tenzor_core import *`
# above (it is a pybind11 submodule), but was never registered in sys.modules.
# That meant `import tenzor.optim as optim` failed with ModuleNotFoundError
# even though `tz.optim.Adam` worked fine.  Register it so the standard import
# form works (matches the pattern used for nn, data, autograd, linalg, etc.).
try:
    from .tenzor_core import optim as _cpp_optim  # type: ignore[import]
    _sys.modules['tenzor.optim'] = _cpp_optim
    optim = _cpp_optim
except (ImportError, AttributeError):
    pass

# Apply the @implements wrapper to the curated set of common ops.
# Only wrap names that exist on this module (the C++ binding surface
# can vary between builds, so be tolerant).
for _op_name in (
    # Arithmetic
    "add", "sub", "mul", "div", "matmul", "bmm", "dot",
    # Reduction
    "sum", "mean", "max", "min", "prod", "var", "std", "norm",
    "argmax", "argmin", "any", "all",
    # Element-wise math
    "sqrt", "exp", "log", "abs", "neg", "pow", "reciprocal",
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "sigmoid",
    "floor", "ceil", "round", "sign", "clamp",
    "log2", "log10", "log1p", "exp2", "expm1",
    "erf", "erfc", "rsqrt", "square",
    # Comparison
    "eq", "ne", "lt", "le", "gt", "ge",
    # Shape
    "reshape", "transpose", "permute", "squeeze", "unsqueeze",
    "flatten", "cat", "stack", "split", "chunk",
    # Indexing
    "gather", "scatter", "index_select", "where", "masked_select",
    # Creation
    "zeros", "ones", "full", "randn", "rand", "arange", "linspace", "eye",
    # Advanced
    "einsum", "topk", "sort", "unique", "flip", "roll",
    # New ops
    "normal", "poisson", "exponential", "bernoulli", "multinomial",
    "cov", "corrcoef", "channel_shuffle",
    "hann_window", "hamming_window", "blackman_window",
    "unique_consecutive",
):
    _op = globals().get(_op_name)
    if _op is not None and callable(_op):
        globals()[_op_name] = overrides.implements(_op)
del _op_name, _op

__version__ = "0.1.0"

__all__ = [
    # Core
    "Tensor",
    "Variable",
    "Device",
    "DeviceType",
    "dtype",
    "tensor",
    "manual_seed",

    # Autograd
    "is_grad_enabled",
    "set_grad_enabled",

    # Tensor creation
    "zeros",
    "ones",
    "randn",
    "rand",
    "randint",
    "eye",
    "arange",
    "linspace",
    "full",
    "empty",
    "randperm",

    # Arithmetic operations
    "matmul",
    "bmm",
    "addmm",
    "addmv",
    "baddbmm",
    "add",
    "sub",
    "mul",
    "div",

    # Reduction operations
    "sum",
    "mean",
    "max",
    "min",
    "argmax",
    "argmin",

    # Math operations
    "sqrt",
    "exp",
    "log",
    "abs",
    "pow",
    "neg",
    "sign",
    "sigmoid",
    "reciprocal",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "log2",
    "log10",
    "log1p",
    "exp2",
    "expm1",
    "erf",
    "erfc",
    "atan2",
    "fmod",
    "remainder",
    "lerp",
    "floor",
    "ceil",
    "round",

    # Clamping
    "clamp",
    "clamp_min",
    "clamp_max",
    "minimum",
    "maximum",

    # Classification
    "isnan",
    "isinf",
    "isfinite",

    # Logical operations
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",

    # Comparison
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",

    # Shape operations
    "reshape",
    "transpose",
    "permute",
    "flatten",
    "squeeze",
    "unsqueeze",
    "contiguous",
    "cat",
    "stack",
    "split",

    # Tensor manipulation
    "triu",
    "tril",
    "diag",
    "trace",
    "flip",
    "chunk",
    "sort",
    "topk",
    "unique",

    # Indexing operations
    "gather",
    "scatter",
    "scatter_add",
    "scatter_reduce",
    "select_scatter",
    "slice_scatter",
    "diagonal_scatter",
    "index_select",
    "masked_select",
    "masked_fill",
    "where",
    "meshgrid",
    "cross",

    # Cumulative operations
    "cumsum",
    "cumprod",
    "logcumsumexp",

    # Histogram / counting
    "bincount",

    # Index reduce
    "index_reduce",

    # Composed math operations
    "diff",
    "logaddexp",
    "logaddexp2",
    "xlogy",
    "tensordot",

    # Search/sampling operations
    "searchsorted",
    "gumbel_softmax",
    "roll",
    "split_with_sizes",

    # Neural network
    "nn",
    "optim",
    "autograd",
    "linalg",

    # Submodules
    "fft",
    "sparse",
    "amp",
    "data",

    # Serialization
    "save",
    "load",

    # Utilities
    "cuda_is_available",
    "no_grad",
    "enable_grad",
    "initialize",
    "empty_cache",
    "memory_stats",
    "reset_memory_stats",
    "set_anomaly_detection",
    "is_anomaly_detection_enabled",
    "memory_format",
    "list_backends",
    "is_backend_available",

    # Quantization
    "quantize_per_tensor",
    "qint8",
    "quint8",
    "qint4x2",

    # JIT compilation
    "compile",

    # FP8 scaling
    "fp8_max_value",
    "compute_amax",
    "compute_fp8_scale",
    "quantize_to_fp8",
    "dequantize_from_fp8",

    # Special functions submodule
    "special",

    # Probability distributions submodule
    "distributions",
    "jit",      # @tz.jit decorator (Phase 13 MLIR pipeline)
    "_jit_mod", # full jit submodule for show_*/cache_stats helpers

    # Special functions (also accessible via tz.special.*) — erf, erfc,
    # xlogy are exported from the earlier "Math operations" block to keep
    # __all__ free of duplicates (audit item H.4).
    "erfinv",
    "lgamma",
    "digamma",
    "polygamma",
    "sinc",
    "zeta",
    "ndtri",
    "xlog1py",
    "bessel_j0",
    "bessel_j1",
    "bessel_y0",
    "bessel_y1",
    "bessel_i0",
    "bessel_i1",

    # Tensor utilities (previously missing)
    "allclose",
    "aminmax",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "block_diag",
    "broadcast_tensors",
    "broadcast_to",
    "cartesian_prod",
    "column_stack",
    "combinations",
    "count_nonzero",
    "cuda_device_count",
    "current_device",
    "deg2rad",
    "dsplit",
    "dstack",
    "enable_auto_checkpoint",
    "expand",
    "float_power",
    "frac",
    "frexp",
    "heaviside",
    "histogramdd",
    "hsplit",
    "hstack",
    "index_add",
    "index_copy",
    "index_fill",
    "isclose",
    "isneginf",
    "isposinf",
    "isreal",
    "kron",
    "ldexp",
    "logit",
    "logspace",
    "moveaxis",
    "movedim",
    "nanmean",
    "nanstd",
    "nansum",
    "nan_to_num",
    "nanvar",
    "narrow_copy",
    "one_hot",
    "oneapi_device_count",
    "oneapi_is_available",
    "pairwise_distance",
    "pdist",
    "pixel_shuffle",
    "pixel_unshuffle",
    "put",
    "rad2deg",
    "randn_with_generator",
    "rand_with_generator",
    "repeat",
    "repeat_interleave",
    "rocm_device_count",
    "rocm_is_available",
    "row_stack",
    "signbit",
    "slice",
    "swapaxes",
    "take",
    "tensor_split",
    "tile",
    "vander",
    "view_as_complex",
    "view_as_real",
    "vsplit",
    "vstack",
    "vulkan_device_count",
    "vulkan_is_available",

    # Multi-tensor foreach optimizer primitives (Phase 9-W2)
    "_foreach_add", "_foreach_add_", "_foreach_sub", "_foreach_sub_",
    "_foreach_mul", "_foreach_mul_", "_foreach_div", "_foreach_div_",
    "_foreach_neg", "_foreach_neg_", "_foreach_abs", "_foreach_abs_",
    "_foreach_sqrt", "_foreach_sqrt_", "_foreach_zero_", "_foreach_copy",
    "_foreach_addcdiv_", "_foreach_addcmul_", "_foreach_lerp_", "_foreach_norm",
]

# Quantized dtype aliases (PyTorch-compatible names)
qint8 = dtype.qint8
quint8 = dtype.quint8
qint4x2 = dtype.qint4x2
