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
"""

# Import sys and importlib for module registration
import sys as _sys
import importlib.util as _importlib_util
import os as _os

# Configure optimal thread count before loading C++ library
# This prevents OpenMP from initializing with too many threads
if 'OMP_NUM_THREADS' not in _os.environ:
    try:
        # Detect physical cores on Linux
        with open('/sys/devices/system/cpu/cpu0/topology/thread_siblings_list') as f:
            siblings = f.read().strip()
            threads_per_core = siblings.count(',') + 1
            logical_cores = _os.cpu_count() or 1
            physical_cores = max(1, logical_cores // threads_per_core)
            _os.environ['OMP_NUM_THREADS'] = str(physical_cores)
    except (OSError, IOError):
        pass  # Use default if detection fails

# Import C++ core module first
# NOTE: This will also register tenzor.nn in sys.modules pointing to C++ nn
from .tenzor_core import *

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

__version__ = "1.0.0"

__all__ = [
    # Core
    "Tensor",
    "Variable",
    "Device",
    "DeviceType",
    "dtype",

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

    # Operations
    "matmul",
    "bmm",
    "add",
    "sub",
    "mul",
    "div",
    "sum",
    "mean",
    "max",
    "min",
    "argmax",
    "argmin",
    "sqrt",
    "exp",
    "log",
    "abs",
    "pow",
    "clamp",

    # Shape operations
    "reshape",
    "transpose",
    "flatten",
    "squeeze",
    "unsqueeze",
    "cat",
    "stack",
    "split",
    "chunk",

    # Neural network
    "nn",

    # Utilities
    "cuda_is_available",
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "is_grad_enabled",
    "initialize",
]
