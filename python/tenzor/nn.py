"""
Tenzor Neural Network Module Wrapper

Provides PyTorch-compatible Module class with automatic submodule/parameter registration.
When you assign a Module or Variable as an attribute, it's automatically registered.

Example:
    class MyNet(tz.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = tz.nn.Linear(10, 20)  # Auto-registered
            self.fc2 = tz.nn.Linear(20, 10)  # Auto-registered

        def forward_impl(self, x):
            x = self.fc1(x)
            x = tz.nn.relu(x)
            return self.fc2(x)
"""

from . import tenzor_core as _core

# Store reference to original C++ Module BEFORE importing *
_CppModule = _core.nn.Module

# Re-export all nn items from the C++ module (including the original Module)
# We'll override Module below
from .tenzor_core.nn import *


class Module(_CppModule):
    """
    Base class for all neural network modules.

    Your models should subclass this class. Modules can contain other Modules,
    allowing them to be nested in a tree structure. You can assign submodules
    as regular attributes, and they will be automatically registered.

    All module state (parameters, buffers, submodules) is stored in C++ and
    accessed through C++ accessors, ensuring Python and C++ state never desync.

    Example::

        import tenzor as tz

        class MyModel(tz.nn.Module):
            def __init__(self):
                super().__init__()
                # These are automatically registered as submodules
                self.conv1 = tz.nn.Conv2d(1, 20, 5)
                self.conv2 = tz.nn.Conv2d(20, 50, 5)
                self.fc1 = tz.nn.Linear(800, 500)
                self.fc2 = tz.nn.Linear(500, 10)

            def forward_impl(self, x):
                x = tz.nn.relu(self.conv1(x))
                x = tz.nn.max_pool2d(x, 2)
                x = tz.nn.relu(self.conv2(x))
                x = tz.nn.max_pool2d(x, 2)
                x = tz.flatten(x, 1)
                x = tz.nn.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = MyModel()
        model.cuda()  # Move to GPU
        output = model(input_tensor)
    """

    def __init__(self):
        super().__init__()

    def __setattr__(self, name: str, value) -> None:
        """
        Automatically register Modules, Variables as submodules/parameters.

        This enables PyTorch-like syntax where assigning a Module attribute
        automatically registers it. All state is stored exclusively in C++
        via register_module/register_parameter/register_buffer.
        """
        # Check if it's an internal attribute (starts with _)
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return

        # Handle Module assignment
        if isinstance(value, _CppModule):
            # Register the module with C++ backend (single source of truth)
            self.register_module(name, value)
            return

        # Handle Variable assignment (potential parameter)
        if isinstance(value, _core.Variable):
            # Check if requires_grad - if so, it's a parameter
            if value.requires_grad():
                self.register_parameter(name, value)
            else:
                # It's a buffer (non-trainable)
                self.register_buffer(name, value)
            return

        # Regular attribute
        object.__setattr__(self, name, value)

    def _get_own_named_params(self):
        """Get this module's own parameters as a dict {name: Variable}.

        Calls C++ named_parameters() which returns all params recursively with
        dotted names. Own params have bare names (no dots). Since the Python
        Module class does not define named_parameters(), the PYBIND11_OVERRIDE
        in the trampoline falls through to the C++ base Module::named_parameters().
        """
        result = {}
        for name, param in self.named_parameters():
            if '.' not in name:
                result[name] = param
        return result

    def _get_own_named_buffers(self):
        """Get this module's own buffers as a dict {name: Variable}.

        Uses C++ named_buffers() which is non-virtual and always calls the
        C++ base implementation directly.
        """
        result = {}
        for name, buf in self.named_buffers():
            if '.' not in name:
                result[name] = buf
        return result

    def __getattr__(self, name: str):
        """
        Get attribute, delegating to C++ accessors for modules/parameters/buffers.

        This ensures we always return the current C++ state, avoiding desync
        issues that arose from maintaining separate Python dicts.

        Lookup order: submodules (O(1) dict), then own parameters, then own
        buffers. Submodule access (the most common case in forward passes)
        is fast via get_submodules() which returns the C++ map directly.
        """
        if name.startswith('_'):
            return object.__getattribute__(self, name)

        # Check in C++ submodules (get_submodules() is non-virtual, always C++)
        submodules = self.get_submodules()
        if name in submodules:
            return submodules[name]

        # Check in C++ own parameters (named_parameters falls through to C++ base)
        own_params = self._get_own_named_params()
        if name in own_params:
            return own_params[name]

        # Check in C++ own buffers (named_buffers is non-virtual, always C++)
        own_buffers = self._get_own_named_buffers()
        if name in own_buffers:
            return own_buffers[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __delattr__(self, name: str) -> None:
        """Remove attribute from appropriate registry.

        Note: C++ does not expose unregister methods for parameters, buffers,
        or submodules. For regular Python attributes, delegates to object.__delattr__.
        For registered C++ state, this is a no-op (the C++ state persists).
        """
        submodules = self.get_submodules()
        if name in submodules:
            return
        own_params = self._get_own_named_params()
        if name in own_params:
            return
        own_buffers = self._get_own_named_buffers()
        if name in own_buffers:
            return
        object.__delattr__(self, name)

    def add_module(self, name: str, module) -> None:
        """
        Add a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name: Name of the child module
            module: Child module to add
        """
        if module is None:
            return

        if not isinstance(module, _CppModule):
            raise TypeError(f"module must be a Module, got {type(module)}")

        self.register_module(name, module)

    def children(self):
        """Returns an iterator over immediate children modules."""
        for name, module in self.get_submodules().items():
            if module is not None:
                yield module

    def named_children(self):
        """Returns an iterator over immediate children modules, yielding tuples of (name, module)."""
        for name, module in self.get_submodules().items():
            if module is not None:
                yield name, module

    def modules(self):
        """Returns an iterator over all modules in the network, including self."""
        yield self
        for name, module in self.get_submodules().items():
            if module is not None:
                if hasattr(module, 'modules'):
                    yield from module.modules()
                else:
                    yield module

    def named_modules(self, prefix: str = ''):
        """Returns an iterator over all modules in the network, yielding tuples of (name, module)."""
        yield prefix, self
        for name, module in self.get_submodules().items():
            if module is not None:
                submodule_prefix = f"{prefix}.{name}" if prefix else name
                if hasattr(module, 'named_modules'):
                    yield from module.named_modules(submodule_prefix)
                else:
                    yield submodule_prefix, module

    def apply(self, fn):
        """
        Apply a function recursively to every submodule and self.

        Args:
            fn: Function to apply to each module

        Returns:
            self
        """
        for module in self.children():
            if hasattr(module, 'apply'):
                module.apply(fn)
            fn(module)
        fn(self)
        return self

    # train(), eval(), training property: delegate to C++ (already bound).
    # The C++ Module::train(mode) sets training_ and recurses into submodules_.
    # The C++ Module::is_training() reads training_ directly.
    # The pybind11 binding exposes these as methods and a read-only property.
    # We do NOT override them here so the C++ implementation is the single
    # source of truth for training mode.

    # parameters(), named_parameters(): delegate to C++ (already bound).
    # The C++ Module::parameters() and Module::named_parameters() read from
    # the C++ parameters_ and submodules_ maps, which are always up-to-date.
    # PYBIND11_OVERRIDE in the trampoline falls through to C++ base because
    # we do NOT define these methods here.

    # state_dict(), load_state_dict(): delegate to C++ (already bound).
    # The C++ implementations read/write the C++ parameters_, buffers_, and
    # submodules_ maps directly, ensuring consistency.
    # PYBIND11_OVERRIDE falls through to C++ base since we do NOT define
    # these methods here.

    # to(device), to(dtype): delegate to C++ (already bound).
    # The C++ Module::to() modifies parameters and buffers in-place within
    # the C++ maps, then recurses into submodules. Since we no longer cache
    # Variable objects in Python dicts, there is no desync.

    def requires_grad_(self, requires_grad: bool = True):
        """
        Change if autograd should record operations on parameters.

        Args:
            requires_grad: Whether to require gradients

        Returns:
            self
        """
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def __repr__(self) -> str:
        """Return a string representation of the module."""
        lines = [f"{self.__class__.__name__}("]
        for name, module in self.get_submodules().items():
            mod_str = repr(module).replace('\n', '\n  ')
            lines.append(f"  ({name}): {mod_str}")
        lines.append(")")
        return '\n'.join(lines) if len(lines) > 2 else f"{self.__class__.__name__}()"


# Store reference to original C++ Sequential BEFORE overriding
_CppSequential = _core.nn.Sequential


class Sequential(_CppSequential):
    """
    A sequential container that holds modules in the order they were added.

    Modules can be passed as arguments to the constructor, or added using append().

    Example::

        import tenzor as tz

        # Using constructor arguments
        model = tz.nn.Sequential(
            tz.nn.Linear(10, 20),
            tz.nn.ReLU(),
            tz.nn.Linear(20, 10)
        )

        # Using append
        model = tz.nn.Sequential()
        model.append(tz.nn.Linear(10, 20))
        model.append(tz.nn.ReLU())

        # Forward pass applies modules in order
        output = model(input)
    """

    def __init__(self, *args):
        """
        Initialize Sequential container.

        Args:
            *args: Variable number of modules to add in order
        """
        super().__init__()
        # Keep Python references alive to prevent garbage collection
        self._py_modules = list(args)
        for module in args:
            self.append(module)

    def __repr__(self) -> str:
        """Return a string representation of the sequential container."""
        lines = ["Sequential("]
        # Use __len__ and __getitem__ to iterate modules
        for i in range(len(self)):
            mod = self[i]
            mod_str = repr(mod).replace('\n', '\n  ')
            lines.append(f"  ({i}): {mod_str}")
        lines.append(")")
        return '\n'.join(lines) if len(self) > 0 else "Sequential()"


class Parameter:
    """
    A kind of Variable that is automatically registered as a parameter when
    assigned as a Module attribute.

    Parameters are Variables that require gradients by default.
    This is a thin wrapper that creates a Variable with requires_grad=True.

    Example::

        import tenzor as tz

        class MyModule(tz.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = tz.nn.Parameter(tz.randn([10, 5]))
                self.bias = tz.nn.Parameter(tz.zeros([5]))

            def forward_impl(self, x):
                return x @ self.weight + self.bias
    """

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            return _core.Variable()
        if isinstance(data, _core.Tensor):
            return _core.Variable(data, requires_grad)
        if isinstance(data, _core.Variable):
            if requires_grad:
                data.requires_grad_(True)
            return data
        raise TypeError(f"Parameter data must be a Tensor, got {type(data)}")


# Override the nn module's Module and Sequential with our wrappers
_core.nn.Module = Module
_core.nn.Sequential = Sequential
_core.nn.Parameter = Parameter

# Expose gradient clipping utilities at nn level (matching PyTorch's torch.nn.utils)
clip_grad_norm_ = _core.nn.clip_grad_norm_
clip_grad_value_ = _core.nn.clip_grad_value_
