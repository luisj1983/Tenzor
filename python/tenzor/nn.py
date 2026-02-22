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
        # Track registered modules/parameters/buffers for attribute access
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, '_parameters', {})
        object.__setattr__(self, '_buffers', {})

    def __setattr__(self, name: str, value) -> None:
        """
        Automatically register Modules, Variables as submodules/parameters.

        This enables PyTorch-like syntax where assigning a Module attribute
        automatically registers it.
        """
        # Check if it's an internal attribute (starts with _)
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return

        # Handle Module assignment
        if isinstance(value, _CppModule):
            # Remove from other registries if present
            if hasattr(self, '_parameters') and name in self._parameters:
                del self._parameters[name]
            if hasattr(self, '_buffers') and name in self._buffers:
                del self._buffers[name]

            # Register the module with C++ backend
            self.register_module(name, value)

            # Also store in Python dict for attribute access
            if hasattr(self, '_modules'):
                self._modules[name] = value

            return

        # Handle Variable assignment (potential parameter)
        if isinstance(value, _core.Variable):
            # Check if requires_grad - if so, it's a parameter
            if value.requires_grad():
                if hasattr(self, '_modules') and name in self._modules:
                    del self._modules[name]
                if hasattr(self, '_buffers') and name in self._buffers:
                    del self._buffers[name]

                self.register_parameter(name, value)
                if hasattr(self, '_parameters'):
                    self._parameters[name] = value
            else:
                # It's a buffer (non-trainable)
                if hasattr(self, '_modules') and name in self._modules:
                    del self._modules[name]
                if hasattr(self, '_parameters') and name in self._parameters:
                    del self._parameters[name]

                self.register_buffer(name, value)
                if hasattr(self, '_buffers'):
                    self._buffers[name] = value

            return

        # Regular attribute
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        """
        Get attribute, checking modules/parameters/buffers dicts first.
        """
        if name.startswith('_'):
            return object.__getattribute__(self, name)

        # Check in modules
        _modules = object.__getattribute__(self, '_modules')
        if name in _modules:
            return _modules[name]

        # Check in parameters
        _parameters = object.__getattribute__(self, '_parameters')
        if name in _parameters:
            return _parameters[name]

        # Check in buffers
        _buffers = object.__getattribute__(self, '_buffers')
        if name in _buffers:
            return _buffers[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __delattr__(self, name: str) -> None:
        """Remove attribute from appropriate registry."""
        if name in self._modules:
            del self._modules[name]
        elif name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        else:
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
            self._modules[name] = None
            return

        if not isinstance(module, _CppModule):
            raise TypeError(f"module must be a Module, got {type(module)}")

        self.register_module(name, module)
        self._modules[name] = module

    def children(self):
        """Returns an iterator over immediate children modules."""
        for name, module in self._modules.items():
            if module is not None:
                yield module

    def named_children(self):
        """Returns an iterator over immediate children modules, yielding tuples of (name, module)."""
        for name, module in self._modules.items():
            if module is not None:
                yield name, module

    def modules(self):
        """Returns an iterator over all modules in the network, including self."""
        yield self
        for name, module in self._modules.items():
            if module is not None:
                if isinstance(module, Module):
                    yield from module.modules()
                else:
                    yield module

    def named_modules(self, prefix: str = ''):
        """Returns an iterator over all modules in the network, yielding tuples of (name, module)."""
        yield prefix, self
        for name, module in self._modules.items():
            if module is not None:
                submodule_prefix = f"{prefix}.{name}" if prefix else name
                if isinstance(module, Module):
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
            if isinstance(module, Module):
                module.apply(fn)
            fn(module)
        fn(self)
        return self

    def train(self, mode: bool = True):
        """
        Set the module in training mode.

        This has effect on certain modules like Dropout and BatchNorm.

        Args:
            mode: Whether to set training mode (True) or eval mode (False)

        Returns:
            self
        """
        object.__setattr__(self, '_training', mode)
        for module in self.children():
            if isinstance(module, Module):
                module.train(mode)
        return self

    def eval(self):
        """
        Set the module in evaluation mode.

        Equivalent to self.train(False).

        Returns:
            self
        """
        return self.train(False)

    @property
    def training(self) -> bool:
        """Whether the module is in training mode."""
        try:
            return object.__getattribute__(self, '_training')
        except AttributeError:
            return True  # Default to training mode

    def parameters(self, recurse: bool = True):
        """
        Return an iterator over module parameters.

        Args:
            recurse: If True, yields parameters of this module and all submodules.
                     If False, yields only parameters that are direct members of this module.

        Yields:
            Variable: Module parameter
        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """
        Return an iterator over module parameters, yielding both the name and the parameter.

        Args:
            prefix: Prefix to prepend to parameter names
            recurse: If True, yields parameters of this module and all submodules

        Yields:
            Tuple[str, Variable]: Tuple of (name, parameter)
        """
        # Yield own parameters
        for name, param in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, param

        # Recurse into submodules
        if recurse:
            for mod_name, module in self._modules.items():
                if module is not None and isinstance(module, Module):
                    submodule_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
                    yield from module.named_parameters(prefix=submodule_prefix, recurse=True)

    def state_dict(self, prefix: str = ''):
        """
        Return a dictionary containing the whole state of the module.

        Both parameters and buffers are included. Keys are the parameter/buffer names.

        Args:
            prefix: Prefix to prepend to keys

        Returns:
            dict: A dictionary containing the module state
        """
        result = {}
        for name, param in self._parameters.items():
            key = f"{prefix}{name}"
            result[key] = param.tensor() if hasattr(param, 'tensor') else param
        for name, buf in self._buffers.items():
            key = f"{prefix}{name}"
            result[key] = buf.tensor() if hasattr(buf, 'tensor') else buf
        for name, module in self._modules.items():
            if module is not None and isinstance(module, Module):
                child_prefix = f"{prefix}{name}."
                result.update(module.state_dict(prefix=child_prefix))
        return result

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """
        Load parameters and buffers from a state_dict.

        Args:
            state_dict: A dict containing parameters and buffers
            strict: If True, keys in state_dict must exactly match this module's state_dict keys

        Returns:
            None
        """
        own_state = self.state_dict()
        missing_keys = []
        unexpected_keys = []

        for key in own_state:
            if key not in state_dict:
                missing_keys.append(key)

        for key in state_dict:
            if key not in own_state:
                unexpected_keys.append(key)

        if strict and (missing_keys or unexpected_keys):
            error_msg = ''
            if missing_keys:
                error_msg += f'Missing keys: {missing_keys}. '
            if unexpected_keys:
                error_msg += f'Unexpected keys: {unexpected_keys}.'
            raise RuntimeError(error_msg)

        # Apply the state dict values
        for name, param in self._parameters.items():
            if name in state_dict:
                # Copy data from state_dict tensor into existing parameter
                param.tensor().copy_(state_dict[name])

        for name, buf in self._buffers.items():
            if name in state_dict:
                buf.tensor().copy_(state_dict[name])

        for name, module in self._modules.items():
            if module is not None and isinstance(module, Module):
                child_prefix = f"{name}."
                child_state = {k[len(child_prefix):]: v for k, v in state_dict.items()
                              if k.startswith(child_prefix)}
                if child_state:
                    module.load_state_dict(child_state, strict=strict)

    def to(self, *args):
        """
        Move and/or cast the parameters and buffers.

        Can be called as:
            - module.to(device) — moves all parameters/buffers to the device
            - module.to(dtype) — casts all parameters/buffers to the dtype

        Args:
            *args: Device or dtype to move/cast to

        Returns:
            self
        """
        for name, param in self._parameters.items():
            if param is not None:
                new_param = param.to(*args) if hasattr(param, 'to') else param
                self._parameters[name] = new_param

        for name, buf in self._buffers.items():
            if buf is not None:
                new_buf = buf.to(*args) if hasattr(buf, 'to') else buf
                self._buffers[name] = new_buf

        for module in self.children():
            if isinstance(module, Module):
                module.to(*args)

        return self

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
        for name, module in self._modules.items():
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


# Override the nn module's Module and Sequential with our wrappers
_core.nn.Module = Module
_core.nn.Sequential = Sequential
