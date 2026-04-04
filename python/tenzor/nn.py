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

        def forward(self, x):
            x = self.fc1(x)
            x = tz.nn.relu(x)
            return self.fc2(x)
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Iterator, Optional

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

            def forward(self, x):
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
        # Lock protects lazy cache invalidation/population from races
        object.__setattr__(self, '_cache_lock', threading.Lock())
        # Lazy caches for submodule/parameter/buffer lookups.
        # Populated on first __getattr__ access, invalidated by __setattr__.
        object.__setattr__(self, '_submodule_cache', None)
        object.__setattr__(self, '_param_cache', None)
        object.__setattr__(self, '_buffer_cache', None)

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

        # Handle Module assignment — hold lock across both cache invalidation
        # and registration to prevent races with concurrent __getattr__
        if isinstance(value, _CppModule):
            with object.__getattribute__(self, '_cache_lock'):
                object.__setattr__(self, '_submodule_cache', None)
                self.register_module(name, value)
            return

        # Handle Variable assignment (potential parameter)
        if isinstance(value, _core.Variable):
            with object.__getattribute__(self, '_cache_lock'):
                if value.requires_grad():
                    object.__setattr__(self, '_param_cache', None)
                    self.register_parameter(name, value)
                else:
                    object.__setattr__(self, '_buffer_cache', None)
                    self.register_buffer(name, value)
            return

        # Warn if assigning a list/tuple containing Modules (should use ModuleList)
        if isinstance(value, (list, tuple)) and any(isinstance(v, _CppModule) for v in value):
            import warnings
            warnings.warn(
                f"Assigning a list of modules to '{name}' — "
                "use ModuleList for auto-registration",
                stacklevel=2,
            )

        # Regular attribute — no cache invalidation needed
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

    def __getattr__(self, name: str) -> Any:
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

        # Lazily populate and reuse caches to avoid repeated C++ boundary crossings.
        # Lock protects cache population against concurrent __setattr__ invalidation.
        lock = object.__getattribute__(self, '_cache_lock')

        with lock:
            submodule_cache = object.__getattribute__(self, '_submodule_cache')
            if submodule_cache is None:
                submodule_cache = self.get_submodules()
                object.__setattr__(self, '_submodule_cache', submodule_cache)
            if name in submodule_cache:
                return submodule_cache[name]

        with lock:
            param_cache = object.__getattribute__(self, '_param_cache')
            if param_cache is None:
                param_cache = self._get_own_named_params()
                object.__setattr__(self, '_param_cache', param_cache)
            if name in param_cache:
                return param_cache[name]

        with lock:
            buffer_cache = object.__getattribute__(self, '_buffer_cache')
            if buffer_cache is None:
                buffer_cache = self._get_own_named_buffers()
                object.__setattr__(self, '_buffer_cache', buffer_cache)
            if name in buffer_cache:
                return buffer_cache[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __delattr__(self, name: str) -> None:
        """Remove attribute from appropriate registry.

        Calls C++ unregister methods for parameters, buffers, and submodules.
        For regular Python attributes, delegates to object.__delattr__.
        """
        lock = object.__getattribute__(self, '_cache_lock')
        submodules = self.get_submodules()
        if name in submodules:
            with lock:
                self.unregister_module(name)
                self._param_cache = None
                self._buffer_cache = None
                self._submodule_cache = None
            return
        own_params = self._get_own_named_params()
        if name in own_params:
            with lock:
                self.unregister_parameter(name)
                self._param_cache = None
            return
        own_buffers = self._get_own_named_buffers()
        if name in own_buffers:
            with lock:
                self.unregister_buffer(name)
                self._buffer_cache = None
            return
        object.__delattr__(self, name)

    def __dir__(self) -> list[str]:
        """Combine Python attributes with C++ submodule/parameter/buffer names."""
        result = set(super().__dir__())
        result.update(self.get_submodules().keys())
        result.update(self._get_own_named_params().keys())
        result.update(self._get_own_named_buffers().keys())
        return sorted(result)

    def add_module(self, name: str, module: Optional[Module]) -> None:
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

    def children(self) -> Iterator[Any]:
        """Returns an iterator over immediate children modules."""
        for name, module in self.get_submodules().items():
            if module is not None:
                yield module

    def named_children(self) -> Iterator[tuple[str, Any]]:
        """Returns an iterator over immediate children modules, yielding tuples of (name, module)."""
        for name, module in self.get_submodules().items():
            if module is not None:
                yield name, module

    def modules(self) -> Iterator[Any]:
        """Returns an iterator over all modules in the network, including self."""
        yield self
        for name, module in self.get_submodules().items():
            if module is not None:
                if hasattr(module, 'modules'):
                    yield from module.modules()
                else:
                    yield module

    def named_modules(self, prefix: str = '') -> Iterator[tuple[str, Any]]:
        """Returns an iterator over all modules in the network, yielding tuples of (name, module)."""
        yield prefix, self
        for name, module in self.get_submodules().items():
            if module is not None:
                submodule_prefix = f"{prefix}.{name}" if prefix else name
                if hasattr(module, 'named_modules'):
                    yield from module.named_modules(submodule_prefix)
                else:
                    yield submodule_prefix, module

    def apply(self, fn: Callable[[Module], None]) -> Module:
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

    def requires_grad_(self, requires_grad: bool = True) -> Module:
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

    # ----------------------------------------------------------------
    # Hook API (methods are inherited from C++ via pybind11 bindings)
    # ----------------------------------------------------------------

    def register_forward_hook(self, hook: Callable) -> int:
        """Register a forward hook on the module.

        The hook is called every time after :meth:`forward` computes an output.

        Args:
            hook: Callable with signature ``hook(module, input, output) -> None or modified output``.

        Returns:
            Hook ID that can be passed to :meth:`remove_hook`.

        Example::

            def print_output(module, input, output):
                print(f"{module.__class__.__name__} output shape: {output.shape()}")

            hook_id = model.fc1.register_forward_hook(print_output)
        """
        return _CppModule.register_forward_hook(self, hook)

    def register_forward_pre_hook(self, hook: Callable) -> int:
        """Register a hook called before each forward call.

        Args:
            hook: Callable with signature ``hook(module, input) -> None or modified input``.

        Returns:
            Hook ID that can be passed to :meth:`remove_hook`.
        """
        return _CppModule.register_forward_pre_hook(self, hook)

    def register_backward_hook(self, hook: Callable) -> int:
        """Register a backward hook on the module.

        .. deprecated::
            Use :meth:`register_full_backward_hook` instead. This method is
            kept for backward compatibility but may be removed in a future release.

        Args:
            hook: Callable with signature ``hook(module, grad_input, grad_output)``.

        Returns:
            Hook ID that can be passed to :meth:`remove_hook`.
        """
        return _CppModule.register_backward_hook(self, hook)

    def register_full_backward_hook(self, hook: Callable) -> int:
        """Register a backward hook on the module.

        The hook is called every time the gradients w.r.t. the module
        inputs and outputs are computed.

        Args:
            hook: Callable with signature ``hook(module, grad_input, grad_output) -> None or modified grad_input``.

        Returns:
            Hook ID that can be passed to :meth:`remove_hook`.

        Example::

            def clip_grad(module, grad_input, grad_output):
                return tuple(g.clamp(-1, 1) if g is not None else g for g in grad_input)

            hook_id = model.fc1.register_full_backward_hook(clip_grad)
        """
        return _CppModule.register_full_backward_hook(self, hook)

    def register_full_backward_pre_hook(self, hook: Callable) -> int:
        """Register a hook called before the backward pass.

        Args:
            hook: Callable with signature ``hook(module, grad_output) -> None or modified grad_output``.

        Returns:
            Hook ID that can be passed to :meth:`remove_hook`.
        """
        return _CppModule.register_full_backward_pre_hook(self, hook)

    def remove_hook(self, hook_id: int) -> None:
        """Remove a previously registered hook.

        Args:
            hook_id: ID returned by one of the ``register_*_hook`` methods.
        """
        _CppModule.remove_hook(self, hook_id)

    def extra_repr(self) -> str:
        """Override in subclasses to show constructor args."""
        # Call C++ extra_repr (inherited from _CppModule)
        return _CppModule.extra_repr(self)

    def __repr__(self) -> str:
        """Return a string representation of the module."""
        extra = self.extra_repr()
        submodules = self.get_submodules()
        if not submodules:
            return f"{self.__class__.__name__}({extra})"
        lines = [f"{self.__class__.__name__}("]
        if extra:
            lines[0] = f"{self.__class__.__name__}({extra}"
            lines.append("")
        for name, module in submodules.items():
            mod_str = repr(module).replace('\n', '\n  ')
            lines.append(f"  ({name}): {mod_str}")
        lines.append(")")
        return '\n'.join(lines)


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

    def __init__(self, *args) -> None:
        """
        Initialize Sequential container.

        Args:
            *args: Variable number of modules to add in order
        """
        super().__init__()
        # _py_modules keeps Python references alive to prevent garbage collection.
        # The C++ Module base class stores raw pointers to child modules; without
        # this list, Python could GC the modules while C++ still references them.
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


class _ParameterMeta(type):
    """Metaclass that enables isinstance(var, Parameter) checks.

    Since Parameter is a factory (returns Variable, not a Parameter instance),
    we override __instancecheck__ to look for the _is_parameter tag set during
    construction.
    """
    def __instancecheck__(cls, instance):
        return getattr(instance, '_is_parameter', False)


class Parameter(metaclass=_ParameterMeta):
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

            def forward(self, x):
                return x @ self.weight + self.bias
    """

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            var = _core.Variable()
        elif isinstance(data, _core.Tensor):
            var = _core.Variable(data, requires_grad)
        elif isinstance(data, _core.Variable):
            if requires_grad:
                data.requires_grad_(True)
            var = data
        else:
            raise TypeError(f"Parameter data must be a Tensor, got {type(data)}")
        var._is_parameter = True
        return var


# Override the nn module's Module and Sequential with our wrappers
_core.nn.Module = Module
_core.nn.Sequential = Sequential
_core.nn.Parameter = Parameter

# Expose gradient clipping utilities at nn level (matching PyTorch's torch.nn.utils)
clip_grad_norm_ = _core.nn.clip_grad_norm_
clip_grad_value_ = _core.nn.clip_grad_value_


# ---------------------------------------------------------------------------
# PackedSequence and RNN utilities
# ---------------------------------------------------------------------------

class PackedSequence:
    """Holds packed variable-length sequences for efficient RNN processing.

    A ``PackedSequence`` stores a batch of variable-length sequences in a
    compact form that avoids wasted computation on padding tokens.  The data
    is sorted by decreasing sequence length and concatenated along the time
    axis, with a ``batch_sizes`` tensor recording how many sequences are
    active at each timestep.

    You normally do not construct ``PackedSequence`` directly.  Instead use
    :func:`pack_padded_sequence` or :func:`pack_sequence`.

    Attributes:
        data (Tensor): Packed tensor of shape ``(total_elements, *features)``.
        batch_sizes (Tensor): 1-D int64 tensor of length ``max_seq_len``
            giving the batch size at each timestep.
        sorted_indices (Tensor or None): Int64 tensor mapping original batch
            indices to their sorted positions.  ``None`` when the input was
            already sorted.
        unsorted_indices (Tensor or None): Int64 tensor that restores the
            original batch order.  ``None`` when the input was already sorted.
    """

    def __init__(self, data, batch_sizes, sorted_indices=None, unsorted_indices=None):
        self.data = data
        self.batch_sizes = batch_sizes
        self.sorted_indices = sorted_indices
        self.unsorted_indices = unsorted_indices

    @classmethod
    def _from_cpp(cls, cpp_packed):
        """Wrap a C++ PackedSequence returned by the backend."""
        obj = cls.__new__(cls)
        obj.data = cpp_packed.data
        obj.batch_sizes = cpp_packed.batch_sizes
        obj.sorted_indices = cpp_packed.sorted_indices
        obj.unsorted_indices = cpp_packed.unsorted_indices
        return obj

    def _to_cpp(self):
        """Convert to C++ PackedSequence for passing into backend functions."""
        cpp = _core.nn.PackedSequence()
        cpp.data = self.data
        cpp.batch_sizes = self.batch_sizes
        if self.sorted_indices is not None:
            cpp.sorted_indices = self.sorted_indices
        if self.unsorted_indices is not None:
            cpp.unsorted_indices = self.unsorted_indices
        return cpp

    def __repr__(self):
        return (
            f"PackedSequence(data={self.data}, "
            f"batch_sizes={self.batch_sizes}, "
            f"sorted_indices={self.sorted_indices}, "
            f"unsorted_indices={self.unsorted_indices})"
        )


def pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True):
    """Pack a padded batch of variable-length sequences.

    Takes a padded tensor and a tensor of sequence lengths, and returns a
    :class:`PackedSequence` suitable for efficient RNN processing.

    Parameters
    ----------
    input : Tensor
        Padded input of shape ``(batch, seq_len, *)`` if *batch_first* is
        ``True``, or ``(seq_len, batch, *)`` otherwise.
    lengths : Tensor
        1-D int64 tensor of actual lengths for each sequence in the batch.
    batch_first : bool, optional
        If ``True``, *input* is expected in ``(batch, seq_len, *)`` layout.
        Default: ``False``.
    enforce_sorted : bool, optional
        If ``True`` (default), the sequences must already be sorted by
        length in descending order.  If ``False`` the function will sort
        them internally and record the permutation so that
        :func:`pad_packed_sequence` can restore the original order.

    Returns
    -------
    PackedSequence
        The packed representation of the input batch.

    Example
    -------
    >>> padded = tz.randn([3, 5, 10])         # 3 seqs, max len 5, 10 feats
    >>> lengths = tz.tensor([5, 3, 1])
    >>> packed = tz.nn.pack_padded_sequence(padded, lengths, batch_first=True)
    """
    cpp_packed = _core.nn.pack_padded_sequence(input, lengths, batch_first, enforce_sorted)
    return PackedSequence._from_cpp(cpp_packed)


def pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):
    """Unpack a :class:`PackedSequence` back to a padded tensor.

    This is the inverse of :func:`pack_padded_sequence`.

    Parameters
    ----------
    sequence : PackedSequence
        The packed sequence to unpad.
    batch_first : bool, optional
        If ``True``, the returned tensor has shape ``(batch, seq_len, *)``.
        Default: ``False``.
    padding_value : float, optional
        Value used for padding positions.  Default: ``0.0``.
    total_length : int or None, optional
        If not ``None``, the output will be padded to this length (must be
        >= the longest sequence).  Useful for ``DataParallel`` where all
        workers need identical shapes.  Default: ``None``.

    Returns
    -------
    tuple[Tensor, Tensor]
        A pair ``(padded_output, lengths)`` where *padded_output* is the
        padded tensor and *lengths* is a 1-D int64 tensor of actual
        sequence lengths (in the original, unsorted order if the input was
        sorted internally by :func:`pack_padded_sequence`).

    Example
    -------
    >>> padded, lengths = tz.nn.pad_packed_sequence(packed, batch_first=True)
    """
    if isinstance(sequence, PackedSequence):
        cpp_packed = sequence._to_cpp()
    else:
        cpp_packed = sequence  # Already a C++ PackedSequence
    tl = total_length if total_length is not None else -1
    return _core.nn.pad_packed_sequence(cpp_packed, batch_first, padding_value, tl)


def pack_sequence(sequences, enforce_sorted=True):
    """Pack a list of variable-length tensors into a :class:`PackedSequence`.

    Tensors are sorted by length (descending) and packed.

    Parameters
    ----------
    sequences : list[Tensor]
        A list of tensors, each of shape ``(length_i, *features)``.
        They must share the same trailing dimensions (feature sizes).
    enforce_sorted : bool, optional
        If ``True`` (default), the tensors must already be sorted by
        length descending.  If ``False``, they will be sorted internally.

    Returns
    -------
    PackedSequence
        The packed representation.

    Example
    -------
    >>> seqs = [tz.randn([5, 10]), tz.randn([3, 10]), tz.randn([1, 10])]
    >>> packed = tz.nn.pack_sequence(seqs)
    """
    cpp_packed = _core.nn.pack_sequence(sequences, enforce_sorted)
    return PackedSequence._from_cpp(cpp_packed)


# Make init submodule accessible
from .tenzor_core.nn import init
