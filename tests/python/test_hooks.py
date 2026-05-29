"""
Tests for module forward and backward hooks.

Tests hook registration, execution, gradient capture/modification,
and removal.
"""

import sys
import os

build_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python')
if os.path.exists(build_dir):
    sys.path.insert(0, os.path.abspath(build_dir))

import tenzor as tz


def _init():
    tz.initialize()


def test_forward_hook_called():
    """Forward hook should be called during forward pass."""
    _init()
    model = tz.nn.Linear(4, 2)
    model.train()

    hook_called = [False]

    def forward_hook(module, input, output):
        hook_called[0] = True

    handle = model.register_forward_hook(forward_hook)

    x = tz.Variable(tz.randn([2, 4]), False)
    _ = model(x)

    assert hook_called[0], "Forward hook should have been called"

    handle.remove()


def test_forward_pre_hook_called():
    """Forward pre-hook should be called before forward pass."""
    _init()
    model = tz.nn.Linear(4, 2)

    pre_hook_called = [False]

    def pre_hook(module, input):
        pre_hook_called[0] = True

    handle = model.register_forward_pre_hook(pre_hook)

    x = tz.Variable(tz.randn([2, 4]), False)
    _ = model(x)

    assert pre_hook_called[0], "Forward pre-hook should have been called"

    handle.remove()


def test_hook_removal():
    """After removal, hook should not be called."""
    _init()
    model = tz.nn.Linear(4, 2)

    call_count = [0]

    def counting_hook(module, input, output):
        call_count[0] += 1

    handle = model.register_forward_hook(counting_hook)

    x = tz.Variable(tz.randn([2, 4]), False)
    _ = model(x)
    assert call_count[0] == 1

    handle.remove()

    _ = model(x)
    assert call_count[0] == 1, "Hook should not be called after removal"


def test_multiple_hooks():
    """Multiple hooks should all be called."""
    _init()
    model = tz.nn.Linear(4, 2)

    calls = []

    def hook_a(module, input, output):
        calls.append('a')

    def hook_b(module, input, output):
        calls.append('b')

    handle_a = model.register_forward_hook(hook_a)
    handle_b = model.register_forward_hook(hook_b)

    x = tz.Variable(tz.randn([2, 4]), False)
    _ = model(x)

    assert 'a' in calls and 'b' in calls, f"Both hooks should be called, got {calls}"

    handle_a.remove()
    handle_b.remove()


def test_register_returns_removable_handle():
    """register_*_hook returns a RemovableHandle exposing .remove()."""
    _init()
    model = tz.nn.Linear(4, 2)
    handle = model.register_forward_hook(lambda m, i, o: None)

    assert type(handle).__name__ == "RemovableHandle"
    assert hasattr(handle, "remove")
    handle.remove()


def test_handle_remove_detaches_hook():
    _init()
    model = tz.nn.Linear(4, 2)
    calls = [0]
    handle = model.register_forward_hook(lambda m, i, o: calls.__setitem__(0, calls[0] + 1))

    x = tz.Variable(tz.randn([2, 4]), False)
    _ = model(x)
    assert calls[0] == 1

    handle.remove()
    _ = model(x)
    assert calls[0] == 1, "hook should not fire after handle.remove()"


def test_handle_remove_is_idempotent():
    _init()
    model = tz.nn.Linear(4, 2)
    handle = model.register_forward_hook(lambda m, i, o: None)
    handle.remove()
    # Second call must not raise
    handle.remove()


def test_handle_survives_module_outliving():
    """The handle should not keep its referenced module alive; it uses a
    weak reference internally. Once the module is dropped, .remove()
    is a silent no-op rather than a crash."""
    _init()
    import weakref
    model = tz.nn.Linear(4, 2)
    handle = model.register_forward_hook(lambda m, i, o: None)
    ref = weakref.ref(model)
    del model
    # Module may or may not be collected immediately — pybind11 holds a
    # strong ref until the Python-side ref drops. Either way, calling
    # handle.remove() must not crash.
    handle.remove()


def test_full_backward_hook_registration():
    """register_full_backward_hook also returns a RemovableHandle."""
    _init()
    model = tz.nn.Linear(4, 2)
    handle = model.register_full_backward_hook(lambda m, gi, go: None)
    assert type(handle).__name__ == "RemovableHandle"
    handle.remove()


def test_inline_lambda_hook_fires():
    """An inline lambda (no external reference) must still fire.

    Previously the hook registry weakref'd every callable, so a lambda with
    no other strong reference was GC'd before the forward pass and silently
    never fired. Lambdas/functions don't implicitly reference the module, so
    they are now held strongly (PyTorch _forward_hooks semantics).
    """
    _init()
    model = tz.nn.Linear(4, 2)
    calls = [0]
    model.register_forward_hook(lambda m, i, o: calls.__setitem__(0, calls[0] + 1))
    _ = model(tz.Variable(tz.randn([2, 4]), False))
    assert calls[0] == 1, "inline lambda hook must fire"


def test_free_function_hook_fires_without_external_ref():
    """A module-level function passed without keeping a reference must fire."""
    _init()
    model = tz.nn.Linear(4, 2)
    seen = []

    def make_and_register():
        # The local `fn` goes out of scope after this returns; the registry
        # must keep it alive (strong ref) for it to fire later.
        def fn(m, i, o):
            seen.append(1)
        model.register_forward_hook(fn)

    make_and_register()
    _ = model(tz.Variable(tz.randn([2, 4]), False))
    assert seen == [1], "free-function hook must fire after its local scope exits"


def test_bound_method_hook_fires_while_module_alive():
    """Bound-method hooks (cycle-prone) still fire while the owner is alive.

    Bound methods are wrapped in weakref.WeakMethod to break the
    owner<->hook reference cycle, but must resolve and fire as long as the
    owning object lives.
    """
    _init()
    model = tz.nn.Linear(4, 2)

    class Recorder:
        def __init__(self):
            self.count = 0

        def on_forward(self, m, i, o):
            self.count += 1

    rec = Recorder()                       # kept alive in this scope
    model.register_forward_hook(rec.on_forward)  # bound method -> WeakMethod
    _ = model(tz.Variable(tz.randn([2, 4]), False))
    assert rec.count == 1, "bound-method hook must fire while owner is alive"


def test_subclass_register_hook_returns_usable_handle():
    """Registering a hook on a Python Module subclass must return a single
    RemovableHandle (the Python wrapper used to double-wrap the C++ handle,
    raising TypeError) and the hook must fire and be removable."""
    _init()

    class M(tz.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = tz.nn.Linear(4, 2)
            self.count = 0
            self.handle = self.register_forward_hook(self._on)

        def _on(self, m, i, o):
            self.count += 1

        def forward(self, x):
            return self.lin(x)

    m = M()
    assert type(m.handle).__name__ == "RemovableHandle"
    _ = m(tz.Variable(tz.randn([2, 4]), False))
    assert m.count == 1, "subclass-registered hook must fire"
    m.handle.remove()
    _ = m(tz.Variable(tz.randn([2, 4]), False))
    assert m.count == 1, "hook must not fire after remove()"


if __name__ == "__main__":
    test_forward_hook_called()
    test_forward_pre_hook_called()
    test_hook_removal()
    test_multiple_hooks()
    test_register_returns_removable_handle()
    test_handle_remove_detaches_hook()
    test_handle_remove_is_idempotent()
    test_handle_survives_module_outliving()
    test_full_backward_hook_registration()
    test_inline_lambda_hook_fires()
    test_free_function_hook_fires_without_external_ref()
    test_bound_method_hook_fires_while_module_alive()
    test_subclass_register_hook_returns_usable_handle()
    print("All hook tests passed!")
