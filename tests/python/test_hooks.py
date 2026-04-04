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

    hook_id = model.register_forward_hook(forward_hook)

    x = tz.Variable(tz.randn([2, 4]), False)
    _ = model(x)

    assert hook_called[0], "Forward hook should have been called"

    model.remove_hook(hook_id)


def test_forward_pre_hook_called():
    """Forward pre-hook should be called before forward pass."""
    _init()
    model = tz.nn.Linear(4, 2)

    pre_hook_called = [False]

    def pre_hook(module, input):
        pre_hook_called[0] = True

    hook_id = model.register_forward_pre_hook(pre_hook)

    x = tz.Variable(tz.randn([2, 4]), False)
    _ = model(x)

    assert pre_hook_called[0], "Forward pre-hook should have been called"

    model.remove_hook(hook_id)


def test_hook_removal():
    """After removal, hook should not be called."""
    _init()
    model = tz.nn.Linear(4, 2)

    call_count = [0]

    def counting_hook(module, input, output):
        call_count[0] += 1

    hook_id = model.register_forward_hook(counting_hook)

    x = tz.Variable(tz.randn([2, 4]), False)
    _ = model(x)
    assert call_count[0] == 1

    model.remove_hook(hook_id)

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

    id_a = model.register_forward_hook(hook_a)
    id_b = model.register_forward_hook(hook_b)

    x = tz.Variable(tz.randn([2, 4]), False)
    _ = model(x)

    assert 'a' in calls and 'b' in calls, f"Both hooks should be called, got {calls}"

    model.remove_hook(id_a)
    model.remove_hook(id_b)


if __name__ == "__main__":
    test_forward_hook_called()
    test_forward_pre_hook_called()
    test_hook_removal()
    test_multiple_hooks()
    print("All hook tests passed!")
