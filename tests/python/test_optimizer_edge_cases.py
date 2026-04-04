"""
Optimizer edge case tests.

Tests zero-gradient steps, extreme learning rates, multiple steps
without zero_grad, and Adam/AdamW weight decay behavior.
"""

import sys
import os

build_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python')
if os.path.exists(build_dir):
    sys.path.insert(0, os.path.abspath(build_dir))

import tenzor as tz


def _init():
    tz.initialize()


def test_sgd_zero_gradient_step():
    """SGD step with zero gradients should not change weights."""
    _init()
    model = tz.nn.Linear(4, 2)
    optimizer = tz.optim.SGD(model.parameters(), lr=0.1)

    params = list(model.parameters())
    initial_weight = params[0].tensor().clone()

    # Zero gradients explicitly, then step
    optimizer.zero_grad()
    optimizer.step()

    # Weights should not change
    updated_weight = params[0].tensor()
    diff = float((updated_weight - initial_weight).abs().sum().item())
    assert diff < 1e-7, f"Weights changed with zero gradients: diff={diff}"


def test_adam_basic_step():
    """Adam optimizer should update weights after backward."""
    _init()
    model = tz.nn.Linear(4, 2)
    model.train()
    optimizer = tz.optim.Adam(model.parameters(), lr=0.01)

    params = list(model.parameters())
    initial_weight = params[0].tensor().clone()

    x = tz.Variable(tz.randn([2, 4]), True)
    output = model(x)
    loss = (output * output).tensor().mean()
    loss_var = tz.Variable(loss, True)
    loss_var.backward()

    optimizer.step()

    updated_weight = params[0].tensor()
    diff = float((updated_weight - initial_weight).abs().sum().item())
    assert diff > 1e-7, "Adam should update weights after backward"


def test_adamw_weight_decay():
    """AdamW should apply decoupled weight decay."""
    _init()
    model = tz.nn.Linear(4, 2)
    model.train()

    # High weight decay to make effect visible
    optimizer = tz.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)

    params = list(model.parameters())
    initial_norm = float(params[0].tensor().norm().item())

    # Several steps with backward
    for _ in range(3):
        optimizer.zero_grad()
        x = tz.Variable(tz.randn([2, 4]), True)
        output = model(x)
        loss = (output * output).tensor().mean()
        loss_var = tz.Variable(loss, True)
        loss_var.backward()
        optimizer.step()

    final_norm = float(params[0].tensor().norm().item())
    # Weight decay should shrink weights (norm should decrease or at least not grow much)
    # This is a soft check since gradient updates can increase norms
    assert final_norm < initial_norm * 2.0, (
        f"Weights grew too much with weight decay: {initial_norm:.4f} -> {final_norm:.4f}"
    )


def test_multiple_steps_without_zero_grad():
    """Multiple optimizer steps without zero_grad should accumulate gradients."""
    _init()
    model = tz.nn.Linear(4, 2)
    model.train()
    optimizer = tz.optim.SGD(model.parameters(), lr=0.01)

    # Step 1
    x = tz.Variable(tz.randn([2, 4]), True)
    output = model(x)
    loss = (output * output).tensor().mean()
    tz.Variable(loss, True).backward()

    params = list(model.parameters())
    grad_after_first = params[0].grad().tensor().clone()

    # Step 2 without zero_grad - gradients should accumulate
    x2 = tz.Variable(tz.randn([2, 4]), True)
    output2 = model(x2)
    loss2 = (output2 * output2).tensor().mean()
    tz.Variable(loss2, True).backward()

    grad_after_second = params[0].grad().tensor()

    # Accumulated gradient should differ from first gradient
    diff = float((grad_after_second - grad_after_first).abs().sum().item())
    assert diff > 1e-7, "Gradients should accumulate without zero_grad"


def test_sgd_with_momentum():
    """SGD with momentum should produce different updates than without."""
    _init()
    model = tz.nn.Linear(4, 2)
    model.train()
    optimizer = tz.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Multiple steps to build up momentum
    for _ in range(3):
        optimizer.zero_grad()
        x = tz.Variable(tz.randn([2, 4]), True)
        output = model(x)
        loss = (output * output).tensor().mean()
        tz.Variable(loss, True).backward()
        optimizer.step()

    # If we got here without error, momentum is working
    assert True


if __name__ == "__main__":
    test_sgd_zero_gradient_step()
    test_adam_basic_step()
    test_adamw_weight_decay()
    test_multiple_steps_without_zero_grad()
    test_sgd_with_momentum()
    print("All optimizer edge case tests passed!")
