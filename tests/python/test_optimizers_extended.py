"""
Tests for optimizer Python bindings that currently have zero coverage.

Covers: RMSprop, Adagrad, Adadelta, RAdam, LAMB, SparseAdam, AdamAtan2.
"""

import pytest
import tenzor as tz


def make_simple_model():
    """Create a simple linear model for optimizer testing."""
    return tz.nn.Linear(4, 2)


def train_step(model, optimizer, device="cpu"):
    """Do a single forward + backward + step."""
    x = tz.Variable(tz.randn([2, 4], device=device), False)
    y = model(x)
    loss_tensor = tz.sum(y.data)
    loss = tz.Variable(loss_tensor, True)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


class TestOptimizers:
    def test_rmsprop(self):
        model = make_simple_model()
        optimizer = tz.optim.RMSprop(model.parameters(), lr=0.01)
        train_step(model, optimizer)

    def test_adagrad(self):
        model = make_simple_model()
        optimizer = tz.optim.Adagrad(model.parameters(), lr=0.01)
        train_step(model, optimizer)

    def test_adadelta(self):
        model = make_simple_model()
        optimizer = tz.optim.Adadelta(model.parameters())
        train_step(model, optimizer)

    def test_radam(self):
        model = make_simple_model()
        optimizer = tz.optim.RAdam(model.parameters(), lr=0.001)
        train_step(model, optimizer)

    def test_lamb(self):
        model = make_simple_model()
        optimizer = tz.optim.LAMB(model.parameters(), lr=0.001)
        train_step(model, optimizer)

    def test_sparse_adam(self):
        model = make_simple_model()
        optimizer = tz.optim.SparseAdam(model.parameters(), lr=0.001)
        train_step(model, optimizer)

    def test_adam_atan2(self):
        model = make_simple_model()
        optimizer = tz.optim.AdamAtan2(model.parameters(), lr=0.001)
        train_step(model, optimizer)

    def test_multiple_steps(self):
        """Verify optimizer can run multiple steps without error."""
        model = make_simple_model()
        optimizer = tz.optim.RAdam(model.parameters(), lr=0.001)
        for _ in range(5):
            train_step(model, optimizer)
