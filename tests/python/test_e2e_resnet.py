"""
End-to-end ResNet-like model training test.

Verifies that a small residual network can:
1. Forward pass through skip connections
2. Compute loss
3. Backpropagate through residual paths
4. Update weights with optimizer
5. Show decreasing loss over multiple steps
"""

import sys
import os

build_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'python')
if os.path.exists(build_dir):
    sys.path.insert(0, os.path.abspath(build_dir))

import tenzor as tz


class ResidualBlock(tz.nn.Module):
    """Simple residual block: x + F(x)."""

    def __init__(self, features):
        super().__init__()
        self.fc1 = tz.nn.Linear(features, features)
        self.fc2 = tz.nn.Linear(features, features)

    def forward(self, x):
        residual = x
        out = tz.nn.relu(self.fc1(x))
        out = self.fc2(out)
        return tz.nn.relu(out + residual)


class SmallResNet(tz.nn.Module):
    """Tiny ResNet for testing: Linear -> ResBlock -> ResBlock -> Linear."""

    def __init__(self, in_features, hidden, num_classes):
        super().__init__()
        self.input_proj = tz.nn.Linear(in_features, hidden)
        self.block1 = ResidualBlock(hidden)
        self.block2 = ResidualBlock(hidden)
        self.classifier = tz.nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = tz.nn.relu(self.input_proj(x))
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


def test_resnet_training_loss_decreases():
    """Train a small ResNet for 5 steps and verify loss decreases."""
    tz.initialize()

    model = SmallResNet(in_features=8, hidden=16, num_classes=4)
    model.train()

    optimizer = tz.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for step in range(5):
        optimizer.zero_grad()

        # Synthetic data
        x = tz.Variable(tz.randn([4, 8]), True)
        target = tz.Variable(tz.randn([4, 4]), False)

        output = model(x)
        assert output.shape() == [4, 4], f"Step {step}: expected [4,4], got {output.shape()}"

        diff = output - target
        loss = (diff * diff).tensor().mean()
        loss_val = float(loss.item())
        losses.append(loss_val)

        loss_var = tz.Variable(loss, True)
        loss_var.backward()
        optimizer.step()

    # Verify loss decreased overall (first -> last)
    assert losses[-1] < losses[0], (
        f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


def test_resnet_gradient_flow_through_skip():
    """Verify gradients flow through the skip connection."""
    tz.initialize()

    model = SmallResNet(in_features=4, hidden=8, num_classes=2)
    model.train()

    x = tz.Variable(tz.randn([2, 4]), True)
    target = tz.Variable(tz.randn([2, 2]), False)

    output = model(x)
    diff = output - target
    loss = (diff * diff).tensor().mean()
    loss_var = tz.Variable(loss, True)
    loss_var.backward()

    # All parameters should have gradients
    for param in model.parameters():
        grad = param.grad()
        assert grad is not None, "All parameters should receive gradients"
        # Gradient should not be all zeros
        grad_sum = abs(float(grad.tensor().sum().item()))
        assert grad_sum > 0, "Gradient should be non-zero"


def test_resnet_eval_mode():
    """Verify model produces output in eval mode without gradient tracking."""
    tz.initialize()

    model = SmallResNet(in_features=4, hidden=8, num_classes=2)
    model.eval()

    x = tz.Variable(tz.randn([2, 4]), False)
    output = model(x)
    assert output.shape() == [2, 2]


if __name__ == "__main__":
    test_resnet_training_loss_decreases()
    test_resnet_gradient_flow_through_skip()
    test_resnet_eval_mode()
    print("All ResNet e2e tests passed!")
