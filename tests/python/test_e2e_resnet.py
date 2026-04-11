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

    import tenzor.nn.functional as F
    optimizer = tz.optim.Adam(model.parameters(), lr=0.05)

    # Use a FIXED input/target across steps so the optimizer can actually
    # converge — previously the test regenerated random data every step,
    # which made loss-decrease expectations noise-dominated. Also go
    # through F.mse_loss on the Variable so the autograd graph stays
    # connected (wrapping a raw Tensor .mean() in a fresh Variable severs
    # the graph and leaves gradients at zero).
    x = tz.Variable(tz.randn([4, 8]), False)
    target = tz.Variable(tz.randn([4, 4]), False)

    losses = []
    for step in range(20):
        optimizer.zero_grad()
        output = model(x)
        assert output.shape == [4, 4], f"Step {step}: expected [4,4], got {output.shape}"
        loss = F.mse_loss(output, target)
        losses.append(float(loss.tensor().item()))
        loss.backward()
        optimizer.step()

    # Verify loss decreased overall (first -> last).
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

    import tenzor.nn.functional as F
    output = model(x)
    loss = F.mse_loss(output, target)
    loss.backward()

    # All parameters should have gradients. Variable.grad returns a
    # Tensor (not a Variable), so sum/item directly on the tensor.
    for param in model.parameters():
        grad = param.grad
        assert grad is not None, "All parameters should receive gradients"
        grad_sum = abs(float(grad.sum().item()))
        assert grad_sum > 0, "Gradient should be non-zero"


def test_resnet_eval_mode():
    """Verify model produces output in eval mode without gradient tracking."""
    tz.initialize()

    model = SmallResNet(in_features=4, hidden=8, num_classes=2)
    model.eval()

    x = tz.Variable(tz.randn([2, 4]), False)
    output = model(x)
    assert output.shape == [2, 2]


if __name__ == "__main__":
    test_resnet_training_loss_decreases()
    test_resnet_gradient_flow_through_skip()
    test_resnet_eval_mode()
    print("All ResNet e2e tests passed!")
