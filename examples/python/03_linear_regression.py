"""
Tenzor Tutorial 03: Linear Regression with SGD
==============================================
Learn how to train a simple linear regression model using Tenzor.
"""

import tenzor as tz
import numpy as np

def main():
    print("=" * 70)
    print("TENZOR TUTORIAL 03: LINEAR REGRESSION WITH SGD")
    print("=" * 70)

    # Initialize Tenzor
    print("\n[Setup] Initializing Tenzor library...")
    tz.initialize()
    print("✓ Tenzor initialized successfully")

    # ========================================================================
    # SECTION 1: Generate Synthetic Data
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 1: Generating Synthetic Training Data")
    print("=" * 70)

    print("\n[1.1] Creating linear relationship: y = 3x + 2 + noise")

    # Generate training data
    np.random.seed(42)
    n_samples = 100

    # True parameters
    true_w = 3.0
    true_b = 2.0

    # Generate x values
    x_np = np.random.randn(n_samples, 1).astype(np.float32)
    # Generate y with noise: y = wx + b + noise
    noise = np.random.randn(n_samples, 1).astype(np.float32) * 0.5
    y_np = true_w * x_np + true_b + noise

    print(f"  Generated {n_samples} training samples")
    print(f"  True parameters: w={true_w}, b={true_b}")
    print(f"  Input shape: {x_np.shape}")
    print(f"  Target shape: {y_np.shape}")

    # Convert to Tenzor tensors (zero-copy where possible)
    print("\n[1.2] Converting to Tenzor tensors...")
    x_data = tz.from_numpy(x_np)
    y_data = tz.from_numpy(y_np)
    print("  ✓ Data prepared")

    # ========================================================================
    # SECTION 2: Define Model
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 2: Defining Linear Model")
    print("=" * 70)

    print("\n[2.1] Model: y = wx + b")
    print("  Initializing parameters randomly...")

    # Initialize parameters with small random values
    w = tz.Variable(tz.randn([1, 1]) * 0.01, requires_grad=True)
    b = tz.Variable(tz.zeros([1]), requires_grad=True)

    print("  ✓ Model parameters initialized")
    print(f"    - Weight (w): shape {w.data.shape}")
    print(f"    - Bias (b): shape {b.data.shape}")

    # ========================================================================
    # SECTION 3: Define Loss Function
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 3: Mean Squared Error Loss")
    print("=" * 70)

    print("\n[3.1] Loss function: MSE = mean((y_pred - y_true)²)")
    print("  This measures how far our predictions are from true values")

    def mse_loss(y_pred, y_true):
        """Mean Squared Error loss (scalar)."""
        diff = y_pred - y_true
        return (diff * diff).mean()

    print("  ✓ Loss function defined")

    # ========================================================================
    # SECTION 4: Setup Optimizer
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 4: Stochastic Gradient Descent (SGD) Optimizer")
    print("=" * 70)

    learning_rate = 0.1
    print(f"\n[4.1] Creating SGD optimizer with lr={learning_rate}")

    # Create optimizer with our parameters
    params = [w, b]
    optimizer = tz.optim.SGD(params, lr=learning_rate)

    print("  ✓ Optimizer created")
    print(f"    - Optimizing {len(params)} parameters")
    print(f"    - Learning rate: {learning_rate}")

    # ========================================================================
    # SECTION 5: Training Loop
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 5: Training the Model")
    print("=" * 70)

    n_epochs = 100
    print(f"\n[5.1] Training for {n_epochs} epochs...")
    print("  Each epoch: Forward pass -> Compute loss -> Backward pass -> Update params")
    print()

    for epoch in range(n_epochs):
        # Forward pass: compute predictions through the autograd graph.
        # y_pred = x @ w + b (x is [100,1], w is [1,1] -> y_pred is [100,1])
        y_pred = tz.matmul(x_data, w) + b

        # Compute loss (scalar)
        loss = mse_loss(y_pred, y_data)

        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # Backward pass: compute gradients
        loss.backward()

        # Update parameters using gradients
        optimizer.step()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1:3d}/{n_epochs}] - loss: {float(loss.tensor().numpy()):.4f}")

    print("\n  ✓ Training complete!")

    # ========================================================================
    # SECTION 6: Evaluate Results
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 6: Evaluating Trained Model")
    print("=" * 70)

    learned_w = float(w.tensor().numpy().reshape(-1)[0])
    learned_b = float(b.tensor().numpy().reshape(-1)[0])
    print(f"\n[6.1] Comparing learned vs true parameters:")
    print(f"  True parameters:    w={true_w:.4f}, b={true_b:.4f}")
    print(f"  Learned parameters: w={learned_w:.4f}, b={learned_b:.4f}")
    assert abs(learned_w - true_w) < 0.5, "w did not converge toward the true value"
    assert abs(learned_b - true_b) < 0.5, "b did not converge toward the true value"
    print("  ✓ Learned parameters converged to the true values")

    print("\n[6.2] Making predictions on test data...")
    # Inference: no gradient tracking needed
    x_test = tz.randn([10, 1])
    with tz.no_grad():
        y_test_pred = tz.matmul(x_test, w.detach()) + b.detach()

    print(f"  Test input shape: {x_test.shape}")
    print(f"  Predictions shape: {y_test_pred.shape}")
    print("  ✓ Model can make predictions on new data")

    # ========================================================================
    # SECTION 7: Training Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    print("\nWhat we accomplished:")
    print("  1. ✓ Generated synthetic linear data (y = 3x + 2 + noise)")
    print("  2. ✓ Initialized model parameters (w, b) randomly")
    print("  3. ✓ Defined MSE loss function")
    print("  4. ✓ Created SGD optimizer")
    print(f"  5. ✓ Trained model for {n_epochs} epochs")
    print("  6. ✓ Model learned to approximate the true relationship")

    print("\nThe training process:")
    print("  For each epoch:")
    print("    1. Forward: Compute y_pred = wx + b")
    print("    2. Loss: Compute MSE between y_pred and y_true")
    print("    3. Backward: Compute gradients ∂L/∂w and ∂L/∂b")
    print("    4. Update: w ← w - lr * ∂L/∂w, b ← b - lr * ∂L/∂b")

    print("\n" + "=" * 70)
    print("TUTORIAL COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Linear regression finds the best line to fit data")
    print("2. MSE loss measures prediction error")
    print("3. SGD optimizer updates parameters to minimize loss")
    print("4. Training loop: forward -> loss -> backward -> update")
    print("5. Gradients tell us how to adjust parameters")
    print("6. Learning rate controls the step size of updates")

    print("\nNext steps:")
    print("- Try different learning rates")
    print("- Add more features (multiple inputs)")
    print("- Experiment with different optimizers (Adam)")
    print("- Visualize the training loss curve")

if __name__ == "__main__":
    main()
