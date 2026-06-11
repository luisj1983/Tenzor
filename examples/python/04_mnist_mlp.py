"""
Tenzor Tutorial 04: MNIST-style MLP Classification
==================================================
Train a real two-layer MLP with ReLU and cross-entropy on an MNIST-shaped
synthetic dataset, and measure actual accuracy.

The data here is synthetic (class-dependent patterns + noise) so the example
runs offline and fast. To train on real MNIST, replace
generate_mnist_like_data() with your favourite MNIST loader producing the
same (N, 784) float32 / (N,) int64 arrays.
"""

import numpy as np
import tenzor as tz


def generate_mnist_like_data(n_samples=1000, n_classes=10, seed=42):
    """
    Synthetic MNIST-shaped dataset that is actually LEARNABLE: each class has
    a distinct random pattern (centroid) and samples are noisy copies of it.

    Returns:
        X: (n_samples, 784) float32 - flattened "images"
        y: (n_samples,)    int64   - class labels 0..9
    """
    # Class centroids are FIXED (seed 0) so train and test sets share the
    # same class structure; only the samples/noise vary with `seed`.
    centroids = np.random.default_rng(0).normal(
        0.0, 1.0, size=(n_classes, 784)).astype(np.float32)
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n_samples).astype(np.int64)
    X = centroids[y] + rng.normal(0.0, 0.5, size=(n_samples, 784)).astype(np.float32)
    return X.astype(np.float32), y


def main():
    print("=" * 70)
    print("TENZOR TUTORIAL 04: MNIST MLP CLASSIFICATION")
    print("=" * 70)

    # Tenzor auto-initializes on import; the explicit call remains a no-op.
    tz.initialize()

    # ========================================================================
    # SECTION 1: Data
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 1: Dataset")
    print("=" * 70)

    n_train, n_test, n_classes = 1000, 200, 10
    input_size, hidden_size = 784, 256

    print("\n[1.1] Generating synthetic MNIST-like dataset...")
    X_train_np, y_train_np = generate_mnist_like_data(n_train, n_classes, seed=42)
    X_test_np, y_test_np = generate_mnist_like_data(n_test, n_classes, seed=7)
    print(f"  Train: {X_train_np.shape}, Test: {X_test_np.shape}")

    # ========================================================================
    # SECTION 2: Model
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 2: Two-Layer MLP")
    print("=" * 70)

    layer1 = tz.nn.Linear(input_size, hidden_size, bias=True)
    layer2 = tz.nn.Linear(hidden_size, n_classes, bias=True)
    print(f"  ✓ Linear({input_size} -> {hidden_size}) -> ReLU -> "
          f"Linear({hidden_size} -> {n_classes})")

    def forward(x):
        h = tz.nn.relu(layer1(x))      # real autograd-aware ReLU
        return layer2(h)               # logits

    # ========================================================================
    # SECTION 3: Loss and Optimizer
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 3: Cross-Entropy Loss + Adam")
    print("=" * 70)

    params = list(layer1.parameters()) + list(layer2.parameters())
    optimizer = tz.optim.Adam(params, lr=1e-3)
    print(f"  ✓ Adam over {len(params)} parameter tensors, "
          f"loss = tz.nn.cross_entropy(logits, labels)")

    # ========================================================================
    # SECTION 4: Training
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 4: Training")
    print("=" * 70)

    n_epochs, batch_size = 5, 100
    n_batches = n_train // batch_size
    print(f"  Epochs: {n_epochs}, batch size: {batch_size}, "
          f"batches/epoch: {n_batches}")
    print("-" * 70)

    first_loss = last_loss = None
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_idx in range(n_batches):
            s, e = batch_idx * batch_size, (batch_idx + 1) * batch_size
            x = tz.from_numpy(X_train_np[s:e])
            labels = tz.from_numpy(y_train_np[s:e])

            logits = forward(x)
            loss = tz.nn.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = float(loss.tensor().numpy())
            epoch_loss += loss_val
            if first_loss is None:
                first_loss = loss_val

        last_loss = epoch_loss / n_batches
        print(f"  Epoch [{epoch + 1}/{n_epochs}]  avg loss: {last_loss:.4f}")

    print("-" * 70)
    assert last_loss < first_loss, "training did not reduce the loss"
    print(f"✓ Training complete (loss {first_loss:.4f} -> {last_loss:.4f})")

    # ========================================================================
    # SECTION 5: Evaluation (real, measured accuracy)
    # ========================================================================
    print("\n" + "=" * 70)
    print("SECTION 5: Evaluation")
    print("=" * 70)

    layer1.eval()
    layer2.eval()

    with tz.no_grad():
        logits = forward(tz.from_numpy(X_test_np))
        pred = logits.tensor().numpy().argmax(axis=1)

    accuracy = float((pred == y_test_np).mean())
    print(f"\n  Test accuracy: {accuracy * 100:.2f}%  "
          f"(chance level: {100.0 / n_classes:.1f}%)")
    assert accuracy > 0.5, "model failed to learn the class patterns"
    print("  ✓ Model learned real structure (measured, not simulated)")

    # ========================================================================
    # SECTION 6: Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print("\nWhat we accomplished:")
    print(f"  1. ✓ Built MLP: {input_size} -> {hidden_size} -> {n_classes}")
    print("  2. ✓ Trained with real ReLU + cross-entropy + Adam")
    print(f"  3. ✓ Loss decreased {first_loss:.4f} -> {last_loss:.4f}")
    print(f"  4. ✓ Measured test accuracy: {accuracy * 100:.1f}%")
    print("\nTo use real MNIST: swap generate_mnist_like_data() for a loader")
    print("returning (N, 784) float32 images and (N,) int64 labels.")


if __name__ == "__main__":
    main()
