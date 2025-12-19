"""
Model Pruning Example

This comprehensive example demonstrates:
- Magnitude-based weight pruning
- Unstructured pruning (individual weights)
- Structured pruning (entire channels/filters)
- Sparsity analysis
- Pruning with fine-tuning

Components used:
- Conv2d, Linear, BatchNorm2d
- Custom pruning masks
- Sparsity computation
"""

import tenzor as tz
import numpy as np


# ============================================================================
# Pruning Utilities
# ============================================================================

def compute_sparsity(weights):
    """Compute sparsity ratio (percentage of zeros)"""
    total = weights.size
    zeros = np.sum(np.abs(weights) < 1e-8)
    return zeros / total


def compute_layer_stats(layer_weights):
    """Compute statistics for a layer's weights"""
    return {
        'mean': np.mean(np.abs(layer_weights)),
        'std': np.std(layer_weights),
        'min': np.min(np.abs(layer_weights)),
        'max': np.max(np.abs(layer_weights)),
        'sparsity': compute_sparsity(layer_weights)
    }


def create_magnitude_mask(weights, sparsity_ratio):
    """
    Create pruning mask based on weight magnitude

    Args:
        weights: Weight tensor
        sparsity_ratio: Fraction of weights to prune (0-1)

    Returns:
        Binary mask (1 = keep, 0 = prune)
    """
    flat = np.abs(weights.flatten())
    threshold_idx = int(len(flat) * sparsity_ratio)

    if threshold_idx == 0:
        return np.ones_like(weights)

    threshold = np.sort(flat)[threshold_idx]
    mask = (np.abs(weights) >= threshold).astype(np.float32)
    return mask


def apply_mask(weights, mask):
    """Apply pruning mask to weights"""
    return weights * mask


# ============================================================================
# Simple CNN Model
# ============================================================================

class SimpleCNN:
    """Simple CNN for pruning demonstration"""

    def __init__(self, num_classes=10):
        # Conv layers
        self.conv1 = tz.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = tz.nn.BatchNorm2d(32)
        self.pool1 = tz.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = tz.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = tz.nn.BatchNorm2d(64)
        self.pool2 = tz.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = tz.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = tz.nn.BatchNorm2d(128)
        self.pool3 = tz.nn.MaxPool2d(kernel_size=2, stride=2)

        # FC layers (assuming 32x32 input -> 4x4 after pooling)
        self.fc1 = tz.nn.Linear(128 * 4 * 4, 256)
        self.fc2 = tz.nn.Linear(256, num_classes)
        self.dropout = tz.nn.Dropout(0.5)

        # Store pruning masks
        self.masks = {}

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = tz.nn.relu(x)
        x = self.pool1(x)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = tz.nn.relu(x)
        x = self.pool2(x)

        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = tz.nn.relu(x)
        x = self.pool3(x)

        # Flatten
        x_np = x.tensor().numpy()
        batch_size = x_np.shape[0]
        x_flat = x_np.reshape(batch_size, -1)
        x = tz.Variable(tz.Tensor.from_numpy(x_flat.astype(np.float32)), requires_grad=True)

        # FC layers
        x = self.fc1(x)
        x = tz.nn.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_weight_layers(self):
        """Get all layers with weights for pruning"""
        return {
            'conv1': self.conv1,
            'conv2': self.conv2,
            'conv3': self.conv3,
            'fc1': self.fc1,
            'fc2': self.fc2,
        }

    def parameters(self):
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.bn1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.bn2.parameters())
        params.extend(self.conv3.parameters())
        params.extend(self.bn3.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params

    def count_parameters(self):
        """Count total and non-zero parameters"""
        total = 0
        nonzero = 0
        for param in self.parameters():
            w = param.tensor().numpy()
            total += w.size
            nonzero += np.sum(np.abs(w) > 1e-8)
        return total, nonzero


# ============================================================================
# Pruning Demonstrations
# ============================================================================

def demo_magnitude_pruning():
    """Demonstrate magnitude-based pruning"""
    print("\n" + "=" * 60)
    print("Magnitude-Based Pruning Demo")
    print("=" * 60)

    # Create sample weights
    np.random.seed(42)
    weights = np.random.randn(64, 32, 3, 3).astype(np.float32)

    print("\nOriginal weights statistics:")
    stats = compute_layer_stats(weights)
    print(f"  Shape: {weights.shape}")
    print(f"  Mean magnitude: {stats['mean']:.4f}")
    print(f"  Std: {stats['std']:.4f}")
    print(f"  Initial sparsity: {stats['sparsity']*100:.2f}%")

    # Apply different sparsity levels
    for sparsity in [0.3, 0.5, 0.7, 0.9]:
        mask = create_magnitude_mask(weights, sparsity)
        pruned = apply_mask(weights, mask)

        actual_sparsity = compute_sparsity(pruned)
        print(f"\nTarget sparsity: {sparsity*100:.0f}%")
        print(f"  Actual sparsity: {actual_sparsity*100:.2f}%")
        print(f"  Non-zero weights: {np.sum(mask):.0f} / {mask.size}")


def demo_layer_wise_pruning():
    """Demonstrate layer-wise pruning on a model"""
    print("\n" + "=" * 60)
    print("Layer-wise Pruning Demo")
    print("=" * 60)

    model = SimpleCNN(num_classes=10)
    layers = model.get_weight_layers()

    print("\nBefore pruning:")
    total, nonzero = model.count_parameters()
    print(f"  Total parameters: {total:,}")
    print(f"  Non-zero parameters: {nonzero:,}")

    # Analyze each layer
    print("\nLayer-wise analysis:")
    print("-" * 50)

    for name, layer in layers.items():
        params = layer.parameters()
        if len(params) > 0:
            weights = params[0].tensor().numpy()
            stats = compute_layer_stats(weights)
            print(f"{name:10} | Shape: {str(weights.shape):20} | "
                  f"Mean: {stats['mean']:.4f} | Sparsity: {stats['sparsity']*100:.1f}%")


def demo_iterative_pruning():
    """Demonstrate iterative pruning with gradual sparsity increase"""
    print("\n" + "=" * 60)
    print("Iterative Pruning Demo")
    print("=" * 60)

    # Simulate iterative pruning schedule
    initial_sparsity = 0.0
    final_sparsity = 0.8
    num_steps = 5

    print("\nIterative pruning schedule (polynomial):")
    print("-" * 40)

    for step in range(num_steps + 1):
        # Polynomial schedule: s_t = s_f * (1 - (1 - t/T)^3)
        t = step / num_steps
        current_sparsity = final_sparsity * (1 - (1 - t) ** 3)
        print(f"Step {step}/{num_steps}: Sparsity = {current_sparsity*100:.1f}%")

    print("\nIterative pruning benefits:")
    print("  - Allows model to adapt to increasing sparsity")
    print("  - Better accuracy retention than one-shot pruning")
    print("  - Fine-tuning between pruning steps is key")


def demo_structured_pruning():
    """Demonstrate structured (channel) pruning concepts"""
    print("\n" + "=" * 60)
    print("Structured Pruning Demo")
    print("=" * 60)

    print("\nUnstructured vs Structured Pruning:")
    print("-" * 50)

    print("\nUnstructured Pruning:")
    print("  - Prunes individual weights")
    print("  - Creates sparse weight matrices")
    print("  - Requires special sparse hardware/libraries")
    print("  - Higher compression ratio possible")

    print("\nStructured Pruning:")
    print("  - Prunes entire channels/filters/heads")
    print("  - Results in smaller dense matrices")
    print("  - Works on standard hardware")
    print("  - More predictable speedup")

    # Simulate channel pruning
    np.random.seed(42)
    weights = np.random.randn(64, 32, 3, 3).astype(np.float32)

    # Compute channel importance (L1 norm of each filter)
    channel_importance = np.sum(np.abs(weights), axis=(1, 2, 3))

    print(f"\nChannel importance scores (first 10 of {len(channel_importance)}):")
    for i in range(min(10, len(channel_importance))):
        print(f"  Channel {i}: {channel_importance[i]:.4f}")

    # Identify channels to prune (lowest 25%)
    prune_ratio = 0.25
    num_to_prune = int(len(channel_importance) * prune_ratio)
    channels_to_prune = np.argsort(channel_importance)[:num_to_prune]

    print(f"\nChannels to prune (lowest {prune_ratio*100:.0f}%): {list(channels_to_prune)}")
    print(f"Remaining channels: {len(channel_importance) - num_to_prune}")


def demo_pruning_with_training():
    """Demonstrate pruning with mock training"""
    print("\n" + "=" * 60)
    print("Pruning with Fine-tuning Demo")
    print("=" * 60)

    model = SimpleCNN(num_classes=10)

    # Count initial parameters
    total, nonzero = model.count_parameters()
    print(f"\nInitial model:")
    print(f"  Total parameters: {total:,}")
    print(f"  Non-zero parameters: {nonzero:,}")

    # Simulate training loop with pruning
    print("\nSimulated pruning + fine-tuning schedule:")
    print("-" * 50)

    sparsity_schedule = [0.2, 0.4, 0.6]
    for i, target_sparsity in enumerate(sparsity_schedule):
        print(f"\nIteration {i+1}:")
        print(f"  1. Prune to {target_sparsity*100:.0f}% sparsity")
        print(f"  2. Fine-tune for N epochs")
        print(f"  3. Evaluate accuracy")

    print("\nKey insights:")
    print("  - Pruning removes low-magnitude weights")
    print("  - Fine-tuning recovers accuracy loss")
    print("  - Iterative approach outperforms one-shot")
    print("  - Final model: same architecture, fewer active weights")


# ============================================================================
# Main
# ============================================================================

def main():
    tz.initialize()

    print("=" * 60)
    print("   Model Pruning Example")
    print("=" * 60)

    print("\nComponents demonstrated:")
    print("  Pruning: Magnitude-based, Unstructured, Structured")
    print("  Analysis: Sparsity computation, Layer statistics")
    print("  Training: Iterative pruning with fine-tuning")

    demo_magnitude_pruning()
    demo_layer_wise_pruning()
    demo_iterative_pruning()
    demo_structured_pruning()
    demo_pruning_with_training()

    print("\n" + "=" * 60)
    print("   Pruning example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
