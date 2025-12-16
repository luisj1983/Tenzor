#!/usr/bin/env python3
"""
ResNet-18 on CIFAR-10
=====================

This example demonstrates:
1. Modern CNN architecture (ResNet with skip connections)
2. Data augmentation for better generalization
3. Learning rate scheduling
4. Mixed precision training (if CUDA available)
5. Model checkpointing and evaluation

ResNet (Residual Network) uses skip connections to enable training
of very deep networks without vanishing gradients.
"""

import tenzor as tz
import numpy as np
import os
import time
from typing import Tuple, List

# Initialize Tenzor library (registers backends)
tz.initialize()

# Configuration
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_CLASSES = 10
# Use CUDA if available (Python modules in Sequential now work with device placement)
DEVICE = tz.Device.cuda(0) if tz.cuda_is_available() else tz.Device.cpu()
USE_MIXED_PRECISION = tz.cuda_is_available()

print(f"Training on: {DEVICE}")
print(f"Mixed precision: {USE_MIXED_PRECISION}")


class BasicBlock(tz.nn.Module):
    """Basic ResNet block with two 3x3 convolutions"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # Main path
        self.conv1 = tz.nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = tz.nn.BatchNorm2d(out_channels)
        self.conv2 = tz.nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = tz.nn.BatchNorm2d(out_channels)

        # Skip connection (identity or projection)
        self.skip = tz.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = tz.nn.Sequential(
                tz.nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                tz.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: tz.Variable) -> tz.Variable:
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = tz.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection
        out = out + self.skip(identity)
        out = tz.nn.relu(out)

        return out


class ResNet18(tz.nn.Module):
    """ResNet-18 architecture for CIFAR-10"""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Initial conv (CIFAR-10 uses smaller kernel than ImageNet)
        self.conv1 = tz.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = tz.nn.BatchNorm2d(64)

        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        # Classifier
        self.avgpool = tz.nn.AdaptiveAvgPool2d(1, 1)
        self.flatten = tz.nn.Flatten(1)  # Flatten from dim 1 onwards
        self.fc = tz.nn.Linear(512, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int,
                    num_blocks: int, stride: int) -> tz.nn.Sequential:
        # Use Sequential now that Python modules are properly supported
        layers = []

        # First block (may have stride > 1)
        layers.append(BasicBlock(in_channels, out_channels, stride))

        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))

        return tz.nn.Sequential(*layers)

    def forward(self, x: tz.Variable) -> tz.Variable:
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = tz.nn.relu(x)

        # ResNet layers (now Sequential works with Python modules!)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classifier
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


def load_cifar10() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load CIFAR-10 dataset (simplified version)
    In practice, use torchvision or similar library
    """
    # This is a placeholder - in real code, load from file or download
    print("Loading CIFAR-10 dataset...")
    print("Note: This example uses synthetic data. Use torchvision.datasets.CIFAR10 for real data.")

    # Synthetic data for demonstration
    train_images = np.random.randn(5000, 3, 32, 32).astype(np.float32)
    train_labels = np.random.randint(0, NUM_CLASSES, 5000).astype(np.int64)

    test_images = np.random.randn(1000, 3, 32, 32).astype(np.float32)
    test_labels = np.random.randint(0, NUM_CLASSES, 1000).astype(np.int64)

    # Normalize (mean=0, std=1 for simplicity)
    train_images = (train_images - train_images.mean()) / (train_images.std() + 1e-7)
    test_images = (test_images - test_images.mean()) / (test_images.std() + 1e-7)

    return (train_images, train_labels), (test_images, test_labels)


def augment_batch(images: np.ndarray) -> np.ndarray:
    """
    Apply data augmentation (random crop, horizontal flip)
    Simplified version - use torchvision.transforms in practice
    """
    augmented = images.copy()

    # Random horizontal flip (50% chance)
    if np.random.rand() > 0.5:
        augmented = augmented[:, :, :, ::-1]

    # Random crop (would need padding in practice)
    # For now, just return as-is

    return augmented


def train_epoch(model: tz.nn.Module, optimizer,
                criterion: tz.nn.Module, train_data: Tuple[np.ndarray, np.ndarray],
                epoch: int, scaler=None) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()

    train_images, train_labels = train_data
    num_samples = len(train_images)
    num_batches = num_samples // BATCH_SIZE

    total_loss = 0.0
    correct = 0
    total = 0

    # Shuffle data
    indices = np.random.permutation(num_samples)

    for batch_idx in range(num_batches):
        # Get batch
        batch_indices = indices[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]
        batch_images = train_images[batch_indices]
        batch_labels = train_labels[batch_indices]

        # Augment
        batch_images = augment_batch(batch_images)

        # Convert to tensors
        inputs = tz.Variable(tz.Tensor.from_numpy(batch_images).to(DEVICE), requires_grad=True)
        targets = tz.Tensor.from_numpy(batch_labels).to(DEVICE)

        # Forward pass (with autocast if using mixed precision)
        if USE_MIXED_PRECISION and scaler is not None:
            # Mixed precision forward pass
            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)

            # Backward pass with gradient scaling
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            scaler.unscale_(optimizer)
            optimizer.step()
            scaler.update()
        else:
            # Regular precision
            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Statistics
        total_loss += loss.tensor().item()

        # Accuracy
        predictions = tz.argmax(outputs.tensor(), dim=1)
        correct += (predictions.numpy() == batch_labels).sum()
        total += len(batch_labels)

        # Print progress
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100.0 * correct / total
            print(f"Epoch {epoch} [{batch_idx + 1}/{num_batches}] "
                  f"Loss: {avg_loss:.4f} Acc: {accuracy:.2f}%")

    epoch_loss = total_loss / num_batches
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def evaluate(model: tz.nn.Module, criterion: tz.nn.Module,
             test_data: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, float]:
    """Evaluate model on test set"""
    model.eval()

    test_images, test_labels = test_data
    num_samples = len(test_images)
    num_batches = num_samples // BATCH_SIZE

    total_loss = 0.0
    correct = 0
    total = 0

    with tz.no_grad():
        for batch_idx in range(num_batches):
            # Get batch
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            batch_images = test_images[start_idx:end_idx]
            batch_labels = test_labels[start_idx:end_idx]

            # Convert to tensors
            inputs = tz.Variable(tz.Tensor.from_numpy(batch_images).to(DEVICE))
            targets = tz.Tensor.from_numpy(batch_labels).to(DEVICE)

            # Forward pass
            outputs = model.forward(inputs)
            loss = criterion(outputs, targets)

            # Statistics
            total_loss += loss.tensor().item()

            # Accuracy
            predictions = tz.argmax(outputs.tensor(), dim=1)
            correct += (predictions.numpy() == batch_labels).sum()
            total += len(batch_labels)

    test_loss = total_loss / num_batches
    test_acc = 100.0 * correct / total

    return test_loss, test_acc


def main():
    """Main training loop"""
    print("="*60)
    print("ResNet-18 CIFAR-10 Training")
    print("="*60)

    # Initialize model
    model = ResNet18(NUM_CLASSES)
    model.to(DEVICE)

    # Count parameters
    num_params = sum(p.tensor().numel for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer with momentum and weight decay
    optimizer = tz.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduler (multi-step decay)
    scheduler = tz.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Loss function
    criterion = tz.nn.CrossEntropyLoss()

    # Gradient scaler for mixed precision
    scaler = tz.amp.GradScaler() if USE_MIXED_PRECISION else None

    # Load data
    train_data, test_data = load_cifar10()

    # Training loop
    best_acc = 0.0

    print("\nStarting training...")
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, optimizer, criterion, train_data, epoch, scaler
        )

        # Evaluate
        test_loss, test_acc = evaluate(model, criterion, test_data)

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()

        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch}/{EPOCHS} - {epoch_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} Test Acc: {test_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"New best accuracy! Saving model...")
            model.save("resnet18_cifar10_best.pth")

        print("-"*60)

    print(f"\nTraining complete! Best test accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
