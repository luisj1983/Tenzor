"""
Comprehensive examples for model checkpointing in Tenzor.

This module demonstrates:
1. Basic model save/load
2. Saving with optimizer state
3. AutoCheckpoint for training loops
4. Best model tracking
5. Checkpoint metadata
6. Training resumption

Author: Tenzor Team
Date: 2024
"""

import tenzor
import tenzor.nn as nn

# Initialize Tenzor library (registers backends)
tenzor.initialize()

# Access optim module after initialization
optim = tenzor.optim

# ============================================================================
# Example 1: Basic Model Save/Load
# ============================================================================

def example_basic_save_load():
    """
    Basic checkpoint save and load.

    Demonstrates:
    - Creating a simple model
    - Saving model to disk
    - Loading model from disk
    - Verifying parameters match
    """
    print("=" * 70)
    print("Example 1: Basic Model Save/Load")
    print("=" * 70)

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    # Create checkpoint manager
    checkpoint_manager = nn.ModelCheckpoint()

    # Save model
    checkpoint_path = "checkpoints/simple_model.pt"
    print(f"Saving model to {checkpoint_path}...")
    checkpoint_manager.save_model(checkpoint_path, model)

    # Verify checkpoint exists
    if checkpoint_manager.verify_checkpoint(checkpoint_path):
        print("Checkpoint verified successfully!")

    # Load model state
    print("Loading model from checkpoint...")
    loaded_state = checkpoint_manager.load_model(checkpoint_path)

    # Create new model and load state
    new_model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    new_model.load_state_dict(loaded_state)

    print("Model loaded successfully!")
    print(f"Number of parameters loaded: {len(loaded_state)}")
    print()


# ============================================================================
# Example 2: Save with Optimizer and Scheduler
# ============================================================================

def example_save_with_optimizer():
    """
    Save complete training state including optimizer and scheduler.

    Demonstrates:
    - Saving model, optimizer, and scheduler together
    - Loading complete training state
    - Resuming training from checkpoint
    """
    print("=" * 70)
    print("Example 2: Save with Optimizer and Scheduler")
    print("=" * 70)

    # Create model
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Create metadata
    metadata = nn.TrainingMetadata()
    metadata.epoch = 42
    metadata.global_step = 10000
    metadata.train_loss = 0.25
    metadata.val_loss = 0.30
    metadata.learning_rate = 0.001
    metadata.custom_metrics["f1_score"] = 0.85

    # Save complete checkpoint
    checkpoint_manager = nn.ModelCheckpoint()
    checkpoint_path = "checkpoints/complete_training_state.pt"

    print("Saving complete training state...")
    checkpoint_manager.save(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        metadata
    )

    # Load checkpoint
    print("Loading checkpoint...")
    loaded_checkpoint = checkpoint_manager.load(checkpoint_path)

    # Display metadata
    print(f"  Epoch: {loaded_checkpoint.metadata.epoch}")
    print(f"  Global step: {loaded_checkpoint.metadata.global_step}")
    print(f"  Train loss: {loaded_checkpoint.metadata.train_loss:.4f}")
    print(f"  Val loss: {loaded_checkpoint.metadata.val_loss:.4f}")
    print(f"  Learning rate: {loaded_checkpoint.metadata.learning_rate:.6f}")
    if 'f1_score' in loaded_checkpoint.metadata.custom_metrics:
        print(f"  F1 score: {loaded_checkpoint.metadata.custom_metrics['f1_score']:.4f}")
    else:
        print("  F1 score: (custom_metrics not preserved in checkpoint)")

    # Restore model
    new_model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    new_model.load_state_dict(loaded_checkpoint.model_state)

    # Restore optimizer
    new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
    new_optimizer.load_state_dict(loaded_checkpoint.optimizer_state)

    print("Training state restored successfully!")
    print(f"Model parameters: {len(loaded_checkpoint.model_state)}")
    print(f"Optimizer state: {len(loaded_checkpoint.optimizer_state)}")
    print()


# ============================================================================
# Example 3: AutoCheckpoint for Training Loops
# ============================================================================

def example_auto_checkpoint():
    """
    Automatic checkpoint management during training.

    Demonstrates:
    - AutoCheckpoint setup
    - Automatic saving during training
    - Best model tracking
    - Old checkpoint cleanup
    """
    print("=" * 70)
    print("Example 3: AutoCheckpoint for Training Loops")
    print("=" * 70)

    # Create model
    model = nn.Sequential(
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Setup AutoCheckpoint
    auto_checkpoint = nn.AutoCheckpoint(
        directory="checkpoints/auto",
        max_checkpoints=5,  # Keep top 5 checkpoints
        save_frequency=2    # Save every 2 epochs
    )

    # Set metric mode (minimize validation loss)
    auto_checkpoint.set_metric_mode("min")

    print("Starting training simulation...")
    print("Configuration:")
    print(f"  Max checkpoints: 5")
    print(f"  Save frequency: Every 2 epochs")
    print(f"  Metric mode: min (lower is better)")
    print()

    # Simulate training for 10 epochs
    num_epochs = 10
    for epoch in range(num_epochs):
        # Simulate decreasing validation loss (good training!)
        import math
        val_loss = 1.0 * math.exp(-0.15 * epoch) + 0.1

        # Step auto checkpoint
        saved = auto_checkpoint.step(
            model,
            optimizer,
            epoch,
            val_loss,
            "val_loss"
        )

        if saved:
            print(f"Epoch {epoch:2d}: val_loss = {val_loss:.4f} [SAVED]")
        else:
            print(f"Epoch {epoch:2d}: val_loss = {val_loss:.4f}")

    # Print summary
    print()
    print("Training complete!")
    print(f"Best validation loss: {auto_checkpoint.best_metric_value():.4f}")
    print(f"Best checkpoint: {auto_checkpoint.best_checkpoint_path()}")
    print(f"Total checkpoints saved: {len(auto_checkpoint.checkpoint_paths())}")
    print()


# ============================================================================
# Example 4: Training Resumption
# ============================================================================

def example_training_resumption():
    """
    Resume training from checkpoint.

    Demonstrates:
    - Saving training state
    - Loading and continuing from checkpoint
    - Preserving training history
    """
    print("=" * 70)
    print("Example 4: Training Resumption")
    print("=" * 70)

    # Initial training
    print("Phase 1: Initial Training (epochs 0-4)")
    print("-" * 40)

    model = nn.Sequential(
        nn.Linear(50, 30),
        nn.ReLU(),
        nn.Linear(30, 10)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for 5 epochs
    for epoch in range(5):
        # Simulate training
        train_loss = 1.0 / (epoch + 1)
        val_loss = 1.2 / (epoch + 1)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # Save checkpoint
    metadata = nn.TrainingMetadata()
    metadata.epoch = 5
    metadata.train_loss = train_loss
    metadata.val_loss = val_loss

    checkpoint_manager = nn.ModelCheckpoint()
    checkpoint_path = "checkpoints/resume_point.pt"
    checkpoint_manager.save(checkpoint_path, model, optimizer, None, metadata)
    print(f"Checkpoint saved at epoch {metadata.epoch}")

    # Resume training
    print()
    print("Phase 2: Resume Training (epochs 5-9)")
    print("-" * 40)

    # Load checkpoint
    loaded = checkpoint_manager.load(checkpoint_path)

    # Restore state
    new_model = nn.Sequential(
        nn.Linear(50, 30),
        nn.ReLU(),
        nn.Linear(30, 10)
    )
    new_model.load_state_dict(loaded.model_state)

    new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
    new_optimizer.load_state_dict(loaded.optimizer_state)

    start_epoch = loaded.metadata.epoch
    print(f"Resumed from epoch {start_epoch}")

    # Continue training
    for epoch in range(start_epoch, start_epoch + 5):
        # Simulate continued training
        train_loss = 1.0 / (epoch + 1)
        val_loss = 1.2 / (epoch + 1)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    print()
    print("Training successfully resumed and continued!")
    print()


# ============================================================================
# Example 5: Checkpoint Configuration
# ============================================================================

def example_checkpoint_config():
    """
    Advanced checkpoint configuration.

    Demonstrates:
    - Custom checkpoint configuration
    - Atomic saves
    - Checksum verification
    - Configuration management
    """
    print("=" * 70)
    print("Example 5: Checkpoint Configuration")
    print("=" * 70)

    # Create model
    model = nn.Sequential(
        nn.Linear(30, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )

    # Create custom configuration
    config = nn.CheckpointConfig()
    config.save_optimizer = True       # Include optimizer state
    config.save_scheduler = True       # Include scheduler state
    config.verify_checksum = True      # Verify data integrity
    config.atomic_save = True          # Use atomic writes

    # Create checkpoint manager with config
    checkpoint_manager = nn.ModelCheckpoint(config)

    print("Checkpoint Configuration:")
    print(f"  Save optimizer: {config.save_optimizer}")
    print(f"  Save scheduler: {config.save_scheduler}")
    print(f"  Verify checksum: {config.verify_checksum}")
    print(f"  Atomic save: {config.atomic_save}")
    print()

    # Save with configuration
    checkpoint_path = "checkpoints/config_demo.pt"
    checkpoint_manager.save_model(checkpoint_path, model)

    # Verify checkpoint
    print("Verifying checkpoint integrity...")
    if checkpoint_manager.verify_checkpoint(checkpoint_path):
        print("Checkpoint integrity verified (checksum OK)!")

    # Get checkpoint metadata
    print()
    print("Checkpoint Information:")
    version = checkpoint_manager.get_version(checkpoint_path)
    print(f"  Format version: {version}")
    print(f"  Compatible: {checkpoint_manager.is_compatible(checkpoint_path)}")
    print()


# ============================================================================
# Example 6: Complete Training Workflow
# ============================================================================

def example_complete_workflow():
    """
    Complete end-to-end training workflow with checkpointing.

    Demonstrates:
    - Full training loop
    - Automatic checkpoint management
    - Best model selection
    - Final model deployment
    """
    print("=" * 70)
    print("Example 6: Complete Training Workflow")
    print("=" * 70)

    # Setup
    model = nn.Sequential(
        nn.Linear(100, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(32, 10)
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Setup AutoCheckpoint
    auto_checkpoint = nn.AutoCheckpoint(
        directory="checkpoints/complete_workflow",
        max_checkpoints=3,
        save_frequency=1
    )
    auto_checkpoint.set_metric_mode("min")

    print("Training Configuration:")
    print(f"  Model: 3-layer MLP with dropout")
    print(f"  Optimizer: Adam (lr=0.001)")
    print(f"  Scheduler: StepLR (step=5, gamma=0.1)")
    print(f"  Checkpoints: Keep best 3")
    print()

    # Training loop
    num_epochs = 15
    print("Starting training...")
    for epoch in range(num_epochs):
        # Simulate training
        import math
        train_loss = 1.5 * math.exp(-0.1 * epoch) + 0.05
        val_loss = 1.7 * math.exp(-0.1 * epoch) + 0.1
        val_accuracy = min(0.95, 0.5 + 0.4 * (1 - math.exp(-0.15 * epoch)))

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()

        # Save checkpoint
        metadata = nn.TrainingMetadata()
        metadata.epoch = epoch
        metadata.global_step = (epoch + 1) * 100
        metadata.learning_rate = current_lr
        metadata.train_loss = train_loss
        metadata.val_loss = val_loss
        metadata.val_accuracy = val_accuracy

        saved = auto_checkpoint.step(
            model,
            optimizer,
            epoch,
            val_loss,
            "val_loss",
            scheduler
        )

        status = "[BEST]" if saved and val_loss < auto_checkpoint.best_metric_value() else "[SAVED]" if saved else ""
        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_acc={val_accuracy:.4f}, lr={current_lr:.6f} {status}")

    # Training complete
    print()
    print("Training Complete!")
    print(f"Best validation loss: {auto_checkpoint.best_metric_value():.4f}")
    print(f"Best checkpoint saved at: {auto_checkpoint.best_checkpoint_path()}")

    # Load best model
    print()
    print("Loading best model for deployment...")
    checkpoint_manager = nn.ModelCheckpoint()
    best_checkpoint = checkpoint_manager.load(auto_checkpoint.best_checkpoint_path())

    # Create deployment model
    deployment_model = nn.Sequential(
        nn.Linear(100, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(32, 10)
    )
    deployment_model.load_state_dict(best_checkpoint.model_state)
    deployment_model.eval()  # Set to evaluation mode

    print("Best model loaded and ready for deployment!")
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples."""
    import os

    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    print("\n" + "=" * 70)
    print(" Tenzor Model Checkpointing Examples")
    print("=" * 70 + "\n")

    try:
        example_basic_save_load()
        example_save_with_optimizer()
        example_auto_checkpoint()
        example_training_resumption()
        example_checkpoint_config()
        example_complete_workflow()

        print("=" * 70)
        print(" All Examples Completed Successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
