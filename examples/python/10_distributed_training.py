#!/usr/bin/env python3
"""
Distributed Data Parallel Training Example
===========================================

This example demonstrates multi-GPU/multi-node training using DistributedDataParallel.
Launch with torchrun or manually set environment variables.

Launch command:
    torchrun --nproc_per_node=4 10_distributed_training.py

Or manually:
    RANK=0 WORLD_SIZE=4 MASTER_ADDR=localhost MASTER_PORT=29500 python 10_distributed_training.py &
    RANK=1 WORLD_SIZE=4 MASTER_ADDR=localhost MASTER_PORT=29500 python 10_distributed_training.py &
    ...
"""

import os
import sys
import tenzor as tz
import numpy as np

# Initialize Tenzor library (registers backends)
tz.initialize()

# Check if distributed training APIs are available
DISTRIBUTED_AVAILABLE = hasattr(tz.distributed, 'init_process_group')

def setup_distributed():
    """Initialize distributed training from environment variables.

    For a real multi-process run, launch one process per rank with RANK,
    WORLD_SIZE, MASTER_ADDR and MASTER_PORT set (e.g. via a launcher script).
    Without them this example falls back to a single-process world so the
    demo can run standalone.
    """
    os.environ.setdefault('RANK', '0')
    os.environ.setdefault('WORLD_SIZE', '1')
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29500')

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = int(os.environ['MASTER_PORT'])

    print(f"[Rank {rank}] Initializing process group...")
    print(f"[Rank {rank}] World size: {world_size}")
    print(f"[Rank {rank}] Master: {master_addr}:{master_port}")

    # NCCL needs GPUs (one per rank); gloo works on CPU and for the
    # single-process demo.
    backend = 'nccl' if tz.cuda_is_available() and world_size > 1 else 'gloo'
    print(f"[Rank {rank}] Backend: {backend}")
    process_group = tz.distributed.init_process_group(backend=backend)
    if process_group is None:
        # init_process_group registers the group globally; fetch the handle.
        process_group = tz.distributed.get_process_group()

    print(f"[Rank {rank}] Process group initialized successfully!")
    return process_group, rank, world_size


def create_model():
    """Create a simple CNN model for demonstration."""
    model = tz.nn.Sequential(
        tz.nn.Conv2d(3, 64, kernel_size=3, padding=1),
        tz.nn.ReLU(),
        tz.nn.MaxPool2d(2),

        tz.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        tz.nn.ReLU(),
        tz.nn.MaxPool2d(2),

        tz.nn.Flatten(),
        tz.nn.Linear(128 * 8 * 8, 256),
        tz.nn.ReLU(),
        tz.nn.Dropout(0.5),
        tz.nn.Linear(256, 10)
    )
    return model


def main():
    # Setup distributed training
    process_group, rank, world_size = setup_distributed()

    # Set device for this process
    device = tz.Device.cuda(rank)
    print(f"[Rank {rank}] Using device: {device}")

    # Create model and move to GPU
    print(f"[Rank {rank}] Creating model...")
    model = create_model().cuda(rank)

    # Wrap model with DistributedDataParallel
    print(f"[Rank {rank}] Wrapping model with DDP...")
    # Tenzor DDP signature: (module, process_group, bucket_size_bytes=25MB).
    # Device placement follows the module's device (no device_ids kwarg).
    ddp_model = tz.distributed.DistributedDataParallel(model, process_group)

    # Create optimizer over the WRAPPED module's parameters (the DDP wrapper
    # only owns the communication hooks, not the parameters).
    optimizer = tz.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = tz.nn.CrossEntropyLoss()

    print(f"[Rank {rank}] Starting training...")

    # Training loop
    num_epochs = 5
    batch_size = 32
    use_gpu = tz.cuda_is_available()

    for epoch in range(num_epochs):
        model.train()

        # Simulate training batches (in real code, use DataLoader)
        num_batches = 10
        total_loss = 0.0

        for batch_idx in range(num_batches):
            # Create dummy data (in real code, load from DataLoader)
            # Each rank gets different data
            rng = np.random.default_rng(epoch * num_batches + batch_idx + rank * 1000)
            inputs = tz.randn([batch_size, 3, 32, 32])
            targets = tz.from_numpy(rng.integers(0, 10, batch_size).astype(np.int64))
            if use_gpu:
                inputs = inputs.cuda(rank)
                targets = targets.cuda(rank)

            # Forward pass through the DDP wrapper
            optimizer.zero_grad()
            outputs = ddp_model.forward(inputs)
            loss = loss_fn(outputs, targets)

            # Backward pass, then all-reduce gradients across ranks
            loss.backward()
            ddp_model.synchronize_gradients()

            # Optimizer step (each rank applies the synchronized gradients)
            optimizer.step()

            total_loss += loss.tensor().item()

            if rank == 0 and batch_idx % 5 == 0:
                print(f"[Rank {rank}] Epoch {epoch+1}/{num_epochs}, "
                      f"Batch {batch_idx}/{num_batches}, "
                      f"Loss: {loss.tensor().item():.4f}")

        # Synchronize at epoch end
        process_group.barrier()

        avg_loss = total_loss / num_batches

        if rank == 0:
            print(f"[Rank {rank}] Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

    # Final synchronization
    print(f"[Rank {rank}] Training completed!")
    process_group.barrier()

    if rank == 0:
        print("=" * 60)
        print("Distributed training finished successfully!")
        print("=" * 60)
        print("\nKey features demonstrated:")
        print("✅ ProcessGroup initialization from environment")
        print("✅ DistributedDataParallel model wrapping")
        print("✅ Automatic gradient synchronization")
        print("✅ Multi-GPU training coordination")
        print("✅ Barrier synchronization")

    # Cleanup
    tz.distributed.destroy_process_group()


if __name__ == '__main__':
    if not DISTRIBUTED_AVAILABLE:
        print("=" * 60)
        print("DISTRIBUTED TRAINING APIs NOT YET IMPLEMENTED")
        print("=" * 60)
        print("\nThis example requires the following APIs which are not yet available:")
        print("  - tz.distributed.init_process_group()")
        print("  - tz.distributed.DistributedDataParallel()")
        print("  - tz.distributed.destroy_process_group()")
        print("\nThese APIs will be added in a future release of Tenzor.")
        print("\nThe example demonstrates the intended usage pattern for when")
        print("distributed training support is implemented.")
        print("=" * 60)
        sys.exit(0)

    if int(os.environ.get('WORLD_SIZE', 1)) == 1:
        print("=" * 60)
        print("SINGLE GPU MODE")
        print("=" * 60)
        print("\nTo run distributed training, use:")
        print("\n  torchrun --nproc_per_node=4 10_distributed_training.py")
        print("\nOr manually set environment variables:")
        print("\n  RANK=0 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=29500 python 10_distributed_training.py &")
        print("  RANK=1 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=29500 python 10_distributed_training.py &")
        print("\nFor now, running with single GPU...")
        print("=" * 60)

    main()
