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

def setup_distributed():
    """Initialize distributed training from environment variables."""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = int(os.environ.get('MASTER_PORT', 29500))

    print(f"[Rank {rank}] Initializing process group...")
    print(f"[Rank {rank}] World size: {world_size}")
    print(f"[Rank {rank}] Master: {master_addr}:{master_port}")

    # Create process group
    process_group = tz.nn.parallel.init_process_group(backend='nccl')

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
    ddp_model = tz.nn.parallel.DistributedDataParallel(
        model,
        process_group,
        device_ids=[rank],
        output_device=rank,
        broadcast_buffers=True,
        find_unused_parameters=False
    )

    # Create optimizer (on rank 0, parameters are broadcast to all ranks)
    optimizer = tz.optim.Adam(ddp_model.parameters(), lr=0.001)
    loss_fn = tz.nn.CrossEntropyLoss()

    print(f"[Rank {rank}] Starting training...")

    # Training loop
    num_epochs = 5
    batch_size = 32

    for epoch in range(num_epochs):
        ddp_model.train()

        # Simulate training batches (in real code, use DataLoader)
        num_batches = 10
        total_loss = 0.0

        for batch_idx in range(num_batches):
            # Create dummy data (in real code, load from DataLoader)
            # Each rank gets different data
            np.random.seed(epoch * num_batches + batch_idx + rank * 1000)
            inputs = tz.randn([batch_size, 3, 32, 32]).cuda(rank)
            targets = tz.Tensor([np.random.randint(0, 10) for _ in range(batch_size)]).cuda(rank)

            # Forward pass (DDP handles gradient synchronization automatically)
            optimizer.zero_grad()
            outputs = ddp_model(tz.Variable(inputs, requires_grad=True))
            loss = loss_fn(outputs, tz.Variable(targets))

            # Backward pass (gradients are all-reduced across processes)
            loss.backward()

            # Optimizer step (each rank updates its local copy)
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
    tz.nn.parallel.destroy_process_group(process_group)


if __name__ == '__main__':
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
