# Distributed Training

Train models across multiple GPUs and machines using Tenzor's distributed backends.

## Overview

Tenzor supports distributed training via:
- **NCCL**: GPU-to-GPU communication (recommended for multi-GPU)
- **Gloo**: CPU-based communication (good for CPU clusters)

## Data Parallel Training (DDP)

```python
import tenzor as tz

# Initialize distributed context
tz.distributed.initialize_from_env()
rank = tz.distributed.get_rank()
world_size = tz.distributed.get_world_size()

# Create model and wrap with DDP
model = tz.nn.Sequential([
    tz.nn.Linear(784, 256),
    tz.nn.ReLU(),
    tz.nn.Linear(256, 10),
])

# DataLoader with distributed sampler
dataset = tz.data.TensorDataset(train_data, train_labels)
sampler = tz.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
loader = tz.data.DataLoader(dataset, batch_size=64, sampler=sampler)

optimizer = tz.optim.Adam(model.parameters(), lr=0.001)
```

## ZeRO Optimizer

For large models that don't fit in a single GPU's memory:

```python
# Stage 2: Shard optimizer states + gradients across GPUs
optimizer = tz.optim.ZeROOptimizer(
    model.parameters(),
    base_optimizer_cls=tz.optim.Adam,
    lr=0.001,
    stage=2,
)
```

## Collective Operations

Low-level collective communication primitives:

```python
# All-reduce: sum tensors across all processes
tz.distributed.all_reduce(tensor, op="sum")

# Broadcast: send tensor from rank 0 to all
tz.distributed.broadcast(tensor, src=0)

# All-gather: collect tensors from all processes
gathered = tz.distributed.all_gather(tensor)
```

## Launch Script

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train.py

# Or with environment variables
WORLD_SIZE=4 RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500 python train.py
```
