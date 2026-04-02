# Mixed Precision Training

Train models faster using FP16/BF16 with automatic loss scaling.

## Basic Usage with GradScaler

```python
import tenzor as tz

model = tz.nn.Sequential([
    tz.nn.Linear(784, 256),
    tz.nn.ReLU(),
    tz.nn.Linear(256, 10),
])
optimizer = tz.optim.Adam(model.parameters(), lr=0.001)
scaler = tz.amp.GradScaler()
criterion = tz.nn.CrossEntropyLoss()

for data, targets in dataloader:
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, targets)

    # Scale loss to prevent underflow in FP16 gradients
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()

    # Unscale gradients, check for inf/nan, step optimizer
    scaler.step(optimizer)
    scaler.update()
```

## GradScaler Parameters

- **`init_scale`** (default: 65536.0): Initial loss scale factor.
- **`growth_factor`** (default: 2.0): Scale multiplier after consecutive non-inf steps.
- **`backoff_factor`** (default: 0.5): Scale multiplier when inf/nan detected.
- **`growth_interval`** (default: 2000): Steps between scale increases.

## BFloat16 Tensors

```python
# Create BF16 tensors directly
x = tz.randn([4, 4], dtype=tz.dtype.bfloat16)

# Convert existing tensors
y = tz.randn([4, 4])
y_bf16 = y.to(tz.dtype.bfloat16)
```

## When to Use Mixed Precision

- **FP16**: Best on NVIDIA GPUs with Tensor Cores (Volta+). 2x memory savings.
- **BF16**: Best on Ampere+ GPUs and Intel. Same range as FP32, less precision.
- **FP8**: Available on Hopper+ GPUs for maximum throughput.
