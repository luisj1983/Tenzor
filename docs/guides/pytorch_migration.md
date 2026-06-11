# PyTorch to Tenzor Migration Guide

This guide helps PyTorch users transition to Tenzor. The API is intentionally similar, but there are key differences to be aware of.

## Import Mapping

| PyTorch | Tenzor |
|---------|--------|
| `import torch` | `import tenzor as tz` |
| `import torch.nn as nn` | `import tenzor.nn as nn` |
| `import torch.nn.functional as F` | `import tenzor.nn.functional as F` |
| `import torch.optim` | `from tenzor import optim` |
| `import torch.autograd` | `from tenzor import autograd` |

## Tensor Creation

| PyTorch | Tenzor | Notes |
|---------|--------|-------|
| `torch.tensor([1, 2, 3])` | `tz.tensor([1, 2, 3])` | |
| `torch.zeros(3, 4)` | `tz.zeros(3, 4)` | List form `tz.zeros([3, 4])` also works |
| `torch.ones(3, 4)` | `tz.ones(3, 4)` | |
| `torch.randn(3, 4)` | `tz.randn(3, 4)` | |
| `torch.empty(3, 4)` | `tz.empty(3, 4)` | Uninitialized |
| `torch.from_numpy(arr)` | `tz.from_numpy(arr)` | Zero-copy |
| `torch.Tensor.from_blob(ptr, shape)` | `tz.Tensor.from_blob(buf, shape)` | Wrap external buffer |

## Key Behavioral Differences

### 1. Variable vs Tensor

Tensor and Variable are merged at the Python level (PyTorch-0.4-style): every
tensor returned by a factory or top-level op IS a Variable with dormant autograd,
so the modern PyTorch idioms port unchanged:

```python
# Both of these work, exactly like PyTorch:
x = tz.randn(3, 3, requires_grad=True)

x = tz.randn(3, 3)
x.requires_grad = True          # settable property, identity preserved

loss = (x * x).sum()            # method-style reductions are autograd-aware
loss.backward()
print(x.grad)

# isinstance checks see one unified type:
isinstance(x, tz.Tensor)        # True
```

Safety contract: a `requires_grad=True` variable refuses to silently flow into an
operation that is not autograd-aware (you get an explicit error instead of a
silently severed graph and zero gradients). Call `.detach()` to leave the graph
intentionally, or wrap the call in `tz.no_grad()` (e.g. in-place parameter init).

The explicit wrapper `tz.Variable(tensor, requires_grad=True)` still works for
raw `Tensor` objects obtained from interop paths.

### 2. Device Specification

Tenzor accepts both `Device` objects and PyTorch-style strings everywhere a device is
expected (`"cpu"`, `"cuda"`, `"cuda:1"`, `"rocm:0"`, `"vulkan"`, `"oneapi"`):

```python
# PyTorch
x = torch.randn(3, 3, device="cuda:0")
x = x.to("cpu")

# Tenzor — identical
x = tz.randn(3, 3, device="cuda:0")
x = x.to("cpu")

# Device objects work too
x = x.to(tz.Device.cuda(0))

# DeviceGuard for multi-GPU context
with tz.DeviceGuard(tz.Device.cuda(1)):
    y = tz.randn([3, 3], device=tz.Device.cuda(1))
```

### 3. Module Subclassing

```python
# PyTorch
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

# Tenzor — identical API
class MyModel(tz.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = tz.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)
```

### 4. Thread Safety

Tenzor requires explicit thread safety opt-in for autograd:

```python
# Tenzor: must call make_thread_safe() for concurrent gradient accumulation
var.make_thread_safe()

# no_grad() is thread-local (does NOT propagate to spawned threads)
with tz.no_grad():
    output = model(input)
```

## API Mapping Table

### Tensor Operations

| PyTorch | Tenzor |
|---------|--------|
| `x + y` | `x + y` |
| `x @ w` | `x @ w` |
| `x.matmul(y)` | `tz.matmul(x, y)` |
| `x.sum()` | `x.sum()` |
| `x.mean(dim=0)` | `x.mean(0)` |
| `x.reshape(3, 4)` | `x.reshape([3, 4])` |
| `x.transpose(0, 1)` | `x.transpose(0, 1)` |
| `x.contiguous()` | `x.contiguous()` |
| `x.clone()` | `x.clone()` |
| `x.detach()` | `x.detach()` |
| `x.numpy()` | `x.numpy()` |
| `x[0:3]` | `x[0:3]` |
| `x.to(device)` | `x.to(device)` |
| `x.to(dtype)` | `x.to(dtype=dtype)` |

### Neural Network Layers

| PyTorch | Tenzor | Notes |
|---------|--------|-------|
| `nn.Linear(in, out)` | `nn.Linear(in, out)` | |
| `nn.Conv2d(in, out, k)` | `nn.Conv2d(in, out, k)` | |
| `nn.BatchNorm2d(n)` | `nn.BatchNorm2d(n)` | |
| `nn.LayerNorm(n)` | `nn.LayerNorm(n)` | |
| `nn.LSTM(in, h)` | `nn.LSTM(in, h)` | |
| `nn.MultiheadAttention(d, h)` | `nn.MultiheadAttention(d, h)` | |
| `nn.Dropout(p)` | `nn.Dropout(p)` | |
| `nn.Embedding(n, d)` | `nn.Embedding(n, d)` | |
| `nn.ReLU()` | `nn.ReLU()` | |
| `nn.Sequential(...)` | `nn.Sequential(...)` | |

### Optimizers

| PyTorch | Tenzor |
|---------|--------|
| `optim.SGD(params, lr)` | `optim.SGD(params, lr)` |
| `optim.Adam(params, lr)` | `optim.Adam(params, lr)` |
| `optim.AdamW(params, lr)` | `optim.AdamW(params, lr)` |
| `optim.RMSprop(params, lr)` | `optim.RMSprop(params, lr)` |

### Loss Functions

| PyTorch | Tenzor |
|---------|--------|
| `nn.CrossEntropyLoss()` | `nn.CrossEntropyLoss()` |
| `nn.MSELoss()` | `nn.MSELoss()` |
| `nn.BCEWithLogitsLoss()` | `nn.BCEWithLogitsLoss()` |
| `nn.L1Loss()` | `nn.L1Loss()` |
| `nn.TripletMarginLoss()` | `nn.TripletMarginLoss()` |

## Training Loop Comparison

```python
# PyTorch                              # Tenzor
model = MyModel()                       model = MyModel()
optimizer = optim.Adam(                 optimizer = optim.Adam(
    model.parameters(), lr=1e-3)            model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()       criterion = nn.CrossEntropyLoss()

for epoch in range(100):                for epoch in range(100):
    optimizer.zero_grad()                   optimizer.zero_grad()
    output = model(input)                   output = model(input)
    loss = criterion(output, target)        loss = criterion(output, target)
    loss.backward()                         loss.backward()
    optimizer.step()                        optimizer.step()
```

## Model Checkpointing

```python
# Save
state = model.state_dict()
model.save("model.pt")

# Load
model.load("model.pt")
# or
model.load_state_dict(state)
```

## Mixed Precision Training

```python
# Tenzor
from tenzor.nn.amp import Autocast, GradScaler

scaler = GradScaler()
with Autocast(enabled=True, dtype=tz.dtype.float16):
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Known Limitations vs PyTorch

1. **No `torch.jit.script`** — Tenzor JIT supports tracing only, not scripting
2. **Sparse tensor autograd** — SpMM/SpMV have backward support, but sparse-sparse ops do not
3. **Distributed training** — Basic DDP available; no FSDP equivalent yet
4. **Ecosystem** — No TorchVision/TorchAudio equivalents yet
5. **Advanced indexing** — Basic slicing works; NumPy-style fancy indexing is limited
6. **No pre-built wheels** — Must compile from source with CMake

## Tenzor Advantages

1. **Faster dispatch** — O(1) array-indexed dispatch (~10-20ns vs PyTorch's ~100-1000ns)
2. **5 backends** — CPU, CUDA, ROCm, OneAPI, Vulkan (PyTorch lacks OneAPI/Vulkan)
3. **Built-in ZeRO** — ZeRO Stage 1/2/3 optimizers included
4. **FP8 native** — FP8 E4M3/E5M2 types built into the dtype system
5. **19 fused kernels** — FusedLinearReLU, FusedAttention, FusedAdamStep, etc.
6. **ONNX export** — Built-in exporter with opset 13+ support
