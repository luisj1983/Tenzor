# Quickstart

Get up and running with Tenzor in 5 minutes.

## Creating Tensors

```python
import tenzor as tz
tz.initialize()

# Basic creation
x = tz.zeros([3, 4])           # 3x4 zero tensor
y = tz.randn([3, 4])           # Random normal
z = tz.ones([3, 4])            # All ones
a = tz.arange(0, 10, 1)        # [0, 1, ..., 9]
e = tz.eye(3)                  # 3x3 identity
```

## Tensor Operations

```python
# Arithmetic
c = x + y
d = x @ y.transpose(0, 1)     # Matrix multiply

# Element-wise
s = tz.sqrt(y)
e = tz.exp(y)

# Reductions
total = tz.sum(y)
avg = tz.mean(y, dim=0)
```

## Automatic Differentiation

```python
# Create variables with gradient tracking
x = tz.Variable(tz.randn([3, 4]), requires_grad=True)
w = tz.Variable(tz.randn([4, 2]), requires_grad=True)

# Forward pass
y = tz.matmul(x, w)
loss = tz.mean(y)

# Backward pass
loss.backward()

# Gradients are now available
print(x.grad.shape)  # [3, 4]
print(w.grad.shape)  # [4, 2]
```

## Neural Networks

```python
# Define a model
model = tz.nn.Sequential([
    tz.nn.Linear(784, 128),
    tz.nn.ReLU(),
    tz.nn.Linear(128, 10),
])

# Forward pass
x = tz.Variable(tz.randn([32, 784]))
logits = model(x)

# Loss and backward
criterion = tz.nn.CrossEntropyLoss()
targets = tz.arange(0, 32, 1, dtype=tz.dtype.int64) % 10
loss = criterion(logits, targets)
loss.backward()
```

## Training Loop

```python
optimizer = tz.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: loss = {loss.data.item():.4f}")
```

## Device Management

```python
if tz.cuda_is_available():
    x_gpu = x.data.to("cuda:0")
    print("On GPU:", x_gpu.device)
```
