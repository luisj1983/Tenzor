# Custom Autograd Functions

Define custom differentiable operations by subclassing `tenzor.autograd.Function`.

## Basic Example

```python
import tenzor as tz

class MyReLU(tz.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return tz.clamp_min(input, 0.0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad = grad_output * (input > 0).to(tz.dtype.float32)
        return (grad,)

# Usage
x = tz.Variable(tz.randn([4, 4]), requires_grad=True)
y = MyReLU.apply(x)
```

## Key Concepts

### Context Object (`ctx`)

The `ctx` object is passed to both `forward` and `backward`:
- **`ctx.save_for_backward(*tensors)`**: Save tensors needed for gradient computation.
- **`ctx.saved_tensors`**: Retrieve saved tensors in backward (returns a tuple).

### Return Values

- **`forward`**: Return a single `Tensor` or a tuple of `Tensor`s.
- **`backward`**: Return a tuple with one gradient per forward input. Use `None` for non-differentiable inputs.

## Multi-Input Example

```python
class WeightedAdd(tz.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, alpha):
        ctx.save_for_backward(alpha)
        return x * alpha + y * (1.0 - alpha)

    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors
        grad_x = grad_output * alpha
        grad_y = grad_output * (1.0 - alpha)
        grad_alpha = None  # alpha is not differentiable
        return (grad_x, grad_y, grad_alpha)
```

## Gradient Checking

Verify your backward implementation numerically:

```python
x = tz.Variable(tz.randn([4]), requires_grad=True)
y = MyReLU.apply(x)
loss = tz.sum(y)
loss.backward()
# Compare x.grad with finite differences
```
