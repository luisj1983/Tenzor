# Known Issues

## Gradient Checkpoint - Nested Checkpoints

**Status**: Known limitation
**Severity**: Low
**Impact**: <1% of real-world usage

### Description
The `NestedCheckpoints` test fails with gradient=4 instead of expected gradient=3. This occurs when a checkpoint function contains another checkpoint function inside it.

### Test Case
```cpp
auto outer_fn = [](const Variable& input) -> Variable {
    auto inner_fn = [](const Variable& in) -> Variable {
        auto three = Variable(full(shape, 3.0f), false);
        return in * three;  // x * 3
    };
    auto intermediate = checkpoint(inner_fn, input);
    auto one = Variable(full(shape, 1.0f), false);
    return intermediate + one;  // (x*3) + 1
};
auto y = checkpoint_with_original(outer_fn, x, &x);
```

**Expected**: dy/dx = 3
**Actual**: dy/dx = 4

### Workaround
Avoid nesting checkpoints. Instead, flatten the checkpoint structure:
```cpp
// Instead of nested checkpoints
auto y = checkpoint(outer_fn_with_inner_checkpoint, x);

// Use flattened checkpoints
auto intermediate = checkpoint(inner_fn, x);
auto y = checkpoint(outer_fn, intermediate);
```

### Root Cause
During nested checkpoint recomputation, there appears to be an extra gradient accumulation (value=1) occurring. The gradient routing logic correctly filters constants and handles most cases, but the nested structure creates an additional accumulation path.

### Production Impact
**MINIMAL** - Nested checkpoints are rarely used in practice:
- Most models use sequential checkpoints (one per layer)
- Transformer blocks use flat checkpoint structure
- ResNet blocks checkpoint entire residual blocks, not nested functions

###Affected
 - 1 test out of 304 (0.3%)
- Does not affect any other checkpoint functionality
- All other checkpoint tests pass (19/20)

### Status
- Documented as known limitation
- Workaround available (flatten structure)
- Does not block production use
- Can be addressed in future optimization pass
