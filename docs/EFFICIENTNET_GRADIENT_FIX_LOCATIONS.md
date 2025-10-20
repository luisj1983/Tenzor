# EfficientNet Gradient Chain Break - Specific Fix Locations

## Quick Reference

| Issue | File | Line(s) | Pattern | Severity |
|-------|------|---------|---------|----------|
| **Swish activation** | src/nn/activations/activations.cpp | 344-348 | `Variable(result, input.requires_grad())` without grad_fn | CRITICAL |
| **Sigmoid activation** | src/nn/activations/activations.cpp | 194-223 | ✓ CORRECT - has all autograd setup | OK |
| **ELU activation** | src/nn/activations/activations.cpp | 322-328 | `Variable(result, input.requires_grad())` without grad_fn | HIGH |
| **SELU activation** | src/nn/activations/activations.cpp | 334-338 | `Variable(result, input.requires_grad())` without grad_fn | HIGH |
| **Mish activation** | src/nn/activations/activations.cpp | 354-358 | `Variable(result, input.requires_grad())` without grad_fn | CRITICAL |

## Root Cause Details

### Pattern 1: Broken Swish (Lines 344-348)

**Current Code:**
```cpp
auto swish(const Variable& input) -> Variable {
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("swish", inputs)[0];
    return Variable(result, input.requires_grad());  // ❌ NO GRAD_FN
}
```

**Problem:**
- `Variable(result, input.requires_grad())` creates a Variable with `grad_fn = nullptr`
- No backward function is attached
- Gradients cannot flow back to input

**Used In EfficientNet:**
- Line 359 in efficientnet.cpp: `x = stem_swish_.forward(x);`
- Line 220 in efficientnet.cpp: `x = swish_.forward(x);` (in MBConvBlock expansion)
- Line 226 in efficientnet.cpp: `x = swish_.forward(x);` (in MBConvBlock depthwise)

### Pattern 2: Broken Sigmoid (Indirectly)

**Current Code (lines 194-223):**
The main sigmoid() function IS CORRECT. However, it's NOT actually being used in a backward-compatible way in some paths.

**Issue Location in SE Module (line 154 in efficientnet.cpp):**
```cpp
auto scale = sigmoid_.forward(expanded);
```

This calls the Sigmoid module which delegates to the correct sigmoid() function, so it works.

### Pattern 3: Broken ELU (Lines 322-328)

**Current Code:**
```cpp
auto elu(const Variable& input, double alpha) -> Variable {
    OpAttributes attrs;
    attrs["alpha"] = std::to_string(static_cast<float>(alpha));
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("elu", inputs, attrs)[0];
    return Variable(result, input.requires_grad());  // ❌ NO GRAD_FN
}
```

**Problem:** Same as swish - no grad_fn attached.

### Pattern 4: Broken SELU (Lines 334-338)

**Current Code:**
```cpp
auto selu(const Variable& input) -> Variable {
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("selu", inputs)[0];
    return Variable(result, input.requires_grad());  // ❌ NO GRAD_FN
}
```

**Problem:** Same as swish - no grad_fn attached.

### Pattern 5: Broken Mish (Lines 354-358)

**Current Code:**
```cpp
auto mish(const Variable& input) -> Variable {
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("mish", inputs)[0];
    return Variable(result, input.requires_grad());  // ❌ NO GRAD_FN
}
```

**Problem:** Same as swish - no grad_fn attached.

## Working Pattern Reference

All of these DO work correctly:

### ReLU (Lines ~160-192)
```cpp
auto relu(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = Dispatcher::dispatch("relu", inputs)[0];
        return Variable(result, false);
    }

    // Compute forward
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = Dispatcher::dispatch("relu", inputs_vec)[0];

    // Set up autograd
    auto grad_fn = std::make_shared<ReLUBackward>();
    grad_fn->save_for_backward({input.tensor(), result_tensor});

    std::vector<std::shared_ptr<Function>> next_funcs;
    if (input.grad_fn()) {
        next_funcs.push_back(input.grad_fn());
    }
    grad_fn->set_next_functions(next_funcs);

    // Track input variable
    std::vector<Variable> input_vars;
    input_vars.push_back(input);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);  // ✓ GRAD_FN ATTACHED
    return output;
}
```

### Sigmoid (Lines 194-223)
```cpp
auto sigmoid(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = Dispatcher::dispatch("sigmoid", inputs)[0];
        return Variable(result, false);
    }

    // Compute forward
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = Dispatcher::dispatch("sigmoid", inputs_vec)[0];

    // Set up autograd
    auto grad_fn = std::make_shared<SigmoidBackward>();
    grad_fn->save_for_backward({result_tensor});

    std::vector<std::shared_ptr<Function>> next_funcs;
    if (input.grad_fn()) {
        next_funcs.push_back(input.grad_fn());
    }
    grad_fn->set_next_functions(next_funcs);

    // Track input variable
    std::vector<Variable> input_vars;
    input_vars.push_back(input);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);  // ✓ GRAD_FN ATTACHED
    return output;
}
```

### GELU (Similar structure - all working correctly)

## Test Case That Fails

### File: tests/unit/test_efficientnet.cpp

**Line 41-55: SqueezeExcitationGradientFlow**
```cpp
TEST_F(EfficientNetTest, SqueezeExcitationGradientFlow) {
    auto se = std::make_shared<EfficientNetSqueezeExcitation>(32, 0.25);

    Variable input(Tensor({1, 32, 7, 7}, DType::Float32, device_), true);
    Variable output = se->forward(input);
    Variable loss = tenzor::sum(output * output);
    loss.backward();

    EXPECT_TRUE(input.grad().has_value());  // ✗ FAILS
    auto params = se->parameters();
    EXPECT_GT(params.size(), 0);
    for (const auto& param : params) {
        EXPECT_TRUE(param->grad().has_value());
    }
}
```

**Why it fails:**
- SE module has sigmoid inside (line 154): `auto scale = sigmoid_.forward(expanded);`
- Sigmoid is correct, but the multiplication `input * scale` needs proper gradient chain
- Actually, the real issue is in how sigmoid is used in the SE module

Wait, let me recheck the SE module...

### Actually: SE Module Uses Module Interface (Lines 144-158 in efficientnet.cpp)
```cpp
auto EfficientNetSqueezeExcitation::forward(const Variable& input) -> Variable {
    auto pooled = pool_->forward(input);
    auto reduced = fc1_->forward(pooled);
    reduced = swish_.forward(reduced);  // Uses swish_ which calls broken swish()
    auto expanded = fc2_->forward(reduced);
    auto scale = sigmoid_.forward(expanded);  // Uses sigmoid_ module
    return input * scale;
}
```

The real issue:
- `swish_.forward(reduced)` - calling Swish module
- Swish module calls the broken `swish()` function from activations.cpp line 341:
  ```cpp
  auto Swish::forward(const Variable& input) -> Variable {
      return swish(input);  // Delegates to broken swish() at line 344-348
  }
  ```

## Call Chain

```
EfficientNetSqueezeExcitation::forward()
    ↓
swish_.forward() [Swish module, line 341]
    ↓
swish(input) [BROKEN function, line 344-348]
    ↓
return Variable(result, input.requires_grad());  // grad_fn = nullptr!
    ↓
gradient chain breaks
```

## Impact Summary

### Direct EfficientNet Impact
1. **Stem**: Uses `stem_swish_` → broken swish()
2. **MBConvBlock expansion**: Uses `swish_.forward()` → broken swish()
3. **MBConvBlock depthwise**: Uses `swish_.forward()` → broken swish()
4. **MBConvBlock SE module**: Uses `swish_.forward()` → broken swish()
5. **MBConvBlock SE module**: Uses `sigmoid_.forward()` → currently OK (sigmoid is correct)
6. **Head**: Uses `head_swish_` → broken swish()

### Result
With 16+ layers × (1-3 swish activations per layer) = 20-48 broken activations per forward pass, the gradient chain is completely severed.

## Cascade of Failures Expected

Once swish() is fixed, these tests should pass:
- `SqueezeExcitationGradientFlow` (line 41)
- `MBConvBlockGradientFlow` (line 86)
- `EfficientNetB0GradientFlow` (line 124)
- `EfficientNetB1GradientFlow` (line 172)
- `EfficientNetB2GradientFlow` (line 199)
- `EfficientNetB7GradientFlow` (line 298)

All other B variants would also pass.

## Neighboring Correct Functions

For reference, these activation functions in the same file ARE implemented correctly:

| Function | Lines | Status | Reason |
|----------|-------|--------|--------|
| ReLU | ~160-192 | ✓ CORRECT | Has full autograd setup |
| Sigmoid | 194-223 | ✓ CORRECT | Has full autograd setup |
| Tanh | ~225-255 | ✓ CORRECT | Has full autograd setup |
| GELU | ~275-305 | ✓ CORRECT | Has full autograd setup |
| LeakyReLU | ~145-158 | ✓ CORRECT | Has full autograd setup |

