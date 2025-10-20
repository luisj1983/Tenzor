# EfficientNet Swish Gradient Fix - Exact Code Changes

## Primary Fix: Swish Function (CRITICAL)

### File
`/home/lee/Projects/Tenzor/src/nn/activations/activations.cpp`

### Location
Lines 344-348

### Current (BROKEN) Code
```cpp
auto swish(const Variable& input) -> Variable {
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("swish", inputs)[0];
    return Variable(result, input.requires_grad());
}
```

### Fixed Code (Copy from sigmoid pattern at lines 194-223)
```cpp
auto swish(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = Dispatcher::dispatch("swish", inputs)[0];
        return Variable(result, false);
    }

    // Compute forward
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = Dispatcher::dispatch("swish", inputs_vec)[0];

    // Set up autograd
    auto grad_fn = std::make_shared<SwishBackward>();
    grad_fn->save_for_backward({input.tensor(), result_tensor});

    std::vector<std::shared_ptr<Function>> next_funcs;
    if (input.grad_fn()) {
        next_funcs.push_back(input.grad_fn());
    }
    grad_fn->set_next_functions(next_funcs);

    // Track input variable for gradient accumulation
    std::vector<Variable> input_vars;
    input_vars.push_back(input);
    grad_fn->set_input_variables(input_vars);

    Variable output(result_tensor, true);
    output.set_grad_fn(grad_fn);  // ✓ CRITICAL LINE - ATTACH GRAD FUNCTION
    return output;
}
```

### What Changed
1. Added early return for non-gradient case
2. Created `SwishBackward` function object
3. Called `save_for_backward()` to store tensors
4. Set up `next_functions` to link to input's gradient function
5. Called `set_input_variables()` to track inputs
6. **CRITICAL**: Added `output.set_grad_fn(grad_fn)` to attach the gradient function

### Why This Fixes It
- `output.set_grad_fn(grad_fn)` connects the output Variable to the computation graph
- When `.backward()` is called on a later loss, the gradient can now flow through SwishBackward
- SwishBackward can access saved tensors and compute gradients for the input

---

## Secondary Fixes (Also Required)

### Fix 2: ELU Function
**File**: `/home/lee/Projects/Tenzor/src/nn/activations/activations.cpp`
**Lines**: 322-328

#### Current (BROKEN)
```cpp
auto elu(const Variable& input, double alpha) -> Variable {
    OpAttributes attrs;
    attrs["alpha"] = std::to_string(static_cast<float>(alpha));
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("elu", inputs, attrs)[0];
    return Variable(result, input.requires_grad());
}
```

#### Fixed
```cpp
auto elu(const Variable& input, double alpha) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        OpAttributes attrs;
        attrs["alpha"] = std::to_string(static_cast<float>(alpha));
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = Dispatcher::dispatch("elu", inputs, attrs)[0];
        return Variable(result, false);
    }

    // Compute forward
    OpAttributes attrs;
    attrs["alpha"] = std::to_string(static_cast<float>(alpha));
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = Dispatcher::dispatch("elu", inputs_vec, attrs)[0];

    // Set up autograd
    auto grad_fn = std::make_shared<ELUBackward>(alpha);
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
    output.set_grad_fn(grad_fn);  // ✓ ATTACH GRAD FUNCTION
    return output;
}
```

### Fix 3: SELU Function
**File**: `/home/lee/Projects/Tenzor/src/nn/activations/activations.cpp`
**Lines**: 334-338

#### Current (BROKEN)
```cpp
auto selu(const Variable& input) -> Variable {
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("selu", inputs)[0];
    return Variable(result, input.requires_grad());
}
```

#### Fixed
```cpp
auto selu(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = Dispatcher::dispatch("selu", inputs)[0];
        return Variable(result, false);
    }

    // Compute forward
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = Dispatcher::dispatch("selu", inputs_vec)[0];

    // Set up autograd
    auto grad_fn = std::make_shared<SELUBackward>();
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
    output.set_grad_fn(grad_fn);  // ✓ ATTACH GRAD FUNCTION
    return output;
}
```

### Fix 4: Mish Function
**File**: `/home/lee/Projects/Tenzor/src/nn/activations/activations.cpp`
**Lines**: 354-358

#### Current (BROKEN)
```cpp
auto mish(const Variable& input) -> Variable {
    std::vector<Tensor> inputs = {input.tensor()};
    auto result = Dispatcher::dispatch("mish", inputs)[0];
    return Variable(result, input.requires_grad());
}
```

#### Fixed
```cpp
auto mish(const Variable& input) -> Variable {
    if (!input.requires_grad() || !is_grad_enabled()) {
        std::vector<Tensor> inputs = {input.tensor()};
        auto result = Dispatcher::dispatch("mish", inputs)[0];
        return Variable(result, false);
    }

    // Compute forward
    std::vector<Tensor> inputs_vec = {input.tensor()};
    auto result_tensor = Dispatcher::dispatch("mish", inputs_vec)[0];

    // Set up autograd
    auto grad_fn = std::make_shared<MishBackward>();
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
    output.set_grad_fn(grad_fn);  // ✓ ATTACH GRAD FUNCTION
    return output;
}
```

---

## Backward Function Requirements

### SwishBackward
Need to verify or implement in the same file. Check if it exists around lines 10-130.

**Expected Pattern** (copy from SigmoidBackward or GeLUBackward):
```cpp
class SwishBackward : public Function {
public:
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("SwishBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad_output = grad_outputs[0];
        const auto& input = saved_tensors()[0];
        const auto& output = saved_tensors()[1];  // swish(x)

        // d_swish/dx = swish(x) + sigmoid(x) * (1 - swish(x))
        // This is the derivative of x * sigmoid(x)
        
        // For now, use numerical approximation or reference PyTorch implementation
        // Simplified: approximately output + sigmoid(x) * (1 - output)
        
        std::vector<Tensor> result;
        // Compute gradient (implementation depends on available ops)
        return result;
    }
};
```

Similar for ELUBackward, SELUBackward, MishBackward if they don't exist.

---

## Testing the Fix

After applying all fixes:

```bash
cd /home/lee/Projects/Tenzor/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run gradient flow tests
./bin/test_efficientnet --gtest_filter="*GradientFlow*"

# Expected output:
# [PASSED] SqueezeExcitationGradientFlow
# [PASSED] MBConvBlockGradientFlow
# [PASSED] EfficientNetB0GradientFlow
# [PASSED] EfficientNetB1GradientFlow
# [PASSED] EfficientNetB2GradientFlow
# [PASSED] EfficientNetB7GradientFlow
```

---

## Summary of Changes

**Files Modified**: 1
- `/home/lee/Projects/Tenzor/src/nn/activations/activations.cpp`

**Lines Changed**: ~60 total
- Swish: 5 lines → 30 lines (add autograd setup)
- ELU: 6 lines → 30 lines (add autograd setup)
- SELU: 4 lines → 30 lines (add autograd setup)
- Mish: 4 lines → 30 lines (add autograd setup)

**Critical Change**:
- Every fixed function must end with `output.set_grad_fn(grad_fn);` before returning

**Tests Fixed**: 6+
- All "*GradientFlow*" tests in test_efficientnet.cpp

