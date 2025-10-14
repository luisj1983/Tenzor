# Gradient Checkpointing Implementation Research

**Date:** 2025-10-13
**Purpose:** Research document on gradient checkpointing implementation patterns from PyTorch and other frameworks to guide the fix for our dangling pointer issue.

---

## Executive Summary

Gradient checkpointing is a memory optimization technique that trades compute for memory by discarding intermediate activations during forward pass and recomputing them during backward pass. The key challenge in C++ implementations is managing the lifetime of recomputed Variables and their associated gradient functions to avoid dangling pointers.

**Key Finding:** The dangling pointer issue occurs because local Variables created during recomputation go out of scope before their gradients can be extracted. PyTorch solves this through careful reference counting with `SavedVariable` and hook mechanisms that ensure tensors remain alive throughout the backward pass.

---

## 1. PyTorch's Implementation Strategy

### 1.1 Core Mechanism

PyTorch's `torch.utils.checkpoint` implements gradient checkpointing through two main variants:

#### **Reentrant Checkpoint** (Legacy)
- Runs forward pass under `torch.no_grad()` during recomputation
- Does not record autograd graph during recomputation
- Simpler but less flexible

#### **Non-Reentrant Checkpoint** (Recommended)
- Records autograd graph during recomputation
- Uses `SavedVariable` hooks to manage tensor lifetimes
- Supports `torch.autograd.grad` and keyword arguments
- More robust for complex use cases

### 1.2 CheckpointFunction Architecture

```python
class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        # 1. Save the function and input arguments
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state

        # 2. Save input tensors for recomputation
        ctx.save_for_backward(*args)

        # 3. Run forward without gradient tracking
        with torch.no_grad():
            outputs = run_function(*args)

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        # 1. Restore RNG state for deterministic recomputation
        # 2. Retrieve saved inputs
        inputs = ctx.saved_tensors

        # 3. Detach inputs to create new computation graph
        inputs = [inp.detach().requires_grad_(True) for inp in inputs]

        # 4. Recompute forward with gradient tracking
        with torch.enable_grad():
            outputs = ctx.run_function(*inputs)

        # 5. Perform backward on recomputed outputs
        torch.autograd.backward(outputs, grad_outputs)

        # 6. Extract gradients from inputs
        grads = tuple(inp.grad if inp.grad is not None else None
                      for inp in inputs)

        return (None, None) + grads  # None for non-tensor ctx arguments
```

### 1.3 Key Design Principles

1. **Detachment**: Inputs are detached before recomputation to break any link to the original computation graph
2. **Fresh Graph**: Recomputation creates an entirely new computation graph
3. **Gradient Extraction**: Gradients are extracted from leaf variables (recomputed inputs) before they go out of scope
4. **RNG Determinism**: RNG states are saved and restored to ensure deterministic recomputation

---

## 2. Variable Lifetime Management in C++

### 2.1 The Dangling Pointer Problem

**Root Cause:** In C++, when Variables are created as local variables during recomputation, their destructors run when they go out of scope. However, if gradient functions hold raw pointers to these Variables, those pointers become dangling.

```cpp
// PROBLEMATIC PATTERN
auto CheckpointFunction::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {

    // Create local Variables for recomputation
    std::vector<Variable> inputs;
    for (const auto& tensor : saved_tensors()) {
        inputs.emplace_back(tensor, true);  // Local Variables!
    }

    // Recompute forward - creates grad_fns with raw pointers to inputs
    auto outputs = recompute_forward(inputs);

    // Call backward on outputs
    outputs[0].backward(grad_outputs[0]);
    // ^ BUG: grad_fns try to access inputs, which may be destroyed

    // Extract gradients
    return extract_grads(inputs);  // Dangling pointer access!
}
```

### 2.2 PyTorch's C++ Solution: SavedVariable

PyTorch uses a specialized `SavedVariable` class that:

1. **Reference Counting**: Uses `shared_ptr<Function>` to keep gradient functions alive
2. **Lifetime Linking**: Links tensor buffer lifetime directly to SavedVariable object
3. **Hook Mechanism**: Provides pack/unpack hooks for custom memory management

```cpp
// PyTorch's SavedVariable pattern
class SavedVariable {
    Tensor tensor_;  // Keeps tensor data alive
    std::shared_ptr<Node> grad_fn_;  // Keeps gradient function alive
    std::unique_ptr<SavedVariableHooks> hooks_;  // Custom lifetime management

    // Unpacking ensures tensor is valid when accessed
    Tensor unpack() {
        if (hooks_) {
            return hooks_->unpack(tensor_);
        }
        return tensor_;
    }
};
```

**Key Invariant:** Buffers are deleted if and only if the SavedVariable is cleared, providing deterministic lifetime management.

### 2.3 Reference Counting Strategy

Variables in autograd systems should use `shared_ptr` for:
- **Gradient Functions**: `shared_ptr<Function>` ensures functions outlive backward pass
- **Intermediate Tensors**: Keep tensor data alive through reference counting
- **Input Variables**: Maintain references to ensure they survive gradient accumulation

---

## 3. Correct Backward Pass Implementation

### 3.1 Manual Topological Execution

The safest approach is to manually execute the backward pass without calling `Variable::backward()`, which can trigger premature cleanup.

```cpp
auto CheckpointFunction::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {

    // STEP 1: Create recomputed Variables and KEEP THEM ALIVE
    std::vector<Variable> inputs;
    inputs.reserve(saved_tensors().size());
    for (const auto& tensor : saved_tensors()) {
        inputs.emplace_back(tensor, true);
    }

    // STEP 2: Recompute forward - builds new computation graph
    auto outputs = recompute_forward(inputs);

    // STEP 3: Collect all Functions in topological order
    // This ensures we process in correct dependency order
    std::vector<std::shared_ptr<Function>> topo_sorted_fns =
        collect_topologically(outputs);

    // STEP 4: Manual backward execution
    std::unordered_map<Function*, std::vector<Tensor>> grad_map;

    // Initialize output gradients
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (outputs[i].grad_fn()) {
            grad_map[outputs[i].grad_fn().get()].push_back(grad_outputs[i]);
        }
    }

    // STEP 5: Execute backward in reverse topological order
    for (auto it = topo_sorted_fns.rbegin();
         it != topo_sorted_fns.rend(); ++it) {
        auto& fn = *it;

        // Get accumulated gradients for this function
        if (!grad_map.count(fn.get())) continue;
        auto& grads = grad_map[fn.get()];

        // Sum accumulated gradients
        Tensor grad = sum_gradients(grads);

        // Compute input gradients
        auto input_grads = fn->backward({grad});

        // STEP 6: Accumulate to leaf variables (inputs)
        const auto& input_vars = fn->input_variables();
        for (size_t i = 0; i < input_grads.size(); ++i) {
            if (i < input_vars.size() && input_vars[i]) {
                if (input_vars[i]->requires_grad() &&
                    input_vars[i]->is_leaf()) {
                    accumulate_grad(input_vars[i], input_grads[i]);
                }
            }

            // Forward gradients to next functions
            const auto& next_fns = fn->next_functions();
            if (i < next_fns.size() && next_fns[i]) {
                grad_map[next_fns[i].get()].push_back(input_grads[i]);
            }
        }
    }

    // STEP 7: Extract gradients BEFORE inputs go out of scope
    std::vector<Tensor> result_grads;
    result_grads.reserve(inputs.size());
    for (const auto& input : inputs) {
        if (input.has_grad()) {
            result_grads.push_back(*input.grad());
        } else {
            result_grads.push_back(Tensor::zeros_like(input.tensor()));
        }
    }

    // Now safe to let inputs go out of scope
    return result_grads;
}
```

### 3.2 Critical Implementation Details

**Rule 1: Keep Variables Alive**
- Store recomputed Variables at the top of backward() function scope
- Ensure they survive until after gradient extraction

**Rule 2: Manual Topological Sort**
- Don't rely on automatic backward traversal
- Build explicit execution order for all functions

**Rule 3: Extract Gradients Before Cleanup**
- Copy gradient tensors before Variables are destroyed
- Use `Tensor::clone()` or copy constructor if needed

**Rule 4: No Variable::backward() in Recomputed Graph**
- Calling `Variable::backward()` can trigger cleanup hooks
- Use manual function execution instead

---

## 4. Alternative Approaches

### 4.1 Weak Pointer Approach

Use weak pointers in gradient functions to gracefully handle destroyed variables:

```cpp
class Function {
    // Store weak references to input variables
    std::vector<std::weak_ptr<Variable>> input_var_refs_;

    auto backward(std::vector<Tensor> grad_outputs)
        -> std::vector<Tensor> {
        // Check if variables still exist
        for (const auto& weak_var : input_var_refs_) {
            if (auto var = weak_var.lock()) {
                // Variable still alive, accumulate gradient
                accumulate_grad(var, grad);
            }
            // If expired, skip - this is expected in checkpointing
        }
    }
};
```

**Pros:** Graceful degradation
**Cons:** Requires architecture changes, may hide bugs

### 4.2 Retained Graph Approach

Keep the recomputed graph alive by storing shared_ptrs:

```cpp
class CheckpointFunction : public Function {
private:
    // Store recomputed graph to keep it alive
    std::vector<std::shared_ptr<Variable>> retained_inputs_;
    std::vector<std::shared_ptr<Function>> retained_functions_;

    auto backward(std::vector<Tensor> grad_outputs)
        -> std::vector<Tensor> {
        // Store in member variables to extend lifetime
        for (auto& input : create_inputs()) {
            retained_inputs_.push_back(
                std::make_shared<Variable>(std::move(input))
            );
        }

        // Recompute and retain functions
        auto outputs = recompute_forward_with_retention();

        // Normal backward is now safe
        outputs[0].backward(grad_outputs[0]);

        // Extract gradients
        auto grads = extract_from_retained();

        // Clear retained references
        retained_inputs_.clear();
        retained_functions_.clear();

        return grads;
    }
};
```

**Pros:** Simpler logic, uses existing backward mechanism
**Cons:** Higher memory usage, careful lifecycle management needed

### 4.3 PyTorch's Actual Approach (Non-Reentrant)

PyTorch's production implementation uses saved tensor hooks:

1. **Pack Hook**: Called when tensor is saved, returns a packed representation
2. **Unpack Hook**: Called when tensor is needed, reconstructs tensor
3. **Holder Pattern**: Uses a Holder object to manage multiple SavedVariables

```cpp
// Conceptual pseudocode
class CheckpointHolder {
    std::vector<Tensor> saved_tensors_;
    std::function<std::vector<Tensor>(std::vector<Tensor>)> recompute_fn_;

    auto pack(const Tensor& tensor) -> PackedTensor {
        // Store minimal info, discard tensor data
        return {tensor.metadata(), /*discard data*/};
    }

    auto unpack(const PackedTensor& packed) -> Tensor {
        // Recompute if needed
        if (!cached_recomputed_) {
            cached_recomputed_ = recompute_fn_(saved_tensors_);
        }
        // Return recomputed tensor
        return find_tensor_by_metadata(packed.metadata);
    }
};
```

---

## 5. Recommendations for Tenzor Implementation

### 5.1 Immediate Fix (Current Implementation)

Your current implementation in `src/autograd/checkpoint.cpp` (lines 124-210) is on the right track:

✅ **Correct Approach:**
- Manual topological sorting of functions
- Explicit backward execution without Variable::backward()
- Gradient extraction before Variables go out of scope

⚠️ **Potential Issues to Verify:**

1. **Input Variable Pointers**: Ensure `fn->input_variables()` returns stable pointers
   ```cpp
   // If input_variables() returns raw pointers, verify they point to
   // the correct Variables (the recomputed inputs, not original inputs)
   ```

2. **Function Retention**: Verify Functions stay alive during backward pass
   ```cpp
   // The shared_ptr<Function> in outputs[i].grad_fn() should keep
   // functions alive during the entire loop
   ```

3. **Gradient Accumulation Order**: Ensure gradients are accumulated correctly
   ```cpp
   // For leaf variables, gradients should accumulate
   // For non-leaf variables, gradients should propagate to next_functions
   ```

### 5.2 Recommended Enhancements

#### **Enhancement 1: Add Lifetime Guards**

```cpp
auto CheckpointFunction::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {

    // Lifetime guard: Keep all shared_ptrs alive explicitly
    struct LifetimeGuard {
        std::vector<Variable> inputs;
        std::vector<Variable> outputs;
        std::vector<std::shared_ptr<Function>> functions;
    } guard;

    // Store everything in guard
    for (const auto& tensor : saved_tensors()) {
        guard.inputs.emplace_back(tensor, true);
    }

    guard.outputs = recompute_forward(guard.inputs);

    // Collect all functions
    for (const auto& output : guard.outputs) {
        collect_all_functions(output, guard.functions);
    }

    // Perform backward with everything safely retained
    auto grads = manual_backward(guard);

    // Extract results before guard is destroyed
    return grads;
}
```

#### **Enhancement 2: Add Validation**

```cpp
// Validate pointer stability during backward
void validate_pointers(const std::vector<Variable>& vars) {
    for (const auto& var : vars) {
        if (var.grad_fn()) {
            const auto& input_vars = var.grad_fn()->input_variables();
            for (const auto* var_ptr : input_vars) {
                assert(var_ptr != nullptr &&
                       "Dangling pointer detected!");
            }
        }
    }
}
```

#### **Enhancement 3: Consider Shared Pointer Storage**

Modify `Function::set_input_variables()` to store shared pointers instead of raw pointers:

```cpp
class Function {
private:
    // Change from raw pointers to shared_ptr
    std::vector<std::weak_ptr<Variable>> input_variables_;

public:
    auto set_input_variables(
        std::vector<std::shared_ptr<Variable>> vars) -> void {
        input_variables_.clear();
        for (auto& var : vars) {
            input_variables_.push_back(var);  // Store weak_ptr
        }
    }

    auto input_variables() const
        -> std::vector<std::shared_ptr<Variable>> {
        std::vector<std::shared_ptr<Variable>> result;
        for (const auto& weak_var : input_variables_) {
            if (auto shared_var = weak_var.lock()) {
                result.push_back(shared_var);
            }
        }
        return result;
    }
};
```

**Benefit:** Automatic lifetime management through reference counting
**Cost:** Requires API changes throughout the codebase

### 5.3 Testing Strategy

Create specific tests for the dangling pointer scenario:

```cpp
TEST(CheckpointTest, RecomputationLifetime) {
    // Create checkpoint function
    auto fn = [](std::vector<Variable> inputs) {
        auto hidden = relu(matmul(inputs[0], weight));
        return std::vector<Variable>{hidden};
    };

    Variable input(Tensor::randn({2, 3}), true);

    // Checkpoint forward
    auto output = checkpoint(fn, {input});

    // Force backward - this should not crash
    output[0].backward(Tensor::ones_like(output[0].tensor()));

    // Verify gradients computed correctly
    ASSERT_TRUE(input.has_grad());
    ASSERT_EQ(input.grad()->shape(), input.tensor().shape());
}

TEST(CheckpointTest, NestedCheckpoints) {
    // Test nested checkpointing to stress-test lifetime management
    auto layer1 = [](Variable x) { return relu(linear(x)); };
    auto layer2 = [](Variable x) { return relu(linear(x)); };

    Variable x(Tensor::randn({2, 10}), true);

    // Nested checkpoints
    auto h1 = checkpoint(layer1, x);
    auto h2 = checkpoint(layer2, h1);

    h2.backward();

    ASSERT_TRUE(x.has_grad());
}
```

---

## 6. Key Insights from PyTorch

### 6.1 The "Recomputed Tensors Are Transient" Rule

PyTorch documents this as "Rule 4":

> Recomputed tensors are considered specific to particular invocations of backward and are always cleared immediately as they are unpacked. Particularly, this is required to happen even if retain_graph=True.

**Implication:** Recomputed tensors have a very short lifetime by design. This is intentional to save memory.

### 6.2 The Weak Reference Architecture

PyTorch's implementation uses weak references extensively:

```python
# Conceptual pattern
class _CheckpointFrame:
    weak_holders: List[ReferenceType] = []

    # Store as WeakKeyDictionary so entries are cleared
    # automatically when SavedVariable is destroyed
    storage_dict = WeakKeyDictionary()
```

This allows graceful handling of destroyed variables without crashes.

### 6.3 The Hook-Based Approach

PyTorch's production-ready non-reentrant implementation uses hooks rather than trying to keep everything alive:

1. **Pack**: Discard tensor data, keep only metadata
2. **Unpack**: Recompute on-demand when needed
3. **Benefit**: Minimal memory footprint, on-demand recomputation

---

## 7. Common Pitfalls to Avoid

### Pitfall 1: Calling Variable::backward() on Recomputed Graph

```cpp
// DON'T DO THIS
auto outputs = recompute_forward(inputs);
outputs[0].backward(grad_outputs[0]);  // Can cause dangling pointers!
```

**Why:** `Variable::backward()` may trigger cleanup of the computation graph, causing Variables to be destroyed while still being accessed.

### Pitfall 2: Storing Raw Pointers to Local Variables

```cpp
// DON'T DO THIS
std::vector<Variable*> input_ptrs;
for (const auto& var : local_variables) {
    input_ptrs.push_back(&var);  // Pointer to local!
}
// local_variables destroyed here!
// input_ptrs now contain dangling pointers
```

### Pitfall 3: Not Extracting Gradients Before Cleanup

```cpp
// DON'T DO THIS
{
    std::vector<Variable> inputs = create_inputs();
    perform_backward(inputs);
}  // inputs destroyed here!

// Trying to access gradients now would be wrong
// Extract gradients BEFORE the closing brace
```

### Pitfall 4: Relying on Automatic Graph Traversal

```cpp
// DON'T DO THIS
// Assume backward engine will handle everything correctly
backward_engine().execute(recomputed_output, grad);
// May not work correctly with local Variables
```

**Better:** Manual topological execution with explicit control.

---

## 8. Conclusion

### Summary of Key Principles

1. **Lifetime Management is Critical**: Variables must survive until gradients are extracted
2. **Manual Control is Safer**: Explicit topological execution avoids automatic cleanup issues
3. **Reference Counting Helps**: shared_ptr ensures objects stay alive when needed
4. **Extract Before Destroy**: Always copy gradients before Variables go out of scope
5. **Test Thoroughly**: Dangling pointers can be subtle and intermittent

### Recommended Path Forward

**Short Term:**
- Your current manual backward implementation is correct
- Add validation checks to detect pointer issues early
- Add comprehensive tests for lifetime scenarios

**Medium Term:**
- Consider adding shared_ptr storage for input variables
- Implement weak_ptr pattern for graceful degradation
- Add memory profiling to verify cleanup

**Long Term:**
- Consider implementing PyTorch-style hook mechanism
- Evaluate retained graph approach for simpler code
- Benchmark different approaches for performance

---

## 9. References

### Academic Papers
- **Training Deep Nets with Sublinear Memory Cost** (Chen et al., 2016)
  Original gradient checkpointing paper, describes O(√n) memory scaling

### PyTorch Documentation
- `torch.utils.checkpoint` - Official documentation
- Autograd mechanics - Detailed explanation of backward pass
- Saved tensor hooks tutorial - Advanced memory management

### PyTorch Source Code
- `torch/utils/checkpoint.py` - CheckpointFunction implementation
- `torch/autograd/function.py` - Base Function class
- C++ autograd implementation - SavedVariable, Node structures

### Key Concepts
- **Activation Checkpointing**: Trading compute for memory
- **Recomputation**: Running forward pass again during backward
- **SavedVariable**: PyTorch's mechanism for managing saved tensors
- **Topological Sorting**: Ordering operations for correct gradient flow
- **Reference Counting**: shared_ptr-based lifetime management

---

## Appendix A: Comparison of Approaches

| Approach | Memory | Complexity | Safety | Performance |
|----------|--------|------------|--------|-------------|
| Manual Backward (Current) | Low | High | High* | Good |
| Retained Graph | Medium | Low | High | Good |
| Weak Pointers | Low | Medium | Medium | Good |
| PyTorch Hooks | Very Low | Very High | Very High | Excellent |

*High safety if implemented correctly

---

## Appendix B: Debugging Dangling Pointers

### Techniques to Detect Issues

1. **Address Sanitizer (ASan)**
   ```bash
   cmake -DCMAKE_CXX_FLAGS="-fsanitize=address" ..
   ./test_checkpoint
   ```

2. **Valgrind**
   ```bash
   valgrind --leak-check=full --track-origins=yes ./test_checkpoint
   ```

3. **Debug Assertions**
   ```cpp
   #ifdef DEBUG
   #define ASSERT_VALID_PTR(ptr) \
       assert(ptr != nullptr && "Dangling pointer!");
   #else
   #define ASSERT_VALID_PTR(ptr)
   #endif
   ```

4. **Logging**
   ```cpp
   std::cout << "Variable address: " << &var << std::endl;
   std::cout << "Function input_var[0]: "
             << fn->input_variables()[0] << std::endl;
   // Compare addresses to verify stability
   ```

---

**End of Research Document**
