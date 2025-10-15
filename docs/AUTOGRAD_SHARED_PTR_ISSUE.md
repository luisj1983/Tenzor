# Autograd shared_ptr Issue Analysis

## Problem Summary

The `AutogradTest.ChainedOperations` test segfaults due to dangling pointer references when using temporary Variables in chained operations.

## Root Cause

### Current Implementation Flaw

In `Variable::operator*` (and other operators):

```cpp
auto Variable::operator*(const Variable& other) const -> Variable {
    auto grad_fn = std::make_shared<MulBackward>();

    // PROBLEM: Using make_variable_ref on 'this' creates shared_ptr with no-op deleter
    grad_fn->set_input_variables({
        make_variable_ref(const_cast<Variable*>(this)),  // Dangling if 'this' is temporary!
        make_variable_ref(const_cast<Variable*>(&other)) // Dangling if 'other' is temporary!
    });

    // ...
    return output;
}
```

### Why This Fails

Consider the expression: `d = (a + b) * c`

1. **Step 1**: `a + b` executes, creating:
   - Temporary Variable `temp1` (result of addition)
   - `AddBackward` function with shared_ptrs to `a` and `b` (OK - they're stack variables)

2. **Step 2**: `temp1 * c` executes, creating:
   - `MulBackward` function with shared_ptrs to `temp1` and `c`
   - **PROBLEM**: `temp1` is a temporary that will be destroyed after this expression!

3. **Step 3**: Full expression completes
   - `temp1` is destroyed (goes out of scope)
   - `MulBackward` still has shared_ptr to destroyed `temp1` → **DANGLING POINTER**

4. **Step 4**: `d.backward()` executes
   - Engine traverses graph, encounters `MulBackward`
   - Tries to access `temp1` through dangling pointer → **SEGFAULT**

## The Lifetime Problem

```cpp
// Expression: d = (a + b) * c
auto d = (a + b) * c;
         ^^^^^^^
         This temporary is destroyed here!
                  ^
                  But MulBackward still references it!
```

## Solution According to DESIGN.md

The DESIGN.md specifies that Variables should use a **handle pattern** with `shared_ptr<VariableImpl>`:

```cpp
class Variable {
private:
    std::shared_ptr<VariableImpl> impl_;  // Handle to implementation
};
```

**Key Insight**: We should track the `impl_` (which is a shared_ptr) in the autograd graph, not raw pointers to Variables!

### Correct Pattern

Instead of storing `shared_ptr<Variable>` with no-op deleters, we should:

1. **Store Variable handles by value** in Functions (the shared_ptr<VariableImpl> will keep data alive)
2. **OR** store shared_ptr<VariableImpl> directly in Functions

This ensures that even if the Variable handle (temporary) is destroyed, the underlying VariableImpl (and its gradient storage) remains alive because it's reference-counted.

## Proposed Fix

### Option 1: Store Variables by Value

```cpp
class Function {
protected:
    std::vector<Variable> input_variables_;  // Store by value, not shared_ptr
};
```

**Pros**:
- Simple and clean
- Variable's impl_ shared_ptr keeps data alive
- No lifetime issues

**Cons**:
- Copies Variable handles (but impl_ is shared, so cheap)

### Option 2: Store shared_ptr<VariableImpl>

```cpp
class Function {
protected:
    std::vector<std::shared_ptr<VariableImpl>> input_impls_;
};
```

**Pros**:
- Direct access to implementation
- Explicit ownership semantics

**Cons**:
- Need to modify API significantly
- Breaks abstraction (exposes impl)

## Recommended Approach: Option 1

Store Variables by value in Functions. This is:
- **Safe**: impl_ shared_ptr keeps data alive
- **Simple**: Minimal API changes
- **Correct**: Follows handle/body idiom properly

### Changes Required

1. **In `function.hpp`**:
   ```cpp
   class Function {
   protected:
       std::vector<Variable> input_variables_;  // Changed from shared_ptr<Variable>
   };
   ```

2. **In `variable.cpp` operators**:
   ```cpp
   auto Variable::operator*(const Variable& other) const -> Variable {
       auto grad_fn = std::make_shared<MulBackward>();

       // Store Variables by value - impl_ shared_ptr keeps data alive
       grad_fn->set_input_variables({*this, other});

       // ...
   }
   ```

3. **In `engine.cpp`**:
   ```cpp
   for (size_t i = 0; i < input_vars.size(); ++i) {
       Variable& var = input_vars[i];  // Direct reference, not shared_ptr

       if (var.is_leaf() || var.retains_grad()) {
           // Accumulate gradient...
       }
   }
   ```

## Testing Plan

1. Fix `Function` to store Variables by value
2. Update `Variable` operators to use new pattern
3. Update `BackwardEngine` to work with value-based storage
4. Test with `ChainedOperations` (should not segfault)
5. Run full test suite to ensure no regressions

## Expected Outcome

After fix:
- No dangling pointers (impl_ shared_ptr keeps data alive)
- Temporary Variables can be safely destroyed
- Gradient accumulation works correctly for all variables (leaf and intermediate)
- All autograd tests pass
