# Critical Bug: Dangling Pointers in Autograd System

## Summary

Discovered a critical bug in the autograd system where `Function::input_variables_` stores raw `Variable*` pointers that become dangling when Variables are moved or destroyed, causing segmentation faults during backward pass.

## Root Cause

**Location**: `/home/lee/Projects/Tenzor/include/tenzor/autograd/function.hpp:146`

```cpp
std::vector<Variable*> input_variables_;  // RAW POINTERS - DANGEROUS!
```

When Variables are:
- Moved (std::move)
- Stored in containers that reallocate (std::vector)
- Destroyed while Functions still reference them

The raw pointers become invalid, leading to crashes when accessing Variable members like `hooks_`, `requires_grad()`, `grad()`, etc.

## How the Bug Was Discovered

### Test Case
`TransformerIntegrationTest.ForwardBackward` - crashed with SIGSEGV during backward pass

### Debugging Process

1. **Initial symptoms**: Crash after processing function 3 of 315 in backward pass
2. **Iteration test**: Confirmed all 315 function pointers are valid (non-null)
3. **Detailed instrumentation**: Added debug output to pinpoint exact crash location
4. **Critical finding**: Function 3, input_vars[0] had `hooks_.size() = 223181891085520640`
   - This massive garbage value indicated memory corruption
   - Function 2's input_vars[0] had `hooks_.size() = 0` (normal)
   - The Variable object was being accessed after it was moved/corrupted

### Exact Crash Location

```cpp
// src/autograd/engine.cpp:87
auto& var = *input_vars[i];  // Dereferencing dangling pointer

// Accessing any member causes crash:
for (auto& hook : var.hooks_) {  // hooks_ contains garbage
    grad_to_apply = hook(grad_to_apply);
}
```

### Debug Output at Crash

```
Processing function 2/315 (LayerNormBackward)
  Accumulating to input vars... [0]{hooks:0}  ← Normal

Processing function 3/315 (LayerNormBackward)
  Accumulating to input vars... [0]{hooks:223181891085520640}  ← CORRUPTED!
  [SEGFAULT]
```

## Temporary Workaround (Applied)

**File**: `/home/lee/Projects/Tenzor/src/autograd/engine.cpp:82-102`

Commented out all access to `input_variables_` to prevent crash:

```cpp
// BUGFIX: Skip input_variables accumulation - Variable* pointers may be dangling
// The issue is that Function stores raw Variable* pointers which become invalid
// when Variables are moved or destroyed. This needs a proper fix using shared_ptr.
// For now, gradient accumulation will happen through next_functions only.
std::cout << "  Skipping input_variables (dangling pointer fix)" << std::endl;
```

### Workaround Impact

✅ **Fixed**: Backward pass no longer crashes - completes all 315 functions
❌ **Broken**: Leaf variables don't receive gradients
  - `TransformerIntegrationTest.ForwardBackward` fails:
    - `src.has_grad()` returns false (expected true)
    - `tgt.has_grad()` returns false (expected true)

## Proper Fix Required

Change `Function::input_variables_` from raw pointers to smart pointers:

### Option 1: Use std::weak_ptr (Recommended)
```cpp
// function.hpp:146
std::vector<std::weak_ptr<Variable>> input_variables_;
```

**Pros**:
- Doesn't prevent Variable destruction
- Can detect if Variable was destroyed (weak_ptr::expired())
- Proper ownership semantics

**Cons**:
- Requires Variables to be stored in shared_ptr
- Need to check expired() before dereferencing

### Option 2: Use std::shared_ptr
```cpp
std::vector<std::shared_ptr<Variable>> input_variables_;
```

**Pros**:
- Prevents premature Variable destruction
- Simpler to use (no expired() checks needed)

**Cons**:
- May cause circular references if not careful
- Variables live longer than needed

### Implementation Steps

1. **Update Function class**:
   ```cpp
   class Function {
       std::vector<std::weak_ptr<Variable>> input_variables_;

       auto input_variables() const -> std::vector<std::weak_ptr<Variable>> {
           return input_variables_;
       }
   };
   ```

2. **Update Variable class** to be managed by shared_ptr

3. **Update all Function subclasses** that store Variables

4. **Update BackwardEngine::execute**:
   ```cpp
   for (auto& weak_var : input_vars) {
       if (auto var = weak_var.lock()) {  // Check if still alive
           // Accumulate gradient...
       }
   }
   ```

5. **Run full test suite** to ensure no regressions

## Test Results

### Before Fix
- 849/853 tests passing (99.5%)
- `TransformerIntegrationTest.ForwardBackward`: **SEGFAULT**

### After Workaround
- Test completes without crash
- `TransformerIntegrationTest.ForwardBackward`: **FAIL** (leaf gradients not set)
- Other autograd tests: **Need verification**

## Related Files

- `/home/lee/Projects/Tenzor/include/tenzor/autograd/function.hpp` - Function class definition
- `/home/lee/Projects/Tenzor/src/autograd/engine.cpp` - Backward execution (workaround applied)
- `/home/lee/Projects/Tenzor/include/tenzor/autograd/variable.hpp` - Variable class definition
- `/home/lee/Projects/Tenzor/tests/unit/test_transformer.cpp` - Failing test

## Priority

**CRITICAL** - This affects the core autograd system and causes segfaults in production code.

The workaround prevents crashes but breaks gradient accumulation to leaf variables, making the library unusable for training.

## Estimated Effort

- Refactoring to shared_ptr/weak_ptr: 4-6 hours
- Testing and validation: 2-3 hours
- **Total**: ~8 hours for proper fix

## Additional Notes

- This bug likely affects ALL tests that use backward() with leaf variables
- The crash only manifests when Variables are moved/reallocated (common in real usage)
- Similar issues may exist in other parts of the codebase that store raw Variable*
