# ROOT CAUSE: Parameter Storage in unordered_map Causes Variable Moves

## Critical Discovery

**Location**: `/home/lee/Projects/Tenzor/include/tenzor/nn/module.hpp:140`

```cpp
std::unordered_map<std::string, Variable> parameters_;  ///< Named parameters
```

## The Problem

1. **Parameters stored by value**: Variables are stored directly in `unordered_map<string, Variable>`
2. **Map rehashing moves Variables**: When the map grows, it rehashes and MOVES all Variable objects to new memory
3. **Pointers become dangling**: Any pointer (raw or shared_ptr with no-op deleter) to the old Variable location becomes invalid
4. **Crash during backward**: When autograd tries to accumulate gradients using these pointers → SEGFAULT

## Why shared_ptr with no-op deleter didn't work

- `shared_ptr<Variable>` with no-op deleter still points to the SAME memory address
- When `unordered_map` moves the Variable object, the memory address changes
- The shared_ptr still points to the OLD (now invalid) address
- No amount of smart pointer magic can fix this if the underlying object moves

## Solutions

### Option 1: Store Parameters as Pointers (RECOMMENDED)
```cpp
// In module.hpp:
std::unordered_map<std::string, std::shared_ptr<Variable>> parameters_;

// Benefits:
// - Variable objects have stable heap addresses
// - No moves when map rehashes
// - Autograd can safely store pointers

// Downside:
// - Requires refactoring all Module subclasses
// - Change register_parameter() signature
```

### Option 2: Use Stable Container
```cpp
// In module.hpp:
std::deque<Variable> parameter_storage_;  // Stable addresses
std::unordered_map<std::string, Variable*> parameters_;  // Points into deque

// Benefits:
// - Variables never move (deque guarantees stability)
// - Minimal API changes

// Downside:
// - More complex memory management
```

### Option 3: Pre-reserve Map Capacity
```cpp
// In Module constructor:
parameters_.reserve(1000);  // Prevent rehashing

// Benefits:
// - Minimal code changes
// - Quick workaround

// Downside:
// - Not a real fix (large models will still crash)
// - Wastes memory
```

### Option 4: Don't Store Variable Pointers in Autograd
```cpp
// Store gradient accumulation differently:
// - Use Tensor data pointer as key
// - Store callback to set gradient
// - Decouple from Variable lifetime

// Benefits:
// - Most robust solution
// - No Variable lifetime issues

// Downside:
// - Requires significant autograd refactoring
```

## Recommended Path Forward

**Short-term**: Option 3 (reserve capacity) + document limitation
**Long-term**: Option 1 (store as shared_ptr) - proper architecture

## Test Case Reproduction

```cpp
TEST(AutogradTest, ParameterMoveCrash) {
    // Create module with parameters
    Linear layer(128, 128);
    auto params = layer.parameters();

    // Force map to rehash by adding many submodules
    for (int i = 0; i < 100; ++i) {
        layer.register_parameter("dummy" + std::to_string(i), Variable(randn({10, 10}), true));
    }

    // Now params[0] points to moved Variable → CRASH in backward()
    Variable input(randn({1, 128}), true);
    Variable output = layer.forward(input);
    output.backward();  // SEGFAULT
}
```

## Files Affected

- `/home/lee/Projects/Tenzor/include/tenzor/nn/module.hpp` - Module class
- `/home/lee/Projects/Tenzor/src/nn/module.cpp` - Module implementation
- All Layer classes (Linear, Conv2d, etc.) - register_parameter calls
- `/home/lee/Projects/Tenzor/include/tenzor/autograd/function.hpp` - autograd tracking

## Priority

**CRITICAL** - This is a fundamental architectural issue that affects ALL models with parameters.

Current status: 849/853 tests passing, but TransformerIntegrationTest.ForwardBackward crashes due to this bug.
