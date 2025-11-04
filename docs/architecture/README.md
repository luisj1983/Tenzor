# Tenzor Runtime Dispatcher Architecture Documentation

This directory contains comprehensive architectural documentation for the runtime operation dispatcher system in Tenzor.

## Overview

The runtime dispatcher architecture eliminates compile-time `#ifdef` checks from the Tenzor frontend, providing clean separation between frontend operations and backend implementations. This enables:

- Backend-agnostic frontend code
- Easy addition of new backends
- Graceful fallback for missing implementations
- Minimal performance overhead
- Better testability and maintainability

## Documents

### 1. [Runtime Dispatcher Design](runtime-dispatcher-design.md)
**Main architectural design document**

Comprehensive overview of the dispatcher architecture including:
- Current architecture analysis
- Proposed design with component diagrams
- Operation registry data structures
- Backend interface design
- Registration API and macros
- Fallback mechanism
- Example implementations for all backends
- Performance considerations
- Testing strategy
- Migration guide

**Read this first** for a complete understanding of the architecture.

### 2. [Dispatch Sequence Diagrams](dispatch-sequence-diagrams.md)
**Visual flow diagrams and performance analysis**

Contains:
- Sequence diagrams for operation dispatch
- Fallback mechanism flows
- Library initialization sequence
- Data structure diagrams
- Decision trees for dispatch logic
- Performance timelines
- Comparison of with/without fallback scenarios
- Scalability analysis

**Best for** understanding runtime behavior and performance characteristics.

### 3. [Implementation Guide](implementation-guide.md)
**Step-by-step implementation instructions**

Practical guide covering:
- Phase 1: Core infrastructure (OperationRegistry, BackendRegistry)
- Phase 2: Backend interface enhancements
- Phase 3: CPU backend refactoring
- Phase 4: CUDA backend refactoring
- Phase 5: Library initialization system
- Phase 6: Testing strategy
- Phase 7: CMake integration

**Use this** when implementing the architecture.

### 4. [Architectural Decisions](architectural-decisions.md)
**Architecture Decision Records (ADRs) and rationale**

Detailed ADRs for:
- ADR-001: Runtime operation dispatcher
- ADR-002: Two-level registry architecture
- ADR-003: Fallback strategy
- ADR-004: Backend initialization strategy
- ADR-005: Kernel function signature
- ADR-006: Thread safety model
- ADR-007: Performance optimization strategy
- ADR-008: Error handling strategy

Plus performance analysis, memory overhead, and future enhancements.

**Reference this** when making architectural decisions or understanding design choices.

## Quick Start

### For Understanding the Architecture

1. Read the [Runtime Dispatcher Design](runtime-dispatcher-design.md) introduction
2. Review the [Sequence Diagrams](dispatch-sequence-diagrams.md) to understand flow
3. Check [Architectural Decisions](architectural-decisions.md) for design rationale

### For Implementation

1. Start with [Implementation Guide](implementation-guide.md) Phase 1
2. Follow each phase sequentially
3. Reference [Runtime Dispatcher Design](runtime-dispatcher-design.md) for API details
4. Use [Architectural Decisions](architectural-decisions.md) for context on design choices

### For Adding a New Backend

1. Review the backend interface in [Runtime Dispatcher Design](runtime-dispatcher-design.md)
2. Follow the backend implementation pattern in [Implementation Guide](implementation-guide.md)
3. Example code:

```cpp
// 1. Create backend class
class MyBackend : public Backend {
public:
    auto name() const -> std::string_view override {
        return "mybackend";
    }

    auto device_type() const -> Device::Type override {
        return Device::Type::MyBackend;
    }

    auto register_operations(OperationRegistry& registry) -> void override {
        // Register kernels
        TENZOR_REGISTER_KERNEL(Device::Type::MyBackend, "add", my_add_kernel);
        TENZOR_REGISTER_KERNEL(Device::Type::MyBackend, "matmul", my_matmul_kernel);

        // Register CPU fallback for unimplemented ops
        TENZOR_REGISTER_FALLBACK("complex_op", Device::Type::MyBackend, Device::Type::CPU);
    }

    // Implement memory management methods...
};

// 2. Create factory function
extern "C" auto create_mybackend_backend() -> std::unique_ptr<Backend> {
    return std::make_unique<MyBackend>();
}

// 3. Add to initialization in init.cpp
#ifdef TENZOR_HAS_MYBACKEND
{
    auto backend = create_mybackend_backend();
    if (backend && backend->is_available()) {
        backend->register_operations(op_reg);
        backend_reg.register_backend(Device::Type::MyBackend, std::move(backend));
    }
}
#endif
```

## Key Concepts

### Operation Registry
Thread-safe mapping from `(operation_name, device_type)` → `kernel_function`

```cpp
auto& registry = operation_registry();
registry.register_kernel("add", Device::Type::CUDA, cuda_add_kernel);
```

### Backend Registry
Manages available backend implementations

```cpp
auto& registry = backend_registry();
auto* backend = registry.get_backend(Device::Type::CUDA);
```

### Dispatcher
Routes operations to appropriate backend

```cpp
// In frontend code (NO #ifdef needed!)
auto result = Dispatcher::dispatch("add", {tensor_a, tensor_b});
```

### Fallback Mechanism
Automatic CPU fallback for missing implementations

```cpp
// If CUDA doesn't implement "custom_op", automatically use CPU
TENZOR_REGISTER_FALLBACK("custom_op", Device::Type::CUDA, Device::Type::CPU);
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│              Frontend Layer                     │
│  (ops/math.hpp, nn/layers/*, autograd/*)       │
│  - Backend-agnostic                             │
│  - No #ifdef checks                             │
└──────────────────┬──────────────────────────────┘
                   │
                   │ Dispatcher::dispatch()
                   │
┌──────────────────▼──────────────────────────────┐
│             Dispatcher Layer                    │
│  - Device compatibility checking                │
│  - Backend selection                            │
└──────────────────┬──────────────────────────────┘
                   │
                   │ Backend::dispatch()
                   │
┌──────────────────▼──────────────────────────────┐
│           Backend Layer                         │
│  - Operation registry lookup                    │
│  - Fallback resolution                          │
│  - Kernel execution                             │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│           Kernel Layer                          │
│  - CPU kernels (C++, SIMD)                     │
│  - CUDA kernels (.cu)                           │
│  - ROCm kernels (HIP)                           │
│  - OneAPI kernels (SYCL)                        │
│  - Vulkan kernels (SPIR-V)                      │
└─────────────────────────────────────────────────┘
```

## Performance Characteristics

| Metric                          | Value        | Notes                    |
|---------------------------------|--------------|--------------------------|
| Dispatch overhead               | ~150 ns      | Per operation            |
| Registry lookup                 | O(1)         | Hash map                 |
| Fallback overhead (small)       | ~400%        | Due to data transfer     |
| Fallback overhead (large)       | ~4900%       | Avoid for hot paths      |
| Memory overhead                 | ~13 KB       | Total for all registries |
| Thread scalability              | Linear       | Reader-writer lock       |

## Design Principles

1. **Zero Frontend Coupling**: Frontend code has no knowledge of backends
2. **Runtime Flexibility**: Operations dispatch based on tensor device at runtime
3. **Graceful Degradation**: Missing implementations fall back to CPU
4. **Type Safety**: Strong typing with clear error messages
5. **Performance**: Minimal overhead through efficient lookup
6. **Extensibility**: Easy to add new backends and operations

## Migration Checklist

- [ ] Phase 1: Implement core infrastructure
  - [ ] BackendRegistry
  - [ ] Enhanced OperationRegistry
  - [ ] Registration macros
- [ ] Phase 2: Refactor backends
  - [ ] CPUBackend
  - [ ] CUDABackend
  - [ ] ROCmBackend
  - [ ] OneAPIBackend
- [ ] Phase 3: Clean frontend
  - [ ] Remove #ifdef from ops/
  - [ ] Remove #ifdef from nn/
  - [ ] Remove #ifdef from autograd/
- [ ] Phase 4: Testing
  - [ ] Unit tests
  - [ ] Integration tests
  - [ ] Performance benchmarks
- [ ] Phase 5: Documentation
  - [ ] API documentation
  - [ ] Migration guide
  - [ ] Performance guide

## Testing Strategy

### Unit Tests
- Backend registration
- Operation registration
- Fallback mechanism
- Device compatibility checking

### Integration Tests
- End-to-end dispatch
- Multi-backend scenarios
- Error handling

### Performance Tests
- Dispatch overhead benchmarks
- Fallback performance
- Memory usage
- Thread scalability

## Future Enhancements

1. **Dynamic Backend Loading**: Plugin system for runtime backend loading
2. **Operation Fusion**: Automatic fusion of common operation patterns
3. **Multi-Device Execution**: Automatic work distribution across devices
4. **JIT Compilation**: Runtime kernel generation
5. **Profiling Integration**: Built-in performance monitoring

## Contributing

When adding new features or backends:

1. Read the relevant documentation sections
2. Follow the implementation patterns
3. Add tests for new functionality
4. Update documentation as needed
5. Submit ADR for significant architectural changes

## Questions?

For questions about the architecture:
1. Check the relevant document above
2. Review the ADRs for design rationale
3. Look at example implementations
4. Open an issue for clarification

## File Structure

```
docs/architecture/
├── README.md                           # This file
├── runtime-dispatcher-design.md        # Main design document
├── dispatch-sequence-diagrams.md       # Flow diagrams and analysis
├── implementation-guide.md             # Step-by-step implementation
└── architectural-decisions.md          # ADRs and performance analysis
```

## Additional Resources

- [Tenzor API Documentation](../api/)
- [Backend Development Guide](../backend/)
- [Performance Optimization Guide](../performance/)
- [Testing Guide](../testing/)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-01
**Status**: Proposed Architecture
