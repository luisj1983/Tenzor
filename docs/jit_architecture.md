# Tenzor JIT/TorchScript Architecture

## Overview

The Tenzor JIT system provides ahead-of-time compilation for neural networks through operation tracing, graph optimization, and serialization. This enables:

- **Performance**: Optimized execution through operator fusion and graph transformations
- **Portability**: Save and load compiled models across platforms
- **Deployment**: Efficient inference without Python dependencies
- **Analysis**: Inspect and optimize computation graphs

## System Components

### 1. Tracer (`tracer.hpp`)

**Purpose**: Record operations during forward pass to build IR graph

**Key Classes**:
- `Tracer`: Thread-local singleton that records operations
- `TracingGuard`: RAII wrapper for automatic trace management
- `TracedOp`: Recorded operation with inputs, outputs, and attributes

**Workflow**:
```cpp
// Start tracing
Tracer& tracer = Tracer::get_instance();
tracer.start_trace();

// Operations are automatically recorded
Variable output = model(input);

// End tracing and get graph
auto graph = tracer.end_trace({input}, {output});
```

**Operation Types**: 45+ operations supported including:
- Arithmetic: Add, Sub, Mul, Div
- Matrix: MatMul, Bmm
- Activations: ReLU, Sigmoid, Tanh, Softmax
- Convolution: Conv2d
- Normalization: BatchNorm2d, LayerNorm
- Shape: Reshape, Transpose, Permute
- Reductions: Sum, Mean, Max
- And more...

### 2. Graph IR (`graph.hpp`)

**Purpose**: Intermediate representation for neural network computations

**Key Classes**:
- `Graph`: Container for nodes and values
- `Node`: Operation node with type, inputs, outputs, attributes
- `Value`: Tensor value flowing between nodes

**Graph Structure**:
```
Inputs → [Node1] → Value1 → [Node2] → Value2 → [Node3] → Outputs
                  ↓
                Value3
                  ↓
              [Node4]
```

**Features**:
- Topological sorting for execution order
- Type inference for shape propagation
- Dynamic execution with runtime inputs
- Attribute storage (float, int, vector, bool, tensor)

**Node Structure**:
```cpp
Node {
    OpType op_type;                    // Operation type
    string name;                        // Node identifier
    vector<Value*> inputs;              // Input values
    vector<Value*> outputs;             // Output values
    map<string, Attribute> attributes;  // Operation parameters
}
```

**Value Structure**:
```cpp
Value {
    string id;                   // Unique identifier
    vector<int64_t> shape;      // Tensor shape
    DType dtype;                 // Data type
    Device device;               // Device location
    Node* producer;              // Producing node
    vector<Node*> consumers;     // Consuming nodes
}
```

### 3. Compiler (`compiler.hpp`)

**Purpose**: Optimize IR graphs for performance

**Optimization Passes**:

#### Dead Code Elimination (DCE)
- Removes nodes not contributing to outputs
- Backward traversal from outputs
- Marks reachable nodes
- Speedup: Eliminates unnecessary computation

#### Constant Folding
- Evaluates constant expressions at compile time
- Replaces operations with constant results
- Example: `2.0 + 3.0` → `5.0`
- Speedup: Reduces runtime computation

#### Operator Fusion

**Conv2d + BatchNorm2d Fusion**:
```
Before:
  y = conv(x, w, b)
  z = bn(y, γ, β, μ, σ²)

After:
  z = conv(x, w', b')
  where:
    w' = γ * w / sqrt(σ² + ε)
    b' = γ * (b - μ) / sqrt(σ² + ε) + β
```
Speedup: 10-20% inference improvement

**Conv2d + ReLU Fusion**:
```
Before:
  y = conv(x)
  z = relu(y)

After:
  z = conv_relu(x)
```
Speedup: 5-15% with optimized kernels

**Linear + ReLU Fusion**:
Similar pattern for fully-connected layers

#### Common Subexpression Elimination (CSE)
- Detects duplicate computations
- Merges equivalent nodes
- Example:
```
a = relu(x)
b = relu(x)  # Duplicate
c = a + b

→

a = relu(x)
c = a + a    # Reuse 'a'
```

#### Algebraic Simplification
- Applies mathematical identities:
  - `x + 0 = x`
  - `x * 1 = x`
  - `x * 0 = 0`
  - `log(exp(x)) = x`

#### Reshape Elimination
- Removes redundant reshapes
- Merges consecutive reshapes
- Example: `reshape(reshape(x, s1), s2)` → `reshape(x, s2)`

**Compiler Pipeline**:
```cpp
Compiler compiler(true);  // Enable default passes
compiler.optimize(graph, max_iterations=10);
```

Passes run iteratively until convergence or max iterations.

### 4. Serialization (`serialization.hpp`)

**Purpose**: Save and load compiled models

**Binary Format**:
```
[Magic: 4 bytes] [Version: 4 bytes]
[Metadata Section]
  - Node count
  - Value count
  - Input/output counts

[Values Section]
  For each value:
    - ID (string)
    - Shape (int64 vector)
    - DType (uint32)
    - Device (uint32 + int64)

[Nodes Section]
  For each node:
    - OpType (uint32)
    - Name (string)
    - Input IDs (string vector)
    - Output IDs (string vector)
    - Attributes (typed maps)

[Tensors Section]
  For each tensor constant:
    - Shape, DType, Device
    - Raw data (bytes)
```

**Features**:
- Compact binary encoding
- Cross-platform compatible
- Fast loading (minimal parsing)
- Version tagging for compatibility

**Utilities**:
- Text export for debugging
- DOT export for visualization (Graphviz)
- Graph statistics
- Integrity verification

## Usage Examples

### Basic Tracing

```cpp
#include "tenzor/jit/tracer.hpp"

// Define model
class SimpleNet : public nn::Module {
public:
    Variable forward(const Variable& x) override {
        auto h1 = fc1->forward(x).relu();
        return fc2->forward(h1);
    }
private:
    std::shared_ptr<nn::Linear> fc1, fc2;
};

// Trace model
auto model = std::make_shared<SimpleNet>();
Variable dummy = Variable(Tensor({1, 784}, DType::Float32, Device::cpu()), false);

auto traced_model = jit::trace(model, dummy);
```

### Full Workflow

```cpp
// 1. Trace
auto graph = jit::trace(model, dummy_input);

// 2. Optimize
jit::Compiler compiler(true);
compiler.set_verbose(true);
int changes = compiler.optimize(*graph);
std::cout << "Applied " << changes << " optimizations\n";

// 3. Verify
auto errors = jit::verify_graph(*graph);
if (!errors.empty()) {
    for (const auto& err : errors) {
        std::cerr << "Error: " << err << "\n";
    }
}

// 4. Save
graph->save("model.pt");

// 5. Load
auto loaded = jit::Graph::load("model.pt");

// 6. Execute
Variable output = loaded->forward({input});
```

### Custom Optimization Pass

```cpp
class MyCustomPass : public jit::Pass {
public:
    bool run(jit::Graph& graph) override {
        bool modified = false;

        for (const auto& node : graph.nodes()) {
            if (should_optimize(node)) {
                optimize_node(node);
                modified = true;
            }
        }

        return modified;
    }

    std::string name() const override {
        return "MyCustomPass";
    }
};

// Use custom pass
jit::Compiler compiler(false);  // Disable defaults
compiler.add_pass(std::make_unique<MyCustomPass>());
compiler.optimize(graph);
```

### Inspection and Analysis

```cpp
// Print graph structure
std::cout << graph->to_string() << "\n";

// Get statistics
std::cout << jit::get_graph_stats(*graph) << "\n";

// Export for visualization
jit::export_graph_dot(*graph, "model.dot");
// Then: dot -Tpng model.dot -o model.png

// Export text format
jit::export_graph_text(*graph, "model.txt");
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    User Code                             │
│              (Neural Network Model)                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   Tracer                                 │
│  - Records operations during forward pass                │
│  - Tracks tensor IDs and metadata                        │
│  - Thread-local singleton                                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                IR Graph Builder                          │
│  - Converts traced ops to nodes/values                   │
│  - Establishes data dependencies                         │
│  - Performs topological sort                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Type Inference                              │
│  - Propagates shapes through graph                       │
│  - Validates type compatibility                          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│             Compiler (Optimizer)                         │
│  ┌─────────────────────────────────────────────┐        │
│  │  Pass 1: Dead Code Elimination               │        │
│  │  Pass 2: Constant Folding                    │        │
│  │  Pass 3: Conv+BN Fusion                      │        │
│  │  Pass 4: Conv+ReLU Fusion                    │        │
│  │  Pass 5: Linear+ReLU Fusion                  │        │
│  │  Pass 6: Algebraic Simplification            │        │
│  │  Pass 7: Reshape Elimination                 │        │
│  │  Pass 8: Common Subexpression Elimination    │        │
│  └─────────────────────────────────────────────┘        │
│            (Iterate until convergence)                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Optimized IR Graph                          │
└────────┬────────────────────────────┬───────────────────┘
         │                            │
         ▼                            ▼
┌──────────────────┐        ┌──────────────────────┐
│  Serialization   │        │   Graph Execution     │
│  - Binary format │        │   - Dynamic inputs    │
│  - Save to disk  │        │   - Runtime execution │
│  - Load from disk│        │   - Return outputs    │
└──────────────────┘        └──────────────────────┘
```

## Performance Characteristics

### Optimization Impact

| Optimization | Typical Speedup | Use Case |
|--------------|----------------|----------|
| Conv+BN Fusion | 10-20% | CNNs with batch norm |
| Conv+ReLU Fusion | 5-15% | CNNs with activations |
| Linear+ReLU Fusion | 3-10% | MLPs |
| Dead Code Elimination | Variable | Complex graphs |
| Constant Folding | 1-5% | Models with constants |
| Algebraic Simplification | 1-3% | Mathematical operations |

### Serialization Performance

- **Save Time**: O(nodes + values)
- **Load Time**: O(nodes + values)
- **File Size**: ~1-2x model parameters size
- **Compression**: Not implemented (future enhancement)

## Limitations and Future Work

### Current Limitations

1. **Script Mode**: Not implemented (only trace mode available)
2. **Control Flow**: Limited support for if/while/for
3. **Dynamic Shapes**: Fixed shapes at trace time
4. **Quantization**: Not integrated with JIT
5. **Multi-GPU**: Single device graphs only

### Future Enhancements

1. **Script Mode**: Parse C++ code for control flow
2. **Shape Functions**: Dynamic shape inference
3. **Quantization Fusion**: Integrate QAT/PTQ
4. **Multi-Device Graphs**: Support for model parallelism
5. **Backend Codegen**: Generate optimized CUDA/CPU kernels
6. **Memory Planning**: Optimal tensor allocation
7. **Profile-Guided Optimization**: Use runtime profiles

## Integration with Tenzor

The JIT system integrates with:

- **Autograd**: Tracing captures gradient computation
- **Module System**: `nn::Module` provides forward() for tracing
- **Operations**: All `ops::` functions are traceable
- **Backends**: Graphs execute on CPU/CUDA/ROCm
- **Parallel**: Future integration with data/model parallelism

## Best Practices

1. **Trace in Eval Mode**: `model->eval()` before tracing
2. **Representative Inputs**: Use typical input shapes
3. **Optimize Before Save**: Run compiler before serialization
4. **Verify Graphs**: Use `verify_graph()` before deployment
5. **Version Models**: Track serialization version compatibility
6. **Test Execution**: Validate loaded models match original

## References

- PyTorch TorchScript: https://pytorch.org/docs/stable/jit.html
- ONNX Format: https://onnx.ai/
- TensorFlow SavedModel: https://www.tensorflow.org/guide/saved_model
- XLA Compiler: https://www.tensorflow.org/xla
