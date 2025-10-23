# Tenzor JIT Examples

## Table of Contents

1. [Basic Tracing](#basic-tracing)
2. [Model Optimization](#model-optimization)
3. [Serialization](#serialization)
4. [Custom Passes](#custom-passes)
5. [Production Deployment](#production-deployment)
6. [Debugging](#debugging)

## Basic Tracing

### Example 1: Simple Linear Model

```cpp
#include "tenzor/jit/tracer.hpp"
#include "tenzor/nn/linear.hpp"
#include "tenzor/nn/module.hpp"

class SimpleModel : public nn::Module {
public:
    SimpleModel() {
        fc = std::make_shared<nn::Linear>(784, 10);
        register_module("fc", fc);
    }

    Variable forward(const Variable& x) override {
        return fc->forward(x);
    }

private:
    std::shared_ptr<nn::Linear> fc;
};

int main() {
    // Create model
    auto model = std::make_shared<SimpleModel>();
    model->eval();

    // Create dummy input
    Variable dummy(Tensor({1, 784}, DType::Float32, Device::cpu()), false);

    // Trace the model
    auto traced_model = jit::trace(model, dummy);

    std::cout << "Traced graph:\n" << traced_model->to_string() << "\n";

    // Execute with real input
    Variable input(Tensor({1, 784}, DType::Float32, Device::cpu()), false);
    auto output = traced_model->forward({input});

    return 0;
}
```

### Example 2: CNN Model

```cpp
#include "tenzor/jit/tracer.hpp"
#include "tenzor/nn/conv.hpp"
#include "tenzor/nn/batchnorm.hpp"
#include "tenzor/nn/linear.hpp"

class ConvNet : public nn::Module {
public:
    ConvNet() {
        conv1 = std::make_shared<nn::Conv2d>(3, 64, 3, 1, 1);
        bn1 = std::make_shared<nn::BatchNorm2d>(64);
        conv2 = std::make_shared<nn::Conv2d>(64, 128, 3, 1, 1);
        bn2 = std::make_shared<nn::BatchNorm2d>(128);
        fc = std::make_shared<nn::Linear>(128 * 8 * 8, 10);

        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("fc", fc);
    }

    Variable forward(const Variable& x) override {
        // Conv1 + BN + ReLU
        auto h1 = conv1->forward(x);
        h1 = bn1->forward(h1);
        h1 = relu(h1);

        // Conv2 + BN + ReLU
        auto h2 = conv2->forward(h1);
        h2 = bn2->forward(h2);
        h2 = relu(h2);

        // Flatten and FC
        auto flat = h2.reshape({-1, 128 * 8 * 8});
        return fc->forward(flat);
    }

private:
    std::shared_ptr<nn::Conv2d> conv1, conv2;
    std::shared_ptr<nn::BatchNorm2d> bn1, bn2;
    std::shared_ptr<nn::Linear> fc;
};

int main() {
    auto model = std::make_shared<ConvNet>();
    model->eval();

    Variable dummy(Tensor({1, 3, 32, 32}, DType::Float32, Device::cpu()), false);
    auto traced = jit::trace(model, dummy);

    std::cout << "CNN traced with " << traced->num_nodes() << " nodes\n";

    return 0;
}
```

## Model Optimization

### Example 3: Applying Optimizations

```cpp
#include "tenzor/jit/tracer.hpp"
#include "tenzor/jit/compiler.hpp"

int main() {
    // Trace model (from previous examples)
    auto model = std::make_shared<ConvNet>();
    model->eval();
    Variable dummy(Tensor({1, 3, 32, 32}, DType::Float32, Device::cpu()), false);
    auto graph = jit::trace(model, dummy);

    std::cout << "Before optimization: " << graph->num_nodes() << " nodes\n";

    // Create compiler with all default passes
    jit::Compiler compiler(true);
    compiler.set_verbose(true);  // Enable logging

    // Optimize
    int changes = compiler.optimize(*graph);

    std::cout << "After optimization: " << graph->num_nodes() << " nodes\n";
    std::cout << "Total optimizations applied: " << changes << "\n";

    // Print optimization statistics
    for (const auto& [pass_name, count] : compiler.get_stats()) {
        std::cout << "  " << pass_name << ": " << count << " times\n";
    }

    return 0;
}
```

### Example 4: Custom Optimization Pipeline

```cpp
#include "tenzor/jit/compiler.hpp"

int main() {
    auto graph = /* ... traced graph ... */;

    // Create compiler without defaults
    jit::Compiler compiler(false);

    // Add specific passes in custom order
    compiler.add_pass(std::make_unique<jit::FuseConvBatchNormPass>());
    compiler.add_pass(std::make_unique<jit::FuseConvReluPass>());
    compiler.add_pass(std::make_unique<jit::DeadCodeEliminationPass>());
    compiler.add_pass(std::make_unique<jit::ConstantFoldingPass>());

    // Optimize with custom iteration limit
    compiler.optimize(*graph, 5);

    return 0;
}
```

## Serialization

### Example 5: Save and Load Model

```cpp
#include "tenzor/jit/serialization.hpp"

int main() {
    // Trace and optimize model
    auto graph = jit::trace(model, dummy);
    jit::optimize_graph(*graph);

    // Save to disk
    graph->save("my_model.pt");
    std::cout << "Model saved to my_model.pt\n";

    // Load from disk
    auto loaded_graph = jit::Graph::load("my_model.pt");
    std::cout << "Model loaded successfully\n";

    // Execute loaded model
    Variable input(Tensor({1, 784}, DType::Float32, Device::cpu()), false);
    auto output = loaded_graph->forward({input});

    return 0;
}
```

### Example 6: Cross-Platform Deployment

```cpp
// Training script (may run on GPU)
void train_and_save() {
    auto model = std::make_shared<MyModel>();
    model->cuda();  // Train on GPU

    // ... training loop ...

    // Trace and optimize
    model->cpu();  // Move to CPU for tracing
    model->eval();
    Variable dummy(Tensor({1, 784}, DType::Float32, Device::cpu()), false);
    auto graph = jit::trace(model, dummy);
    jit::optimize_graph(*graph);

    // Save optimized model
    graph->save("production_model.pt");
}

// Inference script (may run on different device)
void inference() {
    // Load model
    auto graph = jit::Graph::load("production_model.pt");

    // Run inference on CPU
    Variable input(/* ... */);
    auto output = graph->forward({input});

    // Or move to CUDA for inference
    // input = input.cuda();
    // auto output = graph->forward({input});
}
```

## Custom Passes

### Example 7: Custom Optimization Pass

```cpp
#include "tenzor/jit/compiler.hpp"

class BatchNormToInstanceNormPass : public jit::Pass {
public:
    bool run(jit::Graph& graph) override {
        bool modified = false;

        for (const auto& node : graph.nodes()) {
            if (node->op_type() == jit::OpType::BatchNorm2d) {
                // Check if batch size is 1
                if (!node->inputs().empty()) {
                    auto input_shape = node->inputs()[0]->shape();
                    if (input_shape.size() > 0 && input_shape[0] == 1) {
                        // Convert to InstanceNorm
                        // (simplified - would need full implementation)
                        node->set_bool_attr("converted_to_instance_norm", true);
                        modified = true;
                    }
                }
            }
        }

        return modified;
    }

    std::string name() const override {
        return "BatchNormToInstanceNorm";
    }
};

int main() {
    auto graph = /* ... */;

    jit::Compiler compiler(true);
    compiler.add_pass(std::make_unique<BatchNormToInstanceNormPass>());
    compiler.optimize(*graph);

    return 0;
}
```

### Example 8: Pattern Matching Pass

```cpp
class ActivationFusionPass : public jit::Pass {
public:
    bool run(jit::Graph& graph) override {
        bool modified = false;

        for (size_t i = 0; i + 1 < graph.nodes().size(); ++i) {
            auto& node1 = graph.nodes()[i];
            auto& node2 = graph.nodes()[i + 1];

            // Match pattern: Linear -> Activation
            if (node1->op_type() == jit::OpType::Linear) {
                jit::OpType activation = node2->op_type();

                if (activation == jit::OpType::ReLU ||
                    activation == jit::OpType::Sigmoid ||
                    activation == jit::OpType::Tanh) {

                    // Mark for fusion
                    node1->set_int_attr("fused_activation",
                                       static_cast<int>(activation));
                    modified = true;
                }
            }
        }

        return modified;
    }

    std::string name() const override {
        return "ActivationFusion";
    }
};
```

## Production Deployment

### Example 9: Model Server

```cpp
#include "tenzor/jit/graph.hpp"
#include <string>
#include <vector>
#include <memory>

class ModelServer {
public:
    ModelServer(const std::string& model_path) {
        // Load model once
        model_ = jit::Graph::load(model_path);
        std::cout << "Model loaded: " << model_path << "\n";
    }

    std::vector<float> predict(const std::vector<float>& input_data,
                               const std::vector<int64_t>& input_shape) {
        // Create input tensor
        Tensor input(input_shape, DType::Float32, Device::cpu());
        std::copy(input_data.begin(), input_data.end(), input.data<float>());

        Variable input_var(input, false);

        // Run inference
        auto outputs = model_->forward({input_var});

        // Extract results
        const auto& output_tensor = outputs[0].tensor();
        std::vector<float> results(output_tensor.numel());
        std::copy(output_tensor.data<float>(),
                  output_tensor.data<float>() + output_tensor.numel(),
                  results.begin());

        return results;
    }

private:
    std::shared_ptr<jit::Graph> model_;
};

// Usage
int main() {
    ModelServer server("production_model.pt");

    // Serve requests
    while (true) {
        auto request = /* receive from network */;
        auto results = server.predict(request.data, request.shape);
        /* send results back */;
    }

    return 0;
}
```

### Example 10: Batch Inference

```cpp
class BatchInference {
public:
    BatchInference(const std::string& model_path, size_t batch_size)
        : batch_size_(batch_size) {
        model_ = jit::Graph::load(model_path);
    }

    std::vector<Tensor> infer_batch(const std::vector<Tensor>& inputs) {
        // Batch inputs
        if (inputs.size() != batch_size_) {
            throw std::runtime_error("Batch size mismatch");
        }

        // Stack tensors
        Tensor batched = stack(inputs, 0);
        Variable input_var(batched, false);

        // Single forward pass
        auto outputs = model_->forward({input_var});

        // Unstack results
        std::vector<Tensor> results;
        for (size_t i = 0; i < batch_size_; ++i) {
            results.push_back(outputs[0].tensor()[i]);
        }

        return results;
    }

private:
    std::shared_ptr<jit::Graph> model_;
    size_t batch_size_;
};
```

## Debugging

### Example 11: Graph Inspection

```cpp
#include "tenzor/jit/serialization.hpp"

int main() {
    auto graph = jit::trace(model, dummy);

    // Print graph structure
    std::cout << graph->to_string() << "\n\n";

    // Get detailed statistics
    std::cout << jit::get_graph_stats(*graph) << "\n\n";

    // Verify graph integrity
    auto errors = jit::verify_graph(*graph);
    if (!errors.empty()) {
        std::cerr << "Graph validation errors:\n";
        for (const auto& error : errors) {
            std::cerr << "  - " << error << "\n";
        }
    } else {
        std::cout << "Graph is valid\n";
    }

    // Export for visualization
    jit::export_graph_dot(*graph, "graph.dot");
    std::cout << "Graph exported to graph.dot\n";
    std::cout << "Visualize with: dot -Tpng graph.dot -o graph.png\n";

    return 0;
}
```

### Example 12: Performance Profiling

```cpp
#include <chrono>

int main() {
    auto graph = jit::trace(model, dummy);

    // Profile before optimization
    Variable input(Tensor({1, 784}, DType::Float32, Device::cpu()), false);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        graph->forward({input});
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_before = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    // Optimize
    jit::optimize_graph(*graph);

    // Profile after optimization
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        graph->forward({input});
    }
    end = std::chrono::high_resolution_clock::now();
    auto duration_after = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();

    std::cout << "Before optimization: " << duration_before << " μs\n";
    std::cout << "After optimization:  " << duration_after << " μs\n";
    std::cout << "Speedup: " << (double)duration_before / duration_after << "x\n";

    return 0;
}
```

### Example 13: Debugging with Text Export

```cpp
int main() {
    auto graph = jit::trace(model, dummy);

    // Export to human-readable text
    jit::export_graph_text(*graph, "model_debug.txt");

    // Can inspect/edit with text editor
    std::cout << "Model exported to model_debug.txt\n";
    std::cout << "You can now:\n";
    std::cout << "  1. Inspect the graph structure\n";
    std::cout << "  2. Check node attributes\n";
    std::cout << "  3. Verify data flow\n";

    // Could potentially edit and re-import (if import was implemented)
    // auto modified_graph = jit::import_graph_text("model_debug.txt");

    return 0;
}
```

## Advanced Patterns

### Example 14: Multi-Input Multi-Output Model

```cpp
class MultiIOModel : public nn::Module {
public:
    Variable forward(const Variable& x) override {
        // Not ideal for multi-output
        // Better to trace a function directly
        throw std::runtime_error("Use trace() with function");
    }
};

int main() {
    // Trace function with multiple inputs/outputs
    auto func = [](const std::vector<Variable>& inputs) {
        auto x = inputs[0];
        auto y = inputs[1];

        auto sum = x + y;
        auto prod = x * y;

        return std::vector<Variable>{sum, prod};
    };

    Variable x(Tensor({2, 3}, DType::Float32, Device::cpu()), false);
    Variable y(Tensor({2, 3}, DType::Float32, Device::cpu()), false);

    auto graph = jit::trace(func, {x, y});

    std::cout << "Traced multi-I/O graph with " << graph->inputs().size()
              << " inputs and " << graph->outputs().size() << " outputs\n";

    return 0;
}
```

### Example 15: Conditional Compilation

```cpp
int main() {
    auto model = std::make_shared<MyModel>();
    Variable dummy(Tensor({1, 784}, DType::Float32, Device::cpu()), false);
    auto graph = jit::trace(model, dummy);

    // Apply different optimizations based on deployment target
    jit::Compiler compiler(false);

    #ifdef OPTIMIZE_FOR_MOBILE
        // Aggressive optimizations for mobile
        compiler.add_pass(std::make_unique<jit::ConstantFoldingPass>());
        compiler.add_pass(std::make_unique<jit::FuseConvBatchNormPass>());
        compiler.add_pass(std::make_unique<jit::FuseConvReluPass>());
        compiler.add_pass(std::make_unique<jit::DeadCodeEliminationPass>());
    #elif defined(OPTIMIZE_FOR_SERVER)
        // Balanced optimizations for server
        compiler.add_pass(std::make_unique<jit::FuseConvBatchNormPass>());
        compiler.add_pass(std::make_unique<jit::DeadCodeEliminationPass>());
    #else
        // Minimal optimizations for debugging
        compiler.add_pass(std::make_unique<jit::DeadCodeEliminationPass>());
    #endif

    compiler.optimize(*graph);
    graph->save("model_optimized.pt");

    return 0;
}
```

## Best Practices

1. **Always trace in eval mode**: `model->eval()` before tracing
2. **Use representative inputs**: Match production data shapes
3. **Optimize before deployment**: Run compiler on traced graphs
4. **Verify after optimization**: Use `verify_graph()` to catch errors
5. **Profile your optimizations**: Measure actual performance gains
6. **Version your models**: Track which optimizations were applied
7. **Cache loaded models**: Don't reload on every inference
8. **Handle errors gracefully**: Tracing can fail on unsupported ops
9. **Document your graphs**: Use node names and comments
10. **Test serialization**: Verify loaded model matches original

## Troubleshooting

### Issue: Tracing fails with "Operation not supported"
**Solution**: Some operations may not be traceable yet. Check `OpType` enum for supported operations.

### Issue: Optimized model produces different results
**Solution**: Some optimizations may introduce numerical differences. Verify with `verify_graph()` and compare outputs.

### Issue: Serialized model won't load
**Solution**: Check version compatibility. Serialization format may change between versions.

### Issue: Graph execution is slower than original
**Solution**: Not all optimizations help all models. Try different pass combinations or disable optimizations.

### Issue: Large model file size
**Solution**: File size depends on parameter count. Compression not yet implemented.
