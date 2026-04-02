/**
 * @file compiled_module.cpp
 * @brief Implementation of CompiledModule and convenience functions
 *
 * Provides the high-level interface for traced JIT modules, including
 * tracing, execution, optimization, serialization, and metadata management.
 */

#include "../../include/tenzor/jit/compiler.hpp"
#include "../../include/tenzor/jit/tracer.hpp"
#include "../../include/tenzor/jit/serialization.hpp"
#include "../../include/tenzor/core/device.hpp"
#include <algorithm>
#include <stdexcept>

#ifdef TENZOR_HAS_ROCM
#include "../backends/rocm/hip_graph.hpp"
#include <hip/hip_runtime.h>
#endif

namespace tenzor {
namespace jit {

// ============================================================================
// CompiledModule Implementation
// ============================================================================

CompiledModule::~CompiledModule() {
    invalidate_cuda_graph();
}

CompiledModule::CompiledModule(std::shared_ptr<Graph> graph)
    : graph_(std::move(graph)) {}

auto CompiledModule::trace(std::shared_ptr<nn::Module> module,
                            const Variable& example_input) -> std::shared_ptr<CompiledModule> {
    if (!module) {
        throw std::runtime_error("Cannot trace null module");
    }

    // Use the existing jit::trace function to build the graph
    auto graph = jit::trace(module, example_input);

    auto compiled = std::make_shared<CompiledModule>(graph);
    return compiled;
}

auto CompiledModule::trace(std::shared_ptr<nn::Module> module,
                            const Tensor& example_input) -> std::shared_ptr<CompiledModule> {
    return trace(module, Variable(example_input, false));
}

auto CompiledModule::forward(const Variable& input) -> Variable {
    if (!graph_) {
        throw std::runtime_error("CompiledModule has no graph");
    }

    auto results = graph_->forward({input});
    if (results.empty()) {
        throw std::runtime_error("CompiledModule produced no outputs");
    }
    return results[0];
}

auto CompiledModule::forward(const Tensor& input) -> Variable {
    return forward(Variable(input, false));
}

auto CompiledModule::forward(const std::vector<Variable>& inputs) -> std::vector<Variable> {
    if (!graph_) {
        throw std::runtime_error("CompiledModule has no graph");
    }
    return graph_->forward(inputs);
}

auto CompiledModule::optimize_for_inference() -> int {
    if (!graph_) {
        throw std::runtime_error("CompiledModule has no graph to optimize");
    }

    // Use Compiler directly so we can capture the memory plan
    Compiler compiler(true);
    int result = compiler.optimize(*graph_);
    memory_plan_ = compiler.memory_plan();
    return result;
}

auto CompiledModule::save(const std::string& path) const -> void {
    if (!graph_) {
        throw std::runtime_error("CompiledModule has no graph to save");
    }
    // Save the graph using existing serialization
    save_graph(*graph_, path);
}

auto CompiledModule::load(const std::string& path) -> std::shared_ptr<CompiledModule> {
    auto graph = load_graph(path);
    if (!graph) {
        throw std::runtime_error("Failed to load graph from: " + path);
    }
    return std::make_shared<CompiledModule>(graph);
}

auto CompiledModule::add_metadata(const std::string& key, const std::string& value) -> void {
    metadata_[key] = value;
}

auto CompiledModule::get_metadata(const std::string& key) const -> std::string {
    auto it = metadata_.find(key);
    return it != metadata_.end() ? it->second : "";
}

auto CompiledModule::has_metadata(const std::string& key) const -> bool {
    return metadata_.find(key) != metadata_.end();
}

auto CompiledModule::all_metadata() const -> const std::unordered_map<std::string, std::string>& {
    return metadata_;
}

// ============================================================================
// ROCm Graph Adapter (wraps HIPGraph behind the CUDAGraph interface)
// ============================================================================

#ifdef TENZOR_HAS_ROCM
namespace {

class ROCmGraphAdapter : public CUDAGraph {
public:
    explicit ROCmGraphAdapter(int32_t device_id) : device_id_(device_id) {
        hipSetDevice(device_id_);
        auto err = hipStreamCreate(&stream_);
        if (err != hipSuccess) {
            throw std::runtime_error(
                std::string("ROCmGraphAdapter: failed to create stream: ") +
                hipGetErrorString(err));
        }
    }

    ~ROCmGraphAdapter() override {
        if (hip_graph_.is_capturing()) {
            // Abort capture to leave stream in valid state
            hipGraph_t dummy = nullptr;
            hipStreamEndCapture(stream_, &dummy);
            if (dummy) hipGraphDestroy(dummy);
        }
        if (stream_) {
            hipStreamDestroy(stream_);
        }
    }

    void begin_capture() override {
        hipSetDevice(device_id_);
        hip_graph_.begin_capture(stream_);
    }

    void end_capture() override {
        hip_graph_.end_capture(stream_);
    }

    void replay() override {
        hipSetDevice(device_id_);
        hip_graph_.replay(stream_);
    }

    bool is_ready() const override {
        return hip_graph_.is_compiled();
    }

private:
    int32_t device_id_;
    hipStream_t stream_ = nullptr;
    rocm::HIPGraph hip_graph_;
};

} // anonymous namespace
#endif // TENZOR_HAS_ROCM

// ============================================================================
// CUDA / ROCm Graph Capture/Replay
// ============================================================================

auto CompiledModule::capture_cuda_graph(std::vector<Tensor> sample_inputs) -> void {
    if (!graph_) {
        throw std::runtime_error("CompiledModule has no graph");
    }

    // Invalidate any existing captured graph
    invalidate_cuda_graph();

    // Determine device ID and type from the first input tensor
    int32_t device_id = 0;
    Device::Type device_type = Device::Type::CPU;
    if (!sample_inputs.empty()) {
        device_id = sample_inputs[0].device().index;
        device_type = sample_inputs[0].device().type;
    }

    // Create graph capture object based on device type
#ifdef TENZOR_HAS_ROCM
    if (device_type == Device::Type::ROCm) {
        int hip_count = 0;
        hipGetDeviceCount(&hip_count);
        if (hip_count == 0 || device_id >= hip_count) {
            throw std::runtime_error(
                "ROCm is not available; cannot capture HIP graph");
        }
        cuda_graph_ = std::make_unique<ROCmGraphAdapter>(device_id);
    } else
#endif
    {
        // CUDA path (or fallback)
        cuda_graph_ = CUDAGraph::create(device_id);
        if (!cuda_graph_) {
            throw std::runtime_error(
                "GPU is not available; cannot capture graph (device type: " +
                Device{device_type, device_id}.to_string() + ")");
        }
    }

    // Record input shapes for validation during replay
    captured_shapes_.clear();
    captured_shapes_.reserve(sample_inputs.size());
    for (const auto& t : sample_inputs) {
        auto s = t.shape();
        captured_shapes_.emplace_back(s.begin(), s.end());
    }

    // Wrap inputs as Variables for the graph forward pass
    std::vector<Variable> vars;
    vars.reserve(sample_inputs.size());
    for (auto& t : sample_inputs) {
        vars.emplace_back(t, false);
    }

    // Capture: all GPU work submitted between begin/end is recorded
    cuda_graph_->begin_capture();
    try {
        graph_->forward(vars);
    } catch (...) {
        // If forward fails during capture, we must still end capture to
        // leave the stream in a valid state. The graph will be unusable.
        try {
            cuda_graph_->end_capture();
        } catch (...) {
            // Ignore end_capture errors during cleanup
        }
        cuda_graph_.reset();
        captured_shapes_.clear();
        throw;
    }
    cuda_graph_->end_capture();
}

auto CompiledModule::replay_cuda_graph(std::vector<Tensor>& inputs) -> bool {
    if (!cuda_graph_ || !cuda_graph_->is_ready()) {
        return false;
    }

    // Validate input shapes match captured shapes
    if (inputs.size() != captured_shapes_.size()) {
        throw std::runtime_error(
            "CUDA graph replay: expected " +
            std::to_string(captured_shapes_.size()) + " inputs, got " +
            std::to_string(inputs.size()));
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        auto in_shape = inputs[i].shape();
        const auto& cap_shape = captured_shapes_[i];
        if (in_shape.size() != cap_shape.size() ||
            !std::equal(in_shape.begin(), in_shape.end(), cap_shape.begin())) {
            throw std::runtime_error(
                "CUDA graph replay: input " + std::to_string(i) +
                " shape mismatch (graph was captured with different shapes)");
        }
    }

    cuda_graph_->replay();
    return true;
}

auto CompiledModule::invalidate_cuda_graph() -> void {
    cuda_graph_.reset();
    captured_shapes_.clear();
}

auto CompiledModule::has_cuda_graph() const -> bool {
    return cuda_graph_ && cuda_graph_->is_ready();
}

// ============================================================================
// Free function convenience wrappers
// ============================================================================

auto optimize_for_inference(std::shared_ptr<CompiledModule> module) -> int {
    if (!module) {
        throw std::runtime_error("Cannot optimize null module");
    }
    return module->optimize_for_inference();
}

auto save(const std::shared_ptr<CompiledModule>& module, const std::string& path) -> void {
    if (!module) {
        throw std::runtime_error("Cannot save null module");
    }
    module->save(path);
}

auto load(const std::string& path) -> std::shared_ptr<CompiledModule> {
    return CompiledModule::load(path);
}

auto add_metadata(const std::shared_ptr<CompiledModule>& module,
                  const std::string& key, const std::string& value) -> void {
    if (!module) {
        throw std::runtime_error("Cannot add metadata to null module");
    }
    module->add_metadata(key, value);
}

auto get_metadata(const std::shared_ptr<CompiledModule>& module,
                  const std::string& key) -> std::string {
    if (!module) {
        throw std::runtime_error("Cannot get metadata from null module");
    }
    return module->get_metadata(key);
}

} // namespace jit
} // namespace tenzor
