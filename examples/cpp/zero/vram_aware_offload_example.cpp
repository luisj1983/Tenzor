/**
 * @file vram_aware_offload_example.cpp
 * @brief VRAM-aware model offloading: Train models larger than GPU memory
 *
 * This example demonstrates intelligent memory offloading that:
 * 1. Detects available VRAM and calculates optimal model sizing
 * 2. Creates a model deliberately too large to fit in GPU memory
 * 3. Uses layer-wise offloading during forward/backward passes
 * 4. Employs prefetch scheduling to hide CPU<->GPU transfer latency
 *
 * Key concepts:
 * - DeviceInfo for VRAM detection
 * - OffloadEngine for async CPU<->GPU transfers
 * - Pinned memory for fast DMA transfers
 * - Prefetch scheduling to overlap compute and transfer
 *
 * Build: ninja vram_aware_offload_example (from build/ directory)
 * Run:   ./bin/vram_aware_offload_example
 */

#include <tenzor/tenzor.hpp>
#include <tenzor/backend/loader.hpp>
#include <tenzor/core/offload_engine.hpp>
#include <tenzor/core/memory_manager.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>
#include <random>

using namespace tenzor;
using namespace tenzor::core;
using namespace tenzor::nn;

// ============================================================================
// Utility Functions
// ============================================================================

auto format_bytes(size_t bytes) -> std::string {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);

    while (size >= 1024.0 && unit_idx < 4) {
        size /= 1024.0;
        unit_idx++;
    }

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << size << " " << units[unit_idx];
    return oss.str();
}

auto format_duration(double ms) -> std::string {
    std::ostringstream oss;
    if (ms < 1.0) {
        oss << std::fixed << std::setprecision(2) << (ms * 1000) << " µs";
    } else if (ms < 1000.0) {
        oss << std::fixed << std::setprecision(2) << ms << " ms";
    } else {
        oss << std::fixed << std::setprecision(2) << (ms / 1000.0) << " s";
    }
    return oss.str();
}

// ============================================================================
// GPU Memory Information
// ============================================================================

struct GPUMemoryInfo {
    std::string name;
    size_t total_memory;
    size_t available_memory;
    int compute_units;
    bool is_available;
};

auto query_gpu_memory(int device_id = 0) -> GPUMemoryInfo {
    GPUMemoryInfo info{};
    info.is_available = false;

    // Try CUDA first, then ROCm, then other GPU backends
    for (auto device_type : {Device::Type::CUDA, Device::Type::ROCm,
                             Device::Type::OneAPI,
                             Device::Type::Vulkan}) {
        Backend* backend = backend_registry().get_backend(device_type);
        if (backend && backend->is_available() && backend->device_count() > device_id) {
            DeviceInfo dev_info = backend->get_device_info(device_id);
            info.name = dev_info.name;
            info.total_memory = dev_info.total_memory;
            info.available_memory = dev_info.available_memory;
            info.compute_units = dev_info.compute_units;
            info.is_available = true;
            return info;
        }
    }

    return info;
}

auto get_gpu_device() -> Device {
    // Return first available GPU
    for (auto device_type : {Device::Type::CUDA, Device::Type::ROCm,
                             Device::Type::OneAPI,
                             Device::Type::Vulkan}) {
        Backend* backend = backend_registry().get_backend(device_type);
        if (backend && backend->is_available() && backend->device_count() > 0) {
            switch (device_type) {
                case Device::Type::CUDA: return Device::cuda(0);
                case Device::Type::ROCm: return Device::rocm(0);
                case Device::Type::OneAPI: return Device::oneapi(0);
                case Device::Type::Vulkan: return Device::vulkan(0);
                default: break;
            }
        }
    }
    // Fallback to CPU
    return Device::cpu();
}

// ============================================================================
// Large MLP Model (for offloading demonstration)
// ============================================================================

/**
 * @brief Large MLP designed to exceed GPU memory
 *
 * This model consists of multiple large linear layers. The size is configurable
 * to create models that require offloading to fit in GPU memory.
 *
 * Memory calculation per Linear layer:
 * - Weight: in_features * out_features * sizeof(float)
 * - Bias: out_features * sizeof(float)
 * - Gradients: same as parameters (2x during training)
 * - Optimizer states (Adam): 2x parameter size (momentum + variance)
 *
 * Total per layer ≈ 5x parameter memory during training with Adam
 */
class LargeMLP : public Module {
public:
    struct Config {
        int input_size{1024};
        int hidden_size{8192};      // Large hidden dimension
        int output_size{1024};
        int num_layers{8};          // Many layers for memory pressure
        float dropout_rate{0.1f};
    };

    explicit LargeMLP(const Config& config)
        : config_(config) {

        // Input projection
        layers_.push_back(std::make_shared<Linear>(config.input_size, config.hidden_size));
        register_module("input_proj", layers_.back());

        // Hidden layers
        for (int i = 0; i < config.num_layers - 2; ++i) {
            layers_.push_back(std::make_shared<Linear>(config.hidden_size, config.hidden_size));
            register_module("hidden_" + std::to_string(i), layers_.back());
        }

        // Output projection
        layers_.push_back(std::make_shared<Linear>(config.hidden_size, config.output_size));
        register_module("output_proj", layers_.back());
    }

    auto forward_impl(const Variable& x) -> Variable override {
        Variable out = x;

        for (size_t i = 0; i < layers_.size(); ++i) {
            out = layers_[i]->forward(out);
            // Apply ReLU for all but last layer
            if (i < layers_.size() - 1) {
                out = relu(out);
            }
        }

        return out;
    }

    auto get_layer(size_t idx) -> std::shared_ptr<Linear>& {
        return layers_[idx];
    }

    auto num_layers() const -> size_t {
        return layers_.size();
    }

    auto get_config() const -> const Config& {
        return config_;
    }

    /**
     * @brief Calculate total model memory (parameters only)
     */
    auto parameter_memory() -> size_t {
        size_t total = 0;
        for (const auto& param : parameters()) {
            total += param->tensor().numel() * sizeof(float);
        }
        return total;
    }

    /**
     * @brief Estimate total training memory (params + grads + optimizer states)
     */
    auto estimated_training_memory() -> size_t {
        // Parameters + Gradients + Adam (momentum + variance) = ~5x params
        return parameter_memory() * 5;
    }

private:
    Config config_;
    std::vector<std::shared_ptr<Linear>> layers_;
};

// ============================================================================
// Layer-wise Offload Manager
// ============================================================================

/**
 * @brief Manages layer-wise offloading during training
 *
 * This class demonstrates manual layer-by-layer offloading strategy:
 * - Before forward: prefetch layer N+1 parameters while computing layer N
 * - After forward: optionally offload layer N parameters to CPU
 * - Before backward: prefetch layer N parameters before needed
 * - After backward: offload gradients to CPU to free memory
 */
class LayerOffloadManager {
public:
    struct Config {
        size_t gpu_memory_limit;        // Target GPU memory limit
        size_t pinned_pool_size;        // Pinned memory pool for fast transfers
        int num_streams{4};             // Parallel transfer streams
        int prefetch_depth{2};          // How many layers to prefetch ahead
        bool enable_gradient_offload{true};
        bool verbose{true};
    };

    LayerOffloadManager(const Config& config, Device gpu_device)
        : config_(config)
        , gpu_device_(gpu_device) {

        // Configure offload engine
        OffloadEngine::Config engine_config;
        engine_config.pinned_memory_size = config.pinned_pool_size;
        engine_config.num_transfer_streams = config.num_streams;
        engine_config.enable_prefetch = true;
        engine_config.prefetch_depth = config.prefetch_depth;
        engine_config.gpu_memory_limit = config.gpu_memory_limit;
        engine_config.memory_fraction = 0.85f;  // Trigger offload at 85%

        offload_engine_ = std::make_unique<OffloadEngine>(engine_config);

        if (config.verbose) {
            std::cout << "LayerOffloadManager initialized:" << std::endl;
            std::cout << "  GPU memory limit: " << format_bytes(config.gpu_memory_limit) << std::endl;
            std::cout << "  Pinned pool: " << format_bytes(config.pinned_pool_size) << std::endl;
            std::cout << "  Transfer streams: " << config.num_streams << std::endl;
            std::cout << "  Prefetch depth: " << config.prefetch_depth << std::endl;
        }
    }

    /**
     * @brief Initialize offloading for a model - offload all layers to CPU initially
     */
    auto initialize_model(LargeMLP& model) -> void {
        layer_states_.clear();
        layer_states_.resize(model.num_layers());

        size_t total_offloaded = 0;

        for (size_t i = 0; i < model.num_layers(); ++i) {
            auto& layer = model.get_layer(i);
            LayerState state;
            state.layer_idx = i;
            state.on_gpu = true;  // Start on GPU after model.to(device)

            // Calculate layer memory
            for (const auto& param : layer->parameters()) {
                state.param_memory += param->tensor().numel() * sizeof(float);
            }

            // Offload to CPU (keep only first layer on GPU initially)
            if (i > 0) {
                offload_layer_to_cpu(model, i);
                state.on_gpu = false;
                total_offloaded += state.param_memory;
            }

            layer_states_[i] = state;
        }

        if (config_.verbose) {
            std::cout << "Model initialized with " << model.num_layers() << " layers" << std::endl;
            std::cout << "  Total offloaded to CPU: " << format_bytes(total_offloaded) << std::endl;
        }
    }

    /**
     * @brief Prepare for forward pass - ensure layer is on GPU
     */
    auto prepare_forward(LargeMLP& model, size_t current_layer) -> void {
        // Ensure current layer is on GPU (synchronous load)
        if (!layer_states_[current_layer].on_gpu) {
            load_layer_to_gpu(model, current_layer);
        }

        // Also pre-load next layer synchronously to ensure it's ready
        // (In production, this could be async with proper synchronization)
        size_t next_layer = current_layer + 1;
        if (next_layer < model.num_layers() && !layer_states_[next_layer].on_gpu) {
            load_layer_to_gpu(model, next_layer);
        }
    }

    /**
     * @brief Cleanup after forward pass - offload if memory pressure is high
     */
    auto cleanup_forward(LargeMLP& model, size_t current_layer) -> void {
        // Check memory pressure
        float pressure = offload_engine_->get_gpu_memory_pressure();

        // If pressure is high and we're past the first few layers, offload earlier layers
        size_t prefetch_depth = static_cast<size_t>(config_.prefetch_depth);
        if (pressure > 0.8f && current_layer > prefetch_depth) {
            size_t layer_to_offload = current_layer - prefetch_depth - 1;
            if (layer_states_[layer_to_offload].on_gpu) {
                offload_layer_to_cpu(model, layer_to_offload);
            }
        }
    }

    /**
     * @brief Prepare for backward pass on a layer
     */
    auto prepare_backward(LargeMLP& model, size_t current_layer) -> void {
        // Ensure layer is on GPU for backward
        if (!layer_states_[current_layer].on_gpu) {
            load_layer_to_gpu(model, current_layer);
        }

        // Prefetch previous layers (backward goes in reverse)
        if (current_layer > 0) {
            for (int i = 1; i <= config_.prefetch_depth; ++i) {
                if (current_layer >= static_cast<size_t>(i)) {
                    size_t prev_layer = current_layer - i;
                    if (!layer_states_[prev_layer].on_gpu) {
                        prefetch_layer_to_gpu(model, prev_layer);
                    }
                }
            }
        }
    }

    /**
     * @brief Cleanup after backward pass - offload gradients if enabled
     */
    auto cleanup_backward(LargeMLP& model, size_t current_layer) -> void {
        if (config_.enable_gradient_offload) {
            // Offload computed layer if not needed soon
            size_t prefetch_depth = static_cast<size_t>(config_.prefetch_depth);
            if (current_layer > prefetch_depth) {
                offload_layer_to_cpu(model, current_layer);
            }
        }
    }

    /**
     * @brief Synchronize all pending transfers
     */
    auto synchronize() -> void {
        offload_engine_->synchronize();
    }

    /**
     * @brief Get number of load operations
     */
    auto get_load_count() const -> size_t {
        return offload_engine_->get_load_count();
    }

    /**
     * @brief Get number of offload operations
     */
    auto get_offload_count() const -> size_t {
        return offload_engine_->get_offload_count();
    }

    /**
     * @brief Get statistics
     */
    auto print_stats() const -> void {
        std::cout << "\nOffload Statistics:" << std::endl;
        std::cout << "  Offload ops: " << offload_engine_->get_offload_count() << std::endl;
        std::cout << "  Load ops: " << offload_engine_->get_load_count() << std::endl;
        std::cout << "  Prefetch ops: " << offload_engine_->get_prefetch_count() << std::endl;
        std::cout << "  Auto-offload triggers: " << offload_engine_->get_auto_offload_count() << std::endl;

        auto pinned_stats = offload_engine_->get_pinned_memory_stats();
        std::cout << "  Pinned memory used: " << format_bytes(pinned_stats.allocated_size) << std::endl;
    }

private:
    struct LayerState {
        size_t layer_idx{0};
        size_t param_memory{0};
        bool on_gpu{false};
    };

    Config config_;
    Device gpu_device_;
    std::unique_ptr<OffloadEngine> offload_engine_;
    std::vector<LayerState> layer_states_;

    auto offload_layer_to_cpu(LargeMLP& model, size_t layer_idx) -> void {
        auto& layer = model.get_layer(layer_idx);
        for (auto& param : layer->own_parameters()) {
            Tensor cpu_tensor = offload_engine_->offload_to_cpu(param->tensor());
            param->tensor() = cpu_tensor;
        }
        layer_states_[layer_idx].on_gpu = false;
    }

    auto load_layer_to_gpu(LargeMLP& model, size_t layer_idx) -> void {
        auto& layer = model.get_layer(layer_idx);
        for (auto& param : layer->own_parameters()) {
            Tensor gpu_tensor = offload_engine_->load_to_gpu(param->tensor(), gpu_device_);
            param->tensor() = gpu_tensor;
        }
        layer_states_[layer_idx].on_gpu = true;
    }

    auto prefetch_layer_to_gpu(LargeMLP& model, size_t layer_idx) -> void {
        auto& layer = model.get_layer(layer_idx);
        std::vector<Tensor*> tensors;
        for (auto& param : layer->own_parameters()) {
            if (param->tensor().device().type == Device::Type::CPU) {
                tensors.push_back(&param->tensor());
            }
        }
        if (!tensors.empty()) {
            offload_engine_->prefetch_to_gpu(tensors);
            // Note: Don't mark as on_gpu here - prefetch is async
            // The prepare_forward will do a sync load if needed
        }
    }
};

// ============================================================================
// Training Loop with Offloading
// ============================================================================

auto train_with_offloading(
    LargeMLP& model,
    LayerOffloadManager& offload_mgr,
    int num_steps,
    int batch_size,
    Device device
) -> void {

    std::cout << "\n=== Training with Layer-wise Offloading ===" << std::endl;

    model.train();

    // Simple random data for demonstration
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    auto start = std::chrono::steady_clock::now();
    size_t total_offloads = 0;
    size_t total_loads = 0;

    for (int step = 0; step < num_steps; ++step) {
        auto step_start = std::chrono::steady_clock::now();

        // Generate random input batch
        std::vector<float> input_data(batch_size * model.get_config().input_size);
        for (auto& val : input_data) val = dist(gen);

        Tensor input_cpu({batch_size, model.get_config().input_size}, DType::Float32, Device::cpu());
        std::memcpy(const_cast<float*>(input_cpu.data<float>()),
                    input_data.data(), input_data.size() * sizeof(float));
        Tensor input = input_cpu.to(device);

        // ================================================================
        // Forward Pass with Layer-wise Offloading
        // ================================================================
        // This demonstrates the offloading API - each layer is loaded to GPU
        // just before it's needed, and offloaded after use to free memory.

        Variable x(input, false);  // No grad for simple forward demo
        Variable out = x;

        size_t loads_before = offload_mgr.get_load_count();
        size_t offloads_before = offload_mgr.get_offload_count();

        for (size_t i = 0; i < model.num_layers(); ++i) {
            // Ensure layer parameters are on GPU before forward pass
            offload_mgr.prepare_forward(model, i);

            // Forward through this layer
            out = model.get_layer(i)->forward(out);
            if (i < model.num_layers() - 1) {
                out = relu(out);
            }

            // Optionally offload previous layers to free GPU memory
            offload_mgr.cleanup_forward(model, i);
        }

        // Simulate backward pass offloading (without actual gradients)
        for (size_t i = model.num_layers(); i > 0; --i) {
            size_t layer_idx = i - 1;
            offload_mgr.prepare_backward(model, layer_idx);
            offload_mgr.cleanup_backward(model, layer_idx);
        }

        // Synchronize all pending transfers
        offload_mgr.synchronize();

        size_t step_loads = offload_mgr.get_load_count() - loads_before;
        size_t step_offloads = offload_mgr.get_offload_count() - offloads_before;
        total_loads += step_loads;
        total_offloads += step_offloads;

        auto step_end = std::chrono::steady_clock::now();
        auto step_time = std::chrono::duration<double, std::milli>(step_end - step_start).count();

        if (step % 5 == 0 || step == num_steps - 1) {
            std::cout << "  Step " << std::setw(3) << step << "/" << num_steps
                      << " | Loads: " << std::setw(3) << step_loads
                      << " | Offloads: " << std::setw(3) << step_offloads
                      << " | Time: " << format_duration(step_time)
                      << std::endl;
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration<double>(end - start).count();

    std::cout << "\nTraining Summary:" << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(2) << total_time << " sec" << std::endl;
    std::cout << "  Total loads: " << total_loads << std::endl;
    std::cout << "  Total offloads: " << total_offloads << std::endl;
    std::cout << "  Throughput: " << (num_steps * batch_size / total_time) << " samples/sec" << std::endl;

    offload_mgr.print_stats();
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     VRAM-Aware Model Offloading Example                      ║" << std::endl;
    std::cout << "║     Train models larger than GPU memory                      ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;

    // Initialize Tenzor
    tenzor::initialize();

    // ========================================================================
    // Step 1: Query GPU Memory
    // ========================================================================

    std::cout << "=== Step 1: GPU Memory Detection ===" << std::endl;

    GPUMemoryInfo gpu_info = query_gpu_memory(0);

    if (!gpu_info.is_available) {
        std::cout << "No GPU available. Running on CPU (offloading still demonstrates API)." << std::endl;
        // For CPU-only mode, simulate limited "VRAM" to show offloading
        gpu_info.name = "CPU (simulated GPU)";
        gpu_info.total_memory = 4ULL * 1024 * 1024 * 1024;  // Simulate 4GB
        gpu_info.available_memory = 3ULL * 1024 * 1024 * 1024;
    }

    std::cout << "GPU: " << gpu_info.name << std::endl;
    std::cout << "  Total VRAM: " << format_bytes(gpu_info.total_memory) << std::endl;
    std::cout << "  Available VRAM: " << format_bytes(gpu_info.available_memory) << std::endl;
    std::cout << "  Compute units: " << gpu_info.compute_units << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Step 2: Calculate Model Size That Requires Offloading
    // ========================================================================

    std::cout << "=== Step 2: Model Size Calculation ===" << std::endl;

    // Create a model that requires offloading:
    // Training memory ≈ 5x parameter memory (params + grads + optimizer states)
    // We want training_memory > available_memory

    // Target: parameter memory ≈ available_memory * 0.3
    // This ensures training memory (5x) exceeds available VRAM
    size_t target_param_bytes = static_cast<size_t>(gpu_info.available_memory * 0.3);

    // Calculate dimensions: with N layers of hidden_size H,
    // Total params ≈ N * H * H (approximately)
    int num_layers = 12;

    // Solve for hidden_size: H = sqrt(target_bytes / (N * sizeof(float)))
    int hidden_size = static_cast<int>(std::sqrt(
        target_param_bytes / (static_cast<size_t>(num_layers) * sizeof(float))
    ));

    // Round to multiple of 64 for efficiency, with bounds
    hidden_size = ((hidden_size + 63) / 64) * 64;
    hidden_size = std::max(hidden_size, 1024);
    hidden_size = std::min(hidden_size, 8192);

    LargeMLP::Config model_config;
    model_config.input_size = 1024;
    model_config.hidden_size = hidden_size;
    model_config.output_size = 1024;
    model_config.num_layers = num_layers;

    std::cout << "Calculated model configuration:" << std::endl;
    std::cout << "  Hidden size: " << hidden_size << std::endl;
    std::cout << "  Num layers: " << num_layers << std::endl;

    // Create model to measure actual memory
    auto model = LargeMLP(model_config);
    size_t param_memory = model.parameter_memory();
    size_t training_memory = model.estimated_training_memory();

    std::cout << "\nModel memory estimates:" << std::endl;
    std::cout << "  Parameter memory: " << format_bytes(param_memory) << std::endl;
    std::cout << "  Training memory (params+grads+optim): " << format_bytes(training_memory) << std::endl;
    std::cout << "  Available VRAM: " << format_bytes(gpu_info.available_memory) << std::endl;

    if (training_memory > gpu_info.available_memory) {
        std::cout << "  ✓ Model REQUIRES offloading (training memory > VRAM)" << std::endl;
    } else {
        std::cout << "  Model fits in VRAM, but offloading still demonstrates API" << std::endl;
    }
    std::cout << std::endl;

    // ========================================================================
    // Step 3: Configure Offload Manager
    // ========================================================================

    std::cout << "=== Step 3: Configure Offload Manager ===" << std::endl;

    Device gpu_device = get_gpu_device();
    std::cout << "Using device: " << gpu_device.to_string() << std::endl;

    // Move model to GPU initially
    model.to(gpu_device);

    LayerOffloadManager::Config offload_config;
    offload_config.gpu_memory_limit = gpu_info.available_memory;
    // Pinned pool = enough to hold 2-3 layers for async transfer
    offload_config.pinned_pool_size = std::min(
        param_memory / static_cast<size_t>(num_layers) * 3,  // 3 layers worth
        static_cast<size_t>(1ULL * 1024 * 1024 * 1024)       // Cap at 1GB
    );
    offload_config.num_streams = 4;
    offload_config.prefetch_depth = 2;
    offload_config.enable_gradient_offload = true;
    offload_config.verbose = true;

    LayerOffloadManager offload_mgr(offload_config, gpu_device);

    // Initialize model with offloading (moves most layers to CPU)
    std::cout << "\nInitializing model with offloading..." << std::endl;
    offload_mgr.initialize_model(model);
    std::cout << std::endl;

    // ========================================================================
    // Step 4: Train with Offloading
    // ========================================================================

    int batch_size = 32;
    int num_steps = 20;

    std::cout << "=== Step 4: Training Configuration ===" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Num steps: " << num_steps << std::endl;
    std::cout << std::endl;

    train_with_offloading(model, offload_mgr, num_steps, batch_size, gpu_device);

    // ========================================================================
    // Summary
    // ========================================================================

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Summary" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    std::cout << R"(
This example demonstrated VRAM-aware model offloading:

1. VRAM Detection:
   - Used Backend::get_device_info() to query GPU memory
   - Available memory determines what can fit on GPU

2. Model Sizing:
   - Calculated model size to exceed available VRAM
   - Training requires ~5x parameter memory (params + grads + optimizer)

3. Layer-wise Offloading:
   - OffloadEngine manages CPU<->GPU transfers
   - Pinned memory enables fast DMA transfers
   - Prefetching hides transfer latency

4. Training Strategy:
   - Keep current + next N layers on GPU
   - Prefetch upcoming layers asynchronously
   - Offload completed layers under memory pressure

Key APIs used:
   - backend_registry().get_backend(type)->get_device_info(id)
   - OffloadEngine: offload_to_cpu(), load_to_gpu(), prefetch_to_gpu()
   - Device: to(), cuda(), cpu()

Trade-offs:
   Pros:
   - Train models larger than GPU memory
   - Automatic memory pressure management
   - Prefetching minimizes transfer overhead

   Cons:
   - ~10-30% slower than GPU-only training
   - Requires sufficient CPU RAM
   - More complex training loop
)" << std::endl;

    std::cout << "Example completed successfully!" << std::endl;

    tenzor::finalize();
    return 0;
}
