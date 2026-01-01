#include "tenzor/tenzor.hpp"
#include "tenzor/backend/registry.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include <iostream>
#include <filesystem>
#include <dlfcn.h>

namespace tenzor {

// Flag to track initialization
static bool g_initialized = false;

auto initialize() -> void {
    if (g_initialized) {
        return;
    }

    std::cout << "Initializing Tenzor library v1.0.0" << std::endl;

    // Load CPU backend dynamically
    auto& loader = backend_registry();

    // Try to load CPU backend from bin directory (same directory as executables)
    std::filesystem::path bin_path = "/home/lee/Projects/Tenzor/bin";
    std::filesystem::path cpu_backend_path = bin_path / "tenzor_backend_cpu.so";

    if (!std::filesystem::exists(cpu_backend_path)) {
        // Try build directory
        bin_path = "/home/lee/Projects/Tenzor/build/bin";
        cpu_backend_path = bin_path / "tenzor_backend_cpu.so";
    }

    if (!std::filesystem::exists(cpu_backend_path)) {
        // Try current directory
        cpu_backend_path = "./tenzor_backend_cpu.so";
    }

    std::cout << "Loading CPU backend from: " << cpu_backend_path << std::endl;

    auto result = loader.load_backend(cpu_backend_path);
    if (!result) {
        std::cerr << "Error: Failed to load CPU backend: " << result.error() << std::endl;
        throw std::runtime_error("Failed to initialize Tenzor: CPU backend not available");
    }

    // Register the loaded backend
    auto cpu_backend_unique = std::move(result.value());
    auto* cpu_backend_ptr = cpu_backend_unique.get();

    // Register by name
    loader.register_backend(cpu_backend_ptr->name(), std::move(cpu_backend_unique));

    std::cout << "CPU backend registered: " << cpu_backend_ptr->name() << std::endl;

    // Now cpu_backend_ptr points to the registered backend
    auto* cpu_backend = cpu_backend_ptr;

    // =========================================================================
    // NEW O(1) DISPATCH TABLE REGISTRATION
    // =========================================================================
    // Register CPU backend with the dispatch table registry
    DispatchTableRegistry::register_backend(Device::Type::CPU, cpu_backend);

    // Populate dispatch table with direct kernel function pointers
    auto& cpu_table = DispatchTableRegistry::get_table(Device::Type::CPU);

    // Load the register_kernels function from the backend shared library
    void* handle = dlopen(cpu_backend_path.c_str(), RTLD_NOW | RTLD_NOLOAD);
    if (handle) {
        using RegisterFn = void(*)(BackendDispatchTable*);
        auto register_fn = reinterpret_cast<RegisterFn>(dlsym(handle, "register_kernels"));
        if (register_fn) {
            register_fn(&cpu_table);
            std::cout << "CPU dispatch table initialized with O(1) lookup" << std::endl;
        } else {
            std::cerr << "Warning: Could not find register_kernels in CPU backend" << std::endl;
        }
        dlclose(handle);
    } else {
        std::cerr << "Warning: Could not reopen CPU backend for kernel registration" << std::endl;
    }

    // =========================================================================
    // LEGACY DISPATCH REGISTRATION (for backwards compatibility during migration)
    // TODO: Remove this section once all code uses OpId-based dispatch
    // =========================================================================

    // Register all CPU operations with the OperationRegistry
    auto& registry = operation_registry();

    std::cout << "Registering CPU kernels with operation registry" << std::endl;

    // Register operations by forwarding to backend
    registry.register_kernel("add", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("add", inputs, attrs);
        });

    registry.register_kernel("sub", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("sub", inputs, attrs);
        });

    registry.register_kernel("mul", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("mul", inputs, attrs);
        });

    registry.register_kernel("div", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("div", inputs, attrs);
        });

    registry.register_kernel("matmul", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("matmul", inputs, attrs);
        });

    registry.register_kernel("sum", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("sum", inputs, attrs);
        });

    registry.register_kernel("mean", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("mean", inputs, attrs);
        });

    registry.register_kernel("max", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("max", inputs, attrs);
        });

    registry.register_kernel("min", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("min", inputs, attrs);
        });

    registry.register_kernel("argmax", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("argmax", inputs, attrs);
        });

    registry.register_kernel("argmin", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("argmin", inputs, attrs);
        });

    registry.register_kernel("prod", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("prod", inputs, attrs);
        });

    registry.register_kernel("var", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("var", inputs, attrs);
        });

    registry.register_kernel("std", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("std", inputs, attrs);
        });

    registry.register_kernel("norm", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("norm", inputs, attrs);
        });

    // Activation functions
    registry.register_kernel("relu", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("relu", inputs, attrs);
        });

    registry.register_kernel("relu_backward", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("relu_backward", inputs, attrs);
        });

    registry.register_kernel("sigmoid", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("sigmoid", inputs, attrs);
        });

    registry.register_kernel("sigmoid_backward", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("sigmoid_backward", inputs, attrs);
        });

    registry.register_kernel("tanh", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("tanh", inputs, attrs);
        });

    registry.register_kernel("tanh_backward", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("tanh_backward", inputs, attrs);
        });

    registry.register_kernel("gelu", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("gelu", inputs, attrs);
        });

    registry.register_kernel("gelu_backward", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("gelu_backward", inputs, attrs);
        });

    registry.register_kernel("swish", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("swish", inputs, attrs);
        });

    registry.register_kernel("swish_backward", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("swish_backward", inputs, attrs);
        });

    registry.register_kernel("leaky_relu", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("leaky_relu", inputs, attrs);
        });

    registry.register_kernel("leaky_relu_backward", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("leaky_relu_backward", inputs, attrs);
        });

    registry.register_kernel("elu", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("elu", inputs, attrs);
        });

    registry.register_kernel("elu_backward", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("elu_backward", inputs, attrs);
        });

    registry.register_kernel("selu", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("selu", inputs, attrs);
        });

    registry.register_kernel("selu_backward", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("selu_backward", inputs, attrs);
        });

    registry.register_kernel("mish", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("mish", inputs, attrs);
        });

    registry.register_kernel("mish_backward", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("mish_backward", inputs, attrs);
        });

    registry.register_kernel("softplus", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("softplus", inputs, attrs);
        });

    registry.register_kernel("softplus_backward", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("softplus_backward", inputs, attrs);
        });

    registry.register_kernel("softmax", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("softmax", inputs, attrs);
        });

    registry.register_kernel("softmax_backward", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("softmax_backward", inputs, attrs);
        });

    registry.register_kernel("log_softmax", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("log_softmax", inputs, attrs);
        });

    registry.register_kernel("log_softmax_backward", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("log_softmax_backward", inputs, attrs);
        });

    registry.register_kernel("neg", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("neg", inputs, attrs);
        });

    registry.register_kernel("abs", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("abs", inputs, attrs);
        });

    registry.register_kernel("sign", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("sign", inputs, attrs);
        });

    // Math operations
    registry.register_kernel("sqrt", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("sqrt", inputs, attrs);
        });

    registry.register_kernel("exp", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("exp", inputs, attrs);
        });

    registry.register_kernel("log", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("log", inputs, attrs);
        });

    registry.register_kernel("pow", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("pow", inputs, attrs);
        });

    // Trigonometric operations
    registry.register_kernel("sin", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("sin", inputs, attrs);
        });

    registry.register_kernel("cos", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("cos", inputs, attrs);
        });

    registry.register_kernel("tan", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("tan", inputs, attrs);
        });

    registry.register_kernel("asin", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("asin", inputs, attrs);
        });

    registry.register_kernel("acos", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("acos", inputs, attrs);
        });

    registry.register_kernel("atan", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("atan", inputs, attrs);
        });

    // Hyperbolic operations
    registry.register_kernel("sinh", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("sinh", inputs, attrs);
        });

    registry.register_kernel("cosh", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("cosh", inputs, attrs);
        });

    // Rounding operations
    registry.register_kernel("round", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("round", inputs, attrs);
        });

    registry.register_kernel("floor", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("floor", inputs, attrs);
        });

    registry.register_kernel("ceil", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("ceil", inputs, attrs);
        });

    // Reciprocal operation
    registry.register_kernel("reciprocal", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("reciprocal", inputs, attrs);
        });

    // In-place operations
    registry.register_kernel("add_inplace", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("add_inplace", inputs, attrs);
        });

    registry.register_kernel("mul_inplace", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("mul_inplace", inputs, attrs);
        });

    registry.register_kernel("sub_inplace", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("sub_inplace", inputs, attrs);
        });

    registry.register_kernel("div_inplace", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("div_inplace", inputs, attrs);
        });

    registry.register_kernel("dot", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("dot", inputs, attrs);
        });

    registry.register_kernel("clamp", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("clamp", inputs, attrs);
        });

    registry.register_kernel("clamp_min", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("clamp_min", inputs, attrs);
        });

    registry.register_kernel("clamp_max", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("clamp_max", inputs, attrs);
        });

    // Comparison operations
    registry.register_kernel("eq", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("eq", inputs, attrs);
        });

    registry.register_kernel("ne", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("ne", inputs, attrs);
        });

    registry.register_kernel("lt", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("lt", inputs, attrs);
        });

    registry.register_kernel("le", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("le", inputs, attrs);
        });

    registry.register_kernel("gt", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("gt", inputs, attrs);
        });

    registry.register_kernel("ge", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("ge", inputs, attrs);
        });

    // Transform operations
    registry.register_kernel("contiguous", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("contiguous", inputs, attrs);
        });

    registry.register_kernel("fill", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("fill", inputs, attrs);
        });

    registry.register_kernel("clone", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("clone", inputs, attrs);
        });

    registry.register_kernel("reshape", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("reshape", inputs, attrs);
        });

    registry.register_kernel("transpose", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("transpose", inputs, attrs);
        });

    registry.register_kernel("permute", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("permute", inputs, attrs);
        });

    registry.register_kernel("squeeze", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("squeeze", inputs, attrs);
        });

    registry.register_kernel("unsqueeze", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("unsqueeze", inputs, attrs);
        });

    registry.register_kernel("index_select", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("index_select", inputs, attrs);
        });

    registry.register_kernel("gather", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("gather", inputs, attrs);
        });

    registry.register_kernel("scatter", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("scatter", inputs, attrs);
        });

    registry.register_kernel("masked_select", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("masked_select", inputs, attrs);
        });

    registry.register_kernel("masked_fill", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("masked_fill", inputs, attrs);
        });

    registry.register_kernel("where", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("where", inputs, attrs);
        });

    registry.register_kernel("cat", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("cat", inputs, attrs);
        });

    registry.register_kernel("zeros", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("zeros", inputs, attrs);
        });

    // BatchNorm2d operations
    registry.register_kernel("batchnorm2d_mean_var", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("batchnorm2d_mean_var", inputs, attrs);
        });

    registry.register_kernel("batchnorm2d_forward", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("batchnorm2d_forward", inputs, attrs);
        });

    registry.register_kernel("batchnorm2d_forward_affine", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("batchnorm2d_forward_affine", inputs, attrs);
        });

    registry.register_kernel("batchnorm2d_update_running_stats", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("batchnorm2d_update_running_stats", inputs, attrs);
        });

    registry.register_kernel("batchnorm2d_backward", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("batchnorm2d_backward", inputs, attrs);
        });

    // Conv2d operations
    registry.register_kernel("conv2d_forward", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("conv2d_forward", inputs, attrs);
        });

    registry.register_kernel("conv2d_backward_input", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("conv2d_backward_input", inputs, attrs);
        });

    registry.register_kernel("conv2d_backward_weight", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("conv2d_backward_weight", inputs, attrs);
        });

    registry.register_kernel("conv2d_backward_bias", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("conv2d_backward_bias", inputs, attrs);
        });

    registry.register_kernel("conv_transpose2d_forward", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("conv_transpose2d_forward", inputs, attrs);
        });

    registry.register_kernel("im2col", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("im2col", inputs, attrs);
        });

    registry.register_kernel("col2im", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("col2im", inputs, attrs);
        });

    // Fused operations
    registry.register_kernel("fused_linear_relu", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("fused_linear_relu", inputs, attrs);
        });

    registry.register_kernel("fused_conv2d_relu", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("fused_conv2d_relu", inputs, attrs);
        });

    registry.register_kernel("fused_batchnorm_relu", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("fused_batchnorm_relu", inputs, attrs);
        });

    registry.register_kernel("fused_softmax_cross_entropy", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("fused_softmax_cross_entropy", inputs, attrs);
        });

    registry.register_kernel("fused_add_relu", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("fused_add_relu", inputs, attrs);
        });

    registry.register_kernel("fused_gelu", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("fused_gelu", inputs, attrs);
        });

    registry.register_kernel("fused_layer_norm", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("fused_layer_norm", inputs, attrs);
        });

    // Try to load CUDA backend if available
    std::filesystem::path cuda_backend_path = bin_path / "tenzor_backend_cuda.so";

    if (std::filesystem::exists(cuda_backend_path)) {
        std::cout << "Loading CUDA backend from: " << cuda_backend_path << std::endl;

        auto cuda_result = loader.load_backend(cuda_backend_path);
        if (cuda_result) {
            auto cuda_backend_unique = std::move(cuda_result.value());
            auto* cuda_backend_ptr = cuda_backend_unique.get();

            // Check if CUDA is actually available
            if (cuda_backend_ptr->is_available()) {
                loader.register_backend(cuda_backend_ptr->name(), std::move(cuda_backend_unique));
                std::cout << "CUDA backend registered: " << cuda_backend_ptr->name() << std::endl;
                std::cout << "Found " << cuda_backend_ptr->device_count() << " CUDA device(s)" << std::endl;

                auto* cuda_backend = cuda_backend_ptr;

                // =========================================================================
                // O(1) DISPATCH TABLE REGISTRATION FOR CUDA
                // =========================================================================
                DispatchTableRegistry::register_backend(Device::Type::CUDA, cuda_backend);
                auto& cuda_table = DispatchTableRegistry::get_table(Device::Type::CUDA);

                // Load the register_kernels function from the CUDA backend shared library
                void* cuda_handle = dlopen(cuda_backend_path.c_str(), RTLD_NOW | RTLD_NOLOAD);
                if (cuda_handle) {
                    using RegisterFn = void(*)(BackendDispatchTable*);
                    auto register_fn = reinterpret_cast<RegisterFn>(dlsym(cuda_handle, "register_kernels"));
                    if (register_fn) {
                        register_fn(&cuda_table);
                        std::cout << "CUDA dispatch table initialized with O(1) lookup" << std::endl;
                    } else {
                        std::cerr << "Warning: Could not find register_kernels in CUDA backend" << std::endl;
                    }
                    dlclose(cuda_handle);
                } else {
                    std::cerr << "Warning: Could not reopen CUDA backend for kernel registration" << std::endl;
                }

                // =========================================================================
                // LEGACY DISPATCH REGISTRATION (for backwards compatibility)
                // =========================================================================
                std::cout << "Registering CUDA kernels with operation registry" << std::endl;

                registry.register_kernel("add", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("add", inputs, attrs);
                    });

                registry.register_kernel("sub", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("sub", inputs, attrs);
                    });

                registry.register_kernel("mul", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("mul", inputs, attrs);
                    });

                registry.register_kernel("div", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("div", inputs, attrs);
                    });

                registry.register_kernel("matmul", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("matmul", inputs, attrs);
                    });

                registry.register_kernel("sum", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("sum", inputs, attrs);
                    });

                registry.register_kernel("mean", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("mean", inputs, attrs);
                    });

                registry.register_kernel("max", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("max", inputs, attrs);
                    });

                registry.register_kernel("min", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("min", inputs, attrs);
                    });

                registry.register_kernel("argmax", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("argmax", inputs, attrs);
                    });

                registry.register_kernel("argmin", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("argmin", inputs, attrs);
                    });

                registry.register_kernel("prod", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("prod", inputs, attrs);
                    });

                registry.register_kernel("var", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("var", inputs, attrs);
                    });

                registry.register_kernel("std", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("std", inputs, attrs);
                    });

                registry.register_kernel("norm", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("norm", inputs, attrs);
                    });

                // Activation functions
                registry.register_kernel("relu", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("relu", inputs, attrs);
                    });

                registry.register_kernel("relu_backward", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("relu_backward", inputs, attrs);
                    });

                registry.register_kernel("sigmoid", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("sigmoid", inputs, attrs);
                    });

                registry.register_kernel("sigmoid_backward", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("sigmoid_backward", inputs, attrs);
                    });

                registry.register_kernel("tanh", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("tanh", inputs, attrs);
                    });

                registry.register_kernel("tanh_backward", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("tanh_backward", inputs, attrs);
                    });

                registry.register_kernel("gelu", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("gelu", inputs, attrs);
                    });

                registry.register_kernel("gelu_backward", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("gelu_backward", inputs, attrs);
                    });

                registry.register_kernel("swish", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("swish", inputs, attrs);
                    });

                registry.register_kernel("swish_backward", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("swish_backward", inputs, attrs);
                    });

                registry.register_kernel("leaky_relu", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("leaky_relu", inputs, attrs);
                    });

                registry.register_kernel("leaky_relu_backward", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("leaky_relu_backward", inputs, attrs);
                    });

                registry.register_kernel("elu", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("elu", inputs, attrs);
                    });

                registry.register_kernel("elu_backward", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("elu_backward", inputs, attrs);
                    });

                registry.register_kernel("selu", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("selu", inputs, attrs);
                    });

                registry.register_kernel("selu_backward", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("selu_backward", inputs, attrs);
                    });

                registry.register_kernel("mish", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("mish", inputs, attrs);
                    });

                registry.register_kernel("mish_backward", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("mish_backward", inputs, attrs);
                    });

                registry.register_kernel("softplus", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("softplus", inputs, attrs);
                    });

                registry.register_kernel("softplus_backward", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("softplus_backward", inputs, attrs);
                    });

                registry.register_kernel("softmax", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("softmax", inputs, attrs);
                    });

                registry.register_kernel("softmax_backward", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("softmax_backward", inputs, attrs);
                    });

                registry.register_kernel("log_softmax", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("log_softmax", inputs, attrs);
                    });

                registry.register_kernel("log_softmax_backward", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("log_softmax_backward", inputs, attrs);
                    });

                registry.register_kernel("neg", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("neg", inputs, attrs);
                    });

                registry.register_kernel("abs", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("abs", inputs, attrs);
                    });

                registry.register_kernel("sign", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("sign", inputs, attrs);
                    });

                // Math operations
                registry.register_kernel("sqrt", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("sqrt", inputs, attrs);
                    });

                registry.register_kernel("exp", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("exp", inputs, attrs);
                    });

                registry.register_kernel("log", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("log", inputs, attrs);
                    });

                registry.register_kernel("pow", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("pow", inputs, attrs);
                    });

                // Trigonometric functions
                registry.register_kernel("sin", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("sin", inputs, attrs);
                    });

                registry.register_kernel("cos", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("cos", inputs, attrs);
                    });

                registry.register_kernel("tan", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("tan", inputs, attrs);
                    });

                registry.register_kernel("asin", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("asin", inputs, attrs);
                    });

                registry.register_kernel("acos", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("acos", inputs, attrs);
                    });

                registry.register_kernel("atan", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("atan", inputs, attrs);
                    });

                registry.register_kernel("sinh", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("sinh", inputs, attrs);
                    });

                registry.register_kernel("cosh", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("cosh", inputs, attrs);
                    });

                // Rounding functions
                registry.register_kernel("ceil", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("ceil", inputs, attrs);
                    });

                registry.register_kernel("floor", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("floor", inputs, attrs);
                    });

                registry.register_kernel("round", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("round", inputs, attrs);
                    });

                registry.register_kernel("trunc", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("trunc", inputs, attrs);
                    });

                registry.register_kernel("reciprocal", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("reciprocal", inputs, attrs);
                    });

                registry.register_kernel("clamp_min", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("clamp_min", inputs, attrs);
                    });

                registry.register_kernel("clamp_max", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("clamp_max", inputs, attrs);
                    });

                // In-place operations
                registry.register_kernel("add_inplace", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("add_inplace", inputs, attrs);
                    });

                registry.register_kernel("sub_inplace", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("sub_inplace", inputs, attrs);
                    });

                registry.register_kernel("mul_inplace", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("mul_inplace", inputs, attrs);
                    });

                registry.register_kernel("div_inplace", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("div_inplace", inputs, attrs);
                    });

                registry.register_kernel("dot", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("dot", inputs, attrs);
                    });

                registry.register_kernel("clamp", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("clamp", inputs, attrs);
                    });

                // Comparison operations
                registry.register_kernel("eq", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("eq", inputs, attrs);
                    });

                registry.register_kernel("ne", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("ne", inputs, attrs);
                    });

                registry.register_kernel("lt", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("lt", inputs, attrs);
                    });

                registry.register_kernel("le", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("le", inputs, attrs);
                    });

                registry.register_kernel("gt", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("gt", inputs, attrs);
                    });

                registry.register_kernel("ge", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("ge", inputs, attrs);
                    });

                // Transform operations
                registry.register_kernel("expand", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("expand", inputs, attrs);
                    });

                registry.register_kernel("repeat", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("repeat", inputs, attrs);
                    });

                registry.register_kernel("contiguous", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("contiguous", inputs, attrs);
                    });

                registry.register_kernel("fill", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("fill", inputs, attrs);
                    });

                registry.register_kernel("clone", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("clone", inputs, attrs);
                    });

                registry.register_kernel("reshape", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("reshape", inputs, attrs);
                    });

                registry.register_kernel("transpose", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("transpose", inputs, attrs);
                    });

                registry.register_kernel("permute", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("permute", inputs, attrs);
                    });

                registry.register_kernel("squeeze", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("squeeze", inputs, attrs);
                    });

                registry.register_kernel("unsqueeze", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("unsqueeze", inputs, attrs);
                    });

                registry.register_kernel("index_select", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("index_select", inputs, attrs);
                    });

                registry.register_kernel("gather", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("gather", inputs, attrs);
                    });

                registry.register_kernel("scatter", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("scatter", inputs, attrs);
                    });

                registry.register_kernel("masked_select", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("masked_select", inputs, attrs);
                    });

                registry.register_kernel("masked_fill", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("masked_fill", inputs, attrs);
                    });

                registry.register_kernel("where", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("where", inputs, attrs);
                    });

                registry.register_kernel("cat", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("cat", inputs, attrs);
                    });

                registry.register_kernel("zeros", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("zeros", inputs, attrs);
                    });

                // BatchNorm2d operations
                registry.register_kernel("batchnorm2d_mean_var", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("batchnorm2d_mean_var", inputs, attrs);
                    });

                registry.register_kernel("batchnorm2d_forward", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("batchnorm2d_forward", inputs, attrs);
                    });

                registry.register_kernel("batchnorm2d_forward_affine", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("batchnorm2d_forward_affine", inputs, attrs);
                    });

                registry.register_kernel("batchnorm2d_update_running_stats", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("batchnorm2d_update_running_stats", inputs, attrs);
                    });

                registry.register_kernel("batchnorm2d_backward", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("batchnorm2d_backward", inputs, attrs);
                    });

                // Conv2d operations
                registry.register_kernel("im2col", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("im2col", inputs, attrs);
                    });

                registry.register_kernel("col2im", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("col2im", inputs, attrs);
                    });

                registry.register_kernel("conv2d_forward", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("conv2d_forward", inputs, attrs);
                    });

                registry.register_kernel("conv2d_backward_input", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("conv2d_backward_input", inputs, attrs);
                    });

                registry.register_kernel("conv2d_backward_weight", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("conv2d_backward_weight", inputs, attrs);
                    });

                // Pooling operations
                registry.register_kernel("adaptive_avg_pool2d", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("adaptive_avg_pool2d", inputs, attrs);
                    });

                registry.register_kernel("adaptive_avg_pool2d_backward", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("adaptive_avg_pool2d_backward", inputs, attrs);
                    });

                // Vision operations
                registry.register_kernel("gather_relative_position_bias", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("gather_relative_position_bias", inputs, attrs);
                    });

                std::cout << "CUDA operations registered successfully" << std::endl;
            } else {
                std::cout << "CUDA backend loaded but no CUDA devices available" << std::endl;
            }
        } else {
            std::cout << "Warning: Failed to load CUDA backend: " << cuda_result.error() << std::endl;
        }
    } else {
        std::cout << "CUDA backend not found at: " << cuda_backend_path << std::endl;
    }

    // Try to load ROCm backend if available
    std::filesystem::path rocm_backend_path = bin_path / "tenzor_backend_rocm.so";

    if (std::filesystem::exists(rocm_backend_path)) {
        std::cout << "Loading ROCm backend from: " << rocm_backend_path << std::endl;

        auto rocm_result = loader.load_backend(rocm_backend_path);
        if (rocm_result) {
            auto rocm_backend_unique = std::move(rocm_result.value());
            auto* rocm_backend_ptr = rocm_backend_unique.get();

            // Check if ROCm is actually available
            if (rocm_backend_ptr->is_available()) {
                loader.register_backend(rocm_backend_ptr->name(), std::move(rocm_backend_unique));
                std::cout << "ROCm backend registered: " << rocm_backend_ptr->name() << std::endl;
                std::cout << "Found " << rocm_backend_ptr->device_count() << " ROCm device(s)" << std::endl;

                auto* rocm_backend = rocm_backend_ptr;

                // Register all ROCm operations
                std::cout << "Registering ROCm kernels with operation registry" << std::endl;

                // Basic math operations
                registry.register_kernel("add", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("add", inputs, attrs);
                    });

                registry.register_kernel("sub", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("sub", inputs, attrs);
                    });

                registry.register_kernel("mul", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("mul", inputs, attrs);
                    });

                registry.register_kernel("div", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("div", inputs, attrs);
                    });

                registry.register_kernel("matmul", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("matmul", inputs, attrs);
                    });

                registry.register_kernel("sum", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("sum", inputs, attrs);
                    });

                registry.register_kernel("mean", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("mean", inputs, attrs);
                    });

                registry.register_kernel("max", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("max", inputs, attrs);
                    });

                registry.register_kernel("min", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("min", inputs, attrs);
                    });

                // Activation functions
                registry.register_kernel("relu", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("relu", inputs, attrs);
                    });

                registry.register_kernel("relu_backward", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("relu_backward", inputs, attrs);
                    });

                registry.register_kernel("sigmoid", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("sigmoid", inputs, attrs);
                    });

                registry.register_kernel("sigmoid_backward", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("sigmoid_backward", inputs, attrs);
                    });

                registry.register_kernel("tanh", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("tanh", inputs, attrs);
                    });

                registry.register_kernel("tanh_backward", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("tanh_backward", inputs, attrs);
                    });

                registry.register_kernel("gelu", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("gelu", inputs, attrs);
                    });

                registry.register_kernel("gelu_backward", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("gelu_backward", inputs, attrs);
                    });

                registry.register_kernel("leaky_relu", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("leaky_relu", inputs, attrs);
                    });

                registry.register_kernel("leaky_relu_backward", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("leaky_relu_backward", inputs, attrs);
                    });

                registry.register_kernel("softmax", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("softmax", inputs, attrs);
                    });

                registry.register_kernel("softmax_backward", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("softmax_backward", inputs, attrs);
                    });

                registry.register_kernel("log_softmax", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("log_softmax", inputs, attrs);
                    });

                registry.register_kernel("log_softmax_backward", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("log_softmax_backward", inputs, attrs);
                    });

                registry.register_kernel("neg", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("neg", inputs, attrs);
                    });

                registry.register_kernel("abs", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("abs", inputs, attrs);
                    });

                registry.register_kernel("sign", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("sign", inputs, attrs);
                    });

                // Math operations
                registry.register_kernel("sqrt", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("sqrt", inputs, attrs);
                    });

                registry.register_kernel("exp", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("exp", inputs, attrs);
                    });

                registry.register_kernel("log", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("log", inputs, attrs);
                    });

                registry.register_kernel("pow", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("pow", inputs, attrs);
                    });

                registry.register_kernel("clamp", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("clamp", inputs, attrs);
                    });

                // Comparison operations
                registry.register_kernel("eq", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("eq", inputs, attrs);
                    });

                registry.register_kernel("ne", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("ne", inputs, attrs);
                    });

                registry.register_kernel("lt", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("lt", inputs, attrs);
                    });

                registry.register_kernel("le", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("le", inputs, attrs);
                    });

                registry.register_kernel("gt", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("gt", inputs, attrs);
                    });

                registry.register_kernel("ge", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("ge", inputs, attrs);
                    });

                // Transform operations
                registry.register_kernel("contiguous", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("contiguous", inputs, attrs);
                    });

                registry.register_kernel("fill", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("fill", inputs, attrs);
                    });

                registry.register_kernel("clone", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("clone", inputs, attrs);
                    });

                registry.register_kernel("reshape", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("reshape", inputs, attrs);
                    });

                registry.register_kernel("transpose", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("transpose", inputs, attrs);
                    });

                registry.register_kernel("permute", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("permute", inputs, attrs);
                    });

                registry.register_kernel("squeeze", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("squeeze", inputs, attrs);
                    });

                registry.register_kernel("unsqueeze", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("unsqueeze", inputs, attrs);
                    });

                // BatchNorm2d operations
                registry.register_kernel("batchnorm2d_mean_var", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("batchnorm2d_mean_var", inputs, attrs);
                    });

                registry.register_kernel("batchnorm2d_forward", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("batchnorm2d_forward", inputs, attrs);
                    });

                registry.register_kernel("batchnorm2d_forward_affine", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("batchnorm2d_forward_affine", inputs, attrs);
                    });

                registry.register_kernel("batchnorm2d_update_running_stats", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("batchnorm2d_update_running_stats", inputs, attrs);
                    });

                registry.register_kernel("batchnorm2d_backward", Device::Type::ROCm,
                    [rocm_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return rocm_backend->dispatch("batchnorm2d_backward", inputs, attrs);
                    });

                std::cout << "ROCm operations registered successfully" << std::endl;
            } else {
                std::cout << "ROCm backend loaded but no ROCm devices available" << std::endl;
            }
        } else {
            std::cout << "Warning: Failed to load ROCm backend: " << rocm_result.error() << std::endl;
        }
    } else {
        std::cout << "ROCm backend not found at: " << rocm_backend_path << std::endl;
    }

    // Try to load OneAPI backend if available
    std::filesystem::path oneapi_backend_path = bin_path / "tenzor_backend_oneapi.so";

    if (std::filesystem::exists(oneapi_backend_path)) {
        std::cout << "Loading OneAPI backend from: " << oneapi_backend_path << std::endl;

        auto oneapi_result = loader.load_backend(oneapi_backend_path);
        if (oneapi_result) {
            auto oneapi_backend_unique = std::move(oneapi_result.value());
            auto* oneapi_backend_ptr = oneapi_backend_unique.get();

            // Check if OneAPI is actually available
            if (oneapi_backend_ptr->is_available()) {
                loader.register_backend(oneapi_backend_ptr->name(), std::move(oneapi_backend_unique));
                std::cout << "OneAPI backend registered: " << oneapi_backend_ptr->name() << std::endl;
                std::cout << "Found " << oneapi_backend_ptr->device_count() << " OneAPI device(s)" << std::endl;

                auto* oneapi_backend = oneapi_backend_ptr;

                // Register essential OneAPI operations
                std::cout << "Registering OneAPI kernels with operation registry" << std::endl;

                // Tensor creation operations
                registry.register_kernel("zeros", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("zeros", inputs, attrs);
                    });

                registry.register_kernel("ones", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("ones", inputs, attrs);
                    });

                registry.register_kernel("full", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("full", inputs, attrs);
                    });

                // Basic math operations
                registry.register_kernel("add", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("add", inputs, attrs);
                    });

                registry.register_kernel("matmul", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("matmul", inputs, attrs);
                    });

                registry.register_kernel("conv2d_forward", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("conv2d_forward", inputs, attrs);
                    });

                registry.register_kernel("conv2d_backward_input", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("conv2d_backward_input", inputs, attrs);
                    });

                registry.register_kernel("conv2d_backward_weight", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("conv2d_backward_weight", inputs, attrs);
                    });

                registry.register_kernel("conv2d_backward_bias", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("conv2d_backward_bias", inputs, attrs);
                    });

                // Additional binary operations
                registry.register_kernel("sub", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("sub", inputs, attrs);
                    });

                registry.register_kernel("mul", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("mul", inputs, attrs);
                    });

                registry.register_kernel("div", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("div", inputs, attrs);
                    });

                // Unary operations
                registry.register_kernel("sqrt", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("sqrt", inputs, attrs);
                    });

                registry.register_kernel("neg", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("neg", inputs, attrs);
                    });

                registry.register_kernel("abs", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("abs", inputs, attrs);
                    });

                registry.register_kernel("log", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("log", inputs, attrs);
                    });

                registry.register_kernel("exp", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("exp", inputs, attrs);
                    });

                registry.register_kernel("pow", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("pow", inputs, attrs);
                    });

                registry.register_kernel("dot", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("dot", inputs, attrs);
                    });

                // In-place operations
                registry.register_kernel("add_inplace", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("add_inplace", inputs, attrs);
                    });

                registry.register_kernel("sub_inplace", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("sub_inplace", inputs, attrs);
                    });

                registry.register_kernel("mul_inplace", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("mul_inplace", inputs, attrs);
                    });

                registry.register_kernel("div_inplace", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("div_inplace", inputs, attrs);
                    });

                // Activation functions
                registry.register_kernel("relu", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("relu", inputs, attrs);
                    });

                registry.register_kernel("sigmoid", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("sigmoid", inputs, attrs);
                    });

                registry.register_kernel("tanh", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("tanh", inputs, attrs);
                    });

                registry.register_kernel("leaky_relu", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("leaky_relu", inputs, attrs);
                    });

                registry.register_kernel("swish", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("swish", inputs, attrs);
                    });

                registry.register_kernel("swish_backward", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("swish_backward", inputs, attrs);
                    });

                // Reduction operations
                registry.register_kernel("sum", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("sum", inputs, attrs);
                    });

                registry.register_kernel("mean", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("mean", inputs, attrs);
                    });

                registry.register_kernel("max", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("max", inputs, attrs);
                    });

                registry.register_kernel("min", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("min", inputs, attrs);
                    });

                // Transform operations
                registry.register_kernel("reshape", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("reshape", inputs, attrs);
                    });

                registry.register_kernel("transpose", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("transpose", inputs, attrs);
                    });

                registry.register_kernel("permute", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("permute", inputs, attrs);
                    });

                registry.register_kernel("squeeze", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("squeeze", inputs, attrs);
                    });

                registry.register_kernel("unsqueeze", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("unsqueeze", inputs, attrs);
                    });

                registry.register_kernel("index_select", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("index_select", inputs, attrs);
                    });

                registry.register_kernel("gather", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("gather", inputs, attrs);
                    });

                registry.register_kernel("scatter", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("scatter", inputs, attrs);
                    });

                registry.register_kernel("masked_select", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("masked_select", inputs, attrs);
                    });

                registry.register_kernel("masked_fill", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("masked_fill", inputs, attrs);
                    });

                registry.register_kernel("where", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("where", inputs, attrs);
                    });

                registry.register_kernel("contiguous", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("contiguous", inputs, attrs);
                    });

                registry.register_kernel("clone", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("clone", inputs, attrs);
                    });

                // Fill operation
                registry.register_kernel("fill", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("fill", inputs, attrs);
                    });

                // Backward activation functions
                registry.register_kernel("relu_backward", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("relu_backward", inputs, attrs);
                    });

                registry.register_kernel("sigmoid_backward", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("sigmoid_backward", inputs, attrs);
                    });

                registry.register_kernel("tanh_backward", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("tanh_backward", inputs, attrs);
                    });

                registry.register_kernel("leaky_relu_backward", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("leaky_relu_backward", inputs, attrs);
                    });

                registry.register_kernel("gelu", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("gelu", inputs, attrs);
                    });

                registry.register_kernel("gelu_backward", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("gelu_backward", inputs, attrs);
                    });

                registry.register_kernel("softmax", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("softmax", inputs, attrs);
                    });

                registry.register_kernel("softmax_backward", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("softmax_backward", inputs, attrs);
                    });

                registry.register_kernel("log_softmax", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("log_softmax", inputs, attrs);
                    });

                registry.register_kernel("log_softmax_backward", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("log_softmax_backward", inputs, attrs);
                    });

                // Batch normalization operations
                registry.register_kernel("batchnorm2d_mean_var", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("batchnorm2d_mean_var", inputs, attrs);
                    });

                registry.register_kernel("batchnorm2d_update_running_stats", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("batchnorm2d_update_running_stats", inputs, attrs);
                    });

                registry.register_kernel("batchnorm2d_forward", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("batchnorm2d_forward", inputs, attrs);
                    });

                registry.register_kernel("batchnorm2d_forward_affine", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("batchnorm2d_forward_affine", inputs, attrs);
                    });

                registry.register_kernel("batchnorm2d_backward", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("batchnorm2d_backward", inputs, attrs);
                    });

                // Comparison operations
                registry.register_kernel("eq", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("eq", inputs, attrs);
                    });

                registry.register_kernel("ne", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("ne", inputs, attrs);
                    });

                registry.register_kernel("lt", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("lt", inputs, attrs);
                    });

                registry.register_kernel("le", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("le", inputs, attrs);
                    });

                registry.register_kernel("gt", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("gt", inputs, attrs);
                    });

                registry.register_kernel("ge", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("ge", inputs, attrs);
                    });

                // Utility operations
                registry.register_kernel("cat", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("cat", inputs, attrs);
                    });

                registry.register_kernel("clamp", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("clamp", inputs, attrs);
                    });

                registry.register_kernel("sign", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("sign", inputs, attrs);
                    });

                registry.register_kernel("repeat", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("repeat", inputs, attrs);
                    });

                // Vision operations (im2col/col2im)
                registry.register_kernel("im2col", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("im2col", inputs, attrs);
                    });

                registry.register_kernel("col2im", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("col2im", inputs, attrs);
                    });

                registry.register_kernel("expand", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("expand", inputs, attrs);
                    });

                // Pooling operations
                registry.register_kernel("avg_pool2d", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("avg_pool2d", inputs, attrs);
                    });

                registry.register_kernel("max_pool2d", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("max_pool2d", inputs, attrs);
                    });

                registry.register_kernel("adaptive_avg_pool2d", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("adaptive_avg_pool2d", inputs, attrs);
                    });

                registry.register_kernel("adaptive_max_pool2d", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("adaptive_max_pool2d", inputs, attrs);
                    });

                registry.register_kernel("avg_pool2d_backward", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("avg_pool2d_backward", inputs, attrs);
                    });

                registry.register_kernel("max_pool2d_backward", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("max_pool2d_backward", inputs, attrs);
                    });

                registry.register_kernel("adaptive_avg_pool2d_backward", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("adaptive_avg_pool2d_backward", inputs, attrs);
                    });

                registry.register_kernel("std", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("std", inputs, attrs);
                    });

                registry.register_kernel("norm", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("norm", inputs, attrs);
                    });

                registry.register_kernel("var", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("var", inputs, attrs);
                    });

                registry.register_kernel("prod", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("prod", inputs, attrs);
                    });

                registry.register_kernel("argmax", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("argmax", inputs, attrs);
                    });

                registry.register_kernel("argmin", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("argmin", inputs, attrs);
                    });

                // Trigonometric operations
                registry.register_kernel("sin", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("sin", inputs, attrs);
                    });

                registry.register_kernel("cos", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("cos", inputs, attrs);
                    });

                registry.register_kernel("tan", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("tan", inputs, attrs);
                    });

                registry.register_kernel("asin", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("asin", inputs, attrs);
                    });

                registry.register_kernel("acos", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("acos", inputs, attrs);
                    });

                registry.register_kernel("atan", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("atan", inputs, attrs);
                    });

                registry.register_kernel("sinh", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("sinh", inputs, attrs);
                    });

                registry.register_kernel("cosh", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("cosh", inputs, attrs);
                    });

                // Rounding operations
                registry.register_kernel("round", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("round", inputs, attrs);
                    });

                registry.register_kernel("floor", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("floor", inputs, attrs);
                    });

                registry.register_kernel("ceil", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("ceil", inputs, attrs);
                    });

                registry.register_kernel("trunc", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("trunc", inputs, attrs);
                    });

                registry.register_kernel("reciprocal", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("reciprocal", inputs, attrs);
                    });

                // Clamp operations
                registry.register_kernel("clamp_min", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("clamp_min", inputs, attrs);
                    });

                registry.register_kernel("clamp_max", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("clamp_max", inputs, attrs);
                    });

                // Vision operations
                registry.register_kernel("gather_relative_position_bias", Device::Type::OneAPI,
                    [oneapi_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return oneapi_backend->dispatch("gather_relative_position_bias", inputs, attrs);
                    });

                std::cout << "OneAPI operations registered successfully (92 operations)" << std::endl;
            } else {
                std::cout << "OneAPI backend loaded but no OneAPI devices available" << std::endl;
            }
        } else {
            std::cout << "Warning: Failed to load OneAPI backend: " << oneapi_result.error() << std::endl;
        }
    } else {
        std::cout << "OneAPI backend not found at: " << oneapi_backend_path << std::endl;
    }

    // Try to load Vulkan backend if available
    std::filesystem::path vulkan_backend_path = bin_path / "tenzor_backend_vulkan.so";

    if (std::filesystem::exists(vulkan_backend_path)) {
        std::cout << "Loading Vulkan backend from: " << vulkan_backend_path << std::endl;

        auto vulkan_result = loader.load_backend(vulkan_backend_path);
        if (vulkan_result) {
            auto vulkan_backend_unique = std::move(vulkan_result.value());
            auto* vulkan_backend_ptr = vulkan_backend_unique.get();

            // Check if Vulkan is actually available
            if (vulkan_backend_ptr->is_available()) {
                loader.register_backend(vulkan_backend_ptr->name(), std::move(vulkan_backend_unique));
                std::cout << "Vulkan backend registered: " << vulkan_backend_ptr->name() << std::endl;
                std::cout << "Found " << vulkan_backend_ptr->device_count() << " Vulkan device(s)" << std::endl;

                auto* vulkan_backend = vulkan_backend_ptr;

                // =========================================================================
                // O(1) DISPATCH TABLE REGISTRATION FOR VULKAN
                // =========================================================================
                DispatchTableRegistry::register_backend(Device::Type::Vulkan, vulkan_backend);
                auto& vulkan_table = DispatchTableRegistry::get_table(Device::Type::Vulkan);

                // Load the register_kernels function from the Vulkan backend shared library
                void* vulkan_handle = dlopen(vulkan_backend_path.c_str(), RTLD_NOW | RTLD_NOLOAD);
                if (vulkan_handle) {
                    using RegisterFn = void(*)(BackendDispatchTable*);
                    auto register_fn = reinterpret_cast<RegisterFn>(dlsym(vulkan_handle, "register_kernels"));
                    if (register_fn) {
                        register_fn(&vulkan_table);
                        std::cout << "Vulkan dispatch table initialized with O(1) lookup" << std::endl;
                    } else {
                        std::cerr << "Warning: Could not find register_kernels in Vulkan backend" << std::endl;
                    }
                    dlclose(vulkan_handle);
                } else {
                    std::cerr << "Warning: Could not reopen Vulkan backend for kernel registration" << std::endl;
                }

                // =========================================================================
                // LEGACY DISPATCH REGISTRATION (for backwards compatibility)
                // =========================================================================
                std::cout << "Registering Vulkan kernels with operation registry" << std::endl;

                // Binary math operations
                registry.register_kernel("add", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("add", inputs, attrs);
                    });

                registry.register_kernel("sub", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("sub", inputs, attrs);
                    });

                registry.register_kernel("mul", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("mul", inputs, attrs);
                    });

                registry.register_kernel("div", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("div", inputs, attrs);
                    });

                // In-place operations
                registry.register_kernel("add_inplace", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("add_inplace", inputs, attrs);
                    });

                registry.register_kernel("sub_inplace", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("sub_inplace", inputs, attrs);
                    });

                registry.register_kernel("mul_inplace", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("mul_inplace", inputs, attrs);
                    });

                registry.register_kernel("div_inplace", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("div_inplace", inputs, attrs);
                    });

                registry.register_kernel("matmul", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("matmul", inputs, attrs);
                    });

                // Unary operations
                registry.register_kernel("relu", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("relu", inputs, attrs);
                    });

                registry.register_kernel("sigmoid", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("sigmoid", inputs, attrs);
                    });

                registry.register_kernel("tanh", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("tanh", inputs, attrs);
                    });

                registry.register_kernel("gelu", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("gelu", inputs, attrs);
                    });

                registry.register_kernel("leaky_relu", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("leaky_relu", inputs, attrs);
                    });

                registry.register_kernel("swish", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("swish", inputs, attrs);
                    });

                registry.register_kernel("elu", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("elu", inputs, attrs);
                    });

                registry.register_kernel("selu", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("selu", inputs, attrs);
                    });

                registry.register_kernel("mish", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("mish", inputs, attrs);
                    });

                registry.register_kernel("softplus", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("softplus", inputs, attrs);
                    });

                registry.register_kernel("sqrt", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("sqrt", inputs, attrs);
                    });

                registry.register_kernel("exp", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("exp", inputs, attrs);
                    });

                registry.register_kernel("log", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("log", inputs, attrs);
                    });

                registry.register_kernel("neg", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("neg", inputs, attrs);
                    });

                registry.register_kernel("abs", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("abs", inputs, attrs);
                    });

                registry.register_kernel("pow", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("pow", inputs, attrs);
                    });

                registry.register_kernel("sign", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("sign", inputs, attrs);
                    });

                // Trigonometric operations
                registry.register_kernel("sin", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("sin", inputs, attrs);
                    });

                registry.register_kernel("cos", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("cos", inputs, attrs);
                    });

                registry.register_kernel("tan", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("tan", inputs, attrs);
                    });

                registry.register_kernel("asin", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("asin", inputs, attrs);
                    });

                registry.register_kernel("acos", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("acos", inputs, attrs);
                    });

                registry.register_kernel("atan", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("atan", inputs, attrs);
                    });

                // Hyperbolic operations
                registry.register_kernel("sinh", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("sinh", inputs, attrs);
                    });

                registry.register_kernel("cosh", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("cosh", inputs, attrs);
                    });

                // Reduction operations
                registry.register_kernel("sum", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("sum", inputs, attrs);
                    });

                registry.register_kernel("mean", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("mean", inputs, attrs);
                    });

                registry.register_kernel("max", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("max", inputs, attrs);
                    });

                registry.register_kernel("min", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("min", inputs, attrs);
                    });

                registry.register_kernel("argmax", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("argmax", inputs, attrs);
                    });

                registry.register_kernel("argmin", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("argmin", inputs, attrs);
                    });

                registry.register_kernel("var", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("var", inputs, attrs);
                    });

                registry.register_kernel("std", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("std", inputs, attrs);
                    });

                registry.register_kernel("prod", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("prod", inputs, attrs);
                    });

                // Pooling operations
                registry.register_kernel("max_pool2d", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("max_pool2d", inputs, attrs);
                    });

                registry.register_kernel("avg_pool2d", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("avg_pool2d", inputs, attrs);
                    });

                registry.register_kernel("adaptive_max_pool2d", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("adaptive_max_pool2d", inputs, attrs);
                    });

                registry.register_kernel("adaptive_avg_pool2d", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("adaptive_avg_pool2d", inputs, attrs);
                    });

                registry.register_kernel("adaptive_avg_pool2d_backward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("adaptive_avg_pool2d_backward", inputs, attrs);
                    });

                // Normalization operations
                registry.register_kernel("softmax", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("softmax", inputs, attrs);
                    });

                registry.register_kernel("log_softmax", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("log_softmax", inputs, attrs);
                    });

                // Indexing operations
                registry.register_kernel("embedding", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("embedding", inputs, attrs);
                    });

                registry.register_kernel("gather", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("gather", inputs, attrs);
                    });

                registry.register_kernel("scatter", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("scatter", inputs, attrs);
                    });

                registry.register_kernel("index_select", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("index_select", inputs, attrs);
                    });

                // Comparison operations
                registry.register_kernel("eq", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("eq", inputs, attrs);
                    });

                registry.register_kernel("ne", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("ne", inputs, attrs);
                    });

                registry.register_kernel("lt", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("lt", inputs, attrs);
                    });

                registry.register_kernel("le", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("le", inputs, attrs);
                    });

                registry.register_kernel("gt", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("gt", inputs, attrs);
                    });

                registry.register_kernel("ge", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("ge", inputs, attrs);
                    });

                // Shape operations
                registry.register_kernel("zeros", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("zeros", inputs, attrs);
                    });

                registry.register_kernel("fill", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("fill", inputs, attrs);
                    });

                registry.register_kernel("clone", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("clone", inputs, attrs);
                    });

                registry.register_kernel("contiguous", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("contiguous", inputs, attrs);
                    });

                registry.register_kernel("reshape", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("reshape", inputs, attrs);
                    });

                registry.register_kernel("transpose", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("transpose", inputs, attrs);
                    });

                registry.register_kernel("permute", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("permute", inputs, attrs);
                    });

                registry.register_kernel("squeeze", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("squeeze", inputs, attrs);
                    });

                registry.register_kernel("unsqueeze", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("unsqueeze", inputs, attrs);
                    });

                // Backward activation operations
                registry.register_kernel("relu_backward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("relu_backward", inputs, attrs);
                    });

                registry.register_kernel("sigmoid_backward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("sigmoid_backward", inputs, attrs);
                    });

                registry.register_kernel("tanh_backward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("tanh_backward", inputs, attrs);
                    });

                registry.register_kernel("leaky_relu_backward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("leaky_relu_backward", inputs, attrs);
                    });

                registry.register_kernel("gelu_backward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("gelu_backward", inputs, attrs);
                    });

                registry.register_kernel("elu_backward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("elu_backward", inputs, attrs);
                    });

                registry.register_kernel("selu_backward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("selu_backward", inputs, attrs);
                    });

                registry.register_kernel("mish_backward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("mish_backward", inputs, attrs);
                    });

                registry.register_kernel("softplus_backward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("softplus_backward", inputs, attrs);
                    });

                registry.register_kernel("swish_backward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("swish_backward", inputs, attrs);
                    });

                registry.register_kernel("softmax_backward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("softmax_backward", inputs, attrs);
                    });

                registry.register_kernel("log_softmax_backward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("log_softmax_backward", inputs, attrs);
                    });

                // Vision operations (im2col/col2im)
                registry.register_kernel("im2col", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("im2col", inputs, attrs);
                    });

                registry.register_kernel("col2im", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("col2im", inputs, attrs);
                    });

                // Conv2d backward operations
                registry.register_kernel("conv2d_backward_input", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("conv2d_backward_input", inputs, attrs);
                    });

                registry.register_kernel("conv2d_backward_weight", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("conv2d_backward_weight", inputs, attrs);
                    });

                registry.register_kernel("conv2d_backward_bias", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("conv2d_backward_bias", inputs, attrs);
                    });

                // Tensor manipulation operations
                registry.register_kernel("expand", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("expand", inputs, attrs);
                    });

                registry.register_kernel("cat", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("cat", inputs, attrs);
                    });

                registry.register_kernel("clamp", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("clamp", inputs, attrs);
                    });

                registry.register_kernel("clamp_min", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("clamp_min", inputs, attrs);
                    });

                registry.register_kernel("clamp_max", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("clamp_max", inputs, attrs);
                    });

                // BatchNorm2d operations
                registry.register_kernel("batchnorm2d_forward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("batchnorm2d_forward", inputs, attrs);
                    });

                registry.register_kernel("batchnorm2d_forward_affine", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("batchnorm2d_forward_affine", inputs, attrs);
                    });

                registry.register_kernel("batchnorm2d_backward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("batchnorm2d_backward", inputs, attrs);
                    });

                registry.register_kernel("batchnorm2d_mean_var", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("batchnorm2d_mean_var", inputs, attrs);
                    });

                registry.register_kernel("batchnorm2d_update_running_stats", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("batchnorm2d_update_running_stats", inputs, attrs);
                    });

                // Pooling operations (new)
                registry.register_kernel("avg_pool2d_forward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("avg_pool2d_forward", inputs, attrs);
                    });

                registry.register_kernel("max_pool2d_forward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("max_pool2d_forward", inputs, attrs);
                    });

                registry.register_kernel("avg_pool2d_backward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("avg_pool2d_backward", inputs, attrs);
                    });

                registry.register_kernel("max_pool2d_backward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("max_pool2d_backward", inputs, attrs);
                    });

                // Final 3 operations for 100% coverage
                registry.register_kernel("conv2d_forward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("conv2d_forward", inputs, attrs);
                    });

                registry.register_kernel("conv_transpose2d_forward", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("conv_transpose2d_forward", inputs, attrs);
                    });

                registry.register_kernel("full", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("full", inputs, attrs);
                    });

                registry.register_kernel("ones", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("ones", inputs, attrs);
                    });

                // Additional math operations
                registry.register_kernel("dot", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("dot", inputs, attrs);
                    });

                registry.register_kernel("norm", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("norm", inputs, attrs);
                    });

                registry.register_kernel("reciprocal", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("reciprocal", inputs, attrs);
                    });

                // Rounding operations
                registry.register_kernel("round", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("round", inputs, attrs);
                    });

                registry.register_kernel("floor", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("floor", inputs, attrs);
                    });

                registry.register_kernel("ceil", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("ceil", inputs, attrs);
                    });

                // Manipulation operations
                registry.register_kernel("repeat", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("repeat", inputs, attrs);
                    });

                registry.register_kernel("roll", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("roll", inputs, attrs);
                    });

                // Masked operations
                registry.register_kernel("masked_select", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("masked_select", inputs, attrs);
                    });

                registry.register_kernel("masked_fill", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("masked_fill", inputs, attrs);
                    });

                registry.register_kernel("where", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("where", inputs, attrs);
                    });

                // Vision operations
                registry.register_kernel("gather_relative_position_bias", Device::Type::Vulkan,
                    [vulkan_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return vulkan_backend->dispatch("gather_relative_position_bias", inputs, attrs);
                    });

                std::cout << "Vulkan operations registered successfully (101 operations - all core operations added!)" << std::endl;
            } else {
                std::cout << "Vulkan backend loaded but no Vulkan devices available" << std::endl;
            }
        } else {
            std::cout << "Warning: Failed to load Vulkan backend: " << vulkan_result.error() << std::endl;
        }
    } else {
        std::cout << "Vulkan backend not found at: " << vulkan_backend_path << std::endl;
    }

    std::cout << "Tenzor initialization complete - 51 CPU operations registered" << std::endl;

    g_initialized = true;
}

auto finalize() -> void {
    if (!g_initialized) {
        return;
    }

    std::cout << "Finalizing Tenzor library" << std::endl;

    // Cleanup (backend loader doesn't need explicit cleanup)

    g_initialized = false;
}

} // namespace tenzor
