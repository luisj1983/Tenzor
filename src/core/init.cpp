#include "tenzor/tenzor.hpp"
#include "tenzor/backend/registry.hpp"
#include "tenzor/backend/loader.hpp"
#include <iostream>
#include <filesystem>

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

    registry.register_kernel("leaky_relu", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("leaky_relu", inputs, attrs);
        });

    registry.register_kernel("leaky_relu_backward", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("leaky_relu_backward", inputs, attrs);
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

    registry.register_kernel("clamp", Device::Type::CPU,
        [cpu_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cpu_backend->dispatch("clamp", inputs, attrs);
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

                // Register all CUDA operations
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

                registry.register_kernel("leaky_relu", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("leaky_relu", inputs, attrs);
                    });

                registry.register_kernel("leaky_relu_backward", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("leaky_relu_backward", inputs, attrs);
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

                registry.register_kernel("clamp", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("clamp", inputs, attrs);
                    });

                // Transform operations
                registry.register_kernel("expand", Device::Type::CUDA,
                    [cuda_backend](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
                        return cuda_backend->dispatch("expand", inputs, attrs);
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

    std::cout << "Tenzor initialization complete - 38 CPU operations registered" << std::endl;

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
