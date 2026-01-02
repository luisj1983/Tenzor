#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/dispatch.hpp"
#include "tenzor/backend/loader.hpp"
#include <random>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iostream>  // DEBUG

namespace tenzor {

// Global random number generator for reproducible randomness
static std::random_device rd;
static std::mt19937 global_rng(rd());

// Function to access the global RNG
static std::mt19937& get_rng() {
    return global_rng;
}

// Public function to set the random seed
void manual_seed(unsigned int seed) {
    global_rng.seed(seed);
}

// Helper function to convert shape vector to comma-separated string
static auto shape_to_string(const std::vector<int64_t>& shape) -> std::string {
    std::ostringstream oss;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ",";
        oss << shape[i];
    }
    return oss.str();
}

// Helper function to convert DType to string
static auto dtype_to_string(DType dtype) -> std::string {
    switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Float16: return "float16";
        case DType::BFloat16: return "bfloat16";
        case DType::Int8: return "int8";
        case DType::Int16: return "int16";
        case DType::Int32: return "int32";
        case DType::Int64: return "int64";
        case DType::UInt8: return "uint8";
        case DType::UInt16: return "uint16";
        case DType::UInt32: return "uint32";
        case DType::UInt64: return "uint64";
        case DType::Bool: return "bool";
        default: return "unknown";
    }
}

auto zeros(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    // Use backend directly for non-CPU devices
    if (device.type != Device::Type::CPU) {
        auto backend = backend_registry().get_backend(device.type);
        if (!backend) {
            throw std::runtime_error("Backend not available for device type");
        }

        OpAttributes attrs;
        attrs["shape"] = shape_to_string(shape);
        attrs["dtype"] = dtype_to_string(dtype);
        attrs["device_id"] = std::to_string(device.index);

        return backend->dispatch("zeros", {}, attrs)[0];
    }

    // CPU path: use zero-initialized constructor directly
    return Tensor(std::move(shape), dtype, device);
}

auto ones(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    // Use backend directly for non-CPU devices
    if (device.type != Device::Type::CPU) {
        auto backend = backend_registry().get_backend(device.type);
        if (!backend) {
            throw std::runtime_error("Backend not available for device type");
        }

        OpAttributes attrs;
        attrs["shape"] = shape_to_string(shape);
        attrs["dtype"] = dtype_to_string(dtype);
        attrs["device_id"] = std::to_string(device.index);

        // DEBUG
        //if (shape.size() == 1 && shape[0] <= 10) {
        //    std::cerr << "[ONES_DISPATCH] Calling backend->dispatch(\"ones\") with shape=\""
        //              << attrs["shape"] << "\" dtype=\"" << attrs["dtype"] << "\"" << std::endl;
        //}

        return backend->dispatch("ones", {}, attrs)[0];
    }

    // CPU path: use uninitialized allocation (avoid wasteful zeroing before fill)
    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, device);
    if (!tensor.impl() || !tensor.impl()->storage) return tensor;

    size_t numel = tensor.numel();
    void* data = tensor.impl()->storage->data();

    // Fill with ones based on dtype
    switch (dtype) {
        case DType::Float16: {
            Float16* ptr = static_cast<Float16*>(data);
            std::fill(ptr, ptr + numel, Float16(1.0f));
            break;
        }
        case DType::Float32: {
            float* ptr = static_cast<float*>(data);
            std::fill(ptr, ptr + numel, 1.0f);
            break;
        }
        case DType::Float64: {
            double* ptr = static_cast<double*>(data);
            std::fill(ptr, ptr + numel, 1.0);
            break;
        }
        case DType::Int32: {
            int32_t* ptr = static_cast<int32_t*>(data);
            std::fill(ptr, ptr + numel, 1);
            break;
        }
        case DType::Int64: {
            int64_t* ptr = static_cast<int64_t*>(data);
            std::fill(ptr, ptr + numel, 1);
            break;
        }
        case DType::UInt8: {
            uint8_t* ptr = static_cast<uint8_t*>(data);
            std::fill(ptr, ptr + numel, 1);
            break;
        }
        case DType::Int8: {
            int8_t* ptr = static_cast<int8_t*>(data);
            std::fill(ptr, ptr + numel, 1);
            break;
        }
        case DType::Bool: {
            bool* ptr = static_cast<bool*>(data);
            std::fill(ptr, ptr + numel, true);
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for ones()");
    }
    return tensor;
}

auto full(std::vector<int64_t> shape, float value, DType dtype, Device device) -> Tensor {
    // Use backend directly for non-CPU devices
    if (device.type != Device::Type::CPU) {
        auto backend = backend_registry().get_backend(device.type);
        if (!backend) {
            throw std::runtime_error("Backend not available for device type");
        }

        OpAttributes attrs;
        attrs["shape"] = shape_to_string(shape);
        // FIX: Use scientific notation with full precision to avoid loss of small values
        // std::to_string() uses fixed precision and loses values like 1e-7
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
        attrs["value"] = oss.str();
        attrs["dtype"] = dtype_to_string(dtype);
        attrs["device_id"] = std::to_string(device.index);

        return backend->dispatch("full", {}, attrs)[0];
    }

    // CPU path: use uninitialized allocation (avoid wasteful zeroing before fill)
    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, device);
    if (!tensor.impl() || !tensor.impl()->storage) return tensor;

    size_t numel = tensor.numel();
    void* data = tensor.impl()->storage->data();

    // Fill with value based on dtype
    switch (dtype) {
        case DType::Float16: {
            Float16* ptr = static_cast<Float16*>(data);
            std::fill(ptr, ptr + numel, Float16(static_cast<float>(value)));
            break;
        }
        case DType::Float32: {
            float* ptr = static_cast<float*>(data);
            std::fill(ptr, ptr + numel, static_cast<float>(value));
            break;
        }
        case DType::Float64: {
            double* ptr = static_cast<double*>(data);
            std::fill(ptr, ptr + numel, static_cast<double>(value));
            break;
        }
        case DType::Int32: {
            int32_t* ptr = static_cast<int32_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<int32_t>(value));
            break;
        }
        case DType::Int64: {
            int64_t* ptr = static_cast<int64_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<int64_t>(value));
            break;
        }
        case DType::UInt8: {
            uint8_t* ptr = static_cast<uint8_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<uint8_t>(value));
            break;
        }
        case DType::Int8: {
            int8_t* ptr = static_cast<int8_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<int8_t>(value));
            break;
        }
        case DType::Bool: {
            bool* ptr = static_cast<bool*>(data);
            std::fill(ptr, ptr + numel, value != 0.0f);
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for full()");
    }
    return tensor;
}

auto full(std::vector<int64_t> shape, double value, DType dtype, Device device) -> Tensor {
    // Use backend directly for non-CPU devices
    if (device.type != Device::Type::CPU) {
        auto backend = backend_registry().get_backend(device.type);
        if (!backend) {
            throw std::runtime_error("Backend not available for device type");
        }

        OpAttributes attrs;
        attrs["shape"] = shape_to_string(shape);
        // Use scientific notation with full precision for double values
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
        attrs["value"] = oss.str();
        attrs["dtype"] = dtype_to_string(dtype);
        attrs["device_id"] = std::to_string(device.index);

        return backend->dispatch("full", {}, attrs)[0];
    }

    // CPU path: use uninitialized allocation (avoid wasteful zeroing before fill)
    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, device);
    if (!tensor.impl() || !tensor.impl()->storage) return tensor;

    size_t numel = tensor.numel();
    void* data = tensor.impl()->storage->data();

    // Fill with value based on dtype
    switch (dtype) {
        case DType::Float16: {
            Float16* ptr = static_cast<Float16*>(data);
            std::fill(ptr, ptr + numel, Float16(static_cast<float>(value)));
            break;
        }
        case DType::Float32: {
            float* ptr = static_cast<float*>(data);
            std::fill(ptr, ptr + numel, static_cast<float>(value));
            break;
        }
        case DType::Float64: {
            double* ptr = static_cast<double*>(data);
            std::fill(ptr, ptr + numel, value);  // No cast needed, already double
            break;
        }
        case DType::Int32: {
            int32_t* ptr = static_cast<int32_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<int32_t>(value));
            break;
        }
        case DType::Int64: {
            int64_t* ptr = static_cast<int64_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<int64_t>(value));
            break;
        }
        case DType::UInt8: {
            uint8_t* ptr = static_cast<uint8_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<uint8_t>(value));
            break;
        }
        case DType::Int8: {
            int8_t* ptr = static_cast<int8_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<int8_t>(value));
            break;
        }
        case DType::Bool: {
            bool* ptr = static_cast<bool*>(data);
            std::fill(ptr, ptr + numel, value != 0.0);
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for full()");
    }
    return tensor;
}

auto empty(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    // empty() returns truly uninitialized memory - no zeroing
    return Tensor::empty_uninitialized(std::move(shape), dtype, device);
}

auto rand(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    // Use backend directly for non-CPU devices
    if (device.type != Device::Type::CPU) {
        auto backend = backend_registry().get_backend(device.type);
        if (!backend) {
            throw std::runtime_error("Backend not available for device type");
        }

        OpAttributes attrs;
        attrs["shape"] = shape_to_string(shape);
        attrs["dtype"] = dtype_to_string(dtype);
        attrs["device_id"] = std::to_string(device.index);

        return backend->dispatch("rand", {}, attrs)[0];
    }

    // CPU path: use uninitialized allocation (avoid wasteful zeroing before fill)
    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, device);
    if (!tensor.impl() || !tensor.impl()->storage) return tensor;

    size_t numel = tensor.numel();
    void* data = tensor.impl()->storage->data();

    // Use global random number generator (can be seeded via manual_seed)
    auto& gen = get_rng();

    // Fill with uniform random values based on dtype
    switch (dtype) {
        case DType::Float32: {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            float* ptr = static_cast<float*>(data);
            for (size_t i = 0; i < numel; ++i) {
                ptr[i] = dist(gen);
            }
            break;
        }
        case DType::Float64: {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            double* ptr = static_cast<double*>(data);
            for (size_t i = 0; i < numel; ++i) {
                ptr[i] = dist(gen);
            }
            break;
        }
        case DType::Float16: {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            Float16* ptr = static_cast<Float16*>(data);
            for (size_t i = 0; i < numel; ++i) {
                ptr[i] = Float16(dist(gen));
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for rand() - only Float32, Float64, and Float16 are supported");
    }
    return tensor;
}

auto randn(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    // Use backend directly for non-CPU devices
    if (device.type != Device::Type::CPU) {
        auto backend = backend_registry().get_backend(device.type);
        if (!backend) {
            throw std::runtime_error("Backend not available for device type");
        }

        OpAttributes attrs;
        attrs["shape"] = shape_to_string(shape);
        attrs["dtype"] = dtype_to_string(dtype);
        attrs["device_id"] = std::to_string(device.index);

        return backend->dispatch("randn", {}, attrs)[0];
    }

    // CPU path: use uninitialized allocation (avoid wasteful zeroing before fill)
    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, device);
    if (!tensor.impl() || !tensor.impl()->storage) return tensor;

    size_t numel = tensor.numel();
    void* data = tensor.impl()->storage->data();

    // Use global random number generator (can be seeded via manual_seed)
    auto& gen = get_rng();

    // Fill with normal random values N(0,1) based on dtype
    switch (dtype) {
        case DType::Float32: {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            float* ptr = static_cast<float*>(data);
            for (size_t i = 0; i < numel; ++i) {
                ptr[i] = dist(gen);
            }
            break;
        }
        case DType::Float64: {
            std::normal_distribution<double> dist(0.0, 1.0);
            double* ptr = static_cast<double*>(data);
            for (size_t i = 0; i < numel; ++i) {
                ptr[i] = dist(gen);
            }
            break;
        }
        case DType::Float16: {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            Float16* ptr = static_cast<Float16*>(data);
            for (size_t i = 0; i < numel; ++i) {
                ptr[i] = Float16(dist(gen));
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for randn() - only Float32, Float64, and Float16 are supported");
    }
    return tensor;
}

auto arange(float start, float end, float step, DType dtype, Device device) -> Tensor {
    if (step == 0.0f) {
        throw std::invalid_argument("step cannot be zero");
    }
    if ((step > 0 && start >= end) || (step < 0 && start <= end)) {
        throw std::invalid_argument("Invalid start, end, step combination");
    }

    // For non-CPU devices, create on CPU first then transfer
    if (device.type != Device::Type::CPU) {
        auto cpu_tensor = arange(start, end, step, dtype, Device::cpu());
        return cpu_tensor.to(device);
    }

    // Calculate number of elements
    int64_t numel = static_cast<int64_t>(std::ceil((end - start) / step));
    if (numel < 0) numel = 0;

    // Use uninitialized allocation (avoid wasteful zeroing before fill)
    auto tensor = Tensor::empty_uninitialized({numel}, dtype, device);
    if (!tensor.impl() || !tensor.impl()->storage || numel == 0) return tensor;

    void* data = tensor.impl()->storage->data();

    // Fill with range values based on dtype
    switch (dtype) {
        case DType::Float32: {
            float* ptr = static_cast<float*>(data);
            float value = start;
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = value;
                value += step;
            }
            break;
        }
        case DType::Float64: {
            double* ptr = static_cast<double*>(data);
            double value = start;
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = value;
                value += step;
            }
            break;
        }
        case DType::Int32: {
            int32_t* ptr = static_cast<int32_t*>(data);
            int32_t value = static_cast<int32_t>(start);
            int32_t step_int = static_cast<int32_t>(step);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = value;
                value += step_int;
            }
            break;
        }
        case DType::Int64: {
            int64_t* ptr = static_cast<int64_t*>(data);
            int64_t value = static_cast<int64_t>(start);
            int64_t step_int = static_cast<int64_t>(step);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = value;
                value += step_int;
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for arange() - only Float32, Float64, Int32, Int64 are supported");
    }
    return tensor;
}

auto linspace(float start, float end, int64_t steps, DType dtype, Device device) -> Tensor {
    if (steps <= 0) {
        throw std::invalid_argument("steps must be positive");
    }

    // For non-CPU devices, create on CPU first then transfer
    if (device.type != Device::Type::CPU) {
        auto cpu_tensor = linspace(start, end, steps, dtype, Device::cpu());
        return cpu_tensor.to(device);
    }

    // Use uninitialized allocation (avoid wasteful zeroing before fill)
    auto tensor = Tensor::empty_uninitialized({steps}, dtype, device);
    if (!tensor.impl() || !tensor.impl()->storage) return tensor;

    void* data = tensor.impl()->storage->data();

    // Calculate step size
    double step_size = (steps > 1) ? (static_cast<double>(end) - static_cast<double>(start)) / (steps - 1) : 0.0;

    // Fill with linearly spaced values based on dtype
    switch (dtype) {
        case DType::Float32: {
            float* ptr = static_cast<float*>(data);
            if (steps == 1) {
                ptr[0] = start;
            } else {
                for (int64_t i = 0; i < steps; ++i) {
                    ptr[i] = static_cast<float>(start + i * step_size);
                }
                // Ensure the last element is exactly 'end' to avoid floating point errors
                ptr[steps - 1] = end;
            }
            break;
        }
        case DType::Float64: {
            double* ptr = static_cast<double*>(data);
            if (steps == 1) {
                ptr[0] = start;
            } else {
                for (int64_t i = 0; i < steps; ++i) {
                    ptr[i] = start + i * step_size;
                }
                // Ensure the last element is exactly 'end' to avoid floating point errors
                ptr[steps - 1] = end;
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for linspace() - only Float32 and Float64 are supported");
    }
    return tensor;
}

auto eye(int64_t n, std::optional<int64_t> m, DType dtype, Device device) -> Tensor {
    int64_t cols = m.value_or(n);

    // For non-CPU devices, create on CPU first then transfer
    if (device.type != Device::Type::CPU) {
        auto cpu_tensor = eye(n, m, dtype, Device::cpu());
        return cpu_tensor.to(device);
    }

    // Start with zeros (now safe since we're on CPU)
    auto tensor = zeros({n, cols}, dtype, device);
    if (!tensor.impl() || !tensor.impl()->storage) return tensor;

    void* data = tensor.impl()->storage->data();
    int64_t min_dim = std::min(n, cols);

    // Set diagonal to ones based on dtype
    switch (dtype) {
        case DType::Float32: {
            float* ptr = static_cast<float*>(data);
            for (int64_t i = 0; i < min_dim; ++i) {
                ptr[i * cols + i] = 1.0f;
            }
            break;
        }
        case DType::Float64: {
            double* ptr = static_cast<double*>(data);
            for (int64_t i = 0; i < min_dim; ++i) {
                ptr[i * cols + i] = 1.0;
            }
            break;
        }
        case DType::Int32: {
            int32_t* ptr = static_cast<int32_t*>(data);
            for (int64_t i = 0; i < min_dim; ++i) {
                ptr[i * cols + i] = 1;
            }
            break;
        }
        case DType::Int64: {
            int64_t* ptr = static_cast<int64_t*>(data);
            for (int64_t i = 0; i < min_dim; ++i) {
                ptr[i * cols + i] = 1;
            }
            break;
        }
        case DType::UInt8: {
            uint8_t* ptr = static_cast<uint8_t*>(data);
            for (int64_t i = 0; i < min_dim; ++i) {
                ptr[i * cols + i] = 1;
            }
            break;
        }
        case DType::Bool: {
            bool* ptr = static_cast<bool*>(data);
            for (int64_t i = 0; i < min_dim; ++i) {
                ptr[i * cols + i] = true;
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for eye()");
    }
    return tensor;
}

auto zeros_like(const Tensor& tensor) -> Tensor {
    std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
    return zeros(shape, tensor.dtype(), tensor.device());
}

auto ones_like(const Tensor& tensor) -> Tensor {
    std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
    return ones(shape, tensor.dtype(), tensor.device());
}

auto rand_like(const Tensor& tensor) -> Tensor {
    std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
    return rand(shape, tensor.dtype(), tensor.device());
}

auto randn_like(const Tensor& tensor) -> Tensor {
    std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
    return randn(shape, tensor.dtype(), tensor.device());
}

auto randperm(int64_t n, Device device) -> Tensor {
    // Create tensor with sequential integers 0 to n-1
    auto tensor = arange(0.0f, static_cast<float>(n), 1.0f, DType::Int64, device);

    // For CPU, shuffle in place
    if (device.type == Device::Type::CPU) {
        auto data = tensor.data<int64_t>();
        auto& gen = get_rng();
        std::shuffle(data, data + n, gen);
    } else {
        // For other devices, would need backend-specific implementation
        throw std::runtime_error("randperm only supported on CPU currently");
    }

    return tensor;
}

} // namespace tenzor
