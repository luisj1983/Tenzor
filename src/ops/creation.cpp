#include "tenzor/ops/creation.hpp"
#include "tenzor/core/generator.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/backend/dispatch.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include <random>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <complex>

namespace tenzor {

// Thread-local RNG — each thread has independent random state
static thread_local std::mt19937 global_rng(std::random_device{}());
static thread_local bool manual_seed_set = false;
static thread_local uint64_t manual_seed_value = 0;

// Function to access the thread-local RNG
static std::mt19937& get_rng() {
    return global_rng;
}

// Public function to set the random seed (affects calling thread)
void manual_seed(unsigned int seed) {
    global_rng.seed(seed);
    manual_seed_set = true;
    manual_seed_value = seed;
}

// Get a seed for backend RNG: returns the manual seed if set, otherwise a time-based seed
uint64_t get_global_seed() {
    if (manual_seed_set) {
        // Increment so successive calls get different but deterministic seeds
        return manual_seed_value++;
    }
    auto now = std::chrono::high_resolution_clock::now();
    return static_cast<uint64_t>(now.time_since_epoch().count());
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
        case DType::Complex64: return "complex64";
        case DType::Complex128: return "complex128";
        default: return "unknown";
    }
}

auto zeros(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    // Use OpId dispatch for non-CPU devices
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Shape, shape_to_string(shape));
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

        return dispatch_to_device(OpId::Zeros, device.type, {}, attrs)[0];
    }

    // CPU path: validate device index (only index 0 is valid for CPU)
    if (device.index != 0) {
        throw std::runtime_error("Invalid device index " + std::to_string(device.index) +
            " for CPU (only index 0 is valid)");
    }

    // CPU path: use zero-initialized constructor directly
    return Tensor(std::move(shape), dtype, device);
}

auto ones(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    // Use OpId dispatch for non-CPU devices
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Shape, shape_to_string(shape));
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

        return dispatch_to_device(OpId::Ones, device.type, {}, attrs)[0];
    }

    // CPU path: validate device index (only index 0 is valid for CPU)
    if (device.index != 0) {
        throw std::runtime_error("Invalid device index " + std::to_string(device.index) +
            " for CPU (only index 0 is valid)");
    }

    // CPU path: use uninitialized allocation (avoid wasteful zeroing before fill)
    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, device);
    if (!tensor.impl() || !tensor.storage()) return tensor;

    size_t numel = tensor.numel();
    void* data = tensor.storage()->data();

    // Fill with ones based on dtype
    switch (dtype) {
        case DType::Float16: {
            Float16* ptr = static_cast<Float16*>(data);
            std::fill(ptr, ptr + numel, Float16(1.0f));
            break;
        }
        case DType::BFloat16: {
            BFloat16* ptr = static_cast<BFloat16*>(data);
            std::fill(ptr, ptr + numel, BFloat16(1.0f));
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
            std::fill(ptr, ptr + numel, static_cast<uint8_t>(1));
            break;
        }
        case DType::UInt16: {
            uint16_t* ptr = static_cast<uint16_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<uint16_t>(1));
            break;
        }
        case DType::UInt32: {
            uint32_t* ptr = static_cast<uint32_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<uint32_t>(1));
            break;
        }
        case DType::UInt64: {
            uint64_t* ptr = static_cast<uint64_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<uint64_t>(1));
            break;
        }
        case DType::Int8: {
            int8_t* ptr = static_cast<int8_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<int8_t>(1));
            break;
        }
        case DType::Int16: {
            int16_t* ptr = static_cast<int16_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<int16_t>(1));
            break;
        }
        case DType::Bool: {
            bool* ptr = static_cast<bool*>(data);
            std::fill(ptr, ptr + numel, true);
            break;
        }
        case DType::Complex64: {
            auto* ptr = static_cast<std::complex<float>*>(data);
            std::fill(ptr, ptr + numel, std::complex<float>(1.0f, 0.0f));
            break;
        }
        case DType::Complex128: {
            auto* ptr = static_cast<std::complex<double>*>(data);
            std::fill(ptr, ptr + numel, std::complex<double>(1.0, 0.0));
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for ones()");
    }
    return tensor;
}

auto full(std::vector<int64_t> shape, float value, DType dtype, Device device) -> Tensor {
    // Use OpId dispatch for non-CPU devices
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Shape, shape_to_string(shape));
        attrs.set(AttrKey::Value, static_cast<double>(value));
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

        return dispatch_to_device(OpId::Full, device.type, {}, attrs)[0];
    }

    // CPU path: use uninitialized allocation (avoid wasteful zeroing before fill)
    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, device);
    if (!tensor.impl() || !tensor.storage()) return tensor;

    size_t numel = tensor.numel();
    void* data = tensor.storage()->data();

    // Fill with value based on dtype
    switch (dtype) {
        case DType::Float16: {
            Float16* ptr = static_cast<Float16*>(data);
            std::fill(ptr, ptr + numel, Float16(static_cast<float>(value)));
            break;
        }
        case DType::BFloat16: {
            BFloat16* ptr = static_cast<BFloat16*>(data);
            std::fill(ptr, ptr + numel, BFloat16(static_cast<float>(value)));
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
        case DType::UInt16: {
            uint16_t* ptr = static_cast<uint16_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<uint16_t>(value));
            break;
        }
        case DType::UInt32: {
            uint32_t* ptr = static_cast<uint32_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<uint32_t>(value));
            break;
        }
        case DType::UInt64: {
            uint64_t* ptr = static_cast<uint64_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<uint64_t>(value));
            break;
        }
        case DType::Int8: {
            int8_t* ptr = static_cast<int8_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<int8_t>(value));
            break;
        }
        case DType::Int16: {
            int16_t* ptr = static_cast<int16_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<int16_t>(value));
            break;
        }
        case DType::Bool: {
            bool* ptr = static_cast<bool*>(data);
            std::fill(ptr, ptr + numel, value != 0.0f);
            break;
        }
        case DType::Complex64: {
            auto* ptr = static_cast<std::complex<float>*>(data);
            std::fill(ptr, ptr + numel, std::complex<float>(static_cast<float>(value), 0.0f));
            break;
        }
        case DType::Complex128: {
            auto* ptr = static_cast<std::complex<double>*>(data);
            std::fill(ptr, ptr + numel, std::complex<double>(static_cast<double>(value), 0.0));
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for full()");
    }
    return tensor;
}

auto full(std::vector<int64_t> shape, double value, DType dtype, Device device) -> Tensor {
    // Use OpId dispatch for non-CPU devices
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Shape, shape_to_string(shape));
        attrs.set(AttrKey::Value, value);
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

        return dispatch_to_device(OpId::Full, device.type, {}, attrs)[0];
    }

    // CPU path: use uninitialized allocation (avoid wasteful zeroing before fill)
    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, device);
    if (!tensor.impl() || !tensor.storage()) return tensor;

    size_t numel = tensor.numel();
    void* data = tensor.storage()->data();

    // Fill with value based on dtype
    switch (dtype) {
        case DType::Float16: {
            Float16* ptr = static_cast<Float16*>(data);
            std::fill(ptr, ptr + numel, Float16(static_cast<float>(value)));
            break;
        }
        case DType::BFloat16: {
            BFloat16* ptr = static_cast<BFloat16*>(data);
            std::fill(ptr, ptr + numel, BFloat16(static_cast<float>(value)));
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
        case DType::UInt16: {
            uint16_t* ptr = static_cast<uint16_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<uint16_t>(value));
            break;
        }
        case DType::UInt32: {
            uint32_t* ptr = static_cast<uint32_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<uint32_t>(value));
            break;
        }
        case DType::UInt64: {
            uint64_t* ptr = static_cast<uint64_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<uint64_t>(value));
            break;
        }
        case DType::Int8: {
            int8_t* ptr = static_cast<int8_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<int8_t>(value));
            break;
        }
        case DType::Int16: {
            int16_t* ptr = static_cast<int16_t*>(data);
            std::fill(ptr, ptr + numel, static_cast<int16_t>(value));
            break;
        }
        case DType::Bool: {
            bool* ptr = static_cast<bool*>(data);
            std::fill(ptr, ptr + numel, value != 0.0);
            break;
        }
        case DType::Complex64: {
            auto* ptr = static_cast<std::complex<float>*>(data);
            std::fill(ptr, ptr + numel, std::complex<float>(static_cast<float>(value), 0.0f));
            break;
        }
        case DType::Complex128: {
            auto* ptr = static_cast<std::complex<double>*>(data);
            std::fill(ptr, ptr + numel, std::complex<double>(value, 0.0));
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
    // Use OpId dispatch for non-CPU devices
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Shape, shape_to_string(shape));
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

        return dispatch_to_device(OpId::Rand, device.type, {}, attrs)[0];
    }

    // CPU path: use uninitialized allocation (avoid wasteful zeroing before fill)
    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, device);
    if (!tensor.impl() || !tensor.storage()) return tensor;

    size_t numel = tensor.numel();
    void* data = tensor.storage()->data();

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
        case DType::BFloat16: {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            BFloat16* ptr = static_cast<BFloat16*>(data);
            for (size_t i = 0; i < numel; ++i) {
                ptr[i] = BFloat16(dist(gen));
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for rand() - only Float32, Float64, Float16, and BFloat16 are supported");
    }
    return tensor;
}

auto randn(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    // Use OpId dispatch for non-CPU devices
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Shape, shape_to_string(shape));
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

        return dispatch_to_device(OpId::Randn, device.type, {}, attrs)[0];
    }

    // CPU path: use uninitialized allocation (avoid wasteful zeroing before fill)
    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, device);
    if (!tensor.impl() || !tensor.storage()) return tensor;

    size_t numel = tensor.numel();
    void* data = tensor.storage()->data();

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
        case DType::BFloat16: {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            BFloat16* ptr = static_cast<BFloat16*>(data);
            for (size_t i = 0; i < numel; ++i) {
                ptr[i] = BFloat16(dist(gen));
            }
            break;
        }
        case DType::Complex64: {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            auto* ptr = static_cast<std::complex<float>*>(data);
            for (size_t i = 0; i < numel; ++i) {
                ptr[i] = {dist(gen), dist(gen)};
            }
            break;
        }
        case DType::Complex128: {
            std::normal_distribution<double> dist(0.0, 1.0);
            auto* ptr = static_cast<std::complex<double>*>(data);
            for (size_t i = 0; i < numel; ++i) {
                ptr[i] = {dist(gen), dist(gen)};
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for randn() - supported: Float32, Float64, Float16, BFloat16, Complex64, Complex128");
    }
    return tensor;
}

auto randint(int64_t low, int64_t high, std::vector<int64_t> shape, DType dtype, Device device) -> Tensor {
    if (low >= high) {
        throw std::invalid_argument("randint: low must be less than high");
    }

    // Use OpId dispatch for non-CPU devices
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Start, static_cast<int64_t>(low));
        attrs.set(AttrKey::End, static_cast<int64_t>(high));
        attrs.set(AttrKey::Shape, shape_to_string(shape));
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

        return dispatch_to_device(OpId::Randint, device.type, {}, attrs)[0];
    }

    // CPU path: generate directly
    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, Device::cpu());
    if (!tensor.impl() || !tensor.storage()) return tensor;

    size_t numel = tensor.numel();
    void* data = tensor.storage()->data();

    // Use global random number generator
    auto& gen = get_rng();
    std::uniform_int_distribution<int64_t> dist(low, high - 1);

    // Fill with random integers based on dtype
    switch (dtype) {
        case DType::Int64: {
            int64_t* ptr = static_cast<int64_t*>(data);
            for (size_t i = 0; i < numel; ++i) {
                ptr[i] = dist(gen);
            }
            break;
        }
        case DType::Int32: {
            int32_t* ptr = static_cast<int32_t*>(data);
            for (size_t i = 0; i < numel; ++i) {
                ptr[i] = static_cast<int32_t>(dist(gen));
            }
            break;
        }
        case DType::Int16: {
            int16_t* ptr = static_cast<int16_t*>(data);
            for (size_t i = 0; i < numel; ++i) {
                ptr[i] = static_cast<int16_t>(dist(gen));
            }
            break;
        }
        case DType::Int8: {
            int8_t* ptr = static_cast<int8_t*>(data);
            for (size_t i = 0; i < numel; ++i) {
                ptr[i] = static_cast<int8_t>(dist(gen));
            }
            break;
        }
        case DType::UInt8: {
            uint8_t* ptr = static_cast<uint8_t*>(data);
            for (size_t i = 0; i < numel; ++i) {
                ptr[i] = static_cast<uint8_t>(dist(gen));
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for randint() - only integer types are supported");
    }
    return tensor;
}

auto arange(double start, double end, double step, DType dtype, Device device) -> Tensor {
    if (step == 0.0) {
        throw std::invalid_argument("step cannot be zero");
    }
    if ((step > 0 && start >= end) || (step < 0 && start <= end)) {
        throw std::invalid_argument("Invalid start, end, step combination");
    }

    // Use OpId dispatch for non-CPU devices
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Start, start);
        attrs.set(AttrKey::End, end);
        attrs.set(AttrKey::Step, step);
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

        return dispatch_to_device(OpId::Arange, device.type, {}, attrs)[0];
    }

    // Calculate number of elements
    int64_t numel = static_cast<int64_t>(std::ceil((end - start) / step));
    if (numel < 0) numel = 0;

    // Use uninitialized allocation (avoid wasteful zeroing before fill)
    auto tensor = Tensor::empty_uninitialized({numel}, dtype, device);
    if (!tensor.impl() || !tensor.storage() || numel == 0) return tensor;

    void* data = tensor.storage()->data();

    // Fill with range values using start + i * step to avoid accumulation drift
    switch (dtype) {
        case DType::Float32: {
            float* ptr = static_cast<float*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = static_cast<float>(start + i * step);
            }
            break;
        }
        case DType::Float64: {
            double* ptr = static_cast<double*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = start + i * step;
            }
            break;
        }
        case DType::Int32: {
            int32_t* ptr = static_cast<int32_t*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = static_cast<int32_t>(start + i * step);
            }
            break;
        }
        case DType::Int64: {
            int64_t* ptr = static_cast<int64_t*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = static_cast<int64_t>(start + i * step);
            }
            break;
        }
        case DType::Float16: {
            Float16* ptr = static_cast<Float16*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = Float16(static_cast<float>(start + i * step));
            }
            break;
        }
        case DType::Int8: {
            int8_t* ptr = static_cast<int8_t*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = static_cast<int8_t>(start + i * step);
            }
            break;
        }
        case DType::UInt8: {
            uint8_t* ptr = static_cast<uint8_t*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = static_cast<uint8_t>(start + i * step);
            }
            break;
        }
        case DType::Int16: {
            int16_t* ptr = static_cast<int16_t*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = static_cast<int16_t>(start + i * step);
            }
            break;
        }
        case DType::BFloat16: {
            BFloat16* ptr = static_cast<BFloat16*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = BFloat16(static_cast<float>(start + i * step));
            }
            break;
        }
        case DType::UInt16: {
            uint16_t* ptr = static_cast<uint16_t*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = static_cast<uint16_t>(start + i * step);
            }
            break;
        }
        case DType::UInt32: {
            uint32_t* ptr = static_cast<uint32_t*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = static_cast<uint32_t>(start + i * step);
            }
            break;
        }
        case DType::UInt64: {
            uint64_t* ptr = static_cast<uint64_t*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = static_cast<uint64_t>(start + i * step);
            }
            break;
        }
        case DType::Bool: {
            bool* ptr = static_cast<bool*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = (start + i * step) != 0.0;
            }
            break;
        }
        case DType::Complex64: {
            auto* ptr = static_cast<std::complex<float>*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = std::complex<float>(static_cast<float>(start + i * step), 0.0f);
            }
            break;
        }
        case DType::Complex128: {
            auto* ptr = static_cast<std::complex<double>*>(data);
            for (int64_t i = 0; i < numel; ++i) {
                ptr[i] = std::complex<double>(start + i * step, 0.0);
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for arange()");
    }
    return tensor;
}

auto linspace(float start, float end, int64_t steps, DType dtype, Device device) -> Tensor {
    if (steps <= 0) {
        throw std::invalid_argument("steps must be positive");
    }

    // Use OpId dispatch for non-CPU devices
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Start, static_cast<double>(start));
        attrs.set(AttrKey::End, static_cast<double>(end));
        attrs.set(AttrKey::Steps, steps);
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

        return dispatch_to_device(OpId::Linspace, device.type, {}, attrs)[0];
    }

    // Use uninitialized allocation (avoid wasteful zeroing before fill)
    auto tensor = Tensor::empty_uninitialized({steps}, dtype, device);
    if (!tensor.impl() || !tensor.storage()) return tensor;

    void* data = tensor.storage()->data();

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
        case DType::Float16: {
            Float16* ptr = static_cast<Float16*>(data);
            if (steps == 1) {
                ptr[0] = Float16(start);
            } else {
                for (int64_t i = 0; i < steps; ++i) {
                    ptr[i] = Float16(static_cast<float>(start + i * step_size));
                }
                // Ensure the last element is exactly 'end' to avoid floating point errors
                ptr[steps - 1] = Float16(end);
            }
            break;
        }
        case DType::BFloat16: {
            BFloat16* ptr = static_cast<BFloat16*>(data);
            if (steps == 1) {
                ptr[0] = BFloat16(start);
            } else {
                for (int64_t i = 0; i < steps; ++i) {
                    ptr[i] = BFloat16(static_cast<float>(start + i * step_size));
                }
                // Ensure the last element is exactly 'end' to avoid floating point errors
                ptr[steps - 1] = BFloat16(end);
            }
            break;
        }
        case DType::Int8: {
            int8_t* ptr = static_cast<int8_t*>(data);
            for (int64_t i = 0; i < steps; ++i) {
                ptr[i] = static_cast<int8_t>(start + i * step_size);
            }
            if (steps > 1) ptr[steps - 1] = static_cast<int8_t>(end);
            break;
        }
        case DType::Int16: {
            int16_t* ptr = static_cast<int16_t*>(data);
            for (int64_t i = 0; i < steps; ++i) {
                ptr[i] = static_cast<int16_t>(start + i * step_size);
            }
            if (steps > 1) ptr[steps - 1] = static_cast<int16_t>(end);
            break;
        }
        case DType::Int32: {
            int32_t* ptr = static_cast<int32_t*>(data);
            for (int64_t i = 0; i < steps; ++i) {
                ptr[i] = static_cast<int32_t>(start + i * step_size);
            }
            if (steps > 1) ptr[steps - 1] = static_cast<int32_t>(end);
            break;
        }
        case DType::Int64: {
            int64_t* ptr = static_cast<int64_t*>(data);
            for (int64_t i = 0; i < steps; ++i) {
                ptr[i] = static_cast<int64_t>(start + i * step_size);
            }
            if (steps > 1) ptr[steps - 1] = static_cast<int64_t>(end);
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for linspace()");
    }
    return tensor;
}

auto eye(int64_t n, std::optional<int64_t> m, DType dtype, Device device) -> Tensor {
    int64_t cols = m.value_or(n);

    // Use OpId dispatch for non-CPU devices
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::N, n);
        attrs.set(AttrKey::M, cols);
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));

        return dispatch_to_device(OpId::Eye, device.type, {}, attrs)[0];
    }

    // Start with zeros (now safe since we're on CPU)
    auto tensor = zeros({n, cols}, dtype, device);
    if (!tensor.impl() || !tensor.storage()) return tensor;

    void* data = tensor.storage()->data();
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
        case DType::Float16: {
            Float16* ptr = static_cast<Float16*>(data);
            Float16 one(1.0f);
            for (int64_t i = 0; i < min_dim; ++i) {
                ptr[i * cols + i] = one;
            }
            break;
        }
        case DType::Int8: {
            int8_t* ptr = static_cast<int8_t*>(data);
            for (int64_t i = 0; i < min_dim; ++i) {
                ptr[i * cols + i] = 1;
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

auto meshgrid(const std::vector<Tensor>& tensors, const std::string& indexing) -> std::vector<Tensor> {
    if (tensors.empty()) return {};
    if (indexing != "ij" && indexing != "xy")
        throw std::invalid_argument("meshgrid: indexing must be 'ij' or 'xy'");

    size_t ndim = tensors.size();
    for (size_t i = 0; i < ndim; ++i) {
        if (tensors[i].ndim() != 1)
            throw std::invalid_argument("meshgrid: all input tensors must be 1-D");
    }

    // Build output shape
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < ndim; ++i) {
        out_shape.push_back(tensors[i].numel());
    }

    // For "xy" indexing with >= 2 tensors, swap first two dimensions
    if (indexing == "xy" && ndim >= 2) {
        std::swap(out_shape[0], out_shape[1]);
    }

    // Total elements per grid
    int64_t total = 1;
    for (auto s : out_shape) total *= s;

    std::vector<Tensor> result;
    for (size_t ti = 0; ti < ndim; ++ti) {
        // Determine which output dimension this tensor varies along
        size_t dim_idx = ti;
        if (indexing == "xy" && ndim >= 2) {
            if (ti == 0) dim_idx = 1;
            else if (ti == 1) dim_idx = 0;
        }

        auto grid = empty(out_shape, tensors[ti].dtype(), tensors[ti].device());
        auto src = tensors[ti].contiguous();
        size_t elem_size = dtype_size(src.dtype());

        // Compute strides for index calculation
        std::vector<int64_t> strides(ndim);
        int64_t stride = 1;
        for (int64_t d = static_cast<int64_t>(ndim) - 1; d >= 0; --d) {
            strides[d] = stride;
            stride *= out_shape[d];
        }

        auto* dst = static_cast<uint8_t*>(grid.data_ptr());
        auto* src_ptr = static_cast<const uint8_t*>(src.data_ptr());

        for (int64_t flat = 0; flat < total; ++flat) {
            // Extract index along dim_idx from flat index
            int64_t idx = (flat / strides[dim_idx]) % out_shape[dim_idx];
            std::memcpy(dst + flat * elem_size, src_ptr + idx * elem_size, elem_size);
        }

        result.push_back(grid);
    }

    return result;
}

auto multinomial(const Tensor& input, int64_t num_samples, bool replacement) -> Tensor {
    if (input.ndim() < 1 || input.ndim() > 2) {
        throw std::invalid_argument("multinomial: input must be 1D or 2D");
    }
    if (num_samples <= 0) {
        throw std::invalid_argument("multinomial: num_samples must be positive");
    }

    auto inp = input.contiguous();
    std::array<Tensor, 1> inputs = {inp};
    NewOpAttributes attrs;
    attrs.set(AttrKey::NumSamples, num_samples);
    attrs.set(AttrKey::Replacement, replacement);
    return dispatch<OpId::Multinomial>(inputs, attrs)[0];
}

auto bernoulli(const Tensor& probs) -> Tensor {
    auto inp = probs.contiguous();
    std::array<Tensor, 1> inputs = {inp};
    return dispatch<OpId::Bernoulli>(inputs)[0];
}

auto normal(const Tensor& mean, const Tensor& std) -> Tensor {
    auto m = mean.contiguous();
    auto s = std.contiguous();
    std::array<Tensor, 2> inputs = {m, s};
    return dispatch<OpId::NormalSample>(inputs)[0];
}

auto poisson(const Tensor& rates) -> Tensor {
    // Dispatch to backend-specific kernel for proper Poisson sampling
    auto inp = rates.contiguous();
    std::array<Tensor, 1> inputs = {inp};
    return dispatch<OpId::PoissonSample>(inputs)[0];
}

auto exponential(const Tensor& rate) -> Tensor {
    auto r = rate.contiguous();
    std::array<Tensor, 1> inputs = {r};
    return dispatch<OpId::ExponentialSample>(inputs)[0];
}

// ============================================================================
// Generator-aware overloads
// ============================================================================

auto rand(std::vector<int64_t> shape, DType dtype, Device device,
         Generator& generator) -> Tensor {
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Shape, shape_to_string(shape));
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));
        attrs.set(AttrKey::Seed, static_cast<int64_t>(generator.next_seed()));
        return dispatch_to_device(OpId::Rand, device.type, {}, attrs)[0];
    }

    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, device);
    if (!tensor.impl() || !tensor.storage()) return tensor;

    size_t numel = tensor.numel();
    void* data = tensor.storage()->data();
    auto& eng = generator.engine();

    switch (dtype) {
        case DType::Float32: {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            auto* ptr = static_cast<float*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = dist(eng);
            break;
        }
        case DType::Float64: {
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            auto* ptr = static_cast<double*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = dist(eng);
            break;
        }
        case DType::Float16: {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            auto* ptr = static_cast<Float16*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = Float16(dist(eng));
            break;
        }
        case DType::BFloat16: {
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            auto* ptr = static_cast<BFloat16*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = BFloat16(dist(eng));
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for rand() with Generator");
    }
    return tensor;
}

auto randn(std::vector<int64_t> shape, DType dtype, Device device,
          Generator& generator) -> Tensor {
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Shape, shape_to_string(shape));
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));
        attrs.set(AttrKey::Seed, static_cast<int64_t>(generator.next_seed()));
        return dispatch_to_device(OpId::Randn, device.type, {}, attrs)[0];
    }

    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, device);
    if (!tensor.impl() || !tensor.storage()) return tensor;

    size_t numel = tensor.numel();
    void* data = tensor.storage()->data();
    auto& eng = generator.engine();

    switch (dtype) {
        case DType::Float32: {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            auto* ptr = static_cast<float*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = dist(eng);
            break;
        }
        case DType::Float64: {
            std::normal_distribution<double> dist(0.0, 1.0);
            auto* ptr = static_cast<double*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = dist(eng);
            break;
        }
        case DType::Float16: {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            auto* ptr = static_cast<Float16*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = Float16(dist(eng));
            break;
        }
        case DType::BFloat16: {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            auto* ptr = static_cast<BFloat16*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = BFloat16(dist(eng));
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for randn() with Generator");
    }
    return tensor;
}

auto randint(int64_t low, int64_t high, std::vector<int64_t> shape,
            DType dtype, Device device, Generator& generator) -> Tensor {
    if (device.type != Device::Type::CPU) {
        OpAttributes attrs;
        attrs.set(AttrKey::Shape, shape_to_string(shape));
        attrs.set(AttrKey::Dtype, dtype_to_string(dtype));
        attrs.set(AttrKey::Device, static_cast<int64_t>(device.index));
        attrs.set(AttrKey::Low, low);
        attrs.set(AttrKey::High, high);
        attrs.set(AttrKey::Seed, static_cast<int64_t>(generator.next_seed()));
        return dispatch_to_device(OpId::Randint, device.type, {}, attrs)[0];
    }

    auto tensor = Tensor::empty_uninitialized(std::move(shape), dtype, device);
    if (!tensor.impl() || !tensor.storage()) return tensor;

    size_t numel = tensor.numel();
    void* data = tensor.storage()->data();
    auto& eng = generator.engine();
    std::uniform_int_distribution<int64_t> dist(low, high - 1);

    switch (dtype) {
        case DType::Int32: {
            auto* ptr = static_cast<int32_t*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = static_cast<int32_t>(dist(eng));
            break;
        }
        case DType::Int64: {
            auto* ptr = static_cast<int64_t*>(data);
            for (size_t i = 0; i < numel; ++i) ptr[i] = dist(eng);
            break;
        }
        default:
            throw std::runtime_error("Unsupported dtype for randint() with Generator");
    }
    return tensor;
}

auto multinomial(const Tensor& input, int64_t num_samples, bool replacement,
                Generator& generator) -> Tensor {
    if (input.ndim() < 1 || input.ndim() > 2) {
        throw std::invalid_argument("multinomial: input must be 1D or 2D");
    }
    if (num_samples <= 0) {
        throw std::invalid_argument("multinomial: num_samples must be positive");
    }

    auto inp = input.contiguous();
    std::array<Tensor, 1> inputs = {inp};
    NewOpAttributes attrs;
    attrs.set(AttrKey::NumSamples, num_samples);
    attrs.set(AttrKey::Replacement, replacement);
    attrs.set(AttrKey::Seed, static_cast<int64_t>(generator.next_seed()));
    return dispatch<OpId::Multinomial>(inputs, attrs)[0];
}

auto bernoulli(const Tensor& probs, Generator& generator) -> Tensor {
    auto inp = probs.contiguous();
    std::array<Tensor, 1> inputs = {inp};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Seed, static_cast<int64_t>(generator.next_seed()));
    return dispatch<OpId::Bernoulli>(inputs, attrs)[0];
}

auto normal(const Tensor& mean, const Tensor& std, Generator& generator) -> Tensor {
    auto m = mean.contiguous();
    auto s = std.contiguous();
    std::array<Tensor, 2> inputs = {m, s};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Seed, static_cast<int64_t>(generator.next_seed()));
    return dispatch<OpId::NormalSample>(inputs, attrs)[0];
}

auto poisson(const Tensor& rates, Generator& generator) -> Tensor {
    auto inp = rates.contiguous();
    std::array<Tensor, 1> inputs = {inp};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Seed, static_cast<int64_t>(generator.next_seed()));
    return dispatch<OpId::PoissonSample>(inputs, attrs)[0];
}

auto exponential(const Tensor& rate, Generator& generator) -> Tensor {
    auto r = rate.contiguous();
    std::array<Tensor, 1> inputs = {r};
    NewOpAttributes attrs;
    attrs.set(AttrKey::Seed, static_cast<int64_t>(generator.next_seed()));
    return dispatch<OpId::ExponentialSample>(inputs, attrs)[0];
}

auto randperm(int64_t n, Device device, Generator& generator) -> Tensor {
    // Generate sequential tensor then shuffle with Generator's engine
    auto result = tenzor::arange(0.0, static_cast<double>(n), 1.0, DType::Int64, device);
    if (device.type == Device::Type::CPU && n > 1) {
        auto& eng = generator.engine();
        auto* data = result.data<int64_t>();
        for (int64_t i = n - 1; i > 0; --i) {
            std::uniform_int_distribution<int64_t> dist(0, i);
            int64_t j = dist(eng);
            std::swap(data[i], data[j]);
        }
    }
    return result;
}

auto logspace(float start, float end, int64_t steps, double base,
              DType dtype, Device device) -> Tensor {
    auto exponents = tenzor::linspace(start, end, steps, dtype, device);
    // base^exponents = exp(log(base) * exponents)
    float log_base = static_cast<float>(std::log(base));
    auto scaled = tenzor::mul(exponents, tenzor::full({1}, log_base, dtype, device));
    return tenzor::exp(scaled);
}

// =========================================================================
// New creation operations for PyTorch parity (compositions)
// =========================================================================

auto full_like(const Tensor& tensor, double fill_value) -> Tensor {
    std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
    return full(shape, fill_value, tensor.dtype(), tensor.device());
}

auto empty_like(const Tensor& tensor) -> Tensor {
    std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
    return empty(shape, tensor.dtype(), tensor.device());
}

auto randint_like(const Tensor& tensor, int64_t low, int64_t high) -> Tensor {
    std::vector<int64_t> shape(tensor.shape().begin(), tensor.shape().end());
    return randint(low, high, shape, tensor.dtype(), tensor.device());
}

auto tril_indices(int64_t row, int64_t col, int64_t offset,
                  DType /*dtype*/, Device device) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::M, row);
    attrs.set(AttrKey::N, col);
    attrs.set(AttrKey::Diagonal, offset);
    std::vector<Tensor> inputs;  // no input tensors
    return dispatch_to_device(OpId::TrilIndices, device.type, inputs, attrs)[0];
}

auto triu_indices(int64_t row, int64_t col, int64_t offset,
                  DType /*dtype*/, Device device) -> Tensor {
    NewOpAttributes attrs;
    attrs.set(AttrKey::M, row);
    attrs.set(AttrKey::N, col);
    attrs.set(AttrKey::Diagonal, offset);
    std::vector<Tensor> inputs;  // no input tensors
    return dispatch_to_device(OpId::TriuIndices, device.type, inputs, attrs)[0];
}

auto complex(const Tensor& real, const Tensor& imag) -> Tensor {
    // Validate inputs
    auto real_shape = real.shape();
    auto imag_shape = imag.shape();
    if (real_shape.size() != imag_shape.size()) {
        throw std::invalid_argument("complex: real and imag must have the same number of dimensions");
    }
    for (size_t i = 0; i < real_shape.size(); ++i) {
        if (real_shape[i] != imag_shape[i]) {
            throw std::invalid_argument("complex: real and imag must have the same shape");
        }
    }
    if (real.device().type != imag.device().type) {
        throw std::invalid_argument("complex: real and imag must be on the same device");
    }

    // Dispatch through backend for GPU tensors
    if (real.device().type != Device::Type::CPU) {
        std::array<Tensor, 2> inputs = {real, imag};
        return dispatch_single(OpId::ComplexTensor, inputs);
    }

    // CPU path: interleave real and imag into Complex64 storage
    auto r = real.to(DType::Float32).contiguous();
    auto im = imag.to(DType::Float32).contiguous();

    std::vector<int64_t> shape_vec(real_shape.begin(), real_shape.end());
    auto result = empty(shape_vec, DType::Complex64, Device::cpu());

    const float* r_data = r.data<float>();
    const float* i_data = im.data<float>();
    auto* c_data = reinterpret_cast<std::complex<float>*>(result.storage()->data());
    int64_t offset = result.offset();

    int64_t numel = r.numel();
    for (int64_t idx = 0; idx < numel; ++idx) {
        c_data[offset + idx] = std::complex<float>(r_data[idx], i_data[idx]);
    }

    return result;
}

} // namespace tenzor
