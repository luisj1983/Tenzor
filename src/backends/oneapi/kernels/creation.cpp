#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>
#include <random>
#include <chrono>

#ifdef TENZOR_HAS_ONEMKL
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng.hpp>
// Note: Use ::oneapi::mkl to avoid namespace conflict with tenzor::oneapi
#endif

namespace tenzor {
namespace oneapi {

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

// BFloat16 conversion helpers
inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}
inline uint16_t f32_to_bf16(float f32) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f32, sizeof(uint32_t));
    return static_cast<uint16_t>(bits >> 16);
}

/**
 * @brief Generate random numbers from a standard normal distribution (Gaussian with mean=0, stddev=1)
 *
 * Uses Intel oneMKL VSL (Vector Statistics Library) when available for high-performance RNG.
 * Falls back to CPU-based generation using std::normal_distribution when oneMKL is not available.
 *
 * @param shape Shape of the output tensor
 * @param dtype Data type (Float32 or Float64)
 * @param device Device to allocate the tensor on
 * @param queue SYCL queue for execution
 * @return Tensor filled with random values from N(0, 1) distribution
 */
auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, sycl::queue& queue) -> Tensor {
    Tensor output(shape, dtype, device);
    const int64_t numel = output.numel();

    if (numel == 0) {
        return output;
    }

#ifdef TENZOR_HAS_ONEMKL
    // Use Intel oneMKL VSL for high-performance random number generation

    // Use global seed (respects manual_seed) or fall back to time-based
    auto seed = tenzor::get_global_seed();

    try {
        if (dtype == DType::Float32) {
            float* ptr = get_data_ptr<float>(output);

            // Create Philox4x32x10 engine (good quality, high performance)
            ::oneapi::mkl::rng::philox4x32x10 engine(queue, seed);

            // Generate Gaussian distribution with mean=0.0, stddev=1.0
            ::oneapi::mkl::rng::gaussian<float> distribution(0.0f, 1.0f);

            // Generate random numbers directly into device memory
            ::oneapi::mkl::rng::generate(distribution, engine, numel, ptr);
        }
        else if (dtype == DType::Float64) {
            double* ptr = get_data_ptr<double>(output);

            // Create Philox4x32x10 engine
            ::oneapi::mkl::rng::philox4x32x10 engine(queue, seed);

            // Generate Gaussian distribution with mean=0.0, stddev=1.0
            ::oneapi::mkl::rng::gaussian<double> distribution(0.0, 1.0);

            // Generate random numbers directly into device memory
            ::oneapi::mkl::rng::generate(distribution, engine, numel, ptr);
        }
        else if (dtype == DType::Float16) {
            // oneMKL doesn't support half precision directly, so generate float32 and convert
            sycl::half* ptr = get_data_ptr<sycl::half>(output);

            // Create temporary float32 buffer
            Tensor temp_buffer({numel}, DType::Float32, device);
            float* temp_ptr = get_data_ptr<float>(temp_buffer);

            // Create Philox4x32x10 engine
            ::oneapi::mkl::rng::philox4x32x10 engine(queue, seed);

            // Generate Gaussian distribution with mean=0.0, stddev=1.0
            ::oneapi::mkl::rng::gaussian<float> distribution(0.0f, 1.0f);

            // Generate random numbers into temp buffer
            ::oneapi::mkl::rng::generate(distribution, engine, numel, temp_ptr);

            // Convert from float32 to float16
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
                ptr[i] = sycl::half(temp_ptr[i]);
            });
        }
        else if (dtype == DType::BFloat16) {
            // Generate Float32 then convert to BFloat16
            uint16_t* ptr = get_data_ptr<uint16_t>(output);

            Tensor temp_buffer({numel}, DType::Float32, device);
            float* temp_ptr = get_data_ptr<float>(temp_buffer);

            ::oneapi::mkl::rng::philox4x32x10 engine(queue, seed);
            ::oneapi::mkl::rng::gaussian<float> distribution(0.0f, 1.0f);
            ::oneapi::mkl::rng::generate(distribution, engine, numel, temp_ptr);

            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
                uint32_t bits;
                __builtin_memcpy(&bits, &temp_ptr[i], sizeof(uint32_t));
                ptr[i] = static_cast<uint16_t>(bits >> 16);
            });
        }
        else {
            throw std::runtime_error("Unsupported dtype for randn (oneMKL path)");
        }
    }
    catch (const ::oneapi::mkl::exception& e) {
        throw std::runtime_error(
            std::string("oneMKL RNG failed: ") + e.what()
        );
    }
#else
    // Fallback: Generate on host using std::normal_distribution and copy to device

    // Use global seed (respects manual_seed) or fall back to time-based
    auto seed = static_cast<unsigned int>(tenzor::get_global_seed());

    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    if (dtype == DType::Float32) {
        std::vector<float> host_data(numel);
        for (int64_t i = 0; i < numel; ++i) {
            host_data[i] = static_cast<float>(dist(gen));
        }

        float* device_ptr = get_data_ptr<float>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(float)).wait();
    }
    else if (dtype == DType::Float64) {
        std::vector<double> host_data(numel);
        for (int64_t i = 0; i < numel; ++i) {
            host_data[i] = dist(gen);
        }

        double* device_ptr = get_data_ptr<double>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(double)).wait();
    }
    else if (dtype == DType::Float16) {
        // Generate float32 on host and convert to float16
        std::vector<float> host_data(numel);
        for (int64_t i = 0; i < numel; ++i) {
            host_data[i] = static_cast<float>(dist(gen));
        }

        // Upload to temp buffer and convert on device
        Tensor temp_buffer({numel}, DType::Float32, device);
        float* temp_ptr = get_data_ptr<float>(temp_buffer);
        queue.memcpy(temp_ptr, host_data.data(), numel * sizeof(float)).wait();

        sycl::half* device_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = sycl::half(temp_ptr[i]);
        });
    }
    else if (dtype == DType::BFloat16) {
        std::vector<float> host_data(numel);
        for (int64_t i = 0; i < numel; ++i) {
            host_data[i] = static_cast<float>(dist(gen));
        }

        Tensor temp_buffer({numel}, DType::Float32, device);
        float* temp_ptr = get_data_ptr<float>(temp_buffer);
        queue.memcpy(temp_ptr, host_data.data(), numel * sizeof(float)).wait();

        uint16_t* device_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            uint32_t bits;
            __builtin_memcpy(&bits, &temp_ptr[i], sizeof(uint32_t));
            device_ptr[i] = static_cast<uint16_t>(bits >> 16);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for randn (fallback path)");
    }
#endif

    return output;
}

/**
 * @brief Generate random numbers from a uniform distribution [0, 1)
 *
 * Uses Intel oneMKL VSL when available. Falls back to CPU generation otherwise.
 *
 * @param shape Shape of the output tensor
 * @param dtype Data type (Float32 or Float64)
 * @param device Device to allocate the tensor on
 * @param queue SYCL queue for execution
 * @return Tensor filled with random values from U(0, 1) distribution
 */
auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, sycl::queue& queue) -> Tensor {
    Tensor output(shape, dtype, device);
    const int64_t numel = output.numel();

    if (numel == 0) {
        return output;
    }

#ifdef TENZOR_HAS_ONEMKL
    // Use Intel oneMKL VSL for high-performance random number generation

    // Use global seed (respects manual_seed) or fall back to time-based
    auto seed = tenzor::get_global_seed();

    try {
        if (dtype == DType::Float32) {
            float* ptr = get_data_ptr<float>(output);

            ::oneapi::mkl::rng::philox4x32x10 engine(queue, seed);

            // Generate uniform distribution [0, 1)
            ::oneapi::mkl::rng::uniform<float> distribution(0.0f, 1.0f);

            ::oneapi::mkl::rng::generate(distribution, engine, numel, ptr);
        }
        else if (dtype == DType::Float64) {
            double* ptr = get_data_ptr<double>(output);

            ::oneapi::mkl::rng::philox4x32x10 engine(queue, seed);

            ::oneapi::mkl::rng::uniform<double> distribution(0.0, 1.0);

            ::oneapi::mkl::rng::generate(distribution, engine, numel, ptr);
        }
        else if (dtype == DType::Float16) {
            // Generate Float32 then convert to Float16
            sycl::half* ptr = get_data_ptr<sycl::half>(output);

            Tensor temp_buffer({numel}, DType::Float32, device);
            float* temp_ptr = get_data_ptr<float>(temp_buffer);

            ::oneapi::mkl::rng::philox4x32x10 engine(queue, seed);
            ::oneapi::mkl::rng::uniform<float> distribution(0.0f, 1.0f);
            ::oneapi::mkl::rng::generate(distribution, engine, numel, temp_ptr);

            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
                ptr[i] = sycl::half(temp_ptr[i]);
            });
        }
        else if (dtype == DType::BFloat16) {
            // Generate Float32 then convert to BFloat16 (truncate upper 16 bits)
            uint16_t* ptr = get_data_ptr<uint16_t>(output);

            Tensor temp_buffer({numel}, DType::Float32, device);
            float* temp_ptr = get_data_ptr<float>(temp_buffer);

            ::oneapi::mkl::rng::philox4x32x10 engine(queue, seed);
            ::oneapi::mkl::rng::uniform<float> distribution(0.0f, 1.0f);
            ::oneapi::mkl::rng::generate(distribution, engine, numel, temp_ptr);

            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
                // BFloat16: upper 16 bits of float32
                uint32_t bits;
                __builtin_memcpy(&bits, &temp_ptr[i], sizeof(uint32_t));
                ptr[i] = static_cast<uint16_t>(bits >> 16);
            });
        }
        else {
            throw std::runtime_error("Unsupported dtype for rand (oneMKL path)");
        }
    }
    catch (const ::oneapi::mkl::exception& e) {
        throw std::runtime_error(
            std::string("oneMKL RNG failed: ") + e.what()
        );
    }
#else
    // Fallback: Generate on host using std::uniform_real_distribution

    // Use global seed (respects manual_seed) or fall back to time-based
    auto seed_val = static_cast<unsigned int>(tenzor::get_global_seed());

    std::mt19937 gen(seed_val);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    if (dtype == DType::Float32) {
        std::vector<float> host_data(numel);
        for (int64_t i = 0; i < numel; ++i) {
            host_data[i] = static_cast<float>(dist(gen));
        }

        float* device_ptr = get_data_ptr<float>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(float)).wait();
    }
    else if (dtype == DType::Float64) {
        std::vector<double> host_data(numel);
        for (int64_t i = 0; i < numel; ++i) {
            host_data[i] = dist(gen);
        }

        double* device_ptr = get_data_ptr<double>(output);
        queue.memcpy(device_ptr, host_data.data(), numel * sizeof(double)).wait();
    }
    else if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        // Generate Float32 on host, upload, convert on device
        std::vector<float> host_data(numel);
        for (int64_t i = 0; i < numel; ++i) {
            host_data[i] = static_cast<float>(dist(gen));
        }

        Tensor temp_buffer({numel}, DType::Float32, device);
        float* temp_ptr = get_data_ptr<float>(temp_buffer);
        queue.memcpy(temp_ptr, host_data.data(), numel * sizeof(float)).wait();

        if (dtype == DType::Float16) {
            sycl::half* device_ptr = get_data_ptr<sycl::half>(output);
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
                device_ptr[i] = sycl::half(temp_ptr[i]);
            });
        } else {
            uint16_t* device_ptr = get_data_ptr<uint16_t>(output);
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
                uint32_t bits;
                __builtin_memcpy(&bits, &temp_ptr[i], sizeof(uint32_t));
                device_ptr[i] = static_cast<uint16_t>(bits >> 16);
            });
        }
    }
    else {
        throw std::runtime_error("Unsupported dtype for rand (fallback path)");
    }
#endif

    return output;
}

/**
 * @brief Generate a 1D tensor of evenly spaced values in [start, end) with given step
 */
auto arange_kernel(double start, double end, double step, DType dtype, Device device, sycl::queue& queue) -> Tensor {
    int64_t numel = static_cast<int64_t>(std::ceil((end - start) / step));
    if (numel <= 0) numel = 0;

    Tensor output({numel}, dtype, device);
    if (numel == 0) return output;

    if (dtype == DType::Float32) {
        float* ptr = get_data_ptr<float>(output);
        float s = static_cast<float>(start), st = static_cast<float>(step);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            ptr[i] = s + static_cast<float>(i[0]) * st;
        });
    }
    else if (dtype == DType::Float64) {
        double* ptr = get_data_ptr<double>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            ptr[i] = start + static_cast<double>(i[0]) * step;
        });
    }
    else if (dtype == DType::Float16) {
        sycl::half* ptr = get_data_ptr<sycl::half>(output);
        float s = static_cast<float>(start), st = static_cast<float>(step);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            ptr[i] = sycl::half(s + static_cast<float>(i[0]) * st);
        });
    }
    else if (dtype == DType::BFloat16) {
        uint16_t* ptr = get_data_ptr<uint16_t>(output);
        float s = static_cast<float>(start), st = static_cast<float>(step);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            ptr[i] = f32_to_bf16(s + static_cast<float>(i[0]) * st);
        });
    }
    else if (dtype == DType::Int32) {
        int32_t* ptr = get_data_ptr<int32_t>(output);
        int32_t s = static_cast<int32_t>(start), st = static_cast<int32_t>(step);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            ptr[i] = s + static_cast<int32_t>(i[0]) * st;
        });
    }
    else if (dtype == DType::Int64) {
        int64_t* ptr = get_data_ptr<int64_t>(output);
        int64_t s = static_cast<int64_t>(start), st = static_cast<int64_t>(step);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            ptr[i] = s + static_cast<int64_t>(i[0]) * st;
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for arange kernel");
    }

    return output;
}

/**
 * @brief Generate a 1D tensor of `steps` evenly spaced values in [start, end]
 */
auto linspace_kernel(double start, double end, int64_t steps, DType dtype, Device device, sycl::queue& queue) -> Tensor {
    Tensor output({steps}, dtype, device);
    if (steps == 0) return output;

    if (steps == 1) {
        // Single element: just start
        if (dtype == DType::Float32) {
            float val = static_cast<float>(start);
            float* ptr = get_data_ptr<float>(output);
            queue.memcpy(ptr, &val, sizeof(float)).wait();
        } else if (dtype == DType::Float64) {
            double* ptr = get_data_ptr<double>(output);
            queue.memcpy(ptr, &start, sizeof(double)).wait();
        } else if (dtype == DType::Float16) {
            sycl::half val = sycl::half(static_cast<float>(start));
            sycl::half* ptr = get_data_ptr<sycl::half>(output);
            queue.memcpy(ptr, &val, sizeof(sycl::half)).wait();
        } else if (dtype == DType::BFloat16) {
            uint16_t val = f32_to_bf16(static_cast<float>(start));
            uint16_t* ptr = get_data_ptr<uint16_t>(output);
            queue.memcpy(ptr, &val, sizeof(uint16_t)).wait();
        }
        return output;
    }

    double step_size = (end - start) / static_cast<double>(steps - 1);

    if (dtype == DType::Float32) {
        float* ptr = get_data_ptr<float>(output);
        float s = static_cast<float>(start), e = static_cast<float>(end), st = static_cast<float>(step_size);
        int64_t n = steps;
        queue.parallel_for(sycl::range<1>(steps), [=](sycl::id<1> i) {
            if (static_cast<int64_t>(i[0]) == n - 1) {
                ptr[i] = e;  // Ensure last element is exactly end
            } else {
                ptr[i] = s + static_cast<float>(i[0]) * st;
            }
        });
    }
    else if (dtype == DType::Float64) {
        double* ptr = get_data_ptr<double>(output);
        int64_t n = steps;
        queue.parallel_for(sycl::range<1>(steps), [=](sycl::id<1> i) {
            if (static_cast<int64_t>(i[0]) == n - 1) {
                ptr[i] = end;
            } else {
                ptr[i] = start + static_cast<double>(i[0]) * step_size;
            }
        });
    }
    else if (dtype == DType::Float16) {
        sycl::half* ptr = get_data_ptr<sycl::half>(output);
        float s = static_cast<float>(start), e = static_cast<float>(end), st = static_cast<float>(step_size);
        int64_t n = steps;
        queue.parallel_for(sycl::range<1>(steps), [=](sycl::id<1> i) {
            if (static_cast<int64_t>(i[0]) == n - 1) {
                ptr[i] = sycl::half(e);
            } else {
                ptr[i] = sycl::half(s + static_cast<float>(i[0]) * st);
            }
        });
    }
    else if (dtype == DType::BFloat16) {
        uint16_t* ptr = get_data_ptr<uint16_t>(output);
        float s = static_cast<float>(start), e = static_cast<float>(end), st = static_cast<float>(step_size);
        int64_t n = steps;
        queue.parallel_for(sycl::range<1>(steps), [=](sycl::id<1> i) {
            if (static_cast<int64_t>(i[0]) == n - 1) {
                ptr[i] = f32_to_bf16(e);
            } else {
                ptr[i] = f32_to_bf16(s + static_cast<float>(i[0]) * st);
            }
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for linspace kernel");
    }

    return output;
}

/**
 * @brief Generate a 2D identity matrix of shape [n, m]
 */
auto eye_kernel(int64_t n, int64_t m, DType dtype, Device device, sycl::queue& queue) -> Tensor {
    Tensor output({n, m}, dtype, device);
    int64_t total = n * m;

    // Zero-initialize first
    queue.memset(const_cast<void*>(output.data_ptr()), 0, total * output.dtype_size());

    int64_t diag_len = std::min(n, m);

    if (dtype == DType::Float32) {
        float* ptr = get_data_ptr<float>(output);
        queue.parallel_for(sycl::range<1>(diag_len), [=](sycl::id<1> i) {
            ptr[i[0] * m + i[0]] = 1.0f;
        });
    }
    else if (dtype == DType::Float64) {
        double* ptr = get_data_ptr<double>(output);
        queue.parallel_for(sycl::range<1>(diag_len), [=](sycl::id<1> i) {
            ptr[i[0] * m + i[0]] = 1.0;
        });
    }
    else if (dtype == DType::Float16) {
        sycl::half* ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for(sycl::range<1>(diag_len), [=](sycl::id<1> i) {
            ptr[i[0] * m + i[0]] = sycl::half(1.0f);
        });
    }
    else if (dtype == DType::BFloat16) {
        uint16_t* ptr = get_data_ptr<uint16_t>(output);
        uint16_t one = f32_to_bf16(1.0f);
        queue.parallel_for(sycl::range<1>(diag_len), [=](sycl::id<1> i) {
            ptr[i[0] * m + i[0]] = one;
        });
    }
    else if (dtype == DType::Int32) {
        int32_t* ptr = get_data_ptr<int32_t>(output);
        queue.parallel_for(sycl::range<1>(diag_len), [=](sycl::id<1> i) {
            ptr[i[0] * m + i[0]] = 1;
        });
    }
    else if (dtype == DType::Int64) {
        int64_t* ptr = get_data_ptr<int64_t>(output);
        queue.parallel_for(sycl::range<1>(diag_len), [=](sycl::id<1> i) {
            ptr[i[0] * m + i[0]] = 1;
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for eye kernel");
    }

    return output;
}

// ============================================================================
// Randint - Random integers in [low, high)
// ============================================================================

struct RandintKernelInt32 {};
struct RandintKernelInt64 {};

auto randint_kernel(int64_t low, int64_t high, const std::vector<int64_t>& shape,
                    DType dtype, Device device, sycl::queue& queue) -> Tensor {
    Tensor output(shape, dtype, device);
    const int64_t numel = output.numel();
    if (numel == 0) return output;

    auto seed = tenzor::get_global_seed();
    int64_t range = high - low;

    if (dtype == DType::Int32) {
        int32_t* ptr = get_data_ptr<int32_t>(output);
        int32_t lo = static_cast<int32_t>(low);
        int32_t r = static_cast<int32_t>(range);
        uint64_t s = seed;
        queue.parallel_for<RandintKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            // Philox-inspired hash for uniform random
            uint64_t x = static_cast<uint64_t>(idx[0]) ^ s;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            x = x ^ (x >> 31);
            ptr[idx] = lo + static_cast<int32_t>(x % static_cast<uint64_t>(r));
        });
    } else if (dtype == DType::Int64) {
        int64_t* ptr = get_data_ptr<int64_t>(output);
        uint64_t s = seed;
        queue.parallel_for<RandintKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            uint64_t x = static_cast<uint64_t>(idx[0]) ^ s;
            x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
            x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
            x = x ^ (x >> 31);
            ptr[idx] = low + static_cast<int64_t>(x % static_cast<uint64_t>(range));
        });
    } else {
        throw std::runtime_error("randint_kernel: only Int32 and Int64 supported");
    }

    return output;
}

} // namespace oneapi
} // namespace tenzor
