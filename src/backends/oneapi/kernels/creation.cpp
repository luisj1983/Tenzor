#include "tenzor/core/tensor.hpp"
#include "oneapi_kernel_utils.hpp"
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



/**
 * @brief Generate random numbers from a standard normal distribution (Gaussian with mean=0, stddev=1)
 *
 * Uses Intel oneMKL VSL (Vector Statistics Library) when available for high-performance RNG.
 * Falls back to an on-device Philox 4x32-10 + Box-Muller SYCL kernel when oneMKL is not available.
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

            // Convert from float32 to float16. Wait so the function-local
            // temp_buffer is not freed while this conversion kernel still reads
            // temp_ptr, and so the host sees the written output.
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
                ptr[i] = sycl::half(temp_ptr[i]);
            }).wait();
        }
        else if (dtype == DType::BFloat16) {
            // Generate Float32 then convert to BFloat16
            uint16_t* ptr = get_data_ptr<uint16_t>(output);

            Tensor temp_buffer({numel}, DType::Float32, device);
            float* temp_ptr = get_data_ptr<float>(temp_buffer);

            ::oneapi::mkl::rng::philox4x32x10 engine(queue, seed);
            ::oneapi::mkl::rng::gaussian<float> distribution(0.0f, 1.0f);
            ::oneapi::mkl::rng::generate(distribution, engine, numel, temp_ptr);

            // Wait so temp_buffer outlives the conversion kernel reading temp_ptr.
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
                uint32_t bits;
                __builtin_memcpy(&bits, &temp_ptr[i], sizeof(uint32_t));
                ptr[i] = static_cast<uint16_t>(bits >> 16);
            }).wait();
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
#pragma message("WARNING: Building without oneMKL — using on-device Philox RNG fallback")
    // On-device Philox 4x32-10 counter-based RNG with Box-Muller for normal distribution.
    // Each work-item uses its index as counter and the global seed as key.

    uint64_t seed64 = static_cast<uint64_t>(tenzor::get_global_seed());

    // Generate Float32 on device, then convert if needed
    // We generate pairs via Box-Muller, so allocate (numel + 1) / 2 * 2 floats
    int64_t padded = ((numel + 1) / 2) * 2;
    Tensor f32_buf({padded}, DType::Float32, device);
    float* f32_ptr = get_data_ptr<float>(f32_buf);

    // Number of pairs for Box-Muller
    int64_t num_pairs = padded / 2;

    queue.parallel_for(sycl::range<1>(num_pairs), [=](sycl::id<1> idx) {
        // Split the full 64-bit per-element counter across both Philox lanes so
        // the stream period is not exhausted below 2^64 elements, and seed the
        // key schedule with both halves of the 64-bit global seed so that two
        // manual_seed() values differing only above bit 31 produce distinct
        // streams.
        uint64_t ctr = static_cast<uint64_t>(idx[0]);
        uint32_t c0 = static_cast<uint32_t>(ctr & 0xFFFFFFFFu);
        uint32_t c1 = static_cast<uint32_t>(ctr >> 32);
        uint32_t k0 = static_cast<uint32_t>(seed64 & 0xFFFFFFFFu);
        uint32_t k1 = static_cast<uint32_t>(seed64 >> 32);

        // Philox-2x32-10 (Salmon et al., "Parallel Random Numbers", 2011).
        // Each round multiplies the counter lane c0, then mixes the high word
        // with c1 and the key; the key advances by the Weyl constant. Both
        // 64-bit-seed halves participate: k0 mixes into the state directly while
        // k1 (added to the Weyl increment) perturbs the key schedule, ensuring
        // the high seed word affects the output stream.
        constexpr uint32_t PHILOX_M = 0xD256D193u;  // 2x32 multiplier
        constexpr uint32_t PHILOX_W = 0x9E3779B9u;  // Weyl key increment
        for (int round = 0; round < 10; ++round) {
            uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(c0) * PHILOX_M) >> 32);
            uint32_t lo = c0 * PHILOX_M;
            uint32_t new_c0 = hi ^ c1 ^ k0;
            c1 = lo;
            c0 = new_c0;
            k0 += PHILOX_W + k1;
        }

        // Convert two uint32 outputs to uniform [0,1) floats
        constexpr float INV = 2.3283064365386963e-10f; // 1/2^32
        float u1 = (static_cast<float>(c0) + 0.5f) * INV;
        float u2 = (static_cast<float>(c1) + 0.5f) * INV;

        // Clamp to avoid log(0)
        u1 = sycl::fmax(u1, 1e-30f);
        u2 = sycl::fmax(u2, 1e-30f);

        // Box-Muller transform: generate two independent N(0,1) samples
        float r = sycl::sqrt(-2.0f * sycl::log(u1));
        constexpr float TWO_PI = 6.283185307179586f;
        float z0 = r * sycl::cos(TWO_PI * u2);
        float z1 = r * sycl::sin(TWO_PI * u2);

        int64_t base = static_cast<int64_t>(idx[0]) * 2;
        f32_ptr[base] = z0;
        f32_ptr[base + 1] = z1;
    }).wait();

    if (dtype == DType::Float32) {
        float* device_ptr = get_data_ptr<float>(output);
        queue.memcpy(device_ptr, f32_ptr, numel * sizeof(float)).wait();
    }
    else if (dtype == DType::Float64) {
        double* device_ptr = get_data_ptr<double>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = static_cast<double>(f32_ptr[i]);
        }).wait();
    }
    else if (dtype == DType::Float16) {
        sycl::half* device_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = sycl::half(f32_ptr[i]);
        }).wait();
    }
    else if (dtype == DType::BFloat16) {
        uint16_t* device_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            uint32_t bits;
            __builtin_memcpy(&bits, &f32_ptr[i], sizeof(uint32_t));
            device_ptr[i] = static_cast<uint16_t>(bits >> 16);
        }).wait();
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
 * Uses Intel oneMKL VSL when available. Falls back to an on-device Philox 4x32-10 SYCL kernel otherwise.
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

            // Wait so the function-local temp_buffer is not freed while this
            // conversion kernel still reads temp_ptr, and the host sees output.
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
                ptr[i] = sycl::half(temp_ptr[i]);
            }).wait();
        }
        else if (dtype == DType::BFloat16) {
            // Generate Float32 then convert to BFloat16 (truncate upper 16 bits)
            uint16_t* ptr = get_data_ptr<uint16_t>(output);

            Tensor temp_buffer({numel}, DType::Float32, device);
            float* temp_ptr = get_data_ptr<float>(temp_buffer);

            ::oneapi::mkl::rng::philox4x32x10 engine(queue, seed);
            ::oneapi::mkl::rng::uniform<float> distribution(0.0f, 1.0f);
            ::oneapi::mkl::rng::generate(distribution, engine, numel, temp_ptr);

            // Wait so temp_buffer outlives the conversion kernel reading temp_ptr.
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
                // BFloat16: upper 16 bits of float32
                uint32_t bits;
                __builtin_memcpy(&bits, &temp_ptr[i], sizeof(uint32_t));
                ptr[i] = static_cast<uint16_t>(bits >> 16);
            }).wait();
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
    // On-device Philox 4x32-10 counter-based RNG for uniform distribution [0, 1)
    uint64_t seed64 = static_cast<uint64_t>(tenzor::get_global_seed());

    // Generate Float32 on device directly
    Tensor f32_buf({numel}, DType::Float32, device);
    float* f32_ptr = get_data_ptr<float>(f32_buf);

    queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
        // Split the full 64-bit per-element counter across both Philox lanes so
        // the stream period is not exhausted below 2^64 elements, and seed the
        // key schedule with both halves of the 64-bit global seed so distinct
        // 64-bit seeds produce distinct streams. Mirrors the randn fallback.
        uint64_t ctr = static_cast<uint64_t>(idx[0]);
        uint32_t c0 = static_cast<uint32_t>(ctr & 0xFFFFFFFFu);
        uint32_t c1 = static_cast<uint32_t>(ctr >> 32);
        uint32_t k0 = static_cast<uint32_t>(seed64 & 0xFFFFFFFFu);
        uint32_t k1 = static_cast<uint32_t>(seed64 >> 32);

        // Philox-2x32-10 (Salmon et al., "Parallel Random Numbers", 2011).
        // Each round multiplies counter lane c0, mixes the high word with c1 and
        // the key, and advances the key by the Weyl constant perturbed by the
        // high seed word k1 so both seed halves affect the output stream.
        constexpr uint32_t PHILOX_M = 0xD256D193u;  // 2x32 multiplier
        constexpr uint32_t PHILOX_W = 0x9E3779B9u;  // Weyl key increment
        for (int round = 0; round < 10; ++round) {
            uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(c0) * PHILOX_M) >> 32);
            uint32_t lo = c0 * PHILOX_M;
            uint32_t new_c0 = hi ^ c1 ^ k0;
            c1 = lo;
            c0 = new_c0;
            k0 += PHILOX_W + k1;
        }

        // Convert to uniform [0, 1)
        constexpr float INV = 2.3283064365386963e-10f; // 1/2^32
        f32_ptr[idx[0]] = (static_cast<float>(c0) + 0.5f) * INV;
    }).wait();

    if (dtype == DType::Float32) {
        float* device_ptr = get_data_ptr<float>(output);
        queue.memcpy(device_ptr, f32_ptr, numel * sizeof(float)).wait();
    }
    else if (dtype == DType::Float64) {
        double* device_ptr = get_data_ptr<double>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = static_cast<double>(f32_ptr[i]);
        }).wait();
    }
    else if (dtype == DType::Float16) {
        sycl::half* device_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            device_ptr[i] = sycl::half(f32_ptr[i]);
        }).wait();
    }
    else if (dtype == DType::BFloat16) {
        uint16_t* device_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            uint32_t bits;
            __builtin_memcpy(&bits, &f32_ptr[i], sizeof(uint32_t));
            device_ptr[i] = static_cast<uint16_t>(bits >> 16);
        }).wait();
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
    else if (dtype == DType::Int8 || dtype == DType::Int16 || dtype == DType::UInt8 ||
             dtype == DType::UInt16 || dtype == DType::UInt32 || dtype == DType::UInt64) {
        const double s = start, st = step;
        auto run = [&]<typename T>() {
            T* ptr = get_data_ptr<T>(output);
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
                ptr[i] = static_cast<T>(s + static_cast<double>(i[0]) * st);
            });
        };
        switch (dtype) {
            case DType::Int8:   run.template operator()<int8_t>();   break;
            case DType::Int16:  run.template operator()<int16_t>();  break;
            case DType::UInt8:  run.template operator()<uint8_t>();  break;
            case DType::UInt16: run.template operator()<uint16_t>(); break;
            case DType::UInt32: run.template operator()<uint32_t>(); break;
            case DType::UInt64: run.template operator()<uint64_t>(); break;
            default: break;
        }
    }
    else if (dtype == DType::Complex64) {
        float* ptr = get_data_ptr<float>(output);
        const float s = static_cast<float>(start), st = static_cast<float>(step);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            ptr[2 * i[0]] = s + static_cast<float>(i[0]) * st;
            ptr[2 * i[0] + 1] = 0.0f;
        });
    }
    else if (dtype == DType::Complex128) {
        double* ptr = get_data_ptr<double>(output);
        const double s = start, st = step;
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
            ptr[2 * i[0]] = s + static_cast<double>(i[0]) * st;
            ptr[2 * i[0] + 1] = 0.0;
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for arange kernel");
    }

    // Drain the enqueued fill kernel before the host reads the USM-shared output.
    queue.wait_and_throw();
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
        } else {
            auto set_one = [&](auto* ptr) {
                using T = std::remove_pointer_t<decltype(ptr)>;
                T val = static_cast<T>(start);
                queue.memcpy(ptr, &val, sizeof(T)).wait();
            };
            switch (dtype) {
                case DType::Int8:   set_one(get_data_ptr<int8_t>(output)); break;
                case DType::Int16:  set_one(get_data_ptr<int16_t>(output)); break;
                case DType::Int32:  set_one(get_data_ptr<int32_t>(output)); break;
                case DType::Int64:  set_one(get_data_ptr<int64_t>(output)); break;
                case DType::UInt8:  set_one(get_data_ptr<uint8_t>(output)); break;
                case DType::UInt16: set_one(get_data_ptr<uint16_t>(output)); break;
                case DType::UInt32: set_one(get_data_ptr<uint32_t>(output)); break;
                case DType::UInt64: set_one(get_data_ptr<uint64_t>(output)); break;
                default:
                    throw std::runtime_error("Unsupported dtype for linspace kernel");
            }
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
    else if (dtype == DType::Int8 || dtype == DType::Int16 || dtype == DType::Int32 ||
             dtype == DType::Int64 || dtype == DType::UInt8 || dtype == DType::UInt16 ||
             dtype == DType::UInt32 || dtype == DType::UInt64) {
        // Integer linspace: compute the value in double and truncate toward zero,
        // forcing the last element to exactly `end` — matches the CPU backend.
        const double s = start, e = end, st = step_size;
        const int64_t n = steps;
        auto run = [&](auto* ptr) {
            using T = std::remove_pointer_t<decltype(ptr)>;
            queue.parallel_for(sycl::range<1>(steps), [=](sycl::id<1> i) {
                double v = (static_cast<int64_t>(i[0]) == n - 1)
                               ? e : (s + static_cast<double>(i[0]) * st);
                ptr[i] = static_cast<T>(v);
            });
        };
        switch (dtype) {
            case DType::Int8:   run(get_data_ptr<int8_t>(output)); break;
            case DType::Int16:  run(get_data_ptr<int16_t>(output)); break;
            case DType::Int32:  run(get_data_ptr<int32_t>(output)); break;
            case DType::Int64:  run(get_data_ptr<int64_t>(output)); break;
            case DType::UInt8:  run(get_data_ptr<uint8_t>(output)); break;
            case DType::UInt16: run(get_data_ptr<uint16_t>(output)); break;
            case DType::UInt32: run(get_data_ptr<uint32_t>(output)); break;
            case DType::UInt64: run(get_data_ptr<uint64_t>(output)); break;
            default: break;
        }
    }
    else if (dtype == DType::Complex64 || dtype == DType::Complex128) {
        // Complex linspace: real part ramps start->end, imaginary part is 0.
        const double s = start, e = end, st = step_size;
        const int64_t n = steps;
        if (dtype == DType::Complex64) {
            float* ptr = reinterpret_cast<float*>(output.data_ptr());
            queue.parallel_for(sycl::range<1>(steps), [=](sycl::id<1> i) {
                double v = (static_cast<int64_t>(i[0]) == n - 1)
                               ? e : (s + static_cast<double>(i[0]) * st);
                ptr[i[0] * 2]     = static_cast<float>(v);
                ptr[i[0] * 2 + 1] = 0.0f;
            });
        } else {
            double* ptr = reinterpret_cast<double*>(output.data_ptr());
            queue.parallel_for(sycl::range<1>(steps), [=](sycl::id<1> i) {
                double v = (static_cast<int64_t>(i[0]) == n - 1)
                               ? e : (s + static_cast<double>(i[0]) * st);
                ptr[i[0] * 2]     = v;
                ptr[i[0] * 2 + 1] = 0.0;
            });
        }
    }
    else {
        throw std::runtime_error("Unsupported dtype for linspace kernel");
    }

    // Drain the enqueued fill kernel before the host reads the USM-shared output.
    // (The steps==1 fast path already waits via memcpy().wait() and returns early.)
    queue.wait_and_throw();
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
    else if (dtype == DType::Int16) {
        int16_t* ptr = get_data_ptr<int16_t>(output);
        queue.parallel_for(sycl::range<1>(diag_len), [=](sycl::id<1> i) {
            ptr[i[0] * m + i[0]] = 1;
        });
    }
    else if (dtype == DType::Int8 || dtype == DType::UInt8 || dtype == DType::Bool) {
        // 1-byte types (Bool stores 1 for true). Write the diagonal as bytes.
        uint8_t* ptr = get_data_ptr<uint8_t>(output);
        queue.parallel_for(sycl::range<1>(diag_len), [=](sycl::id<1> i) {
            ptr[i[0] * m + i[0]] = uint8_t(1);
        });
    }
    else if (dtype == DType::UInt16) {
        uint16_t* ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for(sycl::range<1>(diag_len), [=](sycl::id<1> i) {
            ptr[i[0] * m + i[0]] = uint16_t(1);
        });
    }
    else if (dtype == DType::UInt32) {
        uint32_t* ptr = get_data_ptr<uint32_t>(output);
        queue.parallel_for(sycl::range<1>(diag_len), [=](sycl::id<1> i) {
            ptr[i[0] * m + i[0]] = uint32_t(1);
        });
    }
    else if (dtype == DType::UInt64) {
        uint64_t* ptr = get_data_ptr<uint64_t>(output);
        queue.parallel_for(sycl::range<1>(diag_len), [=](sycl::id<1> i) {
            ptr[i[0] * m + i[0]] = uint64_t(1);
        });
    }
    else if (dtype == DType::Complex64) {
        // Interleaved (real, imag) float pairs: diagonal = 1 + 0i.
        float* ptr = reinterpret_cast<float*>(output.data_ptr());
        queue.parallel_for(sycl::range<1>(diag_len), [=](sycl::id<1> i) {
            ptr[(i[0] * m + i[0]) * 2]     = 1.0f;
            ptr[(i[0] * m + i[0]) * 2 + 1] = 0.0f;
        });
    }
    else if (dtype == DType::Complex128) {
        double* ptr = reinterpret_cast<double*>(output.data_ptr());
        queue.parallel_for(sycl::range<1>(diag_len), [=](sycl::id<1> i) {
            ptr[(i[0] * m + i[0]) * 2]     = 1.0;
            ptr[(i[0] * m + i[0]) * 2 + 1] = 0.0;
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for eye kernel");
    }

    // Drain the memset + diagonal fill before the host reads the USM-shared output.
    queue.wait_and_throw();
    return output;
}

// ============================================================================
// Randint - Random integers in [low, high)
// ============================================================================

struct RandintKernelInt32 {};
struct RandintKernelInt64 {};

auto randint_kernel(int64_t low, int64_t high, const std::vector<int64_t>& shape,
                    DType dtype, Device device, sycl::queue& queue) -> Tensor {
    if (high <= low) {
        throw std::invalid_argument("randint: high must be greater than low");
    }

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

    // Drain the enqueued fill kernel before the host reads the USM-shared output.
    queue.wait_and_throw();
    return output;
}

} // namespace oneapi
} // namespace tenzor
