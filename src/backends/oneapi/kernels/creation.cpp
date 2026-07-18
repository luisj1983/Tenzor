#include "tenzor/core/tensor.hpp"
#include "oneapi_kernel_utils.hpp"
#include "tenzor/ops/creation.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>
#include <random>
#include <chrono>
#include <limits>

#ifdef TENZOR_HAS_ONEMKL
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng.hpp>
// Note: Use ::oneapi::mkl to avoid namespace conflict with tenzor::oneapi
#endif

namespace tenzor {
namespace oneapi {



// ============================================================================
// F042: on-device Philox 4x32-10 — a bit-for-bit port of the CPU
// tenzor::cpu::philox implementation (src/backends/cpu/kernels/philox.hpp)
// and the CUDA/ROCm device ports, keyed by (seed, element index). Written
// inline in each kernel lambda (matching this file's existing SYCL
// single-source conventions) rather than as free device functions.
//
// This replaces two previously-broken paths:
//   - oneMKL's ::oneapi::mkl::rng::philox4x32x10 engine + gaussian/uniform
//     distribution objects: despite the matching "philox4x32x10" name, MKL's
//     internal counter/key layout and its Gaussian transform (Box-Muller or
//     ziggurat, unspecified) are an opaque implementation detail — not
//     bit-compatible with CPU's hand-written Philox + Box-Muller.
//   - the non-MKL fallback: a structurally different Philox-2x32-10 (only
//     two 32-bit counter/key lanes, multiplier 0xD256D193) that also wrote
//     TWO Box-Muller outputs (cos and sin) per work-item index, rather than
//     one N(0,1) sample per element index drawn from a full 4-word Philox
//     block (out[0], out[1] -> cos only) the way CPU's philox_normal_f32
//     does. Both paths produced statistically-plausible but numerically
//     WRONG values relative to CPU for the same manual_seed().
// ============================================================================

/**
 * @brief Generate random numbers from a standard normal distribution (Gaussian with mean=0, stddev=1)
 *
 * @param shape Shape of the output tensor
 * @param dtype Data type (Float32, Float64, Float16, or BFloat16)
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

    uint64_t seed64 = static_cast<uint32_t>(tenzor::get_global_seed());

    // Generate Float32 on device (one N(0,1) sample per element index), then
    // convert if needed.
    Tensor f32_buf({numel}, DType::Float32, device);
    float* f32_ptr = get_data_ptr<float>(f32_buf);

    queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
        uint32_t ctr[4] = {
            static_cast<uint32_t>(static_cast<uint64_t>(idx[0]) & 0xFFFFFFFFu),
            static_cast<uint32_t>((static_cast<uint64_t>(idx[0]) >> 32) & 0xFFFFFFFFu),
            0u, 0u
        };
        uint32_t key[2] = {
            static_cast<uint32_t>(seed64 & 0xFFFFFFFFu),
            static_cast<uint32_t>((seed64 >> 32) & 0xFFFFFFFFu)
        };

        // Philox 4x32-10 (Salmon et al., "Parallel Random Numbers", 2011) —
        // same round function/constants as CPU's philox.hpp.
        constexpr uint64_t M0 = 0xD2511F53ULL;
        constexpr uint64_t M1 = 0xCD9E8D57ULL;
        constexpr uint32_t W0 = 0x9E3779B9u;
        constexpr uint32_t W1 = 0xBB67AE85u;
        for (int round = 0; round < 10; ++round) {
            uint64_t prod0 = M0 * static_cast<uint64_t>(ctr[0]);
            uint64_t prod1 = M1 * static_cast<uint64_t>(ctr[2]);
            uint32_t hi0 = static_cast<uint32_t>(prod0 >> 32);
            uint32_t lo0 = static_cast<uint32_t>(prod0);
            uint32_t hi1 = static_cast<uint32_t>(prod1 >> 32);
            uint32_t lo1 = static_cast<uint32_t>(prod1);
            uint32_t new0 = hi1 ^ ctr[1] ^ key[0];
            uint32_t new2 = hi0 ^ ctr[3] ^ key[1];
            ctr[0] = new0; ctr[1] = lo1; ctr[2] = new2; ctr[3] = lo0;
            key[0] += W0; key[1] += W1;
        }

        // Box-Muller on the first two output words (out[0], out[1]), matching
        // CPU's philox_normal_f32 exactly: top-24-bits -> uniform, cos-only.
        float u1 = static_cast<float>(ctr[0] >> 8) * (1.0f / 16777216.0f);
        float u2 = static_cast<float>(ctr[1] >> 8) * (1.0f / 16777216.0f);
        u1 = sycl::fmax(u1, 1e-37f);
        constexpr float TWO_PI = 6.28318530718f;
        f32_ptr[idx[0]] = sycl::sqrt(-2.0f * sycl::log(u1)) * sycl::cos(TWO_PI * u2);
    }).wait();

    if (dtype == DType::Float32) {
        float* device_ptr = get_data_ptr<float>(output);
        queue.memcpy(device_ptr, f32_ptr, numel * sizeof(float)).wait();
    }
    else if (dtype == DType::Float64) {
        // Float64 needs its own full-precision Philox pass (paired with
        // CPU's philox_normal_f64, which draws u1/u2 from a 53-bit mantissa
        // built out of two 32-bit words each, i.e. all four output words),
        // not a narrow-then-widen of the Float32 pass above.
        double* device_ptr = get_data_ptr<double>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            uint32_t ctr[4] = {
                static_cast<uint32_t>(static_cast<uint64_t>(idx[0]) & 0xFFFFFFFFu),
                static_cast<uint32_t>((static_cast<uint64_t>(idx[0]) >> 32) & 0xFFFFFFFFu),
                0u, 0u
            };
            uint32_t key[2] = {
                static_cast<uint32_t>(seed64 & 0xFFFFFFFFu),
                static_cast<uint32_t>((seed64 >> 32) & 0xFFFFFFFFu)
            };
            constexpr uint64_t M0 = 0xD2511F53ULL;
            constexpr uint64_t M1 = 0xCD9E8D57ULL;
            constexpr uint32_t W0 = 0x9E3779B9u;
            constexpr uint32_t W1 = 0xBB67AE85u;
            for (int round = 0; round < 10; ++round) {
                uint64_t prod0 = M0 * static_cast<uint64_t>(ctr[0]);
                uint64_t prod1 = M1 * static_cast<uint64_t>(ctr[2]);
                uint32_t hi0 = static_cast<uint32_t>(prod0 >> 32);
                uint32_t lo0 = static_cast<uint32_t>(prod0);
                uint32_t hi1 = static_cast<uint32_t>(prod1 >> 32);
                uint32_t lo1 = static_cast<uint32_t>(prod1);
                uint32_t new0 = hi1 ^ ctr[1] ^ key[0];
                uint32_t new2 = hi0 ^ ctr[3] ^ key[1];
                ctr[0] = new0; ctr[1] = lo1; ctr[2] = new2; ctr[3] = lo0;
                key[0] += W0; key[1] += W1;
            }
            uint64_t bits1 = (static_cast<uint64_t>(ctr[0]) << 21) | (static_cast<uint64_t>(ctr[1]) >> 11);
            uint64_t bits2 = (static_cast<uint64_t>(ctr[2]) << 21) | (static_cast<uint64_t>(ctr[3]) >> 11);
            double u1 = static_cast<double>(bits1) * (1.0 / 9007199254740992.0);
            double u2 = static_cast<double>(bits2) * (1.0 / 9007199254740992.0);
            u1 = sycl::fmax(u1, 1e-300);
            constexpr double TWO_PI = 6.283185307179586;
            device_ptr[idx[0]] = sycl::sqrt(-2.0 * sycl::log(u1)) * sycl::cos(TWO_PI * u2);
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
        throw std::runtime_error("Unsupported dtype for randn");
    }

    return output;
}

/**
 * @brief Generate random numbers from a uniform distribution [0, 1)
 *
 * F042: on-device Philox 4x32-10, bit-for-bit matching CPU's philox_uniform_f32/f64
 * (src/backends/cpu/kernels/philox.hpp). See the comment above randn_kernel for why
 * oneMKL's rng engine and the old Philox-2x32 fallback were both replaced.
 *
 * @param shape Shape of the output tensor
 * @param dtype Data type (Float32, Float64, Float16, or BFloat16)
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

    uint64_t seed64 = static_cast<uint32_t>(tenzor::get_global_seed());

    // Generate Float32 on device directly (one Philox block's out[0] per
    // element index, top 24 bits -> mantissa — matches CPU's
    // philox_uniform_f32 exactly).
    Tensor f32_buf({numel}, DType::Float32, device);
    float* f32_ptr = get_data_ptr<float>(f32_buf);

    queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
        uint32_t ctr[4] = {
            static_cast<uint32_t>(static_cast<uint64_t>(idx[0]) & 0xFFFFFFFFu),
            static_cast<uint32_t>((static_cast<uint64_t>(idx[0]) >> 32) & 0xFFFFFFFFu),
            0u, 0u
        };
        uint32_t key[2] = {
            static_cast<uint32_t>(seed64 & 0xFFFFFFFFu),
            static_cast<uint32_t>((seed64 >> 32) & 0xFFFFFFFFu)
        };
        constexpr uint64_t M0 = 0xD2511F53ULL;
        constexpr uint64_t M1 = 0xCD9E8D57ULL;
        constexpr uint32_t W0 = 0x9E3779B9u;
        constexpr uint32_t W1 = 0xBB67AE85u;
        for (int round = 0; round < 10; ++round) {
            uint64_t prod0 = M0 * static_cast<uint64_t>(ctr[0]);
            uint64_t prod1 = M1 * static_cast<uint64_t>(ctr[2]);
            uint32_t hi0 = static_cast<uint32_t>(prod0 >> 32);
            uint32_t lo0 = static_cast<uint32_t>(prod0);
            uint32_t hi1 = static_cast<uint32_t>(prod1 >> 32);
            uint32_t lo1 = static_cast<uint32_t>(prod1);
            uint32_t new0 = hi1 ^ ctr[1] ^ key[0];
            uint32_t new2 = hi0 ^ ctr[3] ^ key[1];
            ctr[0] = new0; ctr[1] = lo1; ctr[2] = new2; ctr[3] = lo0;
            key[0] += W0; key[1] += W1;
        }
        f32_ptr[idx[0]] = static_cast<float>(ctr[0] >> 8) * (1.0f / 16777216.0f);
    }).wait();

    if (dtype == DType::Float32) {
        float* device_ptr = get_data_ptr<float>(output);
        queue.memcpy(device_ptr, f32_ptr, numel * sizeof(float)).wait();
    }
    else if (dtype == DType::Float64) {
        // Float64 needs its own full-precision Philox pass (53-bit mantissa
        // from out[0]/out[1]), matching CPU's philox_uniform_f64 exactly —
        // not a narrow-then-widen of the Float32 pass above.
        double* device_ptr = get_data_ptr<double>(output);
        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            uint32_t ctr[4] = {
                static_cast<uint32_t>(static_cast<uint64_t>(idx[0]) & 0xFFFFFFFFu),
                static_cast<uint32_t>((static_cast<uint64_t>(idx[0]) >> 32) & 0xFFFFFFFFu),
                0u, 0u
            };
            uint32_t key[2] = {
                static_cast<uint32_t>(seed64 & 0xFFFFFFFFu),
                static_cast<uint32_t>((seed64 >> 32) & 0xFFFFFFFFu)
            };
            constexpr uint64_t M0 = 0xD2511F53ULL;
            constexpr uint64_t M1 = 0xCD9E8D57ULL;
            constexpr uint32_t W0 = 0x9E3779B9u;
            constexpr uint32_t W1 = 0xBB67AE85u;
            for (int round = 0; round < 10; ++round) {
                uint64_t prod0 = M0 * static_cast<uint64_t>(ctr[0]);
                uint64_t prod1 = M1 * static_cast<uint64_t>(ctr[2]);
                uint32_t hi0 = static_cast<uint32_t>(prod0 >> 32);
                uint32_t lo0 = static_cast<uint32_t>(prod0);
                uint32_t hi1 = static_cast<uint32_t>(prod1 >> 32);
                uint32_t lo1 = static_cast<uint32_t>(prod1);
                uint32_t new0 = hi1 ^ ctr[1] ^ key[0];
                uint32_t new2 = hi0 ^ ctr[3] ^ key[1];
                ctr[0] = new0; ctr[1] = lo1; ctr[2] = new2; ctr[3] = lo0;
                key[0] += W0; key[1] += W1;
            }
            uint64_t bits = (static_cast<uint64_t>(ctr[0]) << 21) | (static_cast<uint64_t>(ctr[1]) >> 11);
            device_ptr[idx[0]] = static_cast<double>(bits) * (1.0 / 9007199254740992.0);
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
        throw std::runtime_error("Unsupported dtype for rand");
    }

    return output;
}

/**
 * @brief Generate a 1D tensor of evenly spaced values in [start, end) with given step
 */
auto arange_kernel(double start, double end, double step, DType dtype, Device device, sycl::queue& queue) -> Tensor {
    if (step == 0.0) {
        throw std::runtime_error("arange: step must be non-zero");
    }
    // Length matches PyTorch's torch.arange: ceil((end - start) / step), but a
    // naive ceil over a floating-point ratio rounds an exact-integer quotient
    // (e.g. (1.0 - 0.0) / 0.1 = 9.999999999999998 or 10.000000000000002) up by
    // one, yielding a spurious final element whose value can also drift past
    // `end`. Snap a ratio that is integral within a relative epsilon to that
    // integer before applying ceil, so exact ranges produce the exact count.
    // Ported from CPU's arange_kernel (src/backends/cpu/kernels/creation.cpp:467-481).
    const double ratio = (end - start) / step;
    const double rounded = std::round(ratio);
    double count_d;
    if (std::abs(ratio - rounded) < std::numeric_limits<double>::epsilon() *
                                    std::max(1.0, std::abs(ratio)) * 4.0) {
        count_d = rounded;  // exact integer ratio: half-open interval => `rounded` elements
    } else {
        count_d = std::ceil(ratio);
    }
    int64_t numel = static_cast<int64_t>(count_d);
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
//
// F042: on-device Philox 4x32-10 uniform double in [0,1), scaled to
// [low, high) exactly as CPU randint (low + (int64_t)(philox_uniform_f64 *
// range)) — bit-identical to CPU. The previous implementation used a
// SplitMix64-style hash (`x ^= x >> 30; x *= 0x...`) with `x % range`
// rejection-free modulo mapping — a completely different RNG family and
// range-mapping scheme from CPU's Philox + multiply-by-range-then-truncate,
// producing plausible-looking but numerically wrong integers relative to CPU.
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

    uint64_t seed64 = static_cast<uint32_t>(tenzor::get_global_seed());
    const double range = static_cast<double>(high - low);

    if (dtype == DType::Int32) {
        int32_t* ptr = get_data_ptr<int32_t>(output);
        queue.parallel_for<RandintKernelInt32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            uint32_t ctr[4] = {
                static_cast<uint32_t>(static_cast<uint64_t>(idx[0]) & 0xFFFFFFFFu),
                static_cast<uint32_t>((static_cast<uint64_t>(idx[0]) >> 32) & 0xFFFFFFFFu),
                0u, 0u
            };
            uint32_t key[2] = {
                static_cast<uint32_t>(seed64 & 0xFFFFFFFFu),
                static_cast<uint32_t>((seed64 >> 32) & 0xFFFFFFFFu)
            };
            constexpr uint64_t M0 = 0xD2511F53ULL;
            constexpr uint64_t M1 = 0xCD9E8D57ULL;
            constexpr uint32_t W0 = 0x9E3779B9u;
            constexpr uint32_t W1 = 0xBB67AE85u;
            for (int round = 0; round < 10; ++round) {
                uint64_t prod0 = M0 * static_cast<uint64_t>(ctr[0]);
                uint64_t prod1 = M1 * static_cast<uint64_t>(ctr[2]);
                uint32_t hi0 = static_cast<uint32_t>(prod0 >> 32);
                uint32_t lo0 = static_cast<uint32_t>(prod0);
                uint32_t hi1 = static_cast<uint32_t>(prod1 >> 32);
                uint32_t lo1 = static_cast<uint32_t>(prod1);
                uint32_t new0 = hi1 ^ ctr[1] ^ key[0];
                uint32_t new2 = hi0 ^ ctr[3] ^ key[1];
                ctr[0] = new0; ctr[1] = lo1; ctr[2] = new2; ctr[3] = lo0;
                key[0] += W0; key[1] += W1;
            }
            uint64_t bits = (static_cast<uint64_t>(ctr[0]) << 21) | (static_cast<uint64_t>(ctr[1]) >> 11);
            double r = static_cast<double>(bits) * (1.0 / 9007199254740992.0);
            int64_t val = low + static_cast<int64_t>(r * range);
            if (val >= high) val = high - 1;
            ptr[idx] = static_cast<int32_t>(val);
        });
    } else if (dtype == DType::Int64) {
        int64_t* ptr = get_data_ptr<int64_t>(output);
        queue.parallel_for<RandintKernelInt64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            uint32_t ctr[4] = {
                static_cast<uint32_t>(static_cast<uint64_t>(idx[0]) & 0xFFFFFFFFu),
                static_cast<uint32_t>((static_cast<uint64_t>(idx[0]) >> 32) & 0xFFFFFFFFu),
                0u, 0u
            };
            uint32_t key[2] = {
                static_cast<uint32_t>(seed64 & 0xFFFFFFFFu),
                static_cast<uint32_t>((seed64 >> 32) & 0xFFFFFFFFu)
            };
            constexpr uint64_t M0 = 0xD2511F53ULL;
            constexpr uint64_t M1 = 0xCD9E8D57ULL;
            constexpr uint32_t W0 = 0x9E3779B9u;
            constexpr uint32_t W1 = 0xBB67AE85u;
            for (int round = 0; round < 10; ++round) {
                uint64_t prod0 = M0 * static_cast<uint64_t>(ctr[0]);
                uint64_t prod1 = M1 * static_cast<uint64_t>(ctr[2]);
                uint32_t hi0 = static_cast<uint32_t>(prod0 >> 32);
                uint32_t lo0 = static_cast<uint32_t>(prod0);
                uint32_t hi1 = static_cast<uint32_t>(prod1 >> 32);
                uint32_t lo1 = static_cast<uint32_t>(prod1);
                uint32_t new0 = hi1 ^ ctr[1] ^ key[0];
                uint32_t new2 = hi0 ^ ctr[3] ^ key[1];
                ctr[0] = new0; ctr[1] = lo1; ctr[2] = new2; ctr[3] = lo0;
                key[0] += W0; key[1] += W1;
            }
            uint64_t bits = (static_cast<uint64_t>(ctr[0]) << 21) | (static_cast<uint64_t>(ctr[1]) >> 11);
            double r = static_cast<double>(bits) * (1.0 / 9007199254740992.0);
            int64_t val = low + static_cast<int64_t>(r * range);
            if (val >= high) val = high - 1;
            ptr[idx] = val;
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
