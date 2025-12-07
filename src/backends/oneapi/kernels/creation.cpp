#include "tenzor/core/tensor.hpp"
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

    // Generate seed based on current time for thread-safety
    auto now = std::chrono::high_resolution_clock::now();
    auto seed = static_cast<uint64_t>(now.time_since_epoch().count());

    try {
        if (dtype == DType::Float32) {
            float* ptr = get_data_ptr<float>(output);

            // Create Philox4x32x10 engine (good quality, high performance)
            ::oneapi::mkl::rng::philox4x32x10 engine(queue, seed);

            // Generate Gaussian distribution with mean=0.0, stddev=1.0
            ::oneapi::mkl::rng::gaussian<float> distribution(0.0f, 1.0f);

            // Generate random numbers directly into device memory
            ::oneapi::mkl::rng::generate(distribution, engine, numel, ptr);

            queue.wait();
        }
        else if (dtype == DType::Float64) {
            double* ptr = get_data_ptr<double>(output);

            // Create Philox4x32x10 engine
            ::oneapi::mkl::rng::philox4x32x10 engine(queue, seed);

            // Generate Gaussian distribution with mean=0.0, stddev=1.0
            ::oneapi::mkl::rng::gaussian<double> distribution(0.0, 1.0);

            // Generate random numbers directly into device memory
            ::oneapi::mkl::rng::generate(distribution, engine, numel, ptr);

            queue.wait();
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

            queue.wait();

            // Convert from float32 to float16
            queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> i) {
                ptr[i] = sycl::half(temp_ptr[i]);
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
    // Fallback: Generate on host using std::normal_distribution and copy to device
    // This is slower but works without oneMKL

    // Use time-based seed for different results on each call
    auto now = std::chrono::high_resolution_clock::now();
    auto seed = static_cast<unsigned int>(now.time_since_epoch().count());

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

    auto now = std::chrono::high_resolution_clock::now();
    auto seed = static_cast<uint64_t>(now.time_since_epoch().count());

    try {
        if (dtype == DType::Float32) {
            float* ptr = get_data_ptr<float>(output);

            ::oneapi::mkl::rng::philox4x32x10 engine(queue, seed);

            // Generate uniform distribution [0, 1)
            ::oneapi::mkl::rng::uniform<float> distribution(0.0f, 1.0f);

            ::oneapi::mkl::rng::generate(distribution, engine, numel, ptr);

            queue.wait();
        }
        else if (dtype == DType::Float64) {
            double* ptr = get_data_ptr<double>(output);

            ::oneapi::mkl::rng::philox4x32x10 engine(queue, seed);

            ::oneapi::mkl::rng::uniform<double> distribution(0.0, 1.0);

            ::oneapi::mkl::rng::generate(distribution, engine, numel, ptr);

            queue.wait();
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

    auto now = std::chrono::high_resolution_clock::now();
    auto seed = static_cast<unsigned int>(now.time_since_epoch().count());

    std::mt19937 gen(seed);
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
    else {
        throw std::runtime_error("Unsupported dtype for rand (fallback path)");
    }
#endif

    return output;
}

} // namespace oneapi
} // namespace tenzor
