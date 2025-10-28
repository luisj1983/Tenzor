#include <iostream>
#include <sycl/sycl.hpp>

int main() {
    std::cout << "Direct SYCL Reduction Test\n";
    std::cout << "==========================\n\n";

    try {
        sycl::queue queue{sycl::default_selector_v};
        std::cout << "Running on: " << queue.get_device().get_info<sycl::info::device::name>() << "\n\n";

        // Test 1: Simple sum with malloc_shared
        {
            std::cout << "Test 1: Sum using malloc_shared\n";
            const int N = 4;

            // Allocate shared memory for input and output
            float* input = sycl::malloc_shared<float>(N, queue);
            float* output = sycl::malloc_shared<float>(1, queue);

            // Initialize input to all 1.0
            for (int i = 0; i < N; i++) {
                input[i] = 1.0f;
            }

            output[0] = 0.0f;

            std::cout << "Input values: ";
            for (int i = 0; i < N; i++) {
                std::cout << input[i] << " ";
            }
            std::cout << "\n";

            // Perform reduction
            queue.parallel_for(sycl::range<1>(N), sycl::reduction(output, sycl::plus<float>()),
                [=](sycl::id<1> idx, auto& sum) {
                    sum += input[idx];
                }).wait();

            std::cout << "Sum result: " << output[0] << " (expected: 4.0)\n";
            std::cout << "Test 1: " << (std::abs(output[0] - 4.0f) < 0.001f ? "PASS" : "FAIL") << "\n\n";

            sycl::free(input, queue);
            sycl::free(output, queue);
        }

        // Test 2: Sum with value 3.0
        {
            std::cout << "Test 2: Sum with value 3.0\n";
            const int N = 6;

            float* input = sycl::malloc_shared<float>(N, queue);
            float* output = sycl::malloc_shared<float>(1, queue);

            for (int i = 0; i < N; i++) {
                input[i] = 3.0f;
            }

            output[0] = 0.0f;

            std::cout << "Input values: ";
            for (int i = 0; i < N; i++) {
                std::cout << input[i] << " ";
            }
            std::cout << "\n";

            queue.parallel_for(sycl::range<1>(N), sycl::reduction(output, sycl::plus<float>()),
                [=](sycl::id<1> idx, auto& sum) {
                    sum += input[idx];
                }).wait();

            std::cout << "Sum result: " << output[0] << " (expected: 18.0)\n";
            std::cout << "Test 2: " << (std::abs(output[0] - 18.0f) < 0.001f ? "PASS" : "FAIL") << "\n\n";

            sycl::free(input, queue);
            sycl::free(output, queue);
        }

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception: " << e.what() << "\n";
        return 1;
    }

    std::cout << "All tests complete\n";
    return 0;
}
