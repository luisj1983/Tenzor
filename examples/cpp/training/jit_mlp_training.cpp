/**
 * @file jit_mlp_training.cpp
 * @brief JIT-Compiled MLP Training and Inference Demo
 *
 * This example demonstrates:
 * - Training a small MLP (Linear -> ReLU -> Linear) eagerly via SGD
 * - JIT-compiling the trained model's inference forward pass
 * - Verifying the compiled output matches eager execution
 *
 * R2-02: this is the first example wired into the "examples-as-tests"
 * regression suite (tests/examples/) that actually exercises the JIT
 * subsystem end-to-end (trace -> compile -> execute -> verify), rather than
 * just eager/autograd training.
 */

#include <iostream>

#include "tenzor/tenzor.hpp"
#include "jit_mlp_training_runner.hpp"

using namespace tenzor;

int main(int argc, char* argv[]) {
    tenzor::initialize();

    Device device = Device::cpu();
    if (argc > 1) {
        std::string backend = argv[1];
        if (backend == "cuda") device = Device::cuda();
        else if (backend == "rocm") device = Device::rocm();
        else if (backend == "vulkan") device = Device::vulkan();
    }

    std::cout << "======================================================\n";
    std::cout << "   JIT-Compiled MLP Training and Inference\n";
    std::cout << "   Backend: " << device.to_string() << "\n";
    std::cout << "======================================================\n";
    std::cout << "\nComponents demonstrated:\n";
    std::cout << "  Layers: Linear, ReLU\n";
    std::cout << "  Loss: MSE (mean squared error)\n";
    std::cout << "  Optimizer: manual SGD\n";
    std::cout << "  JIT: trace -> compile -> execute -> verify vs eager\n\n";

    try {
        double init_loss = 0.0, final_loss = 0.0, jit_diff = 0.0;
        auto rc = tenzor::examples::jit_mlp::run_jit_mlp_training(
            /*epochs=*/200, &init_loss, &final_loss, &jit_diff, device,
            /*verbose=*/true);

        std::cout << "\nInitial loss: " << init_loss << "\n";
        std::cout << "Final loss:   " << final_loss << "\n";
        std::cout << "JIT vs eager max abs diff: " << jit_diff << "\n";
        std::cout << "Return code: " << rc << "\n";

        if (rc != 0) {
            std::cerr << "Example reported failure.\n";
            return 1;
        }
        if (!(final_loss < init_loss)) {
            std::cerr << "Loss did not decrease over training.\n";
            return 1;
        }

        std::cout << "\n======================================================\n";
        std::cout << "   JIT MLP example completed successfully!\n";
        std::cout << "======================================================\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
