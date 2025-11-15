#include <iostream>
#include <cstdlib>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/embedding.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Testing Embedding layer with Float16..." << std::endl;

    // Create embedding layer similar to ALBERT
    // vocab_size=30000, embedding_dim=768 (same as ALBERT)
    auto embedding = Embedding(30000, 768);

    // Convert embedding parameters to Float16
    embedding.to(DType::Float16);

    std::cout << "Created Embedding layer with Float16 parameters" << std::endl;

    // Create input indices (batch_size=8, seq_len=128, same as ALBERT test)
    // Create tensor with integer type directly
    auto indices_data = zeros({8, 128}, DType::Int64);
    auto indices_ptr = indices_data.data<int64_t>();

    // Fill with random valid indices in range [0, 29999]
    std::srand(42);  // Fixed seed for reproducibility
    for (int64_t i = 0; i < 8 * 128; ++i) {
        indices_ptr[i] = std::rand() % 30000;
    }

    Variable indices(indices_data, false);  // indices don't need grad

    std::cout << "Created input indices tensor: shape [8, 128]" << std::endl;

    // Run embedding lookup multiple times
    for (int i = 0; i < 100; ++i) {
        auto embedded = embedding.forward(indices);

        if (i % 20 == 0) {
            std::cout << "Iteration " << i << ": embedded.shape = ["
                      << embedded.shape()[0] << ", "
                      << embedded.shape()[1] << ", "
                      << embedded.shape()[2] << "]"
                      << ", dtype=" << (embedded.dtype() == DType::Float16 ? "Float16" : "other")
                      << std::endl;
        }

        // Verify shape and dtype
        if (embedded.shape()[0] != 8 || embedded.shape()[1] != 128 || embedded.shape()[2] != 768) {
            std::cerr << "ERROR: Unexpected output shape!" << std::endl;
            return 1;
        }

        // Create new random indices for next iteration
        if (i < 99) {
            indices_data = zeros({8, 128}, DType::Int64);
            indices_ptr = indices_data.data<int64_t>();
            for (int64_t j = 0; j < 8 * 128; ++j) {
                indices_ptr[j] = std::rand() % 30000;
            }
            indices = Variable(indices_data, false);
        }
    }

    std::cout << "Embedding test completed successfully!" << std::endl;
    return 0;
}
