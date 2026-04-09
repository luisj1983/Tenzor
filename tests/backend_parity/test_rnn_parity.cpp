/**
 * @file test_rnn_parity.cpp
 * @brief RNN and advanced operation parity tests across backends
 *
 * Verifies that LSTM, GRU, and other advanced operations produce
 * identical results across CPU, CUDA, ROCm, OneAPI, and Vulkan backends.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// LSTM Forward Parity (via Module::forward single-input)
// ============================================================================

TEST(RNNParity, LSTM_Forward) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    int64_t input_size = 8;
    int64_t hidden_size = 16;
    int64_t seq_len = 5;
    int64_t batch = 2;

    auto input = randn({seq_len, batch, input_size}, DType::Float32, Device::cpu());

    // Use single-input forward which auto-initializes hidden state
    test_operation_parity([&](const std::vector<Tensor>& inputs) {
        nn::LSTM lstm(input_size, hidden_size, 1);
        return lstm.forward_impl(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-3f, 1e-4f, "LSTM Forward");
}

// ============================================================================
// GRU Forward Parity (via Module::forward single-input)
// ============================================================================

TEST(RNNParity, GRU_Forward) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    int64_t input_size = 8;
    int64_t hidden_size = 16;
    int64_t seq_len = 5;
    int64_t batch = 2;

    auto input = randn({seq_len, batch, input_size}, DType::Float32, Device::cpu());

    test_operation_parity([&](const std::vector<Tensor>& inputs) {
        nn::GRU gru(input_size, hidden_size, 1);
        return gru.forward_impl(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-3f, 1e-4f, "GRU Forward");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
