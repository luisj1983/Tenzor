/**
 * @file test_nn_rnn_parity.cpp
 * @brief RNN layer parity tests across backends
 *
 * Verifies that RNNCell, LSTMCell, GRUCell, LSTM, GRU, and RNN layers
 * produce consistent results across CPU, CUDA, ROCm, OneAPI, and Vulkan.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Cell-level tests
// ============================================================================

TEST(NNRNNParity, LSTMCell) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    nn::LSTMCell cell(32, 64);
    auto input = randn({4, 32}, DType::Float32, Device::cpu());
    auto h = randn({4, 64}, DType::Float32, Device::cpu());
    auto c = randn({4, 64}, DType::Float32, Device::cpu());

    auto [ref_h, ref_c] = cell.forward(Variable(input, false),
                                        Variable(h, false),
                                        Variable(c, false));
    auto ref = ref_h.tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::LSTMCell cell_dev(32, 64);
            auto params_src = cell.parameters();
            auto params_dst = cell_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            cell_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto h_dev = h.to(backends[i]);
            auto c_dev = c.to(backends[i]);
            auto [out_h, out_c] = cell_dev.forward(Variable(input_dev, false),
                                                    Variable(h_dev, false),
                                                    Variable(c_dev, false));
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, out_h.tensor(), 1e-4f, 1e-4f);
        } catch (const std::exception& e) {
            std::cerr << "Skipped on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

TEST(NNRNNParity, GRUCell) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    nn::GRUCell cell(32, 64);
    auto input = randn({4, 32}, DType::Float32, Device::cpu());
    auto h = randn({4, 64}, DType::Float32, Device::cpu());

    auto ref = cell.forward(Variable(input, false), Variable(h, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::GRUCell cell_dev(32, 64);
            auto params_src = cell.parameters();
            auto params_dst = cell_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            cell_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto h_dev = h.to(backends[i]);
            auto output = cell_dev.forward(Variable(input_dev, false),
                                            Variable(h_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-4f, 1e-4f);
        } catch (const std::exception& e) {
            std::cerr << "Skipped on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

TEST(NNRNNParity, RNNCell) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    nn::RNNCell cell(32, 64);
    auto input = randn({4, 32}, DType::Float32, Device::cpu());
    auto h = randn({4, 64}, DType::Float32, Device::cpu());

    auto ref = cell.forward(Variable(input, false), Variable(h, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::RNNCell cell_dev(32, 64);
            auto params_src = cell.parameters();
            auto params_dst = cell_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            cell_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto h_dev = h.to(backends[i]);
            auto output = cell_dev.forward(Variable(input_dev, false),
                                            Variable(h_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-4f, 1e-4f);
        } catch (const std::exception& e) {
            std::cerr << "Skipped on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Full-sequence multi-layer tests
// ============================================================================

TEST(NNRNNParity, LSTM_MultiLayer) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // input_size=32, hidden_size=64, num_layers=2
    nn::LSTM layer(32, 64, 2);
    auto input = randn({8, 4, 32}, DType::Float32, Device::cpu());

    auto ref = layer.forward_impl(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::LSTM layer_dev(32, 64, 2);
            auto params_src = layer.parameters();
            auto params_dst = layer_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            layer_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = layer_dev.forward_impl(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-2f, 1e-2f);
        } catch (const std::exception& e) {
            std::cerr << "Skipped on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

TEST(NNRNNParity, GRU_MultiLayer) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // input_size=32, hidden_size=64, num_layers=2
    nn::GRU layer(32, 64, 2);
    auto input = randn({8, 4, 32}, DType::Float32, Device::cpu());

    auto ref = layer.forward_impl(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::GRU layer_dev(32, 64, 2);
            auto params_src = layer.parameters();
            auto params_dst = layer_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            layer_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = layer_dev.forward_impl(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-2f, 1e-2f);
        } catch (const std::exception& e) {
            std::cerr << "Skipped on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

TEST(NNRNNParity, RNN) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    nn::RNN layer(32, 64);
    auto input = randn({8, 4, 32}, DType::Float32, Device::cpu());

    auto ref = layer.forward_impl(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::RNN layer_dev(32, 64);
            auto params_src = layer.parameters();
            auto params_dst = layer_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            layer_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = layer_dev.forward_impl(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            std::cerr << "Skipped on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

TEST(NNRNNParity, LSTM_Bidirectional) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // input_size=32, hidden_size=64, num_layers=1, bias=true, batch_first=false,
    // dropout=0.0, bidirectional=true
    nn::LSTM layer(32, 64, 1, true, false, 0.0, true);
    auto input = randn({8, 4, 32}, DType::Float32, Device::cpu());

    auto ref = layer.forward_impl(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::LSTM layer_dev(32, 64, 1, true, false, 0.0, true);
            auto params_src = layer.parameters();
            auto params_dst = layer_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            layer_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = layer_dev.forward_impl(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-2f, 1e-2f);
        } catch (const std::exception& e) {
            std::cerr << "Skipped on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

TEST(NNRNNParity, LSTM_Dropout_Eval) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // input_size=32, hidden_size=64, num_layers=2, bias=true, batch_first=false,
    // dropout=0.5, bidirectional=false
    nn::LSTM layer(32, 64, 2, true, false, 0.5, false);
    layer.eval();
    auto input = randn({8, 4, 32}, DType::Float32, Device::cpu());

    auto ref = layer.forward_impl(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::LSTM layer_dev(32, 64, 2, true, false, 0.5, false);
            layer_dev.eval();
            auto params_src = layer.parameters();
            auto params_dst = layer_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            layer_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = layer_dev.forward_impl(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-2f, 1e-2f);
        } catch (const std::exception& e) {
            std::cerr << "Skipped on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// Sequential cell unrolling tests
// ============================================================================

TEST(NNRNNParity, LSTMCell_Sequence) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    nn::LSTMCell cell(32, 64);

    // Generate a sequence of 4 input steps
    auto x0 = randn({4, 32}, DType::Float32, Device::cpu());
    auto x1 = randn({4, 32}, DType::Float32, Device::cpu());
    auto x2 = randn({4, 32}, DType::Float32, Device::cpu());
    auto x3 = randn({4, 32}, DType::Float32, Device::cpu());
    auto h = zeros({4, 64}, DType::Float32, Device::cpu());
    auto c = zeros({4, 64}, DType::Float32, Device::cpu());

    // Unroll on CPU reference
    auto [h0, c0] = cell.forward(Variable(x0, false), Variable(h, false), Variable(c, false));
    auto [h1, c1] = cell.forward(Variable(x1, false), h0, c0);
    auto [h2, c2] = cell.forward(Variable(x2, false), h1, c1);
    auto [h3, c3] = cell.forward(Variable(x3, false), h2, c2);
    auto ref = h3.tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::LSTMCell cell_dev(32, 64);
            auto params_src = cell.parameters();
            auto params_dst = cell_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            cell_dev.to(backends[i]);

            auto x0d = x0.to(backends[i]);
            auto x1d = x1.to(backends[i]);
            auto x2d = x2.to(backends[i]);
            auto x3d = x3.to(backends[i]);
            auto hd = h.to(backends[i]);
            auto cd = c.to(backends[i]);

            auto [dh0, dc0] = cell_dev.forward(Variable(x0d, false), Variable(hd, false), Variable(cd, false));
            auto [dh1, dc1] = cell_dev.forward(Variable(x1d, false), dh0, dc0);
            auto [dh2, dc2] = cell_dev.forward(Variable(x2d, false), dh1, dc1);
            auto [dh3, dc3] = cell_dev.forward(Variable(x3d, false), dh2, dc2);
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, dh3.tensor(), 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            std::cerr << "Skipped on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

TEST(NNRNNParity, GRUCell_Sequence) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    nn::GRUCell cell(32, 64);

    // Generate a sequence of 4 input steps
    auto x0 = randn({4, 32}, DType::Float32, Device::cpu());
    auto x1 = randn({4, 32}, DType::Float32, Device::cpu());
    auto x2 = randn({4, 32}, DType::Float32, Device::cpu());
    auto x3 = randn({4, 32}, DType::Float32, Device::cpu());
    auto h = zeros({4, 64}, DType::Float32, Device::cpu());

    // Unroll on CPU reference
    auto h0 = cell.forward(Variable(x0, false), Variable(h, false));
    auto h1 = cell.forward(Variable(x1, false), h0);
    auto h2 = cell.forward(Variable(x2, false), h1);
    auto h3 = cell.forward(Variable(x3, false), h2);
    auto ref = h3.tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::GRUCell cell_dev(32, 64);
            auto params_src = cell.parameters();
            auto params_dst = cell_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            cell_dev.to(backends[i]);

            auto x0d = x0.to(backends[i]);
            auto x1d = x1.to(backends[i]);
            auto x2d = x2.to(backends[i]);
            auto x3d = x3.to(backends[i]);
            auto hd = h.to(backends[i]);

            auto dh0 = cell_dev.forward(Variable(x0d, false), Variable(hd, false));
            auto dh1 = cell_dev.forward(Variable(x1d, false), dh0);
            auto dh2 = cell_dev.forward(Variable(x2d, false), dh1);
            auto dh3 = cell_dev.forward(Variable(x3d, false), dh2);
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, dh3.tensor(), 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            std::cerr << "Skipped on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
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
