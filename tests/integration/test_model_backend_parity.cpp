/**
 * @file test_model_backend_parity.cpp
 * @brief End-to-end model forward pass parity tests
 *
 * Tests that complete model architectures (MLP, LeNet5, Transformer,
 * LSTM, Autoencoder) produce identical results across all available
 * backends when given the same weights and inputs.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include "../backend_parity/parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Helper: copy parameters from src module to dst module
// ============================================================================

static void copy_params(nn::Module& src, nn::Module& dst) {
    auto src_params = src.parameters();
    auto dst_params = dst.parameters();
    for (size_t p = 0; p < src_params.size(); ++p) {
        dst_params[p]->tensor() = src_params[p]->tensor().clone();
    }
}

// ============================================================================
// Test 1: SimpleMLP — 3-layer MLP (64->32->16->8) with ReLU
// ============================================================================

TEST(ModelBackendParity, SimpleMLP) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("model backend parity");

    nn::Linear fc1(64, 32);
    nn::Linear fc2(32, 16);
    nn::Linear fc3(16, 8);

    auto input = randn({4, 64}, DType::Float32, Device::cpu());

    auto x = nn::relu(fc1.forward(Variable(input, false)));
    x = nn::relu(fc2.forward(x));
    auto ref = fc3.forward(x).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Linear fc1_dev(64, 32);
            nn::Linear fc2_dev(32, 16);
            nn::Linear fc3_dev(16, 8);
            copy_params(fc1, fc1_dev);
            copy_params(fc2, fc2_dev);
            copy_params(fc3, fc3_dev);
            fc1_dev.to(backends[i]);
            fc2_dev.to(backends[i]);
            fc3_dev.to(backends[i]);

            auto in_dev = input.to(backends[i]);
            auto xd = nn::relu(fc1_dev.forward(Variable(in_dev, false)));
            xd = nn::relu(fc2_dev.forward(xd));
            auto out = fc3_dev.forward(xd).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, out, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "SimpleMLP failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// Test 2: LeNet5 — Conv2d(1,6,5) -> pool -> Conv2d(6,16,5) -> pool
//         -> flatten -> linear -> linear -> linear
//         Input: {1, 1, 32, 32}
// ============================================================================

TEST(ModelBackendParity, LeNet5) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("model backend parity");

    nn::Conv2d conv1(1, 6, 5);     // (1,6,28,28)
    nn::MaxPool2d pool1(2);        // (1,6,14,14)
    nn::Conv2d conv2(6, 16, 5);    // (1,16,10,10)
    nn::MaxPool2d pool2(2);        // (1,16,5,5)
    nn::Flatten flatten;           // (1, 400)
    nn::Linear fc1(400, 120);
    nn::Linear fc2(120, 84);
    nn::Linear fc3(84, 10);

    auto input = randn({1, 1, 32, 32}, DType::Float32, Device::cpu());

    auto x = pool1.forward(nn::relu(conv1.forward(Variable(input, false))));
    x = pool2.forward(nn::relu(conv2.forward(x)));
    x = flatten.forward(x);
    x = nn::relu(fc1.forward(x));
    x = nn::relu(fc2.forward(x));
    auto ref = fc3.forward(x).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Conv2d c1d(1, 6, 5);
            nn::MaxPool2d p1d(2);
            nn::Conv2d c2d(6, 16, 5);
            nn::MaxPool2d p2d(2);
            nn::Flatten fld;
            nn::Linear f1d(400, 120);
            nn::Linear f2d(120, 84);
            nn::Linear f3d(84, 10);

            copy_params(conv1, c1d);
            copy_params(conv2, c2d);
            copy_params(fc1, f1d);
            copy_params(fc2, f2d);
            copy_params(fc3, f3d);

            c1d.to(backends[i]); p1d.to(backends[i]);
            c2d.to(backends[i]); p2d.to(backends[i]);
            fld.to(backends[i]);
            f1d.to(backends[i]); f2d.to(backends[i]); f3d.to(backends[i]);

            auto in_dev = input.to(backends[i]);
            auto xd = p1d.forward(nn::relu(c1d.forward(Variable(in_dev, false))));
            xd = p2d.forward(nn::relu(c2d.forward(xd)));
            xd = fld.forward(xd);
            xd = nn::relu(f1d.forward(xd));
            xd = nn::relu(f2d.forward(xd));
            auto out = f3d.forward(xd).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, out, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "LeNet5 failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// Test 3: TransformerEncoder1Layer — TransformerEncoderLayer(64, 4)
//         Input: {8, 4, 64} (seq_len=8, batch=4, d_model=64)
// ============================================================================

TEST(ModelBackendParity, TransformerEncoder1Layer) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("model backend parity");

    int64_t d_model = 64, nhead = 4;

    // dropout=0 for deterministic output
    nn::TransformerEncoderLayer encoder(d_model, nhead, 128, 0.0);
    encoder.eval();

    // Sequence-first input: (seq_len, batch, d_model)
    auto input = randn({8, 4, d_model}, DType::Float32, Device::cpu());

    auto ref = encoder.forward(Variable(input, false), Tensor{}, Tensor{}).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::TransformerEncoderLayer enc_dev(d_model, nhead, 128, 0.0);
            enc_dev.eval();
            copy_params(encoder, enc_dev);
            enc_dev.to(backends[i]);

            auto in_dev = input.to(backends[i]);
            auto out = enc_dev.forward(Variable(in_dev, false), Tensor{}, Tensor{}).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, out, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "TransformerEncoder1Layer failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// Test 4: LSTMSeq2Seq — nn::LSTM(32, 64, 1), input {8, 4, 32}
//         (seq_len=8, batch=4, input_size=32)
// ============================================================================

TEST(ModelBackendParity, LSTMSeq2Seq) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("model backend parity");

    int64_t input_size = 32, hidden_size = 64, num_layers = 1;
    int64_t seq_len = 8, batch = 4;

    nn::LSTM lstm(input_size, hidden_size, num_layers);

    // Sequence-first: (seq_len, batch, input_size)
    auto input = randn({seq_len, batch, input_size}, DType::Float32, Device::cpu());

    auto [output, hidden_state] = lstm.forward(
        Variable(input, false), {Variable{}, Variable{}});
    auto ref = output.tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::LSTM lstm_dev(input_size, hidden_size, num_layers);
            copy_params(lstm, lstm_dev);
            lstm_dev.to(backends[i]);

            auto in_dev = input.to(backends[i]);
            auto [out_dev, _] = lstm_dev.forward(
                Variable(in_dev, false), {Variable{}, Variable{}});
            auto out = out_dev.tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, out, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "LSTMSeq2Seq failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// Test 5: EncoderDecoder — simple conv autoencoder
//         conv2d -> conv2d -> convtranspose2d
// ============================================================================

TEST(ModelBackendParity, EncoderDecoder) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("model backend parity");

    // Encoder
    nn::Conv2d enc1(1, 8, 3, 2, 1);    // (1,1,16,16) -> (1,8,8,8)
    nn::Conv2d enc2(8, 16, 3, 2, 1);   // (1,8,8,8) -> (1,16,4,4)
    // Decoder
    nn::ConvTranspose2d dec1(16, 1, 4, 4);  // (1,16,4,4) -> (1,1,16,16)

    auto input = randn({1, 1, 16, 16}, DType::Float32, Device::cpu());

    auto x = nn::relu(enc1.forward(Variable(input, false)));
    x = nn::relu(enc2.forward(x));
    auto ref = dec1.forward(x).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Conv2d e1d(1, 8, 3, 2, 1);
            nn::Conv2d e2d(8, 16, 3, 2, 1);
            nn::ConvTranspose2d d1d(16, 1, 4, 4);

            copy_params(enc1, e1d);
            copy_params(enc2, e2d);
            copy_params(dec1, d1d);

            e1d.to(backends[i]);
            e2d.to(backends[i]);
            d1d.to(backends[i]);

            auto in_dev = input.to(backends[i]);
            auto xd = nn::relu(e1d.forward(Variable(in_dev, false)));
            xd = nn::relu(e2d.forward(xd));
            auto out = d1d.forward(xd).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, out, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "EncoderDecoder failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    tenzor::finalize();
    return result;
}
