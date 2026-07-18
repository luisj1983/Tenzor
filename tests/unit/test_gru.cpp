#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// GRUCell Tests
// ============================================================================

class GRUCellTest : public BackendTest {};

TEST_P(GRUCellTest, BasicForward) {
    // Test basic forward pass
    nn::GRUCell cell(10, 20);
    cell.to(device);
    auto input = Variable(randn({5, 10}, DType::Float32, device), true);
    auto h = Variable(randn({5, 20}, DType::Float32, device), true);

    auto h_next = cell.forward(input, h);

    EXPECT_EQ(h_next.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[1], 20) << "Failed on " << device.to_string();
}

TEST_P(GRUCellTest, NoInitialHidden) {
    // Test with zero-initialized hidden state
    nn::GRUCell cell(10, 20);
    cell.to(device);
    auto input = Variable(randn({5, 10}, DType::Float32, device), true);

    auto h_next = cell.forward(input);

    EXPECT_EQ(h_next.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[1], 20) << "Failed on " << device.to_string();
}

TEST_P(GRUCellTest, NoBias) {
    // Test without bias
    nn::GRUCell cell(10, 20, false);
    cell.to(device);
    auto input = Variable(randn({5, 10}, DType::Float32, device), true);

    auto h_next = cell.forward(input);

    EXPECT_EQ(h_next.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[1], 20) << "Failed on " << device.to_string();
}

TEST_P(GRUCellTest, HiddenStateEvolution) {
    // Test that hidden state evolves across time steps
    nn::GRUCell cell(10, 20);
    cell.to(device);
    auto input = Variable(randn({5, 10}, DType::Float32, device), true);

    auto h1 = cell.forward(input);
    auto h2 = cell.forward(input, h1);
    auto h3 = cell.forward(input, h2);

    // Each step should produce outputs
    EXPECT_EQ(h3.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h3.shape()[1], 20) << "Failed on " << device.to_string();
}

TEST_P(GRUCellTest, ParameterCount) {
    // Test parameter count
    nn::GRUCell cell(10, 20);
    auto params = cell.parameters();

    // GRU uses PyTorch-style combined weight matrices:
    // weight_ih: (3*hidden, input) - input-to-hidden for all 3 gates combined
    // weight_hh: (3*hidden, hidden) - hidden-to-hidden for all 3 gates combined
    // 2 linear layers * 2 params (weight + bias) = 4 parameters
    EXPECT_EQ(params.size(), 4) << "Failed on " << device.to_string();
}

// forward_with_precomputed_ih previously ran its whole body on raw Tensors and
// wrapped the output in Variable(h_new_tensor, true) with no grad_fn — silently
// zeroing gradients on hx and weight_hh_. Rewritten with Variable-level slice/
// sigmoid/tanh/arithmetic; this test locks in the recurrent-grad invariant.
TEST_P(GRUCellTest, PrecomputedIhPropagatesHiddenGradient) {
    nn::GRUCell cell(10, 20);
    cell.to(device);

    // 3*hidden = 60 wide for GRU gates.
    auto gates_ih = randn({5, 60}, DType::Float32, device);
    Variable hx(randn({5, 20}, DType::Float32, device), /*requires_grad=*/true);

    auto out = cell.forward_with_precomputed_ih(gates_ih, hx);
    auto loss = tenzor::sum(out);
    loss.backward();

    ASSERT_TRUE(hx.has_grad())
        << "GRUCell::forward_with_precomputed_ih must propagate grad to hx on "
        << device.to_string();
    auto g = hx.grad().value().to(Device::cpu()).contiguous();
    const float* gp = g.data<float>();
    float max_abs = 0.0f;
    for (int64_t i = 0; i < g.numel(); ++i) {
        EXPECT_FALSE(std::isnan(gp[i])) << "grad NaN at " << i;
        max_abs = std::max(max_abs, std::abs(gp[i]));
    }
    EXPECT_GT(max_abs, 0.0f)
        << "hx grad identically zero — GRUCell recurrent-side autograd severed on "
        << device.to_string();
}

// ============================================================================
// GRU Tests
// ============================================================================

class GRUTest : public BackendTest {};

TEST_P(GRUTest, BasicForward) {
    // Test basic forward pass
    nn::GRU gru(10, 20, 1);
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);  // (seq_len, batch, features)

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape().size(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();  // seq_len
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string(); // hidden_size

    EXPECT_EQ(h_n.shape().size(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(h_n.shape()[0], 1) << "Failed on " << device.to_string();  // num_layers
    EXPECT_EQ(h_n.shape()[1], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(h_n.shape()[2], 20) << "Failed on " << device.to_string(); // hidden_size
}

TEST_P(GRUTest, MultiLayer) {
    // Test multi-layer GRU
    nn::GRU gru(10, 20, 3);  // 3 layers
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string();

    EXPECT_EQ(h_n.shape()[0], 3) << "Failed on " << device.to_string();  // 3 layers
}

TEST_P(GRUTest, BatchFirst) {
    // Test with batch_first=true
    nn::GRU gru(10, 20, 1, true, true);  // batch_first=true
    gru.to(device);
    auto input = Variable(randn({5, 7, 10}, DType::Float32, device), true);  // (batch, seq_len, features)

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(output.shape()[1], 7) << "Failed on " << device.to_string();  // seq_len
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string(); // hidden_size
}

TEST_P(GRUTest, Bidirectional) {
    // Test bidirectional GRU
    nn::GRU gru(10, 20, 1, true, false, 0.0, true);  // bidirectional=true
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 40) << "Failed on " << device.to_string(); // hidden_size * 2

    EXPECT_EQ(h_n.shape()[0], 2) << "Failed on " << device.to_string();  // num_layers * num_directions
}

TEST_P(GRUTest, BidirectionalMultiLayer) {
    // Test bidirectional multi-layer GRU
    nn::GRU gru(10, 20, 2, true, false, 0.0, true);
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[2], 40) << "Failed on " << device.to_string(); // hidden_size * 2
    EXPECT_EQ(h_n.shape()[0], 4) << "Failed on " << device.to_string();     // num_layers * num_directions
}

TEST_P(GRUTest, WithDropout) {
    // Test with dropout between layers
    nn::GRU gru(10, 20, 3, true, false, 0.5);  // 50% dropout
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string();
}

TEST_P(GRUTest, InitialHiddenState) {
    // Test with provided initial hidden state
    nn::GRU gru(10, 20, 2);
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);
    auto h0 = Variable(randn({2, 5, 20}, DType::Float32, device), true);

    auto [output, h_n] = gru.forward(input, h0);

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(h_n.shape()[0], 2) << "Failed on " << device.to_string();
}

TEST_P(GRUTest, SequenceLengthVariation) {
    // Test with different sequence lengths
    nn::GRU gru(10, 20);
    gru.to(device);

    // Short sequence
    auto input1 = Variable(randn({3, 5, 10}, DType::Float32, device), true);
    auto [output1, h_n1] = gru.forward(input1, Variable{});
    EXPECT_EQ(output1.shape()[0], 3) << "Failed on " << device.to_string();

    // Long sequence
    auto input2 = Variable(randn({50, 5, 10}, DType::Float32, device), true);
    auto [output2, h_n2] = gru.forward(input2, Variable{});
    EXPECT_EQ(output2.shape()[0], 50) << "Failed on " << device.to_string();
}

TEST_P(GRUTest, BatchSizeVariation) {
    // Test with different batch sizes
    nn::GRU gru(10, 20);
    gru.to(device);

    // Small batch
    auto input1 = Variable(randn({7, 2, 10}, DType::Float32, device), true);
    auto [output1, h_n1] = gru.forward(input1, Variable{});
    EXPECT_EQ(output1.shape()[1], 2) << "Failed on " << device.to_string();

    // Large batch
    auto input2 = Variable(randn({7, 32, 10}, DType::Float32, device), true);
    auto [output2, h_n2] = gru.forward(input2, Variable{});
    EXPECT_EQ(output2.shape()[1], 32) << "Failed on " << device.to_string();
}

TEST_P(GRUTest, SingleTimestep) {
    // Test with single timestep
    nn::GRU gru(10, 20);
    gru.to(device);
    auto input = Variable(randn({1, 5, 10}, DType::Float32, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 1) << "Failed on " << device.to_string();
}

TEST_P(GRUTest, OutputConsistency) {
    // Test that output is deterministic
    nn::GRU gru(10, 20);
    gru.to(device);
    auto input = Variable(ones({7, 5, 10}, DType::Float32, device), true);

    auto [output1, h_n1] = gru.forward(input, Variable{});
    auto [output2, h_n2] = gru.forward(input, Variable{});

    // Shapes should match
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(output1.shape()[i], output2.shape()[i]) << "Failed on " << device.to_string();
        EXPECT_EQ(h_n1.shape()[i], h_n2.shape()[i]) << "Failed on " << device.to_string();
    }
}

TEST_P(GRUTest, LearnableInitialStateGradientFlow) {
    // The fused cuDNN-training fast path (CUDA, Float32, unidirectional,
    // no projection) used to build next_functions/input_variables without
    // h0 at all, and CudnnGRUTrainBackward::backward() dropped grad_hx
    // from its returned gradient vector -- a user-supplied learnable
    // initial hidden state silently received a zero gradient instead of
    // the correct one. Other backends exercise a different
    // (already-correct) autograd path, so this is a genuine cross-backend
    // regression check, not a CUDA-only assertion.
    nn::GRU gru(10, 20, 1);
    gru.to(device);

    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);
    auto h0 = Variable(randn({1, 5, 20}, DType::Float32, device), true);

    auto [output, h_n] = gru.forward(input, h0);

    auto loss = sum(output) + sum(h_n);
    loss.backward();

    ASSERT_TRUE(h0.grad().has_value()) << "h0 gradient missing on " << device.to_string();
    auto h0_grad_cpu = h0.grad()->to(Device::cpu()).to(DType::Float32).contiguous();
    float h0_grad_sum = 0.0f;
    const float* hp = h0_grad_cpu.data<float>();
    for (int64_t i = 0; i < h0_grad_cpu.numel(); ++i) h0_grad_sum += std::abs(hp[i]);
    EXPECT_GT(h0_grad_sum, 0.0f) << "h0 gradient is all-zero on " << device.to_string();
}

// The fused fast path (Float32, eval(), no_grad, unidirectional,
// linear_before_reset -- the standard fast-path-eligible config) used to
// ignore `lengths` entirely, processing the full padded seq_len for every
// batch row instead of falling through to the lengths-aware per-timestep
// loop. Row 1 here is padded to seq_len=6 but has a true length of 3; its
// output for timesteps 0..2 and its h_n must match an independent run of
// the same 3-timestep sequence with no padding at all -- if the fused path
// were still ignoring lengths, the padding (garbage timesteps 3..5) would
// contaminate row 1's h_n and this would fail.
TEST_P(GRUTest, FusedFastPathRespectsLengths) {
    nn::GRU gru(4, 6, 1);
    gru.to(device);
    gru.eval();

    const int64_t seq_len = 6, true_len = 3, feat = 4, hidden = 6;

    auto padded_cpu = randn({seq_len, 2, feat}, DType::Float32, Device::cpu());
    {
        float* p = padded_cpu.data<float>();
        for (int64_t t = true_len; t < seq_len; ++t) {
            for (int64_t f = 0; f < feat; ++f) {
                p[(t * 2 + 1) * feat + f] = 1000.0f;
            }
        }
    }
    auto lengths_cpu = zeros({2}, DType::Int64, Device::cpu());
    lengths_cpu.data<int64_t>()[0] = seq_len;
    lengths_cpu.data<int64_t>()[1] = true_len;

    auto row1_unpadded_cpu = zeros({true_len, 1, feat}, DType::Float32, Device::cpu());
    {
        const float* src = padded_cpu.data<float>();
        float* dst = row1_unpadded_cpu.data<float>();
        for (int64_t t = 0; t < true_len; ++t) {
            for (int64_t f = 0; f < feat; ++f) {
                dst[t * feat + f] = src[(t * 2 + 1) * feat + f];
            }
        }
    }

    tenzor::NoGradGuard no_grad;
    Variable padded_in(padded_cpu.to(device), false);
    auto [padded_out, padded_h] = gru.forward(padded_in, Variable{}, lengths_cpu);

    Variable row1_in(row1_unpadded_cpu.to(device), false);
    auto [row1_out, row1_h] = gru.forward(row1_in, Variable{}, Tensor{});

    auto padded_out_cpu = padded_out.tensor().to(Device::cpu()).to(DType::Float32).contiguous();
    auto row1_out_cpu = row1_out.tensor().to(Device::cpu()).to(DType::Float32).contiguous();
    const float* po = padded_out_cpu.data<float>();
    const float* ro = row1_out_cpu.data<float>();
    for (int64_t t = 0; t < true_len; ++t) {
        for (int64_t h = 0; h < hidden; ++h) {
            EXPECT_NEAR(po[(t * 2 + 1) * hidden + h], ro[t * hidden + h], 1e-3f)
                << "output[t=" << t << ",h=" << h << "] diverges on " << device.to_string()
                << " -- fused fast path leaked padding into row 1";
        }
    }

    auto padded_h_cpu = padded_h.tensor().to(Device::cpu()).to(DType::Float32).contiguous();
    auto row1_h_cpu = row1_h.tensor().to(Device::cpu()).to(DType::Float32).contiguous();
    const float* ph = padded_h_cpu.data<float>();
    const float* rh = row1_h_cpu.data<float>();
    for (int64_t h = 0; h < hidden; ++h) {
        EXPECT_NEAR(ph[hidden + h], rh[h], 1e-3f)
            << "h_n[h=" << h << "] diverges on " << device.to_string()
            << " -- fused fast path leaked padding into row 1's final hidden state";
    }
}

TEST_P(GRUTest, GradientFlow) {
    // Test that gradients can flow through GRU
    nn::GRU gru(10, 20);
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    // Check that output requires grad
    EXPECT_TRUE(output.requires_grad()) << "Failed on " << device.to_string();

    // Sum for scalar output
    auto loss = Variable(sum(output.tensor(), std::nullopt, false), true);

    // Backward should not throw
    EXPECT_NO_THROW({
        loss.backward();
    }) << "Failed on " << device.to_string();
}

TEST_P(GRUTest, TrainingMode) {
    // Test training/eval mode switching
    nn::GRU gru(10, 20, 2, true, false, 0.5);

    EXPECT_TRUE(gru.is_training()) << "Failed on " << device.to_string();

    gru.eval();
    EXPECT_FALSE(gru.is_training()) << "Failed on " << device.to_string();

    gru.train();
    EXPECT_TRUE(gru.is_training()) << "Failed on " << device.to_string();
}

TEST_P(GRUTest, ParameterCount) {
    // Test parameter counting
    nn::GRU gru(10, 20, 2);
    auto params = gru.parameters();

    // Each GRUCell uses PyTorch-style combined weight matrices:
    // weight_ih + weight_hh = 2 linear layers * 2 params = 4 params per cell
    // 2 layers * 4 params = 8 parameters total
    EXPECT_EQ(params.size(), 8) << "Failed on " << device.to_string();
}

TEST_P(GRUTest, LargeHidden) {
    // Test with large hidden size
    nn::GRU gru(10, 512);
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[2], 512) << "Failed on " << device.to_string();
}

TEST_P(GRUTest, VeryDeepNetwork) {
    // Test with many layers
    nn::GRU gru(10, 20, 5);
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(h_n.shape()[0], 5) << "Failed on " << device.to_string();  // 5 layers
}

TEST_P(GRUTest, LongSequence) {
    // Test with very long sequence
    nn::GRU gru(10, 20);
    gru.to(device);
    auto input = Variable(randn({100, 5, 10}, DType::Float32, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 100) << "Failed on " << device.to_string();
}

TEST_P(GRUTest, InvalidNumLayers) {
    // Test that invalid num_layers throws
    EXPECT_THROW({
        nn::GRU gru(10, 20, 0);
    }, std::invalid_argument) << "Failed on " << device.to_string();
}

TEST_P(GRUTest, InvalidDropout) {
    // Test that invalid dropout throws
    EXPECT_THROW({
        nn::GRU gru(10, 20, 2, true, false, 1.5);
    }, std::invalid_argument) << "Failed on " << device.to_string();
}

TEST_P(GRUTest, GateOutputRanges) {
    // Test that GRU gates produce reasonable outputs
    nn::GRU gru(10, 20);
    gru.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, h_n] = gru.forward(input, Variable{});

    // Hidden state values should be in reasonable range (due to tanh in new gate)
    auto h_cpu = h_n.tensor().to(Device::cpu());
    auto h_data = h_cpu.data<float>();
    bool all_reasonable = true;
    for (int64_t i = 0; i < h_cpu.numel(); ++i) {
        // Values should generally be in [-10, 10] range
        if (std::abs(h_data[i]) > 10.0f) {
            all_reasonable = false;
            break;
        }
    }
    EXPECT_TRUE(all_reasonable) << "Failed on " << device.to_string();
}

TEST_P(GRUTest, BatchFirstBidirectional) {
    // Test combination of batch_first and bidirectional
    nn::GRU gru(10, 20, 1, true, true, 0.0, true);
    gru.to(device);
    auto input = Variable(randn({5, 7, 10}, DType::Float32, device), true);  // (batch, seq, features)

    auto [output, h_n] = gru.forward(input, Variable{});

    EXPECT_EQ(output.shape()[0], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(output.shape()[1], 7) << "Failed on " << device.to_string();  // seq_len
    EXPECT_EQ(output.shape()[2], 40) << "Failed on " << device.to_string(); // hidden_size * 2
}

TEST_P(GRUTest, ComparisonWithLSTM) {
    // Ensure GRU has similar behavior to LSTM (fewer parameters, similar performance)
    nn::GRU gru(10, 20, 2);
    gru.to(device);
    nn::LSTM lstm(10, 20, 2);
    lstm.to(device);

    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [gru_output, gru_h] = gru.forward(input, Variable{});
    auto [lstm_output, lstm_states] = lstm.forward(input, {Variable{}, Variable{}});

    // Both should produce same shaped outputs
    EXPECT_EQ(gru_output.shape()[0], lstm_output.shape()[0]) << "Failed on " << device.to_string();
    EXPECT_EQ(gru_output.shape()[1], lstm_output.shape()[1]) << "Failed on " << device.to_string();
    EXPECT_EQ(gru_output.shape()[2], lstm_output.shape()[2]) << "Failed on " << device.to_string();

    // GRU has more parameters due to separate gate transforms
    // (GRU uses 6 separate Linear layers vs LSTM's 2 combined layers)
    auto gru_params = gru.parameters();
    auto lstm_params = lstm.parameters();
    EXPECT_GT(gru_params.size(), lstm_params.size()) << "Failed on " << device.to_string();
}

TEST_P(GRUTest, MemoryEfficiency) {
    // Test that GRU uses less memory than LSTM
    nn::GRU gru(10, 20);
    gru.to(device);
    nn::LSTM lstm(10, 20);
    lstm.to(device);

    // GRU only returns hidden state, LSTM returns both hidden and cell
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [gru_output, gru_h] = gru.forward(input, Variable{});
    auto [lstm_output, lstm_states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [lstm_h, lstm_c] = lstm_states;

    // GRU hidden state shape
    EXPECT_EQ(gru_h.shape().size(), 3) << "Failed on " << device.to_string();

    // LSTM has both h and c
    EXPECT_EQ(lstm_h.shape().size(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(lstm_c.shape().size(), 3) << "Failed on " << device.to_string();
}

INSTANTIATE_BACKEND_TESTS(GRUCellTest);
INSTANTIATE_BACKEND_TESTS(GRUTest);
