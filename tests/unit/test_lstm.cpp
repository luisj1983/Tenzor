#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <cmath>
#include <cstdlib>
#include <tuple>
#include <utility>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// LSTMCell Tests
// ============================================================================

class LSTMCellTestFixture : public BackendTest {};

TEST_P(LSTMCellTestFixture, BasicForward) {
    // Test basic forward pass
    nn::LSTMCell cell(10, 20);

    cell.to(device);
    auto input = Variable(randn({5, 10}, DType::Float32, device), true);
    auto h = Variable(randn({5, 20}, DType::Float32, device), true);
    auto c = Variable(randn({5, 20}, DType::Float32, device), true);

    auto [h_next, c_next] = cell.forward(input, h, c);

    EXPECT_EQ(h_next.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[1], 20) << "Failed on " << device.to_string();

    EXPECT_EQ(c_next.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c_next.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(c_next.shape()[1], 20) << "Failed on " << device.to_string();
}

TEST_P(LSTMCellTestFixture, NoInitialStates) {
    // Test with zero-initialized states
    nn::LSTMCell cell(10, 20);

    cell.to(device);
    auto input = Variable(randn({5, 10}, DType::Float32, device), true);

    auto [h_next, c_next] = cell.forward(input, Variable{}, Variable{});

    EXPECT_EQ(h_next.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[1], 20) << "Failed on " << device.to_string();

    EXPECT_EQ(c_next.shape().size(), 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c_next.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(c_next.shape()[1], 20) << "Failed on " << device.to_string();
}

TEST_P(LSTMCellTestFixture, NoBias) {
    // Test without bias
    nn::LSTMCell cell(10, 20, false);

    cell.to(device);
    auto input = Variable(randn({5, 10}, DType::Float32, device), true);

    auto [h_next, c_next] = cell.forward(input, Variable{}, Variable{});

    EXPECT_EQ(h_next.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h_next.shape()[1], 20) << "Failed on " << device.to_string();
}

TEST_P(LSTMCellTestFixture, CellStateEvolution) {
    // Test that cell state evolves across time steps
    nn::LSTMCell cell(10, 20);

    cell.to(device);
    auto input = Variable(randn({5, 10}, DType::Float32, device), true);

    auto [h1, c1] = cell.forward(input, Variable{}, Variable{});
    auto [h2, c2] = cell.forward(input, h1, c1);
    auto [h3, c3] = cell.forward(input, h2, c2);

    // Each step should produce outputs (shapes should be consistent)
    EXPECT_EQ(h3.shape()[0], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h3.shape()[1], 20) << "Failed on " << device.to_string();
}

// forward_with_precomputed_ih previously only fixed the c_new/h_new ops at
// the bottom, but still did `gates_ih + gates_hh.tensor()` (raw Tensor add,
// strips grad_fn from gates_hh) and `chunk(gates_tensor, ...)` (raw chunk),
// severing the grad chain back to hx/cx/weight_hh_. Rewritten to use
// Variable-level add + autograd-aware slice for the gate split. This test
// locks in the recurrent-grad invariant.
TEST_P(LSTMCellTestFixture, PrecomputedIhPropagatesHiddenCellGradient) {
    nn::LSTMCell cell(10, 20);
    cell.to(device);

    // 4*hidden = 80 wide for LSTM gates.
    auto gates_ih = randn({5, 80}, DType::Float32, device);
    Variable hx(randn({5, 20}, DType::Float32, device), /*requires_grad=*/true);
    Variable cx(randn({5, 20}, DType::Float32, device), /*requires_grad=*/true);

    auto [h_new, c_new] = cell.forward_with_precomputed_ih(gates_ih, hx, cx);
    auto loss = tenzor::sum(h_new) + tenzor::sum(c_new);
    loss.backward();

    ASSERT_TRUE(hx.has_grad()) << "hx missing grad on " << device.to_string();
    ASSERT_TRUE(cx.has_grad()) << "cx missing grad on " << device.to_string();

    auto check_nonzero = [&](const Variable& v, const char* name) {
        auto g = v.grad().value().to(Device::cpu()).contiguous();
        const float* gp = g.data<float>();
        float max_abs = 0.0f;
        for (int64_t i = 0; i < g.numel(); ++i) {
            EXPECT_FALSE(std::isnan(gp[i])) << name << " grad NaN";
            max_abs = std::max(max_abs, std::abs(gp[i]));
        }
        EXPECT_GT(max_abs, 0.0f)
            << name << " grad identically zero — LSTMCell recurrent-side autograd severed on "
            << device.to_string();
    };
    check_nonzero(hx, "hx");
    check_nonzero(cx, "cx");
}

INSTANTIATE_BACKEND_TESTS(LSTMCellTestFixture);

// ============================================================================
// LSTM Tests
// ============================================================================

class LSTMTestFixture : public BackendTest {};

TEST_P(LSTMTestFixture, BasicForward) {
    // Test basic forward pass
    nn::LSTM lstm(10, 20, 1);

    lstm.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);  // (seq_len, batch, features)

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape().size(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();  // seq_len
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string(); // hidden_size

    EXPECT_EQ(h_n.shape().size(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(h_n.shape()[0], 1) << "Failed on " << device.to_string();  // num_layers
    EXPECT_EQ(h_n.shape()[1], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(h_n.shape()[2], 20) << "Failed on " << device.to_string(); // hidden_size

    EXPECT_EQ(c_n.shape().size(), 3) << "Failed on " << device.to_string();
    EXPECT_EQ(c_n.shape()[0], 1) << "Failed on " << device.to_string();
    EXPECT_EQ(c_n.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(c_n.shape()[2], 20) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, MultiLayer) {
    // Test multi-layer LSTM
    nn::LSTM lstm(10, 20, 3);

    lstm.to(device);  // 3 layers
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string();

    EXPECT_EQ(h_n.shape()[0], 3) << "Failed on " << device.to_string();  // 3 layers
    EXPECT_EQ(c_n.shape()[0], 3) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, BatchFirst) {
    // Test with batch_first=true
    nn::LSTM lstm(10, 20, 1, true, true);

    lstm.to(device);  // batch_first=true
    auto input = Variable(randn({5, 7, 10}, DType::Float32, device), true);  // (batch, seq_len, features)

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(output.shape()[1], 7) << "Failed on " << device.to_string();  // seq_len
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string(); // hidden_size
}

TEST_P(LSTMTestFixture, Bidirectional) {
    // Test bidirectional LSTM
    nn::LSTM lstm(10, 20, 1, true, false, 0.0, true);

    lstm.to(device);  // bidirectional=true
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 40) << "Failed on " << device.to_string(); // hidden_size * 2

    EXPECT_EQ(h_n.shape()[0], 2) << "Failed on " << device.to_string();  // num_layers * num_directions
    EXPECT_EQ(h_n.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(h_n.shape()[2], 20) << "Failed on " << device.to_string();

    EXPECT_EQ(c_n.shape()[0], 2) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, BidirectionalMultiLayer) {
    // Test bidirectional multi-layer LSTM
    nn::LSTM lstm(10, 20, 2, true, false, 0.0, true);

    lstm.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape()[2], 40) << "Failed on " << device.to_string(); // hidden_size * 2
    EXPECT_EQ(h_n.shape()[0], 4) << "Failed on " << device.to_string();     // num_layers * num_directions
    EXPECT_EQ(c_n.shape()[0], 4) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, WithDropout) {
    // Test with dropout between layers
    nn::LSTM lstm(10, 20, 3, true, false, 0.5);

    lstm.to(device);  // 50% dropout
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[1], 5) << "Failed on " << device.to_string();
    EXPECT_EQ(output.shape()[2], 20) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, InitialStates) {
    // Test with provided initial states
    nn::LSTM lstm(10, 20, 2);

    lstm.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);
    auto h0 = Variable(randn({2, 5, 20}, DType::Float32, device), true);
    auto c0 = Variable(randn({2, 5, 20}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {h0, c0});
    auto [h_n, c_n] = states;

    EXPECT_EQ(output.shape()[0], 7) << "Failed on " << device.to_string();
    EXPECT_EQ(h_n.shape()[0], 2) << "Failed on " << device.to_string();
    EXPECT_EQ(c_n.shape()[0], 2) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, SequenceLengthVariation) {
    // Test with different sequence lengths
    nn::LSTM lstm(10, 20);

    lstm.to(device);

    // Short sequence
    auto input1 = Variable(randn({3, 5, 10}, DType::Float32, device), true);
    auto [output1, states1] = lstm.forward(input1, {Variable{}, Variable{}});
    EXPECT_EQ(output1.shape()[0], 3) << "Failed on " << device.to_string();

    // Long sequence
    auto input2 = Variable(randn({50, 5, 10}, DType::Float32, device), true);
    auto [output2, states2] = lstm.forward(input2, {Variable{}, Variable{}});
    EXPECT_EQ(output2.shape()[0], 50) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, BatchSizeVariation) {
    // Test with different batch sizes
    nn::LSTM lstm(10, 20);

    lstm.to(device);

    // Small batch
    auto input1 = Variable(randn({7, 2, 10}, DType::Float32, device), true);
    auto [output1, states1] = lstm.forward(input1, {Variable{}, Variable{}});
    EXPECT_EQ(output1.shape()[1], 2) << "Failed on " << device.to_string();

    // Large batch
    auto input2 = Variable(randn({7, 32, 10}, DType::Float32, device), true);
    auto [output2, states2] = lstm.forward(input2, {Variable{}, Variable{}});
    EXPECT_EQ(output2.shape()[1], 32) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, SingleTimestep) {
    // Test with single timestep
    nn::LSTM lstm(10, 20);

    lstm.to(device);
    auto input = Variable(randn({1, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 1) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, OutputConsistency) {
    // Test that output is deterministic
    nn::LSTM lstm(10, 20);

    lstm.to(device);
    auto input = Variable(ones({7, 5, 10}, DType::Float32, device), true);

    auto [output1, states1] = lstm.forward(input, {Variable{}, Variable{}});
    auto [output2, states2] = lstm.forward(input, {Variable{}, Variable{}});

    // Shapes should match
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(output1.shape()[i], output2.shape()[i]) << "Failed on " << device.to_string();
    }
}

TEST_P(LSTMTestFixture, LearnableInitialStateGradientFlow) {
    // The fused cuDNN-training fast path (CUDA, Float32, unidirectional,
    // no projection) used to build next_functions/input_variables without
    // h0/c0 at all, and CudnnLSTMTrainBackward::backward() dropped
    // grad_hx/grad_cx from its returned gradient vector -- a user-supplied
    // learnable initial hidden/cell state silently received a zero
    // gradient instead of the correct one. Other backends exercise a
    // different (already-correct) autograd path, so this is a genuine
    // cross-backend regression check, not a CUDA-only assertion.
    nn::LSTM lstm(10, 20, 1);
    lstm.to(device);

    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);
    auto h0 = Variable(randn({1, 5, 20}, DType::Float32, device), true);
    auto c0 = Variable(randn({1, 5, 20}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {h0, c0});
    auto [h_n, c_n] = states;

    auto loss = sum(output) + sum(h_n) + sum(c_n);
    loss.backward();

    ASSERT_TRUE(h0.grad().has_value()) << "h0 gradient missing on " << device.to_string();
    ASSERT_TRUE(c0.grad().has_value()) << "c0 gradient missing on " << device.to_string();

    auto h0_grad_cpu = h0.grad()->to(Device::cpu()).to(DType::Float32).contiguous();
    auto c0_grad_cpu = c0.grad()->to(Device::cpu()).to(DType::Float32).contiguous();
    float h0_grad_sum = 0.0f, c0_grad_sum = 0.0f;
    const float* hp = h0_grad_cpu.data<float>();
    const float* cp = c0_grad_cpu.data<float>();
    for (int64_t i = 0; i < h0_grad_cpu.numel(); ++i) h0_grad_sum += std::abs(hp[i]);
    for (int64_t i = 0; i < c0_grad_cpu.numel(); ++i) c0_grad_sum += std::abs(cp[i]);

    EXPECT_GT(h0_grad_sum, 0.0f) << "h0 gradient is all-zero on " << device.to_string();
    EXPECT_GT(c0_grad_sum, 0.0f) << "c0 gradient is all-zero on " << device.to_string();
}

// Hard requirement: the CPU fused training path (Task 5, see lstm.cpp's
// "FUSED CPU TRAINING PATH" block) must be numerically indistinguishable
// from the existing per-timestep autograd path for the same input. This
// test runs the SAME nn::LSTM twice with the SAME weights/inputs: once with
// the fused path enabled (default), once with TENZOR_DISABLE_FUSED_LSTM_TRAIN=1
// forcing the per-timestep loop, and compares output/h_n/c_n and every
// parameter gradient.
TEST_P(LSTMTestFixture, FusedCpuTrainingMatchesPerTimestepPath) {
    if (device.type != Device::Type::CPU) {
        GTEST_SKIP() << "This test targets the CPU-specific fused training path.";
    }

    const int64_t input_size = 6, hidden_size = 8, seq_len = 5, batch = 3;

    auto run = [&](bool disable_fused) -> std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> {
        if (disable_fused) {
            setenv("TENZOR_DISABLE_FUSED_LSTM_TRAIN", "1", 1);
        } else {
            unsetenv("TENZOR_DISABLE_FUSED_LSTM_TRAIN");
        }

        nn::LSTM lstm(input_size, hidden_size, 1);
        lstm.to(device);
        lstm.train();

        // Deterministic weights so both runs start identical.
        for (auto& [name, param] : lstm.named_parameters()) {
            auto shp = param->tensor().shape();
            const DType pdt = param->tensor().dtype();
            auto p = zeros(std::vector<int64_t>(shp.begin(), shp.end()), DType::Float32, Device::cpu());
            float* data = p.data<float>();
            for (int64_t i = 0; i < p.numel(); ++i) {
                data[i] = 0.01f * static_cast<float>((i * 13 + static_cast<int64_t>(name.size())) % 97) / 97.0f - 0.05f;
            }
            param->set_data_view(p.to(pdt).to(device));
        }

        auto input_cpu = zeros({seq_len, batch, input_size}, DType::Float32, Device::cpu());
        {
            float* d = input_cpu.data<float>();
            for (int64_t i = 0; i < input_cpu.numel(); ++i) d[i] = 0.02f * static_cast<float>((i * 7) % 53) / 53.0f - 0.1f;
        }
        auto h0_cpu = zeros({1, batch, hidden_size}, DType::Float32, Device::cpu());
        auto c0_cpu = zeros({1, batch, hidden_size}, DType::Float32, Device::cpu());
        {
            float* d = h0_cpu.data<float>();
            for (int64_t i = 0; i < h0_cpu.numel(); ++i) d[i] = 0.03f * static_cast<float>((i * 5) % 41) / 41.0f - 0.06f;
        }
        {
            float* d = c0_cpu.data<float>();
            for (int64_t i = 0; i < c0_cpu.numel(); ++i) d[i] = 0.03f * static_cast<float>((i * 3) % 41) / 41.0f - 0.06f;
        }

        auto input = Variable(input_cpu.to(device), true);
        auto h0 = Variable(h0_cpu.to(device), true);
        auto c0 = Variable(c0_cpu.to(device), true);

        auto [output, states] = lstm.forward(input, {h0, c0});
        auto [h_n, c_n] = states;
        auto loss = sum(output) + sum(h_n) + sum(c_n);
        loss.backward();

        auto to_cpu = [](const Tensor& t) { return t.to(Device::cpu()).to(DType::Float32).contiguous(); };
        Tensor grad_input = to_cpu(*input.grad());
        auto named = lstm.named_parameters();
        auto grad_by_name = [&](const std::string& want) -> Tensor {
            for (auto& [name, param] : named) {
                if (name == want) {
                    EXPECT_TRUE(param->grad().has_value()) << want << " has no gradient";
                    return to_cpu(*param->grad());
                }
            }
            ADD_FAILURE() << "parameter not found: " << want;
            return zeros({1}, DType::Float32, Device::cpu());
        };
        Tensor grad_W_ih = grad_by_name("forward_cell_0.weight_ih.weight");
        Tensor grad_W_hh = grad_by_name("forward_cell_0.weight_hh.weight");

        return {to_cpu(output.tensor()), to_cpu(h_n.tensor()), to_cpu(c_n.tensor()),
                grad_input, grad_W_ih, grad_W_hh, to_cpu(*h0.grad())};
    };

    auto [out_fused, hn_fused, cn_fused, gin_fused, gwih_fused, gwhh_fused, gh0_fused] = run(/*disable_fused=*/false);
    auto [out_slow, hn_slow, cn_slow, gin_slow, gwih_slow, gwhh_slow, gh0_slow] = run(/*disable_fused=*/true);
    unsetenv("TENZOR_DISABLE_FUSED_LSTM_TRAIN");

    auto expect_close = [](const Tensor& a, const Tensor& b, const char* label) {
        ASSERT_EQ(a.numel(), b.numel()) << label << " size mismatch";
        const float* pa = a.data<float>();
        const float* pb = b.data<float>();
        for (int64_t i = 0; i < a.numel(); ++i) {
            EXPECT_NEAR(pa[i], pb[i], 1e-3f) << label << " diverged at index " << i;
        }
    };

    expect_close(out_fused, out_slow, "output");
    expect_close(hn_fused, hn_slow, "h_n");
    expect_close(cn_fused, cn_slow, "c_n");
    expect_close(gin_fused, gin_slow, "grad_input");
    expect_close(gwih_fused, gwih_slow, "grad_W_ih");
    expect_close(gwhh_fused, gwhh_slow, "grad_W_hh");
    expect_close(gh0_fused, gh0_slow, "grad_h0");
}

// Same idea, multi-layer: confirms the per-layer chaining in the fused path
// (Task 5) produces the same result as the per-timestep path across layers.
TEST_P(LSTMTestFixture, FusedCpuTrainingMatchesPerTimestepPathMultiLayer) {
    if (device.type != Device::Type::CPU) {
        GTEST_SKIP() << "This test targets the CPU-specific fused training path.";
    }

    const int64_t input_size = 4, hidden_size = 5, seq_len = 4, batch = 2, num_layers = 3;

    auto run = [&](bool disable_fused) -> std::pair<Tensor, Tensor> {
        if (disable_fused) {
            setenv("TENZOR_DISABLE_FUSED_LSTM_TRAIN", "1", 1);
        } else {
            unsetenv("TENZOR_DISABLE_FUSED_LSTM_TRAIN");
        }

        nn::LSTM lstm(input_size, hidden_size, num_layers);
        lstm.to(device);
        lstm.train();
        for (auto& [name, param] : lstm.named_parameters()) {
            auto shp = param->tensor().shape();
            const DType pdt = param->tensor().dtype();
            auto p = zeros(std::vector<int64_t>(shp.begin(), shp.end()), DType::Float32, Device::cpu());
            float* data = p.data<float>();
            for (int64_t i = 0; i < p.numel(); ++i) {
                data[i] = 0.01f * static_cast<float>((i * 11 + static_cast<int64_t>(name.size()) * 3) % 89) / 89.0f - 0.045f;
            }
            param->set_data_view(p.to(pdt).to(device));
        }

        auto input_cpu = zeros({seq_len, batch, input_size}, DType::Float32, Device::cpu());
        float* d = input_cpu.data<float>();
        for (int64_t i = 0; i < input_cpu.numel(); ++i) d[i] = 0.02f * static_cast<float>((i * 9) % 47) / 47.0f - 0.09f;

        auto input = Variable(input_cpu.to(device), true);
        auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
        auto [h_n, c_n] = states;
        auto loss = sum(output) + sum(h_n) + sum(c_n);
        loss.backward();

        auto to_cpu = [](const Tensor& t) { return t.to(Device::cpu()).to(DType::Float32).contiguous(); };
        return {to_cpu(output.tensor()), to_cpu(*input.grad())};
    };

    auto [out_fused, gin_fused] = run(false);
    auto [out_slow, gin_slow] = run(true);
    unsetenv("TENZOR_DISABLE_FUSED_LSTM_TRAIN");

    ASSERT_EQ(out_fused.numel(), out_slow.numel());
    for (int64_t i = 0; i < out_fused.numel(); ++i) {
        EXPECT_NEAR(out_fused.data<float>()[i], out_slow.data<float>()[i], 1e-3f) << "output diverged at " << i;
    }
    ASSERT_EQ(gin_fused.numel(), gin_slow.numel());
    for (int64_t i = 0; i < gin_fused.numel(); ++i) {
        EXPECT_NEAR(gin_fused.data<float>()[i], gin_slow.data<float>()[i], 1e-3f) << "grad_input diverged at " << i;
    }
}

// The fused fast path (Float32, eval(), no_grad, unidirectional, no
// projection -- the standard fast-path-eligible config) used to ignore
// `lengths` entirely, processing the full padded seq_len for every batch
// row instead of falling through to the lengths-aware per-timestep loop.
// Row 1 here is padded to seq_len=6 but has a true length of 3; its output
// for timesteps 0..2 and its h_n/c_n must match an independent run of the
// same 3-timestep sequence with no padding at all -- if the fused path were
// still ignoring lengths, the padding (garbage/zero timesteps 3..5) would
// contaminate row 1's h_n/c_n and this would fail.
TEST_P(LSTMTestFixture, FusedFastPathRespectsLengths) {
    nn::LSTM lstm(4, 6, 1);
    lstm.to(device);
    lstm.eval();

    const int64_t seq_len = 6, true_len = 3, feat = 4, hidden = 6;

    auto padded_cpu = randn({seq_len, 2, feat}, DType::Float32, Device::cpu());
    // Row 1 (batch index 1)'s timesteps beyond true_len are padding -- give
    // them a distinct, easily-wrong-if-leaked value.
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

    // Unpadded reference: row 1's true_len-length sequence run entirely on
    // its own, no padding, no lengths argument.
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
    auto [padded_out, padded_states] = lstm.forward(
        padded_in, {Variable{}, Variable{}}, lengths_cpu);
    auto [padded_h, padded_c] = padded_states;

    Variable row1_in(row1_unpadded_cpu.to(device), false);
    auto [row1_out, row1_states] = lstm.forward(row1_in, {Variable{}, Variable{}}, Tensor{});
    auto [row1_h, row1_c] = row1_states;

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

TEST_P(LSTMTestFixture, GradientFlow) {
    // Test that gradients can flow through LSTM
    nn::LSTM lstm(10, 20);

    lstm.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    // Check that output requires grad
    EXPECT_TRUE(output.requires_grad()) << "Failed on " << device.to_string();

    // Sum for scalar output
    auto loss = Variable(sum(output.tensor(), std::nullopt, false), true);

    // Backward should not throw
    EXPECT_NO_THROW({
        loss.backward();
    }) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, TrainingMode) {
    // Test training/eval mode switching
    nn::LSTM lstm(10, 20, 2, true, false, 0.5);

    lstm.to(device);

    EXPECT_TRUE(lstm.is_training()) << "Failed on " << device.to_string();

    lstm.eval();
    EXPECT_FALSE(lstm.is_training()) << "Failed on " << device.to_string();

    lstm.train();
    EXPECT_TRUE(lstm.is_training()) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, ParameterCount) {
    // Test parameter counting
    nn::LSTM lstm(10, 20, 2);

    lstm.to(device);
    auto params = lstm.parameters();

    // Each layer has 2 combined linear layers (ih and hh) for all 4 gates
    // weight_ih: weight + bias = 2 params
    // weight_hh: weight only = 1 param
    // Layer 0: 3 params
    // Layer 1: 3 params
    // Total: 6 parameters
    EXPECT_EQ(params.size(), 6) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, LargeHidden) {
    // Test with large hidden size
    nn::LSTM lstm(10, 512);

    lstm.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[2], 512) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, VeryDeepNetwork) {
    // Test with many layers
    nn::LSTM lstm(10, 20, 5);

    lstm.to(device);
    auto input = Variable(randn({7, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    EXPECT_EQ(h_n.shape()[0], 5) << "Failed on " << device.to_string();  // 5 layers
    EXPECT_EQ(c_n.shape()[0], 5) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, LongSequence) {
    // Test with very long sequence
    nn::LSTM lstm(10, 20);

    lstm.to(device);
    auto input = Variable(randn({100, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 100) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, InvalidNumLayers) {
    // Test that invalid num_layers throws
    EXPECT_THROW({
        nn::LSTM lstm(10, 20, 0);

        lstm.to(device);
    }, std::invalid_argument) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, InvalidDropout) {
    // Test that invalid dropout throws
    EXPECT_THROW({
        nn::LSTM lstm(10, 20, 2, true, false, 1.5);

        lstm.to(device);
    }, std::invalid_argument) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, CellStateMemory) {
    // Test that cell state carries information across timesteps
    nn::LSTM lstm(10, 20);

    lstm.to(device);
    auto input = Variable(randn({10, 5, 10}, DType::Float32, device), true);

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});
    auto [h_n, c_n] = states;

    // Cell state should not be zero (it carries memory)
    auto c_cpu = c_n.tensor().to(Device::cpu());
    auto c_data = c_cpu.data<float>();
    bool has_nonzero = false;
    for (int64_t i = 0; i < c_cpu.numel(); ++i) {
        if (std::abs(c_data[i]) > 1e-6) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Failed on " << device.to_string();
}

TEST_P(LSTMTestFixture, BatchFirstBidirectional) {
    // Test combination of batch_first and bidirectional
    nn::LSTM lstm(10, 20, 1, true, true, 0.0, true);

    lstm.to(device);
    auto input = Variable(randn({5, 7, 10}, DType::Float32, device), true);  // (batch, seq, features)

    auto [output, states] = lstm.forward(input, {Variable{}, Variable{}});

    EXPECT_EQ(output.shape()[0], 5) << "Failed on " << device.to_string();  // batch
    EXPECT_EQ(output.shape()[1], 7) << "Failed on " << device.to_string();  // seq_len
    EXPECT_EQ(output.shape()[2], 40) << "Failed on " << device.to_string(); // hidden_size * 2
}

INSTANTIATE_BACKEND_TESTS(LSTMTestFixture);
