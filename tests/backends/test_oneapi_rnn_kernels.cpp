#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;

/**
 * @brief Test suite for OneAPI RNN kernels (LSTM, GRU)
 *
 * Mirrors tests/backends/test_rocm_rnn_kernels.cpp (see FINDING 3 in
 * findings.txt: ROCm had a dedicated RNN-cell kernel test file with no
 * OneAPI equivalent). Verifies LSTM/GRU forward output against a CPU
 * reference with identical weights.
 */

namespace {

bool is_oneapi_available() {
    try {
        auto device = Device::oneapi(0);
        auto t = zeros({1}, DType::Float32, device);
        (void)t;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void copy_params(const nn::Module& src, nn::Module& dst) {
    auto src_params = const_cast<nn::Module&>(src).parameters();
    auto dst_params = dst.parameters();
    ASSERT_EQ(src_params.size(), dst_params.size())
        << "Parameter count mismatch between source and destination modules";
    for (size_t i = 0; i < src_params.size(); ++i) {
        dst_params[i]->tensor() = src_params[i]->tensor().clone();
    }
}

float max_abs_diff(const Tensor& a, const Tensor& b) {
    auto a_cpu = a.to(Device::cpu()).contiguous();
    auto b_cpu = b.to(Device::cpu()).contiguous();
    EXPECT_EQ(a_cpu.numel(), b_cpu.numel());
    const float* ap = a_cpu.data<float>();
    const float* bp = b_cpu.data<float>();
    float m = 0.0f;
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        m = std::max(m, std::fabs(ap[i] - bp[i]));
    }
    return m;
}

}  // namespace

class OneAPIRNNKernelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
        if (!is_oneapi_available()) {
            GTEST_SKIP() << "OneAPI device not available, skipping OneAPI RNN kernel tests";
        }
    }
};

TEST_F(OneAPIRNNKernelsTest, LSTM_Forward_Basic) {
    const int64_t batch = 4, seq = 16, in_size = 32, hidden = 32;

    nn::LSTM cpu_layer(in_size, hidden, 1, true, true, 0.0, false);
    cpu_layer.eval();

    nn::LSTM oneapi_layer(in_size, hidden, 1, true, true, 0.0, false);
    copy_params(cpu_layer, oneapi_layer);
    oneapi_layer.eval();
    oneapi_layer.to(Device::oneapi(0));

    auto x_cpu = randn({batch, seq, in_size}, DType::Float32, Device::cpu());
    auto x_oneapi = x_cpu.to(Device::oneapi(0));

    auto cpu_out = cpu_layer.forward_impl(Variable(x_cpu, false)).tensor();
    auto oneapi_out = oneapi_layer.forward_impl(Variable(x_oneapi, false)).tensor();

    ASSERT_EQ(cpu_out.shape().size(), oneapi_out.shape().size());
    for (size_t i = 0; i < cpu_out.shape().size(); ++i) {
        EXPECT_EQ(cpu_out.shape()[i], oneapi_out.shape()[i]);
    }
    EXPECT_LT(max_abs_diff(cpu_out, oneapi_out), 1e-3f)
        << "OneAPI LSTM forward output diverges from CPU reference with identical weights";
}

TEST_F(OneAPIRNNKernelsTest, GRU_Forward_Basic) {
    const int64_t batch = 4, seq = 16, in_size = 32, hidden = 32;

    nn::GRU cpu_layer(in_size, hidden, 1, true, true, 0.0, false);
    cpu_layer.eval();

    nn::GRU oneapi_layer(in_size, hidden, 1, true, true, 0.0, false);
    copy_params(cpu_layer, oneapi_layer);
    oneapi_layer.eval();
    oneapi_layer.to(Device::oneapi(0));

    auto x_cpu = randn({batch, seq, in_size}, DType::Float32, Device::cpu());
    auto x_oneapi = x_cpu.to(Device::oneapi(0));

    auto cpu_out = cpu_layer.forward_impl(Variable(x_cpu, false)).tensor();
    auto oneapi_out = oneapi_layer.forward_impl(Variable(x_oneapi, false)).tensor();

    ASSERT_EQ(cpu_out.shape().size(), oneapi_out.shape().size());
    for (size_t i = 0; i < cpu_out.shape().size(); ++i) {
        EXPECT_EQ(cpu_out.shape()[i], oneapi_out.shape()[i]);
    }
    EXPECT_LT(max_abs_diff(cpu_out, oneapi_out), 1e-3f)
        << "OneAPI GRU forward output diverges from CPU reference with identical weights";
}

TEST_F(OneAPIRNNKernelsTest, LSTM_Shapes) {
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> configs = {
        {1, 4, 8, 16}, {8, 6, 16, 32}, {16, 5, 32, 64}, {32, 3, 64, 128}
    };

    for (const auto& [batch, seq, in_size, hidden] : configs) {
        nn::LSTM cpu_layer(in_size, hidden, 1, true, true, 0.0, false);
        cpu_layer.eval();

        nn::LSTM oneapi_layer(in_size, hidden, 1, true, true, 0.0, false);
        copy_params(cpu_layer, oneapi_layer);
        oneapi_layer.eval();
        oneapi_layer.to(Device::oneapi(0));

        auto x_cpu = randn({batch, seq, in_size}, DType::Float32, Device::cpu());
        auto x_oneapi = x_cpu.to(Device::oneapi(0));

        auto cpu_out = cpu_layer.forward_impl(Variable(x_cpu, false)).tensor();
        auto oneapi_out = oneapi_layer.forward_impl(Variable(x_oneapi, false)).tensor();

        EXPECT_EQ(oneapi_out.shape()[0], batch);
        EXPECT_EQ(oneapi_out.shape()[1], seq);
        EXPECT_EQ(oneapi_out.shape()[2], hidden);
        EXPECT_LT(max_abs_diff(cpu_out, oneapi_out), 1e-3f)
            << "batch=" << batch << " seq=" << seq << " in=" << in_size << " hidden=" << hidden;
    }
}

TEST_F(OneAPIRNNKernelsTest, GRU_Shapes) {
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> configs = {
        {1, 4, 8, 16}, {8, 6, 16, 32}, {16, 5, 32, 64}, {32, 3, 64, 128}
    };

    for (const auto& [batch, seq, in_size, hidden] : configs) {
        nn::GRU cpu_layer(in_size, hidden, 1, true, true, 0.0, false);
        cpu_layer.eval();

        nn::GRU oneapi_layer(in_size, hidden, 1, true, true, 0.0, false);
        copy_params(cpu_layer, oneapi_layer);
        oneapi_layer.eval();
        oneapi_layer.to(Device::oneapi(0));

        auto x_cpu = randn({batch, seq, in_size}, DType::Float32, Device::cpu());
        auto x_oneapi = x_cpu.to(Device::oneapi(0));

        auto cpu_out = cpu_layer.forward_impl(Variable(x_cpu, false)).tensor();
        auto oneapi_out = oneapi_layer.forward_impl(Variable(x_oneapi, false)).tensor();

        EXPECT_EQ(oneapi_out.shape()[0], batch);
        EXPECT_EQ(oneapi_out.shape()[1], seq);
        EXPECT_EQ(oneapi_out.shape()[2], hidden);
        EXPECT_LT(max_abs_diff(cpu_out, oneapi_out), 1e-3f)
            << "batch=" << batch << " seq=" << seq << " in=" << in_size << " hidden=" << hidden;
    }
}

TEST_F(OneAPIRNNKernelsTest, Float64_Support) {
    auto device = Device::oneapi(0);

    int64_t batch_size = 2;
    int64_t hidden_size = 4;

    auto gates_f64 = ones({batch_size, 4 * hidden_size}, DType::Float64, device);
    auto c_prev_f64 = zeros({batch_size, hidden_size}, DType::Float64, device);

    EXPECT_EQ(gates_f64.dtype(), DType::Float64);
}
