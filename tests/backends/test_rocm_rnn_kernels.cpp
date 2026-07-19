#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <hip/hip_runtime.h>
#include <cmath>

using namespace tenzor;

/**
 * @brief Test suite for ROCm RNN kernels (LSTM, GRU)
 *
 * These tests verify LSTM/GRU forward output against a CPU reference with
 * identical weights (see FINDING 5 in findings.txt: the previous version of
 * this file only asserted bare SUCCEED()/tautological tensor shapes and never
 * exercised or checked the actual gate math). Mirrors the CPU-vs-CUDA pattern
 * in tests/unit/test_lstm_cuda_forward_regression.cpp.
 */

namespace {

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

class ROCmRNNKernelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize Tenzor
        tenzor::initialize();

        // Check if ROCm is available
        int device_count = 0;
        hipError_t error = hipGetDeviceCount(&device_count);

        if (error != hipSuccess || device_count == 0) {
            GTEST_SKIP() << "ROCm device not available, skipping ROCm RNN kernel tests";
        }
    }
};

TEST_F(ROCmRNNKernelsTest, LSTM_Forward_Basic) {
    const int64_t batch = 4, seq = 16, in_size = 32, hidden = 32;

    nn::LSTM cpu_layer(in_size, hidden, 1, true, true, 0.0, false);
    cpu_layer.eval();

    nn::LSTM rocm_layer(in_size, hidden, 1, true, true, 0.0, false);
    copy_params(cpu_layer, rocm_layer);
    rocm_layer.eval();
    rocm_layer.to(Device::rocm(0));

    auto x_cpu = randn({batch, seq, in_size}, DType::Float32, Device::cpu());
    auto x_rocm = x_cpu.to(Device::rocm(0));

    auto cpu_out = cpu_layer.forward_impl(Variable(x_cpu, false)).tensor();
    auto rocm_out = rocm_layer.forward_impl(Variable(x_rocm, false)).tensor();
    Device::rocm(0).synchronize();

    ASSERT_EQ(cpu_out.shape().size(), rocm_out.shape().size());
    for (size_t i = 0; i < cpu_out.shape().size(); ++i) {
        EXPECT_EQ(cpu_out.shape()[i], rocm_out.shape()[i]);
    }
    EXPECT_LT(max_abs_diff(cpu_out, rocm_out), 1e-3f)
        << "ROCm LSTM forward output diverges from CPU reference with identical weights";
}

TEST_F(ROCmRNNKernelsTest, GRU_Forward_Basic) {
    const int64_t batch = 4, seq = 16, in_size = 32, hidden = 32;

    nn::GRU cpu_layer(in_size, hidden, 1, true, true, 0.0, false);
    cpu_layer.eval();

    nn::GRU rocm_layer(in_size, hidden, 1, true, true, 0.0, false);
    copy_params(cpu_layer, rocm_layer);
    rocm_layer.eval();
    rocm_layer.to(Device::rocm(0));

    auto x_cpu = randn({batch, seq, in_size}, DType::Float32, Device::cpu());
    auto x_rocm = x_cpu.to(Device::rocm(0));

    auto cpu_out = cpu_layer.forward_impl(Variable(x_cpu, false)).tensor();
    auto rocm_out = rocm_layer.forward_impl(Variable(x_rocm, false)).tensor();
    Device::rocm(0).synchronize();

    ASSERT_EQ(cpu_out.shape().size(), rocm_out.shape().size());
    for (size_t i = 0; i < cpu_out.shape().size(); ++i) {
        EXPECT_EQ(cpu_out.shape()[i], rocm_out.shape()[i]);
    }
    EXPECT_LT(max_abs_diff(cpu_out, rocm_out), 1e-3f)
        << "ROCm GRU forward output diverges from CPU reference with identical weights";
}

TEST_F(ROCmRNNKernelsTest, LSTM_Shapes) {
    // Test various batch sizes and hidden sizes — actually RUNS the LSTM
    // forward pass on ROCm and checks against a CPU reference, rather than
    // just checking random-tensor shapes (the previous version of this test
    // never invoked the LSTM kernel at all).
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> configs = {
        {1, 4, 8, 16}, {8, 6, 16, 32}, {16, 5, 32, 64}, {32, 3, 64, 128}
    };

    for (const auto& [batch, seq, in_size, hidden] : configs) {
        nn::LSTM cpu_layer(in_size, hidden, 1, true, true, 0.0, false);
        cpu_layer.eval();

        nn::LSTM rocm_layer(in_size, hidden, 1, true, true, 0.0, false);
        copy_params(cpu_layer, rocm_layer);
        rocm_layer.eval();
        rocm_layer.to(Device::rocm(0));

        auto x_cpu = randn({batch, seq, in_size}, DType::Float32, Device::cpu());
        auto x_rocm = x_cpu.to(Device::rocm(0));

        auto cpu_out = cpu_layer.forward_impl(Variable(x_cpu, false)).tensor();
        auto rocm_out = rocm_layer.forward_impl(Variable(x_rocm, false)).tensor();
        Device::rocm(0).synchronize();

        EXPECT_EQ(rocm_out.shape()[0], batch);
        EXPECT_EQ(rocm_out.shape()[1], seq);
        EXPECT_EQ(rocm_out.shape()[2], hidden);
        EXPECT_LT(max_abs_diff(cpu_out, rocm_out), 1e-3f)
            << "batch=" << batch << " seq=" << seq << " in=" << in_size << " hidden=" << hidden;
    }
}

TEST_F(ROCmRNNKernelsTest, GRU_Shapes) {
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> configs = {
        {1, 4, 8, 16}, {8, 6, 16, 32}, {16, 5, 32, 64}, {32, 3, 64, 128}
    };

    for (const auto& [batch, seq, in_size, hidden] : configs) {
        nn::GRU cpu_layer(in_size, hidden, 1, true, true, 0.0, false);
        cpu_layer.eval();

        nn::GRU rocm_layer(in_size, hidden, 1, true, true, 0.0, false);
        copy_params(cpu_layer, rocm_layer);
        rocm_layer.eval();
        rocm_layer.to(Device::rocm(0));

        auto x_cpu = randn({batch, seq, in_size}, DType::Float32, Device::cpu());
        auto x_rocm = x_cpu.to(Device::rocm(0));

        auto cpu_out = cpu_layer.forward_impl(Variable(x_cpu, false)).tensor();
        auto rocm_out = rocm_layer.forward_impl(Variable(x_rocm, false)).tensor();
        Device::rocm(0).synchronize();

        EXPECT_EQ(rocm_out.shape()[0], batch);
        EXPECT_EQ(rocm_out.shape()[1], seq);
        EXPECT_EQ(rocm_out.shape()[2], hidden);
        EXPECT_LT(max_abs_diff(cpu_out, rocm_out), 1e-3f)
            << "batch=" << batch << " seq=" << seq << " in=" << in_size << " hidden=" << hidden;
    }
}

TEST_F(ROCmRNNKernelsTest, Float64_Support) {
    auto device = Device::rocm(0);

    int64_t batch_size = 2;
    int64_t hidden_size = 4;

    // Test double precision support
    auto gates_f64 = ones({batch_size, 4 * hidden_size}, DType::Float64, device);
    auto c_prev_f64 = zeros({batch_size, hidden_size}, DType::Float64, device);

    EXPECT_EQ(gates_f64.dtype(), DType::Float64);
}
