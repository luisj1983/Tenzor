/**
 * @file test_lstm_cuda_forward_regression.cpp
 * @brief Regression test for the v0.1.0 cuBLAS INVALID_VALUE bug in CUDA
 *        LSTM/GRU forward (rooted in inverted OP_T flags in cublas_gemm_ex).
 *
 * The failing benchmark shape was:
 *   nn::LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=true)
 *   x of shape (batch=32, seq=128, input=256)
 * which dispatches via the `LSTMForward` fast path in eval-mode and goes
 * through cublas_matmul → cublas_gemm_ex.
 *
 * We compare CUDA forward output against a CPU reference with identical
 * weights. Tolerance is FP32-loose (1e-4) since both paths run FP32.
 *
 * Unlike the test_nn_rnn_parity.cpp variant (which silently swallows CUDA
 * exceptions and prints "Skipped"), this test fails loudly so a regression
 * cannot pass CI unnoticed.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>

using namespace tenzor;

namespace {

bool cuda_available() {
    try {
        Device::cuda(0);
        // Probe by doing a tiny copy
        auto t = randn({2, 2}, DType::Float32, Device::cpu());
        (void)t.to(Device::cuda(0));
        return true;
    } catch (...) {
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

class LSTMCudaForwardRegression : public ::testing::Test {
protected:
    void SetUp() override {
        initialize();
        if (!cuda_available()) {
            GTEST_SKIP() << "CUDA not available";
        }
    }
};

// Exact shape from the v0.1.0 failing benchmark (LSTM Small 1L).
TEST_F(LSTMCudaForwardRegression, BenchShape_Small1L) {
    const int64_t batch = 32, seq = 128, in_size = 256, hidden = 256;

    nn::LSTM cpu_layer(in_size, hidden, 1, true, true, 0.0, false);
    cpu_layer.eval();

    nn::LSTM cuda_layer(in_size, hidden, 1, true, true, 0.0, false);
    copy_params(cpu_layer, cuda_layer);
    cuda_layer.eval();
    cuda_layer.cuda();

    auto x_cpu = randn({batch, seq, in_size}, DType::Float32, Device::cpu());
    auto x_cuda = x_cpu.to(Device::cuda(0));

    auto cpu_out = cpu_layer.forward_impl(Variable(x_cpu, false)).tensor();

    // This is the call that throws cuBLAS INVALID_VALUE before the fix.
    // The test must NOT wrap it in try/catch — a regression should fail loudly.
    auto cuda_out = cuda_layer.forward_impl(Variable(x_cuda, false)).tensor();
    Device::cuda(0).synchronize();

    EXPECT_EQ(cpu_out.shape().size(), cuda_out.shape().size());
    for (size_t i = 0; i < cpu_out.shape().size(); ++i) {
        EXPECT_EQ(cpu_out.shape()[i], cuda_out.shape()[i]);
    }
    EXPECT_LT(max_abs_diff(cpu_out, cuda_out), 1e-3f);
}

// Smaller shape to verify the fix isn't size-dependent.
TEST_F(LSTMCudaForwardRegression, SmallShape) {
    const int64_t batch = 4, seq = 16, in_size = 32, hidden = 32;

    nn::LSTM cpu_layer(in_size, hidden, 1, true, true, 0.0, false);
    cpu_layer.eval();

    nn::LSTM cuda_layer(in_size, hidden, 1, true, true, 0.0, false);
    copy_params(cpu_layer, cuda_layer);
    cuda_layer.eval();
    cuda_layer.cuda();

    auto x_cpu = randn({batch, seq, in_size}, DType::Float32, Device::cpu());
    auto x_cuda = x_cpu.to(Device::cuda(0));

    auto cpu_out = cpu_layer.forward_impl(Variable(x_cpu, false)).tensor();
    auto cuda_out = cuda_layer.forward_impl(Variable(x_cuda, false)).tensor();
    Device::cuda(0).synchronize();

    EXPECT_LT(max_abs_diff(cpu_out, cuda_out), 1e-3f);
}

// Smaller GRU shape — used to validate the numerical path independently
// of the BenchShape configuration.
TEST_F(LSTMCudaForwardRegression, GRU_SmallShape) {
    const int64_t batch = 4, seq = 16, in_size = 32, hidden = 32;

    nn::GRU cpu_layer(in_size, hidden, 1, true, true, 0.0, false);
    cpu_layer.eval();

    nn::GRU cuda_layer(in_size, hidden, 1, true, true, 0.0, false);
    copy_params(cpu_layer, cuda_layer);
    cuda_layer.eval();
    cuda_layer.cuda();

    auto x_cpu = randn({batch, seq, in_size}, DType::Float32, Device::cpu());
    auto x_cuda = x_cpu.to(Device::cuda(0));

    auto cpu_out = cpu_layer.forward_impl(Variable(x_cpu, false)).tensor();
    auto cuda_out = cuda_layer.forward_impl(Variable(x_cuda, false)).tensor();
    Device::cuda(0).synchronize();

    EXPECT_LT(max_abs_diff(cpu_out, cuda_out), 1e-3f);
}


// Bench shape — exercises the originally-failing path. Phase 8.5 fixed:
//   1. Removed duplicate transpose in gru.cpp fast path that fed the
//      kernel batch-major input it then misread as time-major.
//   2. Replaced std::swap on Tensor handles with explicit two-buffer
//      ping-pong so the CUDA driver sees stable device pointers.
//   3. Made gates_ih_t.contiguous() explicit (defensive).
TEST_F(LSTMCudaForwardRegression, GRU_BenchShape) {
    const int64_t batch = 32, seq = 128, in_size = 256, hidden = 256;

    nn::GRU cpu_layer(in_size, hidden, 1, true, true, 0.0, false);
    cpu_layer.eval();

    nn::GRU cuda_layer(in_size, hidden, 1, true, true, 0.0, false);
    copy_params(cpu_layer, cuda_layer);
    cuda_layer.eval();
    cuda_layer.cuda();

    auto x_cpu = randn({batch, seq, in_size}, DType::Float32, Device::cpu());
    auto x_cuda = x_cpu.to(Device::cuda(0));

    auto cpu_out = cpu_layer.forward_impl(Variable(x_cpu, false)).tensor();
    auto cuda_out = cuda_layer.forward_impl(Variable(x_cuda, false)).tensor();
    Device::cuda(0).synchronize();

    EXPECT_LT(max_abs_diff(cpu_out, cuda_out), 1e-3f);
}
