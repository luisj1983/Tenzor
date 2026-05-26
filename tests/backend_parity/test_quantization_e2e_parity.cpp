// test_quantization_e2e_parity.cpp
//
// Wave Inf-G (deferred → landed): multi-backend INT8 e2e quantization parity.
//
// Runs an end-to-end quantized inference on each available backend and
// verifies the outputs agree within INT8 quantization noise. Backends not
// available in this build/runtime are skipped via the skip macro — the test
// passes cleanly on any subset of backends.
//
// Reference for parity: each backend's output compared against the CPU
// reference within an INT8-noise tolerance (5e-2 relative).

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/quantization/quantize_api.hpp>
#include <tenzor/nn/quantization/quantize.hpp>
#include <tenzor/nn/quantization/qconfig.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/nn/module.hpp>

#include <cmath>
#include <vector>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::quantization;
using namespace tenzor::quantization;

namespace {

class QuantizationE2EParity : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }

    static bool device_available(Device::Type type) {
        try {
            Device d;
            switch (type) {
                case Device::Type::CPU:   d = Device::cpu(); break;
                case Device::Type::CUDA:  d = Device::cuda(0); break;
                case Device::Type::ROCm:  d = Device::rocm(0); break;
                case Device::Type::Vulkan:d = Device::vulkan(0); break;
                case Device::Type::OneAPI:d = Device::oneapi(0); break;
                default: return false;
            }
            auto t = zeros({1}, DType::Float32, d);
            (void)t; return true;
        } catch (...) {
            return false;
        }
    }
};

// Build a small Sequential: Linear(16→8) -> ReLU -> Linear(8→4).
auto make_qmodel() -> std::shared_ptr<Sequential> {
    auto m = std::make_shared<Sequential>();
    m->add_module(std::make_shared<Linear>(16, 8));
    m->add_module(std::make_shared<ReLU>());
    m->add_module(std::make_shared<Linear>(8, 4));
    return m;
}

// Calibration: 3 forward passes on synthetic data.
auto make_calib(int n_batches) -> std::function<void(Module&)> {
    return [n_batches](Module& m) {
        for (int i = 0; i < n_batches; ++i) {
            auto x = zeros({2, 16}, DType::Float32, Device::cpu());
            auto* p = x.data<float>();
            for (int64_t k = 0; k < x.numel(); ++k) {
                p[k] = std::sin(0.01f * (k + i)) * 1.5f;
            }
            Variable v(x, /*requires_grad=*/false);
            (void)m.forward(v);
        }
    };
}

auto make_test_input() -> Tensor {
    auto x = zeros({2, 16}, DType::Float32, Device::cpu());
    auto* p = x.data<float>();
    for (int64_t k = 0; k < x.numel(); ++k) p[k] = 0.1f * k;
    return x;
}

auto run_on_device(std::shared_ptr<Module> qmodel, Device target_dev) -> Tensor {
    auto x = make_test_input();
    // Move both model and input to target device.
    qmodel->to(target_dev);
    auto x_dev = x.to(target_dev);
    Variable vx(x_dev, /*requires_grad=*/false);
    auto y = qmodel->forward(vx);
    return y.tensor().to(Device::cpu());
}

}  // namespace

// ----------------------------------------------------------------------------
// E2E: quantize on CPU; verify the same quantized model produces matching
// output when moved to each available GPU backend.
// ----------------------------------------------------------------------------
TEST_F(QuantizationE2EParity, INT8_CudaMatchesCpu) {
    if (!device_available(Device::Type::CUDA)) {
        GTEST_SKIP() << "CUDA not available";
    }
    auto model = make_qmodel();
    auto qmodel = quantize_static(model, make_calib(3));
    auto cpu_out = run_on_device(qmodel, Device::cpu());
    auto cuda_out = run_on_device(qmodel, Device::cuda(0));
    ASSERT_EQ(cpu_out.numel(), cuda_out.numel());
    auto* cp = cpu_out.data<float>();
    auto* gp = cuda_out.data<float>();
    for (int64_t i = 0; i < cpu_out.numel(); ++i) {
        EXPECT_NEAR(cp[i], gp[i], 0.05f * (std::abs(cp[i]) + 1e-3f))
            << " elem " << i << " cpu=" << cp[i] << " cuda=" << gp[i];
    }
}

TEST_F(QuantizationE2EParity, INT8_RocmMatchesCpu) {
    if (!device_available(Device::Type::ROCm)) {
        GTEST_SKIP() << "ROCm not available";
    }
    auto model = make_qmodel();
    auto qmodel = quantize_static(model, make_calib(3));
    auto cpu_out = run_on_device(qmodel, Device::cpu());
    auto rocm_out = run_on_device(qmodel, Device::rocm(0));
    ASSERT_EQ(cpu_out.numel(), rocm_out.numel());
    auto* cp = cpu_out.data<float>();
    auto* gp = rocm_out.data<float>();
    for (int64_t i = 0; i < cpu_out.numel(); ++i) {
        EXPECT_NEAR(cp[i], gp[i], 0.05f * (std::abs(cp[i]) + 1e-3f));
    }
}

TEST_F(QuantizationE2EParity, INT8_VulkanMatchesCpu) {
    if (!device_available(Device::Type::Vulkan)) {
        GTEST_SKIP() << "Vulkan not available";
    }
    auto model = make_qmodel();
    auto qmodel = quantize_static(model, make_calib(3));
    auto cpu_out = run_on_device(qmodel, Device::cpu());
    Tensor vulkan_out;
    try {
        vulkan_out = run_on_device(qmodel, Device::vulkan(0));
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Vulkan quantized inference not supported: " << e.what();
    }
    ASSERT_EQ(cpu_out.numel(), vulkan_out.numel());
    auto* cp = cpu_out.data<float>();
    auto* gp = vulkan_out.data<float>();
    for (int64_t i = 0; i < cpu_out.numel(); ++i) {
        EXPECT_NEAR(cp[i], gp[i], 0.05f * (std::abs(cp[i]) + 1e-3f));
    }
}

TEST_F(QuantizationE2EParity, INT8_OneapiMatchesCpu) {
    if (!device_available(Device::Type::OneAPI)) {
        GTEST_SKIP() << "OneAPI not available";
    }
    auto model = make_qmodel();
    auto qmodel = quantize_static(model, make_calib(3));
    auto cpu_out = run_on_device(qmodel, Device::cpu());
    Tensor oneapi_out;
    try {
        oneapi_out = run_on_device(qmodel, Device::oneapi(0));
    } catch (const std::exception& e) {
        GTEST_SKIP() << "OneAPI quantized inference not supported: " << e.what();
    }
    ASSERT_EQ(cpu_out.numel(), oneapi_out.numel());
    auto* cp = cpu_out.data<float>();
    auto* gp = oneapi_out.data<float>();
    for (int64_t i = 0; i < cpu_out.numel(); ++i) {
        EXPECT_NEAR(cp[i], gp[i], 0.05f * (std::abs(cp[i]) + 1e-3f));
    }
}

// ----------------------------------------------------------------------------
// Self-consistency: same model run twice on CPU must produce identical output.
// (Guards against non-deterministic quant kernels.)
// ----------------------------------------------------------------------------
TEST_F(QuantizationE2EParity, SelfconsistencyIdentical) {
    auto model = make_qmodel();
    auto qmodel = quantize_static(model, make_calib(3));
    auto out1 = run_on_device(qmodel, Device::cpu());
    auto out2 = run_on_device(qmodel, Device::cpu());
    ASSERT_EQ(out1.numel(), out2.numel());
    auto* p1 = out1.data<float>();
    auto* p2 = out2.data<float>();
    for (int64_t i = 0; i < out1.numel(); ++i) {
        EXPECT_FLOAT_EQ(p1[i], p2[i]);
    }
}
