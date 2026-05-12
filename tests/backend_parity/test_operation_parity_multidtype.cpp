/**
 * @file test_operation_parity_multidtype.cpp
 * @brief Multi-dtype backend parity tests.
 *
 * Tests key operations across Float32, Float64, Float16, and BFloat16
 * on all available backends to ensure consistent numerical behavior.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class MultiDTypeParity : public BackendTest {};
// ============================================================================
// Float16 Parity
// ============================================================================

TEST_P(MultiDTypeParity, Add_Float16) {
    auto a = randn({16, 16}, DType::Float16, Device::cpu());
    auto b = randn({16, 16}, DType::Float16, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 1e-2f, 1e-3f, "Add_Float16");
}

TEST_P(MultiDTypeParity, Mul_Float16) {
    auto a = randn({16, 16}, DType::Float16, Device::cpu());
    auto b = randn({16, 16}, DType::Float16, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] * inputs[1];
    }, {a, b}, 1e-2f, 1e-3f, "Mul_Float16");
}

TEST_P(MultiDTypeParity, MatMul_Float16) {
    auto a = randn({8, 16}, DType::Float16, Device::cpu());
    auto b = randn({16, 8}, DType::Float16, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::matmul(inputs[0], inputs[1]);
    }, {a, b}, 1e-1f, 1e-2f, "MatMul_Float16");
}

TEST_P(MultiDTypeParity, Softmax_Float16) {
    auto input = randn({8, 16}, DType::Float16, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch_single(OpId::Softmax, std::span<const Tensor>(ins), attrs);
    }, {input}, 1e-2f, 1e-3f, "Softmax_Float16");
}

// ============================================================================
// BFloat16 Parity
// ============================================================================

TEST_P(MultiDTypeParity, Add_BFloat16) {
    auto a = randn({16, 16}, DType::BFloat16, Device::cpu());
    auto b = randn({16, 16}, DType::BFloat16, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 1e-1f, 1e-2f, "Add_BFloat16");
}

TEST_P(MultiDTypeParity, Mul_BFloat16) {
    auto a = randn({16, 16}, DType::BFloat16, Device::cpu());
    auto b = randn({16, 16}, DType::BFloat16, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] * inputs[1];
    }, {a, b}, 1e-1f, 1e-2f, "Mul_BFloat16");
}

TEST_P(MultiDTypeParity, MatMul_BFloat16) {
    auto a = randn({8, 16}, DType::BFloat16, Device::cpu());
    auto b = randn({16, 8}, DType::BFloat16, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::matmul(inputs[0], inputs[1]);
    }, {a, b}, 5e-1f, 1e-1f, "MatMul_BFloat16");
}

TEST_P(MultiDTypeParity, Softmax_BFloat16) {
    auto input = randn({8, 16}, DType::BFloat16, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch_single(OpId::Softmax, std::span<const Tensor>(ins), attrs);
    }, {input}, 1e-1f, 1e-2f, "Softmax_BFloat16");
}

// ============================================================================
// Float64 Parity (additional ops beyond the extended file)
// ============================================================================

TEST_P(MultiDTypeParity, Mul_Float64) {
    auto a = randn({16, 16}, DType::Float64, Device::cpu());
    auto b = randn({16, 16}, DType::Float64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] * inputs[1];
    }, {a, b}, 1e-14f, 0.0f, "Mul_Float64");
}

TEST_P(MultiDTypeParity, Exp_Float64) {
    auto input = randn({16, 16}, DType::Float64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::exp(inputs[0]);
    }, {input}, 1e-12f, 1e-14f, "Exp_Float64");
}

TEST_P(MultiDTypeParity, Sum_Float64) {
    auto input = randn({32, 32}, DType::Float64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::sum(inputs[0], 1, false);
    }, {input}, 1e-12f, 1e-14f, "Sum_Float64");
}

// ============================================================================
// Conv2d Parity (multiple dtypes)
// ============================================================================

TEST_P(MultiDTypeParity, Conv2d_Float32) {
    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());
    auto weight = randn({8, 3, 3, 3}, DType::Float32, Device::cpu());

    test_operation_parity([&weight](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Stride, int64_t(1));
        attrs.set(AttrKey::Padding, int64_t(1));
        attrs.set(AttrKey::Dilation, int64_t(1));
        attrs.set(AttrKey::Groups, int64_t(1));
        std::vector<Tensor> ins = {inputs[0], weight.to(inputs[0].device())};
        return dispatch<OpId::Conv2dForward>(ins, attrs)[0];
    // Float32 conv2d cross-backend abs diff is bounded by FMA-ordering
    // between the GPU and CPU GEMM kernels (cuDNN IMPLICIT_GEMM vs oneDNN
    // im2col+sgemm), not by any implementation bug — both backends already
    // use CUDNN_FMA_MATH/strict-FP32. PyTorch's own cross-device tests use
    // the same atol/rtol for the same reason.
    }, {input}, 1e-4f, 1e-4f, "Conv2d_Float32");
}

// ============================================================================
// BatchNorm Parity (Float32)
// ============================================================================

TEST_P(MultiDTypeParity, BatchNorm2d_Float32) {
    auto input = randn({2, 8, 4, 4}, DType::Float32, Device::cpu());
    auto weight = ones({8}, DType::Float32, Device::cpu());
    auto bias = zeros({8}, DType::Float32, Device::cpu());
    auto running_mean = zeros({8}, DType::Float32, Device::cpu());
    auto running_var = ones({8}, DType::Float32, Device::cpu());

    test_operation_parity([&weight, &bias, &running_mean, &running_var](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Eps, 1e-5);
        attrs.set(AttrKey::Momentum, 0.1);
        attrs.set(AttrKey::Training, false);
        // BatchNorm2dForwardAffine signature is (input, mean, var, gamma, beta).
        // In inference (Training=false) PyTorch maps:
        //   mean  ← running_mean, var ← running_var, gamma ← weight, beta ← bias
        std::vector<Tensor> ins = {
            inputs[0],
            running_mean.to(inputs[0].device()),
            running_var.to(inputs[0].device()),
            weight.to(inputs[0].device()),
            bias.to(inputs[0].device())
        };
        return dispatch<OpId::BatchNorm2dForwardAffine>(ins, attrs)[0];
    }, {input}, 1e-4f, 1e-6f, "BatchNorm2d_Float32");
}

// ============================================================================
// Conv2d multi-dtype
// ============================================================================

TEST_P(MultiDTypeParity, Conv2d_Float64) {
    auto input = randn({1, 3, 8, 8}, DType::Float64, Device::cpu());
    auto weight = randn({8, 3, 3, 3}, DType::Float64, Device::cpu());

    test_operation_parity([&weight](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Stride, int64_t(1));
        attrs.set(AttrKey::Padding, int64_t(1));
        attrs.set(AttrKey::Dilation, int64_t(1));
        attrs.set(AttrKey::Groups, int64_t(1));
        std::vector<Tensor> ins = {inputs[0], weight.to(inputs[0].device())};
        return dispatch<OpId::Conv2dForward>(ins, attrs)[0];
    }, {input}, 1e-10f, 1e-12f, "Conv2d_Float64");
}

// ============================================================================
// BatchNorm multi-dtype
// ============================================================================

TEST_P(MultiDTypeParity, BatchNorm2d_Float64) {
    auto input = randn({2, 8, 4, 4}, DType::Float64, Device::cpu());
    auto weight = ones({8}, DType::Float64, Device::cpu());
    auto bias = zeros({8}, DType::Float64, Device::cpu());
    auto running_mean = zeros({8}, DType::Float64, Device::cpu());
    auto running_var = ones({8}, DType::Float64, Device::cpu());

    test_operation_parity([&weight, &bias, &running_mean, &running_var](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Eps, 1e-5);
        attrs.set(AttrKey::Momentum, 0.1);
        attrs.set(AttrKey::Training, false);
        std::vector<Tensor> ins = {
            inputs[0],
            running_mean.to(inputs[0].device()),
            running_var.to(inputs[0].device()),
            weight.to(inputs[0].device()),
            bias.to(inputs[0].device())
        };
        return dispatch<OpId::BatchNorm2dForwardAffine>(ins, attrs)[0];
    }, {input}, 1e-10f, 1e-12f, "BatchNorm2d_Float64");
}

TEST_P(MultiDTypeParity, BatchNorm2d_Float16) {
    auto input = randn({2, 8, 4, 4}, DType::Float16, Device::cpu());
    auto weight = ones({8}, DType::Float16, Device::cpu());
    auto bias = zeros({8}, DType::Float16, Device::cpu());
    auto running_mean = zeros({8}, DType::Float16, Device::cpu());
    auto running_var = ones({8}, DType::Float16, Device::cpu());

    test_operation_parity([&weight, &bias, &running_mean, &running_var](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Eps, 1e-5);
        attrs.set(AttrKey::Momentum, 0.1);
        attrs.set(AttrKey::Training, false);
        std::vector<Tensor> ins = {
            inputs[0],
            running_mean.to(inputs[0].device()),
            running_var.to(inputs[0].device()),
            weight.to(inputs[0].device()),
            bias.to(inputs[0].device())
        };
        return dispatch<OpId::BatchNorm2dForwardAffine>(ins, attrs)[0];
    }, {input}, 1e-1f, 1e-2f, "BatchNorm2d_Float16");
}

INSTANTIATE_BACKEND_TESTS(MultiDTypeParity);


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    setenv("TENZOR_DISABLE_TF32", "1", 1);
    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize: " << e.what() << std::endl;
        return 1;
    }
    int result = RUN_ALL_TESTS();
    try { tenzor::finalize(); } catch (...) {}
    return result;
}
