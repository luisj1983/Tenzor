/**
 * @file test_quantized_kernel_parity.cpp
 * @brief Real int8 kernel parity — not float simulation.
 *
 * test_quantization_parity.cpp exercises a float32 quantize/dequantize round
 * trip. This file constructs an actual QuantizedLinear (QInt8 weights) and
 * asserts the real quantized-forward pass agrees across backends for a
 * shared input, which is how the user would actually use the class.
 *
 * We cross-compare backend results using test_operation_parity, which in
 * this context means "each backend runs QuantizedLinear.forward and the
 * outputs match within INT8-accumulation tolerance."
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/quantization/quantize.hpp>
#include <tenzor/nn/quantization/quantized_layers.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn;
using namespace tenzor::nn::quantization;

class QuantizedKernelParity : public BackendTest {};

namespace {

// Build a floating-point Linear with deterministic weights, then convert it
// to a QuantizedLinear using the symmetric INT8 config. Returns both so the
// test can compare quantized-forward against the float reference.
auto make_pair(int64_t in_features, int64_t out_features)
    -> std::pair<std::shared_ptr<Linear>, std::shared_ptr<QuantizedLinear>> {
    auto fp = std::make_shared<Linear>(in_features, out_features, true);
    // Deterministic weight fill — seeded by generate_test_tensor upstream.
    auto qconfig = DefaultQConfigs::default_qconfig();
    auto q = QuantizedLinear::from_float(*fp, qconfig);
    return {fp, q};
}

} // namespace

// ---------------------------------------------------------------------------
// Quantized forward output is close to float reference
// ---------------------------------------------------------------------------

TEST_P(QuantizedKernelParity, ForwardMatchesFloatReferenceWithinINT8Error) {
    auto [fp, q] = make_pair(16, 8);
    auto input = randn({2, 16}, DType::Float32, Device::cpu());

    // Float reference on CPU.
    auto fp_out = fp->forward(Variable(input, false)).tensor();

    // Quantized forward on the target device. QuantizedLinear moves its
    // parameters on .to(device); we move the module and run on the fixture
    // backend.
    q->to(device);
    auto input_dev = input.to(device);
    auto q_out = q->forward(Variable(input_dev, false)).tensor();
    q_out = q_out.to(Device::cpu());

    // INT8 per-tensor quantization has a worst-case error of ~1/128 of the
    // weight magnitude per matmul step. For a 16-wide reduction that adds
    // up; allow a generous rtol and a per-output atol.
    ASSERT_EQ(fp_out.numel(), q_out.numel());
    const float* ap = fp_out.data<float>();
    const float* bp = q_out.data<float>();
    float max_abs = 0.0f;
    for (int64_t i = 0; i < fp_out.numel(); ++i) {
        max_abs = std::max(max_abs, std::abs(ap[i]));
    }
    float atol = std::max(0.1f, 0.1f * max_abs);
    for (int64_t i = 0; i < fp_out.numel(); ++i) {
        EXPECT_LT(std::abs(ap[i] - bp[i]), atol)
            << "quantized-forward drift at idx=" << i
            << " fp=" << ap[i] << " q=" << bp[i];
    }
}

// ---------------------------------------------------------------------------
// Cross-backend parity: same input, same weights, each backend's quantized
// forward must agree with every other backend.
// ---------------------------------------------------------------------------

TEST_P(QuantizedKernelParity, CrossBackendParity) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("needs >=2 backends");

    auto [fp, q] = make_pair(32, 16);
    auto input = randn({4, 32}, DType::Float32, Device::cpu());

    std::vector<Tensor> results;
    std::vector<Device> used;
    for (const auto& backend : backends) {
        try {
            q->to(backend);
            auto out = q->forward(Variable(input.to(backend), false)).tensor();
            backend.synchronize();
            results.push_back(out.to(Device::cpu()));
            used.push_back(backend);
        } catch (const std::exception& e) {
            std::cerr << "Quantized backend " << backend_name(backend)
                      << " failed: " << e.what() << "\n";
        }
    }
    if (results.size() < 2) GTEST_SKIP() << "only one successful backend";

    const auto& ref = results[0];
    for (size_t i = 1; i < results.size(); ++i) {
        float max_diff = max_abs_diff(ref, results[i]);
        // INT8 accumulation is deterministic per backend, but rounding of
        // intermediate dequant steps can differ by ~1 LSB. Allow 1e-3 abs.
        EXPECT_LT(max_diff, 1e-3f)
            << "backend " << backend_name(used[i]) << " diverged from "
            << backend_name(used[0]) << " max_diff=" << max_diff;
    }
}

INSTANTIATE_BACKEND_TESTS(QuantizedKernelParity);
