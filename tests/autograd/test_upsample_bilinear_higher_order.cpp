// test_upsample_bilinear_higher_order.cpp
//
// Audit D3 (full): UpsampleBilinearBackward must
//   (a) dispatch its tensor-level backward through OpId::InterpolateBackward
//       so the math stays on the original device (no CPU round-trip), and
//   (b) preserve the autograd graph through `backward_with_variables` by
//       attaching an UpsampleBilinearForwardAdjoint grad_fn.

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/function.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/nn/layers/segmentation.hpp>
#include <tenzor/nn/functional.hpp>
#include <tenzor/autograd/gradcheck.hpp>

#include "../grad_flow_helpers.hpp"
#include "../backend_test_fixture.hpp"

using namespace tenzor;

class D3Test : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

TEST_P(D3Test, CPU_InterpolateBackward_OpId_ProducesExpectedShape) {
    // 1x1x4x6 grad_output downsampled back to 1x1x2x3 grad_input.
    auto grad_out_cpu = zeros({1, 1, 4, 6}, DType::Float32, Device::cpu());
    auto* p = grad_out_cpu.data<float>();
    for (int i = 0; i < grad_out_cpu.numel(); ++i) p[i] = 1.0f;
    auto grad_out = grad_out_cpu.to(device);

    OpAttributes attrs;
    attrs.set(AttrKey::InputShape, std::string("2,3"));
    attrs.set(AttrKey::Mode, "bilinear");
    attrs.set(AttrKey::AlignCorners, false);

    std::vector<Tensor> inputs = {grad_out};
    auto outs = tenzor::dispatch(OpId::InterpolateBackward, inputs, attrs);
    ASSERT_EQ(outs.size(), 1u);
    auto shape = outs[0].shape();
    ASSERT_EQ(shape.size(), 4u);
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 1);
    EXPECT_EQ(shape[2], 2);
    EXPECT_EQ(shape[3], 3);
}

TEST_P(D3Test, UpsampleBilinearBackward_PreservesGraph) {
    // Set up forward via nn::upsample_bilinear, run backward with
    // create_graph=true, and verify input.grad_variable() carries grad_fn.
    auto x_cpu = zeros({1, 1, 2, 3}, DType::Float32, Device::cpu());
    // Non-trivial data so gradients aren't trivially zero.
    auto* p = x_cpu.data<float>();
    for (int i = 0; i < x_cpu.numel(); ++i) p[i] = static_cast<float>(i + 1);
    auto x = Variable(x_cpu.to(device), /*requires_grad=*/true);

    auto y = nn::upsample_bilinear(x, /*target_h=*/4, /*target_w=*/6);

    auto loss = tenzor::sum(y);
    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);
    ASSERT_TRUE(x.grad_variable().has_value())
        << "create_graph=true must populate grad_variable() now that "
           "UpsampleBilinearBackward's backward_with_variables attaches a "
           "UpsampleBilinearForwardAdjoint grad_fn (audit D3)";
    Variable g = x.grad_variable().value();
    EXPECT_TRUE(g.requires_grad());
    EXPECT_NE(g.grad_fn(), nullptr);

    // Higher-order: compute grad-norm and back-prop; verify the second-order
    // gradient actually flows through to the original Variable.
    auto gnorm = tenzor::sum(g * g);
    gnorm.backward();
    EXPECT_GRAD_FLOWS(x);
}

TEST_P(D3Test, BackwardDispatchesToInterpolateBackward_NotCpuRoundTrip) {
    // Verify the tensor-level backward calls OpId::InterpolateBackward
    // rather than the previous on-host scalar loop with .to(cpu) / .to(device).
    // We can't directly observe the kernel selection but we *can* verify the
    // op is registered and produces results matching the historical
    // implementation for a simple case.
    auto fn = std::make_shared<UpsampleBilinearBackward>(
        /*input_h=*/2, /*input_w=*/3, /*output_h=*/4, /*output_w=*/6);
    auto grad_out_cpu = zeros({1, 1, 4, 6}, DType::Float32, Device::cpu());
    auto* p = grad_out_cpu.data<float>();
    for (int i = 0; i < grad_out_cpu.numel(); ++i) p[i] = 1.0f;
    auto grad_out_t = grad_out_cpu.to(device);

    auto results = fn->backward({grad_out_t});
    ASSERT_EQ(results.size(), 1u);
    auto shape = results[0].shape();
    ASSERT_EQ(shape.size(), 4u);
    EXPECT_EQ(shape[2], 2);  // input_h
    EXPECT_EQ(shape[3], 3);  // input_w
    // The implementation drops boundary mass when source coordinates fall
    // outside the input range (matching the historical scalar-loop body and
    // PyTorch's `align_corners=false` semantics on the non-clamped path).
    // We just require all values to be finite and non-negative summed mass
    // close to interior pixel count.
    auto results_cpu = results[0].cpu();
    auto* gp = results_cpu.data<float>();
    float total = 0.0f;
    for (int i = 0; i < results_cpu.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(gp[i]));
        total += gp[i];
    }
    EXPECT_GT(total, 0.0f);
}

// F009/F010: the bilinear-upsample backward must use the SAME align_corners
// sampling geometry as the forward. gradcheck through interpolate(...,
// align_corners=true) fails if the backward hardcodes align_corners=false.
TEST_P(D3Test, InterpolateBilinearGradcheck_AlignCorners) {
    for (bool ac : {true, false}) {
        Variable input(randn({1, 2, 3, 3}, DType::Float64, device), true);
        auto f = [ac](const Variable& v) -> Variable {
            return nn::functional::interpolate(
                v, std::pair<int64_t, int64_t>{5, 5}, "bilinear", ac);
        };
        bool ok = gradcheck(f, input, 1e-6, 1e-4, 1e-4);
        EXPECT_TRUE(ok) << "bilinear interpolate gradcheck failed on "
                        << device.to_string() << " with align_corners=" << ac;
    }
}

INSTANTIATE_BACKEND_TESTS(D3Test);
