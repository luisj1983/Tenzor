/**
 * @file test_anchor_generator_multidtype.cpp
 * @brief Multi-backend × multi-dtype tests for AnchorGenerator output.
 *
 * AnchorGenerator::generate() returns Float32 boxes regardless of any
 * input dtype — it has no input dtype parameter, only a target Device.
 * This file pins:
 *   - Boxes on a target backend match the CPU reference value-for-value
 *     (the generator is deterministic for fixed sizes/ratios/stride).
 *   - Casting the output to Float64/Float16/BFloat16 preserves the
 *     box geometry within the dtype's representable precision.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/detection/anchors.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn::detection;
using namespace tenzor::testing;

class AnchorGeneratorMultiDTypeTest : public MultiBackendDTypeTest {};

namespace {
// Compute max absolute difference between two Float32-promoted tensors.
double max_abs_diff_f32(const Tensor& a, const Tensor& b) {
    auto a_cpu = a.contiguous().to(Device::cpu()).to(DType::Float32);
    auto b_cpu = b.contiguous().to(Device::cpu()).to(DType::Float32);
    EXPECT_EQ(a_cpu.numel(), b_cpu.numel());
    const float* ap = a_cpu.data<float>();
    const float* bp = b_cpu.data<float>();
    double m = 0.0;
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        m = std::max(m, std::abs(static_cast<double>(ap[i] - bp[i])));
    }
    return m;
}
}  // namespace

// Generate on the test device, compare against the same generator's CPU
// output. AnchorGenerator should produce identical boxes regardless of
// device — generate() is just an arithmetic pattern over (h, w, k).
TEST_P(AnchorGeneratorMultiDTypeTest, GenerateMatchesCPUReference) {
    AnchorGenerator gen({32.0f, 64.0f, 128.0f}, {0.5f, 1.0f, 2.0f});
    auto cpu_boxes = gen.generate(/*H=*/4, /*W=*/4, /*stride=*/16,
                                   Device::cpu());
    auto dev_boxes = gen.generate(/*H=*/4, /*W=*/4, /*stride=*/16, device_);
    EXPECT_EQ(cpu_boxes.numel(), dev_boxes.numel());
    EXPECT_LT(max_abs_diff_f32(cpu_boxes, dev_boxes), 1e-4);
}

// Cast the boxes to the parameterised dtype and verify the cast preserves
// the geometry within the dtype's representable precision.
TEST_P(AnchorGeneratorMultiDTypeTest, CastPreservesBoxes) {
    AnchorGenerator gen({32.0f, 128.0f}, {1.0f});
    auto boxes = gen.generate(/*H=*/2, /*W=*/2, /*stride=*/16, device_);
    auto casted = boxes.to(dtype());
    auto round_trip = casted.to(DType::Float32);

    // Tolerance: Float16 gives ~1e-3 worth of relative error on values
    // around 100; BFloat16 gives ~1e-1; Float32/Float64 are essentially
    // exact at this scale.
    double tol = 0.0;
    if (dtype() == DType::BFloat16)            tol = 1.0;
    else if (dtype() == DType::Float16)        tol = 0.5;
    else                                        tol = 1e-4;
    EXPECT_LT(max_abs_diff_f32(boxes, round_trip), tol)
        << "AnchorGenerator output cast to " << dtype_name(dtype())
        << " on " << device_.to_string() << " lost geometry beyond tolerance.";
}

INSTANTIATE_MULTI_BACKEND_ALL_DTYPE_TESTS(AnchorGeneratorMultiDTypeTest);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }
    int rc = RUN_ALL_TESTS();
    try { tenzor::finalize(); } catch (...) {}
    return rc;
}
