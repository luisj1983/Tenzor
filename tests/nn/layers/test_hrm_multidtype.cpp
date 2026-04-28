/**
 * @file test_hrm_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for Hierarchical Reasoning Model (HRM)
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/hrm.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class HRMMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    HRMConfig make_config() {
        HRMConfig config;
        config.d_model = 64;
        config.n_heads = 4;
        config.d_feedforward = 128;
        config.n_high_cycles = 2;
        config.t_low_steps = 4;
        config.dropout = 0.0;
        config.use_post_norm = true;
        config.deep_supervision = true;
        config.use_act = false;
        config.max_seq_len = 64;
        return config;
    }
};

TEST_P(HRMMultiDTypeTest, RMSNormForwardShape) {
    RMSNorm norm(64);
    convert_model(norm);

    auto input = createInput({2, 8, 64}, false);
    auto output = norm.forward(input);

    EXPECT_EQ(output.shape().size(), 3u);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.shape()[2], 64);
}

TEST_P(HRMMultiDTypeTest, GLUForwardShape) {
    GatedLinearUnit glu(64, 128, GateType::SiLU, false);
    convert_model(glu);

    auto input = createInput({2, 8, 64}, false);
    auto output = glu.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.shape()[2], 64);
}

TEST_P(HRMMultiDTypeTest, GLUParameterCount) {
    GatedLinearUnit glu(64, 128, GateType::SiLU, false);
    convert_model(glu);

    auto params = glu.parameters();
    // gate: 64->128, up: 64->128, down: 128->64
    int64_t expected_params = 64 * 128 + 64 * 128 + 128 * 64;

    int64_t total = 0;
    for (auto& p : params) {
        total += p->tensor().numel();
    }
    EXPECT_EQ(total, expected_params);
}

TEST_P(HRMMultiDTypeTest, HRMConstruction) {
    auto config = make_config();
    EXPECT_NO_THROW({
        HRM hrm(config);
    });
}

TEST_P(HRMMultiDTypeTest, HRMForwardShape) {
    auto config = make_config();
    HRM hrm(config);
    convert_model(hrm);

    auto input = createInput({1, 16, 64}, false);
    auto output = hrm.forward(input);

    EXPECT_EQ(output.shape()[0], 1);
    EXPECT_EQ(output.shape()[2], 64);
}

TEST_P(HRMMultiDTypeTest, HRMOutputFinite) {
    auto config = make_config();
    HRM hrm(config);
    convert_model(hrm);

    auto input = createInput({1, 8, 64}, false);
    auto output = hrm.forward(input);

    auto output_f32 = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data = output_f32.data<float>();
    for (int64_t i = 0; i < output_f32.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(data[i])) << "Inf at index " << i;
    }
}

TEST_P(HRMMultiDTypeTest, HRMParametersNonEmpty) {
    auto config = make_config();
    HRM hrm(config);
    convert_model(hrm);

    auto params = hrm.parameters();
    EXPECT_FALSE(params.empty());
}

// Phase 3-followup #22 / #32: HRM uses one-step gradient training and
// intentionally detaches hidden states between iterations (see hrm.cpp:877:
// "CRITICAL: Detach hidden states for approximate gradient (O(1) memory)").
// Input gradient is therefore expected to be zero/missing BY DESIGN — only
// the final run_h_cycle's parameters get gradient. Verify that parameter
// gradients ARE populated, which is what one-step training requires.
TEST_P(HRMMultiDTypeTest, HRMBackwardWeightsUpdated) {
    // (Float16 skip removed — re-test after Vulkan RMSNormBackward fix #55)
    auto config = make_config();
    HRM hrm(config);
    convert_model(hrm);
    auto input = createInput({1, 8, 64}, /*requires_grad=*/false);
    auto output = hrm.forward(input);
    sum(output).backward();
    int populated = 0;
    for (auto& [name, param] : hrm.named_parameters()) {
        if (param->has_grad()) {
            auto g_cpu_f32 = param->grad()->to(Device::cpu()).to(DType::Float32).contiguous();
            // Reduce in user code to keep the dtype stable: max(abs(...))
            // would return whatever dtype was passed, and item<float>() then
            // throws on Float64. Iterate manually instead.
            float local_max = 0.0f;
            for (int64_t i = 0; i < g_cpu_f32.numel(); ++i) {
                local_max = std::max(local_max, std::abs(g_cpu_f32.data<float>()[i]));
            }
            if (local_max > 0.0f) ++populated;
        }
    }
    EXPECT_GT(populated, 0)
        << "no HRM parameter received gradient on " << device().to_string();
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(HRMMultiDTypeTest);
