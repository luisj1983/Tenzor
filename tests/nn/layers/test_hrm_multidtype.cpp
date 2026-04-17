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
    GatedLinearUnit glu(64, 128, true, false);
    convert_model(glu);

    auto input = createInput({2, 8, 64}, false);
    auto output = glu.forward(input);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 8);
    EXPECT_EQ(output.shape()[2], 64);
}

TEST_P(HRMMultiDTypeTest, GLUParameterCount) {
    GatedLinearUnit glu(64, 128, true, false);
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

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(HRMMultiDTypeTest);
