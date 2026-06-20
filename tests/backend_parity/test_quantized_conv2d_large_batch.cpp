/**
 * @file test_quantized_conv2d_large_batch.cpp
 * @brief Backend parity for quantized conv2d when batch*out_channels > 65535.
 *
 * The CUDA quantized_conv2d kernel folds the (batch, out_channel) pair onto
 * gridDim.z. CUDA's hardware maximum for gridDim.z is 65535 on every compute
 * capability, so any launch with batch*out_channels above that cap would fail
 * with cudaErrorInvalidConfiguration unless the launcher caps grid_z and the
 * kernel grid-strides over the flattened (b, oc) index to cover the remainder.
 *
 * This file exercises exactly that regime (batch=64, out_channels=1100 =>
 * 70400 > 65535) and compares the GPU result element-for-element against the
 * CPU reference. Without the grid-dim-overflow fix the CUDA path either throws
 * on launch or, if grid_z were merely truncated without grid-striding, leaves
 * the (b, oc) pairs past 65535 unwritten (garbage / zero output).
 *
 * The op is driven directly through dispatch<OpId::QuantizedConv2d> rather than
 * the QuantizedConv2d nn module, because the module's forward_quantized always
 * runs on CPU; the dispatch path is what routes to each backend's real kernel.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

namespace {

// Deterministic int8 fill in [-amp, amp] so quantized accumulation is stable
// and reproducible across backends/runs.
Tensor make_int8(const std::vector<int64_t>& shape, int amp, uint32_t salt) {
    auto t = zeros(shape, DType::Int8, Device::cpu());
    auto* p = t.data<int8_t>();
    const int span = 2 * amp + 1;
    for (int64_t i = 0; i < t.numel(); ++i) {
        uint32_t bits = static_cast<uint32_t>(i) * 2654435761u + salt * 40503u;
        int v = static_cast<int>(bits % static_cast<uint32_t>(span)) - amp;
        p[i] = static_cast<int8_t>(v);
    }
    return t;
}

Tensor make_bias_f32(int64_t out_channels) {
    auto t = zeros({out_channels}, DType::Float32, Device::cpu());
    auto* p = t.data<float>();
    for (int64_t i = 0; i < out_channels; ++i) {
        p[i] = 0.01f * static_cast<float>((i % 7) - 3);
    }
    return t;
}

OpAttributes make_attrs() {
    OpAttributes attrs;
    attrs.set(AttrKey::Stride, static_cast<int64_t>(1));
    attrs.set(AttrKey::Padding, static_cast<int64_t>(0));
    attrs.set(AttrKey::Dilation, static_cast<int64_t>(1));
    attrs.set(AttrKey::Groups, static_cast<int64_t>(1));
    // Per-tensor symmetric INT8: zero points are 0, scales are small positive.
    attrs.set(AttrKey::InputScale, static_cast<double>(0.05));
    attrs.set(AttrKey::WeightScaleQ, static_cast<double>(0.02));
    attrs.set(AttrKey::InputZeroPoint, static_cast<int64_t>(0));
    attrs.set(AttrKey::WeightZeroPoint, static_cast<int64_t>(0));
    return attrs;
}

}  // namespace

class QuantizedConv2dLargeBatch : public BackendTest {};

// batch * out_channels = 64 * 1100 = 70400, which exceeds the 65535 gridDim.z
// hardware cap on CUDA. 1x1 input and 1x1 kernel keep the spatial work tiny so
// the test runs fast while still hammering the (b, oc) grid-stride path that
// the overflow fix added.
TEST_P(QuantizedConv2dLargeBatch, GridZOverflow_MatchesCpu) {
    const int64_t batch = 64;
    const int64_t in_channels = 4;
    const int64_t out_channels = 1100;
    const int64_t h_in = 1, w_in = 1;
    const int64_t kh = 1, kw = 1;

    ASSERT_GT(batch * out_channels, 65535)
        << "test must exercise the gridDim.z overflow regime";

    auto input = make_int8({batch, in_channels, h_in, w_in}, /*amp=*/8, /*salt=*/1);
    auto weight = make_int8({out_channels, in_channels, kh, kw}, /*amp=*/6, /*salt=*/2);
    auto bias = make_bias_f32(out_channels);
    auto attrs = make_attrs();

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("quantized conv2d large-batch parity");

    // CPU reference. A throw here is a real bug — let it propagate.
    Tensor ref;
    {
        std::vector<Tensor> inputs = {input, weight, bias};
        ref = dispatch(OpId::QuantizedConv2d, inputs, attrs)[0];
    }
    ASSERT_EQ(ref.numel(), batch * out_channels);  // h_out=w_out=1

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto input_dev = input.to(backends[i]);
            auto weight_dev = weight.to(backends[i]);
            auto bias_dev = bias.to(backends[i]);
            std::vector<Tensor> inputs = {input_dev, weight_dev, bias_dev};
            Tensor out = dispatch(OpId::QuantizedConv2d, inputs, attrs)[0];
            backends[i].synchronize();

            SCOPED_TRACE(std::string("QuantizedConv2d large-batch on ")
                         + backend_name(backends[i]));
            // INT8 conv: CPU and GPU requantize with slightly different rounding
            // of the final float-scale multiply, so a quantization-noise tolerance
            // is required (matching the repo convention in
            // test_quantized_kernel_parity.cpp / test_quantization_e2e_parity.cpp:
            // ~5% relative with a ~0.1-0.2 absolute floor). A real grid-z overflow
            // regression would drop/corrupt whole outputs (errors of full output
            // magnitude), which this tolerance still catches.
            EXPECT_TENSORS_CLOSE(ref, out.to(Device::cpu()), 5e-2f, 2e-1f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "QuantizedConv2d large-batch failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

// Companion case with non-trivial spatial extent so the grid-stride over
// (b, oc) is combined with a real (h_out, w_out) tile grid, confirming the
// fix does not assume single-pixel output.
TEST_P(QuantizedConv2dLargeBatch, GridZOverflow_Spatial_MatchesCpu) {
    const int64_t batch = 72;
    const int64_t in_channels = 3;
    const int64_t out_channels = 1024;  // 72 * 1024 = 73728 > 65535
    const int64_t h_in = 5, w_in = 5;
    const int64_t kh = 3, kw = 3;       // valid conv -> h_out=w_out=3

    ASSERT_GT(batch * out_channels, 65535);

    auto input = make_int8({batch, in_channels, h_in, w_in}, /*amp=*/5, /*salt=*/3);
    auto weight = make_int8({out_channels, in_channels, kh, kw}, /*amp=*/4, /*salt=*/4);
    auto bias = make_bias_f32(out_channels);
    auto attrs = make_attrs();

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("quantized conv2d large-batch spatial parity");

    Tensor ref;
    {
        std::vector<Tensor> inputs = {input, weight, bias};
        ref = dispatch(OpId::QuantizedConv2d, inputs, attrs)[0];
    }
    const int64_t h_out = 3, w_out = 3;
    ASSERT_EQ(ref.numel(), batch * out_channels * h_out * w_out);

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto input_dev = input.to(backends[i]);
            auto weight_dev = weight.to(backends[i]);
            auto bias_dev = bias.to(backends[i]);
            std::vector<Tensor> inputs = {input_dev, weight_dev, bias_dev};
            Tensor out = dispatch(OpId::QuantizedConv2d, inputs, attrs)[0];
            backends[i].synchronize();

            SCOPED_TRACE(std::string("QuantizedConv2d large-batch spatial on ")
                         + backend_name(backends[i]));
            // INT8 quantization-noise tolerance (see note in the non-spatial case
            // above); still large enough to catch a grid-z overflow regression.
            EXPECT_TENSORS_CLOSE(ref, out.to(Device::cpu()), 5e-2f, 2e-1f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "QuantizedConv2d large-batch spatial failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }
}

INSTANTIATE_BACKEND_TESTS(QuantizedConv2dLargeBatch);

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
    int result = RUN_ALL_TESTS();
    try { tenzor::finalize(); } catch (...) {}
    return result;
}
