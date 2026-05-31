/**
 * @file test_quantization_s20.cpp
 * @brief S20 quantization fixes: HistogramObserver re-bin, QuantizedConv1d
 *        real-INT8 path, QuantStub real Q/DQ + STE backward.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/quantization/observer.hpp>
#include <tenzor/nn/quantization/quantized_layers.hpp>
#include <tenzor/nn/quantization/quantize.hpp>
#include <tenzor/nn/quantization/fake_quantize.hpp>

#include "../backend_test_fixture.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace tenzor;
using namespace tenzor::nn::quantization;

namespace {

class QuantizationS20Test : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }

    static int64_t histogram_total(const HistogramObserver& obs) {
        auto [edges, counts] = obs.get_histogram();
        (void)edges;
        int64_t total = 0;
        for (auto c : counts) total += c;
        return total;
    }

    // Helper: build a [N, C, L] tensor with values linearly spanning [lo, hi].
    // Data is filled on the CPU then moved to the target device.
    Tensor make_ramp(std::vector<int64_t> shape, float lo, float hi) {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        std::vector<float> data(static_cast<size_t>(n));
        if (n == 1) {
            data[0] = 0.5f * (lo + hi);
        } else {
            for (int64_t i = 0; i < n; ++i) {
                float t = static_cast<float>(i) / static_cast<float>(n - 1);
                data[static_cast<size_t>(i)] = lo + t * (hi - lo);
            }
        }
        Tensor cpu_t(shape, DType::Float32, Device::cpu());
        std::copy(data.begin(), data.end(),
                  cpu_t.data<float>());
        return cpu_t.to(device);
    }

    static float max_abs_error(const Tensor& a, const Tensor& b) {
        auto ac = a.to(Device::cpu());
        auto bc = b.to(Device::cpu());
        const float* ad = ac.data<float>();
        const float* bd = bc.data<float>();
        float err = 0.0f;
        for (size_t i = 0; i < ac.numel(); ++i) {
            err = std::max(err, std::abs(ad[i] - bd[i]));
        }
        return err;
    }
};

// ----------------------------------------------------------------------------
// Fix 1: HistogramObserver re-bin on range expansion
// ----------------------------------------------------------------------------

TEST_P(QuantizationS20Test, HistogramObserverRebinsOnRangeExpansion) {
    HistogramObserver obs(/*num_bins=*/10);

    auto x1 = make_ramp({100}, 0.0f, 1.0f);
    obs.observe(x1);
    const int64_t count_after_first = histogram_total(obs);
    EXPECT_EQ(count_after_first, 100)
        << "Initial observation should account for every input element";

    auto x2 = make_ramp({60}, -1.0f, 2.0f);
    obs.observe(x2);

    // Re-bin must preserve the total count exactly: every old count survives
    // the resize, plus every new value is binned into the wider range.
    const int64_t count_after_second = histogram_total(obs);
    EXPECT_EQ(count_after_second, 100 + 60)
        << "Re-bin must preserve previous counts AND include new values";

    // Inspect the histogram edges: every bin midpoint must now lie inside the
    // (slightly expanded) [-1, 2] range.
    auto [edges, counts] = obs.get_histogram();
    ASSERT_EQ(static_cast<int64_t>(counts.size()), 10);
    ASSERT_EQ(static_cast<int64_t>(edges.size()), 11);
    EXPECT_LE(edges.front(), -1.0f);
    EXPECT_GE(edges.back(),   2.0f);

    // Bin widths should be uniform (within float epsilon).
    float w0 = edges[1] - edges[0];
    for (size_t i = 1; i + 1 < edges.size(); ++i) {
        EXPECT_NEAR(edges[i + 1] - edges[i], w0, 1e-5f);
    }
}

TEST_P(QuantizationS20Test, HistogramObserverStreamingNonStationary) {
    HistogramObserver obs(/*num_bins=*/16);

    struct Batch { std::vector<int64_t> shape; float lo, hi; };
    std::vector<Batch> batches = {
        {{50}, 0.0f, 0.5f},
        {{50}, 0.5f, 1.0f},
        {{50}, -0.5f, 0.0f},
        {{50}, 1.0f, 1.5f},
    };

    int64_t expected_total = 0;
    for (const auto& b : batches) {
        auto t = make_ramp(b.shape, b.lo, b.hi);
        obs.observe(t);
        int64_t batch_size = 1;
        for (auto d : b.shape) batch_size *= d;
        expected_total += batch_size;
    }

    auto [edges, counts] = obs.get_histogram();
    int64_t total = std::accumulate(counts.begin(), counts.end(),
                                    static_cast<int64_t>(0));
    EXPECT_EQ(total, expected_total)
        << "Streaming histogram must keep every input value across "
           "non-stationary range expansions";

    EXPECT_LE(edges.front(), -0.5f) << "min must cover the most negative batch";
    EXPECT_GE(edges.back(),   1.5f) << "max must cover the most positive batch";
}

// ----------------------------------------------------------------------------
// Fix 2: QuantizedConv1d real INT8 vs FP32 reference
// ----------------------------------------------------------------------------

TEST_P(QuantizationS20Test, QuantizedConv1dRealInt8MatchesFp32Reference) {
    const int64_t batch = 2;
    const int64_t in_c  = 4;
    const int64_t out_c = 3;
    const int64_t k     = 3;
    const int64_t l_in  = 16;
    const int64_t stride = 1;
    const int64_t padding = 1;

    // Random-ish but deterministic FP32 weight in a modest range, so INT8
    // quantisation error stays well-bounded. Built on CPU, moved to device.
    Tensor fp_weight;
    {
        Tensor cpu_w({out_c, in_c, k}, DType::Float32, Device::cpu());
        float* w = cpu_w.data<float>();
        for (int64_t i = 0; i < cpu_w.numel(); ++i) {
            float v = std::sin(static_cast<float>(i) * 0.37f) * 0.5f;
            w[static_cast<size_t>(i)] = v;
        }
        fp_weight = cpu_w.to(device);
    }

    Tensor fp_input;
    {
        Tensor cpu_x({batch, in_c, l_in}, DType::Float32, Device::cpu());
        float* x = cpu_x.data<float>();
        for (int64_t i = 0; i < cpu_x.numel(); ++i) {
            x[static_cast<size_t>(i)] =
                std::cos(static_cast<float>(i) * 0.21f) * 0.7f;
        }
        fp_input = cpu_x.to(device);
    }

    // Quantise weight per-tensor symmetric INT8.
    auto q_weight = quantize_per_tensor_symmetric(fp_weight);

    // Build QuantizedConv1d with that weight.
    QuantizedConv1d qconv(in_c, out_c, k, stride, padding,
                          /*dilation=*/1, /*groups=*/1, q_weight.params());
    qconv.set_weight(q_weight);

    // FP32 reference: compute conv1d directly with the dequantised weight
    // (so both paths see the same effective weight values; only activation
    // quantisation should contribute error). Reference computed host-side on
    // CPU copies of the effective weight and input.
    Tensor effective_weight = q_weight.dequantize();

    const int64_t l_out = (l_in + 2 * padding - (k - 1) - 1) / stride + 1;
    Tensor ref_out({batch, out_c, l_out}, DType::Float32, Device::cpu());
    {
        auto x_cpu = fp_input.to(Device::cpu());
        auto w_cpu = effective_weight.to(Device::cpu());
        const float* x = x_cpu.data<float>();
        const float* w = w_cpu.data<float>();
        float* y = ref_out.data<float>();
        for (int64_t n = 0; n < batch; ++n) {
            for (int64_t oc = 0; oc < out_c; ++oc) {
                for (int64_t ol = 0; ol < l_out; ++ol) {
                    float acc = 0.0f;
                    for (int64_t ic = 0; ic < in_c; ++ic) {
                        for (int64_t kk = 0; kk < k; ++kk) {
                            int64_t il = ol * stride - padding + kk;
                            if (il >= 0 && il < l_in) {
                                int64_t xi = (n * in_c + ic) * l_in + il;
                                int64_t wi = (oc * in_c + ic) * k + kk;
                                acc += x[xi] * w[wi];
                            }
                        }
                    }
                    int64_t yi = (n * out_c + oc) * l_out + ol;
                    y[yi] = acc;
                }
            }
        }
    }

    // Run quantised conv.
    auto q_input = quantize_per_tensor_symmetric(fp_input);
    Tensor q_out = qconv.forward_quantized(q_input);

    ASSERT_EQ(q_out.shape().size(), 3u);
    ASSERT_EQ(q_out.shape()[0], batch);
    ASSERT_EQ(q_out.shape()[1], out_c);
    ASSERT_EQ(q_out.shape()[2], l_in);  // stride=1, padding=1, k=3

    // Quantisation isn't exact: bound the error by ~5% of the FP32 output range.
    float ref_min = std::numeric_limits<float>::infinity();
    float ref_max = -std::numeric_limits<float>::infinity();
    {
        auto refc = ref_out.to(Device::cpu());
        const float* r = refc.data<float>();
        for (size_t i = 0; i < refc.numel(); ++i) {
            ref_min = std::min(ref_min, r[i]);
            ref_max = std::max(ref_max, r[i]);
        }
    }
    const float ref_range = std::max(ref_max - ref_min, 1e-6f);
    const float tol = 0.05f * ref_range;
    const float err = max_abs_error(q_out, ref_out);
    EXPECT_LT(err, tol)
        << "Real-INT8 Conv1d output should agree with FP32 reference within "
           "quantisation tolerance: err=" << err << " tol=" << tol
        << " (ref_range=" << ref_range << ")";
}

// ----------------------------------------------------------------------------
// Fix 3: QuantStub real Q/DQ + STE backward
// ----------------------------------------------------------------------------

TEST_P(QuantizationS20Test, QuantStubObserveModeIsPassthrough) {
    // Build QuantStub with placeholder qparams (overwritten by calibration).
    Tensor scale  = full({1}, 1.0f, DType::Float32, device);
    Tensor zp     = zeros({1}, DType::Int32, device);
    QuantizationParams qp(scale, zp, QuantDType::INT8,
                          QuantizationScheme::PerTensorSymmetric);
    QuantStub stub(qp);

    stub.set_calibrating(true);
    EXPECT_TRUE(stub.is_calibrating());

    auto x = make_ramp({64}, -1.0f, 1.0f);
    Variable v_in(x, /*requires_grad=*/false);
    Variable v_out = stub.forward(v_in);

    // Observe-mode passthrough: bit-for-bit equality with the input.
    const float err = max_abs_error(x, v_out.tensor());
    EXPECT_EQ(err, 0.0f) << "Calibration mode must be exact passthrough";

    // Observer should have data we can finalise into qparams.
    ASSERT_NE(stub.observer(), nullptr);
    EXPECT_TRUE(stub.observer()->has_data());
    stub.update_qparams_from_observer();
}

TEST_P(QuantizationS20Test, QuantStubQuantizeModeInjectsNoise) {
    // Set scale=2/255 so INT8 round-trip is lossy but bounded.
    const float scale_v = 2.0f / 255.0f;
    Tensor scale  = full({1}, scale_v, DType::Float32, device);
    Tensor zp     = zeros({1}, DType::Int32, device);
    QuantizationParams qp(scale, zp, QuantDType::INT8,
                          QuantizationScheme::PerTensorSymmetric);
    QuantStub stub(qp);

    EXPECT_FALSE(stub.is_calibrating());

    // Use values that DON'T land exactly on quantisation grid points so the
    // round-trip is provably lossy. Built on CPU then moved to device.
    Tensor x;
    {
        Tensor cpu_x({128}, DType::Float32, Device::cpu());
        float* d = cpu_x.data<float>();
        for (int64_t i = 0; i < cpu_x.numel(); ++i) {
            float t = static_cast<float>(i) / static_cast<float>(cpu_x.numel() - 1);
            d[i] = -0.9f + 1.8f * t + 0.001f;  // small offset to break grid alignment
        }
        x = cpu_x.to(device);
    }
    Variable v_in(x, /*requires_grad=*/false);
    Variable v_out = stub.forward(v_in);

    const float diff = max_abs_error(x, v_out.tensor());
    EXPECT_GT(diff, 0.0f)
        << "Quantize-mode QuantStub must actually inject Q->DQ noise, "
           "not pass through";
    // Bounded by ~scale (rounding error per element).
    EXPECT_LT(diff, 2.0f * scale_v)
        << "Q->DQ noise should be bounded by ~scale";
}

// QAT path: enable_grad + requires_grad on input. The STE backward passes
// grad_output through to input.grad unchanged (clipped to the representable
// range). Previously this produced no input.grad because
// fake_quantize_with_grad set grad_fn but never wired next_functions /
// input_variables, so the engine had no input edge to accumulate into.
TEST_P(QuantizationS20Test, QuantStubSteBackwardPassesGradThrough) {
    // QAT path: enable_grad + requires_grad on input. The STE backward
    // should pass grad_output through to input.grad unchanged (clipped to
    // the representable range, which our small input is well inside).
    const float scale_v = 0.01f;
    Tensor scale  = full({1}, scale_v, DType::Float32, device);
    Tensor zp     = zeros({1}, DType::Int64, device);
    QuantizationParams qp(scale, zp, QuantDType::INT8,
                          QuantizationScheme::PerTensorSymmetric);
    QuantStub stub(qp);

    // Stay well within +-127*scale = +-1.27 so all activations are in range
    // and the FakeQuantize clip-STE preserves the full gradient. Built on CPU
    // then moved to device.
    Tensor x;
    {
        Tensor cpu_x({8}, DType::Float32, Device::cpu());
        float* d = cpu_x.data<float>();
        for (int64_t i = 0; i < cpu_x.numel(); ++i) {
            d[i] = -0.4f + 0.1f * static_cast<float>(i);
        }
        x = cpu_x.to(device);
    }
    Variable v_in(x, /*requires_grad=*/true);

    set_grad_enabled(true);
    Variable v_out = stub.forward(v_in);
    ASSERT_TRUE(v_out.requires_grad())
        << "QAT path must return a Variable with requires_grad=true";

    // Build a fixed grad_output and call backward.
    std::vector<int64_t> x_shape(x.shape().begin(), x.shape().end());
    Tensor go = ones(x_shape, DType::Float32, device);
    v_out.backward(go);

    ASSERT_TRUE(v_in.grad().has_value());
    Tensor g = *v_in.grad();
    ASSERT_EQ(g.numel(), x.numel());

    // STE: grad_input == grad_output for every in-range activation.
    auto g_cpu = g.to(Device::cpu());
    auto go_cpu = go.to(Device::cpu());
    const float* gd = g_cpu.data<float>();
    const float* od = go_cpu.data<float>();
    for (int64_t i = 0; i < g_cpu.numel(); ++i) {
        EXPECT_NEAR(gd[i], od[i], 1e-5f)
            << "STE backward must pass grad_output through (i=" << i << ")";
    }
}

// Per-channel FakeQuantize STE grad flow (release audit). Per-channel schemes
// previously fell through to a leaf Variable with no grad_fn, silently zeroing
// the input gradient — and QATHelper defaults WEIGHTS to per-channel, so the
// primary QAT path mis-trained. The per-channel STE must now route grad back to
// the input (pass-through for in-range elements, per channel).
TEST_P(QuantizationS20Test, FakeQuantizePerChannelSteGradFlows) {
    const int64_t C = 3, K = 4;  // weight [C, K], per-channel along axis 0

    // Distinct per-channel scales. Built on CPU then moved to device.
    Tensor scale;
    {
        Tensor cpu_scale({C}, DType::Float32, Device::cpu());
        float* s = cpu_scale.data<float>();
        s[0] = 0.01f; s[1] = 0.02f; s[2] = 0.005f;
        scale = cpu_scale.to(device);
    }
    Tensor zp = zeros({C}, DType::Int32, device);
    QuantizationParams qp(scale, zp, QuantDType::INT8,
                          QuantizationScheme::PerChannelSymmetric, /*axis=*/0);

    FakeQuantize fq(QuantDType::INT8, QuantizationScheme::PerChannelSymmetric,
                    /*learnable=*/false, /*observer_enabled=*/false, /*axis=*/0);
    fq.set_qparams(qp);
    fq.enable_fake_quant(true);

    // Keep |x| < 0.5 so every element is within each channel's representable
    // range (min range is channel 2: +-0.635), giving full STE pass-through.
    // Built on CPU then moved to device.
    Tensor x;
    {
        Tensor cpu_x({C, K}, DType::Float32, Device::cpu());
        float* d = cpu_x.data<float>();
        for (int64_t i = 0; i < cpu_x.numel(); ++i) {
            d[i] = -0.3f + 0.05f * static_cast<float>(i % 7);
        }
        x = cpu_x.to(device);
    }
    Variable v_in(x, /*requires_grad=*/true);

    set_grad_enabled(true);
    Variable v_out = fq.forward(v_in);
    ASSERT_TRUE(v_out.requires_grad())
        << "per-channel QAT path must return requires_grad=true";

    std::vector<int64_t> x_shape(x.shape().begin(), x.shape().end());
    Tensor go = ones(x_shape, DType::Float32, device);
    v_out.backward(go);

    ASSERT_TRUE(v_in.grad().has_value())
        << "per-channel STE must produce an input gradient (graph not severed)";
    Tensor g = *v_in.grad();
    ASSERT_EQ(g.numel(), x.numel());

    auto g_cpu = g.to(Device::cpu());
    auto go_cpu = go.to(Device::cpu());
    const float* gd = g_cpu.data<float>();
    const float* od = go_cpu.data<float>();
    for (int64_t i = 0; i < g_cpu.numel(); ++i) {
        EXPECT_NEAR(gd[i], od[i], 1e-5f)
            << "per-channel STE must pass grad_output through in-range (i=" << i << ")";
    }
}

INSTANTIATE_BACKEND_TESTS(QuantizationS20Test);

}  // namespace
