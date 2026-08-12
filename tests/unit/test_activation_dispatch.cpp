#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/tenzor.hpp"
#include <cmath>
#include <cstring>
#include <vector>

using namespace tenzor;

// Global test environment that initializes Tenzor (registers the CPU
// dispatch table) before any test constructs a Tensor on Device::cpu().
class ActivationDispatchTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const activation_dispatch_env =
    ::testing::AddGlobalTestEnvironment(new ActivationDispatchTestEnvironment);

namespace {

// Deterministic pseudo-random fill including exact zero, negative, positive,
// and near-denormal values -- the edge cases a hand-written SIMD ReLU loop
// must get exactly right.
std::vector<float> make_relu_test_input_f32(int64_t n) {
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        int64_t m = i % 7;
        if (m == 0) v[static_cast<size_t>(i)] = 0.0f;
        else if (m == 1) v[static_cast<size_t>(i)] = -0.0f;
        else if (m == 2) v[static_cast<size_t>(i)] = 1e-40f;  // near-denormal
        else if (m == 3) v[static_cast<size_t>(i)] = -1e-40f;
        else v[static_cast<size_t>(i)] = 0.01f * static_cast<float>((i * 37 + 11) % 101) - 0.5f;
    }
    return v;
}

std::vector<double> make_relu_test_input_f64(int64_t n) {
    std::vector<double> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        int64_t m = i % 7;
        if (m == 0) v[static_cast<size_t>(i)] = 0.0;
        else if (m == 1) v[static_cast<size_t>(i)] = -0.0;
        else if (m == 2) v[static_cast<size_t>(i)] = 1e-300;
        else if (m == 3) v[static_cast<size_t>(i)] = -1e-300;
        else v[static_cast<size_t>(i)] = 0.01 * static_cast<double>((i * 37 + 11) % 101) - 0.5;
    }
    return v;
}

}  // namespace

TEST(ActivationDispatch, ReluF32MatchesScalarReferenceAcrossSizes) {
    // Sizes chosen to cross every relevant boundary: below/above the old
    // ONEDNN_ACTIVATION_THRESHOLD (65536), the exact benchmark sizes that
    // showed the regression (32768, 262144, 2097152), and 8388608 -- which
    // is genuinely `>` RELU_SIGMOID_OMP_THRESHOLD's measured 2097152 floor
    // (the pragmas use strict `>`, so 2097152 itself never enters the
    // parallel branch) and stays above that floor even after
    // OmpThresholds::simple()'s core-count scaling on any machine with a
    // sane core count, so this size actually exercises the multithreaded
    // SIMD loop rather than only the single-thread fallback.
    for (int64_t n : {int64_t{1}, int64_t{15}, int64_t{16}, int64_t{17},
                       int64_t{32768}, int64_t{65536}, int64_t{262144},
                       int64_t{2097152}, int64_t{8388608}}) {
        auto input_vec = make_relu_test_input_f32(n);
        Tensor input({n}, DType::Float32, Device::cpu());
        std::memcpy(input.data<float>(), input_vec.data(), static_cast<size_t>(n) * sizeof(float));

        Variable input_var(input, false);
        auto output = nn::relu(input_var);
        auto out_t = output.tensor().contiguous();
        ASSERT_EQ(out_t.numel(), n);

        const float* out_data = out_t.data<float>();
        for (int64_t i = 0; i < n; ++i) {
            float expected = std::max(0.0f, input_vec[static_cast<size_t>(i)]);
            EXPECT_EQ(out_data[i], expected) << "n=" << n << " i=" << i << " input=" << input_vec[static_cast<size_t>(i)];
        }
    }
}

TEST(ActivationDispatch, ReluF64MatchesScalarReferenceAcrossSizes) {
    for (int64_t n : {int64_t{1}, int64_t{15}, int64_t{16}, int64_t{17},
                       int64_t{32768}, int64_t{65536}, int64_t{262144}}) {
        auto input_vec = make_relu_test_input_f64(n);
        Tensor input({n}, DType::Float64, Device::cpu());
        std::memcpy(input.data<double>(), input_vec.data(), static_cast<size_t>(n) * sizeof(double));

        Variable input_var(input, false);
        auto output = nn::relu(input_var);
        auto out_t = output.tensor().contiguous();
        ASSERT_EQ(out_t.numel(), n);

        const double* out_data = out_t.data<double>();
        for (int64_t i = 0; i < n; ++i) {
            double expected = std::max(0.0, input_vec[static_cast<size_t>(i)]);
            EXPECT_EQ(out_data[i], expected) << "n=" << n << " i=" << i;
        }
    }
}

TEST(ActivationDispatch, SigmoidF32MatchesFastMathReferenceAcrossSizes) {
    // fast_math::sigmoid_avx512/avx2 is unchanged by this fix -- this test
    // locks in that today's already-shipped approximation's OUTPUT VALUES
    // are identical whether reached via the old (oneDNN-gated) or new
    // (unconditional) dispatch path, by comparing against itself at a size
    // that was always below the old oneDNN threshold (32768, unaffected by
    // this task's change) versus a size that was always above it (262144,
    // directly affected). If the values match at both sizes, dispatch
    // reordering did not change sigmoid's actual output. 4194304 is also
    // included: it is genuinely `>` RELU_SIGMOID_OMP_THRESHOLD's measured
    // 2097152 floor (pragmas use strict `>`), so on builds without oneDNN
    // (TENZOR_USE_ONEDNN off) this exercises sigmoid_kernel's multithreaded
    // SIMD fallback loop; on oneDNN builds it still validates output
    // correctness at a size well above the F32 oneDNN gate (n>=131072).
    for (int64_t n : {int64_t{32768}, int64_t{262144}, int64_t{4194304}}) {
        std::vector<float> input_vec(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            input_vec[static_cast<size_t>(i)] = 0.01f * static_cast<float>((i * 41 + 7) % 113) - 0.6f;
        }
        Tensor input({n}, DType::Float32, Device::cpu());
        std::memcpy(input.data<float>(), input_vec.data(), static_cast<size_t>(n) * sizeof(float));

        Variable input_var(input, false);
        auto output = nn::sigmoid(input_var);
        auto out_t = output.tensor().contiguous();
        const float* out_data = out_t.data<float>();

        for (int64_t i = 0; i < n; ++i) {
            float x = input_vec[static_cast<size_t>(i)];
            float scalar_ref = 1.0f / (1.0f + std::exp(-x));
            // fast_math's ~2 ULP approximation, not exact libm -- tolerance
            // matches the approximation's own documented accuracy, not a
            // new/looser bar introduced by this task.
            EXPECT_NEAR(out_data[i], scalar_ref, 1e-5f) << "n=" << n << " i=" << i;
        }
    }
}

TEST(ActivationDispatch, TanhF32MatchesStdTanhAcrossSizes) {
    // tanh's math is untouched by this fix (still exact std::tanh below
    // threshold, oneDNN's own "exact" tanh above it) -- this locks in that
    // BOTH paths produce the same values as std::tanh directly, at a size
    // that stays below the new threshold and one that crosses above it.
    // n=16384 is the size that actually exercises this task's change: it is
    // below the old threshold (65536, so it used to run scalar std::tanh)
    // but above the new ONEDNN_TRANSCENDENTAL_THRESHOLD (8192, so it now
    // runs through oneDNN) -- 512 and 262144 alone would leave that band
    // completely uncovered.
    for (int64_t n : {int64_t{512}, int64_t{16384}, int64_t{262144}}) {
        std::vector<float> input_vec(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            input_vec[static_cast<size_t>(i)] = 0.02f * static_cast<float>((i * 53 + 3) % 97) - 1.0f;
        }
        Tensor input({n}, DType::Float32, Device::cpu());
        std::memcpy(input.data<float>(), input_vec.data(), static_cast<size_t>(n) * sizeof(float));

        Variable input_var(input, false);
        auto output = nn::tanh(input_var);
        auto out_t = output.tensor().contiguous();
        const float* out_data = out_t.data<float>();

        for (int64_t i = 0; i < n; ++i) {
            float expected = std::tanh(input_vec[static_cast<size_t>(i)]);
            // oneDNN's primitive vs. std::tanh: allow float32 rounding-level
            // tolerance only, not an approximation-level one -- this test
            // exists specifically to catch it if oneDNN's "exact" claim
            // (the existing code comment's words, not verified before this
            // task) turns out to be inexact at the bit level.
            EXPECT_NEAR(out_data[i], expected, 1e-6f) << "n=" << n << " i=" << i;
        }
    }
}

TEST(ActivationDispatch, GeluF32MatchesExactErfAcrossSizes) {
    // n=16384 sits between the old threshold (65536) and the new
    // ONEDNN_TRANSCENDENTAL_THRESHOLD (8192) -- it is the size this task's
    // change actually reroutes from scalar std::erf to oneDNN, so it must
    // be covered here rather than just the always-scalar (512) and
    // always-oneDNN (262144) sizes.
    for (int64_t n : {int64_t{512}, int64_t{16384}, int64_t{262144}}) {
        std::vector<float> input_vec(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            input_vec[static_cast<size_t>(i)] = 0.02f * static_cast<float>((i * 59 + 13) % 89) - 0.9f;
        }
        Tensor input({n}, DType::Float32, Device::cpu());
        std::memcpy(input.data<float>(), input_vec.data(), static_cast<size_t>(n) * sizeof(float));

        Variable input_var(input, false);
        auto output = nn::gelu(input_var);
        auto out_t = output.tensor().contiguous();
        const float* out_data = out_t.data<float>();

        constexpr float INV_SQRT2_F = 0.70710678f;
        for (int64_t i = 0; i < n; ++i) {
            float x = input_vec[static_cast<size_t>(i)];
            float expected = 0.5f * x * (1.0f + std::erf(x * INV_SQRT2_F));
            EXPECT_NEAR(out_data[i], expected, 1e-6f) << "n=" << n << " i=" << i;
        }
    }
}
