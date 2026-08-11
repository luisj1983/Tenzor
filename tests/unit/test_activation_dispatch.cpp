#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/tenzor.hpp"
#include <cmath>
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
    // ONEDNN_ACTIVATION_THRESHOLD (65536), and the exact benchmark sizes
    // that showed the regression (32768, 262144, 2097152).
    for (int64_t n : {int64_t{1}, int64_t{15}, int64_t{16}, int64_t{17},
                       int64_t{32768}, int64_t{65536}, int64_t{262144}, int64_t{2097152}}) {
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
