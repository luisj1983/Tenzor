// Dedicated multi-backend tests for CosineSimilarity and PairwiseDistance.
//
// Before this file, both modules in tenzor::nn::distance.hpp had no
// dedicated tests. They compose Variable-level ops (sub, abs, pow, sum,
// sqrt, normalize) so autograd flows through automatically — this file
// verifies forward values, gradient propagation, and cross-backend parity.

#include <gtest/gtest.h>

#include <cmath>

#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/layers/distance.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/tenzor.hpp>

#include "../../backend_test_fixture.hpp"

namespace tenzor {

using ::tenzor::testing::BackendTest;

class DistanceLayersTest : public BackendTest {};

// ---------------------------------------------------------------------------
// CosineSimilarity
// ---------------------------------------------------------------------------

TEST_P(DistanceLayersTest, CosineSimilarity_OrthogonalVectorsReturnZero) {
    // Orthogonal unit vectors → similarity ≈ 0.
    auto a = Variable(zeros({1, 4}, DType::Float32, device), false);
    auto b = Variable(zeros({1, 4}, DType::Float32, device), false);
    // a = [1, 0, 0, 0], b = [0, 1, 0, 0]
    auto a_cpu = zeros({1, 4}, DType::Float32, Device::cpu());
    auto b_cpu = zeros({1, 4}, DType::Float32, Device::cpu());
    a_cpu.data<float>()[0] = 1.0f;
    b_cpu.data<float>()[1] = 1.0f;
    a = Variable(a_cpu.to(device), false);
    b = Variable(b_cpu.to(device), false);

    nn::CosineSimilarity cs(/*dim=*/1);
    auto sim = cs(a, b);
    auto sim_cpu = sim.tensor().to(Device::cpu()).to(DType::Float32).contiguous();
    EXPECT_NEAR(sim_cpu.data<float>()[0], 0.0f, 1e-5f);
}

TEST_P(DistanceLayersTest, CosineSimilarity_IdenticalVectorsReturnOne) {
    auto x_cpu = randn({2, 8}, DType::Float32, Device::cpu());
    auto a = Variable(x_cpu.to(device), false);
    auto b = Variable(x_cpu.to(device), false);

    nn::CosineSimilarity cs(/*dim=*/1);
    auto sim = cs(a, b);
    auto sim_cpu = sim.tensor().to(Device::cpu()).to(DType::Float32).contiguous();
    for (int64_t i = 0; i < sim_cpu.numel(); ++i) {
        EXPECT_NEAR(sim_cpu.data<float>()[i], 1.0f, 1e-4f)
            << "Identical vectors should have cosine similarity 1";
    }
}

TEST_P(DistanceLayersTest, CosineSimilarity_GradientFlows) {
    auto a = Variable(randn({2, 4}, DType::Float32, device), /*requires_grad=*/true);
    auto b = Variable(randn({2, 4}, DType::Float32, device), /*requires_grad=*/false);

    nn::CosineSimilarity cs(/*dim=*/1);
    auto sim = cs(a, b);
    auto loss = tenzor::sum(sim);
    loss.backward();

    ASSERT_TRUE(a.has_grad());
    auto g = a.grad().value().to(Device::cpu()).to(DType::Float32).contiguous();
    float max_abs = 0.0f;
    const float* gp = g.data<float>();
    for (int64_t i = 0; i < g.numel(); ++i) {
        max_abs = std::max(max_abs, std::abs(gp[i]));
    }
    EXPECT_GT(max_abs, 0.0f)
        << "CosineSimilarity must propagate gradient to its inputs";
}

// ---------------------------------------------------------------------------
// PairwiseDistance
// ---------------------------------------------------------------------------

TEST_P(DistanceLayersTest, PairwiseDistance_L2_ZeroWhenEqual) {
    auto x_cpu = randn({4, 6}, DType::Float32, Device::cpu());
    auto a = Variable(x_cpu.to(device), false);
    auto b = Variable(x_cpu.to(device), false);

    nn::PairwiseDistance pd(/*p=*/2.0, /*eps=*/1e-8);
    auto d = pd(a, b);
    auto d_cpu = d.tensor().to(Device::cpu()).to(DType::Float32).contiguous();
    // Output length equals batch size (dim -1 is reduced).
    EXPECT_EQ(d_cpu.numel(), 4);
    for (int64_t i = 0; i < d_cpu.numel(); ++i) {
        EXPECT_LT(d_cpu.data<float>()[i], 1e-3f)
            << "Identical inputs should give ~0 distance (eps smoothed)";
    }
}

TEST_P(DistanceLayersTest, PairwiseDistance_L1_MatchesManhattan) {
    // For simple known vectors, verify L1 distance is sum of absolute diffs.
    auto a_cpu = zeros({1, 3}, DType::Float32, Device::cpu());
    auto b_cpu = zeros({1, 3}, DType::Float32, Device::cpu());
    float* ap = a_cpu.data<float>();
    float* bp = b_cpu.data<float>();
    ap[0] = 1.0f; ap[1] = 2.0f; ap[2] = 3.0f;
    bp[0] = 0.0f; bp[1] = 0.0f; bp[2] = 0.0f;
    auto a = Variable(a_cpu.to(device), false);
    auto b = Variable(b_cpu.to(device), false);

    nn::PairwiseDistance pd(/*p=*/1.0, /*eps=*/0.0);
    auto d = pd(a, b);
    auto d_cpu = d.tensor().to(Device::cpu()).to(DType::Float32).contiguous();
    EXPECT_NEAR(d_cpu.data<float>()[0], 6.0f, 1e-4f)
        << "L1 distance between [1,2,3] and [0,0,0] should be 6";
}

TEST_P(DistanceLayersTest, PairwiseDistance_GradientFlows) {
    auto a = Variable(randn({2, 5}, DType::Float32, device), /*requires_grad=*/true);
    auto b = Variable(randn({2, 5}, DType::Float32, device), /*requires_grad=*/false);

    nn::PairwiseDistance pd(/*p=*/2.0, /*eps=*/1e-6);
    auto d = pd(a, b);
    auto loss = tenzor::sum(d);
    loss.backward();

    ASSERT_TRUE(a.has_grad());
    auto g = a.grad().value().to(Device::cpu()).to(DType::Float32).contiguous();
    float max_abs = 0.0f;
    const float* gp = g.data<float>();
    for (int64_t i = 0; i < g.numel(); ++i) {
        max_abs = std::max(max_abs, std::abs(gp[i]));
    }
    EXPECT_GT(max_abs, 0.0f)
        << "PairwiseDistance must propagate gradient to its inputs";
}

INSTANTIATE_BACKEND_TESTS(DistanceLayersTest);

}  // namespace tenzor
