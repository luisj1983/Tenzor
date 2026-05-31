/**
 * @file test_vmap_expanded.cpp
 * @brief Tests for expanded vmap batching rules
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/vmap.hpp>
#include <tenzor/autograd/ops.hpp>
#include "../backend_test_fixture.hpp"

using namespace tenzor;

class VmapExpanded : public ::tenzor::testing::BackendTest {};

// Helper: compare vmap result against loop-and-stack reference
static void verify_vmap(std::function<Variable(const Variable&)> func,
                         const Variable& batched_input,
                         int64_t batch_dim, const char* name,
                         float tol = 1e-4f) {
    auto vmapped = vmap(func, batched_input, batch_dim);

    // Reference: manual loop-and-stack
    auto input_t = batched_input.tensor();
    int64_t batch_size = input_t.shape()[batch_dim];
    std::vector<Tensor> refs;
    for (int64_t i = 0; i < batch_size; ++i) {
        auto slice = tenzor::select(input_t, batch_dim, i);
        Variable sv(slice, false);
        refs.push_back(func(sv).tensor());
    }
    auto ref = tenzor::stack(refs, batch_dim);

    auto v = vmapped.tensor().to(Device::cpu()).contiguous();
    auto r = ref.to(Device::cpu()).contiguous();
    ASSERT_EQ(v.numel(), r.numel()) << name << ": shape mismatch";
    auto* vd = v.data<float>();
    auto* rd = r.data<float>();
    for (int64_t i = 0; i < v.numel(); ++i) {
        EXPECT_NEAR(vd[i], rd[i], tol)
            << name << " vmap mismatch at " << i;
    }
}

TEST_P(VmapExpanded, ElementWise_Exp) {
    auto x = Variable(tenzor::randn({4, 3}, DType::Float32, device), false);
    verify_vmap([](const Variable& v) { return tenzor::exp(v); }, x, 0, "exp");
}

TEST_P(VmapExpanded, ElementWise_Sigmoid) {
    auto x = Variable(tenzor::randn({4, 3}, DType::Float32, device), false);
    verify_vmap([](const Variable& v) { return tenzor::sigmoid(v); }, x, 0, "sigmoid");
}

TEST_P(VmapExpanded, ElementWise_Tanh) {
    auto x = Variable(tenzor::randn({4, 3}, DType::Float32, device), false);
    verify_vmap([](const Variable& v) { return tenzor::tanh(v); }, x, 0, "tanh");
}

TEST_P(VmapExpanded, Reduction_Sum) {
    auto x = Variable(tenzor::randn({4, 3, 5}, DType::Float32, device), false);
    verify_vmap([](const Variable& v) { return tenzor::sum(v); }, x, 0, "sum");
}

TEST_P(VmapExpanded, Softmax) {
    auto x = Variable(tenzor::randn({4, 3, 5}, DType::Float32, device), false);
    verify_vmap([](const Variable& v) {
        // Softmax along last dim
        return tenzor::softmax(v, -1);
    }, x, 0, "softmax");
}

TEST_P(VmapExpanded, ShapePreserved) {
    auto x = Variable(tenzor::randn({4, 3, 5}, DType::Float32, device), false);
    auto result = vmap([](const Variable& v) { return tenzor::exp(v); }, x, 0);
    auto shape = result.tensor().shape();
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 5);
}

TEST_P(VmapExpanded, HasBatchingRules) {
    // Verify key rules are registered
    init_builtin_batching_rules();
    EXPECT_TRUE(has_batching_rule("AddBackward"));
    EXPECT_TRUE(has_batching_rule("MulBackward"));
    EXPECT_TRUE(has_batching_rule("ReLUBackward"));
    EXPECT_TRUE(has_batching_rule("MatMulBackward"));
    EXPECT_TRUE(has_batching_rule("SoftmaxBackward"));
    EXPECT_TRUE(has_batching_rule("SumBackward"));
    EXPECT_TRUE(has_batching_rule("ReshapeBackward"));
    EXPECT_TRUE(has_batching_rule("LinearBackward"));
    EXPECT_TRUE(has_batching_rule("ErfBackward"));
    EXPECT_TRUE(has_batching_rule("GammaBackward"));
}

INSTANTIATE_BACKEND_TESTS(VmapExpanded);
