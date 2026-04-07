/**
 * @file test_embedding_bag_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for EmbeddingBag operations
 *
 * Covers: EmbeddingBag forward (sum, mean modes), backward gradient flow,
 * varying offsets.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/embedding.hpp>
#include <tenzor/autograd/variable.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class EmbeddingBagMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void skipIfIntegerDtype() {
        if (dtype() == DType::Int32 || dtype() == DType::Int64 ||
            dtype() == DType::Int8 || dtype() == DType::UInt8) {
            GTEST_SKIP() << "EmbeddingBag requires floating-point dtype";
        }
    }

    // Create an Int64 tensor from values on the test device
    Tensor makeInt64Tensor(std::initializer_list<int64_t> vals) {
        auto t = tenzor::zeros({static_cast<int64_t>(vals.size())}, DType::Int64, Device::cpu());
        auto* ptr = t.data<int64_t>();
        int64_t i = 0;
        for (auto v : vals) {
            ptr[i++] = v;
        }
        if (device().type != Device::Type::CPU) {
            return t.to(device());
        }
        return t;
    }
};

// ============================================================================
// Sum mode
// ============================================================================

TEST_P(EmbeddingBagMultiDTypeTest, ForwardSumMode) {
    skipIfIntegerDtype();
    nn::EmbeddingBag emb(10, 4, 0.0, 2.0, false, "sum");
    convert_model(emb);

    // Input: flat indices, offsets define bags
    auto indices = makeInt64Tensor({0, 1, 2, 3, 4});
    auto offsets = makeInt64Tensor({0, 2, 3});
    auto input = Variable(indices, false);
    auto offsets_var = Variable(offsets, false);

    auto output = emb.forward(input, offsets_var);
    // 3 bags, embedding_dim=4
    expectShape(output.tensor(), {3, 4});
    expectDevice(output.tensor());
}

// ============================================================================
// Mean mode
// ============================================================================

TEST_P(EmbeddingBagMultiDTypeTest, ForwardMeanMode) {
    skipIfIntegerDtype();
    nn::EmbeddingBag emb(10, 8, 0.0, 2.0, false, "mean");
    convert_model(emb);

    auto indices = makeInt64Tensor({0, 1, 2, 3, 4, 5});
    auto offsets = makeInt64Tensor({0, 3});
    auto input = Variable(indices, false);
    auto offsets_var = Variable(offsets, false);

    auto output = emb.forward(input, offsets_var);
    // 2 bags, embedding_dim=8
    expectShape(output.tensor(), {2, 8});
    expectDevice(output.tensor());
}

// ============================================================================
// Backward gradient flow
// ============================================================================

TEST_P(EmbeddingBagMultiDTypeTest, BackwardGradientFlow) {
    skipIfIntegerDtype();
    nn::EmbeddingBag emb(10, 4, 0.0, 2.0, false, "sum");
    convert_model(emb);

    auto indices = makeInt64Tensor({0, 1, 2, 3});
    auto offsets = makeInt64Tensor({0, 2});
    auto input = Variable(indices, false);
    auto offsets_var = Variable(offsets, false);

    auto output = emb.forward(input, offsets_var);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad_output = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad_output); })
        << "EmbeddingBag backward threw on " << device().to_string();

    // Embedding weight should have gradient
    auto params = emb.parameters();
    ASSERT_FALSE(params.empty()) << "EmbeddingBag has no parameters";
    ASSERT_TRUE(params[0]->grad().has_value())
        << "EmbeddingBag weight gradient not produced on " << device().to_string();
}

// ============================================================================
// Single bag (single offset at 0)
// ============================================================================

TEST_P(EmbeddingBagMultiDTypeTest, ForwardSingleBag) {
    skipIfIntegerDtype();
    nn::EmbeddingBag emb(10, 4, 0.0, 2.0, false, "sum");
    convert_model(emb);

    auto indices = makeInt64Tensor({0, 1, 2});
    auto offsets = makeInt64Tensor({0});
    auto input = Variable(indices, false);
    auto offsets_var = Variable(offsets, false);

    auto output = emb.forward(input, offsets_var);
    // Single bag: 1 bag, embedding_dim=4
    expectShape(output.tensor(), {1, 4});
    expectDevice(output.tensor());
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(EmbeddingBagMultiDTypeTest);
