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
// Numerical Correctness Tests
// ============================================================================

TEST_P(EmbeddingBagMultiDTypeTest, SumModeMatchesManualSum) {
    skipIfIntegerDtype();
    if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
        GTEST_SKIP() << "Numerical correctness check requires Float32/Float64";
    }
    nn::EmbeddingBag emb(5, 3, 0.0, 2.0, false, "sum");
    convert_model(emb);

    // Set the embedding table to a known pattern: row i = [i, i+1, i+2]
    auto params = emb.parameters();
    ASSERT_FALSE(params.empty());
    auto cpu_table = tenzor::zeros({5, 3}, DType::Float32, Device::cpu());
    auto* tbl = cpu_table.data<float>();
    for (int64_t i = 0; i < 5; ++i) {
        tbl[i * 3 + 0] = static_cast<float>(i);
        tbl[i * 3 + 1] = static_cast<float>(i + 1);
        tbl[i * 3 + 2] = static_cast<float>(i + 2);
    }
    auto target_table = cpu_table.to(dtype()).to(device());
    // Replace parameter tensor in-place: zero then add target.
    params[0]->tensor().zero_();
    params[0]->tensor() += target_table;

    // Bag 0: indices [0, 1] → row0+row1 = [0+1, 1+2, 2+3] = [1, 3, 5]
    // Bag 1: indices [2, 3, 4] → row2+row3+row4 = [2+3+4, 3+4+5, 4+5+6] = [9, 12, 15]
    auto indices = makeInt64Tensor({0, 1, 2, 3, 4});
    auto offsets = makeInt64Tensor({0, 2});
    auto output = emb.forward(Variable(indices, false), Variable(offsets, false));

    auto cpu_out = output.tensor().to(Device::cpu()).to(DType::Float32).contiguous();
    auto* out_data = cpu_out.data<float>();
    float tol = std::max(atol() * 100.0f, 1e-3f);
    EXPECT_NEAR(out_data[0], 1.0f, tol);
    EXPECT_NEAR(out_data[1], 3.0f, tol);
    EXPECT_NEAR(out_data[2], 5.0f, tol);
    EXPECT_NEAR(out_data[3], 9.0f, tol);
    EXPECT_NEAR(out_data[4], 12.0f, tol);
    EXPECT_NEAR(out_data[5], 15.0f, tol);
}

// ============================================================================
// Max-mode backward (release-prep A5): gradient must route ONLY to the
// per-feature argmax row, never scatter-add across the whole bag. CPU-only,
// deterministic distinct weights so the argmax is unambiguous.
// ============================================================================

TEST(EmbeddingBagMaxBackward, RoutesGradOnlyToArgmaxRow) {
    tenzor::initialize();
    // num_embeddings=5, dim=3, max_norm disabled, mode="max".
    nn::EmbeddingBag emb(5, 3, /*max_norm=*/0.0, /*norm_type=*/2.0,
                         /*scale_grad_by_freq=*/false, /*mode=*/"max");

    // Distinct, row-monotone weights: weight[r][j] = r*10 + j. Within any
    // contiguous bag the largest row index wins every feature.
    {
        auto& w = emb.weight();
        auto* wp = w.tensor().data<float>();
        for (int64_t r = 0; r < 5; ++r)
            for (int64_t j = 0; j < 3; ++j)
                wp[r * 3 + j] = static_cast<float>(r * 10 + j);
    }

    // indices [0,1,2,3,4], offsets [0,3] -> bag0={rows 0,1,2}, bag1={rows 3,4}.
    auto mk = [](std::initializer_list<int64_t> v) {
        auto t = tenzor::zeros({static_cast<int64_t>(v.size())}, DType::Int64, Device::cpu());
        auto* p = t.data<int64_t>(); int64_t i = 0; for (auto x : v) p[i++] = x; return t;
    };
    auto input = Variable(mk({0, 1, 2, 3, 4}), false);
    auto offsets = Variable(mk({0, 3}), false);

    auto output = emb.forward(input, offsets);  // [2, 3]
    output.backward(tenzor::ones({2, 3}, DType::Float32, Device::cpu()));

    ASSERT_TRUE(emb.weight().grad().has_value());
    const auto* g = emb.weight().grad()->data<float>();

    // Argmax of bag0 is row 2; of bag1 is row 4. Each winning element gets the
    // upstream grad (1.0); every other row stays exactly 0.
    for (int64_t r = 0; r < 5; ++r) {
        float expected = (r == 2 || r == 4) ? 1.0f : 0.0f;
        for (int64_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(g[r * 3 + j], expected)
                << "row " << r << " col " << j;
        }
    }
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(EmbeddingBagMultiDTypeTest);
