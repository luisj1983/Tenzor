/**
 * @file test_embedding.cpp
 * @brief Unit tests for Embedding and EmbeddingBag layers
 */

#include <gtest/gtest.h>
#include "tenzor/nn/layers/embedding.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/variable.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

class EmbeddingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set random seed for reproducibility
    }
};

// ============================================================================
// Embedding Tests
// ============================================================================

TEST_F(EmbeddingTest, BasicLookup) {
    // Create embedding: 10 words, 5-dimensional embeddings
    auto embedding = std::make_shared<Embedding>(10, 5);

    // Input: 3 token indices
    auto input_data = zeros({3}, DType::Int64);
    auto input_ptr = input_data.data<int64_t>();
    input_ptr[0] = 0;
    input_ptr[1] = 5;
    input_ptr[2] = 9;

    auto input = Variable(input_data, false);

    // Forward pass
    auto output = embedding->forward(input);

    // Check output shape: [3, 5]
    auto output_shape = output.tensor().shape();
    ASSERT_EQ(output_shape.size(), 2);
    EXPECT_EQ(output_shape[0], 3);
    EXPECT_EQ(output_shape[1], 5);

    // Verify lookups are different vectors
    auto output_ptr = output.tensor().data<float>();
    bool all_same = true;
    for (int i = 0; i < 5; ++i) {
        if (output_ptr[i] != output_ptr[5 + i]) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);
}

TEST_F(EmbeddingTest, MultiDimensionalInput) {
    // Embedding: 100 tokens, 8-dim
    auto embedding = std::make_shared<Embedding>(100, 8);

    // Input: [batch=2, seq_len=4]
    auto input_data = zeros({2, 4}, DType::Int64);
    auto input_ptr = input_data.data<int64_t>();
    for (int i = 0; i < 8; ++i) {
        input_ptr[i] = i * 10;
    }

    auto input = Variable(input_data, false);
    auto output = embedding->forward(input);

    // Check output shape: [2, 4, 8]
    auto output_shape = output.tensor().shape();
    ASSERT_EQ(output_shape.size(), 3);
    EXPECT_EQ(output_shape[0], 2);
    EXPECT_EQ(output_shape[1], 4);
    EXPECT_EQ(output_shape[2], 8);
}

TEST_F(EmbeddingTest, PaddingIdx) {
    // Create embedding with padding_idx=0
    auto embedding = std::make_shared<Embedding>(10, 5, /*padding_idx=*/0);

    // Check that padding embedding is zero
    auto& weight = embedding->weight();
    auto weight_ptr = weight.tensor().data<float>();

    for (int j = 0; j < 5; ++j) {
        EXPECT_FLOAT_EQ(weight_ptr[j], 0.0f) << "Padding embedding should be zero at index " << j;
    }

    // Forward with padding index
    auto input_data = zeros({3}, DType::Int64);
    auto input_ptr = input_data.data<int64_t>();
    input_ptr[0] = 0;  // Padding
    input_ptr[1] = 5;
    input_ptr[2] = 0;  // Padding

    auto input = Variable(input_data, false);
    auto output = embedding->forward(input);
    auto output_ptr = output.tensor().data<float>();

    // Check padding outputs are zero
    for (int j = 0; j < 5; ++j) {
        EXPECT_FLOAT_EQ(output_ptr[j], 0.0f);
        EXPECT_FLOAT_EQ(output_ptr[10 + j], 0.0f);
    }

    // Check non-padding output is non-zero
    bool has_nonzero = false;
    for (int j = 0; j < 5; ++j) {
        if (std::abs(output_ptr[5 + j]) > 1e-6) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST_F(EmbeddingTest, MaxNormRenormalization) {
    // Create embedding with max_norm=1.0
    auto embedding = std::make_shared<Embedding>(10, 5, -1, /*max_norm=*/1.0);

    // Manually set a large embedding
    auto& weight = embedding->weight();
    auto weight_ptr = weight.tensor().data<float>();

    // Set first embedding to large values
    for (int j = 0; j < 5; ++j) {
        weight_ptr[j] = 10.0f;
    }

    // Forward pass should trigger renormalization
    auto input_data = zeros({1}, DType::Int64);
    auto input_ptr = input_data.data<int64_t>();
    input_ptr[0] = 0;

    auto input = Variable(input_data, false);
    auto output = embedding->forward(input);

    // Compute norm of output
    auto output_ptr = output.tensor().data<float>();
    float norm = 0.0f;
    for (int j = 0; j < 5; ++j) {
        norm += output_ptr[j] * output_ptr[j];
    }
    norm = std::sqrt(norm);

    // Norm should be <= max_norm (with tolerance)
    EXPECT_LE(norm, 1.0f + 1e-5);
}

TEST_F(EmbeddingTest, OutOfRangeIndex) {
    auto embedding = std::make_shared<Embedding>(10, 5);

    // Try invalid index
    auto input_data = zeros({1}, DType::Int64);
    auto input_ptr = input_data.data<int64_t>();
    input_ptr[0] = 10;  // Out of range

    auto input = Variable(input_data, false);

    // Should throw exception
    EXPECT_THROW(embedding->forward(input), std::out_of_range);
}

TEST_F(EmbeddingTest, WeightAccess) {
    auto embedding = std::make_shared<Embedding>(10, 5);

    // Access weight matrix
    auto& weight = embedding->weight();
    auto weight_shape = weight.tensor().shape();

    ASSERT_EQ(weight_shape.size(), 2);
    EXPECT_EQ(weight_shape[0], 10);
    EXPECT_EQ(weight_shape[1], 5);

    // Const access
    const auto& const_embedding = *embedding;
    const auto& const_weight = const_embedding.weight();
    auto const_weight_shape = const_weight.tensor().shape();

    EXPECT_EQ(const_weight_shape[0], 10);
    EXPECT_EQ(const_weight_shape[1], 5);
}

// ============================================================================
// EmbeddingBag Tests
// ============================================================================

TEST_F(EmbeddingTest, EmbeddingBagSum) {
    // Create embedding bag with sum mode
    auto embedding_bag = std::make_shared<EmbeddingBag>(10, 5, 0.0, 2.0, false, "sum");

    // Set known weights for testing
    auto& weight = embedding_bag->weight();
    auto weight_ptr = weight.tensor().data<float>();
    for (int i = 0; i < 50; ++i) {
        weight_ptr[i] = static_cast<float>(i % 5 + 1);  // Values 1-5 repeating
    }

    // Input: [0, 1, 2] as single bag
    auto input_data = zeros({3}, DType::Int64);
    auto input_ptr = input_data.data<int64_t>();
    input_ptr[0] = 0;
    input_ptr[1] = 1;
    input_ptr[2] = 2;

    auto input = Variable(input_data, false);
    auto output = embedding_bag->forward(input, Variable{});

    // Output shape: [1, 5]
    auto output_shape = output.tensor().shape();
    ASSERT_EQ(output_shape.size(), 2);
    EXPECT_EQ(output_shape[0], 1);
    EXPECT_EQ(output_shape[1], 5);

    // Verify sum
    auto output_data = output.tensor();
    auto output_ptr = output_data.data<float>();
    for (int j = 0; j < 5; ++j) {
        float expected = weight_ptr[j] + weight_ptr[5 + j] + weight_ptr[10 + j];
        EXPECT_FLOAT_EQ(output_ptr[j], expected);
    }
}

TEST_F(EmbeddingTest, EmbeddingBagMean) {
    auto embedding_bag = std::make_shared<EmbeddingBag>(10, 5, 0.0, 2.0, false, "mean");

    // Set known weights
    auto& weight = embedding_bag->weight();
    auto weight_ptr = weight.tensor().data<float>();
    for (int i = 0; i < 50; ++i) {
        weight_ptr[i] = static_cast<float>(i % 5 + 1);
    }

    // Input with 3 elements
    auto input_data = zeros({3}, DType::Int64);
    auto input_ptr = input_data.data<int64_t>();
    input_ptr[0] = 0;
    input_ptr[1] = 1;
    input_ptr[2] = 2;

    auto input = Variable(input_data, false);
    auto output = embedding_bag->forward(input, Variable{});

    // Verify mean
    auto output_data = output.tensor();
    auto output_ptr = output_data.data<float>();
    for (int j = 0; j < 5; ++j) {
        float expected = (weight_ptr[j] + weight_ptr[5 + j] + weight_ptr[10 + j]) / 3.0f;
        EXPECT_NEAR(output_ptr[j], expected, 1e-5);
    }
}

TEST_F(EmbeddingTest, EmbeddingBagMax) {
    auto embedding_bag = std::make_shared<EmbeddingBag>(10, 5, 0.0, 2.0, false, "max");

    // Set known weights with clear maximum
    auto& weight = embedding_bag->weight();
    auto weight_ptr = weight.tensor().data<float>();
    for (int i = 0; i < 50; ++i) {
        weight_ptr[i] = 1.0f;
    }
    // Make second embedding largest
    for (int j = 0; j < 5; ++j) {
        weight_ptr[5 + j] = 10.0f;
    }

    // Input: [0, 1, 2]
    auto input_data = zeros({3}, DType::Int64);
    auto input_ptr = input_data.data<int64_t>();
    input_ptr[0] = 0;
    input_ptr[1] = 1;
    input_ptr[2] = 2;

    auto input = Variable(input_data, false);
    auto output = embedding_bag->forward(input, Variable{});

    // All outputs should be 10.0 (max)
    auto output_data = output.tensor();
    auto output_ptr = output_data.data<float>();
    for (int j = 0; j < 5; ++j) {
        EXPECT_FLOAT_EQ(output_ptr[j], 10.0f);
    }
}

TEST_F(EmbeddingTest, EmbeddingBagWithOffsets) {
    auto embedding_bag = std::make_shared<EmbeddingBag>(10, 5, 0.0, 2.0, false, "mean");

    // Set known weights
    auto& weight = embedding_bag->weight();
    auto weight_ptr = weight.tensor().data<float>();
    for (int i = 0; i < 50; ++i) {
        weight_ptr[i] = static_cast<float>(i);
    }

    // Input: [0, 1, 2, 3, 4, 5] split into 3 bags
    auto input_data = zeros({6}, DType::Int64);
    auto input_ptr = input_data.data<int64_t>();
    for (int i = 0; i < 6; ++i) {
        input_ptr[i] = i;
    }

    // Offsets: bag0=[0,1], bag1=[2,3,4], bag2=[5]
    auto offsets_data = zeros({3}, DType::Int64);
    auto offsets_ptr = offsets_data.data<int64_t>();
    offsets_ptr[0] = 0;
    offsets_ptr[1] = 2;
    offsets_ptr[2] = 5;

    auto input = Variable(input_data, false);
    auto offsets = Variable(offsets_data, false);
    auto output = embedding_bag->forward(input, offsets);

    // Output shape: [3, 5]
    auto output_shape = output.tensor().shape();
    ASSERT_EQ(output_shape.size(), 2);
    EXPECT_EQ(output_shape[0], 3);
    EXPECT_EQ(output_shape[1], 5);

    auto output_ptr = output.tensor().data<float>();

    // Verify bag 0: mean of embeddings 0 and 1
    for (int j = 0; j < 5; ++j) {
        float expected = (weight_ptr[j] + weight_ptr[5 + j]) / 2.0f;
        EXPECT_NEAR(output_ptr[j], expected, 1e-4);
    }

    // Verify bag 1: mean of embeddings 2, 3, 4
    for (int j = 0; j < 5; ++j) {
        float expected = (weight_ptr[10 + j] + weight_ptr[15 + j] + weight_ptr[20 + j]) / 3.0f;
        EXPECT_NEAR(output_ptr[5 + j], expected, 1e-4);
    }

    // Verify bag 2: just embedding 5
    for (int j = 0; j < 5; ++j) {
        float expected = weight_ptr[25 + j];
        EXPECT_FLOAT_EQ(output_ptr[10 + j], expected);
    }
}

TEST_F(EmbeddingTest, EmbeddingBagEmptyBag) {
    auto embedding_bag = std::make_shared<EmbeddingBag>(10, 5, 0.0, 2.0, false, "mean");

    // Empty input
    auto input_data = zeros({0}, DType::Int64);
    auto input = Variable(input_data, false);

    // Should handle gracefully (output zeros)
    auto output = embedding_bag->forward(input, Variable{});
    auto output_shape = output.tensor().shape();

    ASSERT_EQ(output_shape.size(), 2);
    EXPECT_EQ(output_shape[0], 1);
    EXPECT_EQ(output_shape[1], 5);
}

TEST_F(EmbeddingTest, EmbeddingBagInvalidMode) {
    // Invalid mode should throw
    EXPECT_THROW(
        std::make_shared<EmbeddingBag>(10, 5, 0.0, 2.0, false, "invalid"),
        std::invalid_argument
    );
}

TEST_F(EmbeddingTest, InvalidParameters) {
    // Negative num_embeddings
    EXPECT_THROW(
        std::make_shared<Embedding>(-1, 5),
        std::invalid_argument
    );

    // Negative embedding_dim
    EXPECT_THROW(
        std::make_shared<Embedding>(10, -5),
        std::invalid_argument
    );

    // Invalid padding_idx
    EXPECT_THROW(
        std::make_shared<Embedding>(10, 5, 10),
        std::invalid_argument
    );
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(EmbeddingTest, EmbeddingInModule) {
    // Test embedding as part of module hierarchy
    auto embedding = std::make_shared<Embedding>(100, 50);

    // Get parameters
    auto params = embedding->parameters();
    ASSERT_EQ(params.size(), 1);

    // Check parameter shape
    auto param_shape = params[0]->tensor().shape();
    EXPECT_EQ(param_shape[0], 100);
    EXPECT_EQ(param_shape[1], 50);
}

TEST_F(EmbeddingTest, EmbeddingTrainEvalMode) {
    auto embedding = std::make_shared<Embedding>(10, 5);

    // Check training mode
    EXPECT_TRUE(embedding->is_training());

    // Set to eval
    embedding->eval();
    EXPECT_FALSE(embedding->is_training());

    // Set to train
    embedding->train();
    EXPECT_TRUE(embedding->is_training());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
