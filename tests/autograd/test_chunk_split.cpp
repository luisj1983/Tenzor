/**
 * @file test_chunk_split.cpp
 * @brief Regression coverage for tenzor::chunk / tenzor::split.
 *
 * Audit-5 V.10 + Y.5 + Z.27:
 *   - V.10 added Variable-level `tenzor::chunk` / `tenzor::split` that
 *     decompose into per-slice SliceBackward so the autograd graph survives
 *     the QKV-style split pattern (which previously severed grad_fn via
 *     `tenzor::chunk(var.tensor(), …)`).
 *   - Y.5 fixed the empty-dim chunk regression where callers doing
 *     `auto [a, b, c] = chunk(x, 3)` would index out of range when the
 *     split dim was zero-length.
 *   - Z.27 (this test): regression cover for the empty-dim contract for
 *     both chunk (must return `chunks` empty outputs) and split (must
 *     return a single empty output, matching PyTorch).
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"

using namespace tenzor;

class ChunkSplitTest : public ::testing::Test {
protected:
    static bool initialized;
    void SetUp() override {
        if (!initialized) {
            tenzor::initialize();
            initialized = true;
        }
    }
};
bool ChunkSplitTest::initialized = false;

// ============================================================================
// Y.5 + Z.27: chunk(zeros({0, 5}), 3, dim=0) must yield 3 empty [0, 5] outputs
// ============================================================================

TEST_F(ChunkSplitTest, ChunkEmptyDimReturnsRequestedChunks) {
    Variable input(zeros({0, 5}, DType::Float32, Device::cpu()),
                   /*requires_grad=*/true);

    auto pieces = tenzor::chunk(input, /*chunks=*/3, /*dim=*/0);

    // PyTorch parity: empty split dim still produces `chunks` empty slices.
    EXPECT_EQ(pieces.size(), 3u)
        << "tenzor::chunk on a zero-length dim must return `chunks` empty "
           "outputs (PyTorch parity), not an empty vector. Y.5 regression.";

    for (size_t i = 0; i < pieces.size(); ++i) {
        const auto& shape = pieces[i].shape();
        ASSERT_EQ(shape.size(), 2u) << "piece " << i << " ndim mismatch";
        EXPECT_EQ(shape[0], 0) << "piece " << i << " dim-0 should be 0";
        EXPECT_EQ(shape[1], 5) << "piece " << i << " dim-1 should be 5";
        EXPECT_EQ(pieces[i].tensor().numel(), 0)
            << "piece " << i << " should be a zero-element tensor";
    }
}

TEST_F(ChunkSplitTest, ChunkEmptyDimAllowsStructuredBinding) {
    // The bug Y.5 surfaced — destructuring assumes `chunks` outputs.
    Variable input(zeros({0, 4}, DType::Float32, Device::cpu()), false);

    auto pieces = tenzor::chunk(input, /*chunks=*/2, /*dim=*/0);
    ASSERT_EQ(pieces.size(), 2u);
    // If Y.5 had not landed, accessing pieces[1] would be out-of-range.
    const auto& second = pieces[1];
    EXPECT_EQ(second.shape()[0], 0);
    EXPECT_EQ(second.shape()[1], 4);
}

// ============================================================================
// Z.27: split(zeros({0, 5}), 2, dim=0) must yield a single empty [0, 5] output
// ============================================================================

TEST_F(ChunkSplitTest, SplitEmptyDimReturnsSingleEmptyOutput) {
    Variable input(zeros({0, 5}, DType::Float32, Device::cpu()),
                   /*requires_grad=*/true);

    auto pieces = tenzor::split(input, /*split_size=*/2, /*dim=*/0);

    // PyTorch parity: torch.split(zeros(0, 5), 2, dim=0) returns a 1-tuple
    // containing a single empty (0, 5) tensor.
    EXPECT_EQ(pieces.size(), 1u)
        << "tenzor::split on a zero-length dim must return a 1-element "
           "vector with a single empty output (PyTorch parity).";

    ASSERT_GE(pieces.size(), 1u);
    const auto& shape = pieces[0].shape();
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 0);
    EXPECT_EQ(shape[1], 5);
    EXPECT_EQ(pieces[0].tensor().numel(), 0);
}

// ============================================================================
// Non-empty sanity: ensure the regression test didn't break the common case.
// ============================================================================

TEST_F(ChunkSplitTest, ChunkNonEmptyDimPartitionsAndPreservesGrad) {
    Variable input(ones({6, 4}, DType::Float32, Device::cpu()),
                   /*requires_grad=*/true);

    auto pieces = tenzor::chunk(input, /*chunks=*/3, /*dim=*/0);
    ASSERT_EQ(pieces.size(), 3u);
    for (const auto& p : pieces) {
        EXPECT_EQ(p.shape()[0], 2);
        EXPECT_EQ(p.shape()[1], 4);
    }
    // Grad-flow check — chunk must decompose into SliceBackward so the
    // graph carries from output through to input (V.10 contract).
    auto sum_chunk = tenzor::sum(pieces[0]);
    sum_chunk.backward();
    ASSERT_TRUE(input.grad().has_value());
    EXPECT_GT(input.grad().value().numel(), 0);
}

TEST_F(ChunkSplitTest, SplitNonEmptyDimPartitions) {
    Variable input(ones({5, 3}, DType::Float32, Device::cpu()),
                   /*requires_grad=*/false);

    auto pieces = tenzor::split(input, /*split_size=*/2, /*dim=*/0);
    // 5 split by size 2 → [2, 2, 1]
    ASSERT_EQ(pieces.size(), 3u);
    EXPECT_EQ(pieces[0].shape()[0], 2);
    EXPECT_EQ(pieces[1].shape()[0], 2);
    EXPECT_EQ(pieces[2].shape()[0], 1);
}
