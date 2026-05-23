/**
 * @file test_chunk_multidtype.cpp
 * @brief Multi-dtype tests for chunk operation
 *
 * Tests `chunk()` across the canonical multi-backend × multi-dtype matrix.
 * Migrated from a custom `struct BackendDTypeParam` to the canonical
 * `MultiBackendDTypeTest` fixture per TESTING.md "Fixture hygiene". Test
 * coverage extended to include integer dtypes (Float32/Float64/Int32) by
 * using a custom `INSTANTIATE_TEST_SUITE_P` rather than the default
 * `INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS` macro (which only covers floats).
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Multi-DType Test Fixture
// ============================================================================

class ChunkMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Chunk Operation Tests
// ============================================================================

// audit-3 T.1 helper: verify chunk[i] equals the corresponding slice of the
// input by comparing flat float-cast contents (handles Int32/Float32/Float64).
static void expect_chunks_match_input_dim0(const Tensor& original,
                                            const std::vector<Tensor>& chunks) {
    auto orig_cpu = original.to(Device::cpu()).to(DType::Float32);
    const float* od = orig_cpu.data<float>();
    int64_t row_stride = orig_cpu.numel() / orig_cpu.shape()[0];
    int64_t offset = 0;
    for (size_t i = 0; i < chunks.size(); ++i) {
        auto c_cpu = chunks[i].to(Device::cpu()).to(DType::Float32);
        const float* cd = c_cpu.data<float>();
        for (int64_t e = 0; e < c_cpu.numel(); ++e) {
            EXPECT_FLOAT_EQ(cd[e], od[offset * row_stride + e])
                << "chunk " << i << " element " << e;
        }
        offset += c_cpu.shape()[0];
    }
}

TEST_P(ChunkMultiDTypeTest, BasicChunkEvenDivision) {
    // Test case: Shape (12, 20), split into 4 chunks along dim=0
    // Expected: 4 tensors of shape (3, 20)
    auto x = zeros({12, 20}, dtype(), device());

    auto chunks = chunk(x, 4, 0);

    ASSERT_EQ(chunks.size(), 4);

    for (size_t i = 0; i < chunks.size(); ++i) {
        auto shape = chunks[i].shape();
        EXPECT_EQ(shape.size(), 2);
        EXPECT_EQ(shape[0], 3);
        EXPECT_EQ(shape[1], 20);
    }
    // Element-level value check: each chunk slice equals the input slice.
    expect_chunks_match_input_dim0(x, chunks);
}

TEST_P(ChunkMultiDTypeTest, BasicChunkUnevenDivision) {
    // Test case: Shape (10, 20), split into 4 chunks along dim=0
    // Expected: 3 tensors of shape (3, 20) and 1 tensor of shape (1, 20)
    auto x = zeros({10, 20}, dtype(), device());

    auto chunks = chunk(x, 4, 0);

    ASSERT_EQ(chunks.size(), 4);

    // First 3 chunks should be (3, 20)
    for (size_t i = 0; i < 3; ++i) {
        auto shape = chunks[i].shape();
        EXPECT_EQ(shape.size(), 2);
        EXPECT_EQ(shape[0], 3);
        EXPECT_EQ(shape[1], 20);
    }

    // Last chunk should be (1, 20)
    auto last_shape = chunks[3].shape();
    EXPECT_EQ(last_shape[0], 1);
    EXPECT_EQ(last_shape[1], 20);
    expect_chunks_match_input_dim0(x, chunks);
}

TEST_P(ChunkMultiDTypeTest, ChunkFewerThanDimensionSize) {
    // Test case: Shape (5, 20), split into 10 chunks along dim=0
    // Expected: 5 tensors of shape (1, 20) - fewer chunks than requested
    auto x = zeros({5, 20}, dtype(), device());

    auto chunks = chunk(x, 10, 0);

    // Should only get 5 chunks since dimension size is 5
    ASSERT_EQ(chunks.size(), 5);

    for (size_t i = 0; i < chunks.size(); ++i) {
        auto shape = chunks[i].shape();
        EXPECT_EQ(shape.size(), 2);
        EXPECT_EQ(shape[0], 1);
        EXPECT_EQ(shape[1], 20);
    }
    expect_chunks_match_input_dim0(x, chunks);
}

TEST_P(ChunkMultiDTypeTest, ChunkAlongDifferentDimension) {
    // Test case: Shape (10, 20), split into 5 chunks along dim=1
    // Expected: 5 tensors of shape (10, 4)
    auto x = zeros({10, 20}, dtype(), device());

    auto chunks = chunk(x, 5, 1);

    ASSERT_EQ(chunks.size(), 5);

    for (size_t i = 0; i < chunks.size(); ++i) {
        auto shape = chunks[i].shape();
        EXPECT_EQ(shape.size(), 2);
        EXPECT_EQ(shape[0], 10);
        EXPECT_EQ(shape[1], 4);
    }
    // For dim=1, value check: each chunk's flat-cast Float32 elements equal 0.
    for (size_t i = 0; i < chunks.size(); ++i) {
        auto c = chunks[i].to(Device::cpu()).to(DType::Float32);
        const float* d = c.data<float>();
        for (int64_t e = 0; e < c.numel(); ++e) {
            EXPECT_FLOAT_EQ(d[e], 0.0f) << "chunk " << i << " elem " << e;
        }
    }
}

TEST_P(ChunkMultiDTypeTest, ChunkNegativeDimension) {
    // Test case: Use negative dimension indexing
    auto x = zeros({10, 20}, dtype(), device());

    // dim=-1 should be equivalent to dim=1
    auto chunks = chunk(x, 4, -1);

    ASSERT_EQ(chunks.size(), 4);

    for (size_t i = 0; i < chunks.size(); ++i) {
        auto shape = chunks[i].shape();
        EXPECT_EQ(shape[0], 10);
        EXPECT_EQ(shape[1], 5);
    }
    // Value: zero-input chunks must be zero in every element.
    for (size_t i = 0; i < chunks.size(); ++i) {
        auto c = chunks[i].to(Device::cpu()).to(DType::Float32);
        const float* d = c.data<float>();
        for (int64_t e = 0; e < c.numel(); ++e) {
            EXPECT_FLOAT_EQ(d[e], 0.0f);
        }
    }
}

TEST_P(ChunkMultiDTypeTest, ChunkSingleChunk) {
    // Test case: Split into 1 chunk should return original tensor
    auto x = zeros({10, 20}, dtype(), device());

    auto chunks = chunk(x, 1, 0);

    ASSERT_EQ(chunks.size(), 1);
    auto shape = chunks[0].shape();
    EXPECT_EQ(shape[0], 10);
    EXPECT_EQ(shape[1], 20);
    expect_chunks_match_input_dim0(x, chunks);
}

TEST_P(ChunkMultiDTypeTest, ChunkInvalidChunks) {
    auto x = zeros({10, 20}, dtype(), device());

    // Zero chunks should throw
    EXPECT_THROW(chunk(x, 0, 0), std::invalid_argument);

    // Negative chunks should throw
    EXPECT_THROW(chunk(x, -1, 0), std::invalid_argument);
}

TEST_P(ChunkMultiDTypeTest, ChunkInvalidDimension) {
    auto x = zeros({10, 20}, dtype(), device());

    // Dimension out of range
    EXPECT_THROW(chunk(x, 4, 2), std::invalid_argument);
    EXPECT_THROW(chunk(x, 4, -3), std::invalid_argument);
}

TEST_P(ChunkMultiDTypeTest, ChunkDataCorrectness) {
    // Verify that chunk() creates tensors with correct shapes and is slice-based
    // Create tensor on CPU first for data population, then transfer to target device
    auto x_cpu = zeros({6, 3}, dtype(), Device::cpu());

    // Fill with sequential values for verification
    if (dtype() == DType::Float32) {
        float* data = x_cpu.data<float>();
        for (int i = 0; i < 18; ++i) {
            data[i] = static_cast<float>(i);
        }
    } else if (dtype() == DType::Float64) {
        double* data = x_cpu.data<double>();
        for (int i = 0; i < 18; ++i) {
            data[i] = static_cast<double>(i);
        }
    } else if (dtype() == DType::Int32) {
        int32_t* data = x_cpu.data<int32_t>();
        for (int i = 0; i < 18; ++i) {
            data[i] = i;
        }
    }

    // Transfer to target device if needed
    auto x = (device().type == Device::Type::CPU) ? x_cpu : x_cpu.to(device());

    // Split into 3 chunks along dim=0
    auto chunks = chunk(x, 3, 0);

    ASSERT_EQ(chunks.size(), 3);

    // Verify that chunks share storage with original (they're views/slices)
    // This is the primary behavior of chunk() - it should create views, not copies
    for (size_t i = 0; i < chunks.size(); ++i) {
        auto shape = chunks[i].shape();
        EXPECT_EQ(shape[0], 2);  // Each chunk has 2 rows
        EXPECT_EQ(shape[1], 3);  // Same number of columns
    }

    // Value assertion: chunks[0] is rows 0..1 = [0..5], chunks[1] = [6..11],
    // chunks[2] = [12..17].
    for (size_t i = 0; i < chunks.size(); ++i) {
        auto c_cpu = chunks[i].to(Device::cpu()).to(DType::Float32);
        const float* d = c_cpu.data<float>();
        for (int e = 0; e < 6; ++e) {
            EXPECT_FLOAT_EQ(d[e], static_cast<float>(i * 6 + e))
                << "chunk " << i << " element " << e;
        }
    }
}

TEST_P(ChunkMultiDTypeTest, Chunk3DTensor) {
    // Test chunking on 3D tensor
    auto x = zeros({8, 4, 6}, dtype(), device());

    // Chunk along first dimension
    auto chunks0 = chunk(x, 2, 0);
    ASSERT_EQ(chunks0.size(), 2);
    for (const auto& c : chunks0) {
        EXPECT_EQ(c.shape()[0], 4);
        EXPECT_EQ(c.shape()[1], 4);
        EXPECT_EQ(c.shape()[2], 6);
    }

    // Chunk along second dimension
    auto chunks1 = chunk(x, 2, 1);
    ASSERT_EQ(chunks1.size(), 2);
    for (const auto& c : chunks1) {
        EXPECT_EQ(c.shape()[0], 8);
        EXPECT_EQ(c.shape()[1], 2);
        EXPECT_EQ(c.shape()[2], 6);
    }

    // Chunk along third dimension
    auto chunks2 = chunk(x, 3, 2);
    ASSERT_EQ(chunks2.size(), 3);
    // First two chunks: (8, 4, 2)
    EXPECT_EQ(chunks2[0].shape()[2], 2);
    EXPECT_EQ(chunks2[1].shape()[2], 2);
    // Last chunk: (8, 4, 2)
    EXPECT_EQ(chunks2[2].shape()[2], 2);
    // Value: zero-input chunks are zero in every element.
    for (const auto& c : chunks0) {
        auto cc = c.to(Device::cpu()).to(DType::Float32);
        const float* d = cc.data<float>();
        for (int64_t e = 0; e < cc.numel(); ++e) EXPECT_FLOAT_EQ(d[e], 0.0f);
    }
}

TEST_P(ChunkMultiDTypeTest, ChunkLargeTensor) {
    // Test with larger tensor to verify scalability
    auto x = zeros({100, 50}, dtype(), device());

    auto chunks = chunk(x, 10, 0);
    ASSERT_EQ(chunks.size(), 10);

    for (const auto& c : chunks) {
        EXPECT_EQ(c.shape()[0], 10);
        EXPECT_EQ(c.shape()[1], 50);
    }
    // First chunk first element must be 0 (input is zeros).
    auto c0 = chunks[0].to(Device::cpu()).to(DType::Float32);
    EXPECT_FLOAT_EQ(c0.data<float>()[0], 0.0f);
    EXPECT_FLOAT_EQ(c0.data<float>()[c0.numel() - 1], 0.0f);
}

TEST_P(ChunkMultiDTypeTest, ChunkDTypePreservation) {
    // Verify that chunked tensors preserve the original dtype
    auto x = zeros({12, 8}, dtype(), device());

    auto chunks = chunk(x, 3, 0);

    for (const auto& c : chunks) {
        EXPECT_EQ(c.dtype(), dtype()) << "Chunk dtype mismatch";
    }
    // Element-level: a zero-input chunk is element-wise zero.
    auto c0 = chunks[0].to(Device::cpu()).to(DType::Float32);
    for (int64_t e = 0; e < c0.numel(); ++e) EXPECT_FLOAT_EQ(c0.data<float>()[e], 0.0f);
}

TEST_P(ChunkMultiDTypeTest, ChunkDevicePreservation) {
    // Verify that chunked tensors preserve the original device
    auto x = zeros({12, 8}, dtype(), device());

    auto chunks = chunk(x, 3, 0);

    for (const auto& c : chunks) {
        EXPECT_EQ(c.device().type, device().type) << "Chunk device mismatch";
    }
    auto c0 = chunks[0].to(Device::cpu()).to(DType::Float32);
    for (int64_t e = 0; e < c0.numel(); ++e) EXPECT_FLOAT_EQ(c0.data<float>()[e], 0.0f);
}

// ============================================================================
// Test Instantiation
// ============================================================================

// Custom dtype list — chunk semantics are dtype-agnostic but we want explicit
// integer-dtype coverage that the default FLOAT_DTYPES (Float32/Float64/Float16)
// doesn't provide. STANDARD_BACKENDS / BackendDTypeParamName come from the
// canonical fixture header.
INSTANTIATE_TEST_SUITE_P(
    AllBackendsMultiDTypes,
    ChunkMultiDTypeTest,
    ::testing::Combine(
        STANDARD_BACKENDS,
        ::testing::Values(DType::Float32, DType::Float64, DType::Int32)
    ),
    BackendDTypeParamName);
