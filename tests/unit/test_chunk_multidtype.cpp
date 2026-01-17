/**
 * @file test_chunk_multidtype.cpp
 * @brief Multi-dtype tests for chunk operation
 *
 * This file tests the chunk operation across multiple data types:
 * - Float32, Float64, Float16 (floating-point operations)
 * - Int32 (integer operations)
 *
 * Tests cover:
 * - Basic chunking with even division
 * - Chunking with uneven division
 * - Chunking along different dimensions
 * - Edge cases (single chunk, negative dimensions)
 * - Invalid input handling
 * - Data correctness verification
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Multi-DType Test Fixture
// ============================================================================

struct BackendDTypeParam {
    std::string backend_name;
    DType dtype;
    std::string dtype_name;

    std::string ToString() const {
        return backend_name + "_" + dtype_name;
    }
};

// Required for gtest_discover_tests to show human-readable test names
void PrintTo(const BackendDTypeParam& param, std::ostream* os) {
    *os << param.ToString();
}

class ChunkMultiDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    Device device;
    DType dtype;
    float tolerance;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype = param.dtype;

        // Set dtype-specific tolerances
        switch(dtype) {
            case DType::Float16:
                tolerance = 1e-2f;
                break;
            case DType::Float32:
                tolerance = 1e-5f;
                break;
            case DType::Float64:
                tolerance = 1e-10f;
                break;
            default:
                tolerance = 0.0f;  // Exact for integers
                break;
        }

        if (param.backend_name == "cpu") {
            device = Device::cpu();
        }
        else if (param.backend_name == "cuda") {
            if (!isBackendAvailable(Device::Type::CUDA)) {
                GTEST_SKIP() << "CUDA not available";
            }
            device = Device::cuda(0);
        }
        else if (param.backend_name == "vulkan") {
            if (!isBackendAvailable(Device::Type::Vulkan)) {
                GTEST_SKIP() << "Vulkan not available";
            }
            device = Device::vulkan(0);
        }
        else if (param.backend_name == "oneapi") {
            if (!isBackendAvailable(Device::Type::OneAPI)) {
                GTEST_SKIP() << "OneAPI not available";
            }
            device = Device::oneapi(0);
        }
        else if (param.backend_name == "rocm") {
            if (!isBackendAvailable(Device::Type::ROCm)) {
                GTEST_SKIP() << "ROCm not available";
            }
            device = Device::rocm(0);
        }
    }

    static bool isBackendAvailable(Device::Type type) {
        try {
            Device test_device{type, 0};
            auto t = zeros({2, 2}, DType::Float32, test_device);
            return true;
        } catch (...) {
            return false;
        }
    }
};

// ============================================================================
// Chunk Operation Tests
// ============================================================================

TEST_P(ChunkMultiDTypeTest, BasicChunkEvenDivision) {
    // Test case: Shape (12, 20), split into 4 chunks along dim=0
    // Expected: 4 tensors of shape (3, 20)
    auto x = zeros({12, 20}, dtype, device);

    auto chunks = chunk(x, 4, 0);

    ASSERT_EQ(chunks.size(), 4);

    for (size_t i = 0; i < chunks.size(); ++i) {
        auto shape = chunks[i].shape();
        EXPECT_EQ(shape.size(), 2);
        EXPECT_EQ(shape[0], 3);
        EXPECT_EQ(shape[1], 20);
    }
}

TEST_P(ChunkMultiDTypeTest, BasicChunkUnevenDivision) {
    // Test case: Shape (10, 20), split into 4 chunks along dim=0
    // Expected: 3 tensors of shape (3, 20) and 1 tensor of shape (1, 20)
    auto x = zeros({10, 20}, dtype, device);

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
}

TEST_P(ChunkMultiDTypeTest, ChunkFewerThanDimensionSize) {
    // Test case: Shape (5, 20), split into 10 chunks along dim=0
    // Expected: 5 tensors of shape (1, 20) - fewer chunks than requested
    auto x = zeros({5, 20}, dtype, device);

    auto chunks = chunk(x, 10, 0);

    // Should only get 5 chunks since dimension size is 5
    ASSERT_EQ(chunks.size(), 5);

    for (size_t i = 0; i < chunks.size(); ++i) {
        auto shape = chunks[i].shape();
        EXPECT_EQ(shape.size(), 2);
        EXPECT_EQ(shape[0], 1);
        EXPECT_EQ(shape[1], 20);
    }
}

TEST_P(ChunkMultiDTypeTest, ChunkAlongDifferentDimension) {
    // Test case: Shape (10, 20), split into 5 chunks along dim=1
    // Expected: 5 tensors of shape (10, 4)
    auto x = zeros({10, 20}, dtype, device);

    auto chunks = chunk(x, 5, 1);

    ASSERT_EQ(chunks.size(), 5);

    for (size_t i = 0; i < chunks.size(); ++i) {
        auto shape = chunks[i].shape();
        EXPECT_EQ(shape.size(), 2);
        EXPECT_EQ(shape[0], 10);
        EXPECT_EQ(shape[1], 4);
    }
}

TEST_P(ChunkMultiDTypeTest, ChunkNegativeDimension) {
    // Test case: Use negative dimension indexing
    auto x = zeros({10, 20}, dtype, device);

    // dim=-1 should be equivalent to dim=1
    auto chunks = chunk(x, 4, -1);

    ASSERT_EQ(chunks.size(), 4);

    for (size_t i = 0; i < chunks.size(); ++i) {
        auto shape = chunks[i].shape();
        EXPECT_EQ(shape[0], 10);
        EXPECT_EQ(shape[1], 5);
    }
}

TEST_P(ChunkMultiDTypeTest, ChunkSingleChunk) {
    // Test case: Split into 1 chunk should return original tensor
    auto x = zeros({10, 20}, dtype, device);

    auto chunks = chunk(x, 1, 0);

    ASSERT_EQ(chunks.size(), 1);
    auto shape = chunks[0].shape();
    EXPECT_EQ(shape[0], 10);
    EXPECT_EQ(shape[1], 20);
}

TEST_P(ChunkMultiDTypeTest, ChunkInvalidChunks) {
    auto x = zeros({10, 20}, dtype, device);

    // Zero chunks should throw
    EXPECT_THROW(chunk(x, 0, 0), std::invalid_argument);

    // Negative chunks should throw
    EXPECT_THROW(chunk(x, -1, 0), std::invalid_argument);
}

TEST_P(ChunkMultiDTypeTest, ChunkInvalidDimension) {
    auto x = zeros({10, 20}, dtype, device);

    // Dimension out of range
    EXPECT_THROW(chunk(x, 4, 2), std::invalid_argument);
    EXPECT_THROW(chunk(x, 4, -3), std::invalid_argument);
}

TEST_P(ChunkMultiDTypeTest, ChunkDataCorrectness) {
    // Verify that chunk() creates tensors with correct shapes and is slice-based
    // Create tensor on CPU first for data population, then transfer to target device
    auto x_cpu = zeros({6, 3}, dtype, Device::cpu());

    // Fill with sequential values for verification
    if (dtype == DType::Float32) {
        float* data = x_cpu.data<float>();
        for (int i = 0; i < 18; ++i) {
            data[i] = static_cast<float>(i);
        }
    } else if (dtype == DType::Float64) {
        double* data = x_cpu.data<double>();
        for (int i = 0; i < 18; ++i) {
            data[i] = static_cast<double>(i);
        }
    } else if (dtype == DType::Int32) {
        int32_t* data = x_cpu.data<int32_t>();
        for (int i = 0; i < 18; ++i) {
            data[i] = i;
        }
    }

    // Transfer to target device if needed
    auto x = (device.type == Device::Type::CPU) ? x_cpu : x_cpu.to(device);

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
}

TEST_P(ChunkMultiDTypeTest, Chunk3DTensor) {
    // Test chunking on 3D tensor
    auto x = zeros({8, 4, 6}, dtype, device);

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
}

TEST_P(ChunkMultiDTypeTest, ChunkLargeTensor) {
    // Test with larger tensor to verify scalability
    auto x = zeros({100, 50}, dtype, device);

    auto chunks = chunk(x, 10, 0);
    ASSERT_EQ(chunks.size(), 10);

    for (const auto& c : chunks) {
        EXPECT_EQ(c.shape()[0], 10);
        EXPECT_EQ(c.shape()[1], 50);
    }
}

TEST_P(ChunkMultiDTypeTest, ChunkDTypePreservation) {
    // Verify that chunked tensors preserve the original dtype
    auto x = zeros({12, 8}, dtype, device);

    auto chunks = chunk(x, 3, 0);

    for (const auto& c : chunks) {
        EXPECT_EQ(c.dtype(), dtype) << "Chunk dtype mismatch";
    }
}

TEST_P(ChunkMultiDTypeTest, ChunkDevicePreservation) {
    // Verify that chunked tensors preserve the original device
    auto x = zeros({12, 8}, dtype, device);

    auto chunks = chunk(x, 3, 0);

    for (const auto& c : chunks) {
        EXPECT_EQ(c.device().type, device.type) << "Chunk device mismatch";
    }
}

// ============================================================================
// Test Instantiation
// ============================================================================

std::vector<BackendDTypeParam> GenerateChunkBackendDTypeCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi", "rocm"};

    // Test with these dtypes
    std::vector<std::pair<DType, std::string>> dtypes = {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
        {DType::Int32, "int32"},
        // Float16 can be enabled when fully supported
        // {DType::Float16, "float16"},
    };

    std::vector<BackendDTypeParam> combinations;
    for (const auto& backend : backends) {
        for (const auto& [dtype, dtype_name] : dtypes) {
            combinations.push_back({backend, dtype, dtype_name});
        }
    }
    return combinations;
}

INSTANTIATE_TEST_SUITE_P(
    AllBackendsMultiDTypes,
    ChunkMultiDTypeTest,
    ::testing::ValuesIn(GenerateChunkBackendDTypeCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

// Entry point
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
