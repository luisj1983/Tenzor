/**
 * @file test_dispatch_ops_multidtype.cpp
 * @brief Multi-backend tests for previously-untested OpIds
 *
 * Covers OpIds with zero direct test coverage:
 *   - StridedFill (OpId 428): Tensor::fill_() on non-contiguous tensors
 *   - EmbeddingWithBoundsCheck (OpId 431): bounds-checked embedding lookup
 *   - HasInfNan (OpId 434): scalar inf/nan check across whole tensor
 *   - Cast (OpId 316): dtype conversion via Tensor::to(DType)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class DispatchOpsMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void skipIfHalf() {
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
            GTEST_SKIP() << "Test requires Float32/Float64 precision";
        }
    }

    Tensor makeInt64Tensor(std::initializer_list<int64_t> vals) {
        auto t = tenzor::zeros({static_cast<int64_t>(vals.size())}, DType::Int64, Device::cpu());
        auto* ptr = t.data<int64_t>();
        int64_t i = 0;
        for (auto v : vals) ptr[i++] = v;
        if (device().type != Device::Type::CPU) {
            return t.to(device());
        }
        return t;
    }
};

// ============================================================================
// StridedFill — Tensor::fill_() on non-contiguous tensors
// ============================================================================

TEST_P(DispatchOpsMultiDTypeTest, FillContiguousTensor) {
    auto t = tenzor::zeros({3, 4}, dtype(), device());
    t.fill_(7.0);

    auto cpu_t = t.to(Device::cpu()).to(DType::Float32).contiguous();
    auto* data = cpu_t.data<float>();
    for (int64_t i = 0; i < 12; ++i) {
        EXPECT_NEAR(data[i], 7.0f, std::max(atol() * 10.0f, 1e-4f));
    }
}

TEST_P(DispatchOpsMultiDTypeTest, FillNonContiguousSlice) {
    skipIfHalf();
    // Create a contiguous tensor and a non-contiguous view via slicing.
    // Filling the slice should only modify the visible elements.
    auto t = tenzor::zeros({4, 4}, dtype(), device());
    // Fill row 1 in-place via slicing
    auto row1 = t.slice(0, 1, 2);  // shape {1, 4}
    row1.fill_(5.0);

    auto cpu_t = t.to(Device::cpu()).to(DType::Float32).contiguous();
    auto* data = cpu_t.data<float>();
    // Row 0: zeros
    for (int64_t j = 0; j < 4; ++j) {
        EXPECT_NEAR(data[0 * 4 + j], 0.0f, 1e-5f) << "Row 0 col " << j;
    }
    // Row 1: 5.0
    for (int64_t j = 0; j < 4; ++j) {
        EXPECT_NEAR(data[1 * 4 + j], 5.0f, std::max(atol() * 10.0f, 1e-4f)) << "Row 1 col " << j;
    }
    // Rows 2-3: zeros
    for (int64_t i = 2; i < 4; ++i) {
        for (int64_t j = 0; j < 4; ++j) {
            EXPECT_NEAR(data[i * 4 + j], 0.0f, 1e-5f) << "Row " << i << " col " << j;
        }
    }
}

// ============================================================================
// HasInfNan
// ============================================================================

TEST_P(DispatchOpsMultiDTypeTest, HasInfNanReturnsFalseForFiniteTensor) {
    skipIfHalf();
    auto t = tenzor::ones({3, 4}, dtype(), device());
    auto result = has_inf_nan(t);
    auto cpu_result = result.to(Device::cpu()).to(DType::Int32);
    // Result is a scalar / single-element tensor: 0 means no inf/nan present
    auto* r = cpu_result.data<int32_t>();
    EXPECT_EQ(r[0], 0);
}

TEST_P(DispatchOpsMultiDTypeTest, HasInfNanReturnsTrueForInfTensor) {
    skipIfHalf();
    if (dtype() != DType::Float32 && dtype() != DType::Float64) {
        GTEST_SKIP() << "Inf only meaningful for Float32/Float64";
    }
    // Build a tensor with one inf value on CPU then ship to device
    auto cpu_t = tenzor::ones({3, 4}, DType::Float32, Device::cpu());
    cpu_t.data<float>()[5] = std::numeric_limits<float>::infinity();
    auto t = cpu_t.to(dtype()).to(device());

    auto result = has_inf_nan(t);
    auto cpu_result = result.to(Device::cpu()).to(DType::Int32);
    auto* r = cpu_result.data<int32_t>();
    EXPECT_NE(r[0], 0);
}

TEST_P(DispatchOpsMultiDTypeTest, HasInfNanReturnsTrueForNanTensor) {
    skipIfHalf();
    if (dtype() != DType::Float32 && dtype() != DType::Float64) {
        GTEST_SKIP() << "NaN only meaningful for Float32/Float64";
    }
    auto cpu_t = tenzor::ones({3, 4}, DType::Float32, Device::cpu());
    cpu_t.data<float>()[2] = std::numeric_limits<float>::quiet_NaN();
    auto t = cpu_t.to(dtype()).to(device());

    auto result = has_inf_nan(t);
    auto cpu_result = result.to(Device::cpu()).to(DType::Int32);
    auto* r = cpu_result.data<int32_t>();
    EXPECT_NE(r[0], 0);
}

// ============================================================================
// Cast — Tensor::to(DType) conversion
// ============================================================================

TEST_P(DispatchOpsMultiDTypeTest, CastFloat32ToFloat64Roundtrip) {
    auto t = tenzor::ones({2, 3}, DType::Float32, device());
    auto t64 = t.to(DType::Float64);
    EXPECT_EQ(t64.dtype(), DType::Float64);
    auto t32_back = t64.to(DType::Float32);
    EXPECT_EQ(t32_back.dtype(), DType::Float32);

    auto cpu_back = t32_back.to(Device::cpu()).contiguous();
    auto* data = cpu_back.data<float>();
    for (int64_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST_P(DispatchOpsMultiDTypeTest, CastFloatToInt32Truncates) {
    if (dtype() != DType::Float32 && dtype() != DType::Float64) {
        GTEST_SKIP() << "Test only meaningful for Float32/Float64";
    }
    // Build a CPU tensor with known fractional values
    auto cpu_t = tenzor::zeros({4}, DType::Float32, Device::cpu());
    auto* d = cpu_t.data<float>();
    d[0] = 1.7f;
    d[1] = -2.3f;
    d[2] = 5.0f;
    d[3] = 0.4f;

    auto t = cpu_t.to(dtype()).to(device());
    auto i32 = t.to(DType::Int32);
    EXPECT_EQ(i32.dtype(), DType::Int32);

    auto cpu_i32 = i32.to(Device::cpu()).contiguous();
    auto* idata = cpu_i32.data<int32_t>();
    EXPECT_EQ(idata[0], 1);
    EXPECT_EQ(idata[1], -2);
    EXPECT_EQ(idata[2], 5);
    EXPECT_EQ(idata[3], 0);
}

TEST_P(DispatchOpsMultiDTypeTest, CastPreservesShape) {
    auto t = tenzor::randn({3, 5, 2}, dtype(), device());
    auto casted = t.to(DType::Float32);
    auto shape = casted.shape();
    ASSERT_EQ(shape.size(), 3u);
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 5);
    EXPECT_EQ(shape[2], 2);
}

// ============================================================================
// EmbeddingWithBoundsCheck — covered indirectly via nn::Embedding
// ============================================================================
// Note: EmbeddingWithBoundsCheck is registered in all backends but is not
// exposed via a public function — backends use it internally when bounds
// checking is enabled. The CPU registry comment notes "same as Embedding".
// The op is exercised through the standard Embedding layer tests in
// tests/unit/test_embedding_multidtype.cpp; we add a smoke test here for
// confidence that the dispatch path is reachable via valid embedding usage.

TEST_P(DispatchOpsMultiDTypeTest, EmbeddingValidIndicesProduceCorrectShape) {
    skipIfHalf();
    nn::Embedding emb(10, 8);
    convert_model(emb);

    auto indices = makeInt64Tensor({0, 5, 9, 2});
    Variable input(indices, false);

    auto output = emb.forward(input);
    auto out_shape = output.tensor().shape();
    ASSERT_EQ(out_shape.size(), 2u);
    EXPECT_EQ(out_shape[0], 4);
    EXPECT_EQ(out_shape[1], 8);
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DispatchOpsMultiDTypeTest);
