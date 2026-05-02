/**
 * @file test_index_scatter_parity.cpp
 * @brief Cross-backend parity tests for scatter / take_along_dim ops.
 *
 * Covers: OpId::SelectScatter (414), SliceScatter (415), DiagonalScatter (416),
 * MaskedScatter (611), TakeAlongDim (610). The audit (2026-05-02) flagged
 * these as having no dedicated parity coverage even though they were
 * registered on every non-MPS backend.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/indexing.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class IndexScatterParity : public BackendTest {};

// ----------------------------------------------------------------------------
// SelectScatter — replace the slice at dim/index with src
// ----------------------------------------------------------------------------

TEST_P(IndexScatterParity, SelectScatter_Dim0) {
    auto input = randn({4, 6}, DType::Float32, Device::cpu());
    auto src = randn({6}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return select_scatter(inputs[0], inputs[1], /*dim=*/0, /*index=*/2);
    }, {input, src}, device, 1e-5f, 1e-7f, "SelectScatter_Dim0");
}

TEST_P(IndexScatterParity, SelectScatter_Dim1Negative) {
    auto input = randn({3, 4, 5}, DType::Float32, Device::cpu());
    auto src = randn({3, 5}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        // dim=-2 = dim 1 of a 3-D tensor.
        return select_scatter(inputs[0], inputs[1], /*dim=*/-2, /*index=*/1);
    }, {input, src}, device, 1e-5f, 1e-7f, "SelectScatter_Dim1Neg");
}

// ----------------------------------------------------------------------------
// SliceScatter — replace the [start:end:step] slice with src
// ----------------------------------------------------------------------------

TEST_P(IndexScatterParity, SliceScatter_FullSpan) {
    auto input = randn({4, 6}, DType::Float32, Device::cpu());
    auto src = randn({4, 6}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return slice_scatter(inputs[0], inputs[1], /*dim=*/0,
                             /*start=*/0, /*end=*/4, /*step=*/1);
    }, {input, src}, device, 1e-5f, 1e-7f, "SliceScatter_FullSpan");
}

TEST_P(IndexScatterParity, SliceScatter_Stride2) {
    auto input = randn({8, 4}, DType::Float32, Device::cpu());
    auto src = randn({4, 4}, DType::Float32, Device::cpu());  // 8 rows, every 2nd

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return slice_scatter(inputs[0], inputs[1], /*dim=*/0,
                             /*start=*/0, /*end=*/8, /*step=*/2);
    }, {input, src}, device, 1e-5f, 1e-7f, "SliceScatter_Stride2");
}

// ----------------------------------------------------------------------------
// DiagonalScatter — place a 1-D src along the (dim1, dim2) diagonal
// ----------------------------------------------------------------------------

TEST_P(IndexScatterParity, DiagonalScatter_MainDiag) {
    auto input = randn({5, 5}, DType::Float32, Device::cpu());
    auto src = randn({5}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return diagonal_scatter(inputs[0], inputs[1], /*offset=*/0,
                                /*dim1=*/0, /*dim2=*/1);
    }, {input, src}, device, 1e-5f, 1e-7f, "DiagonalScatter_Main");
}

TEST_P(IndexScatterParity, DiagonalScatter_OffsetPositive) {
    auto input = randn({5, 7}, DType::Float32, Device::cpu());
    auto src = randn({5}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return diagonal_scatter(inputs[0], inputs[1], /*offset=*/2);
    }, {input, src}, device, 1e-5f, 1e-7f, "DiagonalScatter_OffsetPlus2");
}

TEST_P(IndexScatterParity, DiagonalScatter_OffsetNegative) {
    auto input = randn({6, 6}, DType::Float32, Device::cpu());
    auto src = randn({4}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return diagonal_scatter(inputs[0], inputs[1], /*offset=*/-2);
    }, {input, src}, device, 1e-5f, 1e-7f, "DiagonalScatter_OffsetMinus2");
}

// ----------------------------------------------------------------------------
// MaskedScatter — copy `source` into positions where `mask` is true
// ----------------------------------------------------------------------------

TEST_P(IndexScatterParity, MaskedScatter_Half) {
    auto input = randn({4, 8}, DType::Float32, Device::cpu());
    // Build a deterministic mask: select every other element.
    Tensor mask_bool = full({4, 8}, 0.0, DType::Bool, Device::cpu());
    bool* mask_data = mask_bool.data<bool>();
    for (int64_t i = 0; i < 32; ++i) mask_data[i] = (i % 2) == 0;
    // source must have at least as many elements as mask is true;
    // 16 elements covers the 16 true positions.
    auto source = randn({16}, DType::Float32, Device::cpu());

    // Run CPU and target_device manually so we can dump the values when
    // they differ — masked_scatter has had a Vulkan workgroup-prefix-sum
    // bug that the dispatch-table probe at the bottom of this file
    // surfaced. Keeping the manual diff pipeline on by default makes any
    // future regression on the same path easy to spot in the test log.
    auto cpu_result = masked_scatter(input, mask_bool, source);
    if (device.type == Device::Type::CPU) return;
    auto dev_result = masked_scatter(input.to(device), mask_bool.to(device), source.to(device));
    device.synchronize();
    auto dev_on_cpu = dev_result.to(Device::cpu());

    int64_t mismatch = 0;
    for (int64_t i = 0; i < 32; ++i) {
        if (std::abs(cpu_result.data<float>()[i] - dev_on_cpu.data<float>()[i]) > 1e-5f) {
            ++mismatch;
        }
    }
    ASSERT_EQ(mismatch, 0)
        << "MaskedScatter_Half mismatch on " << backend_name(device)
        << ": " << mismatch << "/32 positions disagree with CPU.";
}

// ----------------------------------------------------------------------------
// TakeAlongDim — gather elements along a specific dim using indices
// ----------------------------------------------------------------------------

TEST_P(IndexScatterParity, TakeAlongDim_LastDim) {
    auto input = randn({4, 8}, DType::Float32, Device::cpu());
    // For each row, pick the indices [0, 2, 5] from columns.
    auto indices_row = full({3}, 0.0, DType::Int64, Device::cpu());
    indices_row.data<int64_t>()[0] = 0;
    indices_row.data<int64_t>()[1] = 2;
    indices_row.data<int64_t>()[2] = 5;
    auto indices = indices_row.unsqueeze(0).expand({4, 3}).contiguous();

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return take_along_dim(inputs[0], inputs[1], /*dim=*/-1);
    }, {input, indices}, device, 1e-5f, 1e-7f, "TakeAlongDim_LastDim");
}

TEST_P(IndexScatterParity, TakeAlongDim_FirstDim) {
    auto input = randn({6, 4}, DType::Float32, Device::cpu());
    // For each column pick rows [1, 3, 5] (top-3 stand-in).
    auto indices_col = full({3}, 0.0, DType::Int64, Device::cpu());
    indices_col.data<int64_t>()[0] = 1;
    indices_col.data<int64_t>()[1] = 3;
    indices_col.data<int64_t>()[2] = 5;
    auto indices = indices_col.unsqueeze(1).expand({3, 4}).contiguous();

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return take_along_dim(inputs[0], inputs[1], /*dim=*/0);
    }, {input, indices}, device, 1e-5f, 1e-7f, "TakeAlongDim_FirstDim");
}

INSTANTIATE_BACKEND_TESTS(IndexScatterParity);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
