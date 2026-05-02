/**
 * @file test_advanced_index_parity.cpp
 * @brief Cross-backend parity tests for NumPy-style advanced (fancy) indexing.
 *
 * Covers OpId::AdvancedIndex (132) and OpId::AdvancedIndexPut (133). Per the
 * audit (2026-05-02), these had zero direct parity test references prior to
 * this file — only the wrapping `index` / `index_put` helpers were used by
 * tests indirectly through other ops.
 */

#include <gtest/gtest.h>
#include <optional>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/indexing.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class AdvancedIndexParity : public BackendTest {};

// ----------------------------------------------------------------------------
// AdvancedIndex (gather): out = input[idx_row, idx_col, ...]
// ----------------------------------------------------------------------------

// Single index dim — picks rows out of a 2-D matrix.
TEST_P(AdvancedIndexParity, GatherRowsByIndex) {
    auto data = randn({6, 4}, DType::Float32, Device::cpu());
    // Permute via argsort of randn — keeps the test deterministic per-run
    // while exercising non-identity gathers.
    auto perm_keys = randn({6}, DType::Float32, Device::cpu());
    auto perm = argsort(perm_keys, /*dim=*/0).to(DType::Int64);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        std::vector<std::optional<Tensor>> indices;
        indices.emplace_back(inputs[1]);
        return index(inputs[0], indices);
    }, {data, perm}, device, 1e-5f, 1e-7f, "AdvancedIndex_RowsByIndex");
}

// Two parallel index dims — gathers (i, j) pairs.
TEST_P(AdvancedIndexParity, GatherPairs2D) {
    auto data = randn({5, 7}, DType::Float32, Device::cpu());
    auto rows = full({4}, 0.0, DType::Int64, Device::cpu());
    rows.data<int64_t>()[0] = 0;
    rows.data<int64_t>()[1] = 2;
    rows.data<int64_t>()[2] = 4;
    rows.data<int64_t>()[3] = 1;
    auto cols = full({4}, 0.0, DType::Int64, Device::cpu());
    cols.data<int64_t>()[0] = 6;
    cols.data<int64_t>()[1] = 3;
    cols.data<int64_t>()[2] = 0;
    cols.data<int64_t>()[3] = 5;

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        std::vector<std::optional<Tensor>> indices;
        indices.emplace_back(inputs[1]);
        indices.emplace_back(inputs[2]);
        return index(inputs[0], indices);
    }, {data, rows, cols}, device, 1e-5f, 1e-7f, "AdvancedIndex_Pairs2D");
}

// Mixed: index first dim, leave second dim full (null in indices vector).
TEST_P(AdvancedIndexParity, GatherFirstDimOnly) {
    auto data = randn({4, 6}, DType::Float32, Device::cpu());
    auto rows = full({3}, 0.0, DType::Int64, Device::cpu());
    rows.data<int64_t>()[0] = 1;
    rows.data<int64_t>()[1] = 3;
    rows.data<int64_t>()[2] = 0;

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        std::vector<std::optional<Tensor>> indices;
        indices.emplace_back(inputs[1]);     // index dim 0
        indices.emplace_back(std::nullopt);  // full slice on dim 1
        return index(inputs[0], indices);
    }, {data, rows}, device, 1e-5f, 1e-7f, "AdvancedIndex_FirstDim");
}

// Broadcasting: row indices shape {3, 1}, col indices shape {1, 4} → output {3, 4}.
TEST_P(AdvancedIndexParity, GatherBroadcastIndices) {
    auto data = randn({5, 5}, DType::Float32, Device::cpu());
    auto rows = full({3, 1}, 0.0, DType::Int64, Device::cpu());
    rows.data<int64_t>()[0] = 0;
    rows.data<int64_t>()[1] = 2;
    rows.data<int64_t>()[2] = 4;
    auto cols = full({1, 4}, 0.0, DType::Int64, Device::cpu());
    cols.data<int64_t>()[0] = 1;
    cols.data<int64_t>()[1] = 3;
    cols.data<int64_t>()[2] = 0;
    cols.data<int64_t>()[3] = 2;

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        std::vector<std::optional<Tensor>> indices;
        indices.emplace_back(inputs[1]);
        indices.emplace_back(inputs[2]);
        return index(inputs[0], indices);
    }, {data, rows, cols}, device, 1e-5f, 1e-7f, "AdvancedIndex_Broadcast");
}

// ----------------------------------------------------------------------------
// AdvancedIndexPut (in-place scatter): input[idx_row, idx_col, ...] = values
// ----------------------------------------------------------------------------

TEST_P(AdvancedIndexParity, ScatterRowsByIndex) {
    auto data = zeros({6, 4}, DType::Float32, Device::cpu());
    auto idx = arange(0, 3, 1, DType::Int64, Device::cpu());  // {0, 1, 2}
    auto vals = randn({3, 4}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto target = inputs[0].clone();
        std::vector<std::optional<Tensor>> indices;
        indices.emplace_back(inputs[1]);
        index_put(target, indices, inputs[2]);
        return target;
    }, {data, idx, vals}, device, 1e-5f, 1e-7f, "AdvancedIndexPut_Rows");
}

TEST_P(AdvancedIndexParity, ScatterPairs2D) {
    auto data = full({5, 7}, 1.0, DType::Float32, Device::cpu());
    auto rows = full({4}, 0.0, DType::Int64, Device::cpu());
    rows.data<int64_t>()[0] = 0;
    rows.data<int64_t>()[1] = 2;
    rows.data<int64_t>()[2] = 4;
    rows.data<int64_t>()[3] = 1;
    auto cols = full({4}, 0.0, DType::Int64, Device::cpu());
    cols.data<int64_t>()[0] = 6;
    cols.data<int64_t>()[1] = 3;
    cols.data<int64_t>()[2] = 0;
    cols.data<int64_t>()[3] = 5;
    auto vals = randn({4}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto target = inputs[0].clone();
        std::vector<std::optional<Tensor>> indices;
        indices.emplace_back(inputs[1]);
        indices.emplace_back(inputs[2]);
        index_put(target, indices, inputs[3]);
        return target;
    }, {data, rows, cols, vals}, device, 1e-5f, 1e-7f, "AdvancedIndexPut_Pairs2D");
}

// Mixed: scatter into first dim only — broadcasting `values` across remaining dims.
TEST_P(AdvancedIndexParity, ScatterFirstDimOnly) {
    auto data = zeros({4, 6}, DType::Float32, Device::cpu());
    auto rows = full({2}, 0.0, DType::Int64, Device::cpu());
    rows.data<int64_t>()[0] = 1;
    rows.data<int64_t>()[1] = 3;
    auto vals = randn({2, 6}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto target = inputs[0].clone();
        std::vector<std::optional<Tensor>> indices;
        indices.emplace_back(inputs[1]);
        indices.emplace_back(std::nullopt);
        index_put(target, indices, inputs[2]);
        return target;
    }, {data, rows, vals}, device, 1e-5f, 1e-7f, "AdvancedIndexPut_FirstDim");
}

INSTANTIATE_BACKEND_TESTS(AdvancedIndexParity);

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
