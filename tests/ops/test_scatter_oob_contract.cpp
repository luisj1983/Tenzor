/**
 * @file test_scatter_oob_contract.cpp
 * @brief Regression: scatter / scatter_add must enforce the PyTorch shape
 *        contract on every backend (notably CUDA).
 *
 * The CUDA scatter / scatter_add launchers previously validated only the
 * scatter index *value* against dim_size; they did NOT check
 *   (a) src.numel() >= index.numel()  (else OOB device read of src), or
 *   (b) index.shape()[d] <= input.shape()[d] on every non-scatter axis
 *       (else the per-dim decode produces an output_offset past
 *        output.numel() -> OOB device write / atomicAdd).
 * The CPU reference throws std::invalid_argument / std::out_of_range for both.
 * This test asserts the CUDA path now throws too (parity), instead of silently
 * corrupting GPU memory.
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/indexing.hpp>

using namespace tenzor;
using namespace tenzor::testing;

class ScatterOOBContractTest : public BackendTest {};

// index has MORE elements than src -> OOB src read on CUDA without the check.
TEST_P(ScatterOOBContractTest, IndexLargerThanSrc_Throws) {
    auto input = zeros({4, 4}, DType::Float32, device);
    // index [4,4] (16 elems), src [4,2] (8 elems): index.numel() > src.numel().
    auto index = zeros({4, 4}, DType::Int64, device);
    auto src   = ones({4, 2}, DType::Float32, device);
    EXPECT_THROW({ auto r = scatter(input, 1, index, src); device.synchronize(); },
                 std::exception);
    EXPECT_THROW({ auto r = scatter_add(input, 1, index, src); device.synchronize(); },
                 std::exception);
}

// index extent on a NON-scatter axis exceeds input's -> OOB output write.
TEST_P(ScatterOOBContractTest, IndexExceedsInputOnNonScatterAxis_Throws) {
    auto input = zeros({3, 4}, DType::Float32, device);
    // scatter along dim=1; index rows (axis 0) = 5 > input rows (3).
    auto index = zeros({5, 4}, DType::Int64, device);
    auto src   = ones({5, 4}, DType::Float32, device);
    EXPECT_THROW({ auto r = scatter(input, 1, index, src); device.synchronize(); },
                 std::exception);
    EXPECT_THROW({ auto r = scatter_add(input, 1, index, src); device.synchronize(); },
                 std::exception);
}

// Sanity: a well-formed scatter still succeeds (the new checks don't over-reject).
TEST_P(ScatterOOBContractTest, ValidScatter_Succeeds) {
    auto input = zeros({3, 4}, DType::Float32, device);
    auto index = zeros({3, 4}, DType::Int64, device);  // all -> column 0
    auto src   = ones({3, 4}, DType::Float32, device);
    Tensor out;
    EXPECT_NO_THROW({ out = scatter(input, 1, index, src); device.synchronize(); });
    auto out_cpu = out.to(Device::cpu()).contiguous();
    const float* d = out_cpu.data<float>();
    // Column 0 of each row receives 1 (last write wins); others stay 0.
    for (int64_t r = 0; r < 3; ++r) {
        EXPECT_FLOAT_EQ(d[r * 4 + 0], 1.0f);
        for (int64_t c = 1; c < 4; ++c) EXPECT_FLOAT_EQ(d[r * 4 + c], 0.0f);
    }
}

INSTANTIATE_BACKEND_TESTS(ScatterOOBContractTest);
