/**
 * @file test_indexing_operator_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for indexing operations
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/indexing.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include "multi_backend_dtype_fixture.hpp"

#include <vector>

using namespace tenzor;
using namespace tenzor::testing;

class IndexingOpMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Audit-T.1: deterministic 4x3 / 8x4 / 10x5 builder so the expected
    // slice/select/narrow values are derivable inline.
    Tensor cpuRange(const std::vector<int64_t>& shape) {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        auto t = tenzor::full(shape, 0.0f, DType::Float32, Device::cpu());
        auto* p = t.data<float>();
        for (int64_t i = 0; i < n; ++i) p[i] = static_cast<float>(i);
        return t;
    }

    // Cast cpuRange to the parametrized dtype/device.
    Tensor rangeOnDevice(const std::vector<int64_t>& shape) {
        return cpuRange(shape).to(dtype()).to(device());
    }
};

TEST_P(IndexingOpMultiDTypeTest, SelectDim0) {
    // Audit-T.1: select(row 2) of a 4x3 range tensor = [6, 7, 8].
    auto cpu_t = cpuRange({4, 3});
    auto t = cpu_t.to(dtype()).to(device());
    auto result = tenzor::select(t, 0, 2);
    expectShape(result, {3});
    expectDevice(result);
    expectDType(result);

    auto expected_cpu = tenzor::full({3}, 0.0f, DType::Float32, Device::cpu());
    auto* ep = expected_cpu.data<float>();
    ep[0] = 6.0f; ep[1] = 7.0f; ep[2] = 8.0f;
    expectTensorNear(result, expected_cpu, atol());
}

TEST_P(IndexingOpMultiDTypeTest, SelectDim1) {
    // Audit-T.1: select column 1 of a 4x3 range tensor = [1, 4, 7, 10].
    auto cpu_t = cpuRange({4, 3});
    auto t = cpu_t.to(dtype()).to(device());
    auto result = tenzor::select(t, 1, 1);
    expectShape(result, {4});

    auto expected_cpu = tenzor::full({4}, 0.0f, DType::Float32, Device::cpu());
    auto* ep = expected_cpu.data<float>();
    ep[0] = 1.0f; ep[1] = 4.0f; ep[2] = 7.0f; ep[3] = 10.0f;
    expectTensorNear(result, expected_cpu, atol());
}

TEST_P(IndexingOpMultiDTypeTest, NarrowDim0) {
    // Audit-T.1: narrow rows [2,6) from 10x5 range = rows 2..5,
    // i.e. values 10..29 in row-major order.
    auto cpu_t = cpuRange({10, 5});
    auto t = cpu_t.to(dtype()).to(device());
    auto result = tenzor::narrow(t, 0, 2, 4);
    expectShape(result, {4, 5});
    expectDevice(result);

    // Cross-check by running narrow on the CPU original — same op, same
    // semantics — and comparing element-wise.
    auto expected = tenzor::narrow(cpu_t, 0, 2, 4);
    expectTensorNear(result, expected, atol());
}

TEST_P(IndexingOpMultiDTypeTest, SliceBasic) {
    // Audit-T.1: slice along dim 0 from 1 to 5 of an 8x4 range tensor =
    // rows 1..4.  Reference produced by running the same op on CPU.
    auto cpu_t = cpuRange({8, 4});
    auto t = cpu_t.to(dtype()).to(device());
    auto result = tenzor::slice(t, 0, 1, 5);
    expectShape(result, {4, 4});

    auto expected = tenzor::slice(cpu_t, 0, 1, 5);
    expectTensorNear(result, expected, atol());
}

TEST_P(IndexingOpMultiDTypeTest, WhereCondition) {
    // Audit-T.1: replace the previous randn-based check with a deterministic
    // condition mask so the output is exactly comparable.
    auto cond_cpu = tenzor::full({3, 4}, 0.0f, DType::Float32, Device::cpu());
    auto* cp = cond_cpu.data<float>();
    // Checkerboard: row+col even => 1.0 (true after gt zero), else 0.0.
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            cp[r * 4 + c] = ((r + c) % 2 == 0) ? 1.0f : 0.0f;

    auto cond_typed = cond_cpu.to(dtype()).to(device());
    auto cond = tenzor::gt(cond_typed,
                           tenzor::zeros({3, 4}, dtype(), device()));
    auto x = createOnes({3, 4});
    auto y = tenzor::zeros({3, 4}, dtype(), device());
    auto result = tenzor::where(cond, x, y);
    expectShape(result, {3, 4});
    expectDevice(result);

    // Expected: same checkerboard with 1.0 where condition holds.
    expectTensorNear(result, cond_cpu, atol());
}

TEST_P(IndexingOpMultiDTypeTest, MaskedFill) {
    // Audit-T.1: deterministic input + deterministic mask so we know the
    // exact output rather than relying on randn() comparability.
    auto cpu_t = tenzor::full({4, 4}, 0.0f, DType::Float32, Device::cpu());
    auto* tp = cpu_t.data<float>();
    for (int i = 0; i < 16; ++i) tp[i] = static_cast<float>(i + 1);

    // Mask the upper-triangular (strictly above diagonal) entries.
    auto mask_cpu = tenzor::full({4, 4}, 0.0f, DType::Float32, Device::cpu());
    auto* mp = mask_cpu.data<float>();
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            mp[r * 4 + c] = (c > r) ? 1.0f : 0.0f;

    auto t = cpu_t.to(dtype()).to(device());
    auto mask = tenzor::gt(mask_cpu.to(dtype()).to(device()),
                           tenzor::zeros({4, 4}, dtype(), device()));
    auto result = tenzor::masked_fill(t, mask, 0.0f);
    expectShape(result, {4, 4});
    expectDevice(result);

    // Expected: lower triangle keeps original values, upper triangle zero.
    auto expected_cpu =
        tenzor::full({4, 4}, 0.0f, DType::Float32, Device::cpu());
    auto* ep = expected_cpu.data<float>();
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            ep[r * 4 + c] = (c > r) ? 0.0f : static_cast<float>(r * 4 + c + 1);
    expectTensorNear(result, expected_cpu, atol());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(IndexingOpMultiDTypeTest);
