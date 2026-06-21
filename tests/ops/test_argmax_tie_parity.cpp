/**
 * @file test_argmax_tie_parity.cpp
 * @brief Regression: argmax/argmin must return the LOWEST index on value ties,
 *        matching the CPU first-occurrence reference, on every backend.
 *
 * The CUDA argmax/argmin tree+warp combine only applied the lower-original-index
 * tie-break for NaN-vs-NaN; on a FINITE value tie it kept whichever operand sat
 * on the tree-left / current lane, so it could return a HIGHER index than CPU.
 * This builds tensors with duplicated max/min values (both full-reduce and
 * along-dim) and asserts CUDA == CPU index.
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>
#include <vector>

using namespace tenzor;
using namespace tenzor::testing;

class ArgmaxTieParityTest : public BackendTest {
protected:
    Tensor from_vec(const std::vector<float>& v, std::vector<int64_t> shape) {
        auto t = zeros(shape, DType::Float32, Device::cpu());
        std::copy(v.begin(), v.end(), t.data<float>());
        return (device.type == Device::Type::CPU) ? t : t.to(device);
    }
    int64_t idx_of(const Tensor& t) {
        auto c = t.to(Device::cpu()).contiguous();
        return c.dtype() == DType::Int64 ? c.data<int64_t>()[0]
                                         : static_cast<int64_t>(c.data<int32_t>()[0]);
    }
};

// Full reduction: max value appears at several positions; lowest index wins.
TEST_P(ArgmaxTieParityTest, ArgmaxFullReduce_LowestIndexOnTie) {
    // Max (5) at indices 2, 5, 9. CPU/PyTorch return 2.
    std::vector<float> v = {1, 3, 5, 2, 4, 5, 0, 1, 4, 5, 2, 3};
    auto t = from_vec(v, {12});
    EXPECT_EQ(idx_of(argmax(t)), 2) << "on " << device.to_string();
}

TEST_P(ArgmaxTieParityTest, ArgminFullReduce_LowestIndexOnTie) {
    // Min (-2) at indices 1, 4, 10. CPU/PyTorch return 1.
    std::vector<float> v = {3, -2, 0, 1, -2, 5, 4, 2, 3, 7, -2, 6};
    auto t = from_vec(v, {12});
    EXPECT_EQ(idx_of(argmin(t)), 1) << "on " << device.to_string();
}

// Along-dim reduction must match CPU index-by-index on ties.
TEST_P(ArgmaxTieParityTest, ArgmaxAlongDim_MatchesCPU) {
    // 3x4; each row has a tied max so the lowest column must be chosen.
    std::vector<float> v = {
        7, 2, 7, 1,   // row0: max 7 at col 0 (tie with col2) -> 0
        3, 9, 4, 9,   // row1: max 9 at col 1 (tie with col3) -> 1
        5, 5, 5, 5,   // row2: all tie -> 0
    };
    auto t_cpu = zeros({3, 4}, DType::Float32, Device::cpu());
    std::copy(v.begin(), v.end(), t_cpu.data<float>());
    auto ref = argmax(t_cpu, 1).to(Device::cpu()).contiguous();
    auto t_dev = (device.type == Device::Type::CPU) ? t_cpu : t_cpu.to(device);
    auto got = argmax(t_dev, 1).to(Device::cpu()).contiguous();
    const int64_t* r = ref.data<int64_t>();
    const int64_t* g = got.data<int64_t>();
    for (int64_t i = 0; i < 3; ++i)
        EXPECT_EQ(r[i], g[i]) << "row " << i << " on " << device.to_string();
    // Sanity: ground truth is the lowest tied column.
    EXPECT_EQ(r[0], 0); EXPECT_EQ(r[1], 1); EXPECT_EQ(r[2], 0);
}

INSTANTIATE_BACKEND_TESTS(ArgmaxTieParityTest);
