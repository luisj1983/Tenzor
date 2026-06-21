/**
 * @file test_math_half_dispatch.cpp
 * @brief Stream S2: math kernels must accept Float16 / BFloat16 inputs.
 *
 * The CPU math.cpp kernel file had ~40 callsites of
 * `TENZOR_DISPATCH_FLOATING_TYPES` (Float32 / Float64 only). Calling any of
 * these kernels with a half-precision tensor used to throw "unsupported
 * dtype (expected floating point)" from dtype_dispatch.hpp. The S2 stream
 * fixed every callsite either by wrapping the kernel entry with
 * `tenzor::utils::widen_narrow_compute` (for transcendentals and
 * multi-element accumulators) or by relying on an existing widen-narrow
 * guard already present in the kernel.
 *
 * This file is a regression net: each op must NOT throw on Float16 /
 * BFloat16 input, must return a tensor of the same half-precision dtype,
 * and must match the Float32 reference within Float16 tolerance for a
 * representative spot-check. It runs across every available backend.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/core/dtype.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/ops/indexing.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/activations/activations.hpp>

#include "backend_test_fixture.hpp"

#include <cmath>
#include <vector>

namespace {

using tenzor::DType;
using tenzor::Tensor;
using tenzor::Float16;
using tenzor::BFloat16;

class MathHalfDispatch : public ::tenzor::testing::BackendTest {};

// Helper to build a half-precision tensor with a single value on the given
// device. The host buffer is filled on CPU (data<T>() requires host memory),
// then moved to the target device.
template <typename HalfT>
auto make_half(const std::vector<int64_t>& shape, float value, DType dt,
               const tenzor::Device& device) -> Tensor {
    auto t = Tensor(shape, dt, tenzor::Device::cpu());
    auto* d = t.data<HalfT>();
    int64_t n = t.numel();
    HalfT h(value);
    for (int64_t i = 0; i < n; ++i) d[i] = h;
    return t.to(device);
}

// ----------------------------------------------------------------------------
// Per-op acceptance tests: for every op touched in S2, verify Float16 and
// BFloat16 inputs do NOT throw and produce a tensor with the matching dtype.
//
// Inputs are deliberately chosen to lie in the domain of each op:
//   asin / acos   -> values in [-1, 1]
//   everything else -> values around 0.5 work universally
// ----------------------------------------------------------------------------

// ---- Transcendentals (approach B: widen-narrow wrap) -----------------------

TEST_P(MathHalfDispatch, AsinAcceptsFloat16) {
    auto x = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::asin(x);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, AsinAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::asin(x);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, AcosAcceptsFloat16) {
    auto x = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::acos(x);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, AcosAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::acos(x);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, AtanAcceptsFloat16) {
    auto x = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::atan(x);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, AtanAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::atan(x);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, SinhAcceptsFloat16) {
    auto x = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::sinh(x);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, SinhAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::sinh(x);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, CoshAcceptsFloat16) {
    auto x = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::cosh(x);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, CoshAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::cosh(x);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

// ---- Simple elementwise (approach A or widen-narrow wrap with mathfn) ------

TEST_P(MathHalfDispatch, RoundAcceptsFloat16) {
    auto x = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::round(x);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, RoundAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::round(x);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, FloorAcceptsFloat16) {
    auto x = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::floor(x);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, FloorAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::floor(x);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, CeilAcceptsFloat16) {
    auto x = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::ceil(x);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, CeilAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::ceil(x);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, TruncAcceptsFloat16) {
    auto x = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::trunc(x);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, TruncAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::trunc(x);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, ReciprocalAcceptsFloat16) {
    auto x = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::reciprocal(x);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, ReciprocalAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::reciprocal(x);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, FracAcceptsFloat16) {
    auto x = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::frac(x);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, FracAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::frac(x);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, HeavisideAcceptsFloat16) {
    auto x = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    auto v = make_half<Float16>({4}, 0.0f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::heaviside(x, v);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, HeavisideAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    auto v = make_half<BFloat16>({4}, 0.0f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::heaviside(x, v);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, NanToNumAcceptsFloat16) {
    auto x = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::nan_to_num(x);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, NanToNumAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::nan_to_num(x);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

// ---- lerp (3-input) --------------------------------------------------------

TEST_P(MathHalfDispatch, LerpAcceptsFloat16) {
    auto s = make_half<Float16>({4}, 0.0f, DType::Float16, device);
    auto e = make_half<Float16>({4}, 1.0f, DType::Float16, device);
    auto w = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::lerp(s, e, w);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, LerpAcceptsBFloat16) {
    auto s = make_half<BFloat16>({4}, 0.0f, DType::BFloat16, device);
    auto e = make_half<BFloat16>({4}, 1.0f, DType::BFloat16, device);
    auto w = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::lerp(s, e, w);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

// ---- angle / polar (complex) -----------------------------------------------

TEST_P(MathHalfDispatch, AngleAcceptsFloat16) {
    auto x = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::angle(x);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, AngleAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::angle(x);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, PolarAcceptsFloat16) {
    auto r = make_half<Float16>({4}, 1.0f, DType::Float16, device);
    auto th = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        // polar promotes to Complex64 internally (no complex_half dtype).
        auto y = tenzor::polar(r, th);
        EXPECT_EQ(y.dtype(), DType::Complex64);
    });
}
TEST_P(MathHalfDispatch, PolarAcceptsBFloat16) {
    auto r = make_half<BFloat16>({4}, 1.0f, DType::BFloat16, device);
    auto th = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::polar(r, th);
        EXPECT_EQ(y.dtype(), DType::Complex64);
    });
}

// ---- log_sigmoid (via Variable API) ----------------------------------------

TEST_P(MathHalfDispatch, LogSigmoidAcceptsFloat16) {
    auto x = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    tenzor::Variable v(x);
    EXPECT_NO_THROW({
        auto y = tenzor::nn::log_sigmoid(v);
        EXPECT_EQ(y.tensor().dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, LogSigmoidAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    tenzor::Variable v(x);
    EXPECT_NO_THROW({
        auto y = tenzor::nn::log_sigmoid(v);
        EXPECT_EQ(y.tensor().dtype(), DType::BFloat16);
    });
}

// ---- rrelu (via Variable API, eval mode = deterministic slope) -------------

TEST_P(MathHalfDispatch, RReluAcceptsFloat16) {
    auto x = make_half<Float16>({4}, -0.5f, DType::Float16, device);
    tenzor::Variable v(x);
    EXPECT_NO_THROW({
        auto y = tenzor::nn::rrelu(v, 1.0 / 8.0, 1.0 / 3.0, /*training=*/false);
        EXPECT_EQ(y.tensor().dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, RReluAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, -0.5f, DType::BFloat16, device);
    tenzor::Variable v(x);
    EXPECT_NO_THROW({
        auto y = tenzor::nn::rrelu(v, 1.0 / 8.0, 1.0 / 3.0, /*training=*/false);
        EXPECT_EQ(y.tensor().dtype(), DType::BFloat16);
    });
}

// ---- nansum / nanmean / nanvar / aminmax (accumulators) --------------------

TEST_P(MathHalfDispatch, NansumAcceptsFloat16) {
    auto x = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::nansum(x);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, NansumAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::nansum(x);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, NansumDimAcceptsFloat16) {
    auto x = make_half<Float16>({2, 3}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::nansum(x, /*dim=*/1);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, NansumDimAcceptsBFloat16) {
    auto x = make_half<BFloat16>({2, 3}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::nansum(x, /*dim=*/1);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, NanmeanAcceptsFloat16) {
    auto x = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::nanmean(x);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, NanmeanAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::nanmean(x);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, NanmeanDimAcceptsFloat16) {
    auto x = make_half<Float16>({2, 3}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::nanmean(x, /*dim=*/1);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, NanmeanDimAcceptsBFloat16) {
    auto x = make_half<BFloat16>({2, 3}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::nanmean(x, /*dim=*/1);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, NanvarAcceptsFloat16) {
    auto x = make_half<Float16>({2, 3}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::nanvar(x, /*dim=*/1);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, NanvarAcceptsBFloat16) {
    auto x = make_half<BFloat16>({2, 3}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::nanvar(x, /*dim=*/1);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, AminmaxAcceptsFloat16) {
    auto x = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    std::pair<Tensor, Tensor> r;
    EXPECT_NO_THROW({ r = tenzor::aminmax(x); });
    EXPECT_EQ(r.first.dtype(), DType::Float16);
    EXPECT_EQ(r.second.dtype(), DType::Float16);
}
TEST_P(MathHalfDispatch, AminmaxAcceptsBFloat16) {
    auto x = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    std::pair<Tensor, Tensor> r;
    EXPECT_NO_THROW({ r = tenzor::aminmax(x); });
    EXPECT_EQ(r.first.dtype(), DType::BFloat16);
    EXPECT_EQ(r.second.dtype(), DType::BFloat16);
}
TEST_P(MathHalfDispatch, AminmaxDimAcceptsFloat16) {
    auto x = make_half<Float16>({2, 3}, 0.5f, DType::Float16, device);
    std::pair<Tensor, Tensor> r;
    EXPECT_NO_THROW({ r = tenzor::aminmax(x, /*dim=*/1); });
    EXPECT_EQ(r.first.dtype(), DType::Float16);
    EXPECT_EQ(r.second.dtype(), DType::Float16);
}
TEST_P(MathHalfDispatch, AminmaxDimAcceptsBFloat16) {
    auto x = make_half<BFloat16>({2, 3}, 0.5f, DType::BFloat16, device);
    std::pair<Tensor, Tensor> r;
    EXPECT_NO_THROW({ r = tenzor::aminmax(x, /*dim=*/1); });
    EXPECT_EQ(r.first.dtype(), DType::BFloat16);
    EXPECT_EQ(r.second.dtype(), DType::BFloat16);
}

// ---- index_add / index_copy / index_fill -----------------------------------

TEST_P(MathHalfDispatch, IndexAddAcceptsFloat16) {
    auto input = make_half<Float16>({4, 3}, 1.0f, DType::Float16, device);
    auto source = make_half<Float16>({2, 3}, 0.5f, DType::Float16, device);
    auto idx_host = Tensor({2}, DType::Int64, tenzor::Device::cpu());
    idx_host.data<int64_t>()[0] = 0;
    idx_host.data<int64_t>()[1] = 2;
    auto idx = idx_host.to(device);
    EXPECT_NO_THROW({
        auto y = tenzor::index_add(input, /*dim=*/0, idx, source);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, IndexAddAcceptsBFloat16) {
    auto input = make_half<BFloat16>({4, 3}, 1.0f, DType::BFloat16, device);
    auto source = make_half<BFloat16>({2, 3}, 0.5f, DType::BFloat16, device);
    auto idx_host = Tensor({2}, DType::Int64, tenzor::Device::cpu());
    idx_host.data<int64_t>()[0] = 0;
    idx_host.data<int64_t>()[1] = 2;
    auto idx = idx_host.to(device);
    EXPECT_NO_THROW({
        auto y = tenzor::index_add(input, /*dim=*/0, idx, source);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, IndexCopyAcceptsFloat16) {
    auto input = make_half<Float16>({4, 3}, 1.0f, DType::Float16, device);
    auto source = make_half<Float16>({2, 3}, 0.5f, DType::Float16, device);
    auto idx_host = Tensor({2}, DType::Int64, tenzor::Device::cpu());
    idx_host.data<int64_t>()[0] = 0;
    idx_host.data<int64_t>()[1] = 2;
    auto idx = idx_host.to(device);
    EXPECT_NO_THROW({
        auto y = tenzor::index_copy(input, /*dim=*/0, idx, source);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, IndexCopyAcceptsBFloat16) {
    auto input = make_half<BFloat16>({4, 3}, 1.0f, DType::BFloat16, device);
    auto source = make_half<BFloat16>({2, 3}, 0.5f, DType::BFloat16, device);
    auto idx_host = Tensor({2}, DType::Int64, tenzor::Device::cpu());
    idx_host.data<int64_t>()[0] = 0;
    idx_host.data<int64_t>()[1] = 2;
    auto idx = idx_host.to(device);
    EXPECT_NO_THROW({
        auto y = tenzor::index_copy(input, /*dim=*/0, idx, source);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, IndexFillAcceptsFloat16) {
    auto input = make_half<Float16>({4, 3}, 1.0f, DType::Float16, device);
    auto idx_host = Tensor({2}, DType::Int64, tenzor::Device::cpu());
    idx_host.data<int64_t>()[0] = 0;
    idx_host.data<int64_t>()[1] = 2;
    auto idx = idx_host.to(device);
    EXPECT_NO_THROW({
        auto y = tenzor::index_fill(input, /*dim=*/0, idx, 7.0);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, IndexFillAcceptsBFloat16) {
    auto input = make_half<BFloat16>({4, 3}, 1.0f, DType::BFloat16, device);
    auto idx_host = Tensor({2}, DType::Int64, tenzor::Device::cpu());
    idx_host.data<int64_t>()[0] = 0;
    idx_host.data<int64_t>()[1] = 2;
    auto idx = idx_host.to(device);
    EXPECT_NO_THROW({
        auto y = tenzor::index_fill(input, /*dim=*/0, idx, 7.0);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

// ---- addcmul / addcdiv -----------------------------------------------------

TEST_P(MathHalfDispatch, AddcmulAcceptsFloat16) {
    auto a = make_half<Float16>({4}, 1.0f, DType::Float16, device);
    auto b = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    auto c = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::addcmul(a, b, c, 1.0);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, AddcmulAcceptsBFloat16) {
    auto a = make_half<BFloat16>({4}, 1.0f, DType::BFloat16, device);
    auto b = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    auto c = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::addcmul(a, b, c, 1.0);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, AddcdivAcceptsFloat16) {
    auto a = make_half<Float16>({4}, 1.0f, DType::Float16, device);
    auto b = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    auto c = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::addcdiv(a, b, c, 1.0);
        EXPECT_EQ(y.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, AddcdivAcceptsBFloat16) {
    auto a = make_half<BFloat16>({4}, 1.0f, DType::BFloat16, device);
    auto b = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    auto c = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto y = tenzor::addcdiv(a, b, c, 1.0);
        EXPECT_EQ(y.dtype(), DType::BFloat16);
    });
}

// ---- trapezoid / cumulative_trapezoid / gradient ---------------------------

TEST_P(MathHalfDispatch, TrapezoidAcceptsFloat16) {
    auto y = make_half<Float16>({8}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto r = tenzor::trapezoid(y, /*dx=*/1.0, /*dim=*/-1);
        EXPECT_EQ(r.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, TrapezoidAcceptsBFloat16) {
    auto y = make_half<BFloat16>({8}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto r = tenzor::trapezoid(y, /*dx=*/1.0, /*dim=*/-1);
        EXPECT_EQ(r.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, CumulativeTrapezoidAcceptsFloat16) {
    auto y = make_half<Float16>({8}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto r = tenzor::cumulative_trapezoid(y, /*dx=*/1.0, /*dim=*/-1);
        EXPECT_EQ(r.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, CumulativeTrapezoidAcceptsBFloat16) {
    auto y = make_half<BFloat16>({8}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto r = tenzor::cumulative_trapezoid(y, /*dx=*/1.0, /*dim=*/-1);
        EXPECT_EQ(r.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, GradientAcceptsFloat16) {
    auto x = make_half<Float16>({8}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto r = tenzor::gradient(x);
        EXPECT_EQ(r.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, GradientAcceptsBFloat16) {
    auto x = make_half<BFloat16>({8}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto r = tenzor::gradient(x);
        EXPECT_EQ(r.dtype(), DType::BFloat16);
    });
}

// ---- pairwise_distance / pdist ---------------------------------------------

TEST_P(MathHalfDispatch, PairwiseDistanceAcceptsFloat16) {
    auto a = make_half<Float16>({4, 3}, 0.5f, DType::Float16, device);
    auto b = make_half<Float16>({4, 3}, 1.0f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto r = tenzor::pairwise_distance(a, b, 2.0);
        EXPECT_EQ(r.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, PairwiseDistanceAcceptsBFloat16) {
    auto a = make_half<BFloat16>({4, 3}, 0.5f, DType::BFloat16, device);
    auto b = make_half<BFloat16>({4, 3}, 1.0f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto r = tenzor::pairwise_distance(a, b, 2.0);
        EXPECT_EQ(r.dtype(), DType::BFloat16);
    });
}

TEST_P(MathHalfDispatch, PdistAcceptsFloat16) {
    auto a = make_half<Float16>({4, 3}, 0.5f, DType::Float16, device);
    EXPECT_NO_THROW({
        auto r = tenzor::pdist(a, 2.0);
        EXPECT_EQ(r.dtype(), DType::Float16);
    });
}
TEST_P(MathHalfDispatch, PdistAcceptsBFloat16) {
    auto a = make_half<BFloat16>({4, 3}, 0.5f, DType::BFloat16, device);
    EXPECT_NO_THROW({
        auto r = tenzor::pdist(a, 2.0);
        EXPECT_EQ(r.dtype(), DType::BFloat16);
    });
}

// ----------------------------------------------------------------------------
// Numeric spot-checks: half-precision output should match Float32 reference
// within Float16's representable precision (~1e-3 relative for values < 1).
//
// We pick: asin (transcendental, Float32-promoted internally),
//          floor (deterministic), nansum (accumulator), and
//          pairwise_distance (multi-step accumulator).
//
// Comparison happens on the host: both half and Float32 results are moved to
// CPU before reading their buffers via data<float>().
// ----------------------------------------------------------------------------

auto half_to_float_max_abs_err(const Tensor& y16, const Tensor& y32) -> float {
    int64_t n = y32.numel();
    auto y16f = y16.cpu().to(DType::Float32);
    auto y32f = y32.cpu();
    const float* a = y16f.data<float>();
    const float* b = y32f.data<float>();
    float worst = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > worst) worst = diff;
    }
    return worst;
}

TEST_P(MathHalfDispatch, AsinFloat16ApproxFloat32) {
    auto x32 = make_half<Float16>({16}, 0.5f, DType::Float16, device).to(DType::Float32);
    auto x16 = x32.to(DType::Float16);
    auto y32 = tenzor::asin(x32);
    auto y16 = tenzor::asin(x16);
    // Float16 precision: ~1e-3 absolute for values in [0, 1].
    EXPECT_LT(half_to_float_max_abs_err(y16, y32), 5e-3f);
}

TEST_P(MathHalfDispatch, FloorFloat16Exact) {
    // Floor of 0.5 is 0.0; representable exactly in both Float16 and Float32.
    auto x16 = make_half<Float16>({4}, 0.5f, DType::Float16, device);
    auto x32 = x16.to(DType::Float32);
    auto y16 = tenzor::floor(x16);
    auto y32 = tenzor::floor(x32);
    EXPECT_LT(half_to_float_max_abs_err(y16, y32), 1e-6f);
}

TEST_P(MathHalfDispatch, NansumFloat16ApproxFloat32) {
    auto x32 = make_half<Float16>({8}, 0.25f, DType::Float16, device).to(DType::Float32);
    auto x16 = x32.to(DType::Float16);
    auto y32 = tenzor::nansum(x32);  // 8 * 0.25 = 2.0
    auto y16 = tenzor::nansum(x16);
    // Sum of 8 elements at 0.25 each; FP32 path is exact, FP16 path widens.
    EXPECT_LT(half_to_float_max_abs_err(y16, y32), 5e-3f);
}

TEST_P(MathHalfDispatch, PairwiseDistanceFloat16ApproxFloat32) {
    auto a16 = make_half<Float16>({4, 3}, 0.5f, DType::Float16, device);
    auto b16 = make_half<Float16>({4, 3}, 1.0f, DType::Float16, device);
    auto a32 = a16.to(DType::Float32);
    auto b32 = b16.to(DType::Float32);
    auto r16 = tenzor::pairwise_distance(a16, b16, 2.0);
    auto r32 = tenzor::pairwise_distance(a32, b32, 2.0);
    // 4-element rows with diff=0.5 -> distance = sqrt(3 * 0.25) ≈ 0.866.
    EXPECT_LT(half_to_float_max_abs_err(r16, r32), 5e-3f);
}

// ----------------------------------------------------------------------------
// Regression: BFloat16 must be accepted by the unary/binary math wrappers that
// previously dispatched only Float32/Float64/Float16 and threw on BF16 on the
// ROCm backend (CPU already supported it). Verify no throw + correct dtype +
// match the Float32 reference within BF16 tolerance.
// ----------------------------------------------------------------------------
namespace {
void check_bf16_unary_matches_f32(const tenzor::Device& device,
                                  Tensor (*op)(const Tensor&), float value) {
    auto xb = make_half<BFloat16>({4}, value, DType::BFloat16, device);
    Tensor yb;
    EXPECT_NO_THROW({ yb = op(xb); });
    EXPECT_EQ(yb.dtype(), DType::BFloat16);
    auto yf = op(xb.to(DType::Float32));
    // BF16 has ~2-3 decimal digits; widen and compare loosely.
    auto diff = tenzor::abs(yb.to(DType::Float32).to(tenzor::Device::cpu()) -
                            yf.to(DType::Float32).to(tenzor::Device::cpu()));
    EXPECT_LT(tenzor::max(diff).item<float>(), 0.1f);
}
}  // namespace

TEST_P(MathHalfDispatch, SqrtAcceptsBFloat16)   { check_bf16_unary_matches_f32(device, tenzor::sqrt, 0.7f); }
TEST_P(MathHalfDispatch, ExpAcceptsBFloat16)    { check_bf16_unary_matches_f32(device, tenzor::exp, 0.5f); }
TEST_P(MathHalfDispatch, LogAcceptsBFloat16)    { check_bf16_unary_matches_f32(device, tenzor::log, 0.7f); }
TEST_P(MathHalfDispatch, SinAcceptsBFloat16)    { check_bf16_unary_matches_f32(device, tenzor::sin, 0.5f); }
TEST_P(MathHalfDispatch, CosAcceptsBFloat16)    { check_bf16_unary_matches_f32(device, tenzor::cos, 0.5f); }
TEST_P(MathHalfDispatch, ErfAcceptsBFloat16)    { check_bf16_unary_matches_f32(device, tenzor::erf, 0.5f); }
TEST_P(MathHalfDispatch, RsqrtAcceptsBFloat16)  { check_bf16_unary_matches_f32(device, tenzor::rsqrt, 0.7f); }
TEST_P(MathHalfDispatch, SquareAcceptsBFloat16) { check_bf16_unary_matches_f32(device, tenzor::square, 0.5f); }
TEST_P(MathHalfDispatch, Log2AcceptsBFloat16)   { check_bf16_unary_matches_f32(device, tenzor::log2, 0.7f); }
TEST_P(MathHalfDispatch, Exp2AcceptsBFloat16)   { check_bf16_unary_matches_f32(device, tenzor::exp2, 0.5f); }

TEST_P(MathHalfDispatch, HypotAcceptsBFloat16) {
    auto a = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    auto b = make_half<BFloat16>({4}, 1.0f, DType::BFloat16, device);
    Tensor y;
    EXPECT_NO_THROW({ y = tenzor::hypot(a, b); });
    EXPECT_EQ(y.dtype(), DType::BFloat16);
}

TEST_P(MathHalfDispatch, Atan2AcceptsBFloat16) {
    auto a = make_half<BFloat16>({4}, 0.5f, DType::BFloat16, device);
    auto b = make_half<BFloat16>({4}, 1.0f, DType::BFloat16, device);
    Tensor y;
    EXPECT_NO_THROW({ y = tenzor::atan2(a, b); });
    EXPECT_EQ(y.dtype(), DType::BFloat16);
}

// isnan/isinf/isfinite must detect special values in a BFloat16 tensor (the
// ROCm kernels previously had no BF16 branch and reported all-false / all-true).
TEST_P(MathHalfDispatch, IsNanInfFiniteBFloat16) {
    auto t = Tensor({3}, DType::BFloat16, tenzor::Device::cpu());
    auto* d = t.data<BFloat16>();
    d[0] = BFloat16(std::numeric_limits<float>::quiet_NaN());
    d[1] = BFloat16(std::numeric_limits<float>::infinity());
    d[2] = BFloat16(1.5f);
    auto tb = t.to(device);

    auto nan_mask = tenzor::isnan(tb).to(tenzor::Device::cpu());
    auto inf_mask = tenzor::isinf(tb).to(tenzor::Device::cpu());
    auto fin_mask = tenzor::isfinite(tb).to(tenzor::Device::cpu());
    const auto* nanb = static_cast<const uint8_t*>(nan_mask.data_ptr());
    const auto* infb = static_cast<const uint8_t*>(inf_mask.data_ptr());
    const auto* finb = static_cast<const uint8_t*>(fin_mask.data_ptr());
    EXPECT_NE(nanb[0], 0);  // NaN -> isnan true
    EXPECT_EQ(nanb[2], 0);
    EXPECT_NE(infb[1], 0);  // Inf -> isinf true
    EXPECT_EQ(infb[2], 0);
    EXPECT_EQ(finb[0], 0);  // NaN -> not finite
    EXPECT_EQ(finb[1], 0);  // Inf -> not finite
    EXPECT_NE(finb[2], 0);  // 1.5 -> finite
}

INSTANTIATE_BACKEND_TESTS(MathHalfDispatch);

}  // namespace
