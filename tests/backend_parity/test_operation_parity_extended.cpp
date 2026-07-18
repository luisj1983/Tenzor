/**
 * @file test_operation_parity_extended.cpp
 * @brief Extended backend parity tests for operations not covered by the
 *        original test_operation_parity.cpp.
 *
 * Covers: indexing, shape manipulation, cumulative, advanced math, and
 * other ops missing from the original 58-test parity suite.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class ExtendedParity : public BackendTest {};
// ============================================================================
// Indexing Operations
// ============================================================================

TEST_P(ExtendedParity, Gather) {
auto input = randn({8, 16}, DType::Float32, Device::cpu());
    auto index = randint(0, 16, {8, 4}, DType::Int64, Device::cpu());

    test_operation_parity([&index](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0], index.to(inputs[0].device())};
        return dispatch<OpId::Gather>(ins, attrs)[0];
    }, {input}, 1e-6f, 1e-8f, "Gather");
}

TEST_P(ExtendedParity, IndexSelect) {
auto input = randn({16, 8}, DType::Float32, Device::cpu());
    auto idx = zeros({4}, DType::Int64, Device::cpu());
    idx.data<int64_t>()[0] = 0; idx.data<int64_t>()[1] = 3;
    idx.data<int64_t>()[2] = 7; idx.data<int64_t>()[3] = 12;

    test_operation_parity([&idx](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(0));
        std::vector<Tensor> ins = {inputs[0], idx.to(inputs[0].device())};
        return dispatch<OpId::IndexSelect>(ins, attrs)[0];
    }, {input}, 1e-6f, 1e-8f, "IndexSelect");
}

TEST_P(ExtendedParity, ScatterAdd) {
auto input = zeros({4, 8}, DType::Float32, Device::cpu());
    auto src = randn({4, 4}, DType::Float32, Device::cpu());
    auto index = randint(0, 8, {4, 4}, DType::Int64, Device::cpu());

    test_operation_parity([&src, &index](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0], index.to(inputs[0].device()), src.to(inputs[0].device())};
        return dispatch<OpId::ScatterAdd>(ins, attrs)[0];
    }, {input}, 1e-5f, 1e-7f, "ScatterAdd");
}

TEST_P(ExtendedParity, Where) {
auto x = randn({8, 8}, DType::Float32, Device::cpu());
    auto y = randn({8, 8}, DType::Float32, Device::cpu());
    auto cond = zeros({8, 8}, DType::Bool, Device::cpu());
    auto* cd = cond.data<bool>();
    for (int64_t i = 0; i < 64; ++i) cd[i] = (i % 3 == 0);

    test_operation_parity([&y, &cond](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {cond.to(inputs[0].device()), inputs[0], y.to(inputs[0].device())};
        return dispatch<OpId::Where>(ins)[0];
    }, {x}, 1e-6f, 1e-8f, "Where");
}

// ============================================================================
// Shape / Manipulation Operations
// ============================================================================

TEST_P(ExtendedParity, Roll) {
auto input = randn({8, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Shift, int64_t(3));
        attrs.set(AttrKey::Dim, int64_t(0));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::Roll>(ins, attrs)[0];
    }, {input}, 0.0f, 0.0f, "Roll");
}

TEST_P(ExtendedParity, Flip) {
auto input = randn({8, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::Flip>(ins)[0];
    }, {input}, 0.0f, 0.0f, "Flip");
}

TEST_P(ExtendedParity, Tril) {
auto input = randn({8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Diagonal, int64_t(0));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::Tril>(ins, attrs)[0];
    }, {input}, 0.0f, 0.0f, "Tril");
}

TEST_P(ExtendedParity, Triu) {
auto input = randn({8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Diagonal, int64_t(0));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::Triu>(ins, attrs)[0];
    }, {input}, 0.0f, 0.0f, "Triu");
}

TEST_P(ExtendedParity, Diag) {
auto input = randn({8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Diagonal, int64_t(0));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::Diag>(ins, attrs)[0];
    }, {input}, 0.0f, 0.0f, "Diag");
}

TEST_P(ExtendedParity, Trace) {
auto input = randn({8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::Trace>(ins)[0];
    }, {input}, 1e-5f, 1e-7f, "Trace");
}

// ============================================================================
// Cumulative Operations
// ============================================================================

TEST_P(ExtendedParity, Cumsum) {
auto input = randn({16, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::CumSum>(ins, attrs)[0];
    }, {input}, 1e-5f, 1e-7f, "Cumsum");
}

TEST_P(ExtendedParity, Cumprod) {
// Small values to prevent overflow
    auto input = full({8, 4}, 0.9f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::CumProd>(ins, attrs)[0];
    }, {input}, 1e-5f, 1e-7f, "Cumprod");
}

// Regression: direct dispatch<OpId::CumSum/CumProd> with dim=-1 bypasses the
// front-door tenzor::cumsum()/cumprod() (which normalize negative dims before
// dispatch) -- ROCm's own kernel indexed shape[-1] directly with no
// normalization (undefined behavior), unlike CPU/CUDA/OneAPI/Vulkan.
TEST_P(ExtendedParity, CumsumNegativeDim) {
    auto input = randn({16, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(-1));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::CumSum>(ins, attrs)[0];
    }, {input}, 1e-5f, 1e-7f, "CumsumNegativeDim");
}

TEST_P(ExtendedParity, CumprodNegativeDim) {
    auto input = full({8, 4}, 0.9f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(-1));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::CumProd>(ins, attrs)[0];
    }, {input}, 1e-5f, 1e-7f, "CumprodNegativeDim");
}

// Regression: direct dispatch<OpId::CumSum/CumProd> with Float16 bypasses the
// front-door tenzor::cumsum()/cumprod() (which widen Float16/BFloat16 to
// Float32 before dispatch) -- ROCm's own kernel threw "unsupported dtype" for
// Float16, unlike CPU/CUDA/OneAPI/Vulkan (all widen inline).
TEST_P(ExtendedParity, CumsumFloat16) {
    auto input = randn({16, 8}, DType::Float16, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::CumSum>(ins, attrs)[0];
    }, {input}, 1e-2f, 1e-3f, "CumsumFloat16");
}

TEST_P(ExtendedParity, CumprodFloat16) {
    auto input = full({8, 4}, 0.9f, DType::Float16, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::CumProd>(ins, attrs)[0];
    }, {input}, 1e-2f, 1e-3f, "CumprodFloat16");
}

// ============================================================================
// Advanced Operations
// ============================================================================

TEST_P(ExtendedParity, TopK) {
auto input = randn({8, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::K, int64_t(5));
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::TopK>(ins, attrs)[0];
    }, {input}, 1e-6f, 1e-8f, "TopK");
}

TEST_P(ExtendedParity, Sort) {
auto input = randn({8, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::Sort>(ins, attrs)[0];
    }, {input}, 1e-6f, 1e-8f, "Sort");
}

TEST_P(ExtendedParity, Unique) {
auto input = randint(0, 10, {32}, DType::Int64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::Unique>(ins)[0];
    }, {input}, 0.0f, 0.0f, "Unique");
}

// ============================================================================
// Special Math Operations
// ============================================================================

TEST_P(ExtendedParity, Erf) {
auto input = randn({16, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::erf(inputs[0]);
    }, {input}, 1e-5f, 1e-7f, "Erf");
}

TEST_P(ExtendedParity, Erfc) {
auto input = randn({16, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::erfc(inputs[0]);
    }, {input}, 1e-5f, 1e-7f, "Erfc");
}

TEST_P(ExtendedParity, Lgamma) {
// Positive values for lgamma
    auto input = tenzor::add(tenzor::abs(randn({16, 16}, DType::Float32, Device::cpu())),
                              full({16, 16}, 0.5f, DType::Float32, Device::cpu()));

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::lgamma(inputs[0]);
    }, {input}, 1e-4f, 1e-6f, "Lgamma");
}

TEST_P(ExtendedParity, Digamma) {
auto input = tenzor::add(tenzor::abs(randn({16, 16}, DType::Float32, Device::cpu())),
                              full({16, 16}, 1.0f, DType::Float32, Device::cpu()));

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::digamma(inputs[0]);
    }, {input}, 1e-4f, 1e-6f, "Digamma");
}

TEST_P(ExtendedParity, Log2) {
auto input = tenzor::add(tenzor::abs(randn({16, 16}, DType::Float32, Device::cpu())),
                              full({16, 16}, 0.1f, DType::Float32, Device::cpu()));

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::log2(inputs[0]);
    }, {input}, 1e-5f, 1e-7f, "Log2");
}

TEST_P(ExtendedParity, Log10) {
auto input = tenzor::add(tenzor::abs(randn({16, 16}, DType::Float32, Device::cpu())),
                              full({16, 16}, 0.1f, DType::Float32, Device::cpu()));

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::log10(inputs[0]);
    }, {input}, 1e-5f, 1e-7f, "Log10");
}

TEST_P(ExtendedParity, Expm1) {
auto input = randn({16, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::expm1(inputs[0]);
    }, {input}, 1e-5f, 1e-7f, "Expm1");
}

TEST_P(ExtendedParity, Sinc) {
    auto input = randn({16, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::sinc(inputs[0]);
    }, {input}, 1e-5f, 1e-7f, "Sinc");
}

// ============================================================================
// Bitwise Operations
// ============================================================================

TEST_P(ExtendedParity, BitwiseAnd) {
auto a = randint(0, 255, {16, 16}, DType::Int32, Device::cpu());
    auto b = randint(0, 255, {16, 16}, DType::Int32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::bitwise_and(inputs[0], inputs[1]);
    }, {a, b}, 0.0f, 0.0f, "BitwiseAnd");
}

TEST_P(ExtendedParity, BitwiseOr) {
auto a = randint(0, 255, {16, 16}, DType::Int32, Device::cpu());
    auto b = randint(0, 255, {16, 16}, DType::Int32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::bitwise_or(inputs[0], inputs[1]);
    }, {a, b}, 0.0f, 0.0f, "BitwiseOr");
}

// ============================================================================
// Multi-DType Parity (key ops across Float32, Float64)
// ============================================================================

TEST_P(ExtendedParity, MatMul_Float64) {
auto a = randn({8, 16}, DType::Float64, Device::cpu());
    auto b = randn({16, 8}, DType::Float64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::matmul(inputs[0], inputs[1]);
    }, {a, b}, 1e-12f, 1e-14f, "MatMul_Float64");
}

// Broadcast batch matmul: a=(B, m, k) @ b=(k, n) — b is rank-2 and must be
// broadcast across the B batch dim. Regression for the OneAPI N-D matmul OOB
// where b_batch_stride*bi ran past b's k*n elements.
TEST_P(ExtendedParity, MatMul_BroadcastBatch_RankMismatch_Float64) {
    auto a = randn({4, 8, 16}, DType::Float64, Device::cpu());
    auto b = randn({16, 8}, DType::Float64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::matmul(inputs[0], inputs[1]);
    }, {a, b}, 1e-12f, 1e-14f, "MatMul_BroadcastBatch_RankMismatch_Float64");
}

TEST_P(ExtendedParity, Softmax_Float64) {
auto input = randn({8, 16}, DType::Float64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> sm_inputs = {inputs[0]};
        return dispatch_single(OpId::Softmax, std::span<const Tensor>(sm_inputs), attrs);
    }, {input}, 1e-12f, 1e-14f, "Softmax_Float64");
}

TEST_P(ExtendedParity, Add_Float64) {
    auto a = randn({16, 16}, DType::Float64, Device::cpu());
    auto b = randn({16, 16}, DType::Float64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 1e-14f, 0.0f, "Add_Float64");
}

// ============================================================================
// FFT Operations
// ============================================================================

TEST_P(ExtendedParity, FFT) {
    // FFT kernels expect complex input. The public tenzor::fft::fft op
    // promotes Float32/Float64 to Complex64/Complex128 and normalises dim
    // before dispatch — when calling dispatch<OpId::FFT> directly we have
    // to provide complex input and a valid dim ourselves.
    auto input = randn({16}, DType::Complex64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {inputs[0].contiguous()};
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(0));
        attrs.set(AttrKey::N, int64_t(16));
        attrs.set(AttrKey::Norm, std::string("backward"));
        return dispatch<OpId::FFT>(ins, attrs)[0];
    }, {input}, 1e-4f, 1e-6f, "FFT");
}

TEST_P(ExtendedParity, IFFT) {
    auto input = randn({16}, DType::Complex64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {inputs[0].contiguous()};
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(0));
        attrs.set(AttrKey::N, int64_t(16));
        attrs.set(AttrKey::Norm, std::string("backward"));
        return dispatch<OpId::IFFT>(ins, attrs)[0];
    }, {input}, 1e-4f, 1e-6f, "IFFT");
}

TEST_P(ExtendedParity, RFFT) {
    // RFFT consumes a real tensor and produces complex output.
    auto input = randn({16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {inputs[0].contiguous()};
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(0));
        attrs.set(AttrKey::N, int64_t(16));
        attrs.set(AttrKey::Norm, std::string("backward"));
        return dispatch<OpId::RFFT>(ins, attrs)[0];
    }, {input}, 1e-4f, 1e-6f, "RFFT");
}

TEST_P(ExtendedParity, IRFFT) {
    // IRFFT requires a Hermitian-symmetric half-spectrum AND the spectrum
    // representation must match the backend's RFFT convention. Round-trip
    // RFFT then IRFFT per-backend exposes any normalization mismatch and
    // exercises the IRFFT path on every backend uniformly.
    auto real_signal = randn({16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> rfft_ins = {inputs[0].contiguous()};
        OpAttributes rfft_attrs;
        rfft_attrs.set(AttrKey::Dim, int64_t(0));
        rfft_attrs.set(AttrKey::N, int64_t(16));
        rfft_attrs.set(AttrKey::Norm, std::string("backward"));
        auto half_spectrum = dispatch<OpId::RFFT>(rfft_ins, rfft_attrs)[0];

        std::vector<Tensor> irfft_ins = {half_spectrum.contiguous()};
        OpAttributes irfft_attrs;
        irfft_attrs.set(AttrKey::Dim, int64_t(0));
        irfft_attrs.set(AttrKey::N, int64_t(16));
        irfft_attrs.set(AttrKey::Norm, std::string("backward"));
        return dispatch<OpId::IRFFT>(irfft_ins, irfft_attrs)[0];
    }, {real_signal}, 1e-4f, 1e-5f, "IRFFT");
}

// ============================================================================
// Sparse Operations
// ============================================================================

TEST_P(ExtendedParity, SparseSpMM) {
    // CSR sparse matrix: 4x4 identity (simple case)
    auto crow = zeros({5}, DType::Int64, Device::cpu());
    crow.data<int64_t>()[0] = 0; crow.data<int64_t>()[1] = 1;
    crow.data<int64_t>()[2] = 2; crow.data<int64_t>()[3] = 3;
    crow.data<int64_t>()[4] = 4;
    auto col = zeros({4}, DType::Int64, Device::cpu());
    col.data<int64_t>()[0] = 0; col.data<int64_t>()[1] = 1;
    col.data<int64_t>()[2] = 2; col.data<int64_t>()[3] = 3;
    auto vals = ones({4}, DType::Float32, Device::cpu());
    auto dense = randn({4, 8}, DType::Float32, Device::cpu());

    test_operation_parity([&crow, &col, &vals](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {
            crow.to(inputs[0].device()),
            col.to(inputs[0].device()),
            vals.to(inputs[0].device()),
            inputs[0]
        };
        OpAttributes attrs;
        attrs.set(AttrKey::M, int64_t(4));
        attrs.set(AttrKey::K, int64_t(4));
        return dispatch<OpId::SparseSpMM>(ins, attrs)[0];
    }, {dense}, 1e-5f, 1e-7f, "SparseSpMM");
}

// ============================================================================
// Additional Missing Operations
// ============================================================================

TEST_P(ExtendedParity, Scatter) {
    auto input = randn({4, 8}, DType::Float32, Device::cpu());
    auto src = randn({4, 3}, DType::Float32, Device::cpu());

    // Scatter behaviour with duplicate indices in the same row is
    // implementation-defined (which write "wins" is order-dependent and
    // differs across CPU/CUDA/Vulkan/ROCm). For a parity check we need
    // unique indices per row so that every backend deterministically
    // produces the same output.
    auto index = zeros({4, 3}, DType::Int64, Device::cpu());
    int64_t* idx_ptr = index.data<int64_t>();
    int64_t unique_cols[3] = {0, 3, 6};  // disjoint, in-range columns
    for (int64_t r = 0; r < 4; ++r) {
        for (int64_t c = 0; c < 3; ++c) {
            idx_ptr[r * 3 + c] = unique_cols[c];
        }
    }

    test_operation_parity([&src, &index](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0], index.to(inputs[0].device()), src.to(inputs[0].device())};
        return dispatch<OpId::Scatter>(ins, attrs)[0];
    }, {input}, 1e-6f, 1e-8f, "Scatter");
}

TEST_P(ExtendedParity, SparseSpMV) {
    // CSR sparse identity 4x4, multiply by vector
    auto crow = zeros({5}, DType::Int64, Device::cpu());
    crow.data<int64_t>()[0] = 0; crow.data<int64_t>()[1] = 1;
    crow.data<int64_t>()[2] = 2; crow.data<int64_t>()[3] = 3;
    crow.data<int64_t>()[4] = 4;
    auto col = zeros({4}, DType::Int64, Device::cpu());
    col.data<int64_t>()[0] = 0; col.data<int64_t>()[1] = 1;
    col.data<int64_t>()[2] = 2; col.data<int64_t>()[3] = 3;
    auto vals = ones({4}, DType::Float32, Device::cpu());
    auto vec = randn({4}, DType::Float32, Device::cpu());

    test_operation_parity([&crow, &col, &vals](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {
            crow.to(inputs[0].device()),
            col.to(inputs[0].device()),
            vals.to(inputs[0].device()),
            inputs[0]
        };
        OpAttributes attrs;
        attrs.set(AttrKey::M, int64_t(4));
        attrs.set(AttrKey::K, int64_t(4));
        return dispatch<OpId::SparseSpMV>(ins, attrs)[0];
    }, {vec}, 1e-5f, 1e-7f, "SparseSpMV");
}

TEST_P(ExtendedParity, SparseAdd) {
    auto crow = zeros({5}, DType::Int64, Device::cpu());
    crow.data<int64_t>()[0] = 0; crow.data<int64_t>()[1] = 1;
    crow.data<int64_t>()[2] = 2; crow.data<int64_t>()[3] = 3;
    crow.data<int64_t>()[4] = 4;
    auto col = zeros({4}, DType::Int64, Device::cpu());
    col.data<int64_t>()[0] = 0; col.data<int64_t>()[1] = 1;
    col.data<int64_t>()[2] = 2; col.data<int64_t>()[3] = 3;
    auto vals = ones({4}, DType::Float32, Device::cpu());
    auto dense = randn({4, 4}, DType::Float32, Device::cpu());

    test_operation_parity([&crow, &col, &vals](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {
            crow.to(inputs[0].device()),
            col.to(inputs[0].device()),
            vals.to(inputs[0].device()),
            inputs[0]
        };
        OpAttributes attrs;
        attrs.set(AttrKey::M, int64_t(4));
        attrs.set(AttrKey::K, int64_t(4));
        return dispatch<OpId::SparseAdd>(ins, attrs)[0];
    }, {dense}, 1e-5f, 1e-7f, "SparseAdd");
}

TEST_P(ExtendedParity, Logcumsumexp) {
    auto input = randn({8, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::Logcumsumexp>(ins, attrs)[0];
    }, {input}, 1e-4f, 1e-6f, "Logcumsumexp");
}

TEST_P(ExtendedParity, IndexAdd) {
    auto input = zeros({8, 4}, DType::Float32, Device::cpu());
    auto idx = zeros({3}, DType::Int64, Device::cpu());
    idx.data<int64_t>()[0] = 1; idx.data<int64_t>()[1] = 3; idx.data<int64_t>()[2] = 5;
    auto src = randn({3, 4}, DType::Float32, Device::cpu());

    test_operation_parity([&idx, &src](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(0));
        std::vector<Tensor> ins = {inputs[0], idx.to(inputs[0].device()), src.to(inputs[0].device())};
        return dispatch<OpId::IndexAdd>(ins, attrs)[0];
    }, {input}, 1e-5f, 1e-7f, "IndexAdd");
}

TEST_P(ExtendedParity, Bucketize) {
    auto input = randn({16}, DType::Float32, Device::cpu());
    // Create sorted boundaries manually
    auto boundaries = randn({8}, DType::Float32, Device::cpu());
    OpAttributes sort_attrs;
    sort_attrs.set(AttrKey::Dim, int64_t(0));
    std::vector<Tensor> sort_ins = {boundaries};
    boundaries = dispatch<OpId::Sort>(sort_ins, sort_attrs)[0];

    test_operation_parity([&boundaries](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {inputs[0], boundaries.to(inputs[0].device())};
        return dispatch<OpId::Bucketize>(ins)[0];
    }, {input}, 0.0f, 0.0f, "Bucketize");
}

TEST_P(ExtendedParity, SearchSorted) {
    auto sorted_data = randn({16}, DType::Float32, Device::cpu());
    OpAttributes sort_attrs;
    sort_attrs.set(AttrKey::Dim, int64_t(0));
    std::vector<Tensor> sort_ins = {sorted_data};
    auto sorted = dispatch<OpId::Sort>(sort_ins, sort_attrs)[0];
    auto values = randn({8}, DType::Float32, Device::cpu());

    test_operation_parity([&sorted](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {sorted.to(inputs[0].device()), inputs[0]};
        return dispatch<OpId::SearchSorted>(ins)[0];
    }, {values}, 0.0f, 0.0f, "SearchSorted");
}

INSTANTIATE_BACKEND_TESTS(ExtendedParity);


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    setenv("TENZOR_DISABLE_TF32", "1", 1);
    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize: " << e.what() << std::endl;
        return 1;
    }
    int result = RUN_ALL_TESTS();
    try { tenzor::finalize(); } catch (...) {}
    return result;
}
