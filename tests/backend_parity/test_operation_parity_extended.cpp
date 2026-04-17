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
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Indexing Operations
// ============================================================================

TEST(ExtendedParity, Gather) {
auto input = randn({8, 16}, DType::Float32, Device::cpu());
    auto index = randint(0, 16, {8, 4}, DType::Int64, Device::cpu());

    test_operation_parity([&index](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0], index.to(inputs[0].device())};
        return dispatch<OpId::Gather>(ins, attrs)[0];
    }, {input}, 1e-6f, 1e-8f, "Gather");
}

TEST(ExtendedParity, IndexSelect) {
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

TEST(ExtendedParity, ScatterAdd) {
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

TEST(ExtendedParity, Where) {
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

TEST(ExtendedParity, Roll) {
auto input = randn({8, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Shift, int64_t(3));
        attrs.set(AttrKey::Dim, int64_t(0));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::Roll>(ins, attrs)[0];
    }, {input}, 0.0f, 0.0f, "Roll");
}

TEST(ExtendedParity, Flip) {
auto input = randn({8, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::Flip>(ins)[0];
    }, {input}, 0.0f, 0.0f, "Flip");
}

TEST(ExtendedParity, Tril) {
auto input = randn({8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Diagonal, int64_t(0));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::Tril>(ins, attrs)[0];
    }, {input}, 0.0f, 0.0f, "Tril");
}

TEST(ExtendedParity, Triu) {
auto input = randn({8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Diagonal, int64_t(0));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::Triu>(ins, attrs)[0];
    }, {input}, 0.0f, 0.0f, "Triu");
}

TEST(ExtendedParity, Diag) {
auto input = randn({8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Diagonal, int64_t(0));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::Diag>(ins, attrs)[0];
    }, {input}, 0.0f, 0.0f, "Diag");
}

TEST(ExtendedParity, Trace) {
auto input = randn({8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::Trace>(ins)[0];
    }, {input}, 1e-5f, 1e-7f, "Trace");
}

// ============================================================================
// Cumulative Operations
// ============================================================================

TEST(ExtendedParity, Cumsum) {
auto input = randn({16, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::CumSum>(ins, attrs)[0];
    }, {input}, 1e-5f, 1e-7f, "Cumsum");
}

TEST(ExtendedParity, Cumprod) {
// Small values to prevent overflow
    auto input = full({8, 4}, 0.9f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::CumProd>(ins, attrs)[0];
    }, {input}, 1e-5f, 1e-7f, "Cumprod");
}

// ============================================================================
// Advanced Operations
// ============================================================================

TEST(ExtendedParity, TopK) {
auto input = randn({8, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::K, int64_t(5));
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::TopK>(ins, attrs)[0];
    }, {input}, 1e-6f, 1e-8f, "TopK");
}

TEST(ExtendedParity, Sort) {
auto input = randn({8, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::Sort>(ins, attrs)[0];
    }, {input}, 1e-6f, 1e-8f, "Sort");
}

TEST(ExtendedParity, Unique) {
auto input = randint(0, 10, {32}, DType::Int64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::Unique>(ins)[0];
    }, {input}, 0.0f, 0.0f, "Unique");
}

// ============================================================================
// Special Math Operations
// ============================================================================

TEST(ExtendedParity, Erf) {
auto input = randn({16, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::erf(inputs[0]);
    }, {input}, 1e-5f, 1e-7f, "Erf");
}

TEST(ExtendedParity, Erfc) {
auto input = randn({16, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::erfc(inputs[0]);
    }, {input}, 1e-5f, 1e-7f, "Erfc");
}

TEST(ExtendedParity, Lgamma) {
// Positive values for lgamma
    auto input = tenzor::add(tenzor::abs(randn({16, 16}, DType::Float32, Device::cpu())),
                              full({16, 16}, 0.5f, DType::Float32, Device::cpu()));

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::lgamma(inputs[0]);
    }, {input}, 1e-4f, 1e-6f, "Lgamma");
}

TEST(ExtendedParity, Digamma) {
auto input = tenzor::add(tenzor::abs(randn({16, 16}, DType::Float32, Device::cpu())),
                              full({16, 16}, 1.0f, DType::Float32, Device::cpu()));

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::digamma(inputs[0]);
    }, {input}, 1e-4f, 1e-6f, "Digamma");
}

TEST(ExtendedParity, Log2) {
auto input = tenzor::add(tenzor::abs(randn({16, 16}, DType::Float32, Device::cpu())),
                              full({16, 16}, 0.1f, DType::Float32, Device::cpu()));

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::log2(inputs[0]);
    }, {input}, 1e-5f, 1e-7f, "Log2");
}

TEST(ExtendedParity, Log10) {
auto input = tenzor::add(tenzor::abs(randn({16, 16}, DType::Float32, Device::cpu())),
                              full({16, 16}, 0.1f, DType::Float32, Device::cpu()));

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::log10(inputs[0]);
    }, {input}, 1e-5f, 1e-7f, "Log10");
}

TEST(ExtendedParity, Expm1) {
auto input = randn({16, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::expm1(inputs[0]);
    }, {input}, 1e-5f, 1e-7f, "Expm1");
}

TEST(ExtendedParity, Sinc) {
auto input = randn({16, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::sinc(inputs[0]);
    }, {input}, 1e-5f, 1e-7f, "Sinc");
}

// ============================================================================
// Bitwise Operations
// ============================================================================

TEST(ExtendedParity, BitwiseAnd) {
auto a = randint(0, 255, {16, 16}, DType::Int32, Device::cpu());
    auto b = randint(0, 255, {16, 16}, DType::Int32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::bitwise_and(inputs[0], inputs[1]);
    }, {a, b}, 0.0f, 0.0f, "BitwiseAnd");
}

TEST(ExtendedParity, BitwiseOr) {
auto a = randint(0, 255, {16, 16}, DType::Int32, Device::cpu());
    auto b = randint(0, 255, {16, 16}, DType::Int32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::bitwise_or(inputs[0], inputs[1]);
    }, {a, b}, 0.0f, 0.0f, "BitwiseOr");
}

// ============================================================================
// Multi-DType Parity (key ops across Float32, Float64)
// ============================================================================

TEST(ExtendedParity, MatMul_Float64) {
auto a = randn({8, 16}, DType::Float64, Device::cpu());
    auto b = randn({16, 8}, DType::Float64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::matmul(inputs[0], inputs[1]);
    }, {a, b}, 1e-12f, 1e-14f, "MatMul_Float64");
}

TEST(ExtendedParity, Softmax_Float64) {
auto input = randn({8, 16}, DType::Float64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> sm_inputs = {inputs[0]};
        return dispatch_single(OpId::Softmax, std::span<const Tensor>(sm_inputs), attrs);
    }, {input}, 1e-12f, 1e-14f, "Softmax_Float64");
}

TEST(ExtendedParity, Add_Float64) {
    auto a = randn({16, 16}, DType::Float64, Device::cpu());
    auto b = randn({16, 16}, DType::Float64, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 1e-14f, 0.0f, "Add_Float64");
}

// ============================================================================
// FFT Operations
// ============================================================================

TEST(ExtendedParity, FFT) {
    auto input = randn({16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::FFT>(ins)[0];
    }, {input}, 1e-4f, 1e-6f, "FFT");
}

TEST(ExtendedParity, IFFT) {
    auto input = randn({16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::IFFT>(ins)[0];
    }, {input}, 1e-4f, 1e-6f, "IFFT");
}

TEST(ExtendedParity, RFFT) {
    auto input = randn({16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::RFFT>(ins)[0];
    }, {input}, 1e-4f, 1e-6f, "RFFT");
}

TEST(ExtendedParity, IRFFT) {
    auto input = randn({16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::IRFFT>(ins)[0];
    }, {input}, 1e-4f, 1e-6f, "IRFFT");
}

// ============================================================================
// Sparse Operations
// ============================================================================

TEST(ExtendedParity, SparseSpMM) {
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

TEST(ExtendedParity, Scatter) {
    auto input = randn({4, 8}, DType::Float32, Device::cpu());
    auto src = randn({4, 3}, DType::Float32, Device::cpu());
    auto index = randint(0, 8, {4, 3}, DType::Int64, Device::cpu());

    test_operation_parity([&src, &index](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0], index.to(inputs[0].device()), src.to(inputs[0].device())};
        return dispatch<OpId::Scatter>(ins, attrs)[0];
    }, {input}, 1e-6f, 1e-8f, "Scatter");
}

TEST(ExtendedParity, SparseSpMV) {
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

TEST(ExtendedParity, SparseAdd) {
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

TEST(ExtendedParity, Logcumsumexp) {
    auto input = randn({8, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        OpAttributes attrs;
        attrs.set(AttrKey::Dim, int64_t(1));
        std::vector<Tensor> ins = {inputs[0]};
        return dispatch<OpId::Logcumsumexp>(ins, attrs)[0];
    }, {input}, 1e-4f, 1e-6f, "Logcumsumexp");
}

TEST(ExtendedParity, IndexAdd) {
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

TEST(ExtendedParity, Bucketize) {
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

TEST(ExtendedParity, SearchSorted) {
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
