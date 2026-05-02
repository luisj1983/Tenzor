/**
 * @file test_stable_math_parity.cpp
 * @brief Cross-backend parity for numerically-stable math ops.
 *
 * Covers OpIds: LogAddExp (680), LogAddExp2 (681), XLogY (682),
 * CosineSimilarity (683), Renorm (684). The audit (2026-05-02) flagged
 * these as missing dedicated parity coverage. The stability properties
 * (no overflow under exp, well-defined 0*log(0) = 0) make these worth
 * exercising on edge values across backends.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class StableMathParity : public BackendTest {};

// ----------------------------------------------------------------------------
// LogAddExp / LogAddExp2 — both should be stable for large operands
// ----------------------------------------------------------------------------

TEST_P(StableMathParity, LogAddExp_TypicalRange) {
    auto a = randn({4, 8}, DType::Float32, Device::cpu());
    auto b = randn({4, 8}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return logaddexp(in[0], in[1]);
    }, {a, b}, device, 1e-5f, 1e-7f, "LogAddExp");
}

TEST_P(StableMathParity, LogAddExp_LargeValues) {
    // Values where naive log(exp(a) + exp(b)) would overflow Float32 (max
    // exp arg ≈ 88). The stable formulation must still produce a finite,
    // matching result.
    auto a = full({4}, 100.0, DType::Float32, Device::cpu());
    auto b = full({4}, 95.0,  DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return logaddexp(in[0], in[1]);
    }, {a, b}, device, 1e-5f, 1e-5f, "LogAddExp_Large");
}

TEST_P(StableMathParity, LogAddExp2_TypicalRange) {
    auto a = randn({4, 8}, DType::Float32, Device::cpu()) * 5.0f;
    auto b = randn({4, 8}, DType::Float32, Device::cpu()) * 5.0f;
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return logaddexp2(in[0], in[1]);
    }, {a, b}, device, 1e-5f, 1e-7f, "LogAddExp2");
}

// ----------------------------------------------------------------------------
// XLogY — well-defined at x=0 (returns 0 even when y=0)
// ----------------------------------------------------------------------------

TEST_P(StableMathParity, XLogY_PositiveY) {
    auto x = randn({4, 8}, DType::Float32, Device::cpu());
    auto y = abs(randn({4, 8}, DType::Float32, Device::cpu())) + 1e-3f;
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return xlogy(in[0], in[1]);
    }, {x, y}, device, 1e-5f, 1e-7f, "XLogY");
}

TEST_P(StableMathParity, XLogY_ZeroX_ZeroY) {
    // The stability guarantee: 0 * log(0) is defined as 0 (not NaN).
    auto x = full({4}, 0.0, DType::Float32, Device::cpu());
    auto y = full({4}, 0.0, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return xlogy(in[0], in[1]);
    }, {x, y}, device, 0.0f, 0.0f, "XLogY_ZeroZero");
}

// ----------------------------------------------------------------------------
// CosineSimilarity
// ----------------------------------------------------------------------------

TEST_P(StableMathParity, CosineSimilarity_Dim1) {
    auto a = randn({4, 16}, DType::Float32, Device::cpu());
    auto b = randn({4, 16}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return cosine_similarity(in[0], in[1], /*dim=*/1);
    }, {a, b}, device, 1e-4f, 1e-5f, "CosineSimilarity_Dim1");
}

TEST_P(StableMathParity, CosineSimilarity_LastDim_NegativeIndex) {
    auto a = randn({3, 5, 8}, DType::Float32, Device::cpu());
    auto b = randn({3, 5, 8}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return cosine_similarity(in[0], in[1], /*dim=*/-1);
    }, {a, b}, device, 1e-4f, 1e-5f, "CosineSimilarity_LastDim_Neg");
}

// ----------------------------------------------------------------------------
// Renorm — re-scales rows along dim so each row has p-norm ≤ maxnorm
// ----------------------------------------------------------------------------

TEST_P(StableMathParity, Renorm_L2_Dim0) {
    auto x = randn({6, 8}, DType::Float32, Device::cpu()) * 5.0f;
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return renorm(in[0], /*p=*/2.0, /*dim=*/0, /*maxnorm=*/3.0);
    }, {x}, device, 1e-5f, 1e-7f, "Renorm_L2_Dim0");
}

TEST_P(StableMathParity, Renorm_L1_Dim1) {
    auto x = randn({4, 6}, DType::Float32, Device::cpu()) * 2.0f;
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return renorm(in[0], /*p=*/1.0, /*dim=*/1, /*maxnorm=*/4.0);
    }, {x}, device, 1e-5f, 1e-7f, "Renorm_L1_Dim1");
}

INSTANTIATE_BACKEND_TESTS(StableMathParity);

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
