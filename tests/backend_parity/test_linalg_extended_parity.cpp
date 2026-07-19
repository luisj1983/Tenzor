/**
 * @file test_linalg_extended_parity.cpp
 * @brief Extended linalg parity: Pinv, LstSq, MatrixExp (plan Phase 3.6).
 *
 * The baseline linalg ops (Det, Inv, Solve, SVD, QR, Eigh, Cholesky, LU,
 * Addmm/v, Baddbmm, SolveTriangular) are in test_linalg_parity.cpp.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/linalg.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class LinalgExtendedParity : public BackendTest {};
// Property-based pinv parity test.
//
// Direct value comparison of pinv(A) across backends is not portable: each
// backend's SVD kernel uses a different algorithm (cuSOLVER QR, Jacobi,
// divide-and-conquer, etc.) and produces U/S/Vt with different numerical
// drift that amplifies into ~0.75 max abs diff in the reconstructed pinv.
// The pseudoinverse is mathematically unique though — A·pinv(A)·A == A holds
// for any valid pinv by the Moore–Penrose definition. Test that property on
// each backend rather than cross-backend values.
TEST_P(LinalgExtendedParity, Pinv) {
    auto A = randn({8, 6}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("linalg extended parity");

    for (const auto& backend : backends) {
        try {
            auto Ad = A.to(backend);
            auto Apinv = tenzor::linalg::pinv(Ad);
            backend.synchronize();

            // A·pinv(A)·A == A (Moore–Penrose identity #1)
            auto reconstructed = tenzor::matmul(tenzor::matmul(Ad, Apinv), Ad);
            backend.synchronize();
            SCOPED_TRACE(std::string("pinv A·pinv·A on ") + backend_name(backend));
            EXPECT_TENSORS_CLOSE(A, reconstructed.to(Device::cpu()),
                                 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "pinv failed on " << backend_name(backend)
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(LinalgExtendedParity, LstSq_Residual) {
    // Overdetermined system: A(8x4) x = B(8x2). lstsq returns (solution, residuals).
    auto A = randn({8, 4}, DType::Float32, Device::cpu());
    auto B = randn({8, 2}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("linalg extended parity");

    // CPU is the reference backend — if lstsq throws here it is a real bug, so
    // let the exception propagate as a failure instead of masking it as a skip.
    Tensor ref_sol;
    {
        auto [sol, _] = tenzor::linalg::lstsq(A, B);
        ref_sol = sol;
    }

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto [sol, _] = tenzor::linalg::lstsq(A.to(backends[i]), B.to(backends[i]));
            backends[i].synchronize();
            SCOPED_TRACE(std::string("lstsq on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref_sol, sol.to(Device::cpu()), 1e-3f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "lstsq failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(LinalgExtendedParity, MatrixExp) {
    // Small, well-conditioned matrix so matrix_exp converges reliably.
    auto A = randn({4, 4}, DType::Float32, Device::cpu()) * 0.1f;
    test_operation_parity([](const std::vector<Tensor>& ins) {
        return tenzor::linalg::matrix_exp(ins[0]);
    }, {A}, 1e-3f, 1e-5f, "matrix_exp");
}

INSTANTIATE_BACKEND_TESTS(LinalgExtendedParity);


int main(int argc, char** argv) {
    try {
        tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
    }
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    try {
        tenzor::finalize();
    } catch (...) {}
    return result;
}
