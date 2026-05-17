/**
 * @file test_softmax_stride_audit.cpp
 * @brief Regression test for audit P0 #1: softmax/log_softmax stride-from-shape bug.
 *
 * Each of the four kernel entry points (softmax, log_softmax, and their
 * backward kernels) indexed input data via
 *   idx = (i * dim_size + j) * inner_size + k
 * without calling .contiguous() first, silently reading wrong memory for
 * any transposed / permuted view tensor. The forward paths are exercised
 * here by comparing output on a non-contiguous transpose view vs. the
 * equivalent explicitly-copied contiguous tensor.
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include <cmath>
#include <limits>

using namespace tenzor;

// Test environment that initialises Tenzor once for the whole binary.
class SoftmaxStrideAuditEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};

static ::testing::Environment* const stride_audit_env =
    ::testing::AddGlobalTestEnvironment(new SoftmaxStrideAuditEnv);

// ---------------------------------------------------------------------------
// Audit P0 #1: softmax must produce identical output on a transposed view
// as on the equivalent explicitly-permuted-and-contiguous tensor. A stride-
// ignoring kernel will silently differ.
// ---------------------------------------------------------------------------
TEST(SoftmaxStrideAudit, TransposedInputMatchesContiguous) {
    // src shape {4, 3}, row-major contiguous
    auto src   = arange(1.0, 13.0, 1.0).reshape({4, 3});
    auto view_t = transpose(src, 0, 1);      // {3, 4} non-contiguous view
    auto ref_t  = view_t.contiguous();       // {3, 4} packed copy

    EXPECT_FALSE(view_t.is_contiguous()) << "Precondition: transpose must yield a non-contiguous tensor";
    EXPECT_TRUE(ref_t.is_contiguous())   << "Precondition: .contiguous() must yield a contiguous tensor";

    // Wrap in Variables (no grad tracking needed for forward-only comparison)
    Variable view_var(view_t, false);
    Variable ref_var(ref_t,   false);

    auto out_view = tenzor::softmax(view_var, -1);
    auto out_ref  = tenzor::softmax(ref_var,  -1);

    // Compare element-wise via abs diff; promote to float64 to avoid rounding
    auto diff_t = abs(out_view.tensor().to(DType::Float64)
                      - out_ref.tensor().to(DType::Float64));
    auto max_diff = max(diff_t).item<double>();

    EXPECT_LT(max_diff, 1e-6)
        << "softmax on transposed view differs from contiguous copy "
           "(stride-from-shape bug, audit P0 #1). max_diff=" << max_diff;
}

TEST(SoftmaxStrideAudit, LogSoftmaxTransposedInputMatchesContiguous) {
    auto src    = arange(1.0, 13.0, 1.0).reshape({4, 3});
    auto view_t = transpose(src, 0, 1);      // {3, 4} non-contiguous view
    auto ref_t  = view_t.contiguous();

    EXPECT_FALSE(view_t.is_contiguous());
    EXPECT_TRUE(ref_t.is_contiguous());

    Variable view_var(view_t, false);
    Variable ref_var(ref_t,   false);

    auto out_view = tenzor::log_softmax(view_var, -1);
    auto out_ref  = tenzor::log_softmax(ref_var,  -1);

    auto diff_t  = abs(out_view.tensor().to(DType::Float64)
                       - out_ref.tensor().to(DType::Float64));
    auto max_diff = max(diff_t).item<double>();

    EXPECT_LT(max_diff, 1e-6)
        << "log_softmax on transposed view differs from contiguous copy "
           "(stride-from-shape bug, audit P0 #1). max_diff=" << max_diff;
}

// Non-last-dim axis: softmax along dim=0 of a transposed view
TEST(SoftmaxStrideAudit, TransposedInputDim0MatchesContiguous) {
    auto src    = arange(1.0, 25.0, 1.0).reshape({4, 6});
    auto view_t = transpose(src, 0, 1);      // {6, 4} non-contiguous view
    auto ref_t  = view_t.contiguous();

    EXPECT_FALSE(view_t.is_contiguous()) << "Precondition: transpose must yield a non-contiguous tensor";
    EXPECT_TRUE(ref_t.is_contiguous())   << "Precondition: .contiguous() must yield a contiguous tensor";

    Variable view_var(view_t, false);
    Variable ref_var(ref_t,   false);

    auto out_view = tenzor::softmax(view_var, 0);
    auto out_ref  = tenzor::softmax(ref_var,  0);

    auto diff_t  = abs(out_view.tensor().to(DType::Float64)
                       - out_ref.tensor().to(DType::Float64));
    auto max_diff = max(diff_t).item<double>();

    EXPECT_LT(max_diff, 1e-6)
        << "softmax(dim=0) on transposed view differs from contiguous copy. "
           "max_diff=" << max_diff;
}

// ---------------------------------------------------------------------------
// Backward-path regression: softmax_backward_kernel must produce identical
// gradients on a transposed view as on the equivalent contiguous tensor.
// A stride-ignoring backward kernel would compute wrong grad values silently.
// ---------------------------------------------------------------------------
TEST(SoftmaxStrideAudit, BackwardTransposedInputMatchesContiguous) {
    // Deterministic source data avoids flaky failures from large random values.
    auto src_t = arange(1.0, 13.0, 1.0).reshape({4, 3});  // {4,3} contiguous

    // Two independent Variables share the same underlying data pattern but
    // one uses a non-contiguous transposed view and one is explicitly packed.
    auto view_t = transpose(src_t, 0, 1);          // {3, 4} non-contiguous
    auto ref_t  = view_t.contiguous();             // {3, 4} packed copy

    EXPECT_FALSE(view_t.is_contiguous()) << "Precondition: view must be non-contiguous";
    EXPECT_TRUE(ref_t.is_contiguous())   << "Precondition: contiguous copy must be contiguous";

    Variable x_view(view_t, /*requires_grad=*/true);
    Variable x_ref (ref_t,  /*requires_grad=*/true);

    auto y_view = softmax(x_view, -1);
    auto y_ref  = softmax(x_ref,  -1);

    auto loss_view = sum(y_view);
    auto loss_ref  = sum(y_ref);

    loss_view.backward();
    loss_ref.backward();

    ASSERT_TRUE(x_view.grad().has_value()) << "No gradient on transposed-view input";
    ASSERT_TRUE(x_ref.grad().has_value())  << "No gradient on contiguous input";

    // View's grad is in the transposed (non-contiguous) layout; pack before diff.
    auto gv = x_view.grad().value().contiguous().to(DType::Float64);
    auto gr = x_ref.grad().value().contiguous().to(DType::Float64);

    auto diff_t   = abs(gv - gr);
    auto max_diff = max(diff_t).item<double>();

    EXPECT_LT(max_diff, 1e-6)
        << "softmax backward on transposed view differs from contiguous copy "
           "(stride-from-shape bug in softmax_backward_kernel). max_diff=" << max_diff;
}

TEST(SoftmaxStrideAudit, BackwardLogSoftmaxTransposedInputMatchesContiguous) {
    auto src_t = arange(1.0, 13.0, 1.0).reshape({4, 3});  // {4,3} contiguous

    auto view_t = transpose(src_t, 0, 1);          // {3, 4} non-contiguous
    auto ref_t  = view_t.contiguous();             // {3, 4} packed copy

    EXPECT_FALSE(view_t.is_contiguous()) << "Precondition: view must be non-contiguous";
    EXPECT_TRUE(ref_t.is_contiguous())   << "Precondition: contiguous copy must be contiguous";

    Variable x_view(view_t, /*requires_grad=*/true);
    Variable x_ref (ref_t,  /*requires_grad=*/true);

    auto y_view = log_softmax(x_view, -1);
    auto y_ref  = log_softmax(x_ref,  -1);

    auto loss_view = sum(y_view);
    auto loss_ref  = sum(y_ref);

    loss_view.backward();
    loss_ref.backward();

    ASSERT_TRUE(x_view.grad().has_value()) << "No gradient on transposed-view input";
    ASSERT_TRUE(x_ref.grad().has_value())  << "No gradient on contiguous input";

    auto gv = x_view.grad().value().contiguous().to(DType::Float64);
    auto gr = x_ref.grad().value().contiguous().to(DType::Float64);

    auto diff_t   = abs(gv - gr);
    auto max_diff = max(diff_t).item<double>();

    EXPECT_LT(max_diff, 1e-6)
        << "log_softmax backward on transposed view differs from contiguous copy "
           "(stride-from-shape bug in log_softmax_backward_kernel). max_diff=" << max_diff;
}
