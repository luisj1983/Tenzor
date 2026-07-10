// Numeric-parity regressions for the MLIR/IREE JIT path.
//
//   M8 (SubprocessFloat32BitExact): the subprocess output path
//       (iree-run-module) must be BIT-EXACT. Its stdout printing rounds floats
//       to ~6 significant figures, silently losing precision vs the InProcess
//       buffer-view path; the fix writes outputs to a NumPy .npy file and reads
//       them back. Verified by forcing the subprocess path on the CPU build
//       (TENZOR_MLIR_FORCE_SUBPROCESS) and requiring an f32 result whose ~O(1e3)
//       magnitude makes 6-sig-fig ASCII lose ~1e-2, far above the tolerance.
//       (f64 is not used: IREE demotes f64→f32, so an f64 input mismatches and
//       degrades to eager — a vacuous test.)
//
//   M3 (Float16Softmax/LayerNormWideAxis): the softmax/layernorm lowerings must
//       accumulate the F16 reductions in F32 (matching the eager kernels, whose
//       sums are F32). Over a long reduction axis, half-precision accumulation
//       diverges; widening keeps compiled == eager.
//
// Every test asserts cache_stats().misses > 0 so a silent eager fallback (which
// would make the comparison vacuously pass) is caught rather than hidden.

#include "tenzor/jit/compile.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/nn/functional.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/tenzor.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <vector>

#include "mlir_target_util.hpp"

namespace {

using namespace tenzor;
namespace mt = ::tenzor::testing::mlir;
namespace mj = ::tenzor::jit::mlir_jit;

auto max_abs_diff_f64(const Tensor& a, const Tensor& b) -> double {
    auto ac = a.to(DType::Float64).to(Device::cpu());
    auto bc = b.to(DType::Float64).to(Device::cpu());
    return ::tenzor::max(::tenzor::abs(ac - bc)).item<double>();
}

void require_cpu_iree() {
    // llvm-cpu must be available in an MLIR-configured build; its absence is a
    // real environment fault, not something to skip past.
    ASSERT_TRUE(mt::target_hw_present("llvm-cpu") &&
                mt::iree_target_supported("llvm-cpu"))
        << "llvm-cpu IREE target unavailable in an MLIR build";
}

// F020: compile `fn` on EVERY available IREE target and assert it matches the
// eager reference on each — the accumulation tests were hardcoded to llvm-cpu, so
// a GPU-only accumulation narrowing went unseen. A target where IREE genuinely
// cannot run the dtype (e.g. f64 transcendentals lack a libm symbol) degrades to
// eager, which IS the correct reference, so the numeric check still holds;
// misses>=1 confirms a compile was attempted per target.
void assert_parity_over_targets(const char* name,
                                ::tenzor::jit::CompiledFunction::FnType fn,
                                const Variable& x, const Tensor& eager,
                                double tol) {
    for (const auto& target : mt::available_iree_targets()) {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend = "mlir";
        cfg.target  = target;
        ::tenzor::jit::CompiledFunction compiled(fn, cfg);
        mj::reset_cache_stats();
        Variable out;
        ASSERT_NO_THROW({ out = compiled(x); })
            << name << " threw on target=" << target;
        ASSERT_GE(mj::cache_stats().misses, 1u)
            << name << " did not attempt an IREE compile on target=" << target;
        EXPECT_LT(max_abs_diff_f64(eager, out.tensor()), tol)
            << name << " diverges from eager on target=" << target;
    }
}

// M8 — the subprocess output path must be bit-exact, not ~6-sig-fig ASCII.
TEST(MlirNumericParity, SubprocessFloat32BitExact) {
    mt::ensure_core_init();
    require_cpu_iree();

    ::setenv("TENZOR_MLIR_FORCE_SUBPROCESS", "1", /*overwrite=*/1);

    auto fn = ::tenzor::jit::CompiledFunction::FnType(
        [](const Variable& x) -> Variable { return (x * x) + x; });
    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target = "llvm-cpu";
    ::tenzor::jit::CompiledFunction compiled(fn, cfg);

    // f(x)=x^2+x with x~[600,950] gives ~[3.6e5,9.0e5]; 6-sig-fig ASCII of a
    // ~9e5 magnitude loses ~1e0 absolute, while the compiled f32 elementwise op
    // matches eager f32 to ~1 ULP (~1e-1 at this magnitude).
    const float vals[6] = {601.125f, 733.5f, 827.875f, 900.0625f,
                           785.75f, 948.5f};
    Tensor x_t({6}, DType::Float32, Device::cpu());
    for (int i = 0; i < 6; ++i) x_t.data<float>()[i] = vals[i];
    Variable x(x_t, /*requires_grad=*/false);

    const Variable eager = (x * x) + x;
    mj::reset_cache_stats();
    Variable out;
    ASSERT_NO_THROW({ out = compiled(x); })
        << "forced-subprocess f32 inference must not throw";

    ASSERT_GE(mj::cache_stats().misses, 1u)
        << "forced-subprocess f32 inference did NOT run through IREE (it fell "
           "back to eager — the precision check below would be vacuous)";

    // .npy path is bit-exact (~1e-4 f32 rounding); ASCII stdout loses ~1e-2.
    EXPECT_LT(max_abs_diff_f64(eager.tensor(), out.tensor()), 1e-3)
        << "subprocess f32 output lost precision (ASCII stdout path?)";

    ::unsetenv("TENZOR_MLIR_FORCE_SUBPROCESS");
}

// Regression guard for the subprocess .npy output path. iree-run-module picks
// its `--output=@path` serialization from the file EXTENSION, so the temp file
// MUST end in ".npy"; otherwise iree writes a raw buffer that read_npy_output
// rejects as "bad .npy magic". In NON-strict mode that failure is a SILENT
// eager fallback — which is exactly why SubprocessFloat32BitExact above passed
// vacuously while ROCm JIT (subprocess-only) never actually ran through IREE.
// Under STRICT the failure THROWS, so this test fails loudly if the extension
// (or any other subprocess-run breakage) ever regresses.
TEST(MlirNumericParity, SubprocessStrictRunsThroughIree) {
    mt::ensure_core_init();
    require_cpu_iree();

    ::setenv("TENZOR_MLIR_FORCE_SUBPROCESS", "1", /*overwrite=*/1);

    auto fn = ::tenzor::jit::CompiledFunction::FnType(
        [](const Variable& x) -> Variable { return (x * x) + x; });
    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target  = "llvm-cpu";
    cfg.strict  = true;  // a broken .npy read must THROW, not fall back to eager
    ::tenzor::jit::CompiledFunction compiled(fn, cfg);

    Tensor x_t({6}, DType::Float32, Device::cpu());
    const float vals[6] = {1.5f, 2.25f, 3.0f, 4.75f, 5.5f, 6.125f};
    for (int i = 0; i < 6; ++i) x_t.data<float>()[i] = vals[i];
    Variable x(x_t, /*requires_grad=*/false);

    const Variable eager = (x * x) + x;
    mj::reset_cache_stats();
    Variable out;
    ASSERT_NO_THROW({ out = compiled(x); })
        << "strict forced-subprocess run threw — the subprocess .npy output "
           "path is broken (temp file likely lost its .npy extension, so "
           "iree-run-module wrote a raw buffer that read_npy_output rejects)";
    ASSERT_GE(mj::cache_stats().misses, 1u)
        << "strict forced-subprocess run did not go through IREE";
    EXPECT_LT(max_abs_diff_f64(eager.tensor(), out.tensor()), 1e-3);

    ::unsetenv("TENZOR_MLIR_FORCE_SUBPROCESS");
}

// M3 — F16 softmax over a long axis: the exp-sum must accumulate in F32 to match
// the eager kernel (Kahan F32 accumulator). F16 accumulation over 2048 elements
// diverges well past F16 ULP.
TEST(MlirNumericParity, Float16SoftmaxWideAxis) {
    mt::ensure_core_init();
    require_cpu_iree();

    auto fn = ::tenzor::jit::CompiledFunction::FnType(
        [](const Variable& x) -> Variable { return nn::softmax(x, -1); });

    // Moderately peaked logits (×4): a handful of probabilities are O(0.1), so
    // the exp-sum accumulation error scales up into a visible per-prob abs diff
    // (a flat distribution spreads the error too thin to observe).
    Tensor x_t = (::tenzor::randn({4, 2048}, DType::Float32, Device::cpu()) * 4.0f)
                     .to(DType::Float16);
    Variable x(x_t, /*requires_grad=*/false);

    const Variable eager = nn::softmax(x, -1);
    // Widened compiled softmax matches eager to ~F16 ULP; unwidened F16
    // accumulation over 2048 elements diverges by >1e-2. Fanned over every IREE
    // target (F020) so a GPU-only narrowing is caught, not just llvm-cpu.
    assert_parity_over_targets("F16 softmax wide-axis", fn, x, eager.tensor(),
                               3e-3);
}

// Fix #10 coverage — the JIT numeric-parity suite was Float32/Float16-dominant,
// so the float64-accumulator divergence class (a compiled reduction silently
// accumulating in F32 while eager uses F64) was untested. F64 softmax over a wide
// axis exercises an internal exp-sum reduction: both eager and the compiled
// (mlir/llvm-cpu) path are F64, so an F32-narrowed accumulation of 4096 exps
// would diverge at ~1e-7 — far above the F64 reduction-order noise this tight
// 1e-9 bound tolerates.
TEST(MlirNumericParity, Float64SoftmaxWideAxisAccumulates) {
    mt::ensure_core_init();
    require_cpu_iree();

    auto fn = ::tenzor::jit::CompiledFunction::FnType(
        [](const Variable& x) -> Variable { return nn::softmax(x, -1); });

    Tensor x_t = (::tenzor::randn({4, 4096}, DType::Float32, Device::cpu()) * 4.0f)
                     .to(DType::Float64);
    Variable x(x_t, /*requires_grad=*/false);

    const Variable eager = nn::softmax(x, -1);
    assert_parity_over_targets("F64 softmax wide-axis", fn, x, eager.tensor(),
                               1e-9);
}

// M3 — F16 LayerNorm over a long axis: mean/variance must accumulate in F32.
TEST(MlirNumericParity, Float16LayerNormWideAxis) {
    mt::ensure_core_init();
    require_cpu_iree();

    const int64_t D = 2048;
    auto fn = ::tenzor::jit::CompiledFunction::FnType(
        [D](const Variable& x) -> Variable {
            return nn::functional::layer_norm(x, {D});
        });
    Tensor x_t = ::tenzor::randn({4, D}, DType::Float32, Device::cpu())
                     .to(DType::Float16);
    Variable x(x_t, /*requires_grad=*/false);

    const Variable eager = nn::functional::layer_norm(x, {D});
    // Fanned over every IREE target (F020): a GPU-only F16 accumulation narrowing
    // in the mean/variance reduction is caught, not just llvm-cpu.
    assert_parity_over_targets("F16 LayerNorm wide-axis", fn, x, eager.tensor(),
                               5e-3);
}

// F16 AvgPool2d (my fix): the reduce-window sum must accumulate in F32 to match
// the eager kernel (float Compute). A 16x16 window sums 256 F16 values, so F16
// accumulation would diverge; widening keeps compiled == eager.
TEST(MlirNumericParity, Float16AvgPoolWideWindow) {
    mt::ensure_core_init();
    require_cpu_iree();

    auto fn = ::tenzor::jit::CompiledFunction::FnType(
        [](const Variable& x) -> Variable {
            return nn::functional::avg_pool2d(x, {16, 16}, {16, 16});
        });
    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target = "llvm-cpu";
    ::tenzor::jit::CompiledFunction compiled(fn, cfg);

    Tensor x_t =
        (::tenzor::randn({1, 4, 32, 32}, DType::Float32, Device::cpu()) * 4.0f)
            .to(DType::Float16);
    Variable x(x_t, /*requires_grad=*/false);

    const Variable eager = nn::functional::avg_pool2d(x, {16, 16}, {16, 16});
    mj::reset_cache_stats();
    Variable out;
    ASSERT_NO_THROW({ out = compiled(x); });
    ASSERT_GE(mj::cache_stats().misses, 1u)
        << "F16 AvgPool did NOT run through IREE (eager fallback)";
    double d = max_abs_diff_f64(eager.tensor(), out.tensor());
    fprintf(stderr, "[parity] F16 AvgPool max_abs_diff = %.6g\n", d);
    EXPECT_LT(d, 5e-3)
        << "F16 AvgPool diverges from eager (reduce-window accumulation not "
           "widened to F32?)";
}

// F16 scaled-dot-product attention: empirically bounds the divergence of the
// MLIR flash-attention expand path vs eager (F32 throughout). If this exceeds
// F16 tolerance, the QK/AV scores need widening in handle_flash_attention_expand.
TEST(MlirNumericParity, Float16ScaledDotProductAttentionParity) {
    mt::ensure_core_init();
    require_cpu_iree();

    auto fn = ::tenzor::jit::CompiledFunction::FnType(
        [](const Variable& x) -> Variable {
            return nn::functional::scaled_dot_product_attention(x, x, x);
        });
    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target = "llvm-cpu";
    ::tenzor::jit::CompiledFunction compiled(fn, cfg);

    Tensor x_t = ::tenzor::randn({1, 4, 32, 32}, DType::Float32, Device::cpu())
                     .to(DType::Float16);
    Variable x(x_t, /*requires_grad=*/false);

    const Variable eager = nn::functional::scaled_dot_product_attention(x, x, x);
    mj::reset_cache_stats();
    Variable out;
    ASSERT_NO_THROW({ out = compiled(x); });
    ASSERT_GE(mj::cache_stats().misses, 1u)
        << "F16 SDPA did NOT run through IREE (eager fallback)";
    double d = max_abs_diff_f64(eager.tensor(), out.tensor());
    fprintf(stderr, "[parity] F16 SDPA max_abs_diff = %.6g\n", d);
    EXPECT_LT(d, 2e-2) << "F16 SDPA diverges from eager beyond F16 tolerance";
}

// F16 Conv2d: empirically bounds MLIR conv (im2col dot_general / convolution in
// storage dtype) vs eager. If beyond F16 tolerance, the conv accumulation needs
// widening in handle_conv2d.
TEST(MlirNumericParity, Float16Conv2dParity) {
    mt::ensure_core_init();
    require_cpu_iree();

    Tensor w_t =
        (::tenzor::randn({8, 4, 3, 3}, DType::Float32, Device::cpu()) * 0.1f)
            .to(DType::Float16);
    Variable w(w_t, /*requires_grad=*/false);
    auto fn = ::tenzor::jit::CompiledFunction::FnType(
        [w](const Variable& x) -> Variable {
            return nn::functional::conv2d(x, w);
        });
    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target = "llvm-cpu";
    ::tenzor::jit::CompiledFunction compiled(fn, cfg);

    Tensor x_t =
        (::tenzor::randn({1, 4, 16, 16}, DType::Float32, Device::cpu()) * 2.0f)
            .to(DType::Float16);
    Variable x(x_t, /*requires_grad=*/false);

    const Variable eager = nn::functional::conv2d(x, w);
    mj::reset_cache_stats();
    Variable out;
    ASSERT_NO_THROW({ out = compiled(x); });
    ASSERT_GE(mj::cache_stats().misses, 1u)
        << "F16 Conv2d did NOT run through IREE (eager fallback)";
    double d = max_abs_diff_f64(eager.tensor(), out.tensor());
    fprintf(stderr, "[parity] F16 Conv2d max_abs_diff = %.6g\n", d);
    EXPECT_LT(d, 2e-2) << "F16 Conv2d diverges from eager beyond F16 tolerance";
}

}  // namespace

int main(int argc, char** argv) {
    ::tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    ::tenzor::finalize();
    return result;
}
