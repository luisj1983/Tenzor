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

// M3 — F16 softmax over a long axis: the exp-sum must accumulate in F32 to match
// the eager kernel (Kahan F32 accumulator). F16 accumulation over 2048 elements
// diverges well past F16 ULP.
TEST(MlirNumericParity, Float16SoftmaxWideAxis) {
    mt::ensure_core_init();
    require_cpu_iree();

    auto fn = ::tenzor::jit::CompiledFunction::FnType(
        [](const Variable& x) -> Variable { return nn::softmax(x, -1); });
    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target = "llvm-cpu";
    ::tenzor::jit::CompiledFunction compiled(fn, cfg);

    // Moderately peaked logits (×4): a handful of probabilities are O(0.1), so
    // the exp-sum accumulation error scales up into a visible per-prob abs diff
    // (a flat distribution spreads the error too thin to observe).
    Tensor x_t = (::tenzor::randn({4, 2048}, DType::Float32, Device::cpu()) * 4.0f)
                     .to(DType::Float16);
    Variable x(x_t, /*requires_grad=*/false);

    const Variable eager = nn::softmax(x, -1);
    mj::reset_cache_stats();
    Variable out;
    ASSERT_NO_THROW({ out = compiled(x); });
    ASSERT_GE(mj::cache_stats().misses, 1u)
        << "F16 softmax did NOT run through IREE (eager fallback)";

    // Widened compiled softmax matches eager to ~F16 ULP; unwidened F16
    // accumulation over 2048 elements diverges by >1e-2.
    EXPECT_LT(max_abs_diff_f64(eager.tensor(), out.tensor()), 3e-3)
        << "F16 softmax diverges from eager (F16 accumulation not widened?)";
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
    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target = "llvm-cpu";
    ::tenzor::jit::CompiledFunction compiled(fn, cfg);

    Tensor x_t = ::tenzor::randn({4, D}, DType::Float32, Device::cpu())
                     .to(DType::Float16);
    Variable x(x_t, /*requires_grad=*/false);

    const Variable eager = nn::functional::layer_norm(x, {D});
    mj::reset_cache_stats();
    Variable out;
    ASSERT_NO_THROW({ out = compiled(x); });
    ASSERT_GE(mj::cache_stats().misses, 1u)
        << "F16 LayerNorm did NOT run through IREE (eager fallback)";

    EXPECT_LT(max_abs_diff_f64(eager.tensor(), out.tensor()), 5e-3)
        << "F16 LayerNorm diverges from eager (F16 accumulation not widened?)";
}

}  // namespace

int main(int argc, char** argv) {
    ::tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    ::tenzor::finalize();
    return result;
}
