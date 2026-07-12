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

#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/jit/compile.hpp"
#include "tenzor/jit/mlir/iree_compile.hpp"
#include "tenzor/nn/functional.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
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

// JIT-R010: BFloat16 has no NPY encoding in iree-run-module's numpy_io (see
// invoke_subprocess's comment on the output side of this same limitation),
// so build_input_arg's file-based marshaling never triggered for it and a
// large (>4096-element) BF16 input always built a giant inline ASCII
// --input= argv string -- risking an OS argv-size (E2BIG) failure on a
// genuinely large tensor. The fix adds a raw-binary
// --input=SHAPExTYPE=@path form (shape/dtype in the flag text, only value
// bytes in the file) specifically for dtypes lacking an NPY encoding. Force
// subprocess mode and use an 8192-element BF16 input (well past the 4096
// threshold) to exercise that path end-to-end and confirm it's still exact.
TEST(MlirNumericParity, SubprocessBFloat16LargeInputMarshals) {
    mt::ensure_core_init();
    require_cpu_iree();

    ::setenv("TENZOR_MLIR_FORCE_SUBPROCESS", "1", /*overwrite=*/1);

    auto fn = ::tenzor::jit::CompiledFunction::FnType(
        [](const Variable& x) -> Variable { return x + x; });
    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target = "llvm-cpu";
    ::tenzor::jit::CompiledFunction compiled(fn, cfg);

    Tensor x_f32 = tenzor::randn({64, 128}, DType::Float32, Device::cpu());  // 8192 elements
    Tensor x_bf16 = x_f32.to(DType::BFloat16);
    Variable x(x_bf16, /*requires_grad=*/false);

    const Variable eager = x + x;
    mj::reset_cache_stats();
    Variable out;
    ASSERT_NO_THROW({ out = compiled(x); })
        << "forced-subprocess large-BF16 inference must not throw (E2BIG?)";

    ASSERT_GE(mj::cache_stats().misses, 1u)
        << "forced-subprocess large-BF16 inference did NOT run through IREE "
           "(it fell back to eager -- the precision check below would be "
           "vacuous)";

    EXPECT_LT(max_abs_diff_f64(eager.tensor(), out.tensor()), 1e-2)
        << "subprocess BF16 large-input marshaling lost precision or "
           "mismatched shape/dtype";

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
// JIT-R029: was hardcoded to llvm-cpu, unlike its siblings (Float16SoftmaxWideAxis
// etc.) already fanned over every IREE target via assert_parity_over_targets
// (F020) -- AvgPool2d's reduce-window sum is exactly the F16-accumulation-
// sensitive pattern F020 exists to protect, so a GPU-only narrowing here would
// have gone undetected. Fanned out to match.
TEST(MlirNumericParity, Float16AvgPoolWideWindow) {
    mt::ensure_core_init();
    require_cpu_iree();

    auto fn = ::tenzor::jit::CompiledFunction::FnType(
        [](const Variable& x) -> Variable {
            return nn::functional::avg_pool2d(x, {16, 16}, {16, 16});
        });

    Tensor x_t =
        (::tenzor::randn({1, 4, 32, 32}, DType::Float32, Device::cpu()) * 4.0f)
            .to(DType::Float16);
    Variable x(x_t, /*requires_grad=*/false);

    const Variable eager = nn::functional::avg_pool2d(x, {16, 16}, {16, 16});
    assert_parity_over_targets("F16 AvgPool wide window", fn, x, eager.tensor(),
                               5e-3);
}

// F16 scaled-dot-product attention: empirically bounds the divergence of the
// MLIR flash-attention expand path vs eager (F32 throughout). If this exceeds
// F16 tolerance, the QK/AV scores need widening in handle_flash_attention_expand.
// JIT-R029: was hardcoded to llvm-cpu. SDPA's QK/AV score accumulation is
// exactly the F16-accumulation-sensitive pattern F020 protects; fanned out to
// every IREE target like its already-fixed siblings so a GPU-only narrowing
// in handle_flash_attention_expand wouldn't go unseen.
TEST(MlirNumericParity, Float16ScaledDotProductAttentionParity) {
    mt::ensure_core_init();
    require_cpu_iree();

    auto fn = ::tenzor::jit::CompiledFunction::FnType(
        [](const Variable& x) -> Variable {
            return nn::functional::scaled_dot_product_attention(x, x, x);
        });

    Tensor x_t = ::tenzor::randn({1, 4, 32, 32}, DType::Float32, Device::cpu())
                     .to(DType::Float16);
    Variable x(x_t, /*requires_grad=*/false);

    const Variable eager = nn::functional::scaled_dot_product_attention(x, x, x);
    assert_parity_over_targets("F16 SDPA", fn, x, eager.tensor(), 2e-2);
}

// F16 Conv2d: empirically bounds MLIR conv (im2col dot_general / convolution in
// storage dtype) vs eager. If beyond F16 tolerance, the conv accumulation needs
// widening in handle_conv2d.
// JIT-R029: was hardcoded to llvm-cpu. Conv2d's im2col dot_general
// accumulation is exactly the F16-accumulation-sensitive pattern F020
// protects; fanned out to every IREE target like its already-fixed siblings
// so a GPU-only narrowing in handle_conv2d wouldn't go unseen.
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

    Tensor x_t =
        (::tenzor::randn({1, 4, 16, 16}, DType::Float32, Device::cpu()) * 2.0f)
            .to(DType::Float16);
    Variable x(x_t, /*requires_grad=*/false);

    const Variable eager = nn::functional::conv2d(x, w);
    assert_parity_over_targets("F16 Conv2d", fn, x, eager.tensor(), 2e-2);
}

// JIT-R003: Conv1d/Conv3d shape inference (Graph::infer_types /
// infer_symbolic_types) guarded output-shape computation on `shape.size() ==
// 4`, so a traced Conv1d (rank-3) or Conv3d (rank-5) node got no static type
// at all -- correct in the native interpreter (execute_node rank-dispatches
// correctly), wrong/type-invalid for the MLIR/IREE compile path. Fixed via
// conv_output_shape()/conv_sym_output_shape() (graph.cpp), which handle any
// rank. This test only verifies that fix does not crash or silently misuse
// out-of-range shape indices when actually reaching MLIR lowering -- NOT
// that Conv1d/Conv3d genuinely compile via IREE: handle_conv2d
// (lowering.cpp) itself still hardcodes a 2-spatial-dim (H, W) attribute
// read for OpType::Conv2d (the same IR node type OpId::Conv1dForward/
// Conv3dForward both map to -- see tracing_interceptor.cpp), so a rank-3/5
// node reaching it throws a clean, caught GraphToMLIR error and the C2
// eager-fallback safety net (compile.cpp) takes over -- never a silent
// wrong-shape read or a crash. Extending handle_conv2d itself to be
// rank-generic (1D/2D/3D stablehlo.convolution dimension_numbers) is a
// separate, substantially larger undertaking tracked as JIT-R003b below.
TEST(MlirNumericParity, Conv1dConv3dShapeInferenceNoCrash) {
    mt::ensure_core_init();
    require_cpu_iree();

    // Conv1d: input [N, C_in, L], weight [C_out, C_in, kL].
    Tensor w1_t = ::tenzor::randn({4, 2, 3}, DType::Float32, Device::cpu()) * 0.1f;
    Variable w1(w1_t, /*requires_grad=*/false);
    auto fn1 = ::tenzor::jit::CompiledFunction::FnType(
        [w1](const Variable& x) -> Variable {
            return nn::functional::conv1d(x, w1);
        });
    ::tenzor::jit::CompileConfig cfg1;
    cfg1.backend = "mlir";
    cfg1.target = "llvm-cpu";
    ::tenzor::jit::CompiledFunction compiled1(fn1, cfg1);
    Tensor x1_t = ::tenzor::randn({1, 2, 10}, DType::Float32, Device::cpu());
    Variable x1(x1_t, /*requires_grad=*/false);
    const Variable eager1 = nn::functional::conv1d(x1, w1);
    Variable out1;
    ASSERT_NO_THROW({ out1 = compiled1(x1); })
        << "Conv1d through the MLIR JIT path must not crash/throw uncaught "
           "(it may still gracefully eager-fallback -- see comment above)";
    EXPECT_LT(max_abs_diff_f64(eager1.tensor(), out1.tensor()), 1e-3)
        << "Conv1d result (whether compiled or eager-fallback) must match eager";

    // Conv3d: input [N, C_in, D, H, W], weight [C_out, C_in, kD, kH, kW].
    Tensor w3_t = ::tenzor::randn({2, 2, 2, 2, 2}, DType::Float32, Device::cpu()) * 0.1f;
    Variable w3(w3_t, /*requires_grad=*/false);
    auto fn3 = ::tenzor::jit::CompiledFunction::FnType(
        [w3](const Variable& x) -> Variable {
            return nn::functional::conv3d(x, w3);
        });
    ::tenzor::jit::CompileConfig cfg3;
    cfg3.backend = "mlir";
    cfg3.target = "llvm-cpu";
    ::tenzor::jit::CompiledFunction compiled3(fn3, cfg3);
    Tensor x3_t = ::tenzor::randn({1, 2, 4, 4, 4}, DType::Float32, Device::cpu());
    Variable x3(x3_t, /*requires_grad=*/false);
    const Variable eager3 = nn::functional::conv3d(x3, w3);
    Variable out3;
    ASSERT_NO_THROW({ out3 = compiled3(x3); })
        << "Conv3d through the MLIR JIT path must not crash/throw uncaught "
           "(it may still gracefully eager-fallback -- see comment above)";
    EXPECT_LT(max_abs_diff_f64(eager3.tensor(), out3.tensor()), 1e-3)
        << "Conv3d result (whether compiled or eager-fallback) must match eager";
}

// JIT-R011: stablehlo.power is exp(y*log(x)) on GPU/SPIR-V HAL targets,
// returning NaN for ANY negative base -- diverging from eager/libm pow,
// which gives the correctly-signed real result for a negative base raised
// to an integer power. This was already worked around for a compile-time-
// constant scalar exponent; the general 2-tensor-operand path (a
// dynamically-computed exponent TENSOR, not a scalar attribute) had no such
// guard. handle_pow's general-case branch (lowering.cpp) now applies the
// same negative-base-safe decomposition there.
//
// This is a LOWERING-LEVEL unit test, not an end-to-end API test: no public
// Tensor/Variable API traces to a genuine 2-input OpType::Pow node.
// autograd::pow(Variable, Variable) is fully DECOMPOSED (eq/where/lt/select
// on top of other ops), so tracing it never produces one. The only way to
// reach handle_pow's general branch directly is a raw dispatch(OpId::Pow,
// ...) call with 2 tensor inputs, which is itself OUT OF CONTRACT --
// OpId::Pow's registered kernel on every backend takes a single tensor input
// plus a scalar AttrKey::Exponent attribute and silently ignores a second
// tensor operand (defaulting to exponent=2.0). Eager fallback for this
// synthetic construction is therefore NOT a valid ground truth; only a
// genuine IREE-compiled run exercises the real (negative-base-safe) code
// path, so the per-target loop below requires an actual compile to have
// happened for the numeric assertions to apply. See
// PowNegativeBaseFloatPowerEndToEnd below for the real, publicly-reachable
// equivalent using tenzor::float_power.
TEST(MlirNumericParity, PowNegativeBaseRuntimeTensorExponent) {
    mt::ensure_core_init();
    require_cpu_iree();

    Tensor y_t({6}, DType::Float32, Device::cpu());
    const float yvals[6] = {2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 0.5f};  // last: fractional
    for (int i = 0; i < 6; ++i) y_t.data<float>()[i] = yvals[i];

    auto fn = ::tenzor::jit::CompiledFunction::FnType(
        [y_t](const Variable& x) -> Variable {
            std::vector<Tensor> inputs = {x.tensor(), y_t};
            auto result = ::tenzor::dispatch(::tenzor::OpId::Pow, inputs,
                                             ::tenzor::OpAttributes{});
            return Variable(result[0], false);
        });

    Tensor x_t({6}, DType::Float32, Device::cpu());
    const float xvals[6] = {-2.0f, -2.0f, -2.0f, -2.0f, 3.0f, -4.0f};
    for (int i = 0; i < 6; ++i) x_t.data<float>()[i] = xvals[i];
    Variable x(x_t, /*requires_grad=*/false);

    // Ground truth computed on the HOST via std::pow (libm), matching
    // real-analysis semantics for a negative base to an integer/fractional
    // power. NOT the eager dispatch(OpId::Pow, ...) result: that kernel's
    // only public tensor-level contract is a SCALAR exponent (see
    // ops/math.hpp's `Tensor pow(Tensor, double)` -- the sole tensor-level
    // pow this codebase exposes publicly); calling dispatch() directly with
    // a multi-element exponent tensor is outside that contract and the
    // kernel silently broadcasts from just y[0], so it cannot serve as
    // ground truth for a genuinely elementwise exponent here.
    float expected[6];
    for (int i = 0; i < 6; ++i) expected[i] = std::pow(xvals[i], yvals[i]);
    EXPECT_NEAR(expected[0], 4.0f, 1e-3f);    // (-2)^2 = 4
    EXPECT_NEAR(expected[1], -8.0f, 1e-3f);   // (-2)^3 = -8
    EXPECT_NEAR(expected[2], 16.0f, 1e-3f);   // (-2)^4 = 16
    EXPECT_NEAR(expected[3], -32.0f, 1e-2f);  // (-2)^5 = -32
    EXPECT_NEAR(expected[4], 9.0f, 1e-3f);    // 3^2 = 9
    ASSERT_TRUE(std::isnan(expected[5]))      // (-4)^0.5 = NaN, matching libm
        << "std::pow(-4, 0.5) should be NaN (real-analysis domain) -- test "
           "setup sanity check";

    for (const auto& target : mt::available_iree_targets()) {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend = "mlir";
        cfg.target  = target;
        ::tenzor::jit::CompiledFunction compiled(fn, cfg);
        mj::reset_cache_stats();
        Variable out;
        ASSERT_NO_THROW({ out = compiled(x); })
            << "PowNegativeBaseRuntimeTensorExponent threw on target=" << target;
        ASSERT_GE(mj::cache_stats().misses, 1u)
            << "PowNegativeBaseRuntimeTensorExponent did not attempt an IREE "
               "compile on target=" << target;

        // JIT-R029/R101: this machine's GPU is sm_120 (Blackwell); the
        // installed IREE 3.11.0rc / LLVM 23.0.0git NVPTX backend compiles
        // sm_80/sm_86 successfully but rejects sm_89+ (including sm_120)
        // with "missing GPU target in #hal.executable.target" -- a
        // toolchain/hardware-generation ceiling, not a Tenzor bug.
        // iree_compile.cpp's compile_via_subprocess (JIT-R101) now retries
        // once with the sm_80 Ampere baseline on exactly this failure
        // signature, which PTX-JITs forward-compatibly onto Blackwell, so
        // the compile now genuinely SUCCEEDS here instead of degrading to
        // eager -- the tight numeric comparison below is a real IREE-
        // compiled result and applies to cuda the same as every other
        // target (previously skipped; see the now-superseded JIT-R029
        // finding in findings.txt for the pre-R101 history).

        const Tensor got = out.tensor().to(DType::Float32).to(Device::cpu());
        const float* g = got.data<float>();
        for (int i = 0; i < 5; ++i) {
            EXPECT_NEAR(g[i], expected[i], 1e-2f)
                << "target=" << target << " index=" << i
                << " diverges from the correctly-signed real value (negative "
                   "base, integer power)";
        }
        EXPECT_TRUE(std::isnan(g[5]))
            << "target=" << target
            << " index=5 (negative base, fractional exponent) should be NaN";
    }
}

// JIT-R011 follow-up (discovered while writing the test above): the only
// publicly-reachable API that produces a genuine tensor-tensor pow --
// tenzor::float_power(Tensor, Tensor), used by the ONNX importer for a
// non-constant Pow exponent -- unconditionally graph-broke the ENTIRE
// compiled graph when traced, discarding JIT compilation for the whole
// function. Root cause: float_power's negative-base decomposition internally
// calls round() and fmod() (`eq(round(exp), exp)`, `fmod(exp, 2)`), and
// OpId::Round / OpId::Fmod had no OpId->OpType mapping in the tracer, so the
// first traced call graph-broke regardless of how small a role it played.
// Fixed by adding OpType::Round / OpType::Fmod (tracing_interceptor.cpp,
// graph.cpp shape inference + execute_node, lowering.cpp via the existing
// generic handle_unary("round_nearest_even") / handle_binary("remainder")).
// This test exercises the REAL public API end to end, across every
// available IREE target, and is a meaningful regression guard for that fix
// (unlike the synthetic dispatch(OpId::Pow, ...) test above).
TEST(MlirNumericParity, PowNegativeBaseFloatPowerEndToEnd) {
    mt::ensure_core_init();
    require_cpu_iree();

    Tensor y_t({6}, DType::Float32, Device::cpu());
    const float yvals[6] = {2.0f, 3.0f, 4.0f, 5.0f, 2.0f, 0.5f};  // last: fractional
    for (int i = 0; i < 6; ++i) y_t.data<float>()[i] = yvals[i];

    auto fn = ::tenzor::jit::CompiledFunction::FnType(
        [y_t](const Variable& x) -> Variable {
            return Variable(::tenzor::float_power(x.tensor(), y_t), false);
        });

    Tensor x_t({6}, DType::Float32, Device::cpu());
    const float xvals[6] = {-2.0f, -2.0f, -2.0f, -2.0f, 3.0f, -4.0f};
    for (int i = 0; i < 6; ++i) x_t.data<float>()[i] = xvals[i];
    Variable x(x_t, /*requires_grad=*/false);

    // tenzor::float_power is itself the reference implementation (matches
    // torch.float_power semantics), so the eager result IS the ground truth
    // here -- unlike the raw dispatch(OpId::Pow, ...) test above, this needs
    // no host-computed std::pow() substitute.
    const Variable eager = Variable(::tenzor::float_power(x_t, y_t), false);
    {
        const Tensor e = eager.tensor().to(DType::Float32).to(Device::cpu());
        const float* eg = e.data<float>();
        EXPECT_NEAR(eg[0], 4.0f, 1e-3f);
        EXPECT_NEAR(eg[1], -8.0f, 1e-3f);
        EXPECT_NEAR(eg[2], 16.0f, 1e-3f);
        EXPECT_NEAR(eg[3], -32.0f, 1e-2f);
        EXPECT_NEAR(eg[4], 9.0f, 1e-3f);
        EXPECT_TRUE(std::isnan(eg[5]));
    }

    for (const auto& target : mt::available_iree_targets()) {
        ::tenzor::jit::CompileConfig cfg;
        cfg.backend = "mlir";
        cfg.target  = target;
        ::tenzor::jit::CompiledFunction compiled(fn, cfg);
        mj::reset_cache_stats();
        Variable out;
        ASSERT_NO_THROW({ out = compiled(x); })
            << "PowNegativeBaseFloatPowerEndToEnd threw on target=" << target;
        ASSERT_GE(mj::cache_stats().misses, 1u)
            << "PowNegativeBaseFloatPowerEndToEnd did not attempt an IREE "
               "compile on target=" << target
            << " (float_power graph-broke again?)";
        // No cuda special-case needed here: whether this specific machine's
        // sm_120 GPU can genuinely compile via IREE or degrades to eager,
        // tenzor::float_power's eager kernel is itself correct, so the
        // comparison holds either way.
        //
        // Element 5 (negative base, fractional exponent) is NaN on both
        // sides by design -- max_abs_diff_f64 subtracts, and NaN - NaN is
        // NaN, which would poison a single whole-vector comparison. Check it
        // separately via isnan and compare only the finite elements 0-4.
        const Tensor got_f32 = out.tensor().to(DType::Float32).to(Device::cpu());
        EXPECT_TRUE(std::isnan(got_f32.data<float>()[5]))
            << "target=" << target
            << " index=5 (negative base, fractional exponent) should be NaN";
        auto slice_finite = [](const Tensor& t) {
            return ::tenzor::slice(t, 0, 0, 5);
        };
        EXPECT_LT(max_abs_diff_f64(slice_finite(eager.tensor()),
                                   slice_finite(out.tensor())),
                  1e-2)
            << "target=" << target << " diverges from eager float_power";
    }
}

// R1-01/JIT-R109 regression: the MLIR backend's trace_single_input_graph
// never declared closure-captured parameters/buffers to the tracer, so
// mlir_invoke_impl baked them as opaque constants inside the compiled
// module instead of treating them as live leaves -- a BatchNorm2d's
// running_mean_/running_var_ (or any with_buffers()/with_parameters()
// capture) went stale after the first compile, unlike backend="nvrtc"
// (already covered by JitTraceOps.BatchNorm2dRunningStatsLiveViaWithBuffers,
// the nvrtc-path sibling of this test). No prior test exercised
// backend="mlir" + with_buffers() + a no-grad inference call.
TEST(MlirNumericParity, BatchNorm2dRunningStatsLiveViaWithBuffers_MlirBackend) {
    mt::ensure_core_init();
    require_cpu_iree();

    nn::BatchNorm2d bn(4);
    bn.eval();
    NoGradGuard no_grad;
    auto fn = ::tenzor::jit::CompiledFunction::FnType(
        [&bn](const Variable& x) { return bn.forward(x); });

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target = "llvm-cpu";
    ::tenzor::jit::CompiledFunction compiled(fn, cfg);
    compiled.set_buffers(bn.buffers());

    auto x = randn({2, 4, 3, 3}, DType::Float32, Device::cpu()) + 1.0f;
    auto before = compiled(Variable(x, /*requires_grad=*/false)).tensor();

    EXPECT_GE(compiled.num_cached(), 1u)
        << "nn::BatchNorm2d(with_buffers, backend=mlir) did not compile/"
           "cache a module -- the op likely graph-broke and silently fell "
           "back to eager, so replay correctness was never exercised";

    // Simulate an eager EMA update happening OUTSIDE the compiled module
    // (e.g. a training step), the same way BatchNorm2dRunningStatsLive-
    // ViaWithBuffers (the nvrtc-path sibling test) does.
    auto rm = bn.get_buffer("running_mean");
    ASSERT_TRUE(rm != nullptr) << "nn::BatchNorm2d has no 'running_mean' buffer";
    rm->tensor() = rm->tensor() + 5.0;

    auto after = compiled(Variable(x, /*requires_grad=*/false)).tensor();
    auto eager_after = fn(Variable(x, false)).tensor();

    EXPECT_GT(max_abs_diff_f64(before, after), 1e-3)
        << "nn::BatchNorm2d MLIR-backend replay did not react to a "
           "running_mean_ mutation -- running stats are still frozen as "
           "trace-time constants despite with_buffers()";
    EXPECT_LT(max_abs_diff_f64(eager_after, after), 1e-4)
        << "nn::BatchNorm2d MLIR-backend replay after mutation != eager "
           "after mutation";
}

// Same fix, exercised via with_parameters() instead of with_buffers(): a
// captured Linear weight must reflect its CURRENT value on every
// backend="mlir" call, not the value at first trace.
TEST(MlirNumericParity, LinearWeightLiveViaWithParameters_MlirBackend) {
    mt::ensure_core_init();
    require_cpu_iree();

    auto weight = std::make_shared<Variable>(
        ::tenzor::full({3, 3}, 1.0f, DType::Float32, Device::cpu()), false);
    NoGradGuard no_grad;
    auto fn = ::tenzor::jit::CompiledFunction::FnType(
        [weight](const Variable& x) -> Variable {
            return matmul(x, *weight);
        });

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target = "llvm-cpu";
    ::tenzor::jit::CompiledFunction compiled(fn, cfg);
    compiled.set_parameters({weight});

    auto x = ::tenzor::full({3, 3}, 2.0f, DType::Float32, Device::cpu());
    auto before = compiled(Variable(x, /*requires_grad=*/false)).tensor();

    EXPECT_GE(compiled.num_cached(), 1u)
        << "Linear-weight closure (with_parameters, backend=mlir) did not "
           "compile/cache a module";

    // Mutate the weight in place (e.g. an optimizer step) OUTSIDE the
    // compiled module.
    weight->tensor() = weight->tensor() + 10.0f;

    auto after = compiled(Variable(x, /*requires_grad=*/false)).tensor();
    auto eager_after = fn(Variable(x, false)).tensor();

    EXPECT_GT(max_abs_diff_f64(before, after), 1e-3)
        << "Linear weight MLIR-backend replay did not react to a weight "
           "mutation -- still frozen as a trace-time constant despite "
           "with_parameters()";
    EXPECT_LT(max_abs_diff_f64(eager_after, after), 1e-4)
        << "Linear weight MLIR-backend replay after mutation != eager "
           "after mutation";
}

// R1-04/JIT-R113 regression: handle_layer_norm/handle_rms_norm_expand's
// mean/variance (resp. sum-of-squares) accumulation dtype `ad` used to widen
// F16/BF16 inputs to Float32 for accumulation, citing a comment that eager
// does the same -- stale, since eager's layer_norm_scalar<T>/RMSNorm kernels
// now accumulate in double UNCONDITIONALLY for every T (the Float32-
// accumulator-inside-template<T> bug class fixed elsewhere in this project).
// The expected numeric divergence from this gap is small (~1e-5 relative,
// scaling with sqrt(norm_size)) and easy for a black-box numeric-tolerance
// test to miss by coincidence, so verify directly and deterministically at
// the MLIR TEXT level instead: an F16 LayerNorm/RMSNorm's dumped StableHLO
// must contain an f64 accumulation step, not stay entirely in f16/f32.
TEST(MlirNumericParity, Float16LayerNormAccumulatesInF64) {
    mt::ensure_core_init();
    require_cpu_iree();

    const int64_t D = 64;
    auto fn = ::tenzor::jit::CompiledFunction::FnType(
        [D](const Variable& x) -> Variable {
            return nn::functional::layer_norm(x, {D});
        });
    Tensor x_t = ::tenzor::randn({2, D}, DType::Float32, Device::cpu())
                     .to(DType::Float16);
    Variable x(x_t, /*requires_grad=*/false);

    ::tenzor::jit::CompileConfig cfg;
    cfg.backend = "mlir";
    cfg.target = "llvm-cpu";
    ::tenzor::jit::CompiledFunction compiled(fn, cfg);
    const std::string mlir_text = compiled.dump_stablehlo(x);

    EXPECT_NE(mlir_text.find("f64"), std::string::npos)
        << "F16 LayerNorm's mean/variance accumulation must widen to f64 "
           "(matching eager's unconditional double accumulation), not stay "
           "in f16/f32 throughout:\n"
        << mlir_text;
}

}  // namespace

int main(int argc, char** argv) {
    ::tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    ::tenzor::finalize();
    return result;
}
