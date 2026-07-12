/**
 * @file extended_codegen.cpp
 * @brief Implementation of extended GPU kernel generation
 */

#include "tenzor/jit/extended_codegen.hpp"
#include "tenzor/jit/codegen.hpp"
#include "tenzor/ops/math.hpp"   // tenzor::matmul for the GEMM-epilogue fusion
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <limits>
#include <string>
#include <stdexcept>
#include <atomic>

namespace tenzor {
namespace jit {

namespace {
// Monotonic count of native fused GPU kernel launches (see
// extended_fused_launch_count). Atomic so a multi-threaded executor still yields
// a coherent nonzero count.
std::atomic<uint64_t> g_extended_fused_launches{0};
}  // namespace

auto extended_fused_launch_count() -> uint64_t {
    return g_extended_fused_launches.load(std::memory_order_relaxed);
}

// ============================================================================
// Signature computation
// ============================================================================

auto ExtendedFusionGroup::compute_signature() -> std::string {
    std::ostringstream ss;
    ss << "xfuse_" << static_cast<int>(kind) << "_" << static_cast<int>(dtype);

    switch (kind) {
        case FusionKind::Reduction:
            ss << "_dim" << reduce_dim << "_rk" << static_cast<int>(reduce_kind)
               << "_pre" << pre_ops.size()
               << "_post" << post_ops.size();
            // Include each pre/post op's scalar (by exact bits) — the kernel
            // source bakes the scalar in, so groups differing only in a
            // MulScalar/AddScalar constant must not share a cache signature.
            for (auto& op : pre_ops) {
                ss << "_" << static_cast<int>(op.op);
                uint64_t bits; std::memcpy(&bits, &op.scalar, sizeof(bits));
                ss << "s" << bits;
            }
            for (auto& op : post_ops) {
                ss << "_" << static_cast<int>(op.op);
                uint64_t bits; std::memcpy(&bits, &op.scalar, sizeof(bits));
                ss << "s" << bits;
            }
            break;
        case FusionKind::GemmEpilogue:
            ss << "_bias" << has_bias << "_hasact" << has_activation
               << "_act" << static_cast<int>(activation_type);
            break;
        case FusionKind::Softmax:
            ss << "_dim" << softmax_dim;
            break;
        case FusionKind::LayerNorm:
            ss << "_axis" << norm_axis << "_aff" << has_affine;
            break;
        case FusionKind::RMSNorm:
            ss << "_axis" << norm_axis << "_aff" << has_affine;
            break;
        case FusionKind::SmallMLP:
            ss << "_h" << hidden_dim << "_act" << static_cast<int>(mlp_activation);
            break;
        default:
            break;
    }

    signature = ss.str();
    return signature;
}

// ============================================================================
// Helpers
// ============================================================================

auto ExtendedKernelCodegen::dtype_to_cuda_type(DType dtype) -> std::string {
    switch (dtype) {
        case DType::Float32:  return "float";
        case DType::Float64:  return "double";
        case DType::Float16:  return "__half";  // same spelling under NVRTC/HIPRTC
        case DType::BFloat16:
            // CUDA and HIP spell bfloat16 differently (__nv_bfloat16 vs
            // __hip_bfloat16). This host TU is compiled for a single backend, but
            // the SAME generated source may be handed to EITHER NVRTC or HIPRTC at
            // runtime (a combined CUDA+ROCm build routes by the tensor's device),
            // so the type name must NOT be fixed by a host-side #if. Emit the
            // neutral "tz_bf16" typedef instead; generate() defines it to the
            // correct target type off the device compiler (__CUDACC_RTC__).
            return "tz_bf16";
        default:
            // Never silently fall back to "float": for Int*/Complex the element
            // size differs, so reinterpreting the buffer as float yields garbage
            // / out-of-bounds reads. Fail loudly instead.
            throw std::runtime_error(
                "Extended codegen: unsupported element dtype for fused GPU "
                "kernel: " + std::string(dtype_name(dtype)));
    }
}

auto ExtendedKernelCodegen::compute_type(DType dtype) -> std::string {
    // Compute/accumulate in double only for Float64; Float32 and the 16-bit
    // types compute in float (the half types are promoted to float for math).
    return (dtype == DType::Float64) ? "double" : "float";
}

auto ExtendedKernelCodegen::literal_suffix(DType dtype) -> std::string {
    return (dtype == DType::Float64) ? "" : "f";
}

auto ExtendedKernelCodegen::fmt_scalar(double v, DType dtype) -> std::string {
    if (!std::isfinite(v)) {
        // Non-finite: emit a valid device constant instead of the invalid
        // "inff"/"nanf" token that "<printed>" + suffix would produce (JIT-002).
        // __int_as_float / __longlong_as_double work in both CUDA and HIP.
        const bool f64 = (dtype == DType::Float64);
        if (std::isnan(v)) {
            return f64 ? "__longlong_as_double(0x7ff8000000000000LL)"
                       : "__int_as_float(0x7fc00000)";
        }
        const std::string inf = f64
            ? "__longlong_as_double(0x7ff0000000000000LL)"
            : "__int_as_float(0x7f800000)";
        return v < 0.0 ? ("(-" + inf + ")") : inf;
    }
    std::ostringstream o;
    o << std::setprecision(std::numeric_limits<double>::max_digits10) << v;
    std::string body = o.str();
    // Ensure a valid floating-literal body: a whole number prints as "2", and
    // appending the 'f' suffix would yield the invalid literal "2f". Give it a
    // decimal point unless it already has one, an exponent, or is inf/nan.
    if (body.find_first_of(".eEni") == std::string::npos) {
        body += ".0";
    }
    return body + literal_suffix(dtype);
}

auto ExtendedKernelCodegen::fn_for(const std::string& base, DType dtype) -> std::string {
    // double overloads are unsuffixed (exp, sqrt, rsqrt, fabs); the float
    // variants take the 'f' suffix (expf, sqrtf, rsqrtf, fabsf).
    return (dtype == DType::Float64) ? base : base + "f";
}

auto ExtendedKernelCodegen::activation_expr(OpType act, const std::string& var,
                                            DType dtype) -> std::string {
    const std::string F = literal_suffix(dtype);
    const std::string tanh_fn = fn_for("tanh", dtype);
    const std::string exp_fn = fn_for("exp", dtype);
    switch (act) {
        // clamp_min(x,0) == `x < 0 ? 0 : x`, which propagates NaN. The prior
        // `x > 0 ? x : 0` returned 0 for NaN and diverged from the eager/CPU
        // ReLU and the elementwise codegen path (codegen.cpp emit_op Relu).
        case OpType::ReLU:    return var + " < 0.0" + F + " ? 0.0" + F + " : " + var;
        case OpType::Sigmoid:
            return "1.0" + F + " / (1.0" + F + " + " + exp_fn + "(-" + var + "))";
        case OpType::Tanh:    return tanh_fn + "(" + var + ")";
        case OpType::GELU:
            // Exact erf GELU 0.5*x*(1 + erf(x / sqrt(2))) to match the eager/CPU
            // kernel and codegen.cpp; the tanh approximation diverged from the
            // exact-erf eager op. erf has float+double device overloads.
            return "0.5" + F + " * " + var + " * (1.0" + F + " + " +
                   fn_for("erf", dtype) + "(" + var +
                   " * 0.7071067811865476" + F + "))";
        default:
            // Never silently return identity for an unhandled activation — that
            // is the same silent-divergence class as the elementwise emit_op
            // default. is_activation() only admits ReLU/Sigmoid/Tanh/GELU, so
            // reaching here means a new fuseable activation was added without a
            // codegen arm. Fail loudly.
            throw std::runtime_error(
                "ExtendedKernelCodegen::activation_expr: unhandled activation "
                "OpType (" + std::to_string(static_cast<int>(act)) + ")");
    }
}

// ============================================================================
// Dispatch
// ============================================================================

auto ExtendedKernelCodegen::generate(const ExtendedFusionGroup& group) -> std::string {
    std::string body;
    switch (group.kind) {
        case FusionKind::Reduction:    body = generate_reduction(group); break;
        case FusionKind::GemmEpilogue: body = generate_gemm_epilogue(group); break;
        case FusionKind::Softmax:      body = generate_softmax(group); break;
        case FusionKind::LayerNorm:    body = generate_layer_norm(group); break;
        case FusionKind::RMSNorm:      body = generate_rms_norm(group); break;
        case FusionKind::SmallMLP:     body = generate_small_mlp(group); break;
        default:                       return "";
    }
    if (body.empty()) return "";
    // The preamble is emitted ahead of the kernel body and, crucially, is
    // resolved by the DEVICE compiler (NVRTC or HIPRTC) at runtime — NOT by this
    // host TU's preprocessor. A combined CUDA+ROCm build compiles this file once
    // (with TENZOR_USE_CUDA) yet must feed correct source to whichever RTC the
    // tensor's device selects, so every target-dependent choice is made with the
    // device compiler's own macro (__CUDACC_RTC__ is defined ONLY by NVRTC).
    std::ostringstream pre;
    // NVRTC/HIPRTC expose no system headers by default; the kernel signatures use
    // int64_t (the element-wise codegen uses `long long`). Provide the typedefs.
    pre << "typedef long long int64_t;\n"
           "typedef unsigned long long uint64_t;\n";
    // Warp-shuffle full-lane mask (BUG H4). __shfl_*_sync's mask must name every
    // participating lane. NVRTC targets a 32-lane warp, so 0xffffffff is the full
    // warp. HIPRTC targets AMD wavefronts that are 64 lanes wide on this hardware:
    // the reduction loop folds lanes 32..63 into 0..31 (offset starts at
    // warpSize/2 == 32), but a 32-bit 0xffffffff mask EXCLUDES lanes 32..63, so
    // those shuffles are undefined and the sum/max/mean/variance is wrong for any
    // reduced extent needing more than 32 lanes. Emit a 64-bit all-ones mask off
    // the device compiler so ROCm gets full-wavefront coverage.
    pre << "#if defined(__CUDACC_RTC__)\n"
           "#define TZ_WARP_MASK 0xffffffffu\n"
           "#else\n"
           "#define TZ_WARP_MASK 0xffffffffffffffffULL\n"
           "#endif\n";
    // Float16/BFloat16 kernels reference __half / tz_bf16, which are not built-in;
    // include the correct device headers and define tz_bf16 to the target's
    // bfloat16 type. Header names AND the bf16 type name differ between CUDA and
    // HIP, so both are selected off the device compiler (see dtype_to_cuda_type).
    if (body.find("__half") != std::string::npos) {
        pre << "#if defined(__CUDACC_RTC__)\n"
               "#include <cuda_fp16.h>\n"
               "#else\n"
               "#include <hip/hip_fp16.h>\n"
               "#endif\n";
    }
    if (body.find("tz_bf16") != std::string::npos) {
        pre << "#if defined(__CUDACC_RTC__)\n"
               "#include <cuda_bf16.h>\n"
               "typedef __nv_bfloat16 tz_bf16;\n"
               "#else\n"
               "#include <hip/hip_bf16.h>\n"
               "typedef __hip_bfloat16 tz_bf16;\n"
               "#endif\n";
    }
    return pre.str() + body;
}

// ============================================================================
// Reduction kernel: block-parallel with warp shuffles
// ============================================================================

auto ExtendedKernelCodegen::generate_reduction(const ExtendedFusionGroup& group) -> std::string {
    auto T = dtype_to_cuda_type(group.dtype);
    auto C = compute_type(group.dtype);          // accumulate in float/double
    const std::string F = literal_suffix(group.dtype);
    const std::string abs_fn = fn_for("fabs", group.dtype);
    const std::string exp_fn = fn_for("exp", group.dtype);
    const std::string log_fn = fn_for("log", group.dtype);
    const std::string sqrt_fn = fn_for("sqrt", group.dtype);
    const std::string pow_fn = fn_for("pow", group.dtype);
    const std::string max_fn = fn_for("fmax", group.dtype);
    const std::string min_fn = fn_for("fmin", group.dtype);
    std::ostringstream ss;

    // Reduction kind drives the accumulator identity, the per-element combine,
    // the warp/block tree-reduction step, and the finalization. Sum/Mean fold
    // with addition (Mean additionally divides by reduce_size); Max/Min fold
    // with fmax/fmin from a -inf/+inf identity.
    const bool is_max = (group.reduce_kind == OpType::Max);
    const bool is_min = (group.reduce_kind == OpType::Min);
    const bool is_mean = (group.reduce_kind == OpType::Mean);
    // Use true ±infinity as the max/min identity, NOT a finite ±1e30 sentinel:
    // real Float32 values range to ±3.4e38, so a finite sentinel would win over
    // legitimate extreme inputs (e.g. reduce_max([-2e30,-3e30]) -> -1e30 instead
    // of -2e30). NVRTC/HIPRTC expose no <math.h> (so INFINITY is undefined), but
    // __int_as_float is a device builtin in both; the float ±inf bit pattern
    // converts exactly to double for a double compute type.
    const std::string ident =
        is_max ? std::string("__int_as_float(0xff800000)") :  // -inf
        is_min ? std::string("__int_as_float(0x7f800000)")  : //  +inf
                 std::string("0");
    // combine(acc, x) -> updated acc, for the element loop and the tree steps.
    auto combine = [&](const std::string& acc, const std::string& x) -> std::string {
        // Max/Min reductions must propagate NaN: eager returns NaN if ANY element
        // is NaN (parallel_simd_max/min_f* in reduction.cpp), whereas fmax/fmin
        // silently drop NaN. Bind x to a temp first — it may be a
        // __shfl_down_sync collective that must be evaluated exactly once — and
        // note `acc + _t` is NaN whenever either operand is NaN.
        if (is_max || is_min) {
            const std::string& f = is_max ? max_fn : min_fn;
            return "{ " + C + " _t = " + x + "; " + acc + " = (" + acc + " != " +
                   acc + " || _t != _t) ? (" + acc + " + _t) : " + f + "(" + acc +
                   ", _t); }";
        }
        return acc + " += " + x + ";";
    };

    // Lower one pre/post element-wise op that operates in place on `v` (either the
    // per-element "val" before reduction or the scalar "result" after). The fused
    // reduction kernel loads a SINGLE input stream, so any op needing a distinct
    // second operand tensor cannot be represented here. BUG M5: the old code hit
    // `default: break` for such ops and SILENTLY OMITTED them from the kernel
    // (producing a wrong result), and hardcoded ElemOp::Mul to `v*v` even for a
    // genuine binary Mul(x, y). This helper instead emits every representable op
    // and, on an unrepresentable one, sets ok=false so generate_reduction refuses
    // to emit the fused kernel (returns ""), which makes execute_extended_fused
    // throw std::runtime_error (extended_codegen.cpp) rather than silently
    // emitting a wrong kernel.
    //
    // R1-09: this `ok=false` path does NOT fall back to normal op dispatch --
    // there is no exception handling anywhere between here and
    // CompiledFunction::operator()'s cache-hit path (compile.cpp), so if ever
    // reached it is an uncaught, unhandled error, not a graceful eager
    // fallback. It should be unreachable in practice: PatternMatcher::
    // match_reduction_chain (pattern_matcher.cpp) now structurally rejects any
    // pre/post op this function can't lower via is_reduction_representable_elem
    // (the exact same unary+self-square-Mul allowlist as this switch), and
    // ExtendedFusionPass::run (compiler.cpp)'s own to_elem check independently
    // re-verifies the same thing before committing a fusion rewrite -- two
    // layers of defense in depth ahead of this function, not one.
    const std::string indent = "        ";
    auto emit_elem = [&](std::ostringstream& out, const std::string& v,
                         const ElemStep& op, bool& ok) {
        const std::string sc = fmt_scalar(op.scalar, group.dtype);
        switch (op.op) {
            case ElemOp::Neg:        out << indent << v << " = -" << v << ";\n"; break;
            case ElemOp::Abs:        out << indent << v << " = " << abs_fn << "(" << v << ");\n"; break;
            case ElemOp::Sign:       out << indent << v << " = (" << v << " > 0) - (" << v << " < 0);\n"; break;
            case ElemOp::Reciprocal: out << indent << v << " = 1.0" << F << " / " << v << ";\n"; break;
            case ElemOp::Exp:        out << indent << v << " = " << exp_fn << "(" << v << ");\n"; break;
            case ElemOp::Log:        out << indent << v << " = " << log_fn << "(" << v << ");\n"; break;
            case ElemOp::Sqrt:       out << indent << v << " = " << sqrt_fn << "(" << v << ");\n"; break;
            case ElemOp::Sin:        out << indent << v << " = " << fn_for("sin", group.dtype) << "(" << v << ");\n"; break;
            case ElemOp::Cos:        out << indent << v << " = " << fn_for("cos", group.dtype) << "(" << v << ");\n"; break;
            case ElemOp::Tan:        out << indent << v << " = " << fn_for("tan", group.dtype) << "(" << v << ");\n"; break;
            case ElemOp::Sinh:       out << indent << v << " = " << fn_for("sinh", group.dtype) << "(" << v << ");\n"; break;
            case ElemOp::Cosh:       out << indent << v << " = " << fn_for("cosh", group.dtype) << "(" << v << ");\n"; break;
            case ElemOp::Tanh:       out << indent << v << " = " << fn_for("tanh", group.dtype) << "(" << v << ");\n"; break;
            case ElemOp::Erf:        out << indent << v << " = " << fn_for("erf", group.dtype) << "(" << v << ");\n"; break;
            case ElemOp::Erfc:       out << indent << v << " = " << fn_for("erfc", group.dtype) << "(" << v << ");\n"; break;
            case ElemOp::Log2:       out << indent << v << " = " << fn_for("log2", group.dtype) << "(" << v << ");\n"; break;
            case ElemOp::Log10:      out << indent << v << " = " << fn_for("log10", group.dtype) << "(" << v << ");\n"; break;
            case ElemOp::Log1p:      out << indent << v << " = " << fn_for("log1p", group.dtype) << "(" << v << ");\n"; break;
            case ElemOp::Exp2:       out << indent << v << " = " << fn_for("exp2", group.dtype) << "(" << v << ");\n"; break;
            case ElemOp::Expm1:      out << indent << v << " = " << fn_for("expm1", group.dtype) << "(" << v << ");\n"; break;
            case ElemOp::Floor:      out << indent << v << " = " << fn_for("floor", group.dtype) << "(" << v << ");\n"; break;
            case ElemOp::Ceil:       out << indent << v << " = " << fn_for("ceil", group.dtype) << "(" << v << ");\n"; break;
            case ElemOp::Round:      out << indent << v << " = " << fn_for("round", group.dtype) << "(" << v << ");\n"; break;
            case ElemOp::Sigmoid:
                out << indent << v << " = 1.0" << F << " / (1.0" << F << " + " << exp_fn << "(-" << v << "));\n"; break;
            case ElemOp::Relu:
                // NaN-propagating select, matching codegen.cpp / clamp_min(x,0).
                out << indent << v << " = (" << v << " < 0) ? 0 : " << v << ";\n"; break;
            case ElemOp::Gelu:
                out << indent << v << " = 0.5" << F << " * " << v << " * (1.0" << F
                    << " + " << fn_for("erf", group.dtype) << "(" << v << " * 0.7071067811865476" << F << "));\n"; break;
            case ElemOp::MulScalar:  out << indent << v << " = " << v << " * " << sc << ";\n"; break;
            case ElemOp::AddScalar:  out << indent << v << " = " << v << " + " << sc << ";\n"; break;
            case ElemOp::PowScalar:  out << indent << v << " = " << pow_fn << "(" << v << ", " << sc << ");\n"; break;
            // clamp_min/clamp_max propagate NaN (see codegen.cpp emit_op); fmax/fmin drop it.
            case ElemOp::ClampMin:   out << indent << v << " = (" << v << " < " << sc << ") ? " << sc << " : " << v << ";\n"; break;
            case ElemOp::ClampMax:   out << indent << v << " = (" << v << " > " << sc << ") ? " << sc << " : " << v << ";\n"; break;
            case ElemOp::Mul:
                // Only a self-square (x*x, same operand) is representable with a
                // single input stream — this is RMSNorm's x². A binary Mul(x, y)
                // with a DISTINCT second operand has no second stream to read, so
                // refuse rather than silently square (the old bug).
                if (op.input_idx == op.second_input_idx) {
                    out << indent << v << " = " << v << " * " << v << ";\n";
                } else {
                    ok = false;
                }
                break;
            default:
                // Add/Sub/Div/Max/Min/Fmod/Pow(binary) and any op needing a second
                // operand stream cannot be faithfully lowered — refuse to fuse.
                ok = false;
                break;
        }
    };

    // For the 16-bit storage types (Float16/BFloat16, where T != C) eager runs
    // each pre-/post-reduction elementwise op as a full tensor op that NARROWS the
    // result back to the 16-bit storage dtype after EVERY step. Chaining the ops
    // in float (narrowing only on the final store) therefore diverges from
    // eager/CPU on any multi-step 16-bit pre/post block. Mirror eager's per-op
    // narrowing by round-tripping the accumulator through T after each op. For
    // Float32/Float64 (T == C) this is skipped — no behavior change, no overhead.
    const bool round_16 = (T != C);
    auto round_to_T = [&](std::ostringstream& out, const std::string& v) {
        if (round_16) {
            out << indent << v << " = static_cast<" << C << ">(static_cast<" << T
                << ">(" << v << "));\n";
        }
    };

    bool representable = true;
    std::ostringstream pre_block;
    for (const auto& op : group.pre_ops) {
        emit_elem(pre_block, "val", op, representable);
        if (!representable) return "";
        round_to_T(pre_block, "val");
    }
    std::ostringstream post_block;
    for (const auto& op : group.post_ops) {
        emit_elem(post_block, "result", op, representable);
        if (!representable) return "";
        round_to_T(post_block, "result");
    }

    ss << R"(
extern "C" __global__ void fused_reduction_kernel(
    const )" << T << R"(* __restrict__ input,
    )" << T << R"(* __restrict__ output,
    int64_t outer_size, int64_t reduce_size, int64_t inner_size) {

    // Grid: (outer_size * inner_size) blocks, 256 threads each. Decode the block
    // index in int64_t: outer_size*inner_size can exceed INT_MAX, and a 32-bit
    // idx would wrap and select the wrong (outer, inner) slice. The 2D term keeps
    // the decode correct even if the grid is launched as (gridDim.x, gridDim.y).
    int64_t idx = static_cast<int64_t>(blockIdx.x) + static_cast<int64_t>(gridDim.x) * blockIdx.y;
    int64_t outer = idx / inner_size;
    int64_t inner = idx % inner_size;
    if (outer >= outer_size) return;

    // Each block reduces one (outer, inner) slice. Accumulate in the compute
    // type (float for the 16-bit storage types) and narrow to T only on store.
    )" << C << R"( sum = )" << ident << R"(;
)";
    // Kahan-compensated accumulation for additive reductions (sum/mean): the CPU
    // reduction sums in compensated float, so a bare running sum on the GPU
    // diverges by more than a few ULP over a large reduce extent. Compensating
    // the per-thread sequential accumulation (the dominant error term) keeps
    // CPU/CUDA/ROCm agreement tight. Max/Min fold exactly, so they skip it.
    const bool use_kahan = !is_max && !is_min;
    if (use_kahan) ss << "    " << C << " kahan_c = 0;\n";
    ss << R"(    for (int64_t r = threadIdx.x; r < reduce_size; r += blockDim.x) {
        )" << C << R"( val = static_cast<)" << C << R"(>(input[outer * reduce_size * inner_size + r * inner_size + inner]);
)";

    // Pre-reduction element-wise ops (lowered + representability-checked above).
    ss << pre_block.str();

    if (use_kahan) {
        ss << "        " << C << " kahan_y = val - kahan_c;\n"
           << "        " << C << " kahan_t = sum + kahan_y;\n"
           << "        kahan_c = (kahan_t - sum) - kahan_y;\n"
           << "        sum = kahan_t;\n";
    } else {
        ss << "        " << combine("sum", "val") << "\n";
    }
    ss << R"(    }
)";
    // Fold the per-thread compensation back before the tree reduction combines
    // the partial sums across threads.
    if (use_kahan) ss << "    sum = sum + kahan_c;\n";
    // NOTE (JIT-F013 / cross-backend tolerance): this warp-shuffle tree folds over
    // `warpSize` lanes, which is 32 under NVRTC (CUDA) but 64 under HIPRTC (ROCm).
    // The lane grouping therefore differs between the two GPU backends, and the
    // cross-lane combine is a plain reduction (Kahan compensation is per-thread
    // only). As a result Sum/Mean/Softmax/Norm reductions are NOT bit-identical
    // between CUDA and ROCm (nor vs the CPU's sequential compensated sum) for
    // large reduce extents — the difference is a small multiple of ULP that grows
    // ~log(N). Cross-backend parity tests over these ops must use a tolerance
    // (not exact equality); do not tighten below the accumulated-rounding bound.
    ss << R"(

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
)";
    ss << "        " << combine("sum", "__shfl_down_sync(TZ_WARP_MASK, sum, offset)") << "\n";
    ss << R"(    }

    // Block-level reduction via shared memory
    __shared__ )" << C << R"( shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) shared[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < blockDim.x / warpSize) ? shared[lane] : )" << ident << R"(;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
)";
    ss << "            " << combine("sum", "__shfl_down_sync(TZ_WARP_MASK, sum, offset)") << "\n";
    ss << R"(        }
    }

    if (threadIdx.x == 0) {
        )" << C << R"( result = sum;
)";
    // Mean finalizes the summed accumulator by dividing by the element count.
    if (is_mean) {
        ss << "        result = result / static_cast<" << C << ">(reduce_size);\n";
    }

    // Post-reduction element-wise ops (lowered + representability-checked above).
    ss << post_block.str();

    ss << "        output[outer * inner_size + inner] = static_cast<" << T << ">(result);\n";
    ss << R"(    }
}
)";

    return ss.str();
}

// ============================================================================
// GEMM epilogue: bias + activation fused with matmul result
// ============================================================================

auto ExtendedKernelCodegen::generate_gemm_epilogue(const ExtendedFusionGroup& group) -> std::string {
    auto T = dtype_to_cuda_type(group.dtype);
    auto C = compute_type(group.dtype);  // float for f32/f16/bf16, double for f64
    std::ostringstream ss;

    ss << R"(
extern "C" __global__ void fused_gemm_epilogue_kernel(
    )" << T << R"(* __restrict__ output,)";

    if (group.has_bias) {
        ss << R"(
    const )" << T << R"(* __restrict__ bias,)";
    }

    ss << R"(
    int64_t rows, int64_t cols) {

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    int64_t col = idx % cols;
    // Promote to the compute type so bias add + activation run in float/double
    // (correct for the 16-bit storage types), then narrow back to T on store.
    )" << C << " val = static_cast<" << C << ">(output[idx]);\n";

    if (group.has_bias) {
        ss << "    val = val + static_cast<" << C << ">(bias[col]);\n";
        // Eager runs matmul -> narrow to T, bias-add -> narrow to T, then
        // activation -> narrow to T. For the 16-bit storage types (T != C) the
        // bias-add result must be rounded to T BEFORE the activation so the
        // activation sees the same 16-bit-rounded input eager does; otherwise a
        // f16/bf16 bias+activation epilogue diverges from eager/CPU. Float32/
        // Float64 (T == C) skip this — no behavior change.
        if (group.has_activation && T != C) {
            ss << "    val = static_cast<" << C << ">(static_cast<" << T << ">(val));\n";
        }
    }

    if (group.has_activation) {
        ss << "    val = " << activation_expr(group.activation_type, "val", group.dtype) << ";\n";
    }

    ss << "    output[idx] = static_cast<" << T << ">(val);\n";
    ss << R"(}
)";

    return ss.str();
}

// ============================================================================
// Softmax: online 2-pass fused per-row
// ============================================================================

auto ExtendedKernelCodegen::generate_softmax(const ExtendedFusionGroup& group) -> std::string {
    auto T = dtype_to_cuda_type(group.dtype);
    auto C = compute_type(group.dtype);          // reductions/exp in float/double
    const std::string F = literal_suffix(group.dtype);
    const std::string exp_fn = fn_for("exp", group.dtype);
    std::ostringstream ss;

    ss << R"(
extern "C" __global__ void fused_softmax_kernel(
    const )" << T << R"(* __restrict__ input,
    )" << T << R"(* __restrict__ output,
    int64_t rows, int64_t cols) {

    // One block per row
    int row = blockIdx.x;
    if (row >= rows) return;

    const )" << T << R"(* row_input = input + row * cols;
    )" << T << R"(* row_output = output + row * cols;

    // Pass 1: find max (for numerical stability). All reductions run in the
    // compute type (float for the 16-bit storage types); only loads/stores touch
    // T.
    )" << C << R"( thread_max = __int_as_float(0xff800000))" << R"(;
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        )" << C << R"( val = static_cast<)" << C << R"(>(row_input[c]);
        thread_max = val > thread_max ? val : thread_max;
    }

    // Warp reduction for max
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        )" << C << R"( other = __shfl_down_sync(TZ_WARP_MASK, thread_max, offset);
        thread_max = other > thread_max ? other : thread_max;
    }

    __shared__ )" << C << R"( shared_max[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) shared_max[warp_id] = thread_max;
    __syncthreads();

    if (warp_id == 0) {
        thread_max = (lane < blockDim.x / warpSize) ? shared_max[lane] : __int_as_float(0xff800000))" << R"(;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            )" << C << R"( other = __shfl_down_sync(TZ_WARP_MASK, thread_max, offset);
            thread_max = other > thread_max ? other : thread_max;
        }
        if (lane == 0) shared_max[0] = thread_max;
    }
    __syncthreads();
    )" << C << R"( row_max = shared_max[0];

    // Pass 2: compute exp(x - max) and sum. Do NOT stage the exp() numerators
    // through the (possibly fp16) output buffer: truncating exp(x-max) to T and
    // reading it back in pass 3 would normalize an already-rounded numerator and
    // diverge from the float reference. Keep the running sum in the compute type
    // and recompute exp() from the untouched input in pass 3.
    )" << C << R"( thread_sum = 0;
    )" << C << R"( sum_c = 0;  // Kahan compensation (match CPU compensated sum)
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        )" << C << R"( val = )" << exp_fn << R"((static_cast<)" << C << R"(>(row_input[c]) - row_max);
        )" << C << R"( ky = val - sum_c;
        )" << C << R"( kt = thread_sum + ky;
        sum_c = (kt - thread_sum) - ky;
        thread_sum = kt;
    }
    thread_sum = thread_sum + sum_c;

    // Warp reduction for sum
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(TZ_WARP_MASK, thread_sum, offset);
    }

    __shared__ )" << C << R"( shared_sum[32];
    if (lane == 0) shared_sum[warp_id] = thread_sum;
    __syncthreads();

    if (warp_id == 0) {
        thread_sum = (lane < blockDim.x / warpSize) ? shared_sum[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(TZ_WARP_MASK, thread_sum, offset);
        }
        if (lane == 0) shared_sum[0] = thread_sum;
    }
    __syncthreads();
    // Pass 3: recompute exp(x - max) in the compute type and normalize with a
    // DIRECT divide (one correctly-rounded division) rather than a
    // reciprocal-then-multiply (two roundings), to bit-match the eager softmax
    // (JIT-F052). The numerator stays in C until the single final narrowing store.
    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        )" << C << R"( val = )" << exp_fn << R"((static_cast<)" << C << R"(>(row_input[c]) - row_max);
        row_output[c] = static_cast<)" << T << R"(>(val / shared_sum[0]);
    }
}
)";

    return ss.str();
}

// ============================================================================
// LayerNorm: Welford online mean+variance, single-kernel
// ============================================================================

auto ExtendedKernelCodegen::generate_layer_norm(const ExtendedFusionGroup& group) -> std::string {
    auto T = dtype_to_cuda_type(group.dtype);
    auto C = compute_type(group.dtype);  // elementwise compute type
    // Run the Welford mean/variance accumulator in double to match the eager
    // kernel's F64 accumulator (JIT-064); computing it in float (compute_type
    // for F32) lost mantissa for long normalized dims. Element reads / the
    // normalize multiply stay in C. Divide by the (double) sqrt of the double
    // statistic, matching eager's 1/sqrt(var+eps) rather than the rsqrt approx.
    const std::string ACC = "double";
    std::ostringstream ss;

    ss << R"(
extern "C" __global__ void fused_layer_norm_kernel(
    const )" << T << R"(* __restrict__ input,
    )" << T << R"(* __restrict__ output,)";

    if (group.has_affine) {
        ss << R"(
    const )" << T << R"(* __restrict__ gamma,
    const )" << T << R"(* __restrict__ beta,)";
    }

    ss << R"(
    int64_t outer_size, int64_t norm_size, )" << C << R"( eps) {

    // One block per normalized instance
    int instance = blockIdx.x;
    if (instance >= outer_size) return;

    const )" << T << R"(* x = input + instance * norm_size;
    )" << T << R"(* y = output + instance * norm_size;

    // Welford online mean computation (accumulate in double to match the eager
    // F64 accumulator; the 16-bit/32-bit storage types are widened for the
    // running mean/variance).
    )" << ACC << R"( mean = 0;
    )" << ACC << R"( m2 = 0;
    int count = 0;

    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        count++;
        )" << ACC << R"( xi = static_cast<)" << ACC << R"(>(x[i]);
        )" << ACC << R"( delta = xi - mean;
        mean += delta / count;
        )" << ACC << R"( delta2 = xi - mean;
        m2 += delta * delta2;
    }

    // Warp-level Welford merge
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        )" << ACC << R"( other_mean = __shfl_down_sync(TZ_WARP_MASK, mean, offset);
        )" << ACC << R"( other_m2 = __shfl_down_sync(TZ_WARP_MASK, m2, offset);
        int other_count = __shfl_down_sync(TZ_WARP_MASK, count, offset);

        int total = count + other_count;
        if (total > 0) {
            )" << ACC << R"( delta = other_mean - mean;
            mean = (count * mean + other_count * other_mean) / total;
            m2 = m2 + other_m2 + delta * delta * count * other_count / total;
            count = total;
        }
    }

    // Block-level merge via shared memory
    __shared__ )" << ACC << R"( s_mean[32], s_m2[32];
    __shared__ int s_count[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) {
        s_mean[warp_id] = mean;
        s_m2[warp_id] = m2;
        s_count[warp_id] = count;
    }
    __syncthreads();

    if (warp_id == 0) {
        int nwarps = blockDim.x / warpSize;
        mean = (lane < nwarps) ? s_mean[lane] : 0;
        m2 = (lane < nwarps) ? s_m2[lane] : 0;
        count = (lane < nwarps) ? s_count[lane] : 0;

        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            )" << ACC << R"( other_mean = __shfl_down_sync(TZ_WARP_MASK, mean, offset);
            )" << ACC << R"( other_m2 = __shfl_down_sync(TZ_WARP_MASK, m2, offset);
            int other_count = __shfl_down_sync(TZ_WARP_MASK, count, offset);

            int total = count + other_count;
            if (total > 0) {
                )" << ACC << R"( delta = other_mean - mean;
                mean = (count * mean + other_count * other_mean) / total;
                m2 = m2 + other_m2 + delta * delta * count * other_count / total;
                count = total;
            }
        }

        if (lane == 0) {
            s_mean[0] = mean;
            s_m2[0] = m2 / count;  // variance
        }
    }
    __syncthreads();

    )" << ACC << R"( final_mean = s_mean[0];
    )" << ACC << R"( inv_std = ((double)1) / sqrt(s_m2[0] + eps);
    // JIT-054c: narrow mean/inv_std to the compute dtype BEFORE the centering
    // multiply, matching eager's SIMD LayerNorm (which narrows mean/inv_std to
    // float before the normalize loop) and the already-fixed MLIR lowering
    // path (handle_layer_norm narrows the same way). Using the double `mean`/
    // `inv_std` directly here double-rounds instead of rounding once at the
    // same point eager does -- a small (~1 ULP) but real divergence from both
    // eager and the MLIR-compiled form of the identical graph.
    )" << C << R"( final_mean_c = static_cast<)" << C << R"(>(final_mean);
    )" << C << R"( inv_std_c = static_cast<)" << C << R"(>(inv_std);

    // Normalize + affine
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        )" << C << R"( val = (static_cast<)" << C << R"(>(x[i]) - final_mean_c) * inv_std_c;)";

    if (group.has_affine) {
        ss << R"(
        val = val * static_cast<)" << C << R"(>(gamma[i]) + static_cast<)" << C << R"(>(beta[i]);)";
    }

    ss << R"(
        y[i] = static_cast<)" << T << R"(>(val);
    }
}
)";

    return ss.str();
}

// ============================================================================
// RMSNorm: square -> mean -> rsqrt -> mul
// ============================================================================

auto ExtendedKernelCodegen::generate_rms_norm(const ExtendedFusionGroup& group) -> std::string {
    auto T = dtype_to_cuda_type(group.dtype);
    auto C = compute_type(group.dtype);  // elementwise compute type
    // Accumulate the sum-of-squares in double to match the eager kernel, which
    // uses an F64 accumulator to avoid mantissa loss in the RMS denominator for
    // long hidden dims (JIT-065). Computing it in float (compute_type for F32)
    // diverged from eager. The element reads/normalize multiply stay in C.
    const std::string ACC = "double";
    // Eager (CPU fused_ops) computes the inverse std as 1.0 / sqrt(var + eps),
    // NOT via the rsqrt intrinsic (a fast approximation differing by 1-2 ULP);
    // we divide by the (double) sqrt of the double statistic to match eager.
    std::ostringstream ss;

    ss << R"(
extern "C" __global__ void fused_rms_norm_kernel(
    const )" << T << R"(* __restrict__ input,
    )" << T << R"(* __restrict__ output,)";

    if (group.has_affine) {
        ss << R"(
    const )" << T << R"(* __restrict__ gamma,)";
    }

    ss << R"(
    int64_t outer_size, int64_t norm_size, )" << C << R"( eps) {

    int instance = blockIdx.x;
    if (instance >= outer_size) return;

    const )" << T << R"(* x = input + instance * norm_size;
    )" << T << R"(* y = output + instance * norm_size;

    // Compute sum of squares (accumulate in compute type; 16-bit storage types
    // are promoted to float for the reduction).
    )" << ACC << R"( sum_sq = 0;
    )" << ACC << R"( sumsq_c = 0;  // Kahan compensation (match CPU compensated sum)
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        )" << ACC << R"( val = static_cast<)" << ACC << R"(>(x[i]);
        )" << ACC << R"( ky = val * val - sumsq_c;
        )" << ACC << R"( kt = sum_sq + ky;
        sumsq_c = (kt - sum_sq) - ky;
        sum_sq = kt;
    }
    sum_sq = sum_sq + sumsq_c;

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(TZ_WARP_MASK, sum_sq, offset);
    }

    __shared__ )" << ACC << R"( shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (lane == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane < blockDim.x / warpSize) ? shared[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(TZ_WARP_MASK, sum_sq, offset);
        }
        if (lane == 0) shared[0] = sum_sq;
    }
    __syncthreads();

    )" << ACC << R"( rms_inv = ((double)1) / sqrt(shared[0] / norm_size + eps);
    // JIT-054c: narrow rms_inv to the compute dtype BEFORE the multiply,
    // matching eager's fused_rms_norm_kernel (which narrows inv_rms to float
    // before multiplying) and the already-fixed MLIR lowering path
    // (handle_rms_norm_expand narrows the same way). Using the double
    // `rms_inv` directly here double-rounds instead of rounding once at the
    // same point eager does -- a small (~1 ULP) but real divergence from both
    // eager and the MLIR-compiled form of the identical graph.
    )" << C << R"( rms_inv_c = static_cast<)" << C << R"(>(rms_inv);

    // Normalize
    for (int64_t i = threadIdx.x; i < norm_size; i += blockDim.x) {
        )" << C << R"( val = static_cast<)" << C << R"(>(x[i]) * rms_inv_c;)";

    if (group.has_affine) {
        ss << R"(
        val = val * static_cast<)" << C << R"(>(gamma[i]);)";
    }

    ss << R"(
        y[i] = static_cast<)" << T << R"(>(val);
    }
}
)";

    return ss.str();
}

// ============================================================================
// Small MLP: shared-memory tiled GEMM with fused activation
// ============================================================================

auto ExtendedKernelCodegen::generate_small_mlp(const ExtendedFusionGroup& group) -> std::string {
    auto T = dtype_to_cuda_type(group.dtype);
    auto C = compute_type(group.dtype);  // GEMM accumulation in float/double
    std::ostringstream ss;

    // For small MLPs, we generate a two-stage kernel:
    // Stage 1: Y = activation(X @ W1 + b1)  (stored in shared memory)
    // Stage 2: Z = Y @ W2 + b2
    // This avoids writing the intermediate Y to global memory.

    ss << R"(
extern "C" __global__ void fused_small_mlp_kernel(
    const )" << T << R"(* __restrict__ input,     // [batch, in_dim]
    const )" << T << R"(* __restrict__ w1,         // [in_dim, hidden_dim]
    const )" << T << R"(* __restrict__ b1,         // [hidden_dim]
    const )" << T << R"(* __restrict__ w2,         // [hidden_dim, out_dim]
    const )" << T << R"(* __restrict__ b2,         // [out_dim]
    )" << T << R"(* __restrict__ output,           // [batch, out_dim]
    int64_t batch, int64_t in_dim, int64_t hidden_dim, int64_t out_dim) {

    // One block per batch element
    int b = blockIdx.x;
    if (b >= batch) return;

    extern __shared__ )" << T << R"( smem[];
    )" << T << R"(* hidden = smem;  // [hidden_dim]

    const )" << T << R"(* x = input + b * in_dim;

    // Stage 1: hidden = activation(x @ W1 + b1). Accumulate the dot product in
    // the compute type (float for the 16-bit storage types) and narrow to T when
    // writing the shared-memory intermediate (whose layout is sized in T).
    for (int64_t h = threadIdx.x; h < hidden_dim; h += blockDim.x) {
        )" << C << R"( acc = static_cast<)" << C << R"(>(b1[h]);
        for (int64_t i = 0; i < in_dim; ++i) {
            acc += static_cast<)" << C << R"(>(x[i]) * static_cast<)" << C << R"(>(w1[i * hidden_dim + h]);
        }
)";
    // Eager runs stage 1 as linear(x, W1, b1) -> narrow to T, then activation ->
    // narrow to T. Stage 1 previously fed the float `acc` (the linear result)
    // straight into the activation, skipping the narrow of the LINEAR result that
    // eager performs. For the 16-bit storage types (T != C) round acc to T before
    // the activation so it sees the same 16-bit-rounded input eager does; a f16/
    // bf16 SmallMLP otherwise diverges from eager/CPU. (Stage 2 already narrows
    // its linear result correctly.) Float32/Float64 (T == C) skip this.
    if (T != C) {
        ss << "        acc = static_cast<" << C << ">(static_cast<" << T << ">(acc));\n";
    }
    ss << R"(        hidden[h] = static_cast<)" << T << R"(>()" << activation_expr(group.mlp_activation, "acc", group.dtype) << R"();
    }
    __syncthreads();

    // Stage 2: output = hidden @ W2 + b2
    )" << T << R"(* out = output + b * out_dim;
    for (int64_t o = threadIdx.x; o < out_dim; o += blockDim.x) {
        )" << C << R"( acc = static_cast<)" << C << R"(>(b2[o]);
        for (int64_t h = 0; h < hidden_dim; ++h) {
            acc += static_cast<)" << C << R"(>(hidden[h]) * static_cast<)" << C << R"(>(w2[h * out_dim + o]);
        }
        out[o] = static_cast<)" << T << R"(>(acc);
    }
}
)";

    return ss.str();
}

// ============================================================================
// Extended kernel execution
// ============================================================================

namespace {

// Entry-point symbol name emitted by each generate_* method, keyed by kind.
// Must stay in sync with the extern "C" __global__ names in the generators.
auto extended_kernel_name(FusionKind kind) -> std::string {
    switch (kind) {
        case FusionKind::Reduction:    return "fused_reduction_kernel";
        case FusionKind::GemmEpilogue: return "fused_gemm_epilogue_kernel";
        case FusionKind::Softmax:      return "fused_softmax_kernel";
        case FusionKind::LayerNorm:    return "fused_layer_norm_kernel";
        case FusionKind::RMSNorm:      return "fused_rms_norm_kernel";
        case FusionKind::SmallMLP:     return "fused_small_mlp_kernel";
        // Not an extended-codegen kind; emitted by the basic element-wise
        // codegen path, which has no extended entry-point symbol.
        case FusionKind::ElementWise:
            return "";
    }
    return "";
}

} // namespace

auto execute_extended_fused(const ExtendedFusionGroup& group,
                            const std::vector<Tensor>& inputs,
                            const std::vector<Tensor>& params) -> Tensor {
    // Generate source code
    auto source = ExtendedKernelCodegen::generate(group);
    if (source.empty()) {
        throw std::runtime_error("Extended codegen: unsupported fusion kind");
    }

    // Compile the EXTENDED kernel source we just generated, keyed by the
    // extended group's signature. Previously this delegated to
    // get_or_compile(FusionGroup), which regenerates and compiles an unrelated
    // element-wise kernel and discards `source` entirely.
    auto& cache = KernelCache::instance();

    if (inputs.empty()) {
        throw std::runtime_error("Extended codegen: no inputs provided");
    }
    // The compiled kernel is bound to one device's context and compute arch, so
    // the cache key MUST encode the device (type + ordinal) and compilation must
    // target that device. Otherwise a kernel built for e.g. cuda:0 is silently
    // served to a cuda:1 input -> illegal cross-device access / corruption.
    // This codegen path emits NVRTC/HIPRTC device kernels, so only CUDA and ROCm
    // are valid launch targets; a Vulkan/OneAPI (or CPU) tensor must be rejected
    // rather than have a CUDA/HIP kernel launched over its foreign pointers.
    const Device dev = inputs[0].device();
    if (dev.type != Device::Type::CUDA && dev.type != Device::Type::ROCm) {
        throw std::runtime_error(
            "execute_extended_fused: fused GPU codegen supports only CUDA and "
            "ROCm devices; got " + dev.to_string());
    }
    // The extended kernels compute in float/double and load/store f16/bf16 via
    // implicit conversion; any other dtype (Int*/Complex) would be reinterpreted
    // as the wrong element type. Reject up front, before compile/transfer.
    if (group.dtype != DType::Float32 && group.dtype != DType::Float64 &&
        group.dtype != DType::Float16 && group.dtype != DType::BFloat16) {
        throw std::runtime_error(
            "execute_extended_fused: fused GPU codegen only supports "
            "Float32/Float64/Float16/BFloat16; got " +
            std::string(dtype_name(group.dtype)));
    }

    std::string base_sig = group.signature.empty()
        ? const_cast<ExtendedFusionGroup&>(group).compute_signature()
        : group.signature;
    std::string signature = base_sig + "_dev" + dev.to_string();
    std::string kernel_name = extended_kernel_name(group.kind);
    if (kernel_name.empty()) {
        throw std::runtime_error("Extended codegen: unsupported fusion kind");
    }

    auto kernel = cache.get_or_compile_source(signature, source, kernel_name,
                                              dev.index,
                                              dev.type == Device::Type::ROCm);
    if (!kernel) {
        throw std::runtime_error("Extended codegen: failed to compile kernel: " + signature);
    }

    // ---- Per-kind output shape, launch geometry, and argument layout ----
    // The extended kernels do NOT share the element-wise [inputs..., output,
    // numel] ABI: each has a distinct signature (GEMM-epilogue puts output
    // first; norms take outer/norm_size+eps; reduction takes outer/reduce/inner;
    // MLP takes weights/biases + four dims) and a distinct launch shape. Build
    // the exact argument list and grid/block for each FusionKind.
    // These extended kernels launch on a GPU via launch_raw and index every
    // operand as contiguous row-major memory. Validate that all operands live
    // on the same GPU device (reading a CPU or non-contiguous tensor's raw
    // data_ptr() as if it were contiguous GPU memory would silently corrupt the
    // result) and make contiguous copies. The copies are kept alive in these
    // vectors so the pointers taken from them below remain valid until launch
    // (mirrors KernelCodegen::execute_fused).
    // `dev` was validated above (CUDA/ROCm only) and folded into the cache key.
    auto require_dev = [&](const Tensor& t, const char* what) {
        if (t.device() != dev) {
            throw std::runtime_error(
                std::string("execute_extended_fused: all ") + what +
                " must be on the same device as inputs[0]");
        }
    };
    std::vector<Tensor> c_inputs;
    c_inputs.reserve(inputs.size());
    for (const auto& t : inputs) { require_dev(t, "inputs"); c_inputs.push_back(t.contiguous()); }
    std::vector<Tensor> c_params;
    c_params.reserve(params.size());
    for (const auto& t : params) { require_dev(t, "params"); c_params.push_back(t.contiguous()); }

    auto in_shape = std::vector<int64_t>(c_inputs[0].shape().begin(), c_inputs[0].shape().end());
    const int64_t ndim = static_cast<int64_t>(in_shape.size());
    const int64_t total = c_inputs[0].numel();
    auto axis_norm = [&](int axis) -> int64_t {
        int64_t a = axis;
        if (a < 0) a += ndim;
        if (a < 0) a = 0;
        if (a >= ndim) a = (ndim > 0 ? ndim - 1 : 0);
        return a;
    };
    auto suffix_prod = [&](int64_t from) -> int64_t {
        int64_t p = 1; for (int64_t d = from; d < ndim; ++d) p *= in_shape[d]; return p;
    };

    constexpr int kBlock = 256;
    Tensor output;
    std::vector<const void*> ptrs;   // buffer args (fully populated before &-taken)
    std::vector<int64_t> ivals;      // integer scalar args (fully populated first)
    // eps must be marshalled at the kernel's COMPUTE precision: a Float64 norm
    // compiles `double eps` (compute_type == double), so passing a 4-byte float
    // by pointer would misalign the kernel arg and feed it garbage. Keep both a
    // float and a double copy alive; select by dtype when appending the arg.
    float eps_val_f = static_cast<float>(group.eps);
    double eps_val_d = group.eps;
    bool has_eps = false;
    int grid = 1;
    unsigned shared = 0;

    // launch_raw takes a 1D int grid (gridDim.x only); blockIdx.y is always 0 for
    // these launches. A grid that exceeds INT_MAX cannot be represented as a 1D
    // int, so silently truncating would launch too few/negative blocks and leave
    // output slices uninitialized. Reject it explicitly instead of wrapping.
    auto grid_to_int = [](int64_t blocks, const char* what) -> int {
        if (blocks < 0 || blocks > static_cast<int64_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                std::string("execute_extended_fused: ") + what +
                " grid size " + std::to_string(blocks) +
                " exceeds the maximum representable 1D launch grid (INT_MAX)");
        }
        return static_cast<int>(blocks);
    };

    switch (group.kind) {
        case FusionKind::Softmax: {
            // The kernel does a per-row reduction over `cols` CONTIGUOUS
            // elements, i.e. it can only normalize the last axis. Using
            // suffix_prod() here would fold every trailing dim into `cols` and
            // normalize over a superset of the intended axis for a non-last-dim
            // softmax. That layout can't be expressed by this (rows, cols)
            // kernel, so reject it rather than silently miscompute.
            int64_t axis = axis_norm(group.softmax_dim);
            if (ndim == 0 || axis != ndim - 1) {
                throw std::runtime_error(
                    "execute_extended_fused: fused Softmax kernel only supports "
                    "normalization over the last dimension (softmax_dim must be "
                    "ndim-1)");
            }
            int64_t cols = std::max<int64_t>(1, in_shape[axis]);
            int64_t rows = total / cols;
            output = Tensor(in_shape, group.dtype, dev);
            ptrs = {c_inputs[0].data_ptr(), output.data_ptr()};
            ivals = {rows, cols};
            grid = grid_to_int(rows, "Softmax");
            break;
        }
        case FusionKind::LayerNorm:
        case FusionKind::RMSNorm: {
            int64_t norm_size = std::max<int64_t>(1, suffix_prod(axis_norm(group.norm_axis)));
            int64_t outer = total / norm_size;
            output = Tensor(in_shape, group.dtype, dev);
            ptrs = {c_inputs[0].data_ptr(), output.data_ptr()};
            if (group.has_affine) {
                ptrs.push_back(c_params.at(0).data_ptr());         // gamma
                if (group.kind == FusionKind::LayerNorm) {
                    ptrs.push_back(c_params.at(1).data_ptr());     // beta
                }
            }
            ivals = {outer, norm_size};
            has_eps = true;
            grid = grid_to_int(outer, "Norm");
            break;
        }
        case FusionKind::Reduction: {
            int64_t a = axis_norm(group.reduce_dim);
            int64_t reduce_size = (ndim > 0) ? in_shape[a] : total;
            int64_t outer = 1; for (int64_t d = 0; d < a; ++d) outer *= in_shape[d];
            int64_t inner = suffix_prod(a + 1);
            std::vector<int64_t> out_shape;
            for (int64_t d = 0; d < ndim; ++d) {
                if (d == a) { if (group.keepdim) out_shape.push_back(1); }
                else out_shape.push_back(in_shape[d]);
            }
            output = Tensor(out_shape, group.dtype, dev);
            ptrs = {c_inputs[0].data_ptr(), output.data_ptr()};
            ivals = {outer, reduce_size, inner};
            grid = grid_to_int(outer * inner, "Reduction");
            break;
        }
        case FusionKind::GemmEpilogue: {
            // The epilogue kernel reads/writes a buffer that ALREADY holds the
            // GEMM result (it only adds bias + applies the activation). So we
            // must compute A @ B here and hand the product to the kernel as the
            // output buffer; the kernel then fuses bias/activation in place.
            // inputs[0] = A (lhs), inputs[1] = B (rhs); bias (if any) is a param.
            if (c_inputs.size() < 2) {
                throw std::runtime_error(
                    "execute_extended_fused: GemmEpilogue requires two matmul inputs (A, B)");
            }
            output = tenzor::matmul(c_inputs[0], c_inputs[1]);
            // Geometry is derived from the GEMM RESULT, not from inputs[0]:
            // the kernel grids over rows*cols of the product and indexes bias by
            // the last (column) dimension.
            auto out_shape = output.shape();
            const int64_t out_ndim = static_cast<int64_t>(out_shape.size());
            const int64_t out_total = output.numel();
            int64_t cols = std::max<int64_t>(1, out_ndim > 0 ? out_shape[out_ndim - 1] : out_total);
            int64_t rows = out_total / cols;
            ptrs = {output.data_ptr()};                            // output FIRST
            if (group.has_bias) ptrs.push_back(c_params.at(0).data_ptr());
            ivals = {rows, cols};
            grid = grid_to_int((rows * cols + kBlock - 1) / kBlock, "GemmEpilogue");
            break;
        }
        case FusionKind::SmallMLP: {
            // params: {w1, b1, w2, b2}
            int64_t in_dim = std::max<int64_t>(1, ndim > 0 ? in_shape[ndim - 1] : total);
            int64_t batch = total / in_dim;
            int64_t hidden_dim = group.hidden_dim;
            int64_t out_dim = static_cast<int64_t>(c_params.at(3).numel());  // b2 = [out_dim]
            output = Tensor({batch, out_dim}, group.dtype, dev);
            ptrs = {c_inputs[0].data_ptr(), c_params.at(0).data_ptr(), c_params.at(1).data_ptr(),
                    c_params.at(2).data_ptr(), c_params.at(3).data_ptr(), output.data_ptr()};
            ivals = {batch, in_dim, hidden_dim, out_dim};
            grid = grid_to_int(batch, "SmallMLP");
            shared = static_cast<unsigned>(hidden_dim * static_cast<int64_t>(c_inputs[0].dtype_size()));
            break;
        }
        default:
            throw std::runtime_error("execute_extended_fused: unsupported fusion kind");
    }

    // Build the kernel argument array (pointers to the stable ptr/dim values),
    // in the order each kernel declares: buffers..., dims..., [eps].
    std::vector<void*> args;
    args.reserve(ptrs.size() + ivals.size() + 1);
    for (auto& p : ptrs) args.push_back(static_cast<void*>(&p));
    for (auto& v : ivals) args.push_back(static_cast<void*>(&v));
    if (has_eps) {
        if (group.dtype == DType::Float64) {
            args.push_back(static_cast<void*>(&eps_val_d));
        } else {
            args.push_back(static_cast<void*>(&eps_val_f));
        }
    }

    kernel->launch_raw(args, grid, kBlock, shared, nullptr);

    // Record that the native codegen path actually launched a kernel (observable
    // proof for the JIT executor that a fusion node ran on the GPU, not eager).
    g_extended_fused_launches.fetch_add(1, std::memory_order_relaxed);

    return output;
}

} // namespace jit
} // namespace tenzor
