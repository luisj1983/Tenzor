/**
 * @file codegen.cpp
 * @brief Runtime GPU kernel generation via NVRTC/HIPRTC
 *
 * Generates, compiles, and caches fused element-wise GPU kernels at runtime.
 */

#include "tenzor/jit/codegen.hpp"
#include "tenzor/jit/autotune.hpp"  // R1-11: AutotuneCache + autotune_mode_active()
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"  // CPU fallback below uses tenzor::{exp,sin,add,...}
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/fast_dispatch.hpp"  // CPU fallback dispatches activations via OpId
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include <array>
#include <sstream>
#include <stdexcept>
#include <mutex>
#include <iomanip>
#include <limits>
#include <cstring>
#include <cstdint>
#include "tenzor/utils/error.hpp"  // NotImplementedError (S25 / audit-12)
#include <iostream>
#include <functional>
#include <tuple>  // std::ignore for intentional discard in destructor cleanup

// Runtime GPU codegen can target CUDA (NVRTC) and/or ROCm (HIPRTC). A combined
// build compiles this ONE host TU with TENZOR_USE_CUDA defined AND TENZOR_HAS_ROCM
// defined (the build never defines TENZOR_USE_ROCM — that macro previously gated a
// dead HIPRTC branch, so ROCm runtime codegen never ran). Both RTCs are selected
// at RUNTIME by the tensor's device type. NVRTC's CUDA driver types (CUresult,
// CUmodule…) and HIP's host-API types (hipError_t, hipModule_t…) have distinct
// names and coexist in one g++ TU as long as we include HIP's HOST api header
// (<hip/hip_runtime_api.h>) rather than <hip/hip_runtime.h> (whose device
// vector-type helpers collide with CUDA's — see core/rocm_transfer.hpp).
#if defined(TENZOR_USE_CUDA)
#include <nvrtc.h>
#include <cuda.h>
#define CODEGEN_CUDA_AVAILABLE 1
#define NVRTC_CHECK(call) do { \
    nvrtcResult res = call; \
    if (res != NVRTC_SUCCESS) { \
        throw std::runtime_error(std::string("NVRTC error: ") + nvrtcGetErrorString(res)); \
    } \
} while(0)
#define CU_CHECK(call) do { \
    CUresult res = call; \
    if (res != CUDA_SUCCESS) { \
        const char* msg = nullptr; \
        cuGetErrorString(res, &msg); \
        throw std::runtime_error(std::string("CUDA Driver error: ") + (msg ? msg : "unknown")); \
    } \
} while(0)
#else
#define CODEGEN_CUDA_AVAILABLE 0
#endif

#if defined(TENZOR_HAS_ROCM)
#include <hip/hiprtc.h>
#include <hip/hip_runtime_api.h>
#define CODEGEN_HIP_AVAILABLE 1
#define HIPRTC_CHECK(call) do { \
    hiprtcResult res = call; \
    if (res != HIPRTC_SUCCESS) { \
        throw std::runtime_error(std::string("HIPRTC error: ") + hiprtcGetErrorString(res)); \
    } \
} while(0)
#define HIP_CHECK(call) do { \
    hipError_t res = call; \
    if (res != hipSuccess) { \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(res)); \
    } \
} while(0)
#else
#define CODEGEN_HIP_AVAILABLE 0
#endif

#define CODEGEN_AVAILABLE (CODEGEN_CUDA_AVAILABLE || CODEGEN_HIP_AVAILABLE)

namespace tenzor::jit {

#if defined(TENZOR_USE_CUDA)
namespace {
// Process-lifetime cache of each CUDA device's retained primary context,
// shared by compile(), launch(), launch_raw(), and ~CompiledKernel() so the
// driver's "current context" (thread-local process state) can always be
// rebound to a kernel's OWN device before it is touched -- mirrors HIP's
// unconditional hipSetDevice(device_index) on every launch. Without this,
// a thread that compiles for cuda:0 then cuda:1 then reuses the cached
// cuda:0 kernel (a cache hit -- compile() not re-invoked) launches against a
// CUfunction bound to device 0's context while device 1's context is
// current: an invalid-context/invalid-handle driver error, or on some
// driver/hardware combinations, silent execution against the wrong device.
// cuDevicePrimaryCtxRetain is itself refcounted by the driver (repeat calls
// for the same device return the identical context handle), so caching here
// is purely to avoid a cuDeviceGet+cuDevicePrimaryCtxRetain round trip on
// every launch, not required for correctness of the retain itself -- but
// deliberately never released (process-lifetime), matching compile()'s
// original intent for its own now-shared cache.
std::mutex g_cuda_ctx_mutex;
std::unordered_map<int, CUcontext> g_cuda_primary_ctxs;

auto cuda_primary_context_for(int device_id) -> CUcontext {
    std::lock_guard<std::mutex> lock(g_cuda_ctx_mutex);
    auto it = g_cuda_primary_ctxs.find(device_id);
    if (it != g_cuda_primary_ctxs.end()) return it->second;
    CUdevice cu_device;
    CU_CHECK(cuDeviceGet(&cu_device, device_id));
    CUcontext cu_context = nullptr;
    CU_CHECK(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
    g_cuda_primary_ctxs.emplace(device_id, cu_context);
    return cu_context;
}

// Destructor-safe variant: destructors must not throw, so this swallows any
// driver error (leaving the current context whatever it was) instead of
// using the throwing CU_CHECK macro. Best-effort, mirroring this file's
// existing "ignore cuModuleUnload's return" destructor pattern.
auto cuda_try_set_current_context_for(int device_id) noexcept -> void {
    try {
        CUresult res = cuCtxSetCurrent(cuda_primary_context_for(device_id));
        (void)res;
    } catch (...) {
        // Best-effort; a failed rebind here just risks the subsequent
        // cuModuleUnload targeting the wrong (but still valid) context,
        // which is what happened unconditionally before this fix.
    }
}
}  // namespace
#endif

// ============================================================================
// FusionGroup
// ============================================================================

auto FusionGroup::compute_signature() -> std::string {
    std::ostringstream ss;
    // Key on the full device (type AND ordinal). A compiled module/function is
    // bound to one device's context and compute arch, so a kernel built for
    // cuda:0 must not be served for a cuda:1 input under an otherwise-identical
    // key (illegal handle / wrong-arch). Mirrors compile.cpp / compiled_module.cpp
    // which encode device().to_string() into their keys.
    ss << "fusion_" << device.to_string() << "_" << num_inputs << "_"
       << static_cast<int>(dtype);
    for (const auto& step : steps) {
        ss << "_" << static_cast<int>(step.op)
           << "i" << step.input_idx
           << "j" << step.second_input_idx;
        // Key on the scalar's exact bit pattern for every step: this both
        // disambiguates scalar-consuming ops that sort below AddScalar
        // (LeakyRelu/Elu/Softplus) and prevents distinct doubles from
        // colliding via printed-precision rounding.
        uint64_t bits;
        std::memcpy(&bits, &step.scalar, sizeof(bits));
        ss << "s" << bits;
    }
    signature = ss.str();
    return signature;
}

auto build_fusion(std::vector<ElemStep> steps, int num_inputs, DType dtype)
    -> FusionGroup {
    FusionGroup group;
    group.steps = std::move(steps);
    group.num_inputs = num_inputs;
    group.dtype = dtype;
    group.compute_signature();
    return group;
}

// ============================================================================
// Code Generation
// ============================================================================

auto KernelCodegen::dtype_to_cuda_type(DType dtype) -> std::string {
    switch (dtype) {
        case DType::Float32:  return "float";
        case DType::Float64:  return "double";
        case DType::Float16:  return "__half";   // same spelling under NVRTC/HIPRTC
        // Neutral bfloat16 typedef defined in the generated preamble off the
        // device compiler (__nv_bfloat16 under NVRTC, __hip_bfloat16 under HIPRTC),
        // so the same source compiles on either backend in a combined build.
        case DType::BFloat16: return "tz_bf16";
        default:
            // No lossy "float" default: reinterpreting an Int8/Complex buffer as
            // float reads the wrong element size and yields garbage / OOB. Fail
            // loudly (execute_fused also rejects non-float dtypes before this).
            throw std::runtime_error(
                "KernelCodegen: unsupported element dtype for fused GPU kernel: " +
                std::string(dtype_name(dtype)));
    }
}

// In-kernel compute/accumulation type for a storage dtype. Float64 computes in
// double; Float32 and the 16-bit storage types (Float16/BFloat16) compute in
// float — half-precision math is numerically wrong and the device math
// intrinsics don't apply to __half/bfloat16 directly, so those load as float,
// compute in float, and narrow back on store (matching the eager widen-narrow).
static auto elementwise_compute_type(DType dtype) -> std::string {
    switch (dtype) {
        case DType::Float64: return "double";
        default:             return "float";  // Float32/Float16/BFloat16
    }
}

auto KernelCodegen::emit_op(const ElemStep& step, const std::string& vp,
                            DType dtype) -> std::string {
    // vp = variable prefix. step.input_idx refers to either an input array or
    // the previous result (if -1, means "previous result" = vp + "val")
    // Inputs are referenced through per-iteration locals x0,x1,… that generate()
    // pre-widens to the compute type. Reading inpN[i] (the storage type) directly
    // would make comparisons / ?: on __half/bfloat16 ambiguous against float
    // literals (e.g. `half < 0.0f ? 0.0f : half`); widening once fixes that and
    // keeps every op in the compute type.
    auto input = [&](int idx) -> std::string {
        if (idx < 0) return vp + "val";
        return "x" + std::to_string(idx);
    };

    auto a = input(step.input_idx);
    auto b = input(step.second_input_idx);
    // For a Float64 kernel, emit literals WITHOUT the 'f' suffix and with full
    // double precision; an 'f'-suffixed constant in a double kernel is rounded
    // to single precision (~7 digits), defeating the f64 path.
    const bool f64 = (dtype == DType::Float64);
    const std::string F = f64 ? "" : "f";          // float-literal suffix
    const std::string ZERO = "0.0" + F;
    const std::string ONE = "1.0" + F;
    // Emit scalar literals with full round-trip (double) precision; the older
    // std::to_string rounds to ~6 digits and silently defeats the f64 path.
    auto fmt_double = [](double v) -> std::string {
        std::ostringstream o;
        o << std::setprecision(std::numeric_limits<double>::max_digits10) << v;
        std::string str = o.str();
        // Ensure the result is a valid floating literal body: a whole number
        // prints as e.g. "2", and appending the 'f' suffix would yield the
        // invalid literal "2f". Give it a decimal point unless it already has
        // a '.', an exponent, or is inf/nan.
        if (str.find_first_of(".eEni") == std::string::npos) {
            str += ".0";
        }
        return str;
    };
    // Emit a COMPLETE scalar-operand literal (already suffixed). For non-finite
    // values, emit a valid device constant instead of the invalid "inff"/"nanf"
    // token that "<printed>" + F previously produced, which failed NVRTC/HIPRTC
    // compilation (JIT-002). __int_as_float / __longlong_as_double are available
    // in both CUDA and HIP device code.
    auto emit_scalar = [&](double v) -> std::string {
        if (!std::isfinite(v)) {
            if (std::isnan(v)) {
                return f64 ? "__longlong_as_double(0x7ff8000000000000LL)"
                           : "__int_as_float(0x7fc00000)";
            }
            const std::string inf = f64
                ? "__longlong_as_double(0x7ff0000000000000LL)"
                : "__int_as_float(0x7f800000)";
            return v < 0.0 ? ("(-" + inf + ")") : inf;
        }
        return fmt_double(v) + F;  // finite: full literal incl. suffix
    };
    std::string s = emit_scalar(step.scalar);       // complete scalar operand literal

    // Activation scalars default to their PyTorch nn.functional values when the
    // step records 0.0 (no explicit slope/alpha). Mirror execute_fused_cpu
    // (LeakyRelu -> 0.01, Elu -> 1.0); without this the negative branch would
    // collapse to 0 (LeakyRelu) or use a 0 coefficient (Elu), diverging from
    // the CPU fallback for the same fused group.
    // JIT-R023: use emit_scalar (NaN/Inf-safe device intrinsics), not raw
    // fmt_double -- a traced NaN/Inf LeakyReLU slope or Elu/Softplus alpha
    // would otherwise emit the invalid device literal token "nanf"/"inff"
    // (the exact JIT-002 bug emit_scalar exists to fix), failing NVRTC/HIPRTC
    // compilation identically on both backends. Returns a COMPLETE literal
    // (dtype suffix already included for the finite case, no suffix needed
    // for the non-finite intrinsic-call case) -- callers must NOT append `F`.
    auto activation_scalar = [&](double dflt) -> std::string {
        return emit_scalar(step.scalar != 0.0 ? step.scalar : dflt);
    };

    // Device math intrinsic name for this kernel's compute precision. The
    // unsuffixed names (exp, sqrt, erf…) resolve to the DOUBLE overload even in a
    // Float32 kernel, so a Float32 fusion computed in double and diverged from the
    // eager Float32 (expf) path (and was slower). Emit the single-precision
    // variants (expf, sqrtf, erff…) for everything except Float64. F is already
    // "" for Float64 and "f" otherwise, so `fn(base)` == base + F is exactly the
    // right overload. Half/BFloat16 compute in float, so they take the f-suffixed
    // intrinsics too.
    auto fn = [&](const char* base) -> std::string { return std::string(base) + F; };

    switch (step.op) {
        // Unary
        case ElemOp::Neg:        return vp + "val = -" + a + ";";
        case ElemOp::Abs:        return vp + "val = " + fn("fabs") + "(" + a + ");";
        case ElemOp::Sign:       return vp + "val = (" + a + " > 0) - (" + a + " < 0);";
        case ElemOp::Reciprocal: return vp + "val = " + ONE + " / " + a + ";";
        case ElemOp::Exp:        return vp + "val = " + fn("exp") + "(" + a + ");";
        case ElemOp::Log:        return vp + "val = " + fn("log") + "(" + a + ");";
        case ElemOp::Sqrt:       return vp + "val = " + fn("sqrt") + "(" + a + ");";
        case ElemOp::Pow:        return vp + "val = " + fn("pow") + "(" + a + ", " + b + ");";
        case ElemOp::Sin:        return vp + "val = " + fn("sin") + "(" + a + ");";
        case ElemOp::Cos:        return vp + "val = " + fn("cos") + "(" + a + ");";
        case ElemOp::Tan:        return vp + "val = " + fn("tan") + "(" + a + ");";
        case ElemOp::Asin:       return vp + "val = " + fn("asin") + "(" + a + ");";
        case ElemOp::Acos:       return vp + "val = " + fn("acos") + "(" + a + ");";
        case ElemOp::Atan:       return vp + "val = " + fn("atan") + "(" + a + ");";
        case ElemOp::Sinh:       return vp + "val = " + fn("sinh") + "(" + a + ");";
        case ElemOp::Cosh:       return vp + "val = " + fn("cosh") + "(" + a + ");";
        case ElemOp::Tanh:       return vp + "val = " + fn("tanh") + "(" + a + ");";
        case ElemOp::Sigmoid:    return vp + "val = " + ONE + " / (" + ONE + " + " + fn("exp") + "(-" + a + "));";
        // Match the CPU fallback (clamp_min(x, 0) == `x < 0 ? 0 : x`), which
        // propagates NaN; fmax(NaN, 0) returns 0 and diverged from CPU/eager.
        case ElemOp::Relu:       return vp + "val = (" + a + " < " + ZERO + ") ? " + ZERO + " : " + a + ";";
        case ElemOp::LeakyRelu:  return vp + "val = " + a + " > 0 ? " + a + " : " + activation_scalar(0.01) + " * " + a + ";";
        case ElemOp::Elu:        return vp + "val = " + a + " > 0 ? " + a + " : " + activation_scalar(1.0) + " * (" + fn("exp") + "(" + a + ") - " + ONE + ");";
        case ElemOp::Selu: {
            std::string lam = "1.0507009873554805" + F;
            std::string alp = "1.6732632423543772" + F;
            return vp + "val = " + a + " > 0 ? " + lam + " * " + a + " : " + lam + " * " + alp + " * (" + fn("exp") + "(" + a + ") - " + ONE + ");";
        }
        case ElemOp::Gelu:
            // Exact erf GELU 0.5*x*(1 + erf(x / sqrt(2))) to match the CPU/eager
            // kernel (cpu::gelu_kernel, approximate='none'); 1/sqrt(2) =
            // 0.7071067811865476. erf is available in NVRTC/HIPRTC device math.
            // The older tanh approximation diverged from CPU by ~1e-3, breaking
            // cross-backend parity / gradcheck for any fused group with GELU.
            return vp + "val = 0.5" + F + " * " + a + " * (" + ONE + " + " + fn("erf") + "(" + a + " * 0.7071067811865476" + F + "));";
        case ElemOp::Mish:
            return vp + "val = " + a + " * " + fn("tanh") + "(" + fn("log") + "(" + ONE + " + " + fn("exp") + "(" + a + ")));";
        case ElemOp::Softplus: {
            // Match the CPU fallback (OpId::Softplus, beta from scalar/default 1,
            // threshold 20): softplus_beta(x) = (1/beta)*log1p(exp(beta*x)), but
            // return x once beta*x > threshold. The old plain log(1+exp(x))
            // overflowed to +inf for large x (CPU returns x).
            std::string beta = activation_scalar(1.0);
            return vp + "val = (" + beta + " * " + a + " > 20.0" + F + ") ? " +
                   a + " : " + fn("log1p") + "(" + fn("exp") + "(" + beta + " * " + a + ")) / (" + beta + ");";
        }
        case ElemOp::Erf:        return vp + "val = " + fn("erf") + "(" + a + ");";
        case ElemOp::Erfc:       return vp + "val = " + fn("erfc") + "(" + a + ");";
        case ElemOp::Log2:       return vp + "val = " + fn("log2") + "(" + a + ");";
        case ElemOp::Log10:      return vp + "val = " + fn("log10") + "(" + a + ");";
        case ElemOp::Log1p:      return vp + "val = " + fn("log1p") + "(" + a + ");";
        case ElemOp::Exp2:       return vp + "val = " + fn("exp2") + "(" + a + ");";
        case ElemOp::Expm1:      return vp + "val = " + fn("expm1") + "(" + a + ");";
        case ElemOp::Floor:      return vp + "val = " + fn("floor") + "(" + a + ");";
        case ElemOp::Ceil:       return vp + "val = " + fn("ceil") + "(" + a + ");";
        case ElemOp::Round:      return vp + "val = " + fn("round") + "(" + a + ");";

        // Binary
        case ElemOp::Add:  return vp + "val = " + a + " + " + b + ";";
        case ElemOp::Sub:  return vp + "val = " + a + " - " + b + ";";
        case ElemOp::Mul:  return vp + "val = " + a + " * " + b + ";";
        case ElemOp::Div:  return vp + "val = " + a + " / " + b + ";";
        // JIT-054b: match eager maximum_typed/minimum_typed's sel() EXACTLY --
        // check the first operand for NaN, then the second, only THEN compare
        // (see math.cpp maximum_typed/minimum_typed's `sel` lambda: `if (fx !=
        // fx) return x; if (fy != fy) return y; return fx < fy ? x : y;`).
        // The previous plain `(a > b) ? a : b` / `(a < b) ? a : b` only
        // propagated a NaN SECOND operand (comparison-with-NaN is always
        // false, so a NaN in the THEN-branch operand `a` fell through to the
        // ELSE-branch `b`, silently dropping it) -- confirmed reproducible on
        // ROCm/HIPRTC for Min with a NaN first operand (ExecuteFusedNaN-
        // PropagationAllBackends). `x != x` is the portable, IEEE-754-defined
        // NaN test (true iff NaN) and does not rely on isnan()/fmax()/fmin(),
        // which this project's own history already flagged as unreliable
        // under ROCm/HIP. fmax/fmin drop NaN entirely and would diverge from
        // CPU/eager regardless.
        case ElemOp::Max:
            return vp + "val = (" + a + " != " + a + ") ? " + a + " : (" + b +
                   " != " + b + ") ? " + b + " : ((" + a + " > " + b + ") ? " +
                   a + " : " + b + ");";
        case ElemOp::Min:
            return vp + "val = (" + a + " != " + a + ") ? " + a + " : (" + b +
                   " != " + b + ") ? " + b + " : ((" + a + " < " + b + ") ? " +
                   a + " : " + b + ");";
        case ElemOp::Fmod: return vp + "val = " + fn("fmod") + "(" + a + ", " + b + ");";

        // Scalar ops
        case ElemOp::AddScalar:  return vp + "val = " + a + " + " + s + ";";
        case ElemOp::MulScalar:  return vp + "val = " + a + " * " + s + ";";
        case ElemOp::PowScalar:  return vp + "val = " + fn("pow") + "(" + a + ", " + s + ");";
        // clamp_min/clamp_max == std::max/std::min(x, s), which propagate NaN
        // (clamp_min_kernel/clamp_max_kernel blend the NaN back over AVX
        // max/min); fmax/fmin drop NaN and diverge from CPU/eager.
        case ElemOp::ClampMin:   return vp + "val = (" + a + " < " + s + ") ? " + s + " : " + a + ";";
        case ElemOp::ClampMax:   return vp + "val = (" + a + " > " + s + ") ? " + s + " : " + a + ";";
    }
    // No `default:` — `-Wswitch` flags any ElemOp without an explicit case above
    // (promoted to an error under the CI/Release warning flags), and the throw is
    // the runtime backstop. The previous silent identity default (`val = a`) let a
    // newly-added ElemOp that was wired into the CPU twin (execute_fused_cpu) but
    // missed here compile into a passthrough kernel, silently diverging CUDA/ROCm
    // from CPU. The throw is unreachable for any valid enum value and satisfies
    // the non-void return.
    throw std::runtime_error(
        "KernelCodegen::emit_op: unhandled ElemOp (" +
        std::to_string(static_cast<int>(step.op)) + ")");
}

auto KernelCodegen::generate(const FusionGroup& group) -> std::string {
    const std::string T = dtype_to_cuda_type(group.dtype);        // storage type
    const std::string C = elementwise_compute_type(group.dtype);  // compute type
    // JIT-R134: naming purely by step COUNT let two semantically different
    // fusion groups (e.g. a lone Sigmoid vs. a lone Tanh, or [Add,Relu] vs
    // [Sub,Gelu]) with equal step count/dtype/device-arch/numel collapse
    // onto the SAME AutotuneCache key (CompiledKernel::launch() builds the
    // key from this name) even though FusionGroup::compute_signature()
    // correctly distinguishes them as separate cached CompiledKernel
    // objects -- a block size autotuned for one kernel silently got reused
    // for a completely different one. Fold a hash of the full op-sequence
    // signature into the name so distinct fusion patterns get distinct
    // autotune keys; purely a naming addition, does not change compile()'s
    // C-identifier validity (hex digits only).
    std::string kernel_name = "fused_elementwise_" +
        std::to_string(group.steps.size()) + "_ops_" +
        [&group]() {
            std::ostringstream hex;
            hex << std::hex << std::hash<std::string>{}(group.signature);
            return hex.str();
        }();

    std::ostringstream body;

    // Header. Pointers use the STORAGE type T (e.g. __half); the per-element
    // computation runs in the COMPUTE type C (float for the 16-bit types), so the
    // half operands widen to float on load and narrow back on the single store —
    // matching the eager widen-compute-narrow kernels.
    body << "extern \"C\" __global__ void " << kernel_name << "(\n";
    for (int i = 0; i < group.num_inputs; ++i) {
        body << "    const " << T << "* __restrict__ inp" << i << ",\n";
    }
    body << "    " << T << "* __restrict__ out,\n";
    body << "    long long numel\n";
    body << ") {\n";

    // Grid-stride loop
    body << "    for (long long i = blockIdx.x * blockDim.x + threadIdx.x;\n";
    body << "         i < numel;\n";
    body << "         i += blockDim.x * gridDim.x) {\n";

    // Compute in C. Widen each input operand to the compute type once (x0,x1,…);
    // the 16-bit storage types (__half/bfloat16) promote to float here so all
    // subsequent ops (including comparisons / ?:) are unambiguous. The result
    // narrows back to the storage type T on the single store.
    for (int i = 0; i < group.num_inputs; ++i) {
        body << "        " << C << " x" << i << " = static_cast<" << C
             << ">(inp" << i << "[i]);\n";
    }
    // For the 16-bit storage types (Float16/BFloat16, where T != C) eager runs
    // each fused elementwise op as a full tensor op that NARROWS the result back
    // to the 16-bit storage dtype after EVERY step. Keeping `val` in float across
    // all steps (narrowing only on the final store) therefore diverges from
    // eager/CPU on any multi-step 16-bit fusion. Mirror eager's per-op narrowing
    // by round-tripping `val` through the storage type T after each step. For
    // Float32/Float64 (T == C) this is skipped entirely — no behavior change, no
    // overhead.
    const bool round_each_step = (T != C);
    body << "        " << C << " val;\n";
    for (size_t s = 0; s < group.steps.size(); ++s) {
        body << "        " << emit_op(group.steps[s], "", group.dtype) << "\n";
        if (round_each_step) {
            body << "        val = static_cast<" << C << ">(static_cast<" << T
                 << ">(val));\n";
        }
    }
    body << "        out[i] = static_cast<" << T << ">(val);\n";
    body << "    }\n";
    body << "}\n";

    // Preamble: pull in the half/bfloat16 device headers and define the neutral
    // tz_bf16 type when the kernel references those storage types. The header
    // names AND the bfloat16 type name differ between CUDA and HIP, so both are
    // selected off the DEVICE compiler (__CUDACC_RTC__ only under NVRTC) — the
    // same source then compiles under either NVRTC or HIPRTC in a combined build.
    std::string src = body.str();
    std::ostringstream pre;
    if (src.find("__half") != std::string::npos) {
        pre << "#if defined(__CUDACC_RTC__)\n"
               "#include <cuda_fp16.h>\n"
               "#else\n"
               "#include <hip/hip_fp16.h>\n"
               "#endif\n";
    }
    if (src.find("tz_bf16") != std::string::npos) {
        pre << "#if defined(__CUDACC_RTC__)\n"
               "#include <cuda_bf16.h>\n"
               "typedef __nv_bfloat16 tz_bf16;\n"
               "#else\n"
               "#include <hip/hip_bf16.h>\n"
               "typedef __hip_bfloat16 tz_bf16;\n"
               "#endif\n";
    }
    return pre.str() + src;
}

// ============================================================================
// Compiled Kernel
// ============================================================================

CompiledKernel::~CompiledKernel() {
#if CODEGEN_HIP_AVAILABLE
    if (is_hip) {
        // hipSetDevice's return isn't checked either (best-effort, mirrors
        // hipModuleUnload immediately below) -- a destructor must not throw.
        std::ignore = hipSetDevice(device_index);
        if (module) std::ignore = hipModuleUnload(static_cast<hipModule_t>(module));
        return;
    }
#endif
#if defined(TENZOR_USE_CUDA)
    // JIT-054: rebind before unload, same rationale as launch()/launch_raw()
    // -- best-effort/non-throwing since destructors must not throw.
    if (module) {
        cuda_try_set_current_context_for(device_index);
        cuModuleUnload(static_cast<CUmodule>(module));
    }
#endif
}

#if CODEGEN_AVAILABLE
namespace {

// R1-11: candidate block sizes considered by real kernel-launch autotuning.
// 1024 is a safe upper bound on both CUDA and ROCm (max threads/block).
constexpr std::array<int, 6> kAutotuneBlockSizes = {32, 64, 128, 256, 512, 1024};

// Grid size for `numel` elements at `block_size` threads/block, clamped to
// the driver's per-dimension grid limit. Computed in int64 and clamped
// BEFORE narrowing to int: casting the raw (numel + block - 1)/block to int
// first overflows for numel > ~2^31*block (possibly negative), and a
// negative value slips past the 65535 clamp. The kernel uses a grid-stride
// loop, so clamping the grid is safe.
auto autotune_grid_size(int64_t numel, int block_size) -> int {
    int64_t grid_blocks = (numel + block_size - 1) / block_size;
    if (grid_blocks > 65535) grid_blocks = 65535;
    if (grid_blocks < 1) grid_blocks = 1;
    return static_cast<int>(grid_blocks);
}

// Original static numel-threshold heuristic. Used when there is no
// AutotuneCache entry for this kernel/shape AND the caller is not running in
// max-autotune mode (so no benchmarking is performed). Small tensors benefit
// from fewer threads (less launch overhead); large tensors need more threads
// for full GPU occupancy.
auto autotune_static_heuristic_block_size(int64_t numel) -> int {
    if (numel <= 256) return 64;
    if (numel <= 4096) return 128;
    if (numel <= 65536) return 256;
    return 512;
}

// Memoized device-architecture identifier for AutotuneCache keys. Computed
// once per (device_index, is_hip) pair -- hipGetDeviceProperties /
// cuDeviceGetAttribute are cheap but not free, and launch() runs on every
// kernel invocation. See AutotuneCache::make_key's doc comment for why the
// key must be arch-qualified, not just backend+ordinal.
auto autotune_device_arch(int device_index, bool is_hip) -> std::string {
    static std::mutex mtx;
    static std::unordered_map<std::string, std::string> cache;
    const std::string map_key = (is_hip ? "rocm#" : "cuda#") + std::to_string(device_index);
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = cache.find(map_key);
        if (it != cache.end()) return it->second;
    }
    std::string arch = "unknown";
#if CODEGEN_HIP_AVAILABLE
    if (is_hip) {
        hipDeviceProp_t prop{};
        arch = (hipGetDeviceProperties(&prop, device_index) == hipSuccess)
                   ? (std::string("rocm:") + prop.gcnArchName)
                   : "rocm:unknown";
    }
#endif
#if defined(TENZOR_USE_CUDA)
    if (!is_hip) {
        CUdevice cu_device;
        if (cuDeviceGet(&cu_device, device_index) == CUDA_SUCCESS) {
            int major = 0, minor = 0;
            cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_device);
            cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_device);
            arch = "cuda:sm_" + std::to_string(major) + std::to_string(minor);
        } else {
            arch = "cuda:unknown";
        }
    }
#endif
    std::lock_guard<std::mutex> lock(mtx);
    cache[map_key] = arch;
    return arch;
}

} // namespace
#endif // CODEGEN_AVAILABLE

auto CompiledKernel::launch(const std::vector<const void*>& input_ptrs,
                            void* output_ptr, int64_t numel, void* stream) -> void {
#if CODEGEN_AVAILABLE
    if (!function) {
        throw std::runtime_error("CompiledKernel::launch: kernel not compiled");
    }

    // Build kernel arguments: input pointers + output pointer + numel
    std::vector<void*> args;
    args.reserve(input_ptrs.size() + 2);
    // CUDA driver API requires pointers-to-pointers for kernel args
    std::vector<const void*> input_ptrs_copy = input_ptrs;
    for (auto& ptr : input_ptrs_copy) {
        args.push_back(&ptr);
    }
    args.push_back(&output_ptr);
    args.push_back(&numel);

    // Issues ONE launch at the given geometry. Shared by the real launch
    // below and by the R1-11 autotune benchmarking loop further down --
    // every candidate computes the exact same result (the generated kernel
    // loops over numel with a grid-stride loop, per autotune_grid_size's
    // comment), so timing candidates is a pure performance exercise with
    // zero correctness risk: whichever candidate ran LAST left the
    // bit-identical correct result in output_ptr.
    auto do_launch = [&](int grid_size, int block_size) {
#if CODEGEN_HIP_AVAILABLE
        if (is_hip) {
            HIP_CHECK(hipSetDevice(device_index));
            HIP_CHECK(hipModuleLaunchKernel(
                static_cast<hipFunction_t>(function),
                grid_size, 1, 1,
                block_size, 1, 1,
                0,
                static_cast<hipStream_t>(stream),
                args.data(),
                nullptr));
            return;
        }
#endif
#if defined(TENZOR_USE_CUDA)
        // JIT-054: rebind the CUDA driver's current context to this kernel's
        // OWN device before every launch, mirroring the HIP branch's
        // unconditional hipSetDevice(device_index) above. Without this, a
        // cache-hit launch of a kernel compiled for a different device than
        // whatever context is currently active on this thread fails (or on
        // some drivers, silently targets the wrong GPU) -- see
        // cuda_primary_context_for's doc comment.
        CU_CHECK(cuCtxSetCurrent(cuda_primary_context_for(device_index)));
        CU_CHECK(cuLaunchKernel(
            static_cast<CUfunction>(function),
            grid_size, 1, 1,    // grid
            block_size, 1, 1,    // block
            0,                    // shared mem
            static_cast<CUstream>(stream),
            args.data(),
            nullptr
        ));
#else
        (void)grid_size; (void)block_size;
        throw NotImplementedError("GPU codegen not available (no CUDA/ROCm)");
#endif
    };

    // Times ONE candidate launch in milliseconds using the backend's own
    // event API (GPU-side timing -- launches are asynchronous, so a CPU wall
    // clock around do_launch would mostly measure launch-queue overhead, not
    // kernel execution time).
    auto time_launch_ms = [&](int grid_size, int block_size) -> double {
#if CODEGEN_HIP_AVAILABLE
        if (is_hip) {
            hipEvent_t start = nullptr, stop = nullptr;
            HIP_CHECK(hipEventCreate(&start));
            HIP_CHECK(hipEventCreate(&stop));
            HIP_CHECK(hipEventRecord(start, static_cast<hipStream_t>(stream)));
            do_launch(grid_size, block_size);
            HIP_CHECK(hipEventRecord(stop, static_cast<hipStream_t>(stream)));
            HIP_CHECK(hipEventSynchronize(stop));
            float ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
            HIP_CHECK(hipEventDestroy(start));
            HIP_CHECK(hipEventDestroy(stop));
            return static_cast<double>(ms);
        }
#endif
#if defined(TENZOR_USE_CUDA)
        CU_CHECK(cuCtxSetCurrent(cuda_primary_context_for(device_index)));
        CUevent start = nullptr, stop = nullptr;
        CU_CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
        CU_CHECK(cuEventCreate(&stop, CU_EVENT_DEFAULT));
        CU_CHECK(cuEventRecord(start, static_cast<CUstream>(stream)));
        do_launch(grid_size, block_size);
        CU_CHECK(cuEventRecord(stop, static_cast<CUstream>(stream)));
        CU_CHECK(cuEventSynchronize(stop));
        float ms = 0.0f;
        CU_CHECK(cuEventElapsedTime(&ms, start, stop));
        CU_CHECK(cuEventDestroy(start));
        CU_CHECK(cuEventDestroy(stop));
        return static_cast<double>(ms);
#else
        (void)grid_size; (void)block_size;
        throw NotImplementedError("GPU codegen not available (no CUDA/ROCm)");
#endif
    };

    // R1-11: adaptive block-size selection, cheapest tier first.
    //   1. AutotuneCache hit for this (kernel, dtype, arch, numel) -> reuse
    //      it. Loaded lazily (once per process) from disk so tuning from a
    //      prior run is picked up even when this call is not itself running
    //      in max-autotune mode.
    //   2. Cache miss AND the calling CompiledFunction is running in
    //      max-autotune mode (AutotuneModeGuard, set by
    //      CompiledFunction::operator()) -> benchmark every candidate with
    //      REAL GPU-timed launches (see do_launch's correctness note above),
    //      record every result, persist the winner to disk.
    //   3. Otherwise -> the original static numel-threshold heuristic.
    static std::once_flag autotune_cache_loaded;
    std::call_once(autotune_cache_loaded, [] {
        AutotuneCache::instance().load_default();
    });

    const std::string autotune_key = AutotuneCache::make_key(
        name, std::string(dtype_name(dtype)),
        autotune_device_arch(device_index, is_hip), {{numel}});

    auto cached_id = AutotuneCache::instance().lookup(
        autotune_key, static_cast<int>(kAutotuneBlockSizes.size()));

    if (cached_id) {
        int block_size = kAutotuneBlockSizes[static_cast<size_t>(*cached_id)];
        do_launch(autotune_grid_size(numel, block_size), block_size);
        return;
    }

    if (autotune_mode_active()) {
        double best_time = std::numeric_limits<double>::max();
        int best_id = -1;
        for (size_t i = 0; i < kAutotuneBlockSizes.size(); ++i) {
            int cand_block = kAutotuneBlockSizes[i];
            double t = time_launch_ms(autotune_grid_size(numel, cand_block), cand_block);
            AutotuneCache::instance().record(autotune_key, static_cast<int>(i), t);
            if (t < best_time) { best_time = t; best_id = static_cast<int>(i); }
        }
        if (best_id >= 0) {
            AutotuneCache::instance().save_default();
            return;
        }
        // All candidate timings were degenerate (e.g. NaN elapsed time from
        // the driver) -- fall through to the static heuristic below rather
        // than leaving the kernel unlaunched.
    }

    {
        int block_size = autotune_static_heuristic_block_size(numel);
        do_launch(autotune_grid_size(numel, block_size), block_size);
    }
#else
    throw NotImplementedError("GPU codegen not available (no CUDA/ROCm)");
#endif
}

auto CompiledKernel::launch_raw(const std::vector<void*>& kernel_args,
                                int grid_size, int block_size, unsigned shared_bytes,
                                void* stream) -> void {
#if CODEGEN_AVAILABLE
    if (!function) {
        throw std::runtime_error("CompiledKernel::launch_raw: kernel not compiled");
    }
    if (grid_size < 1) grid_size = 1;
    if (block_size < 1) block_size = 1;
    // cuLaunchKernel/hipModuleLaunchKernel want a non-const void** to the array
    // of argument pointers; the pointed-to values are owned by the caller.
    std::vector<void*> args = kernel_args;
#if CODEGEN_HIP_AVAILABLE
    if (is_hip) {
        HIP_CHECK(hipSetDevice(device_index));
        HIP_CHECK(hipModuleLaunchKernel(
            static_cast<hipFunction_t>(function),
            grid_size, 1, 1,
            block_size, 1, 1,
            shared_bytes,
            static_cast<hipStream_t>(stream),
            args.data(),
            nullptr));
        return;
    }
#endif
#if defined(TENZOR_USE_CUDA)
    // JIT-054: see launch()'s identical rebind for the rationale.
    CU_CHECK(cuCtxSetCurrent(cuda_primary_context_for(device_index)));
    CU_CHECK(cuLaunchKernel(
        static_cast<CUfunction>(function),
        grid_size, 1, 1,
        block_size, 1, 1,
        shared_bytes,
        static_cast<CUstream>(stream),
        args.data(),
        nullptr));
#else
    throw NotImplementedError("GPU codegen not available (no CUDA/ROCm)");
#endif
#else
    (void)kernel_args; (void)grid_size; (void)block_size; (void)shared_bytes; (void)stream;
    throw NotImplementedError("GPU codegen not available (no CUDA/ROCm)");
#endif
}

// ============================================================================
// Kernel Cache
// ============================================================================

auto KernelCache::instance() -> KernelCache& {
    static KernelCache cache;
    return cache;
}

auto KernelCache::get_or_compile(const FusionGroup& group) -> std::shared_ptr<CompiledKernel> {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_.find(group.signature);
    if (it != cache_.end()) {
        cache_hits_++;
        return it->second;
    }

    // Generate source code
    auto source = KernelCodegen::generate(group);
    // JIT-R134: naming purely by step COUNT let two semantically different
    // fusion groups (e.g. a lone Sigmoid vs. a lone Tanh, or [Add,Relu] vs
    // [Sub,Gelu]) with equal step count/dtype/device-arch/numel collapse
    // onto the SAME AutotuneCache key (CompiledKernel::launch() builds the
    // key from this name) even though FusionGroup::compute_signature()
    // correctly distinguishes them as separate cached CompiledKernel
    // objects -- a block size autotuned for one kernel silently got reused
    // for a completely different one. Fold a hash of the full op-sequence
    // signature into the name so distinct fusion patterns get distinct
    // autotune keys; purely a naming addition, does not change compile()'s
    // C-identifier validity (hex digits only).
    std::string kernel_name = "fused_elementwise_" +
        std::to_string(group.steps.size()) + "_ops_" +
        [&group]() {
            std::ostringstream hex;
            hex << std::hex << std::hash<std::string>{}(group.signature);
            return hex.str();
        }();

    // Compile for the group's target GPU (its context + compute arch), not a
    // hardcoded device 0. Select CUDA vs ROCm runtime by the group's device type.
    auto kernel = compile(source, kernel_name, group.device.index,
                          group.device.type == Device::Type::ROCm, group.dtype);
    kernel->source = source;
    kernel->num_inputs = group.num_inputs;
    cache_[group.signature] = kernel;
    compilations_++;

    return kernel;
}

auto KernelCache::get_or_compile_source(const std::string& signature,
                                        const std::string& source,
                                        const std::string& kernel_name,
                                        int device_index,
                                        bool is_rocm)
    -> std::shared_ptr<CompiledKernel> {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_.find(signature);
    if (it != cache_.end()) {
        cache_hits_++;
        return it->second;
    }

    // Compile for the requesting device's context/compute arch, not a hardcoded
    // device 0. The caller must also fold the device into `signature` so a
    // kernel built for one device is never served to another. is_rocm selects
    // HIPRTC vs NVRTC at runtime (combined build compiles both).
    auto kernel = compile(source, kernel_name, device_index, is_rocm);
    if (kernel) {
        kernel->source = source;
        cache_[signature] = kernel;
        compilations_++;
    }
    return kernel;
}

auto KernelCache::compile(const std::string& source, const std::string& kernel_name,
                          int device_index, bool is_rocm, DType dtype)
    -> std::shared_ptr<CompiledKernel> {
#if CODEGEN_AVAILABLE
    auto kernel = std::make_shared<CompiledKernel>();
    kernel->name = kernel_name;
    kernel->device_index = device_index;
    kernel->dtype = dtype;

    // ---- ROCm target: compile with HIPRTC and load via the HIP driver. ----
#if CODEGEN_HIP_AVAILABLE
    if (is_rocm) {
        HIP_CHECK(hipInit(0));
        // Bind to the target device so hipModuleLoadData/launch use its context
        // and HIPRTC compiles for its architecture.
        HIP_CHECK(hipSetDevice(device_index));
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, device_index));
        // Compile for the resident GPU's ISA. Without an explicit --offload-arch,
        // HIPRTC has no target and either fails or emits code for the wrong arch
        // (e.g. gfx1150 wavefronts are 64-wide — the mask/reduction correctness in
        // the generated source depends on the right target).
        std::string arch_flag = std::string("--offload-arch=") + prop.gcnArchName;

        hiprtcProgram prog;
        HIPRTC_CHECK(hiprtcCreateProgram(&prog, source.c_str(), "fused_kernel.hip",
                                         0, nullptr, nullptr));
        // Provide the ROCm include dir so <hip/hip_fp16.h> / <hip/hip_bf16.h>
        // (Float16/BFloat16 fusions) resolve under HIPRTC.
        // -ffp-contract=off disables FP multiply-add contraction (clang/HIP
        // defaults to "fast", which fuses to a single-rounding fma). Round mul
        // then add separately to match the precise reference / eager CPU and the
        // codebase's double-accumulator bit-match effort (JIT-F004).
        std::vector<std::string> opt_storage = {arch_flag, "--std=c++17",
                                                "-ffp-contract=off"};
#ifdef TENZOR_HIPRTC_INCLUDE
        opt_storage.push_back(std::string("-I") + TENZOR_HIPRTC_INCLUDE);
#endif
        std::vector<const char*> opts;
        opts.reserve(opt_storage.size());
        for (const auto& o : opt_storage) opts.push_back(o.c_str());
        hiprtcResult compile_result =
            hiprtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());
        if (compile_result != HIPRTC_SUCCESS) {
            size_t log_size = 0;
            hiprtcGetProgramLogSize(prog, &log_size);
            std::string log(log_size, '\0');
            hiprtcGetProgramLog(prog, log.data());
            hiprtcDestroyProgram(&prog);
            throw std::runtime_error("HIPRTC compilation failed:\n" + log +
                                     "\nSource:\n" + source);
        }
        size_t code_size = 0;
        HIPRTC_CHECK(hiprtcGetCodeSize(prog, &code_size));
        std::string code(code_size, '\0');
        HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));
        hiprtcDestroyProgram(&prog);

        hipModule_t hip_module;
        HIP_CHECK(hipModuleLoadData(&hip_module, code.data()));
        kernel->module = hip_module;
        hipFunction_t hip_func;
        HIP_CHECK(hipModuleGetFunction(&hip_func, hip_module, kernel_name.c_str()));
        kernel->function = hip_func;
        kernel->is_hip = true;
        return kernel;
    }
#else
    if (is_rocm) {
        throw std::runtime_error(
            "KernelCache::compile: ROCm (HIPRTC) codegen requested but this build "
            "has no ROCm runtime");
    }
#endif

    // ---- CUDA target: compile with NVRTC and load via the CUDA driver. ----
#if defined(TENZOR_USE_CUDA)
    (void)is_rocm;
    // Initialize CUDA driver API and ensure a context is active. Bind to the
    // caller's target device (multi-GPU): a module compiled/loaded under device
    // 0's context yields a CUfunction that is illegal to launch on cuda:1.
    cuInit(0);
    int device_id = device_index;
    CUdevice cu_device;
    CU_CHECK(cuDeviceGet(&cu_device, device_id));

    // Ensure the CURRENT context targets cu_device before we load the module:
    // the resulting CUfunction is bound to whatever device's context is active,
    // so on multi-GPU we must not rely on (or inherit) a context for a different
    // device. Retain each device's primary context EXACTLY ONCE per process
    // (reference-counted, process-lived) keyed by ordinal — a single global
    // once-retain would pin every device to whichever GPU compiled first.
    // Shared with launch()/launch_raw()/~CompiledKernel() via
    // cuda_primary_context_for() so every driver-API touch point (not just
    // compile) rebinds to its kernel's own device (JIT-054).
    CUcontext cu_context = cuda_primary_context_for(device_id);
    CU_CHECK(cuCtxSetCurrent(cu_context));

    // Detect GPU compute capability via driver API
    int major = 7, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cu_device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cu_device);
    std::string arch_flag = "--gpu-architecture=compute_" +
                            std::to_string(major) + std::to_string(minor);

    // Compile with NVRTC
    nvrtcProgram prog;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, source.c_str(), "fused_kernel.cu",
                                    0, nullptr, nullptr));

    // NVRTC bundles no headers: a kernel that #includes <cuda_fp16.h> /
    // <cuda_bf16.h> (Float16/BFloat16 fusions) fails to open the file unless we
    // hand it the CUDA include directory (baked in at build time).
    std::vector<std::string> opt_storage = {
        // --fmad=false disables FP multiply-add contraction so `a*b + c` rounds
        // twice (mul then add), matching the eager CPU/precise reference and the
        // codebase's double-accumulator / 1-over-sqrt bit-match effort. Leaving it
        // on lets the GPU fuse to a single-rounding fma, drifting ~1 ULP/op over
        // long normalized dims (JIT-F004).
        arch_flag, "--std=c++17", "-default-device", "--fmad=false"
    };
#ifdef TENZOR_NVRTC_INCLUDE
    opt_storage.push_back(std::string("--include-path=") + TENZOR_NVRTC_INCLUDE);
#endif
    std::vector<const char*> opts;
    opts.reserve(opt_storage.size());
    for (const auto& o : opt_storage) opts.push_back(o.c_str());
    nvrtcResult compile_result =
        nvrtcCompileProgram(prog, static_cast<int>(opts.size()), opts.data());

    if (compile_result != NVRTC_SUCCESS) {
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        nvrtcGetProgramLog(prog, log.data());
        nvrtcDestroyProgram(&prog);
        throw std::runtime_error("NVRTC compilation failed:\n" + log + "\nSource:\n" + source);
    }

    // Get PTX
    size_t ptx_size;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
    std::string ptx(ptx_size, '\0');
    NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
    nvrtcDestroyProgram(&prog);

    // Load module and get function
    CUmodule cu_module;
    CU_CHECK(cuModuleLoadDataEx(&cu_module, ptx.data(), 0, nullptr, nullptr));
    kernel->module = cu_module;

    CUfunction cu_func;
    CU_CHECK(cuModuleGetFunction(&cu_func, cu_module, kernel_name.c_str()));
    kernel->function = cu_func;

    return kernel;
#else
    // No CUDA runtime in this build. A ROCm-only build reaches here only if
    // is_rocm was false (handled/thrown above), so a CUDA kernel was requested
    // without a CUDA runtime.
    throw std::runtime_error(
        "KernelCache::compile: CUDA (NVRTC) codegen requested but this build has "
        "no CUDA runtime");
#endif
#else
    (void)device_index; (void)is_rocm;
    throw NotImplementedError("GPU codegen not available (no CUDA/ROCm)");
#endif
}

auto KernelCache::clear() -> void {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

auto KernelCache::num_cached() const -> size_t {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

// ============================================================================
// High-level API
// ============================================================================

// Forward declaration; defined further below. The GPU path in execute_fused
// also delegates here on failure modes that require a CPU fallback.
auto execute_fused_cpu(const FusionGroup& group,
                       const std::vector<Tensor>& inputs) -> Tensor;

auto execute_fused(const FusionGroup& group,
                   const std::vector<Tensor>& inputs) -> Tensor {
    if (inputs.empty()) {
        throw std::runtime_error("execute_fused: no inputs provided");
    }

    // Validate inputs
    auto orig_device = inputs[0].device();
    int64_t numel = inputs[0].numel();
    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i].numel() != numel) {
            throw std::runtime_error("execute_fused: all inputs must have same numel");
        }
    }

#if CODEGEN_AVAILABLE
    // The generated kernel loads/stores Float32/Float64/Float16/BFloat16 (the
    // 16-bit types widen to float for the math, compute in float, and narrow back
    // on store — see KernelCodegen::generate). Reject any OTHER dtype up front —
    // before any device transfer, kernel compilation, or output allocation —
    // instead of reinterpreting its bytes as float (which silently produced
    // garbage / OOB for Int*/Complex whose element size differs). Validating here
    // also avoids compiling/caching a dead kernel and wasting CPU->GPU transfers.
    if (group.dtype != DType::Float32 && group.dtype != DType::Float64 &&
        group.dtype != DType::Float16 && group.dtype != DType::BFloat16) {
        throw std::runtime_error(
            "KernelCodegen::execute_fused: fused GPU codegen only supports "
            "Float32/Float64/Float16/BFloat16; got " +
            std::string(dtype_name(group.dtype)) +
            " — route this dtype through the eager fallback");
    }

    // Move inputs to GPU if needed (codegen runs on GPU). CODEGEN_AVAILABLE is
    // set for both CUDA and ROCm builds, so derive the concrete GPU device from
    // the active backend — directing tensors to a CUDA device on a ROCm build
    // fails device validation / lands data on the wrong backend.
#if defined(TENZOR_USE_CUDA)
    auto gpu_device = Device::cuda(0);
#elif defined(TENZOR_HAS_ROCM)
    auto gpu_device = Device::rocm(0);
#else
    auto gpu_device = Device::cuda(0);
#endif
    // Prefer the device of an input already resident on a GPU, so a model on
    // cuda:1 / rocm:1 doesn't get its output allocated on device 0 (which would
    // make the kernel access memory across devices). Fall back to the default.
    for (const auto& inp : inputs) {
        if (inp.device().type != Device::Type::CPU) {
            gpu_device = inp.device();
            break;
        }
    }

    // The generated kernel is an NVRTC/HIPRTC device kernel; it can only be
    // compiled for and launched on a CUDA or ROCm device. A Vulkan/OneAPI tensor
    // reaching here (possible in a combined build where CODEGEN_AVAILABLE is set
    // by CUDA/ROCm) must NOT be treated as the GPU target — compiling a CUDA/HIP
    // kernel and launching it over that device's memory is illegal. Reject
    // loudly so the caller routes this group through the correct backend path.
    // Mirrors execute_extended_fused's guard.
    if (gpu_device.type != Device::Type::CUDA &&
        gpu_device.type != Device::Type::ROCm) {
        throw std::runtime_error(
            "KernelCodegen::execute_fused: native fused codegen targets only "
            "CUDA/ROCm; got a " + gpu_device.to_string() +
            " tensor — route this group through its backend's own path");
    }

    std::vector<Tensor> gpu_inputs;
    gpu_inputs.reserve(inputs.size());
    for (const auto& inp : inputs) {
        if (inp.device().type == Device::Type::CPU) {
            // A CPU-resident operand (e.g. a hoisted scalar constant) is moved
            // onto the target GPU. This is not a fallback: the kernel still runs
            // on the GPU.
            gpu_inputs.push_back(inp.to(gpu_device));
        } else if (inp.device() == gpu_device) {
            gpu_inputs.push_back(inp);
        } else {
            // A GPU-resident operand on a DIFFERENT device than the launch
            // device. Passing its data_ptr() to a kernel launched on gpu_device
            // is an illegal cross-device access. Reject loudly rather than
            // silently corrupt (mirrors execute_extended_fused's require_dev).
            throw std::runtime_error(
                "KernelCodegen::execute_fused: fused inputs span multiple GPU "
                "devices (" + inp.device().to_string() + " vs the launch device "
                + gpu_device.to_string() + "); a single fused kernel cannot "
                "read across devices");
        }
    }

    // Get or compile the kernel for the ACTUAL target GPU. The fusion group was
    // built without knowing which device it would run on; pin it to the resident
    // device now so both the cache key (compute_signature) and the driver-side
    // compile (which selects the device context + compute arch) target this GPU
    // rather than always device 0.
    FusionGroup device_group = group;
    device_group.device = gpu_device;
    device_group.compute_signature();
    auto kernel = KernelCache::instance().get_or_compile(device_group);

    // Allocate output on GPU
    std::vector<int64_t> shape(inputs[0].shape().begin(), inputs[0].shape().end());
    Tensor output(shape, group.dtype, gpu_device);

    // Collect input data pointers (now on GPU). contiguous() may allocate a
    // fresh tensor whose storage is NOT shared with gpu_inputs[i]; storing a
    // pointer into a loop-scoped temporary would dangle once it is destroyed,
    // so replace each entry in place with its contiguous version and take the
    // pointer from the long-lived vector element (kept alive until launch).
    for (auto& inp : gpu_inputs) {
        inp = inp.contiguous();
    }
    // Raw device pointers, dtype-agnostic: the kernel's pointer type (float /
    // double / __half / bfloat16) matches group.dtype, and data_ptr() returns the
    // untyped base of the contiguous storage. Using data<float>() for every dtype
    // would misread a half/double buffer.
    std::vector<const void*> input_ptrs;
    input_ptrs.reserve(gpu_inputs.size());
    for (auto& inp : gpu_inputs) {
        input_ptrs.push_back(inp.data_ptr());
    }

    void* output_ptr = output.data_ptr();

    // Launch on default stream
    kernel->launch(input_ptrs, output_ptr, numel, nullptr);

    // Synchronize and move back to original device if needed
    if (orig_device.type == Device::Type::CPU) {
        return output.to(orig_device);
    }
    return output;
#else
    // GPU codegen not compiled in — delegate to the CPU eager fallback.
    return execute_fused_cpu(group, inputs);
#endif
}

auto execute_fused_cpu(const FusionGroup& group,
                       const std::vector<Tensor>& inputs) -> Tensor {
    if (inputs.empty()) {
        throw std::runtime_error("execute_fused_cpu: no inputs provided");
    }
    const int64_t numel = inputs[0].numel();
    for (size_t i = 1; i < inputs.size(); ++i) {
        if (inputs[i].numel() != numel) {
            throw std::runtime_error("execute_fused_cpu: all inputs must have same numel");
        }
    }

    // Enforce the SAME dtype contract as execute_fused (the GPU twin): the fused
    // elementwise codegen supports only Float32/Float64/Float16/BFloat16. Without
    // this guard a non-float group threw on a CUDA/ROCm build (execute_fused) but
    // silently computed here on a CPU-only build (the CODEGEN_AVAILABLE #else
    // routes to this function) — the same group giving an error on one backend
    // and a result on another. Reject identically so behaviour is backend-agnostic.
    if (group.dtype != DType::Float32 && group.dtype != DType::Float64 &&
        group.dtype != DType::Float16 && group.dtype != DType::BFloat16) {
        throw std::runtime_error(
            "execute_fused_cpu: fused elementwise codegen only supports "
            "Float32/Float64/Float16/BFloat16; got " +
            std::string(dtype_name(group.dtype)));
    }

    // Execute operations sequentially using existing eager ops.
    //
    // Every ElemOp value declared in include/tenzor/jit/codegen.hpp must be
    // handled here — the `default:` branch throws rather than silently
    // returning the unmodified input, which would produce silently-wrong
    // output (the bug fixed in Phase P0/Fix-1 of the audit cleanup).
    //
    // Activation ops (LeakyRelu/Elu/Selu/Gelu/Mish/Softplus) don't have
    // Tensor-level free functions, so they dispatch through OpId into the
    // main backend kernel registry. This matches eager behaviour byte for
    // byte and reuses the optimised CPU kernels.
    Tensor result = inputs[0].clone();
    for (const auto& step : group.steps) {
        Tensor a = (step.input_idx < 0) ? result : inputs[step.input_idx];
        auto binary_b = [&]() -> Tensor {
            return (step.second_input_idx < 0) ? result : inputs[step.second_input_idx];
        };
        const double s = step.scalar;

        switch (step.op) {
            // ----- Unary, direct tensor wrappers -------------------------
            case ElemOp::Neg:        result = tenzor::neg(a); break;
            case ElemOp::Abs:        result = tenzor::abs(a); break;
            case ElemOp::Sign:       result = tenzor::sign(a); break;
            case ElemOp::Reciprocal: result = tenzor::reciprocal(a); break;
            case ElemOp::Exp:        result = tenzor::exp(a); break;
            case ElemOp::Log:        result = tenzor::log(a); break;
            case ElemOp::Sqrt:       result = tenzor::sqrt(a); break;
            case ElemOp::Sin:        result = tenzor::sin(a); break;
            case ElemOp::Cos:        result = tenzor::cos(a); break;
            case ElemOp::Tan:        result = tenzor::tan(a); break;
            case ElemOp::Asin:       result = tenzor::asin(a); break;
            case ElemOp::Acos:       result = tenzor::acos(a); break;
            case ElemOp::Atan:       result = tenzor::atan(a); break;
            case ElemOp::Sinh:       result = tenzor::sinh(a); break;
            case ElemOp::Cosh:       result = tenzor::cosh(a); break;
            case ElemOp::Tanh:       result = tenzor::tanh(a); break;
            case ElemOp::Sigmoid:    result = tenzor::sigmoid(a); break;
            case ElemOp::Relu:       result = tenzor::clamp_min(a, 0.0f); break;
            case ElemOp::Erf:        result = tenzor::erf(a); break;
            case ElemOp::Erfc:       result = tenzor::erfc(a); break;
            case ElemOp::Log2:       result = tenzor::log2(a); break;
            case ElemOp::Log10:      result = tenzor::log10(a); break;
            case ElemOp::Log1p:      result = tenzor::log1p(a); break;
            case ElemOp::Exp2:       result = tenzor::exp2(a); break;
            case ElemOp::Expm1:      result = tenzor::expm1(a); break;
            case ElemOp::Floor:      result = tenzor::floor(a); break;
            case ElemOp::Ceil:       result = tenzor::ceil(a); break;
            case ElemOp::Round:      result = tenzor::round(a); break;

            // ----- Activations via OpId dispatch -------------------------
            //
            // Each routes to the registered CPU kernel with the appropriate
            // AttrKey value pulled from step.scalar. Default values match
            // PyTorch's nn.functional defaults (matches the eager autograd
            // path in src/autograd/ops.cpp).
            case ElemOp::LeakyRelu: {
                OpAttributes attrs;
                attrs.set(AttrKey::Alpha, s != 0.0 ? s : 0.01);
                const Tensor in_arr[1] = {a};
                result = tenzor::dispatch(OpId::LeakyReLU,
                                          std::span<const Tensor>{in_arr, 1}, attrs)[0];
                break;
            }
            case ElemOp::Elu: {
                OpAttributes attrs;
                attrs.set(AttrKey::Alpha, s != 0.0 ? s : 1.0);
                const Tensor in_arr[1] = {a};
                result = tenzor::dispatch(OpId::Elu,
                                          std::span<const Tensor>{in_arr, 1}, attrs)[0];
                break;
            }
            case ElemOp::Selu: {
                const Tensor in_arr[1] = {a};
                result = tenzor::dispatch(OpId::Selu,
                                          std::span<const Tensor>{in_arr, 1}, {})[0];
                break;
            }
            case ElemOp::Gelu: {
                const Tensor in_arr[1] = {a};
                result = tenzor::dispatch(OpId::Gelu,
                                          std::span<const Tensor>{in_arr, 1}, {})[0];
                break;
            }
            case ElemOp::Mish: {
                const Tensor in_arr[1] = {a};
                result = tenzor::dispatch(OpId::Mish,
                                          std::span<const Tensor>{in_arr, 1}, {})[0];
                break;
            }
            case ElemOp::Softplus: {
                OpAttributes attrs;
                attrs.set(AttrKey::Beta, s != 0.0 ? s : 1.0);
                attrs.set(AttrKey::Threshold, 20.0);
                const Tensor in_arr[1] = {a};
                result = tenzor::dispatch(OpId::Softplus,
                                          std::span<const Tensor>{in_arr, 1}, attrs)[0];
                break;
            }

            // ----- Binary ops --------------------------------------------
            case ElemOp::Add: { Tensor b = binary_b(); result = tenzor::add(a, b); break; }
            case ElemOp::Sub: { Tensor b = binary_b(); result = tenzor::sub(a, b); break; }
            case ElemOp::Mul: { Tensor b = binary_b(); result = tenzor::mul(a, b); break; }
            case ElemOp::Div: { Tensor b = binary_b(); result = tenzor::div(a, b); break; }
            case ElemOp::Max: { Tensor b = binary_b(); result = tenzor::maximum(a, b); break; }
            case ElemOp::Min: { Tensor b = binary_b(); result = tenzor::minimum(a, b); break; }
            case ElemOp::Fmod: { Tensor b = binary_b(); result = tenzor::fmod(a, b); break; }
            case ElemOp::Pow: {
                // tenzor::float_power implements genuine elementwise
                // tensor-tensor pow with correct negative-base handling.
                // dispatch(OpId::Pow, ...) is NOT the right call here: that
                // kernel's contract (see cpu_kernel_registry.cpp) is a single
                // tensor input plus a scalar AttrKey::Exponent attribute --
                // it silently ignores a second tensor operand and defaults
                // to exponent=2.0, diverging from the genuine pow(a,b) the
                // generated device code above (emit_op's ElemOp::Pow case)
                // actually computes.
                Tensor b = binary_b();
                result = tenzor::float_power(a, b);
                break;
            }

            // ----- Scalar binary ops -------------------------------------
            case ElemOp::AddScalar: result = tenzor::add(a, s); break;
            case ElemOp::MulScalar: result = tenzor::mul(a, s); break;
            // Keep full double precision (these ops take a double scalar and the
            // GPU kernel bakes the scalar at double precision via fmt_double);
            // the old static_cast<float> truncated it and diverged for Float64.
            case ElemOp::PowScalar: result = tenzor::pow(a, s); break;
            case ElemOp::ClampMin:  result = tenzor::clamp_min(a, s); break;
            case ElemOp::ClampMax:  result = tenzor::clamp_max(a, s); break;
        }
        // No `default:` — every ElemOp value must have a `case` arm above.
        // If the enum grows and a new value lands without being handled
        // here, the compiler's -Wswitch warning (treated as error by the
        // project's build flags) will catch it at compile time.
    }
    return result;
}

} // namespace tenzor::jit
