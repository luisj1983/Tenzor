/**
 * @file codegen.cpp
 * @brief Runtime GPU kernel generation via NVRTC/HIPRTC
 *
 * Generates, compiles, and caches fused element-wise GPU kernels at runtime.
 */

#include "tenzor/jit/codegen.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"  // CPU fallback below uses tenzor::{exp,sin,add,...}
#include "tenzor/core/dtype.hpp"
#include "tenzor/backend/fast_dispatch.hpp"  // CPU fallback dispatches activations via OpId
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
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

#if defined(TENZOR_USE_CUDA)
#include <nvrtc.h>
#include <cuda.h>
#define CODEGEN_AVAILABLE 1
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
#elif defined(TENZOR_USE_ROCM)
#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>
#define CODEGEN_AVAILABLE 1
#define NVRTC_CHECK(call) do { \
    hiprtcResult res = call; \
    if (res != HIPRTC_SUCCESS) { \
        throw std::runtime_error(std::string("HIPRTC error: ") + hiprtcGetErrorString(res)); \
    } \
} while(0)
#define CU_CHECK(call) do { \
    hipError_t res = call; \
    if (res != hipSuccess) { \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(res)); \
    } \
} while(0)
#else
#define CODEGEN_AVAILABLE 0
#endif

namespace tenzor::jit {

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
        case DType::Float32: return "float";
        case DType::Float64: return "double";
        case DType::Int32:   return "int";
        case DType::Int64:   return "long long";
        default: return "float";  // Default to float
    }
}

auto KernelCodegen::emit_op(const ElemStep& step, const std::string& vp,
                            DType dtype) -> std::string {
    // vp = variable prefix. step.input_idx refers to either an input array or
    // the previous result (if -1, means "previous result" = vp + "val")
    auto input = [&](int idx) -> std::string {
        if (idx < 0) return vp + "val";
        return "inp" + std::to_string(idx) + "[i]";
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
    std::string s = fmt_double(step.scalar);        // scalar operand literal

    // Activation scalars default to their PyTorch nn.functional values when the
    // step records 0.0 (no explicit slope/alpha). Mirror execute_fused_cpu
    // (LeakyRelu -> 0.01, Elu -> 1.0); without this the negative branch would
    // collapse to 0 (LeakyRelu) or use a 0 coefficient (Elu), diverging from
    // the CPU fallback for the same fused group.
    auto activation_scalar = [&](double dflt) -> std::string {
        return fmt_double(step.scalar != 0.0 ? step.scalar : dflt);
    };

    switch (step.op) {
        // Unary
        case ElemOp::Neg:        return vp + "val = -" + a + ";";
        case ElemOp::Abs:        return vp + "val = fabs(" + a + ");";
        case ElemOp::Sign:       return vp + "val = (" + a + " > 0) - (" + a + " < 0);";
        case ElemOp::Reciprocal: return vp + "val = " + ONE + " / " + a + ";";
        case ElemOp::Exp:        return vp + "val = exp(" + a + ");";
        case ElemOp::Log:        return vp + "val = log(" + a + ");";
        case ElemOp::Sqrt:       return vp + "val = sqrt(" + a + ");";
        case ElemOp::Pow:        return vp + "val = pow(" + a + ", " + b + ");";
        case ElemOp::Sin:        return vp + "val = sin(" + a + ");";
        case ElemOp::Cos:        return vp + "val = cos(" + a + ");";
        case ElemOp::Tan:        return vp + "val = tan(" + a + ");";
        case ElemOp::Asin:       return vp + "val = asin(" + a + ");";
        case ElemOp::Acos:       return vp + "val = acos(" + a + ");";
        case ElemOp::Atan:       return vp + "val = atan(" + a + ");";
        case ElemOp::Sinh:       return vp + "val = sinh(" + a + ");";
        case ElemOp::Cosh:       return vp + "val = cosh(" + a + ");";
        case ElemOp::Tanh:       return vp + "val = tanh(" + a + ");";
        case ElemOp::Sigmoid:    return vp + "val = " + ONE + " / (" + ONE + " + exp(-" + a + "));";
        case ElemOp::Relu:       return vp + "val = fmax(" + a + ", " + ZERO + ");";
        case ElemOp::LeakyRelu:  return vp + "val = " + a + " > 0 ? " + a + " : " + activation_scalar(0.01) + F + " * " + a + ";";
        case ElemOp::Elu:        return vp + "val = " + a + " > 0 ? " + a + " : " + activation_scalar(1.0) + F + " * (exp(" + a + ") - " + ONE + ");";
        case ElemOp::Selu: {
            std::string lam = "1.0507009873554805" + F;
            std::string alp = "1.6732632423543772" + F;
            return vp + "val = " + a + " > 0 ? " + lam + " * " + a + " : " + lam + " * " + alp + " * (exp(" + a + ") - " + ONE + ");";
        }
        case ElemOp::Gelu:
            // Exact erf GELU 0.5*x*(1 + erf(x / sqrt(2))) to match the CPU/eager
            // kernel (cpu::gelu_kernel, approximate='none'); 1/sqrt(2) =
            // 0.7071067811865476. erf is available in NVRTC/HIPRTC device math.
            // The older tanh approximation diverged from CPU by ~1e-3, breaking
            // cross-backend parity / gradcheck for any fused group with GELU.
            return vp + "val = 0.5" + F + " * " + a + " * (" + ONE + " + erf(" + a + " * 0.7071067811865476" + F + "));";
        case ElemOp::Mish:
            return vp + "val = " + a + " * tanh(log(" + ONE + " + exp(" + a + ")));";
        case ElemOp::Softplus:   return vp + "val = log(" + ONE + " + exp(" + a + "));";
        case ElemOp::Erf:        return vp + "val = erf(" + a + ");";
        case ElemOp::Erfc:       return vp + "val = erfc(" + a + ");";
        case ElemOp::Log2:       return vp + "val = log2(" + a + ");";
        case ElemOp::Log10:      return vp + "val = log10(" + a + ");";
        case ElemOp::Log1p:      return vp + "val = log1p(" + a + ");";
        case ElemOp::Exp2:       return vp + "val = exp2(" + a + ");";
        case ElemOp::Expm1:      return vp + "val = expm1(" + a + ");";
        case ElemOp::Floor:      return vp + "val = floor(" + a + ");";
        case ElemOp::Ceil:       return vp + "val = ceil(" + a + ");";
        case ElemOp::Round:      return vp + "val = round(" + a + ");";

        // Binary
        case ElemOp::Add:  return vp + "val = " + a + " + " + b + ";";
        case ElemOp::Sub:  return vp + "val = " + a + " - " + b + ";";
        case ElemOp::Mul:  return vp + "val = " + a + " * " + b + ";";
        case ElemOp::Div:  return vp + "val = " + a + " / " + b + ";";
        case ElemOp::Max:  return vp + "val = fmax(" + a + ", " + b + ");";
        case ElemOp::Min:  return vp + "val = fmin(" + a + ", " + b + ");";
        case ElemOp::Fmod: return vp + "val = fmod(" + a + ", " + b + ");";

        // Scalar ops
        case ElemOp::AddScalar:  return vp + "val = " + a + " + " + s + F + ";";
        case ElemOp::MulScalar:  return vp + "val = " + a + " * " + s + F + ";";
        case ElemOp::PowScalar:  return vp + "val = pow(" + a + ", " + s + F + ");";
        case ElemOp::ClampMin:   return vp + "val = fmax(" + a + ", " + s + F + ");";
        case ElemOp::ClampMax:   return vp + "val = fmin(" + a + ", " + s + F + ");";

        default: return vp + "val = " + a + "; // unknown op";
    }
}

auto KernelCodegen::generate(const FusionGroup& group) -> std::string {
    auto ctype = dtype_to_cuda_type(group.dtype);
    std::string kernel_name = "fused_elementwise_" +
        std::to_string(group.steps.size()) + "_ops";

    std::ostringstream ss;

    // Header
    ss << "extern \"C\" __global__ void " << kernel_name << "(\n";

    // Input pointers
    for (int i = 0; i < group.num_inputs; ++i) {
        ss << "    const " << ctype << "* __restrict__ inp" << i << ",\n";
    }
    ss << "    " << ctype << "* __restrict__ out,\n";
    ss << "    long long numel\n";
    ss << ") {\n";

    // Grid-stride loop
    ss << "    for (long long i = blockIdx.x * blockDim.x + threadIdx.x;\n";
    ss << "         i < numel;\n";
    ss << "         i += blockDim.x * gridDim.x) {\n";

    // Emit operations
    ss << "        " << ctype << " val;\n";
    for (size_t s = 0; s < group.steps.size(); ++s) {
        ss << "        " << emit_op(group.steps[s], "", group.dtype) << "\n";
    }

    // Store result
    ss << "        out[i] = val;\n";
    ss << "    }\n";
    ss << "}\n";

    return ss.str();
}

// ============================================================================
// Compiled Kernel
// ============================================================================

CompiledKernel::~CompiledKernel() {
#if CODEGEN_AVAILABLE
#if defined(TENZOR_USE_CUDA)
    if (module) cuModuleUnload(static_cast<CUmodule>(module));
#elif defined(TENZOR_USE_ROCM)
    if (module) hipModuleUnload(static_cast<hipModule_t>(module));
#endif
#endif
}

auto CompiledKernel::launch(const std::vector<const void*>& input_ptrs,
                            void* output_ptr, int64_t numel, void* stream) -> void {
#if CODEGEN_AVAILABLE
    if (!function) {
        throw std::runtime_error("CompiledKernel::launch: kernel not compiled");
    }

    // Adaptive block size selection based on problem size.
    // Small tensors benefit from fewer threads (less launch overhead),
    // while large tensors need more threads for full GPU occupancy.
    int block_size;
    if (numel <= 256) block_size = 64;
    else if (numel <= 4096) block_size = 128;
    else if (numel <= 65536) block_size = 256;
    else block_size = 512;

    int grid_size = static_cast<int>((numel + block_size - 1) / block_size);
    if (grid_size > 65535) grid_size = 65535;

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

#if defined(TENZOR_USE_CUDA)
    CU_CHECK(cuLaunchKernel(
        static_cast<CUfunction>(function),
        grid_size, 1, 1,    // grid
        block_size, 1, 1,    // block
        0,                    // shared mem
        static_cast<CUstream>(stream),
        args.data(),
        nullptr
    ));
#elif defined(TENZOR_USE_ROCM)
    CU_CHECK(hipModuleLaunchKernel(
        static_cast<hipFunction_t>(function),
        grid_size, 1, 1,
        block_size, 1, 1,
        0,
        static_cast<hipStream_t>(stream),
        args.data(),
        nullptr
    ));
#endif
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
#if defined(TENZOR_USE_CUDA)
    CU_CHECK(cuLaunchKernel(
        static_cast<CUfunction>(function),
        grid_size, 1, 1,
        block_size, 1, 1,
        shared_bytes,
        static_cast<CUstream>(stream),
        args.data(),
        nullptr));
#elif defined(TENZOR_USE_ROCM)
    CU_CHECK(hipModuleLaunchKernel(
        static_cast<hipFunction_t>(function),
        grid_size, 1, 1,
        block_size, 1, 1,
        shared_bytes,
        static_cast<hipStream_t>(stream),
        args.data(),
        nullptr));
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
    std::string kernel_name = "fused_elementwise_" +
        std::to_string(group.steps.size()) + "_ops";

    // Compile for the group's target GPU (its context + compute arch), not a
    // hardcoded device 0.
    auto kernel = compile(source, kernel_name, group.device.index);
    kernel->source = source;
    kernel->num_inputs = group.num_inputs;
    cache_[group.signature] = kernel;
    compilations_++;

    return kernel;
}

auto KernelCache::get_or_compile_source(const std::string& signature,
                                        const std::string& source,
                                        const std::string& kernel_name)
    -> std::shared_ptr<CompiledKernel> {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_.find(signature);
    if (it != cache_.end()) {
        cache_hits_++;
        return it->second;
    }

    auto kernel = compile(source, kernel_name);
    if (kernel) {
        kernel->source = source;
        cache_[signature] = kernel;
        compilations_++;
    }
    return kernel;
}

auto KernelCache::compile(const std::string& source, const std::string& kernel_name,
                          int device_index)
    -> std::shared_ptr<CompiledKernel> {
#if CODEGEN_AVAILABLE
    auto kernel = std::make_shared<CompiledKernel>();
    kernel->name = kernel_name;

#if defined(TENZOR_USE_CUDA)
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
    CUcontext cu_context = nullptr;
    {
        static std::mutex ctx_mutex;
        static std::unordered_map<int, CUcontext> primary_ctxs;
        std::lock_guard<std::mutex> ctx_lock(ctx_mutex);
        auto it = primary_ctxs.find(device_id);
        if (it != primary_ctxs.end()) {
            cu_context = it->second;
        } else {
            CU_CHECK(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
            primary_ctxs.emplace(device_id, cu_context);
        }
    }
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

    const char* opts[] = {
        arch_flag.c_str(),
        "--std=c++17",
        "-default-device"
    };
    nvrtcResult compile_result = nvrtcCompileProgram(prog, 3, opts);

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

#elif defined(TENZOR_USE_ROCM)
    // Compile with HIPRTC
    hiprtcProgram prog;
    NVRTC_CHECK(hiprtcCreateProgram(&prog, source.c_str(), "fused_kernel.hip",
                                    0, nullptr, nullptr));

    const char* opts[] = {"--std=c++17"};
    hiprtcResult compile_result = hiprtcCompileProgram(prog, 1, opts);

    if (compile_result != HIPRTC_SUCCESS) {
        size_t log_size;
        hiprtcGetProgramLogSize(prog, &log_size);
        std::string log(log_size, '\0');
        hiprtcGetProgramLog(prog, log.data());
        hiprtcDestroyProgram(&prog);
        throw std::runtime_error("HIPRTC compilation failed:\n" + log + "\nSource:\n" + source);
    }

    size_t code_size;
    NVRTC_CHECK(hiprtcGetCodeSize(prog, &code_size));
    std::string code(code_size, '\0');
    NVRTC_CHECK(hiprtcGetCode(prog, code.data()));
    hiprtcDestroyProgram(&prog);

    hipModule_t hip_module;
    CU_CHECK(hipModuleLoadData(&hip_module, code.data()));
    kernel->module = hip_module;

    hipFunction_t hip_func;
    CU_CHECK(hipModuleGetFunction(&hip_func, hip_module, kernel_name.c_str()));
    kernel->function = hip_func;
#endif

    return kernel;
#else
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
    // The generated kernel only implements Float32/Float64 math. Reject any
    // other dtype up front — before any device transfer, kernel compilation, or
    // output allocation — instead of reinterpreting its bytes as float (which
    // silently produced garbage / OOB for Int32/Int64 whose element size
    // differs). Validating here (rather than after get_or_compile/allocation)
    // also avoids compiling and caching a dead NVRTC/HIPRTC kernel and wasting
    // CPU->GPU transfers for a dtype that always throws.
    if (group.dtype != DType::Float32 && group.dtype != DType::Float64) {
        throw std::runtime_error(
            "KernelCodegen::execute_fused: fused GPU codegen only supports "
            "Float32/Float64; got " + std::string(dtype_name(group.dtype)) +
            " — route this dtype through the eager fallback");
    }

    // Move inputs to GPU if needed (codegen runs on GPU). CODEGEN_AVAILABLE is
    // set for both CUDA and ROCm builds, so derive the concrete GPU device from
    // the active backend — directing tensors to a CUDA device on a ROCm build
    // fails device validation / lands data on the wrong backend.
#if defined(TENZOR_USE_CUDA)
    auto gpu_device = Device::cuda(0);
#elif defined(TENZOR_USE_ROCM)
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
    std::vector<Tensor> gpu_inputs;
    gpu_inputs.reserve(inputs.size());
    for (const auto& inp : inputs) {
        if (inp.device().type == Device::Type::CPU) {
            gpu_inputs.push_back(inp.to(gpu_device));
        } else {
            gpu_inputs.push_back(inp);
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
    std::vector<const void*> input_ptrs;
    input_ptrs.reserve(gpu_inputs.size());
    for (auto& inp : gpu_inputs) {
        if (group.dtype == DType::Float32) {
            input_ptrs.push_back(inp.data<float>());
        } else {  // Float64 (guarded above)
            input_ptrs.push_back(inp.data<double>());
        }
    }

    void* output_ptr = (group.dtype == DType::Float32)
                           ? static_cast<void*>(output.data<float>())
                           : static_cast<void*>(output.data<double>());

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
                // No tensor-tensor pow free function; route through dispatch.
                Tensor b = binary_b();
                const Tensor in_arr[2] = {a, b};
                result = tenzor::dispatch(OpId::Pow,
                                          std::span<const Tensor>{in_arr, 2}, {})[0];
                break;
            }

            // ----- Scalar binary ops -------------------------------------
            case ElemOp::AddScalar: result = tenzor::add(a, s); break;
            case ElemOp::MulScalar: result = tenzor::mul(a, s); break;
            case ElemOp::PowScalar: result = tenzor::pow(a, static_cast<float>(s)); break;
            case ElemOp::ClampMin:  result = tenzor::clamp_min(a, static_cast<float>(s)); break;
            case ElemOp::ClampMax:  result = tenzor::clamp_max(a, static_cast<float>(s)); break;
        }
        // No `default:` — every ElemOp value must have a `case` arm above.
        // If the enum grows and a new value lands without being handled
        // here, the compiler's -Wswitch warning (treated as error by the
        // project's build flags) will catch it at compile time.
    }
    return result;
}

} // namespace tenzor::jit
