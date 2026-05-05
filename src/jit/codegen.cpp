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
#include <sstream>
#include <stdexcept>
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
    ss << "fusion_" << num_inputs << "_" << static_cast<int>(dtype);
    for (const auto& step : steps) {
        ss << "_" << static_cast<int>(step.op)
           << "i" << step.input_idx
           << "j" << step.second_input_idx;
        if (step.op >= ElemOp::AddScalar) {
            ss << "s" << step.scalar;
        }
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

auto KernelCodegen::emit_op(const ElemStep& step, const std::string& vp) -> std::string {
    // vp = variable prefix. step.input_idx refers to either an input array or
    // the previous result (if -1, means "previous result" = vp + "val")
    auto input = [&](int idx) -> std::string {
        if (idx < 0) return vp + "val";
        return "inp" + std::to_string(idx) + "[i]";
    };

    auto a = input(step.input_idx);
    auto b = input(step.second_input_idx);
    std::string s = std::to_string(step.scalar);

    switch (step.op) {
        // Unary
        case ElemOp::Neg:        return vp + "val = -" + a + ";";
        case ElemOp::Abs:        return vp + "val = fabs(" + a + ");";
        case ElemOp::Sign:       return vp + "val = (" + a + " > 0) - (" + a + " < 0);";
        case ElemOp::Reciprocal: return vp + "val = 1.0f / " + a + ";";
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
        case ElemOp::Sigmoid:    return vp + "val = 1.0f / (1.0f + exp(-" + a + "));";
        case ElemOp::Relu:       return vp + "val = fmax(" + a + ", 0.0f);";
        case ElemOp::LeakyRelu:  return vp + "val = " + a + " > 0 ? " + a + " : " + s + "f * " + a + ";";
        case ElemOp::Elu:        return vp + "val = " + a + " > 0 ? " + a + " : " + s + "f * (exp(" + a + ") - 1.0f);";
        case ElemOp::Selu: {
            std::string lam = "1.0507009873554805f";
            std::string alp = "1.6732632423543772f";
            return vp + "val = " + a + " > 0 ? " + lam + " * " + a + " : " + lam + " * " + alp + " * (exp(" + a + ") - 1.0f);";
        }
        case ElemOp::Gelu:
            return vp + "val = 0.5f * " + a + " * (1.0f + tanh(0.7978845608f * (" + a + " + 0.044715f * " + a + " * " + a + " * " + a + ")));";
        case ElemOp::Mish:
            return vp + "val = " + a + " * tanh(log(1.0f + exp(" + a + ")));";
        case ElemOp::Softplus:   return vp + "val = log(1.0f + exp(" + a + "));";
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
        case ElemOp::AddScalar:  return vp + "val = " + a + " + " + s + "f;";
        case ElemOp::MulScalar:  return vp + "val = " + a + " * " + s + "f;";
        case ElemOp::PowScalar:  return vp + "val = pow(" + a + ", " + s + "f);";
        case ElemOp::ClampMin:   return vp + "val = fmax(" + a + ", " + s + "f);";
        case ElemOp::ClampMax:   return vp + "val = fmin(" + a + ", " + s + "f);";

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
        ss << "        " << emit_op(group.steps[s], "") << "\n";
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
    throw std::runtime_error("GPU codegen not available (no CUDA/ROCm)");
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

    // Compile
    auto kernel = compile(source, kernel_name);
    kernel->source = source;
    kernel->num_inputs = group.num_inputs;
    cache_[group.signature] = kernel;
    compilations_++;

    return kernel;
}

auto KernelCache::compile(const std::string& source, const std::string& kernel_name)
    -> std::shared_ptr<CompiledKernel> {
#if CODEGEN_AVAILABLE
    auto kernel = std::make_shared<CompiledKernel>();
    kernel->name = kernel_name;

#if defined(TENZOR_USE_CUDA)
    // Initialize CUDA driver API and ensure a context is active
    cuInit(0);
    int device_id = 0;
    CUdevice cu_device;
    CU_CHECK(cuDeviceGet(&cu_device, device_id));

    CUcontext cu_context = nullptr;
    cuCtxGetCurrent(&cu_context);
    if (!cu_context) {
        // Use primary context (compatible with CUDA runtime API)
        CU_CHECK(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
        CU_CHECK(cuCtxSetCurrent(cu_context));
    }

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
    throw std::runtime_error("GPU codegen not available (no CUDA/ROCm)");
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
    // Move inputs to GPU if needed (codegen runs on GPU)
    auto gpu_device = Device::cuda(0);
    std::vector<Tensor> gpu_inputs;
    gpu_inputs.reserve(inputs.size());
    for (const auto& inp : inputs) {
        if (inp.device().type == Device::Type::CPU) {
            gpu_inputs.push_back(inp.to(gpu_device));
        } else {
            gpu_inputs.push_back(inp);
        }
    }

    // Get or compile the kernel
    auto kernel = KernelCache::instance().get_or_compile(group);

    // Allocate output on GPU
    std::vector<int64_t> shape(inputs[0].shape().begin(), inputs[0].shape().end());
    Tensor output(shape, group.dtype, gpu_device);

    // Collect input data pointers (now on GPU)
    std::vector<const void*> input_ptrs;
    input_ptrs.reserve(gpu_inputs.size());
    for (auto& inp : gpu_inputs) {
        auto c = inp.contiguous();
        if (group.dtype == DType::Float32) {
            input_ptrs.push_back(c.data<float>());
        } else if (group.dtype == DType::Float64) {
            input_ptrs.push_back(c.data<double>());
        } else {
            input_ptrs.push_back(c.data<float>());
        }
    }

    void* output_ptr = nullptr;
    if (group.dtype == DType::Float32) {
        output_ptr = output.data<float>();
    } else if (group.dtype == DType::Float64) {
        output_ptr = output.data<double>();
    } else {
        output_ptr = output.data<float>();
    }

    // Launch on default stream
    kernel->launch(input_ptrs, output_ptr, numel, nullptr);

    // Synchronize and move back to original device if needed
    if (orig_device.type == Device::Type::CPU) {
        return output.to(orig_device);
    }
    return output;
#else
    // CPU fallback: execute operations sequentially using existing ops
    Tensor result = inputs[0].clone();
    for (const auto& step : group.steps) {
        Tensor a = (step.input_idx < 0) ? result : inputs[step.input_idx];
        switch (step.op) {
            case ElemOp::Exp:    result = tenzor::exp(a); break;
            case ElemOp::Log:    result = tenzor::log(a); break;
            case ElemOp::Sqrt:   result = tenzor::sqrt(a); break;
            case ElemOp::Sin:    result = tenzor::sin(a); break;
            case ElemOp::Cos:    result = tenzor::cos(a); break;
            case ElemOp::Neg:    result = tenzor::neg(a); break;
            case ElemOp::Abs:    result = tenzor::abs(a); break;
            case ElemOp::Sigmoid: result = tenzor::sigmoid(a); break;
            case ElemOp::Relu:   result = tenzor::clamp_min(a, 0.0f); break;
            case ElemOp::Tanh:   result = tenzor::tanh(a); break;
            case ElemOp::Erf:    result = tenzor::erf(a); break;
            case ElemOp::Add: {
                Tensor b = (step.second_input_idx < 0) ? result : inputs[step.second_input_idx];
                result = tenzor::add(a, b); break;
            }
            case ElemOp::Mul: {
                Tensor b = (step.second_input_idx < 0) ? result : inputs[step.second_input_idx];
                result = tenzor::mul(a, b); break;
            }
            case ElemOp::Sub: {
                Tensor b = (step.second_input_idx < 0) ? result : inputs[step.second_input_idx];
                result = tenzor::sub(a, b); break;
            }
            case ElemOp::Div: {
                Tensor b = (step.second_input_idx < 0) ? result : inputs[step.second_input_idx];
                result = tenzor::div(a, b); break;
            }
            case ElemOp::AddScalar: result = tenzor::add(a, static_cast<double>(step.scalar)); break;
            case ElemOp::MulScalar: result = tenzor::mul(a, static_cast<double>(step.scalar)); break;
            default: break;  // Unsupported in CPU fallback
        }
    }
    return result;
#endif
}

} // namespace tenzor::jit
