/**
 * @file codegen.hpp
 * @brief Runtime GPU kernel generation via NVRTC/HIPRTC
 *
 * Provides an element-wise kernel fusion engine that generates and compiles
 * GPU kernels at runtime. Given a chain of element-wise operations from the
 * JIT IR, it emits a single fused CUDA/HIP kernel source, compiles it via
 * NVRTC/HIPRTC, and caches the result for reuse.
 *
 * This eliminates kernel launch overhead and intermediate memory allocations
 * for common patterns like: x -> relu -> mul(scalar) -> add(bias) -> sigmoid
 *
 * Architecture:
 *   1. FusionGroup identifies chains of fusible element-wise ops
 *   2. KernelCodegen emits CUDA/HIP C++ source for the fused kernel
 *   3. KernelCache compiles and caches kernels keyed by operation signature
 *   4. CodegenPass integrates with the JIT compiler pipeline
 */

#pragma once

#include "../core/tensor.hpp"
#include "../core/dtype.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <functional>

namespace tenzor::jit {

// ============================================================================
// Fusion kind categories (shared with pattern_matcher and extended_codegen)
// ============================================================================

/**
 * @brief Categories of kernel fusion patterns.
 *
 * Used by PatternMatcher to classify matched patterns and by
 * ExtendedKernelCodegen to dispatch to the appropriate generator.
 */
enum class FusionKind : uint8_t {
    ElementWise,      ///< Existing element-wise chain fusion
    Reduction,        ///< Reduction with pre/post element-wise ops
    GemmEpilogue,     ///< MatMul/Linear + Add(bias) + activation
    Softmax,          ///< sub(max) -> exp -> sum -> div pattern
    LayerNorm,        ///< mean -> sub -> var -> rsqrt -> mul -> add
    RMSNorm,          ///< square -> mean -> rsqrt -> mul
    SmallMLP,         ///< Linear -> activation -> Linear (hidden <= 4096)
    SwiGLU,           ///< Linear -> Slice (split) -> Sigmoid -> Mul (gate) -> Linear
    GeluVariant,      ///< GELU approximation: tanh-based (Pow->Mul->Add->Tanh) or erf-based
    RotaryEmbedding,  ///< cos/sin rotation: Slice -> Mul(cos) -> Slice -> Mul(sin) -> Sub/Add
};

// ============================================================================
// Element-wise operation representation for codegen
// ============================================================================

/**
 * @brief Supported element-wise operations for code generation.
 */
enum class ElemOp : uint8_t {
    // Unary
    Neg, Abs, Sign, Reciprocal,
    Exp, Log, Sqrt, Pow,
    Sin, Cos, Tan,
    Asin, Acos, Atan,
    Sinh, Cosh, Tanh,
    Sigmoid,
    Relu, LeakyRelu, Elu, Selu, Gelu, Mish, Softplus,
    Erf, Erfc,
    Log2, Log10, Log1p,
    Exp2, Expm1,
    Floor, Ceil, Round,

    // Binary
    Add, Sub, Mul, Div,
    Max, Min,
    Fmod,

    // Scalar binary (second operand is a constant)
    AddScalar, MulScalar, PowScalar, ClampMin, ClampMax,
};

/**
 * @brief Single step in a fused element-wise computation.
 */
struct ElemStep {
    ElemOp op;
    int input_idx;          ///< Index of input variable (for binary: first operand)
    int second_input_idx;   ///< Index of second input (for binary ops), -1 if N/A
    double scalar;          ///< Scalar constant for *Scalar ops
};

/**
 * @brief A group of element-wise operations to fuse into one kernel.
 */
struct FusionGroup {
    std::vector<ElemStep> steps;
    int num_inputs;         ///< Number of distinct input tensors
    DType dtype;            ///< Data type for computation
    std::string signature;  ///< Unique signature for cache lookup

    /**
     * @brief Compute a cache key from the operation sequence.
     */
    auto compute_signature() -> std::string;
};

// ============================================================================
// Kernel code generation
// ============================================================================

/**
 * @brief Generates CUDA/HIP kernel source code from a FusionGroup.
 */
class KernelCodegen {
public:
    /**
     * @brief Generate kernel source code for a fusion group.
     *
     * Emits a grid-stride loop kernel with all fused operations inlined.
     * Supports Float32, Float64, Float16, BFloat16.
     *
     * @param group The fusion group to generate code for
     * @return CUDA/HIP source code as a string
     */
    static auto generate(const FusionGroup& group) -> std::string;

private:
    static auto emit_op(const ElemStep& step, const std::string& var_prefix,
                        DType dtype = DType::Float32) -> std::string;
    static auto dtype_to_cuda_type(DType dtype) -> std::string;
};

// ============================================================================
// Compiled kernel handle
// ============================================================================

/**
 * @brief Handle to a compiled GPU kernel.
 */
struct CompiledKernel {
    void* module{nullptr};     ///< CUmodule or hipModule_t
    void* function{nullptr};   ///< CUfunction or hipFunction_t
    std::string name;          ///< Kernel function name
    std::string source;        ///< Original source (for debugging)
    int num_inputs{0};

    ~CompiledKernel();

    /**
     * @brief Launch the compiled kernel.
     *
     * @param inputs Input tensors
     * @param output Pre-allocated output tensor
     * @param numel Number of elements
     * @param stream GPU stream
     */
    auto launch(const std::vector<const void*>& input_ptrs,
                void* output_ptr, int64_t numel, void* stream) -> void;
};

// ============================================================================
// Kernel compilation and caching
// ============================================================================

/**
 * @brief Compiles and caches GPU kernels generated by KernelCodegen.
 *
 * Thread-safe. Kernels are cached by their signature string so identical
 * fusion patterns are compiled only once.
 */
class KernelCache {
public:
    static auto instance() -> KernelCache&;

    /**
     * @brief Get or compile a kernel for the given fusion group.
     *
     * @param group Fusion group describing the operations
     * @return Shared pointer to the compiled kernel
     */
    auto get_or_compile(const FusionGroup& group) -> std::shared_ptr<CompiledKernel>;

    /**
     * @brief Get or compile a kernel from pre-generated source, keyed by an
     * explicit signature.
     *
     * Used by the extended-fusion path, which generates its own (non
     * element-wise) kernel source and supplies the entry-point name. Without
     * this the caller would have to route through get_or_compile(FusionGroup),
     * which regenerates an unrelated element-wise kernel and discards the
     * provided source.
     *
     * @param signature  Cache key uniquely identifying this kernel
     * @param source     Complete kernel source to compile
     * @param kernel_name Entry-point (extern "C" __global__) symbol name
     * @return Shared pointer to the compiled kernel
     */
    auto get_or_compile_source(const std::string& signature,
                               const std::string& source,
                               const std::string& kernel_name)
        -> std::shared_ptr<CompiledKernel>;

    /**
     * @brief Clear all cached kernels.
     */
    auto clear() -> void;

    /**
     * @brief Get cache statistics.
     */
    auto num_cached() const -> size_t;
    auto num_compilations() const -> size_t { return compilations_; }
    auto num_cache_hits() const -> size_t { return cache_hits_; }

private:
    KernelCache() = default;

    auto compile(const std::string& source, const std::string& kernel_name)
        -> std::shared_ptr<CompiledKernel>;

    std::unordered_map<std::string, std::shared_ptr<CompiledKernel>> cache_;
    mutable std::mutex mutex_;
    size_t compilations_{0};
    size_t cache_hits_{0};
};

// ============================================================================
// High-level fusion API
// ============================================================================

/**
 * @brief Execute a fused element-wise operation on GPU tensors.
 *
 * Takes a FusionGroup and input tensors, generates/compiles/caches the
 * kernel, and launches it.
 *
 * @param group Fusion group describing the operations
 * @param inputs Input tensors
 * @return Output tensor
 */
auto execute_fused(const FusionGroup& group,
                   const std::vector<Tensor>& inputs) -> Tensor;

/**
 * @brief Execute a FusionGroup via the CPU eager fallback path.
 *
 * Runs each ElemStep sequentially via `tenzor::*` / OpId dispatch into the
 * main backend kernel registry. Unlike `execute_fused`, this never invokes
 * the GPU code generator — it always runs on CPU semantics, regardless of
 * whether CUDA/HIP support is compiled in. Used by `execute_fused` as the
 * fallback when GPU codegen is unavailable or fails, and by tests that
 * need to validate the fallback path explicitly.
 *
 * @param group Fusion group describing the operations
 * @param inputs Input tensors (any device; copied via existing tensor ops)
 * @return Output tensor with same shape as inputs[0]
 * @throws std::runtime_error if `inputs` is empty or numel mismatches
 */
auto execute_fused_cpu(const FusionGroup& group,
                       const std::vector<Tensor>& inputs) -> Tensor;

/**
 * @brief Build a FusionGroup from a simple lambda-style description.
 *
 * Convenience for building fusion groups programmatically.
 *
 * @code
 * auto group = build_fusion({
 *     {ElemOp::Mul, 0, 1, 0.0},        // tmp = input0 * input1
 *     {ElemOp::AddScalar, -1, -1, 1.0}, // tmp = tmp + 1.0
 *     {ElemOp::Sigmoid, -1, -1, 0.0},   // out = sigmoid(tmp)
 * }, 2, DType::Float32);
 * @endcode
 */
auto build_fusion(std::vector<ElemStep> steps, int num_inputs, DType dtype)
    -> FusionGroup;

} // namespace tenzor::jit
