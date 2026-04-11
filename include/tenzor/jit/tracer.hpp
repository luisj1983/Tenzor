/**
 * @file tracer.hpp
 * @brief JIT operation tracing system for recording computation graphs
 *
 * Provides trace mode execution that records operations during forward pass
 * to build an intermediate representation (IR) graph. This is the primary
 * JIT compilation mode for Tenzor.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include "../core/tensor.hpp"
#include "../autograd/variable.hpp"
#include "../nn/module.hpp"
#include "graph.hpp"

namespace tenzor {
namespace jit {

/**
 * @brief Operation types supported by the tracer.
 *
 * Each type corresponds to a kernel operation that can be traced
 * and compiled into the IR graph.
 */
enum class OpType {
    // Arithmetic operations
    Add,
    Sub,
    Mul,
    Div,

    // Matrix operations
    MatMul,
    Bmm,

    // Activations
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LogSoftmax,

    // Pooling
    MaxPool2d,
    AvgPool2d,
    AdaptiveAvgPool2d,

    // Convolution
    Conv2d,

    // Normalization
    BatchNorm2d,
    LayerNorm,

    // Shape operations
    Reshape,
    Transpose,
    Permute,
    Squeeze,
    Unsqueeze,
    Flatten,

    // Reductions
    Sum,
    Mean,
    Max,
    Min,

    // Element-wise
    Exp,
    Log,
    Sqrt,
    Pow,
    Abs,
    Neg,
    Clamp,

    // Indexing
    Slice,
    Cat,

    // Other
    Dropout,
    Linear,
    Embedding,

    // Activations (extended)
    GELU,

    // Linear algebra
    Det,
    Inv,
    Solve,
    Cholesky,
    Svd,
    Qr,
    Eigh,
    Eigvalsh,
    Norm,
    Slogdet,

    // Constants
    Constant,
    Input,
    Output,

    // Fused operations
    FlashAttention,   ///< Fused multi-head attention (Q*K^T -> scale -> softmax -> *V)
    FusedFFN,         ///< Fused feed-forward network (Linear -> GELU/ReLU -> Linear)
    ResidualAdd,      ///< Residual connection marker (x + sublayer(x))

    // Shape guard (for dynamic shape support)
    ShapeGuard,       ///< Runtime shape check that triggers re-trace on mismatch
    GuardNode,        ///< Data-dependent branch guard; failure triggers retrace

    // Memory management pseudo-ops
    SwapOut,          ///< GPU -> CPU async transfer for memory pressure relief
    SwapIn,           ///< CPU -> GPU async prefetch before reuse

    // Control flow
    If,     ///< Conditional branch: cond → then_branch / else_branch subgraphs
    Loop    ///< Loop: (max_iter, cond, carried...) → body subgraph → (carried...)
};

/**
 * @brief Converts OpType to string for debugging and serialization.
 *
 * @param type Operation type
 * @return String representation
 */
auto op_type_to_string(OpType type) -> std::string;

/**
 * @brief Converts string to OpType for deserialization.
 *
 * @param str String representation
 * @return Operation type
 * @throws std::runtime_error if string is not recognized
 */
auto string_to_op_type(const std::string& str) -> OpType;

/**
 * @brief Recorded operation during tracing.
 *
 * Captures all information needed to reconstruct and optimize
 * an operation in the IR graph.
 */
struct TracedOp {
    OpType type;                                     ///< Operation type
    std::vector<std::string> inputs;                 ///< Input tensor IDs
    std::vector<std::string> outputs;                ///< Output tensor IDs
    std::unordered_map<std::string, float> attrs;    ///< Float attributes (e.g., dropout rate)
    std::unordered_map<std::string, int64_t> int_attrs;  ///< Int attributes (e.g., dimensions)
    std::unordered_map<std::string, std::vector<int64_t>> vec_attrs;  ///< Vector attributes (e.g., shape)
    std::unordered_map<std::string, bool> bool_attrs;  ///< Boolean attributes (e.g., bias)
    std::unordered_map<std::string, Tensor> tensor_attrs;  ///< Tensor constants (e.g., weights)

    /**
     * @brief Construct traced operation.
     *
     * @param op_type Type of operation
     * @param input_ids Input tensor identifiers
     * @param output_ids Output tensor identifiers
     */
    TracedOp(OpType op_type, std::vector<std::string> input_ids, std::vector<std::string> output_ids)
        : type(op_type), inputs(std::move(input_ids)), outputs(std::move(output_ids)) {}
};

/**
 * @brief Tensor metadata tracked during tracing.
 *
 * Records shape, dtype, and device information for each tensor
 * in the computation graph.
 */
struct TensorInfo {
    std::vector<int64_t> shape;  ///< Tensor shape
    DType dtype;                 ///< Data type
    Device device;               ///< Device location
    bool is_param{false};        ///< True if this is a model parameter

    TensorInfo() = default;
    TensorInfo(std::vector<int64_t> s, DType dt, Device dev, bool param = false)
        : shape(std::move(s)), dtype(dt), device(dev), is_param(param) {}
};

/**
 * @brief Tracing context that records operations during execution.
 *
 * The Tracer maintains a global state while tracing is active,
 * recording all operations performed on tensors and variables.
 * It builds a complete IR graph that can be optimized and serialized.
 *
 * Thread-local storage ensures thread safety for multi-threaded tracing.
 *
 * @code
 * Tracer tracer;
 * tracer.start_trace();
 *
 * // Operations are automatically recorded
 * Variable output = model(input);
 *
 * auto graph = tracer.end_trace({input}, {output});
 * @endcode
 */
class Tracer {
public:
    /**
     * @brief Construct tracer instance.
     */
    Tracer() = default;

    /**
     * @brief Start recording operations.
     *
     * Activates tracing mode globally. All tensor/variable operations
     * will be recorded until end_trace() is called.
     */
    auto start_trace() -> void;

    /**
     * @brief Stop recording and build IR graph.
     *
     * Finalizes the trace and constructs an optimized IR graph.
     *
     * @param inputs Input tensors/variables that started the trace
     * @param outputs Output tensors/variables produced by the trace
     * @return IR graph representing the computation
     */
    auto end_trace(const std::vector<Variable>& inputs,
                   const std::vector<Variable>& outputs) -> std::shared_ptr<Graph>;

    /**
     * @brief Record an operation during tracing.
     *
     * Called automatically by overloaded operators and functions
     * when tracing is active.
     *
     * @param op Traced operation to record
     */
    auto record_op(TracedOp op) -> void;

    /**
     * @brief Register a tensor in the trace.
     *
     * Assigns a unique ID to the tensor and records its metadata.
     *
     * @param tensor Tensor to register
     * @return Unique tensor ID
     */
    auto register_tensor(const Tensor& tensor) -> std::string;

    /**
     * @brief Register a variable in the trace.
     *
     * @param var Variable to register
     * @return Unique tensor ID
     */
    auto register_tensor(const Variable& var) -> std::string;

    /**
     * @brief Get metadata for a tensor ID.
     *
     * @param tensor_id Tensor identifier
     * @return Tensor metadata
     */
    auto get_tensor_info(const std::string& tensor_id) const -> const TensorInfo&;

    /**
     * @brief Check if tracing is currently active.
     *
     * @return true if operations are being recorded
     */
    auto is_tracing() const -> bool { return tracing_; }

    /**
     * @brief Trace a conditional branch (if/else).
     *
     * Records both branches as subgraphs in the IR. At execution time,
     * the condition tensor determines which branch to evaluate.
     *
     * @param condition Boolean scalar tensor (evaluated at runtime)
     * @param then_fn Function executed when condition is true
     * @param else_fn Function executed when condition is false
     * @param inputs Input variables available to both branches
     * @return Output variables from the selected branch
     */
    auto trace_if(const Tensor& condition,
                  std::function<std::vector<Variable>(const std::vector<Variable>&)> then_fn,
                  std::function<std::vector<Variable>(const std::vector<Variable>&)> else_fn,
                  const std::vector<Variable>& inputs) -> std::vector<Variable>;

    /**
     * @brief Trace a loop with carried state.
     *
     * Records the loop body as a subgraph. The loop executes up to max_iter
     * times, with carried variables passed between iterations.
     *
     * @param max_iter Maximum number of iterations
     * @param cond_fn Function returning bool tensor (continue condition)
     * @param body_fn Function computing one iteration (takes carried, returns updated carried)
     * @param carried Initial carried state variables
     * @return Final carried state after loop completes
     */
    auto trace_loop(int64_t max_iter,
                    std::function<Tensor(const std::vector<Variable>&)> cond_fn,
                    std::function<std::vector<Variable>(const std::vector<Variable>&)> body_fn,
                    const std::vector<Variable>& carried) -> std::vector<Variable>;

    /**
     * @brief Clear all recorded operations and reset state.
     */
    auto clear() -> void;

    /**
     * @brief Get global tracer instance (thread-local).
     *
     * @return Reference to thread-local tracer
     */
    static auto get_instance() -> Tracer&;

private:
    bool tracing_{false};                                       ///< Tracing active flag
    std::vector<TracedOp> ops_;                                 ///< Recorded operations
    std::unordered_map<std::string, TensorInfo> tensor_info_;   ///< Tensor metadata
    std::unordered_map<void*, std::string> tensor_id_map_;      ///< Pointer to ID mapping
    int64_t next_tensor_id_{0};                                 ///< Counter for unique IDs

    /**
     * @brief Generate unique tensor ID.
     *
     * @return Unique identifier string
     */
    auto generate_tensor_id() -> std::string;
};

/**
 * @brief RAII guard for tracing scope.
 *
 * Automatically starts tracing on construction and stops on destruction.
 * Ensures tracing is properly cleaned up even if exceptions occur.
 *
 * @code
 * {
 *     TracingGuard guard;
 *     Variable output = model(input);
 *     auto graph = guard.get_graph({input}, {output});
 * }
 * @endcode
 */
class TracingGuard {
public:
    /**
     * @brief Start tracing.
     */
    TracingGuard();

    /**
     * @brief Stop tracing and clean up.
     */
    ~TracingGuard();

    /**
     * @brief Get traced graph.
     *
     * @param inputs Input variables
     * @param outputs Output variables
     * @return IR graph
     */
    auto get_graph(const std::vector<Variable>& inputs,
                   const std::vector<Variable>& outputs) -> std::shared_ptr<Graph>;

    TracingGuard(const TracingGuard&) = delete;
    TracingGuard& operator=(const TracingGuard&) = delete;

private:
    Tracer& tracer_;
    bool interceptor_installed_{false};
};

/**
 * @brief Trace a module's forward pass.
 *
 * Records all operations during module execution and returns an
 * optimized IR graph. This is the main entry point for JIT compilation.
 *
 * @param module Module to trace (must have forward() method)
 * @param dummy_input Example input for tracing
 * @return Traced and compiled graph
 *
 * @code
 * auto model = std::make_shared<MyNetwork>();
 * Variable dummy = Variable(Tensor({1, 3, 224, 224}, DType::Float32, Device::cpu()));
 * auto traced = trace(model, dummy);
 * traced->save("model.pt");
 * @endcode
 */
auto trace(std::shared_ptr<nn::Module> module,
           const Variable& dummy_input) -> std::shared_ptr<Graph>;

/**
 * @brief Trace a function.
 *
 * Records operations from a callable function.
 *
 * @param func Function to trace
 * @param inputs Input variables
 * @return Traced graph
 */
auto trace(std::function<std::vector<Variable>(const std::vector<Variable>&)> func,
           const std::vector<Variable>& inputs) -> std::shared_ptr<Graph>;

// Forward declaration
class CompiledModule;

/**
 * @brief Trace a module with a raw Tensor input, returning a CompiledModule.
 *
 * Convenience overload that wraps the input in a Variable and returns
 * a CompiledModule for optimized inference. This is the primary entry
 * point for JIT compilation.
 *
 * @param module Module to trace
 * @param dummy_input Example input tensor for tracing
 * @return Compiled module wrapping the traced graph
 *
 * @code
 * auto model = std::make_shared<MyNetwork>();
 * Tensor input({1, 3, 224, 224}, DType::Float32, Device::cpu());
 * auto traced = trace(model, input);
 * traced->forward(new_input);
 * @endcode
 */
auto trace(std::shared_ptr<nn::Module> module,
           const Tensor& dummy_input) -> std::shared_ptr<CompiledModule>;

/**
 * @brief Helper macros for automatic operation tracing.
 *
 * These macros should be inserted into tensor operations to enable tracing.
 */
#define TRACE_OP_BEGIN(op_type, inputs, outputs) \
    do { \
        if (::tenzor::jit::Tracer::get_instance().is_tracing()) { \
            ::tenzor::jit::TracedOp trace_op((op_type), (inputs), (outputs));

#define TRACE_OP_ATTR(name, value) \
            trace_op.attrs[(name)] = (value);

#define TRACE_OP_INT_ATTR(name, value) \
            trace_op.int_attrs[(name)] = (value);

#define TRACE_OP_VEC_ATTR(name, value) \
            trace_op.vec_attrs[(name)] = (value);

#define TRACE_OP_BOOL_ATTR(name, value) \
            trace_op.bool_attrs[(name)] = (value);

#define TRACE_OP_TENSOR_ATTR(name, value) \
            trace_op.tensor_attrs[(name)] = (value);

#define TRACE_OP_END() \
            ::tenzor::jit::Tracer::get_instance().record_op(std::move(trace_op)); \
        } \
    } while (0)

} // namespace jit
} // namespace tenzor
