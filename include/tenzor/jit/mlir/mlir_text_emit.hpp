// Phase 13 / Task A.6′ — MLIR text emission helpers.
//
// Per the 2026-05-19 spec amendment, Tenzor emits MLIR (StableHLO + custom_call)
// as text bytes via std::ostringstream and feeds the result into the IREE
// Compiler embedding API. This header provides the small, focused renderers
// used by the higher-level program builder.
//
// Design notes:
//   * No MLIR C++ dialect linkage. Output is plain ASCII matching the
//     official MLIR generic / pretty syntax that StableHLO accepts.
//   * Renderers take an output stream and append to it — they do NOT print
//     newlines themselves; the caller controls formatting.
//   * Scalar tensors render as `tensor<T>` (rank-0). Shaped tensors render
//     as `tensor<DxDxDxT>`. Dynamic dims aren't part of MVP-1.

#pragma once

#include "tenzor/core/dtype.hpp"

#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace tenzor::jit::mlir_jit {

/// Render a Tenzor DType as the MLIR scalar type identifier.
///
/// Mapping:
///   Float32 → "f32"
///   Float64 → "f64"
///   Float16 → "f16"
///   BFloat16 → "bf16"
///   Int8 → "i8", Int16 → "i16", Int32 → "i32", Int64 → "i64"
///   UInt8 → "ui8", UInt16 → "ui16", UInt32 → "ui32", UInt64 → "ui64"
///   Bool → "i1"
///   Complex64 → "complex<f32>"
///   Complex128 → "complex<f64>"
///
/// Throws std::invalid_argument for unsupported DType values.
auto mlir_type_name(::tenzor::DType d) -> std::string;

/// Render `tensor<DxDxDxT>` for the given shape/dtype. Rank-0 (empty shape)
/// renders as `tensor<T>`.
auto mlir_tensor_type(const std::vector<int64_t>& shape,
                      ::tenzor::DType d) -> std::string;

/// Emit `%result = stablehlo.<mnemonic> %a, %b : tensor<...>` to `os`.
///
/// `mnemonic` is the StableHLO op spelling (e.g. "add", "multiply",
/// "subtract", "divide"). Operands are SSA value names with leading `%`
/// stripped — this function adds the `%` prefix.
auto emit_stablehlo_binary(std::ostream& os, const std::string& mnemonic,
                           const std::string& result, const std::string& a,
                           const std::string& b,
                           const std::vector<int64_t>& shape,
                           ::tenzor::DType d) -> void;

/// Emit `%result = stablehlo.<mnemonic> %a : tensor<...>` to `os`.
auto emit_stablehlo_unary(std::ostream& os, const std::string& mnemonic,
                          const std::string& result, const std::string& a,
                          const std::vector<int64_t>& shape,
                          ::tenzor::DType d) -> void;

/// Emit `%result = stablehlo.<mnemonic> %a, %b, %c : tensor<...>` to `os`.
/// Used for ternary ops like `select` (cond, on_true, on_false) and
/// `clamp` (min, x, max). All three operands and the result share the
/// same tensor type for the simple element-wise case.
auto emit_stablehlo_ternary(std::ostream& os, const std::string& mnemonic,
                            const std::string& result, const std::string& a,
                            const std::string& b, const std::string& c,
                            const std::vector<int64_t>& shape,
                            ::tenzor::DType d) -> void;

/// Emit `%result = stablehlo.constant dense<value> : tensor<...>`.
/// `value_literal` is the textual MLIR attribute payload (e.g. "0.0",
/// "1.0", "0xFFC00000"). For full-tensor splats StableHLO accepts the
/// shorthand `dense<scalar>` and broadcasts to the result type.
auto emit_stablehlo_splat_constant(std::ostream& os,
                                   const std::string& result,
                                   const std::string& value_literal,
                                   const std::vector<int64_t>& shape,
                                   ::tenzor::DType d) -> void;

/// Emit a `stablehlo.reduce` block.
///
/// Generated form:
///   `%result = stablehlo.reduce(%a init: %init) applies stablehlo.<reducer>
///       across dimensions = [d0, d1, ...] : (tensor<...>, tensor<T>) ->
///       tensor<...>`
///
/// `reducer` is one of `add`, `multiply`, `maximum`, `minimum`.
/// `dims` are the input axes being reduced over.
/// `init_name` is the SSA name of the init constant.
auto emit_stablehlo_reduce(std::ostream& os, const std::string& result,
                           const std::string& operand,
                           const std::string& init_name,
                           const std::string& reducer,
                           const std::vector<int64_t>& dims,
                           const std::vector<int64_t>& operand_shape,
                           const std::vector<int64_t>& result_shape,
                           ::tenzor::DType d) -> void;

/// Emit `%result = stablehlo.reshape %a : tensor<src> -> tensor<dst>`.
auto emit_stablehlo_reshape(std::ostream& os, const std::string& result,
                            const std::string& a,
                            const std::vector<int64_t>& src_shape,
                            const std::vector<int64_t>& dst_shape,
                            ::tenzor::DType d) -> void;

/// Emit `%result = stablehlo.transpose %a, dims = [...] : tensor<src> ->
/// tensor<dst>`.
auto emit_stablehlo_transpose(std::ostream& os, const std::string& result,
                              const std::string& a,
                              const std::vector<int64_t>& perm,
                              const std::vector<int64_t>& src_shape,
                              const std::vector<int64_t>& dst_shape,
                              ::tenzor::DType d) -> void;

/// Emit `%result = stablehlo.broadcast_in_dim %a, dims = [...]
///   : tensor<src> -> tensor<dst>`.
auto emit_stablehlo_broadcast_in_dim(std::ostream& os,
                                     const std::string& result,
                                     const std::string& a,
                                     const std::vector<int64_t>& bcast_dims,
                                     const std::vector<int64_t>& src_shape,
                                     const std::vector<int64_t>& dst_shape,
                                     ::tenzor::DType d) -> void;

/// Emit `%result = stablehlo.convert %a : tensor<src_dtype> ->
/// tensor<dst_dtype>`.
auto emit_stablehlo_convert(std::ostream& os, const std::string& result,
                            const std::string& a,
                            const std::vector<int64_t>& shape,
                            ::tenzor::DType src_dtype,
                            ::tenzor::DType dst_dtype) -> void;

/// Emit `%result = stablehlo.concatenate %a, %b, ... dim = N
///   : (tensor<...>, ...) -> tensor<...>`.
auto emit_stablehlo_concatenate(
    std::ostream& os, const std::string& result,
    const std::vector<std::string>& operand_names,
    const std::vector<std::vector<int64_t>>& operand_shapes,
    int64_t dim, const std::vector<int64_t>& result_shape,
    ::tenzor::DType d) -> void;

/// Emit `%result = stablehlo.slice %a [start:limit:stride, ...]
///   : tensor<src> -> tensor<dst>`.
auto emit_stablehlo_slice(std::ostream& os, const std::string& result,
                          const std::string& a,
                          const std::vector<int64_t>& starts,
                          const std::vector<int64_t>& limits,
                          const std::vector<int64_t>& strides,
                          const std::vector<int64_t>& src_shape,
                          const std::vector<int64_t>& dst_shape,
                          ::tenzor::DType d) -> void;

/// Emit `%result = stablehlo.pad %a, %padval, low = [...], high = [...],
///   interior = [...] : (tensor<src>, tensor<>) -> tensor<dst>`.
auto emit_stablehlo_pad(std::ostream& os, const std::string& result,
                        const std::string& a, const std::string& padval,
                        const std::vector<int64_t>& low,
                        const std::vector<int64_t>& high,
                        const std::vector<int64_t>& interior,
                        const std::vector<int64_t>& src_shape,
                        const std::vector<int64_t>& dst_shape,
                        ::tenzor::DType d) -> void;

/// Emit a `stablehlo.dot_general` op with explicit batching and
/// contracting dim lists. Output type is required (dot_general doesn't
/// auto-infer from operands).
auto emit_stablehlo_dot_general(std::ostream& os, const std::string& result,
                                const std::string& a, const std::string& b,
                                const std::vector<int64_t>& lhs_batch,
                                const std::vector<int64_t>& rhs_batch,
                                const std::vector<int64_t>& lhs_contracting,
                                const std::vector<int64_t>& rhs_contracting,
                                const std::vector<int64_t>& lhs_shape,
                                const std::vector<int64_t>& rhs_shape,
                                const std::vector<int64_t>& result_shape,
                                ::tenzor::DType d) -> void;

/// Emit a `stablehlo.custom_call @<callee>(...)` invocation.
///
/// Generated form:
///   `%result = stablehlo.custom_call @callee(%a, %b) {backend_config = "..."}
///       : (tensor<...>, tensor<...>) -> tensor<...>`
///
/// `operand_names`, `operand_shapes`, and `operand_dtypes` must all have the
/// same length. `backend_config` is emitted verbatim inside double quotes;
/// callers must escape internal quotes/backslashes themselves.
auto emit_custom_call(std::ostream& os, const std::string& callee,
                      const std::string& result,
                      const std::vector<std::string>& operand_names,
                      const std::vector<std::vector<int64_t>>& operand_shapes,
                      const std::vector<::tenzor::DType>& operand_dtypes,
                      const std::vector<int64_t>& result_shape,
                      ::tenzor::DType result_dtype,
                      const std::string& backend_config) -> void;

/// Wrap a function body into a `module { func.func @main(...) -> ... { <body>
/// return %... : ... } }` shell.
///
/// `inputs` / `outputs` are (shape, dtype) pairs declaring the function
/// signature. `return_names` are the SSA value names (without `%`) returned
/// from the body. `body_os.str()` is consumed as-is; callers are expected to
/// have populated it with one statement per line.
///
/// Returns the complete module text.
auto emit_module_wrapper(
    std::ostream& body_os,
    const std::vector<std::pair<std::vector<int64_t>, ::tenzor::DType>>& inputs,
    const std::vector<std::pair<std::vector<int64_t>, ::tenzor::DType>>& outputs,
    const std::vector<std::string>& return_names) -> std::string;

}  // namespace tenzor::jit::mlir_jit
