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
