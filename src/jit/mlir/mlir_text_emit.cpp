// Phase 13 / Task A.6′ — MLIR text emission helpers (implementation).

#include "tenzor/jit/mlir/mlir_text_emit.hpp"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tenzor::jit::mlir_jit {

namespace {

auto write_tensor_type(std::ostream& os, const std::vector<int64_t>& shape,
                       ::tenzor::DType d) -> void {
    os << "tensor<";
    for (auto dim : shape) {
        os << dim << 'x';
    }
    os << mlir_type_name(d) << '>';
}

}  // namespace

auto mlir_type_name(::tenzor::DType d) -> std::string {
    using ::tenzor::DType;
    switch (d) {
        case DType::Float32:    return "f32";
        case DType::Float64:    return "f64";
        case DType::Float16:    return "f16";
        case DType::BFloat16:   return "bf16";
        case DType::Int8:       return "i8";
        case DType::Int16:      return "i16";
        case DType::Int32:      return "i32";
        case DType::Int64:      return "i64";
        case DType::UInt8:      return "ui8";
        case DType::UInt16:     return "ui16";
        case DType::UInt32:     return "ui32";
        case DType::UInt64:     return "ui64";
        case DType::Bool:       return "i1";
        case DType::Complex64:  return "complex<f32>";
        case DType::Complex128: return "complex<f64>";
    }
    throw std::invalid_argument(
        "mlir_type_name: unsupported DType (value=" +
        std::to_string(static_cast<int>(d)) + ")");
}

auto mlir_tensor_type(const std::vector<int64_t>& shape,
                      ::tenzor::DType d) -> std::string {
    std::ostringstream os;
    write_tensor_type(os, shape, d);
    return os.str();
}

auto emit_stablehlo_binary(std::ostream& os, const std::string& mnemonic,
                           const std::string& result, const std::string& a,
                           const std::string& b,
                           const std::vector<int64_t>& shape,
                           ::tenzor::DType d) -> void {
    os << '%' << result << " = stablehlo." << mnemonic << " %" << a << ", %"
       << b << " : ";
    write_tensor_type(os, shape, d);
}

auto emit_stablehlo_unary(std::ostream& os, const std::string& mnemonic,
                          const std::string& result, const std::string& a,
                          const std::vector<int64_t>& shape,
                          ::tenzor::DType d) -> void {
    os << '%' << result << " = stablehlo." << mnemonic << " %" << a << " : ";
    write_tensor_type(os, shape, d);
}

auto emit_chlo_unary(std::ostream& os, const std::string& mnemonic,
                     const std::string& result, const std::string& a,
                     const std::vector<int64_t>& shape,
                     ::tenzor::DType d) -> void {
    os << '%' << result << " = chlo." << mnemonic << " %" << a << " : ";
    write_tensor_type(os, shape, d);
    os << " -> ";
    write_tensor_type(os, shape, d);
}

auto emit_stablehlo_ternary(std::ostream& os, const std::string& mnemonic,
                            const std::string& result, const std::string& a,
                            const std::string& b, const std::string& c,
                            const std::vector<int64_t>& shape,
                            ::tenzor::DType d) -> void {
    os << '%' << result << " = stablehlo." << mnemonic << " %" << a << ", %"
       << b << ", %" << c << " : ";
    write_tensor_type(os, shape, d);
}

auto emit_stablehlo_splat_constant(std::ostream& os,
                                   const std::string& result,
                                   const std::string& value_literal,
                                   const std::vector<int64_t>& shape,
                                   ::tenzor::DType d) -> void {
    os << '%' << result << " = stablehlo.constant dense<" << value_literal
       << "> : ";
    write_tensor_type(os, shape, d);
}

namespace {

auto write_dims_list(std::ostream& os, const std::vector<int64_t>& dims)
    -> void {
    os << '[';
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i != 0) os << ", ";
        os << dims[i];
    }
    os << ']';
}

}  // namespace

auto emit_stablehlo_reduce(std::ostream& os, const std::string& result,
                           const std::string& operand,
                           const std::string& init_name,
                           const std::string& reducer,
                           const std::vector<int64_t>& dims,
                           const std::vector<int64_t>& operand_shape,
                           const std::vector<int64_t>& result_shape,
                           ::tenzor::DType d) -> void {
    os << '%' << result << " = stablehlo.reduce(%" << operand << " init: %"
       << init_name << ") applies stablehlo." << reducer
       << " across dimensions = ";
    write_dims_list(os, dims);
    os << " : (";
    write_tensor_type(os, operand_shape, d);
    os << ", ";
    write_tensor_type(os, {}, d);
    os << ") -> ";
    write_tensor_type(os, result_shape, d);
}

auto emit_stablehlo_reshape(std::ostream& os, const std::string& result,
                            const std::string& a,
                            const std::vector<int64_t>& src_shape,
                            const std::vector<int64_t>& dst_shape,
                            ::tenzor::DType d) -> void {
    os << '%' << result << " = stablehlo.reshape %" << a << " : (";
    write_tensor_type(os, src_shape, d);
    os << ") -> ";
    write_tensor_type(os, dst_shape, d);
}

auto emit_stablehlo_transpose(std::ostream& os, const std::string& result,
                              const std::string& a,
                              const std::vector<int64_t>& perm,
                              const std::vector<int64_t>& src_shape,
                              const std::vector<int64_t>& dst_shape,
                              ::tenzor::DType d) -> void {
    os << '%' << result << " = stablehlo.transpose %" << a << ", dims = ";
    write_dims_list(os, perm);
    os << " : (";
    write_tensor_type(os, src_shape, d);
    os << ") -> ";
    write_tensor_type(os, dst_shape, d);
}

auto emit_stablehlo_broadcast_in_dim(std::ostream& os,
                                     const std::string& result,
                                     const std::string& a,
                                     const std::vector<int64_t>& bcast_dims,
                                     const std::vector<int64_t>& src_shape,
                                     const std::vector<int64_t>& dst_shape,
                                     ::tenzor::DType d) -> void {
    os << '%' << result << " = stablehlo.broadcast_in_dim %" << a
       << ", dims = ";
    write_dims_list(os, bcast_dims);
    os << " : (";
    write_tensor_type(os, src_shape, d);
    os << ") -> ";
    write_tensor_type(os, dst_shape, d);
}

auto emit_stablehlo_convert(std::ostream& os, const std::string& result,
                            const std::string& a,
                            const std::vector<int64_t>& shape,
                            ::tenzor::DType src_dtype,
                            ::tenzor::DType dst_dtype) -> void {
    os << '%' << result << " = stablehlo.convert %" << a << " : (";
    write_tensor_type(os, shape, src_dtype);
    os << ") -> ";
    write_tensor_type(os, shape, dst_dtype);
}

auto emit_stablehlo_concatenate(
    std::ostream& os, const std::string& result,
    const std::vector<std::string>& operand_names,
    const std::vector<std::vector<int64_t>>& operand_shapes,
    int64_t dim, const std::vector<int64_t>& result_shape,
    ::tenzor::DType d) -> void {
    if (operand_names.size() != operand_shapes.size()) {
        throw std::invalid_argument(
            "emit_stablehlo_concatenate: operand_names and operand_shapes "
            "must have equal length");
    }
    os << '%' << result << " = stablehlo.concatenate";
    for (std::size_t i = 0; i < operand_names.size(); ++i) {
        os << (i == 0 ? " %" : ", %") << operand_names[i];
    }
    os << ", dim = " << dim << " : (";
    for (std::size_t i = 0; i < operand_shapes.size(); ++i) {
        if (i != 0) os << ", ";
        write_tensor_type(os, operand_shapes[i], d);
    }
    os << ") -> ";
    write_tensor_type(os, result_shape, d);
}

auto emit_stablehlo_slice(std::ostream& os, const std::string& result,
                          const std::string& a,
                          const std::vector<int64_t>& starts,
                          const std::vector<int64_t>& limits,
                          const std::vector<int64_t>& strides,
                          const std::vector<int64_t>& src_shape,
                          const std::vector<int64_t>& dst_shape,
                          ::tenzor::DType d) -> void {
    os << '%' << result << " = stablehlo.slice %" << a;
    os << " [";
    for (std::size_t i = 0; i < starts.size(); ++i) {
        if (i != 0) os << ", ";
        os << starts[i] << ':' << limits[i] << ':' << strides[i];
    }
    os << "] : (";
    write_tensor_type(os, src_shape, d);
    os << ") -> ";
    write_tensor_type(os, dst_shape, d);
}

auto emit_stablehlo_pad(std::ostream& os, const std::string& result,
                        const std::string& a, const std::string& padval,
                        const std::vector<int64_t>& low,
                        const std::vector<int64_t>& high,
                        const std::vector<int64_t>& interior,
                        const std::vector<int64_t>& src_shape,
                        const std::vector<int64_t>& dst_shape,
                        ::tenzor::DType d) -> void {
    os << '%' << result << " = stablehlo.pad %" << a << ", %" << padval
       << ", low = ";
    write_dims_list(os, low);
    os << ", high = ";
    write_dims_list(os, high);
    os << ", interior = ";
    write_dims_list(os, interior);
    os << " : (";
    write_tensor_type(os, src_shape, d);
    os << ", ";
    write_tensor_type(os, {}, d);
    os << ") -> ";
    write_tensor_type(os, dst_shape, d);
}

auto emit_stablehlo_dot_general(std::ostream& os, const std::string& result,
                                const std::string& a, const std::string& b,
                                const std::vector<int64_t>& lhs_batch,
                                const std::vector<int64_t>& rhs_batch,
                                const std::vector<int64_t>& lhs_contracting,
                                const std::vector<int64_t>& rhs_contracting,
                                const std::vector<int64_t>& lhs_shape,
                                const std::vector<int64_t>& rhs_shape,
                                const std::vector<int64_t>& result_shape,
                                ::tenzor::DType d) -> void {
    os << '%' << result << " = stablehlo.dot_general %" << a << ", %" << b
       << ", batching_dims = [";
    for (std::size_t i = 0; i < lhs_batch.size(); ++i) {
        if (i != 0) os << ", ";
        os << lhs_batch[i];
    }
    os << "] x [";
    for (std::size_t i = 0; i < rhs_batch.size(); ++i) {
        if (i != 0) os << ", ";
        os << rhs_batch[i];
    }
    os << "], contracting_dims = [";
    for (std::size_t i = 0; i < lhs_contracting.size(); ++i) {
        if (i != 0) os << ", ";
        os << lhs_contracting[i];
    }
    os << "] x [";
    for (std::size_t i = 0; i < rhs_contracting.size(); ++i) {
        if (i != 0) os << ", ";
        os << rhs_contracting[i];
    }
    os << "]";
    // Force HIGHEST precision for float GEMMs: half-precision (F16/BF16) inputs
    // then accumulate in f32 (matching eager MKL/oneDNN/cuBLAS f32-accumulate)
    // and F32 inputs avoid TF32 reduced-precision tensor cores — keeping results
    // consistent with eager and across HAL targets. Integer dot_general ignores
    // precision, so restrict to float dtypes.
    {
        using ::tenzor::DType;
        const bool is_float = (d == DType::Float16 || d == DType::BFloat16 ||
                               d == DType::Float32 || d == DType::Float64);
        if (is_float) os << ", precision = [HIGHEST, HIGHEST]";
    }
    os << " : (";
    write_tensor_type(os, lhs_shape, d);
    os << ", ";
    write_tensor_type(os, rhs_shape, d);
    os << ") -> ";
    write_tensor_type(os, result_shape, d);
}

auto emit_custom_call(std::ostream& os, const std::string& callee,
                      const std::string& result,
                      const std::vector<std::string>& operand_names,
                      const std::vector<std::vector<int64_t>>& operand_shapes,
                      const std::vector<::tenzor::DType>& operand_dtypes,
                      const std::vector<int64_t>& result_shape,
                      ::tenzor::DType result_dtype,
                      const std::string& backend_config) -> void {
    if (operand_names.size() != operand_shapes.size() ||
        operand_names.size() != operand_dtypes.size()) {
        throw std::invalid_argument(
            "emit_custom_call: operand_names, operand_shapes, and "
            "operand_dtypes must have equal length");
    }

    os << '%' << result << " = stablehlo.custom_call @" << callee << '(';
    for (std::size_t i = 0; i < operand_names.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        os << '%' << operand_names[i];
    }
    os << ") {backend_config = \"" << backend_config << "\"} : (";
    for (std::size_t i = 0; i < operand_names.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        write_tensor_type(os, operand_shapes[i], operand_dtypes[i]);
    }
    os << ") -> ";
    write_tensor_type(os, result_shape, result_dtype);
}

auto emit_plugin_call(
    std::ostream& os, const std::string& callee, const std::string& result,
    const std::vector<std::string>& tensor_operand_names,
    const std::vector<std::vector<int64_t>>& tensor_operand_shapes,
    const std::vector<::tenzor::DType>& tensor_operand_dtypes,
    const std::vector<std::pair<std::string, std::string>>& scalar_args,
    const std::vector<int64_t>& result_shape,
    ::tenzor::DType result_dtype) -> void {
    if (tensor_operand_names.size() != tensor_operand_shapes.size() ||
        tensor_operand_names.size() != tensor_operand_dtypes.size()) {
        throw std::invalid_argument(
            "emit_plugin_call: tensor_operand_{names,shapes,dtypes} must have "
            "equal length");
    }

    os << '%' << result << " = call @" << callee << '(';
    bool first = true;
    for (std::size_t i = 0; i < tensor_operand_names.size(); ++i) {
        if (!first) os << ", ";
        os << '%' << tensor_operand_names[i];
        first = false;
    }
    for (const auto& [name, _ty] : scalar_args) {
        if (!first) os << ", ";
        os << '%' << name;
        first = false;
    }
    os << ") : (";
    first = true;
    for (std::size_t i = 0; i < tensor_operand_names.size(); ++i) {
        if (!first) os << ", ";
        write_tensor_type(os, tensor_operand_shapes[i],
                          tensor_operand_dtypes[i]);
        first = false;
    }
    for (const auto& [_name, ty] : scalar_args) {
        if (!first) os << ", ";
        os << ty;
        first = false;
    }
    os << ") -> ";
    write_tensor_type(os, result_shape, result_dtype);
}

auto emit_module_wrapper(
    std::ostream& body_os,
    const std::vector<std::pair<std::vector<int64_t>, ::tenzor::DType>>& inputs,
    const std::vector<std::pair<std::vector<int64_t>, ::tenzor::DType>>& outputs,
    const std::vector<std::string>& return_names,
    const std::vector<std::string>& extern_decls) -> std::string {
    if (outputs.size() != return_names.size()) {
        throw std::invalid_argument(
            "emit_module_wrapper: outputs and return_names must have equal "
            "length");
    }

    // body_os is a std::ostream& — extract its accumulated text. The expected
    // usage is to pass an ostringstream, in which case str() reads it
    // directly. Otherwise fall back to copying via rdbuf() into a fresh
    // stringbuf (works for any ostream whose streambuf supports underflow).
    std::string body_text;
    if (auto* oss = dynamic_cast<std::ostringstream*>(&body_os)) {
        body_text = oss->str();
    } else {
        std::ostringstream body_capture;
        body_capture << body_os.rdbuf();
        body_text = body_capture.str();
    }

    std::ostringstream out;
    out << "module {\n";

    // External `func.func private @<name>(...) -> ...` declarations the
    // plugin path uses to declare tenzor_plugin.<op> imports. Emitted
    // before @main so the call sites in the body resolve against them.
    for (const auto& decl : extern_decls) {
        out << "  " << decl;
        if (decl.empty() || decl.back() != '\n') {
            out << '\n';
        }
    }

    out << "  func.func @main(";
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << "%arg" << i << ": ";
        write_tensor_type(out, inputs[i].first, inputs[i].second);
    }
    out << ") -> (";
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        write_tensor_type(out, outputs[i].first, outputs[i].second);
    }
    out << ") {\n";

    out << body_text;
    if (!body_text.empty() && body_text.back() != '\n') {
        out << '\n';
    }

    out << "    return";
    for (std::size_t i = 0; i < return_names.size(); ++i) {
        out << (i == 0 ? " %" : ", %") << return_names[i];
    }
    if (!return_names.empty()) {
        out << " : ";
        for (std::size_t i = 0; i < outputs.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            write_tensor_type(out, outputs[i].first, outputs[i].second);
        }
    }
    out << '\n';
    out << "  }\n";
    out << "}\n";
    return out.str();
}

}  // namespace tenzor::jit::mlir_jit
