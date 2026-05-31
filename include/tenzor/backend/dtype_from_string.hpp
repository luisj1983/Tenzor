/// \file dtype_from_string.hpp
/// \brief Shared helper to parse a dtype string into a DType.
///
/// Consolidates the previously-duplicated `dtype_from_string` definitions that
/// lived inline in each backend kernel registry (CUDA, ROCm, Vulkan). The
/// canonical version handles the full set of dtype strings; backends that
/// previously recognised a subset gain the missing entries (they would have
/// fallen through to `default_val` before, so this only adds coverage).

#pragma once

#include <string_view>

#include "tenzor/core/dtype.hpp"

namespace tenzor {

/// Parse a dtype string (e.g. "float32", "int8", "complex64") into a DType.
/// Returns `default_val` for an empty or unrecognised string.
inline DType dtype_from_string(std::string_view s, DType default_val = DType::Float32) {
    if (s == "float32") return DType::Float32;
    if (s == "float64") return DType::Float64;
    if (s == "float16") return DType::Float16;
    if (s == "bfloat16") return DType::BFloat16;
    if (s == "int32") return DType::Int32;
    if (s == "int64") return DType::Int64;
    if (s == "int16") return DType::Int16;
    if (s == "int8") return DType::Int8;
    if (s == "uint8") return DType::UInt8;
    if (s == "uint16") return DType::UInt16;
    if (s == "uint32") return DType::UInt32;
    if (s == "uint64") return DType::UInt64;
    if (s == "bool") return DType::Bool;
    if (s == "complex64") return DType::Complex64;
    if (s == "complex128") return DType::Complex128;
    return default_val;
}

}  // namespace tenzor
