/**
 * @file types.hpp
 * @brief Shared ONNX types for import and export functionality
 */

#pragma once

#include <cstdint>
#include "../core/dtype.hpp"

namespace tenzor {
namespace onnx {

/**
 * @brief ONNX data type enumeration mapping to Tenzor DType
 */
enum class ONNXDataType : int32_t {
    UNDEFINED = 0,
    FLOAT = 1,      // float32
    UINT8 = 2,      // uint8
    INT8 = 3,       // int8
    UINT16 = 4,     // uint16
    INT16 = 5,      // int16
    INT32 = 6,      // int32
    INT64 = 7,      // int64
    STRING = 8,     // string
    BOOL = 9,       // bool
    FLOAT16 = 10,   // float16
    DOUBLE = 11,    // float64
    UINT32 = 12,    // uint32
    UINT64 = 13,    // uint64
    COMPLEX64 = 14, // complex64
    COMPLEX128 = 15,// complex128
    BFLOAT16 = 16   // bfloat16
};

/**
 * @brief Convert Tenzor DType to ONNX data type
 *
 * @param dtype Tenzor data type
 * @return Corresponding ONNX data type
 */
auto dtype_to_onnx(DType dtype) -> ONNXDataType;

/**
 * @brief Convert ONNX data type to Tenzor DType
 *
 * @param onnx_dtype ONNX data type
 * @return Corresponding Tenzor DType
 */
auto onnx_to_dtype(ONNXDataType onnx_dtype) -> DType;

} // namespace onnx
} // namespace tenzor
