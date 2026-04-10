/* Vendored DLPack C ABI header (version 0.8).
 *
 * DLPack is the open in-memory tensor exchange protocol used across
 * NumPy >= 1.22, PyTorch, TensorFlow, JAX, MXNet, CuPy, and more. This
 * file is a minimal self-contained copy of the reference header from
 * https://github.com/dmlc/dlpack (Apache-2.0 licensed).
 *
 * Keep the ABI stable. Do not modify struct layouts.
 */

#ifndef TENZOR_DLPACK_H_
#define TENZOR_DLPACK_H_

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
extern "C" {
#else
#include <stddef.h>
#include <stdint.h>
#endif

#define DLPACK_VERSION 80
#define DLPACK_ABI_VERSION 1

/*!
 * \brief The device type in DLDevice.
 */
typedef enum {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLOpenCL = 4,
    kDLVulkan = 7,
    kDLMetal = 8,
    kDLVPI = 9,
    kDLROCM = 10,
    kDLROCMHost = 11,
    kDLExtDev = 12,
    kDLCUDAManaged = 13,
    kDLOneAPI = 14,
    kDLWebGPU = 15,
    kDLHexagon = 16,
} DLDeviceType;

typedef struct {
    DLDeviceType device_type;
    int32_t device_id;
} DLDevice;

typedef enum {
    kDLInt = 0,
    kDLUInt = 1,
    kDLFloat = 2,
    kDLOpaqueHandle = 3,
    kDLBfloat = 4,
    kDLComplex = 5,
    kDLBool = 6,
} DLDataTypeCode;

typedef struct {
    uint8_t code;  ///< DLDataTypeCode
    uint8_t bits;  ///< Number of bits (8, 16, 32, 64, 128)
    uint16_t lanes; ///< Number of lanes (1 for scalar)
} DLDataType;

/*!
 * \brief Plain C tensor object, does not manage memory.
 */
typedef struct {
    void* data;          ///< Raw data pointer, aligned as required by dtype
    DLDevice device;     ///< Device
    int32_t ndim;        ///< Number of dimensions
    DLDataType dtype;    ///< Element dtype
    int64_t* shape;      ///< Shape, length ndim
    int64_t* strides;    ///< Strides in elements (NOT bytes); NULL = contiguous
    uint64_t byte_offset; ///< Byte offset added to `data` before the tensor starts
} DLTensor;

/*!
 * \brief C Tensor object managed by DLPack protocol.
 *
 * The producer of this object owns the memory referenced by the embedded
 * DLTensor. The `deleter` must be called to release ownership (along with
 * any resources associated with `manager_ctx`).
 */
typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;                                     ///< Opaque context
    void (*deleter)(struct DLManagedTensor* self);         ///< Releases manager_ctx
} DLManagedTensor;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TENZOR_DLPACK_H_
