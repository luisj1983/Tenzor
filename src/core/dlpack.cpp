#include "tenzor/core/dlpack.hpp"

#include "tenzor/core/device.hpp"
#include "tenzor/core/dtype.hpp"

#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace {

// Map Tenzor DType -> DLPack DLDataType. Throws on FP8 / quantized dtypes
// which have no stable DLPack encoding.
auto dtype_to_dlpack(DType dt) -> DLDataType {
    DLDataType out{};
    out.lanes = 1;
    switch (dt) {
        case DType::Float16:    out.code = kDLFloat;  out.bits = 16; return out;
        case DType::Float32:    out.code = kDLFloat;  out.bits = 32; return out;
        case DType::Float64:    out.code = kDLFloat;  out.bits = 64; return out;
        case DType::BFloat16:   out.code = kDLBfloat; out.bits = 16; return out;
        case DType::Int8:       out.code = kDLInt;    out.bits = 8;  return out;
        case DType::Int16:      out.code = kDLInt;    out.bits = 16; return out;
        case DType::Int32:      out.code = kDLInt;    out.bits = 32; return out;
        case DType::Int64:      out.code = kDLInt;    out.bits = 64; return out;
        case DType::UInt8:      out.code = kDLUInt;   out.bits = 8;  return out;
        case DType::UInt16:     out.code = kDLUInt;   out.bits = 16; return out;
        case DType::UInt32:     out.code = kDLUInt;   out.bits = 32; return out;
        case DType::UInt64:     out.code = kDLUInt;   out.bits = 64; return out;
        case DType::Bool:       out.code = kDLBool;   out.bits = 8;  return out;
        case DType::Complex64:  out.code = kDLComplex; out.bits = 64;  return out;
        case DType::Complex128: out.code = kDLComplex; out.bits = 128; return out;

        case DType::FP8_E4M3:
        case DType::FP8_E5M2:
            throw std::runtime_error("to_dlpack: FP8 dtypes have no stable DLPack encoding");

        case DType::QInt8:
        case DType::QUInt8:
        case DType::QInt4x2:
            throw std::runtime_error("to_dlpack: quantized dtypes are not DLPack-representable");
    }
    throw std::runtime_error("to_dlpack: unknown dtype");
}

auto dlpack_to_dtype(DLDataType dl) -> DType {
    if (dl.lanes != 1) {
        throw std::runtime_error("from_dlpack: multi-lane dtypes are not supported");
    }
    switch (dl.code) {
        case kDLFloat:
            switch (dl.bits) {
                case 16: return DType::Float16;
                case 32: return DType::Float32;
                case 64: return DType::Float64;
                default: break;
            }
            break;
        case kDLBfloat:
            if (dl.bits == 16) return DType::BFloat16;
            break;
        case kDLInt:
            switch (dl.bits) {
                case 8:  return DType::Int8;
                case 16: return DType::Int16;
                case 32: return DType::Int32;
                case 64: return DType::Int64;
                default: break;
            }
            break;
        case kDLUInt:
            switch (dl.bits) {
                case 8:  return DType::UInt8;
                case 16: return DType::UInt16;
                case 32: return DType::UInt32;
                case 64: return DType::UInt64;
                default: break;
            }
            break;
        case kDLBool:
            if (dl.bits == 8) return DType::Bool;
            break;
        case kDLComplex:
            switch (dl.bits) {
                case 64:  return DType::Complex64;
                case 128: return DType::Complex128;
                default: break;
            }
            break;
    }
    throw std::runtime_error("from_dlpack: unsupported DLDataType (code=" +
                             std::to_string(dl.code) + ", bits=" +
                             std::to_string(dl.bits) + ")");
}

auto device_to_dlpack(Device d) -> DLDevice {
    DLDevice out{};
    out.device_id = d.index;
    switch (d.type) {
        case Device::Type::CPU:    out.device_type = kDLCPU;    return out;
        case Device::Type::CUDA:   out.device_type = kDLCUDA;   return out;
        case Device::Type::ROCm:   out.device_type = kDLROCM;   return out;
        case Device::Type::OneAPI: out.device_type = kDLOneAPI; return out;
        case Device::Type::Vulkan: out.device_type = kDLVulkan; return out;
        case Device::Type::MPS:    out.device_type = kDLMetal;  return out;
        case Device::Type::COUNT:  break;
    }
    throw std::runtime_error("to_dlpack: unsupported device type");
}

auto dlpack_to_device(DLDevice d) -> Device {
    Device::Type t;
    switch (d.device_type) {
        case kDLCPU:    t = Device::Type::CPU;    break;
        case kDLCUDA:
        case kDLCUDAHost:
        case kDLCUDAManaged:
                        t = Device::Type::CUDA;   break;
        case kDLROCM:
        case kDLROCMHost:
                        t = Device::Type::ROCm;   break;
        case kDLOneAPI: t = Device::Type::OneAPI; break;
        case kDLVulkan: t = Device::Type::Vulkan; break;
        case kDLMetal:  t = Device::Type::MPS;    break;
        default:
            throw std::runtime_error("from_dlpack: unsupported DLDevice type " +
                                     std::to_string(d.device_type));
    }
    return Device{t, d.device_id};
}

// Manager context for to_dlpack(): holds a strong reference to the source
// tensor's storage so the data stays alive until the DLPack consumer is
// done, plus heap-allocated shape and stride buffers referenced by the
// DLTensor.
struct ExportCtx {
    Tensor source;                 // Keeps storage alive
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
};

void export_deleter(DLManagedTensor* self) {
    if (!self) return;
    delete static_cast<ExportCtx*>(self->manager_ctx);
    delete self;
}

} // namespace

auto to_dlpack(const Tensor& tensor) -> DLManagedTensor* {
    auto* managed = new DLManagedTensor{};
    auto* ctx = new ExportCtx{tensor,
                              std::vector<int64_t>(tensor.shape().begin(),
                                                   tensor.shape().end()),
                              std::vector<int64_t>(tensor.strides().begin(),
                                                   tensor.strides().end())};

    try {
        managed->dl_tensor.data = const_cast<void*>(tensor.data_ptr());
        managed->dl_tensor.device = device_to_dlpack(tensor.device());
        managed->dl_tensor.ndim = static_cast<int32_t>(ctx->shape.size());
        managed->dl_tensor.dtype = dtype_to_dlpack(tensor.dtype());
        managed->dl_tensor.shape = ctx->shape.data();
        managed->dl_tensor.strides = ctx->strides.empty() ? nullptr : ctx->strides.data();
        managed->dl_tensor.byte_offset = 0;  // tensor.data_ptr() already includes offset
        managed->manager_ctx = ctx;
        managed->deleter = &export_deleter;
    } catch (...) {
        delete ctx;
        delete managed;
        throw;
    }
    return managed;
}

auto from_dlpack(DLManagedTensor* managed) -> Tensor {
    if (!managed) {
        throw std::runtime_error("from_dlpack: null DLManagedTensor");
    }
    const auto& dl = managed->dl_tensor;

    DType dtype = dlpack_to_dtype(dl.dtype);
    Device device = dlpack_to_device(dl.device);

    if (dl.ndim < 0) {
        throw std::runtime_error("from_dlpack: negative ndim");
    }
    std::vector<int64_t> shape(dl.shape, dl.shape + dl.ndim);

    // Phase 5.1: only accept contiguous imports for now. Non-contiguous
    // incoming DLPack tensors would need a strided from_blob overload.
    if (dl.strides != nullptr) {
        int64_t expected_stride = 1;
        for (int32_t i = dl.ndim - 1; i >= 0; --i) {
            if (dl.strides[i] != expected_stride) {
                throw std::runtime_error(
                    "from_dlpack: non-contiguous DLPack tensors are not yet "
                    "supported. Request the producer to materialize a "
                    "contiguous copy before passing it across the boundary.");
            }
            expected_stride *= shape[i];
        }
    }

    // Apply byte_offset to get the actual data pointer.
    auto* data_base = static_cast<uint8_t*>(dl.data);
    auto* data_ptr = data_base + dl.byte_offset;

    // Wrap the external buffer with a deleter that invokes the producer's
    // DLPack deleter exactly once. from_blob's deleter runs when the new
    // Tensor's storage is released.
    Tensor t = Tensor::from_blob(data_ptr, std::move(shape), dtype, device,
                                 [managed](void*) {
                                     if (managed && managed->deleter) {
                                         managed->deleter(managed);
                                     }
                                 });
    return t;
}

} // namespace tenzor
