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
    // `lanes` describes a packed vector element (e.g. float32x4). Tenzor has no
    // vector dtype, so we map to the SCALAR base dtype here; from_dlpack() turns a
    // lanes>1 element into a trailing contiguous dimension of size `lanes`.
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

void export_deleter_versioned(DLManagedTensorVersioned* self) {
    if (!self) return;
    delete static_cast<ExportCtx*>(self->manager_ctx);
    delete self;
}

// Populate a DLTensor view (data/device/shape/strides/dtype) from a Tenzor,
// borrowing the shape/stride storage held by `ctx`. Shared by both the
// unversioned and versioned exporters so they cannot drift.
void fill_dl_tensor(DLTensor& dl, const Tensor& tensor, ExportCtx& ctx) {
    dl.data = const_cast<void*>(tensor.data_ptr());
    dl.device = device_to_dlpack(tensor.device());
    dl.ndim = static_cast<int32_t>(ctx.shape.size());
    dl.dtype = dtype_to_dlpack(tensor.dtype());
    dl.shape = ctx.shape.data();
    dl.strides = ctx.strides.empty() ? nullptr : ctx.strides.data();
    dl.byte_offset = 0;  // tensor.data_ptr() already includes the offset
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
        fill_dl_tensor(managed->dl_tensor, tensor, *ctx);
        managed->manager_ctx = ctx;
        managed->deleter = &export_deleter;
    } catch (...) {
        delete ctx;
        delete managed;
        throw;
    }
    return managed;
}

auto to_dlpack_versioned(const Tensor& tensor) -> DLManagedTensorVersioned* {
    auto* managed = new DLManagedTensorVersioned{};
    auto* ctx = new ExportCtx{tensor,
                              std::vector<int64_t>(tensor.shape().begin(),
                                                   tensor.shape().end()),
                              std::vector<int64_t>(tensor.strides().begin(),
                                                   tensor.strides().end())};
    try {
        managed->version.major = DLPACK_MAJOR_VERSION;
        managed->version.minor = DLPACK_MINOR_VERSION;
        managed->flags = 0;  // writable view, not a copy
        fill_dl_tensor(managed->dl_tensor, tensor, *ctx);
        managed->manager_ctx = ctx;
        managed->deleter = &export_deleter_versioned;
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

    // Multi-lane (packed vector) elements: a lanes>1 element is `lanes` contiguous
    // scalars (innermost). Represent it as a trailing dimension of size `lanes`.
    // `shape`/`dl.strides` stay in vector-element units for the strided walk below;
    // `out_shape` is the scalar-element shape exposed to the caller.
    const int lanes = (dl.dtype.lanes < 1) ? 1 : static_cast<int>(dl.dtype.lanes);
    std::vector<int64_t> out_shape = shape;
    if (lanes > 1) {
        out_shape.push_back(static_cast<int64_t>(lanes));
    }

    // Audit item F.8: support non-contiguous DLPack imports.  Producers
    // (NumPy views, PyTorch slices, …) hand us strides; we copy into a
    // contiguous Tensor on import rather than refusing.  The copy is on
    // the producer's device (CPU only here — GPU dlpack with strides
    // would need a backend-specific strided memcpy; keep that path on
    // the throw side until we wire it).
    bool is_contiguous = true;
    if (dl.strides != nullptr) {
        int64_t expected_stride = 1;
        for (int32_t i = dl.ndim - 1; i >= 0; --i) {
            // DLPack strides are in ELEMENTS, not bytes.
            if (dl.strides[i] != expected_stride) {
                is_contiguous = false;
                break;
            }
            expected_stride *= shape[i];
        }
    }

    // Apply byte_offset to get the actual data pointer.
    auto* data_base = static_cast<uint8_t*>(dl.data);
    auto* data_ptr = data_base + dl.byte_offset;

    if (!is_contiguous) {
        // Copy through strided indexing into a fresh contiguous Tensor.
        // Only CPU is supported; GPU-side strided dlpack would need a
        // device memcpy kernel (e.g. cudaMemcpy3D with non-unit strides),
        // which has not been wired yet — fail clearly there.
        if (device.type != Device::Type::CPU) {
            throw std::runtime_error(
                "from_dlpack: non-contiguous DLPack tensors are only "
                "supported on CPU.  For GPU producers, request a "
                "contiguous copy on the producer side first.");
        }
        // Per (vector) element = `lanes` contiguous scalars; for lanes==1 this is
        // just the scalar size (unchanged behavior).
        const size_t elem_bytes = dtype_size(dtype) * static_cast<size_t>(lanes);
        Tensor out(out_shape, dtype, device);
        auto* dst = static_cast<uint8_t*>(out.data_ptr());

        // DLPack permits negative strides (reverse views): a producer anchors
        // byte_offset near the end of the buffer and walks backward, so the
        // per-index element offset is legitimately negative for many indices.
        // Keep the offset arithmetic signed and compute the minimum reachable
        // element offset across all index combinations. For each axis the
        // extreme contributions are stride*0 and stride*(extent-1); a negative
        // stride contributes its negative extreme to the minimum.
        int64_t min_elem_off = 0;
        for (size_t d = 0; d < shape.size(); ++d) {
            if (shape[d] <= 0) continue;
            const int64_t extreme = dl.strides[d] * (shape[d] - 1);
            if (extreme < 0) {
                min_elem_off += extreme;
            }
        }

        // Anchor at the lowest reachable element so all subsequent offsets are
        // non-negative relative to `base`. data_ptr already includes
        // byte_offset; shift it down by the (possibly negative) minimum.
        auto* base = data_ptr + min_elem_off * static_cast<int64_t>(elem_bytes);
        if (base < data_base) {
            throw std::runtime_error(
                "from_dlpack: strided capsule references memory before the "
                "start of its data buffer (invalid negative offset).");
        }

        // Walk every index of the output (row-major) and dereference the
        // matching strided position in the source.
        int64_t total = 1;
        for (auto s : shape) total *= s;
        std::vector<int64_t> idx(shape.size(), 0);
        for (int64_t lin = 0; lin < total; ++lin) {
            // Compute source element offset from per-axis strides (in elements),
            // kept signed and rebased against the minimum reachable offset so it
            // is always non-negative. Casting a genuinely-negative running sum
            // straight to size_t (the previous code) wrapped to a huge value and
            // read out of bounds on negative-stride capsules.
            int64_t src_elem_off = -min_elem_off;
            for (size_t d = 0; d < shape.size(); ++d) {
                src_elem_off += idx[d] * dl.strides[d];
            }
            const size_t src_byte_off = static_cast<size_t>(src_elem_off) * elem_bytes;
            std::memcpy(dst + static_cast<size_t>(lin) * elem_bytes,
                        base + src_byte_off, elem_bytes);

            // Increment row-major index.
            for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
                if (++idx[d] < shape[d]) break;
                idx[d] = 0;
            }
        }

        // Producer's deleter still needs to run when we drop our reference
        // to the source.  Run it immediately since we have made our own
        // copy and no longer need the original buffer.
        if (managed->deleter != nullptr) {
            managed->deleter(managed);
        }
        return out;
    }

    // Wrap the external buffer with a deleter that invokes the producer's
    // DLPack deleter exactly once. from_blob's deleter runs when the new
    // Tensor's storage is released.
    Tensor t = Tensor::from_blob(data_ptr, std::move(out_shape), dtype, device,
                                 [managed](void*) {
                                     if (managed && managed->deleter) {
                                         managed->deleter(managed);
                                     }
                                 });
    return t;
}

} // namespace tenzor
