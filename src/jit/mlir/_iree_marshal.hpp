// Phase 13 / Path A — Marshalling helpers between tenzor::Tensor and
// iree_hal_buffer_view_t. Used by both the IreeInvoker (input/output
// shuffle around @main) and the tenzor_plugin native VM module
// (callback inputs/outputs).
//
// All copies are host-side: the IREE buffer_view is allocated host-mappable
// and we memcpy the bytes. For CPU targets (local-task / local-sync) this
// is the entire transfer cost. For GPU targets the underlying allocator
// transparently stages through a host-visible buffer; callers above this
// layer don't need to know which is in use.

#pragma once

#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"

#include <iree/hal/api.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor::jit::mlir_jit::marshal {

inline auto dtype_to_iree(::tenzor::DType d) -> iree_hal_element_type_t {
    switch (d) {
        case ::tenzor::DType::Float32: return IREE_HAL_ELEMENT_TYPE_FLOAT_32;
        case ::tenzor::DType::Float64: return IREE_HAL_ELEMENT_TYPE_FLOAT_64;
        case ::tenzor::DType::Int32:   return IREE_HAL_ELEMENT_TYPE_INT_32;
        case ::tenzor::DType::Int64:   return IREE_HAL_ELEMENT_TYPE_INT_64;
        default: break;
    }
    throw std::invalid_argument(
        "IREE marshal: unsupported DType (only Float32/64, Int32/64). value=" +
        std::to_string(static_cast<int>(d)));
}

inline auto iree_to_dtype(iree_hal_element_type_t e) -> ::tenzor::DType {
    switch (e) {
        case IREE_HAL_ELEMENT_TYPE_FLOAT_32:  return ::tenzor::DType::Float32;
        case IREE_HAL_ELEMENT_TYPE_FLOAT_64:  return ::tenzor::DType::Float64;
        case IREE_HAL_ELEMENT_TYPE_FLOAT_16:  return ::tenzor::DType::Float16;
        case IREE_HAL_ELEMENT_TYPE_BFLOAT_16: return ::tenzor::DType::BFloat16;
        case IREE_HAL_ELEMENT_TYPE_INT_8:     return ::tenzor::DType::Int8;
        case IREE_HAL_ELEMENT_TYPE_INT_16:    return ::tenzor::DType::Int16;
        case IREE_HAL_ELEMENT_TYPE_INT_32:    return ::tenzor::DType::Int32;
        case IREE_HAL_ELEMENT_TYPE_INT_64:    return ::tenzor::DType::Int64;
        case IREE_HAL_ELEMENT_TYPE_UINT_8:    return ::tenzor::DType::UInt8;
        case IREE_HAL_ELEMENT_TYPE_UINT_16:   return ::tenzor::DType::UInt16;
        case IREE_HAL_ELEMENT_TYPE_UINT_32:   return ::tenzor::DType::UInt32;
        case IREE_HAL_ELEMENT_TYPE_UINT_64:   return ::tenzor::DType::UInt64;
        case IREE_HAL_ELEMENT_TYPE_BOOL_8:    return ::tenzor::DType::Bool;
        case IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64:  return ::tenzor::DType::Complex64;
        case IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128: return ::tenzor::DType::Complex128;
        default: break;
    }
    throw std::invalid_argument(
        "IREE marshal: unsupported element type 0x" +
        std::to_string(static_cast<unsigned>(e)));
}

/// Allocate a fresh host-visible buffer_view of the given shape/dtype on the
/// device, copying the contiguous data from the Tensor's host storage.
/// Returns IREE ok status on success; failure status on allocator error.
/// The returned buffer_view ownership is transferred to the caller via
/// `*out_view` and must be released with `iree_hal_buffer_view_release`.
inline auto tensor_to_buffer_view(
    iree_hal_device_t* device,
    iree_hal_allocator_t* allocator,
    const ::tenzor::Tensor& t,
    iree_hal_buffer_view_t** out_view) -> iree_status_t {
    const ::tenzor::Tensor cpu = t.cpu().contiguous();
    std::vector<iree_hal_dim_t> dims;
    dims.reserve(cpu.shape().size());
    for (auto d : cpu.shape()) {
        dims.push_back(static_cast<iree_hal_dim_t>(d));
    }
    iree_hal_element_type_t etype = dtype_to_iree(cpu.dtype());
    iree_hal_buffer_params_t params = {};
    params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;

    const std::size_t bytes =
        static_cast<std::size_t>(cpu.numel()) * cpu.element_size();
    const void* host_ptr = cpu.data_ptr();

    return iree_hal_buffer_view_allocate_buffer_copy(
        device, allocator, dims.size(), dims.data(), etype,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
        iree_make_const_byte_span(host_ptr, bytes), out_view);
}

/// Read a buffer_view's contents back into a fresh host-side
/// (CPU-device) Tensor. Maps the buffer for read, memcpys, then unmaps.
inline auto buffer_view_to_tensor(iree_hal_buffer_view_t* bv)
    -> ::tenzor::Tensor {
    const iree_host_size_t rank = iree_hal_buffer_view_shape_rank(bv);
    std::vector<int64_t> shape(rank);
    for (iree_host_size_t i = 0; i < rank; ++i) {
        shape[i] = static_cast<int64_t>(iree_hal_buffer_view_shape_dim(bv, i));
    }
    const auto dt = iree_to_dtype(iree_hal_buffer_view_element_type(bv));

    // Allocate raw storage of the correct dtype directly; the subsequent
    // host-map / d2h transfer overwrites every byte, so the (uninitialized)
    // contents don't matter. Using the generic constructor (rather than
    // ::tenzor::full, which only has float/double scalar overloads) lets every
    // dtype iree_to_dtype accepts — Float16/BFloat16, the integer/unsigned
    // types, Bool and complex — round-trip correctly, matching the Subprocess
    // parser's supported set instead of throwing in the default InProcess mode.
    ::tenzor::Tensor out(shape, dt, ::tenzor::Device::cpu());

    iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(bv);
    const iree_device_size_t byte_length =
        iree_hal_buffer_view_byte_length(bv);

    // Try host-mapping first: free on CPU drivers (local-task / local-sync)
    // since the buffer storage is host-visible there. Fall back to an
    // explicit device-to-host transfer when the buffer is device-local
    // (CUDA/Vulkan/HIP — host can't map them directly).
    iree_hal_buffer_mapping_t mapping;
    iree_status_t status = iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
        /*byte_offset=*/0, byte_length, &mapping);
    if (iree_status_is_ok(status)) {
        std::memcpy(out.data_ptr(),
                    mapping.contents.data,
                    static_cast<std::size_t>(byte_length));
        iree_status_ignore(iree_hal_buffer_unmap_range(&mapping));
        return out;
    }
    iree_status_ignore(status);

    // Map failed: use a device→host transfer routed through the buffer's
    // associated device (recovered from the allocation placement).
    iree_hal_buffer_t* allocated_buffer =
        iree_hal_buffer_allocated_buffer(buffer);
    iree_hal_buffer_placement_t placement =
        iree_hal_buffer_allocation_placement(allocated_buffer);
    iree_hal_device_t* device = placement.device;
    if (!device) {
        throw std::runtime_error(
            "buffer_view_to_tensor: buffer is not host-mappable and has no "
            "associated device for d2h transfer");
    }
    iree_status_t ts = iree_hal_device_transfer_d2h(
        device, buffer, /*source_offset=*/0, out.data_ptr(), byte_length,
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout());
    if (!iree_status_is_ok(ts)) {
        iree_status_ignore(ts);
        throw std::runtime_error(
            "buffer_view_to_tensor: iree_hal_device_transfer_d2h failed");
    }
    return out;
}

}  // namespace tenzor::jit::mlir_jit::marshal
