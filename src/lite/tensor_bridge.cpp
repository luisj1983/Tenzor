/**
 * @file tensor_bridge.cpp
 * @brief LiteTensor <-> Tensor non-owning views (Phase 1).
 */

#include "tenzor/lite/tensor_bridge.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace tenzor::lite {

auto view_as_tensor(const LiteTensor& lt) -> Tensor {
    if (lt.ndim < 0 || lt.ndim > static_cast<int32_t>(kMaxDims)) {
        throw std::invalid_argument("view_as_tensor: invalid ndim");
    }
    std::vector<int64_t> shape;
    shape.reserve(static_cast<size_t>(lt.ndim));
    for (int32_t i = 0; i < lt.ndim; ++i) {
        shape.push_back(lt.shape[i]);
    }

    // Tensor::from_blob always derives contiguous row-major strides and has no
    // strides parameter, so a non-contiguous (transposed/sliced) LiteTensor
    // would be silently reinterpreted as contiguous, reading data in the wrong
    // element order. Validate that lt.strides match the contiguous row-major
    // layout for lt.shape and reject otherwise. Strides on dims of extent 0 or
    // 1 are irrelevant to element order, so they are not checked.
    {
        int64_t expected = 1;
        for (int32_t i = lt.ndim - 1; i >= 0; --i) {
            if (lt.shape[i] > 1 && lt.strides[i] != expected) {
                throw std::invalid_argument(
                    "view_as_tensor: non-contiguous LiteTensor strides are not "
                    "supported (expected row-major contiguous layout)");
            }
            expected *= lt.shape[i];
        }
    }

    // Empty deleter — the Tensor view does not own LiteTensor's buffer.
    return Tensor::from_blob(
        const_cast<void*>(lt.data),
        std::move(shape),
        lt.dtype,
        Device::cpu(),
        /*deleter=*/[](void*) noexcept {});
}

auto to_lite_tensor(const Tensor& t) -> LiteTensor {
    if (t.ndim() > static_cast<int64_t>(kMaxDims)) {
        throw std::invalid_argument(
            "to_lite_tensor: tensor has more dims than LiteTensor::kMaxDims");
    }

    LiteTensor out;
    out.ndim = static_cast<int32_t>(t.ndim());
    out.dtype = t.dtype();
    out.owns_data = true;

    int64_t numel = 1;
    for (int32_t i = 0; i < out.ndim; ++i) {
        out.shape[i] = t.size(i);
        numel *= out.shape[i];
    }
    // Row-major (contiguous) strides for the output buffer.
    for (int32_t i = out.ndim - 1; i >= 0; --i) {
        out.strides[i] = (i == out.ndim - 1) ? 1 : out.strides[i + 1] * out.shape[i + 1];
    }

    const auto nbytes = static_cast<size_t>(numel * dtype_size(out.dtype));
    out.data = (numel > 0) ? std::malloc(nbytes) : nullptr;
    if (numel > 0 && out.data == nullptr) {
        throw std::bad_alloc{};
    }

    if (numel > 0) {
        // Ensure source is contiguous on CPU before memcpy. .contiguous()
        // returns a self-reference when already contiguous, so this is cheap
        // in the common case.
        Tensor src = t;
        if (src.device().type != Device::Type::CPU) {
            src = src.to(Device::cpu());
        }
        if (!src.is_contiguous()) {
            src = src.contiguous();
        }
        std::memcpy(out.data, src.data_ptr(), nbytes);
    }
    return out;
}

}  // namespace tenzor::lite
