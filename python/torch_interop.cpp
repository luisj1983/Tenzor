/**
 * @file torch_interop.cpp
 * @brief Implementation of PyTorch tensor interoperability
 */

#include "torch_interop.hpp"
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>   // Audit J7: at::toDLPack / at::fromDLPack
#include <tenzor/core/dlpack.hpp>  // Audit J7: tenzor::to_dlpack / from_dlpack
#include <c10/core/ScalarType.h>
#include <stdexcept>
#include <sstream>

namespace tenzor {
namespace torch_interop {

auto can_zero_copy_to_torch(const Tensor& tensor) -> bool {
    // Zero-copy requires:
    // 1. Contiguous memory layout
    // 2. Compatible device (CPU or CUDA)
    // 3. Supported data type

    if (!tensor.is_contiguous()) {
        return false;
    }

    auto device = tensor.device();
    if (device.type != Device::Type::CPU && device.type != Device::Type::CUDA) {
        return false;
    }

    // Check if dtype is supported by PyTorch
    try {
        dtype_to_torch(tensor.dtype());
        return true;
    } catch (...) {
        return false;
    }
}

auto can_zero_copy_from_torch(const torch::Tensor& torch_tensor) -> bool {
    // Zero-copy requires:
    // 1. Contiguous memory layout
    // 2. Compatible device
    // 3. Supported data type

    if (!torch_tensor.is_contiguous()) {
        return false;
    }

    auto device = torch_tensor.device();
    if (!device.is_cpu() && !device.is_cuda()) {
        return false;
    }

    // Check if dtype is supported
    try {
        dtype_from_torch(static_cast<int>(torch_tensor.scalar_type()));
        return true;
    } catch (...) {
        return false;
    }
}

auto dtype_to_torch(DType dtype) -> int {
    switch (dtype) {
        case DType::Float32:
            return static_cast<int>(torch::kFloat32);
        case DType::Float64:
            return static_cast<int>(torch::kFloat64);
        case DType::Float16:
            return static_cast<int>(torch::kFloat16);
        case DType::BFloat16:
            return static_cast<int>(torch::kBFloat16);
        case DType::Int8:
            return static_cast<int>(torch::kInt8);
        case DType::Int16:
            return static_cast<int>(torch::kInt16);
        case DType::Int32:
            return static_cast<int>(torch::kInt32);
        case DType::Int64:
            return static_cast<int>(torch::kInt64);
        case DType::UInt8:
            return static_cast<int>(torch::kUInt8);
        // 5th-audit B1: PyTorch added UInt16/UInt32/UInt64 in 2.3+.
        case DType::UInt16:
            return static_cast<int>(torch::kUInt16);
        case DType::UInt32:
            return static_cast<int>(torch::kUInt32);
        case DType::UInt64:
            return static_cast<int>(torch::kUInt64);
        case DType::Bool:
            return static_cast<int>(torch::kBool);
        case DType::Complex64:
            return static_cast<int>(torch::kComplexFloat);
        case DType::Complex128:
            return static_cast<int>(torch::kComplexDouble);
        default:
            throw std::runtime_error("Unsupported DType for PyTorch conversion");
    }
}

auto dtype_from_torch(int torch_dtype) -> DType {
    auto scalar_type = static_cast<torch::ScalarType>(torch_dtype);

    switch (scalar_type) {
        case torch::kFloat32:
            return DType::Float32;
        case torch::kFloat64:
            return DType::Float64;
        case torch::kFloat16:
            return DType::Float16;
        case torch::kBFloat16:
            return DType::BFloat16;
        case torch::kInt8:
            return DType::Int8;
        case torch::kInt16:
            return DType::Int16;
        case torch::kInt32:
            return DType::Int32;
        case torch::kInt64:
            return DType::Int64;
        case torch::kUInt8:
            return DType::UInt8;
        // 5th-audit B1: PyTorch added UInt16/UInt32/UInt64 in 2.3+.
        case torch::kUInt16:
            return DType::UInt16;
        case torch::kUInt32:
            return DType::UInt32;
        case torch::kUInt64:
            return DType::UInt64;
        case torch::kBool:
            return DType::Bool;
        case torch::kComplexFloat:
            return DType::Complex64;
        case torch::kComplexDouble:
            return DType::Complex128;
        default:
            throw std::runtime_error("Unsupported PyTorch ScalarType for Tenzor");
    }
}

auto device_to_torch_string(const Device& device) -> std::string {
    switch (device.type) {
        case Device::Type::CPU:
            return "cpu";
        case Device::Type::CUDA:
            return "cuda:" + std::to_string(device.index);
        default:
            throw std::runtime_error("Unsupported device type for PyTorch");
    }
}

auto device_from_torch_string(const std::string& device_str) -> Device {
    if (device_str == "cpu") {
        return Device::cpu();
    }

    if (device_str.substr(0, 5) == "cuda:") {
        int index = std::stoi(device_str.substr(5));
        return Device::cuda(index);
    }

    if (device_str == "cuda") {
        return Device::cuda(0);
    }

    throw std::runtime_error("Unsupported PyTorch device string: " + device_str);
}

auto tensor_to_torch(const Tensor& tensor, bool requires_grad) -> torch::Tensor {
    // Get tensor metadata
    auto shape = tensor.shape();
    auto dtype = tensor.dtype();
    auto device = tensor.device();

    // Convert shape to torch IntArrayRef
    std::vector<int64_t> torch_shape(shape.begin(), shape.end());

    // Convert dtype
    auto torch_dtype = static_cast<torch::ScalarType>(dtype_to_torch(dtype));

    // Convert device
    torch::Device torch_device(torch::kCPU);
    if (device.type == Device::Type::CUDA) {
        torch_device = torch::Device(torch::kCUDA, device.index);
    }

    torch::Tensor torch_tensor;

    if (can_zero_copy_to_torch(tensor)) {
        // 5th-audit B2: zero-copy `torch::from_blob` MUST be given a deleter
        // that keeps the source Tenzor storage alive for at least as long as
        // the returned PyTorch tensor. Pre-fix this call had no deleter — if
        // the source Tenzor went out of scope first, PyTorch ended up reading
        // freed memory (use-after-free).
        //
        // We heap-allocate an `intrusive_ptr<Storage>` ticket and free it
        // from the deleter (mirroring the numpy capsule pattern at B'2).
        // The capture-by-value `storage_ticket` parameter in the deleter
        // closure cannot move, so we use the raw-ptr-and-delete idiom.
        auto* storage_ticket = new intrusive_ptr<Storage>(tensor.storage());
        torch_tensor = torch::from_blob(
            const_cast<void*>(tensor.data<void>()),
            torch_shape,
            /*deleter=*/[storage_ticket](void* /*data*/) {
                delete storage_ticket;
            },
            torch::TensorOptions().dtype(torch_dtype)
        );

        // If CUDA, move to correct device
        if (device.type == Device::Type::CUDA) {
            torch_tensor = torch_tensor.to(torch_device);
        }
    } else {
        // Need to copy data
        // First make tensor contiguous
        Tensor contiguous_tensor = tensor.is_contiguous() ? tensor : tensor.contiguous();

        // Create PyTorch tensor and copy data
        torch_tensor = torch::empty(torch_shape, torch::TensorOptions()
                                                    .dtype(torch_dtype)
                                                    .device(torch_device));

        if (device.type == Device::Type::CPU && torch_device.is_cpu()) {
            std::memcpy(torch_tensor.data_ptr(),
                       contiguous_tensor.data<void>(),
                       contiguous_tensor.numel() * dtype_size(dtype));
        } else if (contiguous_tensor.device().type == Device::Type::CUDA
                   || torch_device.is_cuda()) {
            // 5th-audit B3 / 6th-audit Fix #3: CUDA path. Select the correct
            // cudaMemcpyKind from the actual {source, destination} device
            // pair. Pre-5th-audit this path hardcoded
            // `cudaMemcpyDeviceToDevice`; the 5th-audit fix selected on
            // CUDA-vs-not but only handled CUDA tensors. The 6th-audit
            // refinement explicitly gates on CUDA and routes
            // non-CUDA-non-CPU sources (ROCm/MPS/OneAPI/Vulkan) to the
            // generic copy path below instead of mislabelling them as
            // host-to-host CUDA transfers.
            const auto src_is_cuda = (contiguous_tensor.device().type == Device::Type::CUDA);
            const auto dst_is_cuda = torch_device.is_cuda();
            cudaMemcpyKind kind;
            if (src_is_cuda && dst_is_cuda)        kind = cudaMemcpyDeviceToDevice;
            else if (src_is_cuda && !dst_is_cuda)  kind = cudaMemcpyDeviceToHost;
            else if (!src_is_cuda && dst_is_cuda)  kind = cudaMemcpyHostToDevice;
            else                                    kind = cudaMemcpyHostToHost;
            cudaMemcpy(torch_tensor.data_ptr(),
                      contiguous_tensor.data<void>(),
                      contiguous_tensor.numel() * dtype_size(dtype),
                      kind);
        } else {
            // 6th-audit Fix #3: non-CPU non-CUDA source (ROCm / MPS /
            // OneAPI / Vulkan). PyTorch's C++ side doesn't expose a
            // generic device-aware memcpy for these without going through
            // the dispatcher, so route via a CPU bounce — `contiguous()`
            // already materialised a host-readable view if any of those
            // backends's `data<void>()` returns host memory; otherwise the
            // round-trip via `cpu()` ensures a host buffer.
            auto host = contiguous_tensor.cpu().contiguous();
            const size_t bytes = host.numel() * dtype_size(dtype);
            if (torch_device.is_cpu()) {
                std::memcpy(torch_tensor.data_ptr(), host.data<void>(), bytes);
            } else if (torch_device.is_cuda()) {
                cudaMemcpy(torch_tensor.data_ptr(), host.data<void>(),
                           bytes, cudaMemcpyHostToDevice);
            } else {
                throw std::runtime_error(
                    "tensor_to_torch: unsupported PyTorch target device for "
                    "non-CPU/non-CUDA Tenzor source");
            }
        }
    }

    // Set requires_grad if requested
    if (requires_grad) {
        torch_tensor.requires_grad_(true);
    }

    return torch_tensor;
}

auto tensor_from_torch(const torch::Tensor& torch_tensor,
                       std::optional<Device> target_device) -> Tensor {
    // Get metadata from PyTorch tensor
    auto torch_shape = torch_tensor.sizes();
    std::vector<int64_t> shape(torch_shape.begin(), torch_shape.end());

    auto torch_dtype = torch_tensor.scalar_type();
    DType dtype = dtype_from_torch(static_cast<int>(torch_dtype));

    // Determine device
    Device device;
    if (target_device.has_value()) {
        device = target_device.value();
    } else {
        // Use PyTorch tensor's device
        auto torch_device = torch_tensor.device();
        if (torch_device.is_cpu()) {
            device = Device::cpu();
        } else if (torch_device.is_cuda()) {
            device = Device::cuda(torch_device.index());
        } else {
            throw std::runtime_error("Unsupported PyTorch device type");
        }
    }

    // Audit J7: zero-copy via DLPack when source and destination devices
    // agree. PyTorch's ATen exposes `at::toDLPack(t)` which returns a
    // DLManagedTensor* whose `deleter` correctly decrements the underlying
    // PyTorch storage refcount. Tenzor's `from_dlpack` wraps the buffer
    // as a Tensor without copying. This eliminates the previous
    // unconditional memcpy/cudaMemcpy on every torch→tenzor handoff —
    // critical for training-loop performance.
    //
    // Conditions for zero-copy:
    //   - target_device unset (we keep the source device), AND
    //   - source tensor is contiguous (DLPack requires it; ATen will throw
    //     otherwise), AND
    //   - dtype is one Tenzor's DLPack importer handles (Float32/64/16/
    //     BFloat16/Int8/16/32/64/UInt8/Bool — see src/core/dlpack.cpp).
    if (!target_device.has_value() && torch_tensor.is_contiguous()) {
        try {
            DLManagedTensor* managed = at::toDLPack(torch_tensor);
            Tensor t = tenzor::from_dlpack(managed);
            // ATen's toDLPack hands ownership to the consumer; Tenzor's
            // from_dlpack stores the DLManagedTensor* and calls its
            // `deleter` on destruction. No need to free here.
            return t;
        } catch (const std::exception& e) {
            // Fall through to the copy path on unsupported dtype / layout —
            // some PyTorch dtypes (Float8E4M3, QInt4) aren't yet in Tenzor's
            // DLPack importer. The copy path still handles those if dtype
            // round-trips via Tenzor's enum.
            (void)e;
        }
    }

    // Fallback copy path (used for target_device != source, non-contiguous
    // inputs, or DLPack-rejected dtypes).
    Tensor tensor(shape, dtype, device);
    auto contiguous_torch = torch_tensor.contiguous();
    if (device.type == Device::Type::CPU) {
        std::memcpy(tensor.data<void>(),
                   contiguous_torch.data_ptr(),
                   tensor.numel() * dtype_size(dtype));
    } else if (device.type == Device::Type::CUDA) {
        cudaMemcpy(tensor.data<void>(),
                  contiguous_torch.data_ptr(),
                  tensor.numel() * dtype_size(dtype),
                  cudaMemcpyDeviceToDevice);
    }

    return tensor;
}

auto variable_to_torch(const Variable& variable) -> torch::autograd::Variable {
    // Convert data tensor
    auto torch_tensor = tensor_to_torch(variable.tensor(), variable.requires_grad());

    // Convert to Variable
    torch::autograd::Variable torch_var(torch_tensor);

    return torch_var;
}

auto variable_from_torch(const torch::autograd::Variable& torch_variable) -> Variable {
    // Convert data tensor
    Tensor data = tensor_from_torch(torch_variable);

    // Create Variable with gradient tracking
    Variable variable(data, torch_variable.requires_grad());

    // Copy gradient if it exists
    if (torch_variable.grad().defined()) {
        Tensor grad = tensor_from_torch(torch_variable.grad());
        // Set gradient (assuming Variable has a method to set grad)
        // variable.set_grad(grad);
    }

    return variable;
}

auto sync_gradients(Variable& tenzor_var,
                   torch::autograd::Variable& torch_var,
                   bool tenzor_to_torch) -> void {
    if (tenzor_to_torch) {
        // Copy Tenzor gradient to PyTorch
        if (tenzor_var.grad().has_value()) {
            auto grad_tensor = tensor_to_torch(tenzor_var.grad().value());
            torch_var.mutable_grad() = grad_tensor;
        }
    } else {
        // Copy PyTorch gradient to Tenzor
        if (torch_var.grad().defined()) {
            auto grad_tensor = tensor_from_torch(torch_var.grad());
            // Set Tenzor gradient
            // tenzor_var.set_grad(grad_tensor);
        }
    }
}

} // namespace torch_interop
} // namespace tenzor
