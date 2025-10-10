#include "numpy_interop.hpp"
#include <tenzor/core/storage.hpp>
#include <stdexcept>
#include <cstring>
#include <sstream>

namespace tenzor {
namespace numpy {

// DType to NumPy format mapping
auto dtype_to_numpy_format(DType dtype) -> std::string {
    switch (dtype) {
        case DType::Float32: return py::format_descriptor<float>::format();
        case DType::Float64: return py::format_descriptor<double>::format();
        case DType::Int8: return py::format_descriptor<int8_t>::format();
        case DType::Int16: return py::format_descriptor<int16_t>::format();
        case DType::Int32: return py::format_descriptor<int32_t>::format();
        case DType::Int64: return py::format_descriptor<int64_t>::format();
        case DType::UInt8: return py::format_descriptor<uint8_t>::format();
        case DType::UInt16: return py::format_descriptor<uint16_t>::format();
        case DType::UInt32: return py::format_descriptor<uint32_t>::format();
        case DType::UInt64: return py::format_descriptor<uint64_t>::format();
        case DType::Bool: return py::format_descriptor<bool>::format();
        case DType::Complex64: return py::format_descriptor<std::complex<float>>::format();
        case DType::Complex128: return py::format_descriptor<std::complex<double>>::format();
        default:
            throw std::runtime_error("Unsupported dtype for NumPy conversion: " +
                                   std::string(dtype_name(dtype)));
    }
}

// NumPy dtype to Tenzor DType mapping
auto numpy_dtype_to_tenzor(const py::array& arr) -> DType {
    auto dtype = arr.dtype();
    auto kind = dtype.kind();
    auto itemsize = dtype.itemsize();

    // Floating-point types
    if (kind == 'f') {
        if (itemsize == 4) return DType::Float32;
        if (itemsize == 8) return DType::Float64;
        if (itemsize == 2) return DType::Float16;
    }
    // Signed integer types
    else if (kind == 'i') {
        if (itemsize == 1) return DType::Int8;
        if (itemsize == 2) return DType::Int16;
        if (itemsize == 4) return DType::Int32;
        if (itemsize == 8) return DType::Int64;
    }
    // Unsigned integer types
    else if (kind == 'u') {
        if (itemsize == 1) return DType::UInt8;
        if (itemsize == 2) return DType::UInt16;
        if (itemsize == 4) return DType::UInt32;
        if (itemsize == 8) return DType::UInt64;
    }
    // Boolean type
    else if (kind == 'b') {
        return DType::Bool;
    }
    // Complex types
    else if (kind == 'c') {
        if (itemsize == 8) return DType::Complex64;
        if (itemsize == 16) return DType::Complex128;
    }

    std::ostringstream oss;
    oss << "Unsupported NumPy dtype: kind=" << kind << ", itemsize=" << itemsize;
    throw std::runtime_error(oss.str());
}

auto get_numpy_itemsize(const py::array& arr) -> size_t {
    return arr.dtype().itemsize();
}

auto can_zero_copy_tensor_to_numpy(const Tensor& tensor) -> bool {
    // Zero-copy only possible for CPU tensors that are contiguous
    return tensor.device().type == Device::Type::CPU && tensor.is_contiguous();
}

auto can_zero_copy_numpy_to_tensor(const py::array& arr) -> bool {
    // Zero-copy only if contiguous and C-style (row-major)
    auto flags = arr.flags();
    return (flags & py::array::c_style) && !(flags & py::array::f_style);
}

// Tensor to NumPy conversion
auto tensor_to_numpy(const Tensor& tensor) -> py::array {
    // Get tensor properties
    auto shape = tensor.shape();
    auto strides = tensor.strides();
    auto dtype = tensor.dtype();
    auto device = tensor.device();

    // Convert shape to vector
    std::vector<ssize_t> np_shape(shape.begin(), shape.end());

    // Convert strides from element counts to byte counts
    std::vector<ssize_t> np_strides;
    np_strides.reserve(strides.size());
    size_t element_size = dtype_size(dtype);
    for (auto s : strides) {
        np_strides.push_back(s * element_size);
    }

    // Get NumPy format string
    std::string format = dtype_to_numpy_format(dtype);

    // Handle CUDA tensors - must copy to CPU first
    if (device.type == Device::Type::CUDA) {
        // Copy to CPU
        Tensor cpu_tensor = tensor.cpu();

        // Create NumPy array with copied data
        py::array result(py::dtype(format), np_shape, np_strides);

        // Copy data
        void* src = const_cast<void*>(cpu_tensor.impl()->storage->data());
        void* dst = result.mutable_data();
        size_t total_bytes = cpu_tensor.numel() * element_size;
        std::memcpy(dst, src, total_bytes);

        return result;
    }

    // CPU tensor - attempt zero-copy if contiguous
    if (tensor.is_contiguous()) {
        // Zero-copy path: share memory with tensor
        void* data_ptr = const_cast<void*>(tensor.impl()->storage->data());

        // Create capsule for memory management
        // The capsule will keep the tensor's storage alive by incrementing shared_ptr refcount
        // We need to create a new shared_ptr copy that will be owned by the capsule
        auto storage_ptr = new std::shared_ptr<Storage>(tensor.impl()->storage);

        py::capsule capsule(storage_ptr, [](void* ptr) {
            // Destructor is called when NumPy array is deallocated
            // Delete the shared_ptr copy, which decrements the refcount
            delete static_cast<std::shared_ptr<Storage>*>(ptr);
        });

        // Create NumPy array with shared memory
        return py::array(py::dtype(format), np_shape, np_strides, data_ptr, capsule);
    } else {
        // Non-contiguous CPU tensor - must copy
        Tensor contiguous_tensor = tensor.contiguous();

        // Create NumPy array
        py::array result(py::dtype(format), np_shape);

        // Copy data
        void* src = const_cast<void*>(contiguous_tensor.impl()->storage->data());
        void* dst = result.mutable_data();
        size_t total_bytes = contiguous_tensor.numel() * element_size;
        std::memcpy(dst, src, total_bytes);

        return result;
    }
}

// NumPy to Tensor conversion
auto numpy_to_tensor(py::array arr, Device device) -> Tensor {
    // Get NumPy array properties
    auto dtype = numpy_dtype_to_tenzor(arr);

    // Get shape
    std::vector<int64_t> shape;
    shape.reserve(arr.ndim());
    for (ssize_t i = 0; i < arr.ndim(); ++i) {
        shape.push_back(arr.shape(i));
    }

    // Calculate total elements
    int64_t numel = 1;
    for (auto dim : shape) {
        numel *= dim;
    }

    // Create tensor with requested device
    Tensor tensor(shape, dtype, device);

    // Check if we can do zero-copy (CPU device, contiguous array)
    bool can_zero_copy = (device.type == Device::Type::CPU) &&
                         can_zero_copy_numpy_to_tensor(arr);

    if (can_zero_copy) {
        // Zero-copy path: share memory with NumPy array
        // Note: This is tricky because we need to ensure the NumPy array stays alive
        // For safety, we'll create a shared_ptr that holds a Python reference

        void* numpy_data = arr.mutable_data();
        size_t size_bytes = numel * dtype_size(dtype);

        // For now, we'll copy the data for safety
        // True zero-copy would require custom storage implementation
        // which is complex to implement safely with Python's GIL
        void* tensor_data = tensor.impl()->storage->data();
        std::memcpy(tensor_data, numpy_data, size_bytes);

        return tensor;
    } else {
        // Copy path: copy data from NumPy to tensor

        // Request buffer from NumPy array
        py::buffer_info buf = arr.request();
        void* numpy_data = buf.ptr;

        // Get tensor data pointer
        void* tensor_data = tensor.impl()->storage->data();

        // If array is C-contiguous, we can copy directly
        if (can_zero_copy_numpy_to_tensor(arr)) {
            size_t size_bytes = numel * dtype_size(dtype);
            std::memcpy(tensor_data, numpy_data, size_bytes);
        } else {
            // Non-contiguous array - need to copy element by element
            // or convert to contiguous first
            py::array contiguous = py::array::ensure(arr, py::array::c_style);
            py::buffer_info contiguous_buf = contiguous.request();
            void* contiguous_data = contiguous_buf.ptr;
            size_t size_bytes = numel * dtype_size(dtype);
            std::memcpy(tensor_data, contiguous_data, size_bytes);
        }

        // If target device is CUDA, we need to copy to GPU
        if (device.type == Device::Type::CUDA) {
            // The tensor constructor already placed it on the correct device
            // and the backend will handle the CPU-to-GPU transfer
            // (assuming the backend is properly implemented)
        }

        return tensor;
    }
}

} // namespace numpy
} // namespace tenzor
