#include "numpy_interop.hpp"
#include <tenzor/core/storage.hpp>
#include <stdexcept>
#include <cstring>
#include <sstream>

namespace tenzor {
namespace numpy {

namespace {

// Try to get ml_dtypes.bfloat16 numpy dtype. Returns the dtype object on
// success, or py::none() if ml_dtypes is not installed.
auto get_ml_dtypes_bfloat16() -> py::object {
    static py::object cached = py::none();
    static bool tried = false;
    if (!tried) {
        tried = true;
        try {
            auto ml_dtypes = py::module_::import("ml_dtypes");
            cached = py::dtype::from_args(ml_dtypes.attr("bfloat16"));
        } catch (const py::error_already_set&) {
            // ml_dtypes not installed — cached stays as py::none()
        }
    }
    return cached;
}

// If the tensor dtype is BFloat16, try to reinterpret a uint16 NumPy array
// as ml_dtypes.bfloat16. Issues a warning if ml_dtypes is unavailable.
auto apply_bfloat16_dtype(py::array result, DType dtype) -> py::array {
    if (dtype != DType::BFloat16) {
        return result;
    }
    auto bf16_dtype = get_ml_dtypes_bfloat16();
    if (!bf16_dtype.is_none()) {
        // View the uint16 data as bfloat16 (same binary layout, zero-copy)
        return result.attr("view")(bf16_dtype);
    }
    PyErr_WarnEx(PyExc_UserWarning,
        "ml_dtypes package not available; BFloat16 tensor is exposed as raw "
        "uint16 bits. Install ml_dtypes for proper bfloat16 NumPy support: "
        "pip install ml_dtypes", 1);
    return result;
}

} // anonymous namespace

// DType to NumPy format mapping
auto dtype_to_numpy_format(DType dtype) -> std::string {
    switch (dtype) {
        case DType::Float32: return py::format_descriptor<float>::format();
        case DType::Float64: return py::format_descriptor<double>::format();
        case DType::Float16: return "e";  // NumPy native float16 format string
        // BFloat16: format string is always uint16 (same binary layout).
        // The actual ml_dtypes dtype is applied in tensor_to_numpy() when available.
        case DType::BFloat16: return py::format_descriptor<uint16_t>::format();
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
    }
    throw std::runtime_error("Unsupported dtype for NumPy conversion: " +
                           std::string(dtype_name(dtype)));
}

// NumPy dtype to Tenzor DType mapping
auto numpy_dtype_to_tenzor(const py::array& arr) -> DType {
    auto dtype = arr.dtype();
    auto kind = dtype.kind();
    auto itemsize = dtype.itemsize();

    // Check for BFloat16 (ml_dtypes.bfloat16 shows as kind='V', itemsize=2)
    if (kind == 'V' && itemsize == 2) {
        try {
            auto ml_dtypes = py::module_::import("ml_dtypes");
            auto bf16_dtype = ml_dtypes.attr("bfloat16");
            if (dtype.equal(py::dtype::from_args(bf16_dtype))) {
                return DType::BFloat16;
            }
        } catch (const py::error_already_set&) {
            // ml_dtypes not available — fall back to string matching
            std::string dtype_name = py::str(dtype);
            if (dtype_name.find("bfloat16") != std::string::npos) {
                return DType::BFloat16;
            }
        }
    }

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
    oss << "Unsupported NumPy dtype: kind=" << kind << ", itemsize=" << itemsize
        << ", name=" << py::str(dtype).cast<std::string>();
    throw std::runtime_error(oss.str());
}

auto get_numpy_itemsize(const py::array& arr) -> size_t {
    return arr.dtype().itemsize();
}

auto can_zero_copy_tensor_to_numpy(const Tensor& tensor) -> bool {
    // Zero-copy possible for all CPU tensors (NumPy supports strided arrays)
    return tensor.device().type == Device::Type::CPU;
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

        return apply_bfloat16_dtype(result, dtype);
    }

    // CPU tensor - zero-copy path sharing storage with the tensor.
    // Works for both contiguous and non-contiguous (strided) tensors since
    // NumPy natively supports strided arrays.
    {
        // Validate that max accessible offset falls within storage bounds
        int64_t max_offset = tensor.impl()->offset;
        for (size_t d = 0; d < shape.size(); ++d) {
            if (shape[d] > 0) {
                max_offset += (shape[d] - 1) * strides[d];
            }
        }
        int64_t storage_elements = static_cast<int64_t>(
            tensor.impl()->storage->size_bytes() / element_size);

        if (max_offset >= storage_elements) {
            // Strided view exceeds storage — fall back to contiguous copy.
            // This can happen with advanced slicing that creates views with
            // strides exceeding the underlying storage bounds.
            PyErr_WarnEx(PyExc_RuntimeWarning,
                "Strided tensor view exceeds storage bounds, "
                "falling back to contiguous copy for NumPy conversion", 1);
            Tensor contiguous = tensor.contiguous();
            py::array result(py::dtype(format), np_shape);
            void* src = const_cast<void*>(contiguous.impl()->storage->data());
            void* dst = result.mutable_data();
            std::memcpy(dst, src, contiguous.numel() * element_size);
            return apply_bfloat16_dtype(result, dtype);
        }

        // Account for storage offset
        auto* base_ptr = static_cast<char*>(
            const_cast<void*>(tensor.impl()->storage->data()));
        void* data_ptr = base_ptr + tensor.impl()->offset * element_size;

        // Create capsule that keeps the tensor's storage alive via shared_ptr refcount.
        auto* storage_ptr = new std::shared_ptr<Storage>(tensor.impl()->storage);
        py::capsule capsule(storage_ptr, [](void* ptr) {
            delete static_cast<std::shared_ptr<Storage>*>(ptr);
        });

        // Create NumPy array with shared memory and original strides
        py::array result(py::dtype(format), np_shape, np_strides, data_ptr, capsule);
        return apply_bfloat16_dtype(result, dtype);
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

    // Always copy data from NumPy to Tensor for memory safety
    // (NumPy and Tensor have independent lifetime management)
    void* tensor_data = tensor.impl()->storage->data();

    if (can_zero_copy_numpy_to_tensor(arr)) {
        // C-contiguous array — direct memcpy
        size_t size_bytes = numel * dtype_size(dtype);
        std::memcpy(tensor_data, arr.data(), size_bytes);
    } else {
        // Non-contiguous array — make contiguous copy first
        py::array contiguous = py::array::ensure(arr, py::array::c_style);
        py::buffer_info contiguous_buf = contiguous.request();
        size_t size_bytes = numel * dtype_size(dtype);
        std::memcpy(tensor_data, contiguous_buf.ptr, size_bytes);
    }

    return tensor;
}

} // namespace numpy
} // namespace tenzor
