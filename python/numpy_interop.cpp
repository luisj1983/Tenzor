#include "numpy_interop.hpp"
#include <tenzor/core/storage.hpp>
#include <tenzor/core/checked_math.hpp>
#include <tenzor/utils/error.hpp>
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
        case DType::FP8_E4M3:
        case DType::FP8_E5M2:
            // R.26: FP8 is a sub-byte float, not a quantized integer — there is
            // no .dequantize() on FP8. Point at the .to(Float32).numpy() path
            // which is the correct (and only) lossless route.
            throw ::tenzor::TypeError(
                std::string("FP8 tensors (dtype ") +
                std::string(dtype_name(dtype)) +
                ") cannot be exposed to NumPy directly because NumPy has no "
                "native FP8 dtype. Cast to Float32 first: "
                "t.to(DType::Float32).numpy()");
        case DType::QInt4x2:
        case DType::QInt8:
        case DType::QUInt8:
            throw ::tenzor::TypeError(
                std::string("Quantized tensors (dtype ") +
                std::string(dtype_name(dtype)) +
                ") cannot be exposed to NumPy directly. Quantized tensors "
                "must be dequantized before calling .numpy() — use "
                "t.dequantize().numpy() or t.to(DType::Float32).numpy() first");
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

    // R.26: detect ml_dtypes.float8_e4m3fn / float8_e5m2 (kind='V', itemsize=1).
    // Without this branch, FP8 NumPy arrays surface as "Unsupported NumPy dtype".
    if (kind == 'V' && itemsize == 1) {
        try {
            auto ml_dtypes = py::module_::import("ml_dtypes");
            auto e4m3 = ml_dtypes.attr("float8_e4m3fn");
            if (dtype.equal(py::dtype::from_args(e4m3))) {
                return DType::FP8_E4M3;
            }
            auto e5m2 = ml_dtypes.attr("float8_e5m2");
            if (dtype.equal(py::dtype::from_args(e5m2))) {
                return DType::FP8_E5M2;
            }
        } catch (const py::error_already_set&) {
            // ml_dtypes not available — fall back to dtype-name string match.
            std::string dtype_name = py::str(dtype);
            if (dtype_name.find("float8_e4m3") != std::string::npos) {
                return DType::FP8_E4M3;
            }
            if (dtype_name.find("float8_e5m2") != std::string::npos) {
                return DType::FP8_E5M2;
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
    if (tensor.device().type != Device::Type::CPU) return false;
    // Check alignment: data pointer must be aligned to dtype size
    auto align = tensor.dtype_size();
    if (align > 0 && reinterpret_cast<uintptr_t>(tensor.data_ptr()) % align != 0) return false;
    // Check for negative strides (not safely representable in NumPy zero-copy)
    for (auto s : tensor.strides()) {
        if (s < 0) return false;
    }
    return true;
}

auto can_zero_copy_numpy_to_tensor(const py::array& arr) -> bool {
    // Zero-copy only if contiguous and C-style (row-major).
    auto flags = arr.flags();
    if (!(flags & py::array::c_style) || (flags & py::array::f_style)) {
        return false;
    }
    // 5th-audit B5: explicitly reject negative-stride (reversed slice) and
    // zero-stride (broadcasted) arrays. A naive memcpy past these strides
    // either reads backwards or repeatedly reads the same byte — both
    // corrupt the resulting tensor. The non-zero-copy path below already
    // forces a contiguous copy via `py::array::ensure(arr, c_style)`.
    for (ssize_t i = 0; i < arr.ndim(); ++i) {
        if (arr.strides(i) <= 0 && arr.shape(i) > 1) {
            return false;
        }
    }
    return true;
}

// Phase 1: Prepare tensor for NumPy (pure C++, GIL not required)
auto prepare_tensor_for_numpy(const Tensor& tensor) -> Tensor {
    if (tensor.device().type != Device::Type::CPU) {
        return tensor.to(Device::cpu());
    }
    return tensor;
}

// Phase 2: Create NumPy array from CPU tensor (requires GIL)
auto create_numpy_array(const Tensor& tensor, DType original_dtype,
                        bool want_no_copy) -> py::array {
    auto shape = tensor.shape();
    auto strides = tensor.strides();
    auto dtype = tensor.dtype();
    size_t element_size = dtype_size(dtype);

    std::vector<ssize_t> np_shape(shape.begin(), shape.end());
    std::vector<ssize_t> np_strides;
    np_strides.reserve(strides.size());
    for (auto s : strides) {
        np_strides.push_back(s * element_size);
    }

    std::string format = dtype_to_numpy_format(dtype);

    // Validate that max accessible offset falls within storage bounds
    int64_t max_offset = tensor.offset();
    for (size_t d = 0; d < shape.size(); ++d) {
        if (shape[d] > 0) {
            max_offset = checked_add(max_offset,
                checked_mul(static_cast<int64_t>(shape[d] - 1),
                            static_cast<int64_t>(strides[d])));
        }
    }
    int64_t storage_elements = static_cast<int64_t>(
        tensor.storage()->size_bytes() / element_size);

    if (max_offset >= storage_elements) {
        // Y.25: NumPy 2.0 ``__array__(copy=False)`` strict contract — when
        // the caller has demanded zero-copy but our strided view exceeds
        // storage bounds (which would require a contiguous fallback), raise
        // ``ValueError`` instead of silently warning-and-copying. W.16
        // closed this for non-CPU / dtype-cast paths but not this branch.
        if (want_no_copy) {
            throw py::value_error(
                "Unable to avoid copy: tensor stride pattern exceeds "
                "storage bounds and requires a contiguous copy");
        }
        PyErr_WarnEx(PyExc_RuntimeWarning,
            "Strided tensor view exceeds storage bounds, "
            "falling back to contiguous copy for NumPy conversion", 1);
        Tensor contiguous = tensor.contiguous();
        py::array result(py::dtype(format), np_shape);
        void* src = const_cast<void*>(contiguous.storage()->data());
        void* dst = result.mutable_data();
        std::memcpy(dst, src, contiguous.numel() * element_size);
        return apply_bfloat16_dtype(result, original_dtype);
    }

    // Account for storage offset
    auto* base_ptr = static_cast<char*>(
        const_cast<void*>(tensor.storage()->data()));
    void* data_ptr = base_ptr + checked_mul(static_cast<int64_t>(tensor.offset()),
                                            static_cast<int64_t>(element_size));

    // 5th-audit B'2: lifetime contract for the zero-copy NumPy array.
    //
    //   - We heap-allocate an `intrusive_ptr<Storage>` that holds one
    //     additional reference to the tensor's storage.
    //   - The capsule owns that allocation; NumPy guarantees the capsule's
    //     deleter fires exactly once when the array is garbage-collected
    //     (CPython destroys the array first, then walks the base chain to
    //     the capsule and runs its destructor).
    //   - The deleter deletes the intrusive_ptr, which decrements the
    //     storage refcount. If the source Tensor has already gone out of
    //     scope by then, this is the last reference and the storage is
    //     freed; otherwise the storage stays alive until the Tensor side
    //     drops its reference.
    //   - There is no double-decrement: `intrusive_ptr` is move-aware and
    //     this allocation is the only one with a "ticket" to call decref
    //     here.
    auto* storage_ptr = new intrusive_ptr<Storage>(tensor.storage());
    py::capsule capsule(storage_ptr, [](void* ptr) {
        delete static_cast<intrusive_ptr<Storage>*>(ptr);
    });

    py::array result(py::dtype(format), np_shape, np_strides, data_ptr, capsule);
    return apply_bfloat16_dtype(result, original_dtype);
}

// Tensor to NumPy conversion (convenience wrapper)
auto tensor_to_numpy(const Tensor& tensor) -> py::array {
    Tensor cpu_tensor = prepare_tensor_for_numpy(tensor);
    return create_numpy_array(cpu_tensor, tensor.dtype());
}

// NumPy to Tensor conversion
auto numpy_to_tensor(py::array arr, Device device) -> Tensor {
    // 5th-audit B'4: reject object-dtype arrays at the entry. They have
    // PyObject* elements (not raw bytes), so a generic dtype-mapping error
    // deeper in the call would surface as "Unsupported NumPy dtype" without
    // context. Catch them here with a precise message.
    if (arr.dtype().kind() == 'O') {
        throw std::invalid_argument(
            "NumPy object-dtype arrays are not supported. Convert to a "
            "numeric dtype first (e.g. arr.astype(np.float32)).");
    }
    // Warn about Fortran-contiguous (column-major) arrays — data will be copied
    // as row-major which may silently transpose the data layout
    auto flags = arr.flags();
    if ((flags & py::array::f_style) && !(flags & py::array::c_style) && arr.ndim() > 1) {
        PyErr_WarnEx(PyExc_UserWarning,
            "Converting a Fortran-contiguous (column-major) NumPy array to a "
            "row-major Tenzor tensor. The data will be copied in C order, which "
            "may not match the original memory layout. Use np.ascontiguousarray() "
            "to explicitly convert before passing.", 1);
    }

    // 5th-audit B'8: on a misaligned source pointer, force a contiguous
    // (and therefore freshly aligned) copy. Pre-fix we issued a warning and
    // then memcpy'd from the misaligned buffer, which is undefined behaviour
    // on strict-alignment hardware (ARMv7, SPARC, some RISC-V). On x86-64
    // misalignment is "merely" slow, but the new path is uniformly safe.
    auto itemsize = arr.dtype().itemsize();
    if (itemsize > 1 && reinterpret_cast<uintptr_t>(arr.data()) % itemsize != 0) {
        arr = py::array::ensure(arr, py::array::c_style);
    }

    // Get NumPy array properties
    auto dtype = numpy_dtype_to_tenzor(arr);

    // 5th-audit B4: explicit 0-dim handling. `np.array(3.14)` has ndim==0
    // and an empty shape vector; the loops below would happily produce a
    // numel=1 tensor with shape `{}`, which is the *correct* representation
    // of a numpy scalar, but make the contract explicit and document it so
    // future maintainers don't "fix" the empty-shape path away.
    std::vector<int64_t> shape;
    shape.reserve(arr.ndim());
    if (arr.ndim() == 0) {
        // 0-D scalar array: leave `shape` empty; tenzor::Tensor with empty
        // shape vector is a 0-d (scalar) tensor with numel==1.
    } else {
        for (ssize_t i = 0; i < arr.ndim(); ++i) {
            shape.push_back(arr.shape(i));
        }
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
    void* tensor_data = tensor.storage()->data();

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
