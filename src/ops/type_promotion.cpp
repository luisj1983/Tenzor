#include "tenzor/ops/type_promotion.hpp"

namespace tenzor {

namespace {

// Priority ordering for type promotion (higher = wider type)
// Maps DType to a priority used for promotion decisions.
constexpr int dtype_priority(DType dt) {
    switch (dt) {
        case DType::Bool:       return 0;
        case DType::UInt8:      return 1;
        case DType::Int8:       return 2;
        case DType::UInt16:     return 3;
        case DType::Int16:      return 4;
        case DType::UInt32:     return 5;
        case DType::Int32:      return 6;
        case DType::UInt64:     return 7;
        case DType::Int64:      return 8;
        case DType::Float16:    return 9;
        case DType::BFloat16:   return 10;
        case DType::Float32:    return 11;
        case DType::Float64:    return 12;
        case DType::Complex64:  return 13;
        case DType::Complex128: return 14;
        default:                return -1;
    }
}

constexpr bool is_floating(DType dt) {
    return dt == DType::Float16 || dt == DType::BFloat16 ||
           dt == DType::Float32 || dt == DType::Float64;
}

constexpr bool is_complex(DType dt) {
    return dt == DType::Complex64 || dt == DType::Complex128;
}

constexpr bool is_integer(DType dt) {
    return dt == DType::Int8 || dt == DType::Int16 || dt == DType::Int32 || dt == DType::Int64 ||
           dt == DType::UInt8 || dt == DType::UInt16 || dt == DType::UInt32 || dt == DType::UInt64;
}

} // anonymous namespace

auto promote_types(DType a, DType b) -> DType {
    // Same type: no promotion needed
    if (a == b) return a;

    // Bool promotes to anything
    if (a == DType::Bool) return b;
    if (b == DType::Bool) return a;

    // Complex wins over everything
    if (is_complex(a) && is_complex(b)) {
        return dtype_priority(a) >= dtype_priority(b) ? a : b;
    }
    if (is_complex(a)) {
        // Promote complex precision if other is wider float
        if (b == DType::Float64) return DType::Complex128;
        return a;
    }
    if (is_complex(b)) {
        if (a == DType::Float64) return DType::Complex128;
        return b;
    }

    // Float wins over integer.
    // Float16/BFloat16 can only represent integers up to 2048 exactly
    // (mantissa is 10/7 bits respectively), so any integer type mixed with
    // Float16/BFloat16 promotes to Float32 to avoid silent precision loss.
    // Examples: Float16 + Int32 -> Float32, BFloat16 + Int64 -> Float32.
    // This matches NumPy/PyTorch semantics (torch.result_type).
    if (is_floating(a) && is_integer(b)) {
        if (a == DType::Float16 || a == DType::BFloat16) {
            return DType::Float32;
        }
        return a;
    }
    if (is_floating(b) && is_integer(a)) {
        if (b == DType::Float16 || b == DType::BFloat16) {
            return DType::Float32;
        }
        return b;
    }

    // Both floating: promote to wider
    if (is_floating(a) && is_floating(b)) {
        return dtype_priority(a) >= dtype_priority(b) ? a : b;
    }

    // Both integer: promote to wider
    if (is_integer(a) && is_integer(b)) {
        return dtype_priority(a) >= dtype_priority(b) ? a : b;
    }

    // Fallback: take the higher priority type
    return dtype_priority(a) >= dtype_priority(b) ? a : b;
}

auto result_type(const Tensor& a, const Tensor& b) -> DType {
    return promote_types(a.dtype(), b.dtype());
}

auto promote_inputs(const Tensor& a, const Tensor& b) -> std::pair<Tensor, Tensor> {
    DType target = promote_types(a.dtype(), b.dtype());
    Tensor a_out = (a.dtype() == target) ? a : a.to(target);
    Tensor b_out = (b.dtype() == target) ? b : b.to(target);
    return {a_out, b_out};
}

} // namespace tenzor
