/**
 * @file simd_dispatch.hpp
 * @brief Function pointer dispatch system for SIMD operations
 *
 * Eliminates runtime overhead of CPU feature checking by using function pointers
 * that are initialized once at startup based on detected CPU features.
 *
 * This provides:
 * - Zero-overhead dispatch after initialization
 * - Clean fallback hierarchy (AVX-512 → AVX2 → SSE → Scalar)
 * - Thread-safe initialization
 * - Runtime ISA level override via TENZOR_FORCE_SIMD_LEVEL env var
 */

#pragma once

#include "tenzor/backends/cpu/simd.hpp"
#include <cstddef>
#include <string>

namespace tenzor {
namespace cpu {
namespace dispatch {

// ============================================================================
// Function pointer types
// ============================================================================

// Unary operations (float32): out[i] = f(a[i])
using UnaryOp = void(*)(const float*, float*, size_t);

// Binary operations (float32): out[i] = f(a[i], b[i])
using BinaryOp = void(*)(const float*, const float*, float*, size_t);

// Ternary operations (float32): out[i] = f(a[i], b[i], c[i])
using TernaryOp = void(*)(const float*, const float*, const float*, float*, size_t);

// Unary operations (float64): out[i] = f(a[i])
using UnaryOpF64 = void(*)(const double*, double*, size_t);

// Binary operations (float64): out[i] = f(a[i], b[i])
using BinaryOpF64 = void(*)(const double*, const double*, double*, size_t);

// ============================================================================
// Dispatch table
// ============================================================================

struct SIMDDispatch {
    // Float32 math operations
    BinaryOp add;
    BinaryOp sub;
    BinaryOp mul;
    BinaryOp div;
    UnaryOp sqrt;
    UnaryOp exp;
    UnaryOp log;
    TernaryOp fma;

    // Float32 unary ops (in scope: neg, abs)
    UnaryOp neg;
    UnaryOp abs_f32;

    // Float32 activation functions
    UnaryOp relu;
    UnaryOp sigmoid;
    UnaryOp tanh;
    UnaryOp gelu;

    // Float64 binary ops
    BinaryOpF64 add_f64;
    BinaryOpF64 sub_f64;
    BinaryOpF64 mul_f64;
    BinaryOpF64 div_f64;

    // Float64 unary ops
    UnaryOpF64 sqrt_f64;
    UnaryOpF64 neg_f64;
    UnaryOpF64 abs_f64;

    // Initialization flag
    bool initialized;

    // Active ISA level string (set during init, useful for testing)
    const char* simd_level;
};

// Global dispatch table
extern SIMDDispatch g_dispatch;

// ============================================================================
// Initialization
// ============================================================================

/**
 * @brief Initialize the dispatch table based on CPU features
 *
 * This function is thread-safe and will only initialize once.
 * Should be called early in program startup.
 *
 * Respects TENZOR_FORCE_SIMD_LEVEL env var:
 *   "avx512"  — force AVX-512 (only if hardware supports it)
 *   "avx2"    — force AVX2 (only if hardware supports it)
 *   "sse2"    — force SSE2 scalar-width fallback
 *   "scalar"  — force scalar fallback
 */
void init_dispatch();

/**
 * @brief Re-initialize the dispatch table, re-reading TENZOR_FORCE_SIMD_LEVEL
 *
 * Intended for testing: allows forcing a different SIMD level mid-process.
 * Not thread-safe with concurrent kernel calls — only use from single-threaded
 * test code after setting/unsetting the env var.
 */
void reinit_dispatch();

/**
 * @brief Return the active SIMD level string ("avx512", "avx2", "sse2", "scalar")
 *
 * Returns the level that was selected when the dispatch table was last
 * initialised. Valid after init_dispatch() or reinit_dispatch().
 */
std::string get_simd_level();

} // namespace dispatch
} // namespace cpu
} // namespace tenzor
