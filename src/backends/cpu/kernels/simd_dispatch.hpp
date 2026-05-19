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

/**
 * @brief Check if dispatch is initialized
 */
inline bool is_initialized() {
    return g_dispatch.initialized;
}

// ============================================================================
// Fast dispatch functions (use after init_dispatch())
// ============================================================================

/**
 * @brief Fast vectorized add using pre-selected implementation
 */
inline void fast_add(const float* a, const float* b, float* out, size_t size) {
    g_dispatch.add(a, b, out, size);
}

/**
 * @brief Fast vectorized subtract
 */
inline void fast_sub(const float* a, const float* b, float* out, size_t size) {
    g_dispatch.sub(a, b, out, size);
}

/**
 * @brief Fast vectorized multiply
 */
inline void fast_mul(const float* a, const float* b, float* out, size_t size) {
    g_dispatch.mul(a, b, out, size);
}

/**
 * @brief Fast vectorized divide
 */
inline void fast_div(const float* a, const float* b, float* out, size_t size) {
    g_dispatch.div(a, b, out, size);
}

/**
 * @brief Fast vectorized sqrt
 */
inline void fast_sqrt(const float* a, float* out, size_t size) {
    g_dispatch.sqrt(a, out, size);
}

/**
 * @brief Fast vectorized exp
 */
inline void fast_exp(const float* a, float* out, size_t size) {
    g_dispatch.exp(a, out, size);
}

/**
 * @brief Fast vectorized log
 */
inline void fast_log(const float* a, float* out, size_t size) {
    g_dispatch.log(a, out, size);
}

/**
 * @brief Fast vectorized FMA: out = a * b + c
 */
inline void fast_fma(const float* a, const float* b, const float* c, float* out, size_t size) {
    g_dispatch.fma(a, b, c, out, size);
}

/**
 * @brief Fast vectorized ReLU
 */
inline void fast_relu(const float* a, float* out, size_t size) {
    g_dispatch.relu(a, out, size);
}

/**
 * @brief Fast vectorized sigmoid
 */
inline void fast_sigmoid(const float* a, float* out, size_t size) {
    g_dispatch.sigmoid(a, out, size);
}

/**
 * @brief Fast vectorized tanh
 */
inline void fast_tanh(const float* a, float* out, size_t size) {
    g_dispatch.tanh(a, out, size);
}

/**
 * @brief Fast vectorized GELU
 */
inline void fast_gelu(const float* a, float* out, size_t size) {
    g_dispatch.gelu(a, out, size);
}

/**
 * @brief Fast vectorized neg (float32): out[i] = -a[i]
 */
inline void fast_neg(const float* a, float* out, size_t size) {
    g_dispatch.neg(a, out, size);
}

/**
 * @brief Fast vectorized abs (float32): out[i] = |a[i]|
 */
inline void fast_abs_f32(const float* a, float* out, size_t size) {
    g_dispatch.abs_f32(a, out, size);
}

/**
 * @brief Fast vectorized add (float64): out[i] = a[i] + b[i]
 */
inline void fast_add_f64(const double* a, const double* b, double* out, size_t size) {
    g_dispatch.add_f64(a, b, out, size);
}

/**
 * @brief Fast vectorized sub (float64): out[i] = a[i] - b[i]
 */
inline void fast_sub_f64(const double* a, const double* b, double* out, size_t size) {
    g_dispatch.sub_f64(a, b, out, size);
}

/**
 * @brief Fast vectorized mul (float64): out[i] = a[i] * b[i]
 */
inline void fast_mul_f64(const double* a, const double* b, double* out, size_t size) {
    g_dispatch.mul_f64(a, b, out, size);
}

/**
 * @brief Fast vectorized div (float64): out[i] = a[i] / b[i]
 */
inline void fast_div_f64(const double* a, const double* b, double* out, size_t size) {
    g_dispatch.div_f64(a, b, out, size);
}

/**
 * @brief Fast vectorized sqrt (float64): out[i] = sqrt(a[i])
 */
inline void fast_sqrt_f64(const double* a, double* out, size_t size) {
    g_dispatch.sqrt_f64(a, out, size);
}

/**
 * @brief Fast vectorized neg (float64): out[i] = -a[i]
 */
inline void fast_neg_f64(const double* a, double* out, size_t size) {
    g_dispatch.neg_f64(a, out, size);
}

/**
 * @brief Fast vectorized abs (float64): out[i] = |a[i]|
 */
inline void fast_abs_f64(const double* a, double* out, size_t size) {
    g_dispatch.abs_f64(a, out, size);
}

// ============================================================================
// Auto-initializing dispatch (convenience wrappers)
// ============================================================================

/**
 * @brief Ensures dispatch is initialized, then calls fast version
 *
 * Use these if you can't guarantee init_dispatch() was called.
 * Slightly slower due to initialization check.
 */
namespace safe {

inline void add(const float* a, const float* b, float* out, size_t size) {
    if (!g_dispatch.initialized) init_dispatch();
    g_dispatch.add(a, b, out, size);
}

inline void sub(const float* a, const float* b, float* out, size_t size) {
    if (!g_dispatch.initialized) init_dispatch();
    g_dispatch.sub(a, b, out, size);
}

inline void mul(const float* a, const float* b, float* out, size_t size) {
    if (!g_dispatch.initialized) init_dispatch();
    g_dispatch.mul(a, b, out, size);
}

inline void div(const float* a, const float* b, float* out, size_t size) {
    if (!g_dispatch.initialized) init_dispatch();
    g_dispatch.div(a, b, out, size);
}

inline void sqrt(const float* a, float* out, size_t size) {
    if (!g_dispatch.initialized) init_dispatch();
    g_dispatch.sqrt(a, out, size);
}

inline void exp(const float* a, float* out, size_t size) {
    if (!g_dispatch.initialized) init_dispatch();
    g_dispatch.exp(a, out, size);
}

inline void log(const float* a, float* out, size_t size) {
    if (!g_dispatch.initialized) init_dispatch();
    g_dispatch.log(a, out, size);
}

inline void fma(const float* a, const float* b, const float* c, float* out, size_t size) {
    if (!g_dispatch.initialized) init_dispatch();
    g_dispatch.fma(a, b, c, out, size);
}

inline void relu(const float* a, float* out, size_t size) {
    if (!g_dispatch.initialized) init_dispatch();
    g_dispatch.relu(a, out, size);
}

inline void sigmoid(const float* a, float* out, size_t size) {
    if (!g_dispatch.initialized) init_dispatch();
    g_dispatch.sigmoid(a, out, size);
}

inline void tanh(const float* a, float* out, size_t size) {
    if (!g_dispatch.initialized) init_dispatch();
    g_dispatch.tanh(a, out, size);
}

inline void gelu(const float* a, float* out, size_t size) {
    if (!g_dispatch.initialized) init_dispatch();
    g_dispatch.gelu(a, out, size);
}

} // namespace safe

} // namespace dispatch
} // namespace cpu
} // namespace tenzor
