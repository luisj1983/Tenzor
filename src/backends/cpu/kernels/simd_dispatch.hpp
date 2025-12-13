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
 */

#pragma once

#include "tenzor/backends/cpu/simd.hpp"
#include "simd_fast_math.hpp"
#include <cstddef>
#include <atomic>
#include <mutex>

namespace tenzor {
namespace cpu {
namespace dispatch {

// ============================================================================
// Function pointer types
// ============================================================================

// Unary operations: out[i] = f(a[i])
using UnaryOp = void(*)(const float*, float*, size_t);

// Binary operations: out[i] = f(a[i], b[i])
using BinaryOp = void(*)(const float*, const float*, float*, size_t);

// Ternary operations: out[i] = f(a[i], b[i], c[i])
using TernaryOp = void(*)(const float*, const float*, const float*, float*, size_t);

// ============================================================================
// Dispatch table
// ============================================================================

struct SIMDDispatch {
    // Math operations
    BinaryOp add;
    BinaryOp sub;
    BinaryOp mul;
    BinaryOp div;
    UnaryOp sqrt;
    UnaryOp exp;
    UnaryOp log;
    TernaryOp fma;

    // Activation functions
    UnaryOp relu;
    UnaryOp sigmoid;
    UnaryOp tanh;
    UnaryOp gelu;

    // Initialization flag
    bool initialized;
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
 */
void init_dispatch();

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
