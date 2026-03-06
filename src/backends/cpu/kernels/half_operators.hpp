/**
 * @file half_operators.hpp
 * @brief Arithmetic operator overloads for Float16 and BFloat16
 *
 * These inline helpers allow Float16/BFloat16 to work with template code that
 * uses arithmetic operators. All operations are performed in Float32 precision.
 *
 * Include this header inside an anonymous namespace (within tenzor::cpu) to
 * prevent ODR violations across translation units.
 */

#pragma once

#include "tenzor/core/dtype.hpp"
#include <cmath>

// ============================================================================
// Float16 Arithmetic Operators
// ============================================================================

inline tenzor::Float16 operator+(const tenzor::Float16& a, const tenzor::Float16& b) {
    return tenzor::Float16(static_cast<float>(a) + static_cast<float>(b));
}

inline tenzor::Float16 operator-(const tenzor::Float16& a, const tenzor::Float16& b) {
    return tenzor::Float16(static_cast<float>(a) - static_cast<float>(b));
}

inline tenzor::Float16 operator*(const tenzor::Float16& a, const tenzor::Float16& b) {
    return tenzor::Float16(static_cast<float>(a) * static_cast<float>(b));
}

inline tenzor::Float16 operator/(const tenzor::Float16& a, const tenzor::Float16& b) {
    return tenzor::Float16(static_cast<float>(a) / static_cast<float>(b));
}

inline tenzor::Float16& operator+=(tenzor::Float16& a, const tenzor::Float16& b) {
    a = tenzor::Float16(static_cast<float>(a) + static_cast<float>(b));
    return a;
}

inline tenzor::Float16& operator-=(tenzor::Float16& a, const tenzor::Float16& b) {
    a = tenzor::Float16(static_cast<float>(a) - static_cast<float>(b));
    return a;
}

inline tenzor::Float16& operator*=(tenzor::Float16& a, const tenzor::Float16& b) {
    a = tenzor::Float16(static_cast<float>(a) * static_cast<float>(b));
    return a;
}

inline tenzor::Float16& operator/=(tenzor::Float16& a, const tenzor::Float16& b) {
    a = tenzor::Float16(static_cast<float>(a) / static_cast<float>(b));
    return a;
}

// ============================================================================
// BFloat16 Arithmetic Operators
// ============================================================================

inline tenzor::BFloat16 operator+(const tenzor::BFloat16& a, const tenzor::BFloat16& b) {
    return tenzor::BFloat16(static_cast<float>(a) + static_cast<float>(b));
}

inline tenzor::BFloat16 operator-(const tenzor::BFloat16& a, const tenzor::BFloat16& b) {
    return tenzor::BFloat16(static_cast<float>(a) - static_cast<float>(b));
}

inline tenzor::BFloat16 operator*(const tenzor::BFloat16& a, const tenzor::BFloat16& b) {
    return tenzor::BFloat16(static_cast<float>(a) * static_cast<float>(b));
}

inline tenzor::BFloat16 operator/(const tenzor::BFloat16& a, const tenzor::BFloat16& b) {
    return tenzor::BFloat16(static_cast<float>(a) / static_cast<float>(b));
}

inline tenzor::BFloat16& operator+=(tenzor::BFloat16& a, const tenzor::BFloat16& b) {
    a = tenzor::BFloat16(static_cast<float>(a) + static_cast<float>(b));
    return a;
}

inline tenzor::BFloat16& operator-=(tenzor::BFloat16& a, const tenzor::BFloat16& b) {
    a = tenzor::BFloat16(static_cast<float>(a) - static_cast<float>(b));
    return a;
}

inline tenzor::BFloat16& operator*=(tenzor::BFloat16& a, const tenzor::BFloat16& b) {
    a = tenzor::BFloat16(static_cast<float>(a) * static_cast<float>(b));
    return a;
}

inline tenzor::BFloat16& operator/=(tenzor::BFloat16& a, const tenzor::BFloat16& b) {
    a = tenzor::BFloat16(static_cast<float>(a) / static_cast<float>(b));
    return a;
}

// ============================================================================
// Math Helper Templates
// ============================================================================

template<typename T>
inline T safe_sqrt(const T& x) {
    return std::sqrt(x);
}

template<>
inline tenzor::Float16 safe_sqrt<tenzor::Float16>(const tenzor::Float16& x) {
    return tenzor::Float16(std::sqrt(static_cast<float>(x)));
}

template<>
inline tenzor::BFloat16 safe_sqrt<tenzor::BFloat16>(const tenzor::BFloat16& x) {
    return tenzor::BFloat16(std::sqrt(static_cast<float>(x)));
}
