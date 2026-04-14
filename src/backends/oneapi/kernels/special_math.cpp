/**
 * @file special_math.cpp
 * @brief Native OneAPI/SYCL implementations of special math functions.
 *
 * Replaces the previous CPU-roundtrip fallbacks in oneapi_kernel_registry.cpp.
 * SYCL provides sycl::tgamma and sycl::lgamma natively; the rest (Bessel
 * functions, ErfInv, Sinc, Zeta, Polygamma, BetaInc) use the same Cephes /
 * Abramowitz polynomial approximations as the CPU backend, ported to
 * device-side functions that the SYCL kernel lambdas call directly.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#ifdef TENZOR_HAS_ONEMKL
#include <oneapi/mkl/vm.hpp>
#endif

namespace tenzor {
namespace oneapi {

// =========================================================================
// Helpers (mirror those in math.cpp)
// =========================================================================

template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

inline float bf16_to_f32_h(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}
inline uint16_t f32_to_bf16_h(float f32) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f32, sizeof(uint32_t));
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

// =========================================================================
// Device-side approximations (callable from SYCL kernel lambdas)
// =========================================================================

inline float digamma_dev_f32(float x) {
    float result = 0.0f;
    if (x < 0.5f) {
        float y = 1.0f - x;
        float r = 0.0f;
        while (y < 7.0f) { r -= 1.0f / y; y += 1.0f; }
        float y2 = 1.0f / (y * y);
        r += sycl::log(y) - 0.5f / y
            - y2 * (0.0833333333f - y2 * (0.00833333333f - y2 * (0.00396825397f
            - y2 * (0.00416666667f - y2 * 0.00757575758f))));
        return r - 3.14159265358979f / sycl::tan(3.14159265358979f * x);
    }
    while (x < 7.0f) { result -= 1.0f / x; x += 1.0f; }
    float x2 = 1.0f / (x * x);
    result += sycl::log(x) - 0.5f / x
            - x2 * (0.0833333333f - x2 * (0.00833333333f - x2 * (0.00396825397f
            - x2 * (0.00416666667f - x2 * 0.00757575758f))));
    return result;
}
inline double digamma_dev_f64(double x) {
    double result = 0.0;
    if (x < 0.5) {
        double y = 1.0 - x;
        double r = 0.0;
        while (y < 7.0) { r -= 1.0 / y; y += 1.0; }
        double y2 = 1.0 / (y * y);
        r += sycl::log(y) - 0.5 / y
            - y2 * (1.0/12.0 - y2 * (1.0/120.0 - y2 * (1.0/252.0
            - y2 * (1.0/240.0 - y2 * (1.0/132.0)))));
        return r - 3.14159265358979323846 / sycl::tan(3.14159265358979323846 * x);
    }
    while (x < 7.0) { result -= 1.0 / x; x += 1.0; }
    double x2 = 1.0 / (x * x);
    result += sycl::log(x) - 0.5 / x
            - x2 * (1.0/12.0 - x2 * (1.0/120.0 - x2 * (1.0/252.0
            - x2 * (1.0/240.0 - x2 * (1.0/132.0)))));
    return result;
}

// Bessel J0 — Abramowitz & Stegun polynomial approximation
inline float bessel_j0_dev_f32(float x_in) {
    float x = sycl::fabs(x_in);
    if (x <= 3.0f) {
        float y = x * x / 9.0f;
        return 1.0f - y * (2.2499997f - y * (1.2656208f - y * (0.3163866f
               - y * (0.0444479f - y * (0.0039444f - y * 0.0002100f)))));
    }
    float ax = 3.0f / x;
    float p = 0.79788456f - ax * (0.00000077f + ax * (0.00552740f + ax * (0.00009512f
             - ax * (0.00137237f - ax * (0.00072805f - ax * 0.00014476f)))));
    float q = -0.04166397f - ax * (0.00003954f - ax * (0.00262573f - ax * (0.00054125f
             + ax * (0.00029333f - ax * (0.00013558f)))));
    float z = x - 0.785398164f;
    return sycl::sqrt(2.0f / (3.14159265358979f * x)) * (p * sycl::cos(z) - q * sycl::sin(z));
}
inline double bessel_j0_dev_f64(double x_in) {
    double x = sycl::fabs(x_in);
    if (x <= 3.0) {
        double y = x * x / 9.0;
        return 1.0 - y * (2.2499997 - y * (1.2656208 - y * (0.3163866
               - y * (0.0444479 - y * (0.0039444 - y * 0.0002100)))));
    }
    double ax = 3.0 / x;
    double p = 0.79788456 - ax * (0.00000077 + ax * (0.00552740 + ax * (0.00009512
             - ax * (0.00137237 - ax * (0.00072805 - ax * 0.00014476)))));
    double q = -0.04166397 - ax * (0.00003954 - ax * (0.00262573 - ax * (0.00054125
             + ax * (0.00029333 - ax * (0.00013558)))));
    double z = x - 0.785398164;
    return sycl::sqrt(2.0 / (3.14159265358979323846 * x)) * (p * sycl::cos(z) - q * sycl::sin(z));
}

// Bessel J1
inline float bessel_j1_dev_f32(float x_in) {
    float sign = (x_in < 0.0f) ? -1.0f : 1.0f;
    float x = sycl::fabs(x_in);
    if (x <= 3.0f) {
        float y = x * x / 9.0f;
        return sign * x * (0.5f - y * (0.56249985f - y * (0.21093573f - y * (0.03954289f
               - y * (0.00443319f - y * (0.00031761f - y * 0.00001109f))))));
    }
    float ax = 3.0f / x;
    float p = 0.79788456f + ax * (0.00000156f + ax * (0.01659667f + ax * (0.00017105f
             - ax * (0.00249511f + ax * (0.00113653f - ax * 0.00020033f)))));
    float q = 0.12499612f + ax * (0.00005650f - ax * (0.00637879f + ax * (0.00074348f
             + ax * (0.00079824f - ax * (0.00029166f)))));
    float z = x - 2.356194491f;
    return sign * sycl::sqrt(2.0f / (3.14159265358979f * x)) * (p * sycl::cos(z) - q * sycl::sin(z));
}
inline double bessel_j1_dev_f64(double x_in) {
    double sign = (x_in < 0.0) ? -1.0 : 1.0;
    double x = sycl::fabs(x_in);
    if (x <= 3.0) {
        double y = x * x / 9.0;
        return sign * x * (0.5 - y * (0.56249985 - y * (0.21093573 - y * (0.03954289
               - y * (0.00443319 - y * (0.00031761 - y * 0.00001109))))));
    }
    double ax = 3.0 / x;
    double p = 0.79788456 + ax * (0.00000156 + ax * (0.01659667 + ax * (0.00017105
             - ax * (0.00249511 + ax * (0.00113653 - ax * 0.00020033)))));
    double q = 0.12499612 + ax * (0.00005650 - ax * (0.00637879 + ax * (0.00074348
             + ax * (0.00079824 - ax * (0.00029166)))));
    double z = x - 2.356194491;
    return sign * sycl::sqrt(2.0 / (3.14159265358979323846 * x)) * (p * sycl::cos(z) - q * sycl::sin(z));
}

// Bessel Y0
inline float bessel_y0_dev_f32(float x) {
    if (x <= 0.0f) return -INFINITY;
    if (x <= 3.0f) {
        float y = x * x / 9.0f;
        // y0 needs j0 inside the small-x branch
        return (2.0f / 3.14159265358979f) * sycl::log(x / 2.0f) * bessel_j0_dev_f32(x)
               + 0.36746691f + y * (0.60559366f - y * (0.74350384f - y * (0.25300117f
               - y * (0.04261214f - y * (0.00427916f - y * 0.00024846f)))));
    }
    float ax = 3.0f / x;
    float p = 0.79788456f - ax * (0.00000077f + ax * (0.00552740f + ax * (0.00009512f
             - ax * (0.00137237f - ax * (0.00072805f - ax * 0.00014476f)))));
    float q = -0.04166397f - ax * (0.00003954f - ax * (0.00262573f - ax * (0.00054125f
             + ax * (0.00029333f - ax * (0.00013558f)))));
    float z = x - 0.785398164f;
    return sycl::sqrt(2.0f / (3.14159265358979f * x)) * (p * sycl::sin(z) + q * sycl::cos(z));
}
inline double bessel_y0_dev_f64(double x) {
    if (x <= 0.0) return -INFINITY;
    if (x <= 3.0) {
        double y = x * x / 9.0;
        return (2.0 / 3.14159265358979323846) * sycl::log(x / 2.0) * bessel_j0_dev_f64(x)
               + 0.36746691 + y * (0.60559366 - y * (0.74350384 - y * (0.25300117
               - y * (0.04261214 - y * (0.00427916 - y * 0.00024846)))));
    }
    double ax = 3.0 / x;
    double p = 0.79788456 - ax * (0.00000077 + ax * (0.00552740 + ax * (0.00009512
             - ax * (0.00137237 - ax * (0.00072805 - ax * 0.00014476)))));
    double q = -0.04166397 - ax * (0.00003954 - ax * (0.00262573 - ax * (0.00054125
             + ax * (0.00029333 - ax * (0.00013558)))));
    double z = x - 0.785398164;
    return sycl::sqrt(2.0 / (3.14159265358979323846 * x)) * (p * sycl::sin(z) + q * sycl::cos(z));
}

// Bessel Y1
inline float bessel_y1_dev_f32(float x) {
    if (x <= 0.0f) return -INFINITY;
    if (x <= 3.0f) {
        float y = x * x / 9.0f;
        return (2.0f / 3.14159265358979f) * (sycl::log(x / 2.0f) * bessel_j1_dev_f32(x) - 1.0f / x)
               + x * (0.02635537f + y * (-0.04985710f + y * (-0.00121547f + y * (0.00127120f
               - y * (0.00023895f + y * (0.00002535f))))));
    }
    float ax = 3.0f / x;
    float p = 0.79788456f + ax * (0.00000156f + ax * (0.01659667f + ax * (0.00017105f
             - ax * (0.00249511f + ax * (0.00113653f - ax * 0.00020033f)))));
    float q = 0.12499612f + ax * (0.00005650f - ax * (0.00637879f + ax * (0.00074348f
             + ax * (0.00079824f - ax * (0.00029166f)))));
    float z = x - 2.356194491f;
    return sycl::sqrt(2.0f / (3.14159265358979f * x)) * (p * sycl::sin(z) + q * sycl::cos(z));
}
inline double bessel_y1_dev_f64(double x) {
    if (x <= 0.0) return -INFINITY;
    if (x <= 3.0) {
        double y = x * x / 9.0;
        return (2.0 / 3.14159265358979323846) * (sycl::log(x / 2.0) * bessel_j1_dev_f64(x) - 1.0 / x)
               + x * (0.02635537 + y * (-0.04985710 + y * (-0.00121547 + y * (0.00127120
               - y * (0.00023895 + y * (0.00002535))))));
    }
    double ax = 3.0 / x;
    double p = 0.79788456 + ax * (0.00000156 + ax * (0.01659667 + ax * (0.00017105
             - ax * (0.00249511 + ax * (0.00113653 - ax * 0.00020033)))));
    double q = 0.12499612 + ax * (0.00005650 - ax * (0.00637879 + ax * (0.00074348
             + ax * (0.00079824 - ax * (0.00029166)))));
    double z = x - 2.356194491;
    return sycl::sqrt(2.0 / (3.14159265358979323846 * x)) * (p * sycl::sin(z) + q * sycl::cos(z));
}

// Modified Bessel I0 (Abramowitz & Stegun 9.8.1/9.8.2)
inline float bessel_i0_dev_f32(float x_in) {
    float ax = sycl::fabs(x_in);
    if (ax < 3.75f) {
        float t = x_in / 3.75f;
        t = t * t;
        return 1.0f + t * (3.5156229f + t * (3.0899424f + t * (1.2067492f
               + t * (0.2659732f + t * (0.0360768f + t * 0.0045813f)))));
    }
    float t = 3.75f / ax;
    return (sycl::exp(ax) / sycl::sqrt(ax)) * (0.39894228f + t * (0.01328592f
           + t * (0.00225319f - t * (0.00157565f - t * (0.00916281f
           - t * (0.02057706f - t * (0.02635537f - t * (0.01647633f
           - t * 0.00392377f))))))));
}
inline double bessel_i0_dev_f64(double x_in) {
    double ax = sycl::fabs(x_in);
    if (ax < 3.75) {
        double t = x_in / 3.75;
        t = t * t;
        return 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492
               + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
    }
    double t = 3.75 / ax;
    return (sycl::exp(ax) / sycl::sqrt(ax)) * (0.39894228 + t * (0.01328592
           + t * (0.00225319 - t * (0.00157565 - t * (0.00916281
           - t * (0.02057706 - t * (0.02635537 - t * (0.01647633
           - t * 0.00392377))))))));
}

// Modified Bessel I1
inline float bessel_i1_dev_f32(float x_in) {
    float ax = sycl::fabs(x_in);
    float result;
    if (ax < 3.75f) {
        float t = x_in / 3.75f;
        t = t * t;
        result = ax * (0.5f + t * (0.87890594f + t * (0.51498869f + t * (0.15084934f
                 + t * (0.02658733f + t * (0.00301532f + t * 0.00032411f))))));
    } else {
        float t = 3.75f / ax;
        result = (sycl::exp(ax) / sycl::sqrt(ax)) * (0.39894228f - t * (0.03988024f
                 - t * (0.00362018f + t * (0.00163801f - t * (0.01031555f
                 - t * (0.02282967f - t * (0.02895312f - t * (0.01787654f
                 - t * 0.00420059f))))))));
    }
    return (x_in < 0.0f) ? -result : result;
}
inline double bessel_i1_dev_f64(double x_in) {
    double ax = sycl::fabs(x_in);
    double result;
    if (ax < 3.75) {
        double t = x_in / 3.75;
        t = t * t;
        result = ax * (0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934
                 + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))));
    } else {
        double t = 3.75 / ax;
        result = (sycl::exp(ax) / sycl::sqrt(ax)) * (0.39894228 - t * (0.03988024
                 - t * (0.00362018 + t * (0.00163801 - t * (0.01031555
                 - t * (0.02282967 - t * (0.02895312 - t * (0.01787654
                 - t * 0.00420059))))))));
    }
    return (x_in < 0.0) ? -result : result;
}

// erfinv via Winitzki rational approximation + Newton refinement
inline float erfinv_dev_f32(float x) {
    if (x <= -1.0f) return -INFINITY;
    if (x >= 1.0f) return INFINITY;
    if (x == 0.0f) return 0.0f;

    float a = 0.147f;
    float ln1mx2 = sycl::log(1.0f - x * x);
    float t1 = 2.0f / (3.14159265358979f * a) + 0.5f * ln1mx2;
    float t2 = ln1mx2 / a;
    float sign = (x > 0.0f) ? 1.0f : -1.0f;
    float result = sign * sycl::sqrt(sycl::sqrt(t1 * t1 - t2) - t1);

    // Two Newton iterations
    for (int i = 0; i < 2; ++i) {
        float err = sycl::erf(result) - x;
        float deriv = 2.0f / sycl::sqrt(3.14159265358979f) * sycl::exp(-result * result);
        result -= err / deriv;
    }
    return result;
}
inline double erfinv_dev_f64(double x) {
    if (x <= -1.0) return -INFINITY;
    if (x >= 1.0) return INFINITY;
    if (x == 0.0) return 0.0;

    double a = 0.147;
    double ln1mx2 = sycl::log(1.0 - x * x);
    double t1 = 2.0 / (3.14159265358979323846 * a) + 0.5 * ln1mx2;
    double t2 = ln1mx2 / a;
    double sign = (x > 0.0) ? 1.0 : -1.0;
    double result = sign * sycl::sqrt(sycl::sqrt(t1 * t1 - t2) - t1);

    for (int i = 0; i < 2; ++i) {
        double err = sycl::erf(result) - x;
        double deriv = 2.0 / sycl::sqrt(3.14159265358979323846) * sycl::exp(-result * result);
        result -= err / deriv;
    }
    return result;
}

// Sinc
inline float sinc_dev_f32(float x) {
    if (x == 0.0f) return 1.0f;
    float px = 3.14159265358979f * x;
    return sycl::sin(px) / px;
}
inline double sinc_dev_f64(double x) {
    if (x == 0.0) return 1.0;
    double px = 3.14159265358979323846 * x;
    return sycl::sin(px) / px;
}

// Hurwitz zeta
inline float zeta_dev_f32(float s, float a) {
    float result = 0.0f;
    for (int n = 0; n < 12; ++n) {
        result += sycl::pow(a + static_cast<float>(n), -s);
    }
    float aN = a + 12.0f;
    if (s != 1.0f) result += sycl::pow(aN, 1.0f - s) / (s - 1.0f);
    result += 0.5f * sycl::pow(aN, -s);
    return result;
}
inline double zeta_dev_f64(double s, double a) {
    double result = 0.0;
    for (int n = 0; n < 12; ++n) {
        result += sycl::pow(a + static_cast<double>(n), -s);
    }
    double aN = a + 12.0;
    if (s != 1.0) result += sycl::pow(aN, 1.0 - s) / (s - 1.0);
    result += 0.5 * sycl::pow(aN, -s);
    return result;
}

// Polygamma ψ^(n)(x)
inline double polygamma_dev_f64(int n, double x) {
    if (n == 0) return digamma_dev_f64(x);
    double fact_n = 1.0;
    for (int k = 1; k <= n; ++k) fact_n *= static_cast<double>(k);
    double sign = ((n + 1) % 2 == 0) ? 1.0 : -1.0;
    double sum = 0.0;
    for (int k = 0; k < 100; ++k) {
        double term = sycl::pow(x + k, -static_cast<double>(n + 1));
        sum += term;
        if (term < 1e-15 * sycl::fabs(sum)) break;
    }
    return sign * fact_n * sum;
}
inline float polygamma_dev_f32(int n, float x) {
    return static_cast<float>(polygamma_dev_f64(n, static_cast<double>(x)));
}

// Regularized incomplete beta — Lentz continued fraction
inline double betainc_dev_f64(double a, double b, double x) {
    if (x < 0.0 || x > 1.0) return sycl::nan(0u);
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;

    bool flipped = false;
    if (x > (a + 1.0) / (a + b + 2.0)) {
        double tmp_a = a; a = b; b = tmp_a;
        x = 1.0 - x;
        flipped = true;
    }

    double lbeta = sycl::lgamma(a) + sycl::lgamma(b) - sycl::lgamma(a + b);
    double front = sycl::exp(sycl::log(x) * a + sycl::log(1.0 - x) * b - lbeta) / a;

    double f = 1.0, c = 1.0, d = 1.0 - (a + b) * x / (a + 1.0);
    if (sycl::fabs(d) < 1e-30) d = 1e-30;
    d = 1.0 / d;
    f = d;

    for (int m = 1; m <= 200; ++m) {
        double num = static_cast<double>(m) * (b - m) * x
                   / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 + num * d; if (sycl::fabs(d) < 1e-30) d = 1e-30; d = 1.0 / d;
        c = 1.0 + num / c; if (sycl::fabs(c) < 1e-30) c = 1e-30;
        f *= d * c;

        num = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 + num * d; if (sycl::fabs(d) < 1e-30) d = 1e-30; d = 1.0 / d;
        c = 1.0 + num / c; if (sycl::fabs(c) < 1e-30) c = 1e-30;
        double delta = d * c;
        f *= delta;
        if (sycl::fabs(delta - 1.0) < 1e-12) break;
    }
    double val = front * f;
    return flipped ? (1.0 - val) : val;
}

// =========================================================================
// SYCL kernel name tags (one per op × dtype to keep DPC++ kernel naming happy)
// =========================================================================
#define SPECIAL_TAGS(NAME) \
    struct NAME##KernelF32 {}; \
    struct NAME##KernelF64 {}; \
    struct NAME##KernelF16 {}; \
    struct NAME##KernelBF16 {};

SPECIAL_TAGS(Gamma)
SPECIAL_TAGS(Lgamma)
SPECIAL_TAGS(Digamma)
SPECIAL_TAGS(Polygamma)
SPECIAL_TAGS(Beta)
SPECIAL_TAGS(BetaInc)
SPECIAL_TAGS(BesselJ0)
SPECIAL_TAGS(BesselJ1)
SPECIAL_TAGS(BesselY0)
SPECIAL_TAGS(BesselY1)
SPECIAL_TAGS(BesselI0)
SPECIAL_TAGS(BesselI1)
SPECIAL_TAGS(ErfInv)
SPECIAL_TAGS(Sinc)
SPECIAL_TAGS(Zeta)
SPECIAL_TAGS(LogAddExp)
SPECIAL_TAGS(LogAddExp2)
SPECIAL_TAGS(XLogY)
SPECIAL_TAGS(I0e)
SPECIAL_TAGS(I1e)
SPECIAL_TAGS(Entr)
SPECIAL_TAGS(SphericalBesselJ0)
SPECIAL_TAGS(Ndtr)
SPECIAL_TAGS(LogNdtr)
SPECIAL_TAGS(Multigammaln)
SPECIAL_TAGS(CosineSimilarity)
SPECIAL_TAGS(Renorm)
#undef SPECIAL_TAGS

// =========================================================================
// Generic unary launcher
//
// When OneMKL is available, the bessel/gamma/erfinv functions dispatch to
// oneapi::mkl::vm:: versions which are accuracy-validated by Intel.
// Falls back to the polynomial device functions otherwise.
// =========================================================================
#ifdef TENZOR_HAS_ONEMKL
// Helper to launch the float16/bfloat16 path via the polynomial fallbacks
// (OneMKL does support FP16 but for consistency with the FP32 cast path
// already used elsewhere, we promote half/bfloat16 to float and call the VM
// function for the float32 case via a small compute kernel — actually OneMKL
// directly supports half so we can just call it; but for BF16 promotion is needed).
#define ONEMKL_DISPATCH_UNARY(MKL_FN, TAG_PREFIX, NAME, F32_FALLBACK, F64_FALLBACK) \
    auto NAME##_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {        \
        Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()), \
                      input.dtype(), input.device());                              \
        const int64_t numel = input.numel();                                       \
        if (numel == 0) return output;                                             \
        if (input.dtype() == DType::Float32) {                                     \
            ::oneapi::mkl::vm::MKL_FN(queue, numel,                                  \
                get_data_ptr<const float>(input),                                  \
                get_data_ptr<float>(output));                                      \
        } else if (input.dtype() == DType::Float64) {                              \
            ::oneapi::mkl::vm::MKL_FN(queue, numel,                                  \
                get_data_ptr<const double>(input),                                 \
                get_data_ptr<double>(output));                                     \
        } else if (input.dtype() == DType::Float16) {                              \
            ::oneapi::mkl::vm::MKL_FN(queue, numel,                                  \
                get_data_ptr<const sycl::half>(input),                             \
                get_data_ptr<sycl::half>(output));                                 \
        } else if (input.dtype() == DType::BFloat16) {                             \
            /* OneMKL VM has no native BF16 path; promote to float32 via scratch */ \
            float* a_f32 = sycl::malloc_device<float>(numel, queue);               \
            float* y_f32 = sycl::malloc_device<float>(numel, queue);               \
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);          \
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);                    \
            queue.parallel_for<TAG_PREFIX##KernelBF16>(sycl::range<1>(numel),      \
                [=](sycl::id<1> idx) {                                             \
                    a_f32[idx] = bf16_to_f32_h(in_ptr[idx]);                       \
                }).wait();                                                         \
            ::oneapi::mkl::vm::MKL_FN(queue, numel, a_f32, y_f32).wait();            \
            queue.parallel_for<TAG_PREFIX##KernelF16>(sycl::range<1>(numel),       \
                [=](sycl::id<1> idx) {                                             \
                    out_ptr[idx] = f32_to_bf16_h(y_f32[idx]);                      \
                }).wait();                                                         \
            sycl::free(a_f32, queue);                                              \
            sycl::free(y_f32, queue);                                              \
        } else {                                                                   \
            throw std::runtime_error(#NAME ": unsupported dtype");                 \
        }                                                                          \
        return output;                                                             \
    }

ONEMKL_DISPATCH_UNARY(tgamma, Gamma,    gamma,     sycl::tgamma, sycl::tgamma)
ONEMKL_DISPATCH_UNARY(lgamma, Lgamma,   lgamma,    sycl::lgamma, sycl::lgamma)
ONEMKL_DISPATCH_UNARY(j0,     BesselJ0, bessel_j0, bessel_j0_dev_f32, bessel_j0_dev_f64)
ONEMKL_DISPATCH_UNARY(j1,     BesselJ1, bessel_j1, bessel_j1_dev_f32, bessel_j1_dev_f64)
ONEMKL_DISPATCH_UNARY(y0,     BesselY0, bessel_y0, bessel_y0_dev_f32, bessel_y0_dev_f64)
ONEMKL_DISPATCH_UNARY(y1,     BesselY1, bessel_y1, bessel_y1_dev_f32, bessel_y1_dev_f64)
ONEMKL_DISPATCH_UNARY(i0,     BesselI0, bessel_i0, bessel_i0_dev_f32, bessel_i0_dev_f64)
ONEMKL_DISPATCH_UNARY(i1,     BesselI1, bessel_i1, bessel_i1_dev_f32, bessel_i1_dev_f64)
ONEMKL_DISPATCH_UNARY(erfinv, ErfInv,   erfinv,    erfinv_dev_f32, erfinv_dev_f64)
#undef ONEMKL_DISPATCH_UNARY
#endif // TENZOR_HAS_ONEMKL

// Polynomial-fallback launcher (used for digamma/sinc which OneMKL VM lacks,
// and for the entire op set when OneMKL is unavailable).
#define DEFINE_ONEAPI_SPECIAL_UNARY(NAME, TAG_PREFIX, F32_FN, F64_FN)                          \
    auto NAME##_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {                    \
        Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),        \
                      input.dtype(), input.device());                                          \
        const int64_t numel = input.numel();                                                   \
        if (numel == 0) return output;                                                         \
        if (input.dtype() == DType::Float32) {                                                 \
            const float* in_ptr = get_data_ptr<const float>(input);                            \
            float* out_ptr = get_data_ptr<float>(output);                                      \
            queue.parallel_for<TAG_PREFIX##KernelF32>(sycl::range<1>(numel), [=](sycl::id<1> idx) { \
                out_ptr[idx] = (F32_FN)(in_ptr[idx]);                                          \
            });                                                                                \
        } else if (input.dtype() == DType::Float64) {                                          \
            const double* in_ptr = get_data_ptr<const double>(input);                          \
            double* out_ptr = get_data_ptr<double>(output);                                    \
            queue.parallel_for<TAG_PREFIX##KernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) { \
                out_ptr[idx] = (F64_FN)(in_ptr[idx]);                                          \
            });                                                                                \
        } else if (input.dtype() == DType::Float16) {                                          \
            const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);                  \
            sycl::half* out_ptr = get_data_ptr<sycl::half>(output);                            \
            queue.parallel_for<TAG_PREFIX##KernelF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) { \
                out_ptr[idx] = sycl::half((F32_FN)(static_cast<float>(in_ptr[idx])));          \
            });                                                                                \
        } else if (input.dtype() == DType::BFloat16) {                                         \
            const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);                      \
            uint16_t* out_ptr = get_data_ptr<uint16_t>(output);                                \
            queue.parallel_for<TAG_PREFIX##KernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) { \
                out_ptr[idx] = f32_to_bf16_h((F32_FN)(bf16_to_f32_h(in_ptr[idx])));            \
            });                                                                                \
        } else {                                                                               \
            throw std::runtime_error(#NAME ": only Float32/64/16 + BF16 supported");           \
        }                                                                                      \
        return output;                                                                         \
    }

DEFINE_ONEAPI_SPECIAL_UNARY(digamma,   Digamma,   digamma_dev_f32, digamma_dev_f64)
DEFINE_ONEAPI_SPECIAL_UNARY(sinc,      Sinc,      sinc_dev_f32, sinc_dev_f64)

#ifndef TENZOR_HAS_ONEMKL
DEFINE_ONEAPI_SPECIAL_UNARY(gamma,     Gamma,     sycl::tgamma, sycl::tgamma)
DEFINE_ONEAPI_SPECIAL_UNARY(lgamma,    Lgamma,    sycl::lgamma, sycl::lgamma)
DEFINE_ONEAPI_SPECIAL_UNARY(bessel_j0, BesselJ0,  bessel_j0_dev_f32, bessel_j0_dev_f64)
DEFINE_ONEAPI_SPECIAL_UNARY(bessel_j1, BesselJ1,  bessel_j1_dev_f32, bessel_j1_dev_f64)
DEFINE_ONEAPI_SPECIAL_UNARY(bessel_y0, BesselY0,  bessel_y0_dev_f32, bessel_y0_dev_f64)
DEFINE_ONEAPI_SPECIAL_UNARY(bessel_y1, BesselY1,  bessel_y1_dev_f32, bessel_y1_dev_f64)
DEFINE_ONEAPI_SPECIAL_UNARY(bessel_i0, BesselI0,  bessel_i0_dev_f32, bessel_i0_dev_f64)
DEFINE_ONEAPI_SPECIAL_UNARY(bessel_i1, BesselI1,  bessel_i1_dev_f32, bessel_i1_dev_f64)
DEFINE_ONEAPI_SPECIAL_UNARY(erfinv,    ErfInv,    erfinv_dev_f32, erfinv_dev_f64)
#endif

// =========================================================================
// I0e: exp(-|x|) * BesselI0(x) -- exponentially scaled modified Bessel
// =========================================================================
inline float i0e_dev_f32(float x) {
    float ax = sycl::fabs(x);
    if (ax < 3.75f) {
        float t = x / 3.75f;
        t = t * t;
        float i0 = 1.0f + t * (3.5156229f + t * (3.0899424f + t * (1.2067492f
                   + t * (0.2659732f + t * (0.0360768f + t * 0.0045813f)))));
        return sycl::exp(-ax) * i0;
    }
    float t = 3.75f / ax;
    return (1.0f / sycl::sqrt(ax)) * (0.39894228f + t * (0.01328592f
           + t * (0.00225319f - t * (0.00157565f - t * (0.00916281f
           - t * (0.02057706f - t * (0.02635537f - t * (0.01647633f
           - t * 0.00392377f))))))));
}
inline double i0e_dev_f64(double x) {
    double ax = sycl::fabs(x);
    if (ax < 3.75) {
        double t = x / 3.75;
        t = t * t;
        double i0 = 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492
                    + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
        return sycl::exp(-ax) * i0;
    }
    double t = 3.75 / ax;
    return (1.0 / sycl::sqrt(ax)) * (0.39894228 + t * (0.01328592
           + t * (0.00225319 - t * (0.00157565 - t * (0.00916281
           - t * (0.02057706 - t * (0.02635537 - t * (0.01647633
           - t * 0.00392377))))))));
}
DEFINE_ONEAPI_SPECIAL_UNARY(i0e, I0e, i0e_dev_f32, i0e_dev_f64)

// =========================================================================
// I1e: exp(-|x|) * BesselI1(x) -- exponentially scaled modified Bessel
// =========================================================================
inline float i1e_dev_f32(float x) {
    float ax = sycl::fabs(x);
    float result;
    if (ax < 3.75f) {
        float t = x / 3.75f;
        t = t * t;
        result = ax * (0.5f + t * (0.87890594f + t * (0.51498869f + t * (0.15084934f
                 + t * (0.02658733f + t * (0.00301532f + t * 0.00032411f))))));
        result = sycl::exp(-ax) * result;
    } else {
        float t = 3.75f / ax;
        result = (1.0f / sycl::sqrt(ax)) * (0.39894228f - t * (0.03988024f
                 - t * (0.00362018f + t * (0.00163801f - t * (0.01031555f
                 - t * (0.02282967f - t * (0.02895312f - t * (0.01787654f
                 - t * 0.00420059f))))))));
    }
    return (x < 0.0f) ? -result : result;
}
inline double i1e_dev_f64(double x) {
    double ax = sycl::fabs(x);
    double result;
    if (ax < 3.75) {
        double t = x / 3.75;
        t = t * t;
        result = ax * (0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934
                 + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))));
        result = sycl::exp(-ax) * result;
    } else {
        double t = 3.75 / ax;
        result = (1.0 / sycl::sqrt(ax)) * (0.39894228 - t * (0.03988024
                 - t * (0.00362018 + t * (0.00163801 - t * (0.01031555
                 - t * (0.02282967 - t * (0.02895312 - t * (0.01787654
                 - t * 0.00420059))))))));
    }
    return (x < 0.0) ? -result : result;
}
DEFINE_ONEAPI_SPECIAL_UNARY(i1e, I1e, i1e_dev_f32, i1e_dev_f64)

// =========================================================================
// Entr: -x*log(x), 0->0, negative->-inf
// =========================================================================
inline float entr_dev_f32(float x) {
    if (x > 0.0f) return -x * sycl::log(x);
    if (x == 0.0f) return 0.0f;
    return -std::numeric_limits<float>::infinity();
}
inline double entr_dev_f64(double x) {
    if (x > 0.0) return -x * sycl::log(x);
    if (x == 0.0) return 0.0;
    return -std::numeric_limits<double>::infinity();
}
DEFINE_ONEAPI_SPECIAL_UNARY(entr, Entr, entr_dev_f32, entr_dev_f64)

// =========================================================================
// SphericalBesselJ0: sin(x)/x, j0(0)=1
// =========================================================================
inline float spherical_bessel_j0_dev_f32(float x) {
    if (x == 0.0f) return 1.0f;
    return sycl::sin(x) / x;
}
inline double spherical_bessel_j0_dev_f64(double x) {
    if (x == 0.0) return 1.0;
    return sycl::sin(x) / x;
}
DEFINE_ONEAPI_SPECIAL_UNARY(spherical_bessel_j0, SphericalBesselJ0, spherical_bessel_j0_dev_f32, spherical_bessel_j0_dev_f64)

// --- Ndtr: Normal CDF Phi(x) = 0.5 * erfc(-x * M_SQRT1_2) ---
inline float ndtr_dev_f32(float x) {
    return 0.5f * sycl::erfc(-x * 0.7071067811865476f);
}
inline double ndtr_dev_f64(double x) {
    return 0.5 * sycl::erfc(-x * 0.7071067811865476);
}
DEFINE_ONEAPI_SPECIAL_UNARY(ndtr, Ndtr, ndtr_dev_f32, ndtr_dev_f64)

// --- LogNdtr: log(Phi(x)) with stable tail for x < -5 ---
inline float log_ndtr_dev_f32(float x) {
    if (x >= -5.0f) {
        return sycl::log(0.5f * sycl::erfc(-x * 0.7071067811865476f));
    }
    float x2 = x * x;
    float inv_x2 = 1.0f / x2;
    float series = 1.0f - inv_x2 * (1.0f - inv_x2 * (3.0f - inv_x2 * 15.0f));
    return -0.5f * x2 - sycl::log(-x) - 0.9189385332046727f + sycl::log(series);
}
inline double log_ndtr_dev_f64(double x) {
    if (x >= -5.0) {
        return sycl::log(0.5 * sycl::erfc(-x * 0.7071067811865476));
    }
    double x2 = x * x;
    double inv_x2 = 1.0 / x2;
    double series = 1.0 - inv_x2 * (1.0 - inv_x2 * (3.0 - inv_x2 * 15.0));
    return -0.5 * x2 - sycl::log(-x) - 0.9189385332046727 + sycl::log(series);
}
DEFINE_ONEAPI_SPECIAL_UNARY(log_ndtr, LogNdtr, log_ndtr_dev_f32, log_ndtr_dev_f64)

#undef DEFINE_ONEAPI_SPECIAL_UNARY

// =========================================================================
// Multigammaln: sum_{j=0}^{d-1} lgamma(x - j/2) + d*(d-1)/4 * log(pi)
// =========================================================================

// SYCL kernel name tags
class MultigammalnKernelF32;
class MultigammalnKernelF64;
class MultigammalnKernelF16;
class MultigammalnKernelBF16;

auto multigammaln_kernel(const Tensor& input, int64_t d, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    const int64_t numel = input.numel();
    if (numel == 0) return output;
    float log_pi_coeff = static_cast<float>(d) * static_cast<float>(d - 1) / 4.0f
                       * 1.1447298858494002f; // log(pi)
    int d_val = static_cast<int>(d);

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<MultigammalnKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float val = log_pi_coeff;
            for (int j = 0; j < d_val; ++j)
                val += sycl::lgamma(in_ptr[idx] - static_cast<float>(j) * 0.5f);
            out_ptr[idx] = val;
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        double log_pi_coeff_d = static_cast<double>(d) * static_cast<double>(d - 1) / 4.0
                              * 1.1447298858494002;
        queue.parallel_for<MultigammalnKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double val = log_pi_coeff_d;
            for (int j = 0; j < d_val; ++j)
                val += sycl::lgamma(in_ptr[idx] - static_cast<double>(j) * 0.5);
            out_ptr[idx] = val;
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<MultigammalnKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(in_ptr[idx]);
            float val = log_pi_coeff;
            for (int j = 0; j < d_val; ++j)
                val += sycl::lgamma(x - static_cast<float>(j) * 0.5f);
            out_ptr[idx] = sycl::half(val);
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<MultigammalnKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32_h(in_ptr[idx]);
            float val = log_pi_coeff;
            for (int j = 0; j < d_val; ++j)
                val += sycl::lgamma(x - static_cast<float>(j) * 0.5f);
            out_ptr[idx] = f32_to_bf16_h(val);
        });
    } else {
        throw std::runtime_error("multigammaln: only Float32/64/16 + BF16 supported");
    }
    return output;
}

// =========================================================================
// Beta(a, b) = exp(lgamma(a) + lgamma(b) - lgamma(a + b))
// =========================================================================
inline float beta_dev_f32(float a, float b) {
    return sycl::exp(sycl::lgamma(a) + sycl::lgamma(b) - sycl::lgamma(a + b));
}
inline double beta_dev_f64(double a, double b) {
    return sycl::exp(sycl::lgamma(a) + sycl::lgamma(b) - sycl::lgamma(a + b));
}

auto beta_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());
    const int64_t numel = a.numel();
    if (numel == 0) return output;

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<BetaKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = beta_dev_f32(a_ptr[idx], b_ptr[idx]);
        });
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<BetaKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = beta_dev_f64(a_ptr[idx], b_ptr[idx]);
        });
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<BetaKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(beta_dev_f32(static_cast<float>(a_ptr[idx]),
                                                    static_cast<float>(b_ptr[idx])));
        });
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<BetaKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16_h(beta_dev_f32(bf16_to_f32_h(a_ptr[idx]),
                                                       bf16_to_f32_h(b_ptr[idx])));
        });
    } else {
        throw std::runtime_error("beta: unsupported dtype");
    }
    return output;
}

// =========================================================================
// Hurwitz zeta ζ(s, q)
// =========================================================================
auto zeta_kernel(const Tensor& s, const Tensor& q, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(s.shape().begin(), s.shape().end()),
                  s.dtype(), s.device());
    const int64_t numel = s.numel();
    if (numel == 0) return output;

    if (s.dtype() == DType::Float32) {
        const float* s_ptr = get_data_ptr<const float>(s);
        const float* q_ptr = get_data_ptr<const float>(q);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<ZetaKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = zeta_dev_f32(s_ptr[idx], q_ptr[idx]);
        });
    } else if (s.dtype() == DType::Float64) {
        const double* s_ptr = get_data_ptr<const double>(s);
        const double* q_ptr = get_data_ptr<const double>(q);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<ZetaKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = zeta_dev_f64(s_ptr[idx], q_ptr[idx]);
        });
    } else if (s.dtype() == DType::Float16) {
        const sycl::half* s_ptr = get_data_ptr<const sycl::half>(s);
        const sycl::half* q_ptr = get_data_ptr<const sycl::half>(q);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<ZetaKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(zeta_dev_f32(static_cast<float>(s_ptr[idx]),
                                                    static_cast<float>(q_ptr[idx])));
        });
    } else if (s.dtype() == DType::BFloat16) {
        const uint16_t* s_ptr = get_data_ptr<const uint16_t>(s);
        const uint16_t* q_ptr = get_data_ptr<const uint16_t>(q);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<ZetaKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16_h(zeta_dev_f32(bf16_to_f32_h(s_ptr[idx]),
                                                       bf16_to_f32_h(q_ptr[idx])));
        });
    } else {
        throw std::runtime_error("zeta: unsupported dtype");
    }
    return output;
}

// =========================================================================
// Polygamma ψ^(n)(x)
// =========================================================================
auto polygamma_kernel(int64_t n, const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    const int64_t numel = input.numel();
    if (numel == 0) return output;
    int n_int = static_cast<int>(n);

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<PolygammaKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = polygamma_dev_f32(n_int, in_ptr[idx]);
        });
    } else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<PolygammaKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = polygamma_dev_f64(n_int, in_ptr[idx]);
        });
    } else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<PolygammaKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(polygamma_dev_f32(n_int, static_cast<float>(in_ptr[idx])));
        });
    } else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<PolygammaKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16_h(polygamma_dev_f32(n_int, bf16_to_f32_h(in_ptr[idx])));
        });
    } else {
        throw std::runtime_error("polygamma: unsupported dtype");
    }
    return output;
}

// =========================================================================
// Regularized incomplete beta I_x(a, b)
// =========================================================================
auto betainc_kernel(const Tensor& a, const Tensor& b, const Tensor& x, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());
    const int64_t numel = a.numel();
    if (numel == 0) return output;

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        const float* x_ptr = get_data_ptr<const float>(x);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<BetaIncKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = static_cast<float>(betainc_dev_f64(
                static_cast<double>(a_ptr[idx]),
                static_cast<double>(b_ptr[idx]),
                static_cast<double>(x_ptr[idx])));
        });
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        const double* x_ptr = get_data_ptr<const double>(x);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<BetaIncKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = betainc_dev_f64(a_ptr[idx], b_ptr[idx], x_ptr[idx]);
        });
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        const sycl::half* x_ptr = get_data_ptr<const sycl::half>(x);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<BetaIncKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double r = betainc_dev_f64(
                static_cast<double>(static_cast<float>(a_ptr[idx])),
                static_cast<double>(static_cast<float>(b_ptr[idx])),
                static_cast<double>(static_cast<float>(x_ptr[idx])));
            out_ptr[idx] = sycl::half(static_cast<float>(r));
        });
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        const uint16_t* x_ptr = get_data_ptr<const uint16_t>(x);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<BetaIncKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double r = betainc_dev_f64(
                static_cast<double>(bf16_to_f32_h(a_ptr[idx])),
                static_cast<double>(bf16_to_f32_h(b_ptr[idx])),
                static_cast<double>(bf16_to_f32_h(x_ptr[idx])));
            out_ptr[idx] = f32_to_bf16_h(static_cast<float>(r));
        });
    } else {
        throw std::runtime_error("betainc: unsupported dtype");
    }
    return output;
}

// =========================================================================
// LogAddExp: max(a,b) + log1p(exp(-|a-b|))
// =========================================================================
auto logaddexp_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());
    const int64_t numel = a.numel();
    if (numel == 0) return output;

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<LogAddExpKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float va = a_ptr[idx], vb = b_ptr[idx];
            float m = sycl::fmax(va, vb);
            out_ptr[idx] = m + sycl::log1p(sycl::exp(-sycl::fabs(va - vb)));
        });
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<LogAddExpKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double va = a_ptr[idx], vb = b_ptr[idx];
            double m = sycl::fmax(va, vb);
            out_ptr[idx] = m + sycl::log1p(sycl::exp(-sycl::fabs(va - vb)));
        });
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<LogAddExpKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float va = static_cast<float>(a_ptr[idx]), vb = static_cast<float>(b_ptr[idx]);
            float m = sycl::fmax(va, vb);
            out_ptr[idx] = sycl::half(m + sycl::log1p(sycl::exp(-sycl::fabs(va - vb))));
        });
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<LogAddExpKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float va = bf16_to_f32_h(a_ptr[idx]), vb = bf16_to_f32_h(b_ptr[idx]);
            float m = sycl::fmax(va, vb);
            out_ptr[idx] = f32_to_bf16_h(m + sycl::log1p(sycl::exp(-sycl::fabs(va - vb))));
        });
    } else {
        throw std::runtime_error("logaddexp: unsupported dtype");
    }
    return output;
}

// =========================================================================
// LogAddExp2: max(a,b) + log2(1 + exp2(-|a-b|))
// =========================================================================
auto logaddexp2_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());
    const int64_t numel = a.numel();
    if (numel == 0) return output;

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<LogAddExp2KernelF32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float va = a_ptr[idx], vb = b_ptr[idx];
            float m = sycl::fmax(va, vb);
            out_ptr[idx] = m + sycl::log2(1.0f + sycl::exp2(-sycl::fabs(va - vb)));
        });
    } else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<LogAddExp2KernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double va = a_ptr[idx], vb = b_ptr[idx];
            double m = sycl::fmax(va, vb);
            out_ptr[idx] = m + sycl::log2(1.0 + sycl::exp2(-sycl::fabs(va - vb)));
        });
    } else if (a.dtype() == DType::Float16) {
        const sycl::half* a_ptr = get_data_ptr<const sycl::half>(a);
        const sycl::half* b_ptr = get_data_ptr<const sycl::half>(b);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<LogAddExp2KernelF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float va = static_cast<float>(a_ptr[idx]), vb = static_cast<float>(b_ptr[idx]);
            float m = sycl::fmax(va, vb);
            out_ptr[idx] = sycl::half(m + sycl::log2(1.0f + sycl::exp2(-sycl::fabs(va - vb))));
        });
    } else if (a.dtype() == DType::BFloat16) {
        const uint16_t* a_ptr = get_data_ptr<const uint16_t>(a);
        const uint16_t* b_ptr = get_data_ptr<const uint16_t>(b);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<LogAddExp2KernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float va = bf16_to_f32_h(a_ptr[idx]), vb = bf16_to_f32_h(b_ptr[idx]);
            float m = sycl::fmax(va, vb);
            out_ptr[idx] = f32_to_bf16_h(m + sycl::log2(1.0f + sycl::exp2(-sycl::fabs(va - vb))));
        });
    } else {
        throw std::runtime_error("logaddexp2: unsupported dtype");
    }
    return output;
}

// =========================================================================
// XLogY: x*log(y), with x==0 -> 0
// =========================================================================
auto xlogy_kernel(const Tensor& x, const Tensor& y, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(x.shape().begin(), x.shape().end()),
                  x.dtype(), x.device());
    const int64_t numel = x.numel();
    if (numel == 0) return output;

    if (x.dtype() == DType::Float32) {
        const float* x_ptr = get_data_ptr<const float>(x);
        const float* y_ptr = get_data_ptr<const float>(y);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<XLogYKernelF32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float xv = x_ptr[idx];
            out_ptr[idx] = (xv == 0.0f) ? 0.0f : xv * sycl::log(y_ptr[idx]);
        });
    } else if (x.dtype() == DType::Float64) {
        const double* x_ptr = get_data_ptr<const double>(x);
        const double* y_ptr = get_data_ptr<const double>(y);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<XLogYKernelF64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double xv = x_ptr[idx];
            out_ptr[idx] = (xv == 0.0) ? 0.0 : xv * sycl::log(y_ptr[idx]);
        });
    } else if (x.dtype() == DType::Float16) {
        const sycl::half* x_ptr = get_data_ptr<const sycl::half>(x);
        const sycl::half* y_ptr = get_data_ptr<const sycl::half>(y);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<XLogYKernelF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float xv = static_cast<float>(x_ptr[idx]);
            out_ptr[idx] = sycl::half((xv == 0.0f) ? 0.0f : xv * sycl::log(static_cast<float>(y_ptr[idx])));
        });
    } else if (x.dtype() == DType::BFloat16) {
        const uint16_t* x_ptr = get_data_ptr<const uint16_t>(x);
        const uint16_t* y_ptr = get_data_ptr<const uint16_t>(y);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<XLogYKernelBF16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float xv = bf16_to_f32_h(x_ptr[idx]);
            out_ptr[idx] = f32_to_bf16_h((xv == 0.0f) ? 0.0f : xv * sycl::log(bf16_to_f32_h(y_ptr[idx])));
        });
    } else {
        throw std::runtime_error("xlogy: unsupported dtype");
    }
    return output;
}

// =========================================================================
// CosineSimilarity: sum(a*b, dim) / (norm(a, dim) * norm(b, dim) + eps)
// Composed from existing OneAPI primitives via the dispatch layer.
// =========================================================================
auto cosine_similarity_kernel(const Tensor& a, const Tensor& b,
                               int64_t dim, double eps, sycl::queue& queue) -> Tensor {
    // Use the same high-level composition as CPU: mul -> sum -> norm -> div
    auto ab = tenzor::mul(a, b);
    auto dot = tenzor::sum(ab, dim, false);
    auto norm_a = tenzor::norm(a, 2.0, dim, false);
    auto norm_b = tenzor::norm(b, 2.0, dim, false);
    auto norms = tenzor::mul(norm_a, norm_b);
    // Add eps to denominator
    auto eps_tensor = tenzor::full_like(norms, eps);
    auto denom = tenzor::add(norms, eps_tensor);
    return tenzor::div(dot, denom);
}

// =========================================================================
// Renorm: scale slices along dim so p-norm <= maxnorm
// Composed from existing OneAPI primitives via the dispatch layer.
// =========================================================================
auto renorm_kernel(const Tensor& input, double p, int64_t dim, double maxnorm,
                    sycl::queue& queue) -> Tensor {
    // Compute p-norm along dim (keepdim=true for broadcasting)
    auto norm_val = tenzor::norm(input, p, dim, true);
    // clamp: max(norm, maxnorm)
    auto maxnorm_tensor = tenzor::full_like(norm_val, maxnorm);
    auto clamped = tenzor::maximum(norm_val, maxnorm_tensor);
    // scale = maxnorm / max(norm, maxnorm)
    auto scale = tenzor::div(maxnorm_tensor, clamped);
    return tenzor::mul(input, scale);
}

} // namespace oneapi
} // namespace tenzor
