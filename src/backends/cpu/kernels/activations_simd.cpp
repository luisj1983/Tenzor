/**
 * @file activations_simd.cpp
 * @brief SIMD implementations of activation functions (currently unused)
 *
 * All relu/sigmoid/tanh/gelu SIMD implementations that used to live here
 * (scalar::, avx2::, and the cpu::simd:: runtime-dispatch wrapper) were
 * removed as dead code (F-029): activations.cpp dispatches through oneDNN
 * instead, and neither SimdTrait nor the g_dispatch function-pointer table
 * ever read a relu/sigmoid/tanh/gelu entry. See simd_dispatch.cpp/.hpp for
 * the live g_dispatch fields (add/sub/mul/div/sqrt + f64 variants) and
 * math_simd.cpp/math_simd_avx512.cpp for their surviving implementations.
 *
 * This TU is kept (empty) because it is still listed as a per-file
 * -mavx2/-mavx512f compile unit in src/backends/cpu/CMakeLists.txt.
 */

#include "tenzor/backends/cpu/simd.hpp"

namespace tenzor {
namespace cpu {
} // namespace cpu
} // namespace tenzor
