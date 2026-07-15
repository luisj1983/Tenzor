/**
 * @file activations_simd_avx512.cpp
 * @brief AVX-512 implementations of activation functions (currently unused)
 *
 * All relu/sigmoid/tanh/gelu SIMD implementations that used to live here
 * were removed as dead code (F-029) — see activations_simd.cpp for the
 * full explanation. This TU is kept (empty) because it is still listed as
 * a per-file -mavx512f compile unit in src/backends/cpu/CMakeLists.txt.
 */

#include "tenzor/backends/cpu/simd.hpp"

namespace tenzor {
namespace cpu {
} // namespace cpu
} // namespace tenzor
