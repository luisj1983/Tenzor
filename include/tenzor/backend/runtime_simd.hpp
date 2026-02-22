/**
 * @file runtime_simd.hpp
 * @brief Runtime SIMD capability detection via CPUID
 *
 * Provides portable runtime detection of CPU SIMD instruction set extensions.
 * Uses CPUID on x86/x64 with proper XSAVE/XCR0 verification to confirm OS
 * support for extended register state. Results are cached after first detection.
 *
 * This is the foundation for portable binary distribution: pre-built binaries
 * can ship kernels compiled at multiple ISA levels and select the best one
 * at runtime based on the actual CPU capabilities.
 *
 * @see SIMDKernelTable for multi-ISA kernel registration and resolution
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace tenzor {
namespace backend {

/**
 * @brief SIMD instruction set level, ordered from lowest to highest capability.
 *
 * Each level implies support for all lower levels (e.g., AVX2 implies SSE2,
 * SSE4.2, and AVX). The ordering is used by SIMDKernelTable to select the
 * best available kernel: the highest level that does not exceed the detected
 * CPU capability.
 */
enum class SIMDLevel : uint8_t {
    None       = 0,   ///< No SIMD (scalar fallback)
    SSE2       = 1,   ///< SSE2 (128-bit integer/FP, baseline x86-64)
    SSE42      = 2,   ///< SSE4.2 (string/CRC instructions, implies SSE2-4.1)
    AVX        = 3,   ///< AVX (256-bit FP, VEX encoding)
    AVX2       = 4,   ///< AVX2 (256-bit integer, FMA3)
    AVX512F    = 5,   ///< AVX-512 Foundation (512-bit vectors)
    AVX512BW   = 6,   ///< AVX-512 Byte/Word (fine-grained 512-bit integer)
    AVX512VNNI = 7,   ///< AVX-512 VNNI (int8/int16 dot product)
    AVX512BF16 = 8,   ///< AVX-512 BF16 (bfloat16 conversion/dot product)
    NEON       = 20,  ///< ARM NEON (128-bit SIMD, mandatory in AArch64)
    SVE        = 21,  ///< ARM SVE (scalable vector, stub for future use)
};

/**
 * @brief Detect the highest SIMD level supported by the current CPU.
 *
 * On x86/x64, uses CPUID with XSAVE/XCR0 validation to confirm the OS
 * has enabled the necessary register state saving (YMM for AVX, ZMM for
 * AVX-512). This prevents illegal instruction faults on CPUs where the
 * hardware supports an ISA but the OS kernel does not save the state.
 *
 * On ARM AArch64, returns SIMDLevel::NEON (mandatory in ARMv8-A).
 * On unknown architectures, returns SIMDLevel::None.
 *
 * The result is computed once and cached in a static local variable.
 * Thread-safe via C++11 static local initialization guarantees.
 *
 * @return The highest available SIMDLevel
 */
auto detect_simd_level() -> SIMDLevel;

/**
 * @brief Get a human-readable name for a SIMD level.
 *
 * @param level The SIMD level
 * @return String representation (e.g., "AVX2", "AVX-512F", "None")
 */
auto simd_level_name(SIMDLevel level) -> const char*;

/**
 * @brief Check if a required SIMD feature level is available on this CPU.
 *
 * For x86 levels (SSE2 through AVX512BF16), this is an ordered comparison:
 * the detected level must be >= the required level. For ARM levels, exact
 * match or higher within the ARM family is checked.
 *
 * @param required The minimum SIMD level needed
 * @return true if the CPU supports the required level
 */
auto has_simd_feature(SIMDLevel required) -> bool;

/**
 * @brief Detailed SIMD feature flags from CPUID.
 *
 * Unlike SIMDLevel which provides a single "best level" summary, this
 * structure exposes individual feature bits for fine-grained queries
 * (e.g., checking for FMA independently of AVX2, or AVX512BW without
 * requiring AVX512VNNI).
 */
struct SIMDFeatures {
    // x86 features
    bool sse2       = false;
    bool sse3       = false;
    bool ssse3      = false;
    bool sse41      = false;
    bool sse42      = false;
    bool avx        = false;
    bool avx2       = false;
    bool fma        = false;
    bool f16c       = false;
    bool avx512f    = false;
    bool avx512dq   = false;
    bool avx512bw   = false;
    bool avx512vl   = false;
    bool avx512vnni = false;
    bool avx512bf16 = false;

    // ARM features
    bool neon       = false;
    bool sve        = false;

    // OS support for extended register state
    bool os_avx     = false;  ///< OS saves YMM registers (XCR0 bits 1:2)
    bool os_avx512  = false;  ///< OS saves ZMM registers (XCR0 bits 5:7)

    /**
     * @brief Get the highest SIMDLevel based on detected features.
     *
     * Takes into account both hardware support (CPUID) and OS support
     * (XSAVE/XCR0) to determine the highest safely usable level.
     */
    auto best_level() const -> SIMDLevel;

    /**
     * @brief Get a summary string of all detected features.
     *
     * Example: "AVX512BF16 AVX512VNNI AVX512BW AVX512F AVX2 FMA AVX SSE4.2 SSE2"
     */
    auto to_string() const -> std::string;
};

/**
 * @brief Get the full set of detected SIMD features.
 *
 * More detailed than detect_simd_level() -- exposes individual feature flags.
 * Cached after first call, thread-safe.
 *
 * @return Reference to the cached feature detection result
 */
auto get_simd_features() -> const SIMDFeatures&;

} // namespace backend
} // namespace tenzor
