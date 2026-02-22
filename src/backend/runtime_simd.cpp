/**
 * @file runtime_simd.cpp
 * @brief Runtime SIMD capability detection implementation
 *
 * Uses CPUID (x86/x64) or hwcaps (ARM) for feature detection, with proper
 * XSAVE/XCR0 verification to confirm the OS saves extended register state
 * before reporting AVX/AVX-512 support.
 *
 * Detection runs once on first call and is cached via static local variables
 * (thread-safe by C++11 guarantee).
 */

#include "tenzor/backend/runtime_simd.hpp"
#include "tenzor/backend/simd_kernel_table.hpp"
#include <iostream>
#include <sstream>

// ============================================================================
// Platform-specific CPUID / feature detection includes
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define TENZOR_X86 1
    #if defined(_MSC_VER)
        #include <intrin.h>  // __cpuid, __cpuidex, _xgetbv
    #elif defined(__GNUC__) || defined(__clang__)
        #include <cpuid.h>
        // xgetbv requires inline asm on GCC/Clang (immintrin.h _xgetbv
        // requires -mxsave which may not be enabled at baseline)
        #include <cstdint>
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define TENZOR_AARCH64 1
    #if defined(__linux__)
        #include <sys/auxv.h>
        #include <asm/hwcap.h>
    #endif
#elif defined(__ARM_NEON) || defined(_M_ARM)
    #define TENZOR_ARM32 1
    #if defined(__linux__)
        #include <sys/auxv.h>
        #include <asm/hwcap.h>
    #endif
#endif

namespace tenzor {
namespace backend {

// ============================================================================
// x86/x64 CPUID helpers
// ============================================================================

#ifdef TENZOR_X86

namespace {

struct CpuidResult {
    uint32_t eax;
    uint32_t ebx;
    uint32_t ecx;
    uint32_t edx;
};

auto cpuid(uint32_t leaf, uint32_t subleaf = 0) -> CpuidResult {
    CpuidResult r{};
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    r.eax = static_cast<uint32_t>(regs[0]);
    r.ebx = static_cast<uint32_t>(regs[1]);
    r.ecx = static_cast<uint32_t>(regs[2]);
    r.edx = static_cast<uint32_t>(regs[3]);
#elif defined(__GNUC__) || defined(__clang__)
    __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
    return r;
}

/**
 * @brief Read the XCR0 (Extended Control Register 0) via XGETBV.
 *
 * XCR0 indicates which extended CPU state components the OS has enabled
 * for saving/restoring on context switches. We must check this before
 * using AVX (YMM registers) or AVX-512 (ZMM, opmask registers).
 *
 * Bit 0: x87 FP state
 * Bit 1: SSE state (XMM registers)
 * Bit 2: AVX state (upper 128 bits of YMM registers)
 * Bit 5: AVX-512 opmask registers (k0-k7)
 * Bit 6: AVX-512 upper 256 bits of ZMM0-ZMM15
 * Bit 7: AVX-512 ZMM16-ZMM31
 */
auto xgetbv(uint32_t xcr_index) -> uint64_t {
#if defined(_MSC_VER)
    return _xgetbv(xcr_index);
#elif defined(__GNUC__) || defined(__clang__)
    uint32_t eax, edx;
    __asm__ __volatile__(
        "xgetbv"
        : "=a"(eax), "=d"(edx)
        : "c"(xcr_index)
    );
    return (static_cast<uint64_t>(edx) << 32) | eax;
#else
    (void)xcr_index;
    return 0;
#endif
}

} // anonymous namespace

#endif // TENZOR_X86

// ============================================================================
// SIMDFeatures detection
// ============================================================================

namespace {

auto detect_features_impl() -> SIMDFeatures {
    SIMDFeatures f{};

#ifdef TENZOR_X86
    // -- Leaf 0: max basic leaf and vendor string --
    auto leaf0 = cpuid(0, 0);
    uint32_t max_leaf = leaf0.eax;

    // -- Leaf 1: basic feature flags --
    if (max_leaf >= 1) {
        auto leaf1 = cpuid(1, 0);

        // EDX features
        f.sse2 = (leaf1.edx & (1u << 26)) != 0;

        // ECX features
        f.sse3  = (leaf1.ecx & (1u << 0)) != 0;
        f.ssse3 = (leaf1.ecx & (1u << 9)) != 0;
        f.sse41 = (leaf1.ecx & (1u << 19)) != 0;
        f.sse42 = (leaf1.ecx & (1u << 20)) != 0;
        f.fma   = (leaf1.ecx & (1u << 12)) != 0;
        f.f16c  = (leaf1.ecx & (1u << 29)) != 0;

        // AVX requires both the CPU flag AND OS XSAVE support
        bool cpu_avx = (leaf1.ecx & (1u << 28)) != 0;
        bool osxsave = (leaf1.ecx & (1u << 27)) != 0;

        if (cpu_avx && osxsave) {
            // Verify OS saves YMM state via XCR0
            uint64_t xcr0 = xgetbv(0);
            // Bits 1 (SSE/XMM) and 2 (AVX/YMM) must both be set
            f.os_avx = ((xcr0 & 0x6) == 0x6);
            f.avx = f.os_avx;

            // Check for AVX-512 OS support: bits 5, 6, 7 (opmask, ZMM upper, ZMM16-31)
            f.os_avx512 = ((xcr0 & 0xE0) == 0xE0) && f.os_avx;
        }
    }

    // -- Leaf 7, subleaf 0: extended feature flags --
    if (max_leaf >= 7) {
        auto leaf7 = cpuid(7, 0);

        // AVX2: EBX bit 5 (also requires OS AVX support for YMM)
        if (f.os_avx) {
            f.avx2 = (leaf7.ebx & (1u << 5)) != 0;
        }

        // AVX-512 features: all require OS AVX-512 support
        if (f.os_avx512) {
            f.avx512f    = (leaf7.ebx & (1u << 16)) != 0;
            f.avx512dq   = (leaf7.ebx & (1u << 17)) != 0;
            f.avx512bw   = (leaf7.ebx & (1u << 30)) != 0;
            f.avx512vl   = (leaf7.ebx & (1u << 31)) != 0;
            f.avx512vnni = (leaf7.ecx & (1u << 11)) != 0;
        }

        // AVX-512 BF16: leaf 7, subleaf 1, EAX bit 5
        // Check max subleaf count first
        if (f.os_avx512 && f.avx512f) {
            uint32_t max_subleaf = leaf7.eax;
            if (max_subleaf >= 1) {
                auto leaf7_1 = cpuid(7, 1);
                f.avx512bf16 = (leaf7_1.eax & (1u << 5)) != 0;
            }
        }
    }

#endif // TENZOR_X86

#ifdef TENZOR_AARCH64
    // NEON is mandatory in ARMv8-A (AArch64)
    f.neon = true;

    #if defined(__linux__)
    // SVE detection via hwcaps
    unsigned long hwcap = getauxval(AT_HWCAP);
    // HWCAP_SVE is bit 22 on AArch64 Linux
    #ifndef HWCAP_SVE
        #define HWCAP_SVE (1 << 22)
    #endif
    f.sve = (hwcap & HWCAP_SVE) != 0;
    #endif
#endif

#ifdef TENZOR_ARM32
    #if defined(__ARM_NEON)
        f.neon = true;
    #elif defined(__linux__)
        unsigned long hwcap = getauxval(AT_HWCAP);
        f.neon = (hwcap & HWCAP_NEON) != 0;
    #endif
#endif

    return f;
}

/// Log the detection results once on first call
auto log_detection(const SIMDFeatures& f, SIMDLevel level) -> void {
    std::cout << "[tenzor] Runtime SIMD detection: " << simd_level_name(level)
              << " (" << f.to_string() << ")" << std::endl;
    if (!f.os_avx && (f.sse42 || f.sse2)) {
        // This would be unusual on modern systems
        std::cout << "[tenzor] Note: OS does not support AVX register state saving"
                  << std::endl;
    }
}

} // anonymous namespace

// ============================================================================
// SIMDFeatures methods
// ============================================================================

auto SIMDFeatures::best_level() const -> SIMDLevel {
    // ARM levels
    if (sve)  return SIMDLevel::SVE;
    if (neon) return SIMDLevel::NEON;

    // x86 levels: return highest supported, ordered descending
    if (avx512bf16 && avx512f) return SIMDLevel::AVX512BF16;
    if (avx512vnni && avx512f) return SIMDLevel::AVX512VNNI;
    if (avx512bw   && avx512f) return SIMDLevel::AVX512BW;
    if (avx512f)               return SIMDLevel::AVX512F;
    if (avx2)                  return SIMDLevel::AVX2;
    if (avx)                   return SIMDLevel::AVX;
    if (sse42)                 return SIMDLevel::SSE42;
    if (sse2)                  return SIMDLevel::SSE2;

    return SIMDLevel::None;
}

auto SIMDFeatures::to_string() const -> std::string {
    std::ostringstream oss;
    bool first = true;

    auto append = [&](const char* name) {
        if (!first) oss << ' ';
        oss << name;
        first = false;
    };

    // Print highest to lowest for readability
    if (avx512bf16) append("AVX512-BF16");
    if (avx512vnni) append("AVX512-VNNI");
    if (avx512bw)   append("AVX512-BW");
    if (avx512vl)   append("AVX512-VL");
    if (avx512dq)   append("AVX512-DQ");
    if (avx512f)    append("AVX512-F");
    if (avx2)       append("AVX2");
    if (fma)        append("FMA");
    if (f16c)       append("F16C");
    if (avx)        append("AVX");
    if (sse42)      append("SSE4.2");
    if (sse41)      append("SSE4.1");
    if (ssse3)      append("SSSE3");
    if (sse3)       append("SSE3");
    if (sse2)       append("SSE2");
    if (sve)        append("SVE");
    if (neon)       append("NEON");

    if (first) {
        oss << "None";
    }

    return oss.str();
}

// ============================================================================
// Public API
// ============================================================================

auto get_simd_features() -> const SIMDFeatures& {
    static const SIMDFeatures features = [] {
        auto f = detect_features_impl();
        log_detection(f, f.best_level());
        return f;
    }();
    return features;
}

auto detect_simd_level() -> SIMDLevel {
    return get_simd_features().best_level();
}

auto simd_level_name(SIMDLevel level) -> const char* {
    switch (level) {
        case SIMDLevel::None:       return "None";
        case SIMDLevel::SSE2:       return "SSE2";
        case SIMDLevel::SSE42:      return "SSE4.2";
        case SIMDLevel::AVX:        return "AVX";
        case SIMDLevel::AVX2:       return "AVX2";
        case SIMDLevel::AVX512F:    return "AVX-512F";
        case SIMDLevel::AVX512BW:   return "AVX-512BW";
        case SIMDLevel::AVX512VNNI: return "AVX-512VNNI";
        case SIMDLevel::AVX512BF16: return "AVX-512BF16";
        case SIMDLevel::NEON:       return "NEON";
        case SIMDLevel::SVE:        return "SVE";
    }
    return "Unknown";
}

auto has_simd_feature(SIMDLevel required) -> bool {
    SIMDLevel detected = detect_simd_level();

    // ARM and x86 are separate families; cannot compare across them
    auto is_x86 = [](SIMDLevel l) {
        auto v = static_cast<uint8_t>(l);
        return v >= static_cast<uint8_t>(SIMDLevel::SSE2)
            && v <= static_cast<uint8_t>(SIMDLevel::AVX512BF16);
    };
    auto is_arm = [](SIMDLevel l) {
        auto v = static_cast<uint8_t>(l);
        return v >= static_cast<uint8_t>(SIMDLevel::NEON)
            && v <= static_cast<uint8_t>(SIMDLevel::SVE);
    };

    if (required == SIMDLevel::None) return true;

    if (is_x86(required) && is_x86(detected)) {
        return static_cast<uint8_t>(detected) >= static_cast<uint8_t>(required);
    }
    if (is_arm(required) && is_arm(detected)) {
        return static_cast<uint8_t>(detected) >= static_cast<uint8_t>(required);
    }

    // Cross-family: required ARM on x86 or vice versa
    return false;
}

// ============================================================================
// Global SIMD kernel table singleton
// ============================================================================

auto get_simd_kernel_table() -> SIMDKernelTable& {
    static SIMDKernelTable table;
    return table;
}

} // namespace backend
} // namespace tenzor
