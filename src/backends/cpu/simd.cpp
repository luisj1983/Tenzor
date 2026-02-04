/**
 * @file simd.cpp
 * @brief CPU feature detection using CPUID
 */

#include "tenzor/backends/cpu/simd.hpp"
#include <sstream>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define TENZOR_X86
    #if defined(_MSC_VER)
        #include <intrin.h>
    #elif defined(__GNUC__) || defined(__clang__)
        #include <cpuid.h>
    #endif
#endif

namespace tenzor {
namespace cpu {

#ifdef TENZOR_X86

auto CPUInfo::cpuid(uint32_t leaf, uint32_t subleaf)
    -> std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {

    uint32_t eax, ebx, ecx, edx;

    #if defined(_MSC_VER)
        int cpu_info[4];
        __cpuidex(cpu_info, leaf, subleaf);
        eax = cpu_info[0];
        ebx = cpu_info[1];
        ecx = cpu_info[2];
        edx = cpu_info[3];
    #elif defined(__GNUC__) || defined(__clang__)
        __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
    #else
        eax = ebx = ecx = edx = 0;
    #endif

    return {eax, ebx, ecx, edx};
}

#else

auto CPUInfo::cpuid(uint32_t leaf, uint32_t subleaf)
    -> std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
    return {0, 0, 0, 0};
}

#endif

CPUInfo::CPUInfo() {
    detect_features();
}

auto CPUInfo::get() -> const CPUInfo& {
    static CPUInfo instance;
    return instance;
}

auto CPUInfo::detect_features() -> void {
#ifdef TENZOR_X86
    // Get vendor string
    auto [max_leaf, vendor_ebx, vendor_edx, vendor_ecx] = cpuid(0, 0);

    char vendor_buf[13] = {0};
    std::memcpy(vendor_buf + 0, &vendor_ebx, 4);
    std::memcpy(vendor_buf + 4, &vendor_edx, 4);
    std::memcpy(vendor_buf + 8, &vendor_ecx, 4);
    vendor_ = vendor_buf;

    // Get brand string
    auto [max_ext_leaf, _, __, ___] = cpuid(0x80000000, 0);
    if (max_ext_leaf >= 0x80000004) {
        char brand_buf[49] = {0};
        for (uint32_t i = 0; i < 3; ++i) {
            auto [eax, ebx, ecx, edx] = cpuid(0x80000002 + i, 0);
            std::memcpy(brand_buf + i * 16 + 0, &eax, 4);
            std::memcpy(brand_buf + i * 16 + 4, &ebx, 4);
            std::memcpy(brand_buf + i * 16 + 8, &ecx, 4);
            std::memcpy(brand_buf + i * 16 + 12, &edx, 4);
        }
        brand_ = brand_buf;
    }

    if (max_leaf >= 1) {
        auto [eax1, ebx1, ecx1, edx1] = cpuid(1, 0);

        // EDX features
        if (edx1 & (1 << 25)) features_ = features_ | CPUFeature::SSE;
        if (edx1 & (1 << 26)) features_ = features_ | CPUFeature::SSE2;

        // ECX features
        if (ecx1 & (1 << 0))  features_ = features_ | CPUFeature::SSE3;
        if (ecx1 & (1 << 9))  features_ = features_ | CPUFeature::SSSE3;
        if (ecx1 & (1 << 19)) features_ = features_ | CPUFeature::SSE41;
        if (ecx1 & (1 << 20)) features_ = features_ | CPUFeature::SSE42;
        if (ecx1 & (1 << 28)) features_ = features_ | CPUFeature::AVX;
        if (ecx1 & (1 << 12)) features_ = features_ | CPUFeature::FMA;
    }

    if (max_leaf >= 7) {
        auto [eax7, ebx7, ecx7, edx7] = cpuid(7, 0);

        // EBX features
        if (ebx7 & (1 << 5))  features_ = features_ | CPUFeature::AVX2;
        if (ebx7 & (1 << 16)) features_ = features_ | CPUFeature::AVX512F;
        if (ebx7 & (1 << 17)) features_ = features_ | CPUFeature::AVX512DQ;
        if (ebx7 & (1 << 30)) features_ = features_ | CPUFeature::AVX512BW;
        if (ebx7 & (1 << 31)) features_ = features_ | CPUFeature::AVX512VL;

        // ECX features (leaf 7, subleaf 0)
        if (ecx7 & (1 << 11)) features_ = features_ | CPUFeature::AVX512VNNI;
    }
#endif
}

auto CPUInfo::feature_string() const -> std::string {
    std::ostringstream oss;

    if (has(CPUFeature::AVX512VNNI)) oss << "AVX512-VNNI ";
    if (has(CPUFeature::AVX512F))  oss << "AVX-512 ";
    if (has(CPUFeature::AVX2))     oss << "AVX2 ";
    if (has(CPUFeature::AVX))      oss << "AVX ";
    if (has(CPUFeature::FMA))      oss << "FMA ";
    if (has(CPUFeature::SSE42))    oss << "SSE4.2 ";
    if (has(CPUFeature::SSE41))    oss << "SSE4.1 ";
    if (has(CPUFeature::SSSE3))    oss << "SSSE3 ";
    if (has(CPUFeature::SSE3))     oss << "SSE3 ";
    if (has(CPUFeature::SSE2))     oss << "SSE2 ";
    if (has(CPUFeature::SSE))      oss << "SSE ";

    std::string result = oss.str();
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }

    return result;
}

} // namespace cpu
} // namespace tenzor
