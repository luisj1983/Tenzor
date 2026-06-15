/**
 * @file rocm_arch_detect.hpp
 * @brief AMD GPU architecture detection and capability queries
 *
 * Detects CDNA/RDNA architecture generation from hipDeviceProp_t::gcnArchName
 * and exposes wavefront size and MFMA capability information for kernel tuning.
 */

#pragma once

#include <hip/hip_runtime.h>
#include <string>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <mutex>

namespace tenzor {
namespace rocm {

enum class AMDArch {
    Unknown,
    GCN,      // Older GCN (gfx8xx)
    CDNA,     // MI100 (gfx908)
    CDNA2,    // MI210/MI250 (gfx90a)
    CDNA3,    // MI300 (gfx942)
    RDNA2,    // RX 6000 series (gfx1030)
    RDNA3,    // RX 7000 series (gfx1100/gfx1150)
    RDNA4     // RX 9000 series (gfx12xx)
};

namespace detail {

inline auto parse_arch_from_name(const char* gcn_arch_name) -> AMDArch {
    // gcnArchName is typically something like "gfx908:sramecc+:xnack-"
    // We only need the prefix up to the first colon or end of string.
    std::string name(gcn_arch_name);
    auto colon_pos = name.find(':');
    if (colon_pos != std::string::npos) {
        name = name.substr(0, colon_pos);
    }

    // CDNA3: MI300 series
    if (name == "gfx942" || name == "gfx940" || name == "gfx941") {
        return AMDArch::CDNA3;
    }

    // CDNA2: MI210/MI250 series
    if (name == "gfx90a") {
        return AMDArch::CDNA2;
    }

    // CDNA: MI100
    if (name == "gfx908") {
        return AMDArch::CDNA;
    }

    // RDNA4: RX 9000 series (gfx12xx)
    if (name.size() >= 5 && name.substr(0, 5) == "gfx12") {
        return AMDArch::RDNA4;
    }

    // RDNA3: RX 7000 series (gfx1100, gfx1101, gfx1102, gfx1150, gfx1151)
    if (name == "gfx1100" || name == "gfx1101" || name == "gfx1102" ||
        name == "gfx1150" || name == "gfx1151") {
        return AMDArch::RDNA3;
    }

    // RDNA2: RX 6000 series (gfx1030, gfx1031, gfx1032, gfx1033, gfx1034, gfx1035, gfx1036)
    if (name.size() >= 7 && name.substr(0, 6) == "gfx103") {
        return AMDArch::RDNA2;
    }

    // Older GCN architectures (gfx8xx, gfx900, gfx902, gfx906)
    if (name.size() >= 4 && name.substr(0, 4) == "gfx8") {
        return AMDArch::GCN;
    }
    if (name == "gfx900" || name == "gfx902" || name == "gfx906") {
        return AMDArch::GCN;
    }

    return AMDArch::Unknown;
}

/// Cached device properties to avoid repeated hipGetDeviceProperties calls.
/// Safe for concurrent reads after initialization (per-device lazy init).
struct CachedDeviceInfo {
    AMDArch arch = AMDArch::Unknown;
    int wavefront_size = 64;
    std::string gcn_arch_name;  ///< e.g. "gfx1150" (colon suffix stripped)
    bool initialized = false;
};

inline auto get_cached_info(int device_id) -> const CachedDeviceInfo& {
    // Support up to 16 devices; static array avoids dynamic allocation.
    static constexpr int kMaxDevices = 16;
    static CachedDeviceInfo cache[kMaxDevices];
    // One once_flag per device guards the lazy init so concurrent first calls
    // for the same device cannot race on the std::string / bool writes.
    static std::once_flag init_flags[kMaxDevices];

    if (device_id < 0 || device_id >= kMaxDevices) {
        throw std::runtime_error(
            "get_cached_info: device_id " + std::to_string(device_id) +
            " out of range [0, " + std::to_string(kMaxDevices) + ")");
    }

    auto& info = cache[device_id];
    std::call_once(init_flags[device_id], [device_id, &info]() {
        hipDeviceProp_t props;
        hipError_t err = hipGetDeviceProperties(&props, device_id);
        if (err != hipSuccess) {
            throw std::runtime_error(
                std::string("hipGetDeviceProperties failed for device ") +
                std::to_string(device_id) + ": " + hipGetErrorString(err));
        }
        info.arch = parse_arch_from_name(props.gcnArchName);
        info.wavefront_size = props.warpSize;

        // Strip ":sramecc+:xnack-" feature suffix; keep only the gfx name.
        std::string name(props.gcnArchName);
        auto colon_pos = name.find(':');
        info.gcn_arch_name = (colon_pos != std::string::npos)
            ? name.substr(0, colon_pos)
            : name;

        info.initialized = true;
    });
    return info;
}

} // namespace detail

/// Detect AMD GPU architecture for the given device
inline auto detect_arch(int device_id = 0) -> AMDArch {
    return detail::get_cached_info(device_id).arch;
}

/// Get the warp/wavefront size for the given device (64 for CDNA, 32 for RDNA3+)
inline auto get_wavefront_size(int device_id = 0) -> int {
    return detail::get_cached_info(device_id).wavefront_size;
}

/// Check whether MIOpen is usable on this device.
///
/// Upstream MIOpen ships a prebuilt kernel database (the .ufdb / .ktn / .tn
/// files under /opt/rocm/share/miopen/db/) for CDNA parts only — gfx908
/// (MI100), gfx90a (MI200), gfx942 (MI300), gfx950. Every RDNA part falls
/// back to HIPRTC JIT at first use.
///
/// HIPRTC needs ROCM_PATH (or the HIP headers in its default search path) to
/// resolve <hip/hip_runtime.h>; without it, JIT fails with "file not found"
/// and on some devices the failed compile wedges the GPU. The ROCm backend's
/// constructor (rocm_backend.cpp::ensure_rocm_path_for_hiprtc) sets ROCM_PATH
/// before any MIOpen call when /opt/rocm/include/hip/hip_runtime.h exists,
/// which is the canonical install layout. With ROCM_PATH in place, JIT works
/// and the GPU does not hang.
///
/// We therefore enable MIOpen on every architecture we recognise. The
/// TENZOR_ROCM_DISABLE_MIOPEN=1 escape hatch forces the native HIP fallback
/// for environments where MIOpen still misbehaves (custom ROCm install, no
/// HIP headers on disk, etc.).
inline auto is_miopen_available(int device_id = 0) -> bool {
    if (const char* env = std::getenv("TENZOR_ROCM_DISABLE_MIOPEN");
        env && env[0] == '1') {
        return false;
    }
    if (const char* env = std::getenv("TENZOR_ROCM_FORCE_MIOPEN");
        env && env[0] == '1') {
        return true;
    }
    AMDArch arch = detect_arch(device_id);
    return arch != AMDArch::Unknown;
}

} // namespace rocm
} // namespace tenzor
