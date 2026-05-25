// HH.25: shared CLI helpers for the C++ benchmark binaries.
//
// Pre-HH.25 every ``benchmark_*.cpp`` hard-coded ``Device::cpu()`` even when
// the binary clearly intended to exercise GPU kernels (``benchmark_attention``
// etc.). The runner in ``benchmarks/python/run_benchmarks.py`` invokes these
// binaries through argv, so the fix is a tiny argv parser that every main()
// can call: ``--device {cpu|cuda|rocm|vulkan|oneapi}`` plus an optional
// ``--device-id N``.
//
// Kept header-only so it can be included from binaries without adding a new
// CMake source target.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>

#include <tenzor/core/device.hpp>

namespace tenzor::bench {

/// Parse ``--device <name>`` and ``--device-id <N>`` from argv. Unknown args
/// are ignored (each benchmark may have its own additional flags). Returns
/// ``Device::cpu()`` when no flag is present so existing CPU-only callers
/// stay correct.
inline Device parse_device_arg(int argc, char** argv) {
    std::string device_name = "cpu";
    int device_id = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device_name = argv[++i];
        } else if (std::strcmp(argv[i], "--device-id") == 0 && i + 1 < argc) {
            device_id = std::atoi(argv[++i]);
        }
    }
    if (device_name == "cpu") return Device::cpu();
    if (device_name == "cuda") return Device(Device::Type::CUDA, device_id);
    if (device_name == "rocm") return Device(Device::Type::ROCm, device_id);
    if (device_name == "vulkan") return Device(Device::Type::Vulkan, device_id);
    if (device_name == "oneapi") return Device(Device::Type::OneAPI, device_id);
    if (device_name == "mps") return Device(Device::Type::MPS, device_id);
    // Unknown -> fall back to CPU so a typo doesn't crash the binary on a
    // pristine machine. The CLI prints the resolved device so this is
    // observable.
    return Device::cpu();
}

} // namespace tenzor::bench
