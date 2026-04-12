/**
 * @file fuzz_onnx_importer.cpp
 * @brief libFuzzer target for the ONNX import path.
 *
 * Feeds random bytes to ONNXImporter::import_from_bytes(). The importer
 * hand-rolls its own protobuf wire-format parser in src/onnx/importer.cpp
 * (see read_varint / read_length_delimited at ~lines 38-99), which makes
 * it a high-value fuzzing target — any crash, out-of-bounds read, or
 * infinite loop reachable via untrusted .onnx bytes would be a real
 * security issue for users loading models from HuggingFace / Hub / disk.
 *
 * Build: cmake -B build -G Ninja -DTENZOR_FUZZ=ON -DCMAKE_CXX_COMPILER=clang++
 * Run:   ./build/fuzz/fuzz_onnx_importer -max_total_time=600
 */

#include "tenzor/onnx/importer.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Bound the input size — libFuzzer will sometimes produce very large
    // inputs, and ONNX parsing on a pathological huge blob can dominate
    // wall-clock fuzzing budget without improving coverage.
    if (size > 1 << 20) return 0;

    std::vector<uint8_t> bytes(data, data + size);
    tenzor::onnx::ONNXImporter importer(/*verbose=*/false);
    try {
        // Importer builds a whole module graph; most fuzzed inputs will
        // fail validation and throw. We care about crashes / UAF / OOB —
        // not thrown exceptions.
        auto module = importer.import_from_bytes(bytes);
        if (module) {
            (void)module->extra_repr();
        }
    } catch (const std::exception&) {
        // Expected for almost all fuzz inputs.
    } catch (...) {
        // Non-std exception — swallow so the fuzzer keeps running. Still
        // a stability signal worth investigating if this ever fires, but
        // it should not halt fuzzing.
    }
    return 0;
}
