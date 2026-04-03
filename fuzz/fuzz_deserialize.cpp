/**
 * @file fuzz_deserialize.cpp
 * @brief libFuzzer target for Tenzor native format deserialization
 *
 * Feeds random bytes to Serializer::load() to find crashes, OOM,
 * or undefined behavior in the deserialization path.
 */

#include "tenzor/nn/serialize.hpp"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <filesystem>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Write fuzzer input to a temp file in /dev/shm for speed
    static int counter = 0;
    std::string path = "/dev/shm/tenzor_fuzz_deser_" +
                        std::to_string(getpid()) + "_" +
                        std::to_string(counter++) + ".bin";

    {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open()) return 0;
        f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }

    try {
        auto result = tenzor::nn::Serializer::load(path);
        // Exercise the loaded tensors to catch deferred issues
        for (const auto& [name, tensor] : result) {
            (void)tensor.numel();
            (void)tensor.dtype();
            (void)tensor.device();
        }
    } catch (...) {
        // Expected for most fuzzed inputs
    }

    std::filesystem::remove(path);
    return 0;
}
