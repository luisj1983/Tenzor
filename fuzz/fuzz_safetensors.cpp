/**
 * @file fuzz_safetensors.cpp
 * @brief libFuzzer target for SafeTensors format deserialization
 *
 * Feeds random bytes to SafeTensorsSerializer::load() to find crashes
 * or undefined behavior in JSON parsing and data loading.
 */

#include "tenzor/nn/safetensors.hpp"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <filesystem>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    static int counter = 0;
    std::string path = "/dev/shm/tenzor_fuzz_st_" +
                        std::to_string(getpid()) + "_" +
                        std::to_string(counter++) + ".safetensors";

    {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open()) return 0;
        f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }

    try {
        auto result = tenzor::nn::SafeTensorsSerializer::load(path);
        for (const auto& [name, tensor] : result) {
            (void)tensor.numel();
            (void)tensor.dtype();
        }
    } catch (...) {
        // Expected for most fuzzed inputs
    }

    std::filesystem::remove(path);
    return 0;
}
