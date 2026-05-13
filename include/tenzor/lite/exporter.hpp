/**
 * @file exporter.hpp
 * @brief Export an `nn::Module` to a `.tzlite` file for the Lite runtime.
 *
 * The exporter walks the module's submodule tree, recognises supported layer
 * types via `dynamic_cast`, and emits a `LiteGraph` plus a SafeTensors-style
 * weight blob through `TZLiteWriter::save`. The resulting file loads via
 * `LiteRuntime::load` and produces numerically identical output to
 * `module.forward(x).detach()` for the same inputs.
 *
 * Phase 3 supported layers:
 *   - tenzor::nn::Linear   (with or without bias)
 *   - tenzor::nn::Sequential
 *   - tenzor::nn::ReLU, Sigmoid, Tanh, GELU
 *
 * Unsupported layer types throw `std::runtime_error` with the offending
 * class name in the message. Phase 5+ extends coverage.
 */

#pragma once

#include "../core/device.hpp"
#include "../core/dtype.hpp"
#include "model_format.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace tenzor {
namespace nn { class Module; }
namespace lite {

struct ExportOptions {
    /** Shape of the single input tensor the exported model accepts.
     *  Phase 3 supports exactly one input; Phase 5+ will lift this. */
    std::vector<int64_t> input_shape;

    /** Dtype of the single input tensor. */
    DType input_dtype{DType::Float32};

    /** Device the exported model targets. Today, only matters for the
     *  embedded metadata; runtime can override via load(path, device). */
    Device device{Device::cpu()};

    /** Optional free-form metadata stored in the META section. */
    std::unordered_map<std::string, std::string> metadata;
};

/** Export a module to `.tzlite`. The module's parameter / buffer set is
 *  captured at call time — subsequent training of `module` does not affect
 *  the saved file.
 *
 *  Throws `std::runtime_error` if `module` contains a layer type the Phase 3
 *  exporter doesn't support.
 */
auto export_to_tzlite(nn::Module& module,
                      const std::string& path,
                      const ExportOptions& opts) -> void;

}  // namespace lite
}  // namespace tenzor
