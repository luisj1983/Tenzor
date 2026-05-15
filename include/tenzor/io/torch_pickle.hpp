/**
 * @file torch_pickle.hpp
 * @brief Native PyTorch checkpoint (.pth / .pt / .bin) loader.
 *
 * Audit H2-followup: parses the torch.save ZIP archive layout used by
 * PyTorch >= 1.6 and the embedded Python pickle stream that encodes the
 * state_dict, then reconstructs each tensor's raw bytes (via the
 * `_rebuild_tensor_v2` reduce protocol) into a Tenzor `Tensor`.
 *
 * Scope of this minimal parser:
 *   - State-dict-shaped checkpoints (nested dicts of tensors), the only
 *     thing the audit cares about. General arbitrary-Python-object pickles
 *     are intentionally out of scope.
 *   - Pickle protocol 2 through 5 (everything PyTorch's `_use_new_zipfile_serialization`
 *     path emits).
 *   - Standard `torch.FloatStorage` / `torch.LongStorage` / etc. via
 *     `_rebuild_tensor_v2(storage, storage_offset, size, stride,
 *      requires_grad, backward_hooks)`.
 *   - Float32 / Float64 / Float16 / BFloat16 / Int32 / Int64 / Int8 /
 *     UInt8 / Bool storages.
 *
 * Out of scope:
 *   - Legacy non-zipfile PyTorch checkpoints (< 1.6). Those have a
 *     completely different binary layout.
 *   - Compressed (DEFLATEd) archive entries — torch.save uses STORED
 *     (compression method 0) for tensor data; the pickle entry may be
 *     either STORED or DEFLATEd. We support STORED for both.
 *   - Tensors with non-default stride that doesn't match contiguous
 *     row-major (we throw a clear error in that case).
 *
 * Usage:
 *
 *     auto state = tenzor::io::load_torch_pickle("resnet50.pth");
 *     // state is std::unordered_map<std::string, Tensor> keyed by param name.
 */

#pragma once

#include "tenzor/core/tensor.hpp"

#include <string>
#include <unordered_map>

namespace tenzor::io {

/**
 * @brief Load a PyTorch .pth/.pt checkpoint into a Tenzor state-dict map.
 *
 * @param path Filesystem path to the checkpoint file. May be either the
 *             newer `torch.save(state_dict, f)` ZIP archive layout (most
 *             modern checkpoints) or, on a best-effort basis, the legacy
 *             non-zip pickle layout.
 *
 * @return `std::unordered_map<std::string, Tensor>` keyed by parameter
 *         name. Tensors land on CPU; callers can `.to(device)` if needed.
 *
 * @throws std::runtime_error if the file is not a recognised PyTorch
 *         checkpoint, uses an unsupported pickle opcode, references a
 *         tensor dtype that Tenzor doesn't represent, or contains
 *         compressed (DEFLATE) entries.
 */
auto load_torch_pickle(const std::string& path)
    -> std::unordered_map<std::string, Tensor>;

}  // namespace tenzor::io
