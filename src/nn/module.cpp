#include "tenzor/nn/module.hpp"
#include "tenzor/nn/serialize.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/autograd/checkpoint.hpp"
#include "tenzor/utils/error.hpp"  // NotImplementedError (S25 / audit-12)
#include <algorithm>
#include <unordered_set>

namespace tenzor::nn {

// Forward declaration — defined in src/nn/utils/parametrize.cpp. Called
// from ~Module() so that the parametrize registry doesn't accumulate stale
// entries for Modules that have been destroyed. Pulling this in via a
// forward decl (rather than including parametrize.hpp) keeps the dependency
// inverted: Module doesn't depend on the utils header.
namespace utils {
void unregister_parametrization_for_module(uint64_t module_id);
}

namespace detail {
// Shared in-place parameter/buffer load used by Module::load_state_dict and the
// ParameterList/ParameterDict overrides. Adapts the checkpoint payload to the
// live tensor's dtype + device, then writes via ``dst.zero_(); add_(dst, src)``
// so the live Storage handle is preserved (R.18 in-place-write invariant):
// aliasing views (FSDP2 shards, saved-for-backward captures, optimizer-held
// shared_ptrs) observe the new bytes instead of dangling at the pre-load
// buffer. Shapes must match exactly; only dtype/device are adapted.
void load_param_in_place(Tensor& dst, const Tensor& src,
                         const std::string& name, const char* kind) {
    if (dst.shape().size() != src.shape().size() ||
        !std::equal(dst.shape().begin(), dst.shape().end(), src.shape().begin())) {
        throw std::runtime_error(std::string("Shape mismatch for ") + kind + " '" + name + "'");
    }
    Tensor adapted = src;
    if (adapted.dtype() != dst.dtype()) {
        adapted = adapted.to(dst.dtype());
    }
    if (adapted.device() != dst.device()) {
        adapted = adapted.to(dst.device());
    }
    dst.zero_();
    add_(dst, adapted);
}
} // namespace detail

namespace {
// Process-monotonic counter. Starts at 1 so that id 0 is reserved for
// "no module" / sentinel use. memory_order_relaxed is sufficient — we only
// require uniqueness, not happens-before across threads observing the id.
auto next_module_id() -> uint64_t {
    static std::atomic<uint64_t> counter{0};
    return counter.fetch_add(1, std::memory_order_relaxed) + 1;
}
}

Module::Module() : id_(next_module_id()) {}

Module::~Module() {
    // Tear down any external registrations keyed by this Module's UID.
    // We *must* do this here rather than relying on the user to call
    // remove_parametrizations() because the registry's lifetime is global
    // and otherwise outlives every Module that ever registered into it.
    //
    // id_ == 0 marks a moved-from shell whose UID now belongs to another live
    // Module; unregistering would tear down the live module's parametrizations.
    if (id_ != 0) {
        utils::unregister_parametrization_for_module(id_);
    }
}

auto Module::parameters() -> std::vector<std::shared_ptr<Variable>> {
    std::vector<std::shared_ptr<Variable>> params;

    // Add own parameters in a consistent order (weight before bias)
    // This ensures tests can rely on params[0] being weight
    if (parameters_.find("weight") != parameters_.end()) {
        params.push_back(parameters_["weight"]);
    }
    if (parameters_.find("bias") != parameters_.end()) {
        params.push_back(parameters_["bias"]);
    }

    // Add any other parameters not named "weight" or "bias"
    for (auto& [name, param] : parameters_) {
        if (name != "weight" && name != "bias") {
            params.push_back(param);
        }
    }

    // Add submodule parameters
    for (auto& [name, module] : submodules_) {
        auto sub_params = module->parameters();
        params.insert(params.end(), sub_params.begin(), sub_params.end());
    }

    return params;
}

auto Module::own_parameters() -> std::vector<std::shared_ptr<Variable>> {
    std::vector<std::shared_ptr<Variable>> params;

    // Add ONLY this module's direct parameters, not submodules'
    // Order: weight, bias, then others
    if (parameters_.find("weight") != parameters_.end()) {
        params.push_back(parameters_["weight"]);
    }
    if (parameters_.find("bias") != parameters_.end()) {
        params.push_back(parameters_["bias"]);
    }

    for (auto& [name, param] : parameters_) {
        if (name != "weight" && name != "bias") {
            params.push_back(param);
        }
    }

    return params;
}

auto Module::named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> {
    std::vector<std::pair<std::string, std::shared_ptr<Variable>>> params;

    for (auto& [name, param] : parameters_) {
        params.emplace_back(name, param);
    }

    // Sort own parameters by name for deterministic ordering
    // (parameters_ is an unordered_map — iteration order is non-deterministic)
    std::sort(params.begin(), params.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    for (auto& [name, module] : submodules_) {
        auto sub_params = module->named_parameters();
        for (auto& [sub_name, sub_param] : sub_params) {
            params.emplace_back(name + "." + sub_name, sub_param);
        }
    }

    return params;
}

auto Module::train(bool mode) -> void {
    training_ = mode;
    for (auto& [_, module] : submodules_) {
        module->train(mode);
    }
}

auto Module::eval() -> void {
    train(false);
}

auto Module::to(Device device) -> void {
    // std::cerr << "[DEBUG] Module::to() called with device type " << static_cast<int>(device.type) << std::endl;

    // Transfer parameters
    // std::cerr << "[DEBUG] Transferring " << parameters_.size() << " parameters..." << std::endl;
    for (auto& [name, param] : parameters_) {
        // std::cerr << "[DEBUG] Transfer parameter '" << name << "' from device "
        //           << static_cast<int>(param->tensor().device().type) << " to " << static_cast<int>(device.type) << std::endl;
        param->tensor() = param->tensor().to(device);
        // std::cerr << "[DEBUG] Parameter '" << name << "' transferred successfully" << std::endl;
    }

    // Transfer buffers (running_mean, running_var, etc.)
    // std::cerr << "[DEBUG] Transferring " << buffers_.size() << " buffers..." << std::endl;
    for (auto& [_, buffer] : buffers_) {
        buffer->tensor() = buffer->tensor().to(device);
    }

    // Recursively transfer submodules
    // std::cerr << "[DEBUG] Transferring " << submodules_.size() << " submodules..." << std::endl;
    for (auto& [_, module] : submodules_) {
        module->to(device);
    }

    // Device transfer moves each parameter/buffer in place and recurses into
    // submodules without reconstructing them, so every (sub)module retains its
    // own training_ flag. We must NOT call train(training_) here: that recurses
    // and overwrites intentional per-submodule eval state (e.g. a frozen child
    // inside a training parent) with the parent's flag.

    // std::cerr << "[DEBUG] Module::to() completed successfully" << std::endl;
}

auto Module::to(DType dtype) -> void {
    // Only convert floating-point tensors (matches PyTorch behavior).
    // Integer buffers (e.g. num_batches_tracked) are preserved.
    auto is_floating = [](DType dt) {
        return dt == DType::Float32 || dt == DType::Float64 ||
               dt == DType::Float16 || dt == DType::BFloat16 ||
               dt == DType::Complex64 || dt == DType::Complex128;
    };

    // Convert floating-point parameters to target dtype
    for (auto& [name, param] : parameters_) {
        if (is_floating(param->tensor().dtype())) {
            param->tensor() = param->tensor().to(dtype);
        }
    }

    // Convert floating-point buffers to target dtype
    for (auto& [_, buffer] : buffers_) {
        if (is_floating(buffer->tensor().dtype())) {
            buffer->tensor() = buffer->tensor().to(dtype);
        }
    }

    // Recursively convert submodules
    for (auto& [_, module] : submodules_) {
        module->to(dtype);
    }

    // As in to(Device): dtype conversion is in-place and does not reconstruct
    // submodules, so per-submodule training_ flags are preserved. Calling
    // train(training_) here would recurse and clobber intentional per-submodule
    // eval state with the parent's flag.
}

auto Module::cuda(int device_id) -> void {
    to(Device::cuda(device_id));
}

auto Module::cpu() -> void {
    to(Device::cpu());
}

auto Module::zero_grad() -> void {
    for (auto& param : parameters()) {
        if (param) {
            param->zero_grad();
        }
    }
}

auto Module::register_parameter(std::string name, Variable param) -> void {
    // Wrap Variable in shared_ptr for stable address (prevents dangling pointers in autograd)
    auto ptr = std::make_shared<Variable>(std::move(param));
    // Enable thread-safe gradient accumulation for multi-threaded training
    ptr->make_thread_safe();
    parameters_[std::move(name)] = ptr;
}

auto Module::register_buffer(std::string name, Variable buffer) -> void {
    // Wrap Variable in shared_ptr for stable address
    buffers_[std::move(name)] = std::make_shared<Variable>(std::move(buffer));
}

auto Module::register_parameter_shared(std::string name,
                                       std::shared_ptr<Variable> param) -> void {
    if (!param) {
        throw std::invalid_argument(
            "Module::register_parameter_shared: param shared_ptr is null (name='" + name + "')");
    }
    // Enable thread-safe gradient accumulation, matching register_parameter().
    param->make_thread_safe();
    parameters_[std::move(name)] = std::move(param);
}

auto Module::register_buffer_shared(std::string name,
                                    std::shared_ptr<Variable> buffer) -> void {
    if (!buffer) {
        throw std::invalid_argument(
            "Module::register_buffer_shared: buffer shared_ptr is null (name='" + name + "')");
    }
    buffers_[std::move(name)] = std::move(buffer);
}

auto Module::register_module(std::string name, std::shared_ptr<Module> module) -> void {
    submodules_[std::move(name)] = std::move(module);
}

auto Module::get_parameter(const std::string& name) const -> std::shared_ptr<Variable> {
    auto it = parameters_.find(name);
    if (it == parameters_.end()) {
        throw std::out_of_range("Parameter not found: " + name);
    }
    return it->second;
}

auto Module::get_buffer(const std::string& name) const -> std::shared_ptr<Variable> {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) {
        throw std::out_of_range("Buffer not found: " + name);
    }
    return it->second;
}

auto Module::unregister_parameter(const std::string& name) -> void {
    auto it = parameters_.find(name);
    if (it == parameters_.end()) {
        throw std::out_of_range("Parameter not found: " + name);
    }
    parameters_.erase(it);
}

auto Module::unregister_buffer(const std::string& name) -> void {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) {
        throw std::out_of_range("Buffer not found: " + name);
    }
    buffers_.erase(it);
}

auto Module::replace_module(const std::string& name, std::shared_ptr<Module> module) -> void {
    submodules_[name] = std::move(module);
}

auto Module::unregister_module(const std::string& name) -> void {
    auto it = submodules_.find(name);
    if (it == submodules_.end()) {
        throw std::out_of_range("Module not found: " + name);
    }
    submodules_.erase(it);
}

auto Module::buffers() -> std::vector<std::shared_ptr<Variable>> {
    std::vector<std::shared_ptr<Variable>> bufs;

    for (auto& [name, buffer] : buffers_) {
        bufs.push_back(buffer);
    }

    for (auto& [name, module] : submodules_) {
        auto sub_bufs = module->buffers();
        bufs.insert(bufs.end(), sub_bufs.begin(), sub_bufs.end());
    }

    return bufs;
}

auto Module::own_buffers() -> std::vector<std::shared_ptr<Variable>> {
    std::vector<std::shared_ptr<Variable>> bufs;

    // Return only this module's direct buffers, not submodules'
    for (auto& [name, buffer] : buffers_) {
        bufs.push_back(buffer);
    }

    return bufs;
}

auto Module::named_buffers() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> {
    std::vector<std::pair<std::string, std::shared_ptr<Variable>>> bufs;

    for (auto& [name, buffer] : buffers_) {
        bufs.emplace_back(name, buffer);
    }

    std::sort(bufs.begin(), bufs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    for (auto& [name, module] : submodules_) {
        auto sub_bufs = module->named_buffers();
        for (auto& [sub_name, sub_buf] : sub_bufs) {
            bufs.emplace_back(name + "." + sub_name, sub_buf);
        }
    }

    return bufs;
}

auto Module::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    // Add own parameters
    for (const auto& [name, param] : parameters_) {
        state[name] = param->tensor().clone();
    }

    // Add own buffers
    for (const auto& [name, buffer] : buffers_) {
        state[name] = buffer->tensor().clone();
    }

    // Add submodule state with prefixed names
    for (const auto& [name, module] : submodules_) {
        auto sub_state = module->state_dict();
        for (auto& [sub_name, tensor] : sub_state) {
            state[name + "." + sub_name] = std::move(tensor);
        }
    }

    return state;
}

auto Module::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    load_state_dict(state, true);
}

// Audit-4 U.10: load_state_dict preserves the live parameter/buffer storage
// and adapts the checkpoint payload to the module's current dtype/device.
//
// Behaviour:
//   * Shapes still must match exactly -- silent reshape would mask user bugs.
//   * If the checkpoint dtype differs from the live parameter dtype, the
//     loaded tensor is cast to the module's dtype (supports the canonical
//     mixed-precision flow: load an F32 checkpoint into an F16 inference
//     model, or load a BF16 checkpoint back into an F32 reference model).
//   * If the checkpoint device differs (e.g. CPU checkpoint into a CUDA
//     module, the standard distributed/restore pattern), the loaded tensor
//     is transferred to the module's current device.
//   * Data is written via ``dst.zero_(); add_(dst, src)`` so the live
//     Tensor's underlying ``Storage`` handle is preserved (the R.18
//     invariant FSDP2/saved-for-backward depend on).  Direct
//     ``dst = src.clone()`` would swap the TensorImpl pointer and orphan
//     any aliasing view, including FSDP2 shard views and any activation
//     that captured the pre-load Tensor in save_for_backward.
auto Module::load_state_dict(const std::unordered_map<std::string, Tensor>& state, bool strict) -> void {
    // Audit-4 W.17: delegate to the out-param variant so the keep/throw
    // path and the introspect path share a single implementation.
    std::vector<std::string> missing_keys;
    std::vector<std::string> unexpected_keys;
    load_state_dict(state, strict, missing_keys, unexpected_keys);
}

auto Module::load_state_dict(const std::unordered_map<std::string, Tensor>& state,
                             bool strict,
                             std::vector<std::string>& missing_keys,
                             std::vector<std::string>& unexpected_keys) -> void {
    missing_keys.clear();
    unexpected_keys.clear();

    // Track which state keys are consumed to detect unexpected keys
    std::unordered_set<std::string> consumed_keys;

    // Local helper: adapt loaded tensor to live tensor's dtype + device and
    // copy in-place so the live Storage handle is preserved.
    // Adapt the checkpoint payload to the live tensor's dtype + device, then
    // write in-place so the live Storage handle is preserved. This realises the
    // advertised mixed-precision flow (an F32 checkpoint can be loaded into an
    // F16 inference model and vice-versa). Shared with the
    // ParameterList/ParameterDict overrides via detail::load_param_in_place.
    auto copy_into_live = [](Tensor& dst, const Tensor& src, const std::string& name,
                             const char* kind) {
        detail::load_param_in_place(dst, src, name, kind);
    };

    // Load own parameters
    for (auto& [name, param] : parameters_) {
        auto it = state.find(name);
        if (it != state.end()) {
            consumed_keys.insert(name);
            copy_into_live(param->tensor(), it->second, name, "parameter");
        }
    }

    // Load own buffers
    for (auto& [name, buffer] : buffers_) {
        auto it = state.find(name);
        if (it != state.end()) {
            consumed_keys.insert(name);
            copy_into_live(buffer->tensor(), it->second, name, "buffer");
        }
    }

    // Load submodule state
    for (auto& [name, module] : submodules_) {
        // Extract submodule state with matching prefix
        std::unordered_map<std::string, Tensor> sub_state;
        std::string prefix = name + ".";
        for (const auto& [key, tensor] : state) {
            if (key.rfind(prefix, 0) == 0) {
                // Key starts with prefix
                consumed_keys.insert(key);
                std::string sub_key = key.substr(prefix.length());
                sub_state[sub_key] = tensor;
            }
        }
        // Y.21: propagate the submodule's missing/unexpected keys into the
        // root's aggregate report, re-prefixed with `name + "."`. Previously
        // the recursion called the throw-on-mismatch overload with
        // strict=false, which silently discarded the submodule's report —
        // so the root's strict-mode throw only ever saw mismatches in its
        // *own* parameters_/buffers_, not anything inside submodules.
        std::vector<std::string> sub_missing;
        std::vector<std::string> sub_unexpected;
        module->load_state_dict(sub_state, /*strict=*/false, sub_missing, sub_unexpected);
        for (auto& k : sub_missing) {
            missing_keys.push_back(prefix + k);
        }
        for (auto& k : sub_unexpected) {
            unexpected_keys.push_back(prefix + k);
        }
    }

    // Collect missing keys (expected but not in state)
    for (const auto& [name, _] : parameters_) {
        if (state.find(name) == state.end()) {
            missing_keys.push_back(name);
        }
    }
    for (const auto& [name, _] : buffers_) {
        if (state.find(name) == state.end()) {
            missing_keys.push_back(name);
        }
    }

    // Collect unexpected keys
    for (const auto& [key, _] : state) {
        if (consumed_keys.find(key) == consumed_keys.end()) {
            unexpected_keys.push_back(key);
        }
    }

    // Report both missing and unexpected keys in strict mode
    if (strict && (!missing_keys.empty() || !unexpected_keys.empty())) {
        std::string msg = "Error(s) in loading state_dict:";
        if (!missing_keys.empty()) {
            msg += "\n  Missing key(s): ";
            for (size_t i = 0; i < missing_keys.size(); ++i) {
                if (i > 0) msg += ", ";
                msg += "\"" + missing_keys[i] + "\"";
            }
        }
        if (!unexpected_keys.empty()) {
            msg += "\n  Unexpected key(s): ";
            for (size_t i = 0; i < unexpected_keys.size(); ++i) {
                if (i > 0) msg += ", ";
                msg += "\"" + unexpected_keys[i] + "\"";
            }
        }
        throw std::runtime_error(msg);
    }
}

auto Module::save(const std::string& path) const -> void {
    auto state = state_dict();
    Serializer::save(state, path);
}

auto Module::load(const std::string& path) -> void {
    auto state = Serializer::load(path);
    load_state_dict(state);
}

// ============================================================================
// ModuleList implementation
// ============================================================================

auto ModuleList::append(std::shared_ptr<Module> module) -> ModuleList& {
    std::string name = std::to_string(modules_.size());
    modules_.push_back(module);
    register_module(name, module);
    return *this;
}

auto ModuleList::at(size_t idx) const -> std::shared_ptr<Module> {
    if (idx >= modules_.size()) {
        throw std::out_of_range("ModuleList index out of range: " +
                                std::to_string(idx) + " >= " + std::to_string(modules_.size()));
    }
    return modules_[idx];
}

auto ModuleList::forward_impl(const Variable& /*input*/) -> Variable {
    throw NotImplementedError("ModuleList does not implement forward(). "
                              "Use it to store modules and iterate over them manually.");
}

auto ModuleList::parameters() -> std::vector<std::shared_ptr<Variable>> {
    std::vector<std::shared_ptr<Variable>> params;
    for (auto& module : modules_) {
        auto sub_params = module->parameters();
        params.insert(params.end(), sub_params.begin(), sub_params.end());
    }
    return params;
}

auto ModuleList::named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> {
    std::vector<std::pair<std::string, std::shared_ptr<Variable>>> params;
    for (size_t i = 0; i < modules_.size(); ++i) {
        std::string prefix = std::to_string(i);
        auto sub_params = modules_[i]->named_parameters();
        for (auto& [name, param] : sub_params) {
            params.emplace_back(prefix + "." + name, param);
        }
    }
    return params;
}

auto ModuleList::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;
    for (size_t i = 0; i < modules_.size(); ++i) {
        std::string prefix = std::to_string(i);
        auto sub_state = modules_[i]->state_dict();
        for (auto& [key, tensor] : sub_state) {
            state[prefix + "." + key] = tensor;
        }
    }
    return state;
}

auto ModuleList::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    for (size_t i = 0; i < modules_.size(); ++i) {
        std::string prefix = std::to_string(i) + ".";
        std::unordered_map<std::string, Tensor> sub_state;
        for (const auto& [key, tensor] : state) {
            if (key.rfind(prefix, 0) == 0) {
                sub_state[key.substr(prefix.length())] = tensor;
            }
        }
        modules_[i]->load_state_dict(sub_state);
    }
}

// ============================================================================
// ModuleDict implementation
// ============================================================================

auto ModuleDict::insert(const std::string& key, std::shared_ptr<Module> module) -> ModuleDict& {
    // If key already exists, replace the module and update the registration
    if (modules_.count(key)) {
        modules_[key] = module;
        // Re-register (Module::register_module handles replacement internally)
        register_module(key, module);
    } else {
        order_.push_back(key);
        modules_[key] = module;
        register_module(key, module);
    }
    return *this;
}

auto ModuleDict::at(const std::string& key) const -> std::shared_ptr<Module> {
    auto it = modules_.find(key);
    if (it == modules_.end()) {
        throw std::out_of_range("ModuleDict key not found: " + key);
    }
    return it->second;
}

auto ModuleDict::contains(const std::string& key) const -> bool {
    return modules_.count(key) > 0;
}

auto ModuleDict::erase(const std::string& key) -> void {
    auto it = modules_.find(key);
    if (it == modules_.end()) {
        throw std::out_of_range("ModuleDict key not found: " + key);
    }
    modules_.erase(it);
    order_.erase(std::remove(order_.begin(), order_.end(), key), order_.end());
    submodules_.erase(key);
}

auto ModuleDict::values() const -> std::vector<std::shared_ptr<Module>> {
    std::vector<std::shared_ptr<Module>> result;
    result.reserve(order_.size());
    for (const auto& key : order_) {
        result.push_back(modules_.at(key));
    }
    return result;
}

auto ModuleDict::items() const -> std::vector<std::pair<std::string, std::shared_ptr<Module>>> {
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> result;
    result.reserve(order_.size());
    for (const auto& key : order_) {
        result.emplace_back(key, modules_.at(key));
    }
    return result;
}

auto ModuleDict::forward_impl(const Variable& /*input*/) -> Variable {
    throw NotImplementedError("ModuleDict does not implement forward(). "
                             "Use it to store modules and access them by key.");
}

auto ModuleDict::parameters() -> std::vector<std::shared_ptr<Variable>> {
    std::vector<std::shared_ptr<Variable>> params;
    for (const auto& key : order_) {
        auto sub_params = modules_.at(key)->parameters();
        params.insert(params.end(), sub_params.begin(), sub_params.end());
    }
    return params;
}

auto ModuleDict::named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> {
    std::vector<std::pair<std::string, std::shared_ptr<Variable>>> params;
    for (const auto& key : order_) {
        auto sub_params = modules_.at(key)->named_parameters();
        for (auto& [name, param] : sub_params) {
            params.emplace_back(key + "." + name, param);
        }
    }
    return params;
}

auto ModuleDict::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;
    for (const auto& key : order_) {
        auto sub_state = modules_.at(key)->state_dict();
        for (auto& [k, tensor] : sub_state) {
            state[key + "." + k] = tensor;
        }
    }
    return state;
}

auto ModuleDict::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    for (const auto& key : order_) {
        std::string prefix = key + ".";
        std::unordered_map<std::string, Tensor> sub_state;
        for (const auto& [k, tensor] : state) {
            if (k.rfind(prefix, 0) == 0) {
                sub_state[k.substr(prefix.length())] = tensor;
            }
        }
        modules_.at(key)->load_state_dict(sub_state);
    }
}

// Sequential implementation
auto Sequential::add_module(std::shared_ptr<Module> module) -> Sequential& {
    // Generate unique name for this module
    std::string name = "module_" + std::to_string(modules_.size());

    // Add to both modules_ vector (for forward pass) and submodules_ map (for state_dict)
    modules_.push_back(module);
    register_module(name, module);

    return *this;
}

auto Sequential::forward_impl(const Variable& input) -> Variable {
    auto output = input;
    for (auto& module : modules_) {
        if (gradient_checkpointing_ && output.requires_grad()) {
            // Recompute this module's activations in backward instead of
            // retaining them. Only when the input tracks gradients (otherwise
            // there is nothing to recompute for).
            Module* m = module.get();
            auto outs = ::tenzor::autograd::checkpoint(
                [m](const std::vector<Variable>& v) -> std::vector<Variable> {
                    return { m->forward(v[0]) };
                },
                std::vector<Variable>{output});
            output = outs[0];
        } else {
            output = module->forward(output);
        }
    }
    return output;
}

auto Sequential::parameters() -> std::vector<std::shared_ptr<Variable>> {
    std::vector<std::shared_ptr<Variable>> params;
    // Iterate over modules_ in order (not submodules_ which is unordered)
    for (auto& module : modules_) {
        auto sub_params = module->parameters();
        params.insert(params.end(), sub_params.begin(), sub_params.end());
    }
    return params;
}

auto Sequential::named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> {
    std::vector<std::pair<std::string, std::shared_ptr<Variable>>> params;
    // Use module index to generate names in order
    for (size_t i = 0; i < modules_.size(); ++i) {
        std::string prefix = "module_" + std::to_string(i);
        auto sub_params = modules_[i]->named_parameters();
        for (auto& [sub_name, sub_param] : sub_params) {
            params.emplace_back(prefix + "." + sub_name, sub_param);
        }
    }
    return params;
}

auto Sequential::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;
    // Use module index to generate names in order
    for (size_t i = 0; i < modules_.size(); ++i) {
        std::string prefix = "module_" + std::to_string(i);
        auto sub_state = modules_[i]->state_dict();
        for (auto& [sub_name, tensor] : sub_state) {
            state[prefix + "." + sub_name] = std::move(tensor);
        }
    }
    return state;
}

auto Sequential::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    // Load state for each module in order
    for (size_t i = 0; i < modules_.size(); ++i) {
        std::string prefix = "module_" + std::to_string(i) + ".";

        // Extract submodule state with matching prefix
        std::unordered_map<std::string, Tensor> sub_state;
        for (const auto& [key, tensor] : state) {
            if (key.rfind(prefix, 0) == 0) {
                // Key starts with prefix
                std::string sub_key = key.substr(prefix.length());
                sub_state[sub_key] = tensor;
            }
        }

        // Load state into this module
        modules_[i]->load_state_dict(sub_state);
    }
}

// ============================================================================
// ParameterList implementation
// ============================================================================

auto ParameterList::append(Variable param) -> ParameterList& {
    std::string name = std::to_string(params_.size());
    auto param_ptr = std::make_shared<Variable>(std::move(param));
    param_ptr->make_thread_safe();
    params_.push_back(param_ptr);
    // Alias the SAME shared_ptr into the base parameters_ map so that
    // Module::to(Device)/to(DType) (which mutate parameters_) and params_
    // refer to one Variable object. register_parameter() would store a copy,
    // leaving params_ stale after a device/dtype move.
    register_parameter_shared(name, param_ptr);
    return *this;
}

auto ParameterList::at(size_t idx) const -> std::shared_ptr<Variable> {
    if (idx >= params_.size()) {
        throw std::out_of_range("ParameterList index out of range: " +
                                std::to_string(idx) + " >= " + std::to_string(params_.size()));
    }
    return params_[idx];
}

auto ParameterList::forward_impl(const Variable& /*input*/) -> Variable {
    throw NotImplementedError("ParameterList does not implement forward(). "
                             "Use it to store parameters and access them by index.");
}

auto ParameterList::parameters() -> std::vector<std::shared_ptr<Variable>> {
    return params_;
}

auto ParameterList::named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> {
    std::vector<std::pair<std::string, std::shared_ptr<Variable>>> result;
    for (size_t i = 0; i < params_.size(); ++i) {
        result.emplace_back(std::to_string(i), params_[i]);
    }
    return result;
}

auto ParameterList::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;
    for (size_t i = 0; i < params_.size(); ++i) {
        // Clone so the snapshot does not alias the live parameter Storage,
        // matching Module::state_dict(). Returning the live tensor would let a
        // later in-place update mutate the saved snapshot.
        state[std::to_string(i)] = params_[i]->tensor().clone();
    }
    return state;
}

auto ParameterList::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    for (size_t i = 0; i < params_.size(); ++i) {
        auto key = std::to_string(i);
        auto it = state.find(key);
        if (it != state.end()) {
            // R.18-preserving, dtype/device-adapting in-place copy (mirrors
            // Module::load_state_dict). Rebinding the Storage via `= clone()`
            // would orphan aliasing views (FSDP2 shards, saved-for-backward
            // captures, optimizer-held shared_ptrs) and silently drop the
            // checkpoint's dtype/device adaptation.
            detail::load_param_in_place(params_[i]->tensor(), it->second, key, "parameter");
        }
    }
}

// ============================================================================
// ParameterDict implementation
// ============================================================================

auto ParameterDict::insert(const std::string& key, Variable param) -> ParameterDict& {
    auto param_ptr = std::make_shared<Variable>(std::move(param));
    param_ptr->make_thread_safe();
    if (params_.count(key)) {
        params_[key] = param_ptr;
    } else {
        order_.push_back(key);
        params_[key] = param_ptr;
    }
    // Alias the SAME shared_ptr into the base parameters_ map so that
    // Module::to(Device)/to(DType) (which mutate parameters_) and params_
    // refer to one Variable object. register_parameter() would store a copy,
    // leaving params_ stale after a device/dtype move.
    register_parameter_shared(key, param_ptr);
    return *this;
}

auto ParameterDict::at(const std::string& key) const -> std::shared_ptr<Variable> {
    auto it = params_.find(key);
    if (it == params_.end()) {
        throw std::out_of_range("ParameterDict key not found: " + key);
    }
    return it->second;
}

auto ParameterDict::contains(const std::string& key) const -> bool {
    return params_.count(key) > 0;
}

auto ParameterDict::erase(const std::string& key) -> void {
    auto it = params_.find(key);
    if (it == params_.end()) {
        throw std::out_of_range("ParameterDict key not found: " + key);
    }
    params_.erase(it);
    order_.erase(std::remove(order_.begin(), order_.end(), key), order_.end());
    parameters_.erase(key);
}

auto ParameterDict::forward_impl(const Variable& /*input*/) -> Variable {
    throw NotImplementedError("ParameterDict does not implement forward(). "
                             "Use it to store parameters and access them by key.");
}

auto ParameterDict::parameters() -> std::vector<std::shared_ptr<Variable>> {
    std::vector<std::shared_ptr<Variable>> result;
    for (const auto& key : order_) {
        result.push_back(params_.at(key));
    }
    return result;
}

auto ParameterDict::named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> {
    std::vector<std::pair<std::string, std::shared_ptr<Variable>>> result;
    for (const auto& key : order_) {
        result.emplace_back(key, params_.at(key));
    }
    return result;
}

auto ParameterDict::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;
    for (const auto& key : order_) {
        // Clone so the snapshot does not alias the live parameter Storage,
        // matching Module::state_dict().
        state[key] = params_.at(key)->tensor().clone();
    }
    return state;
}

auto ParameterDict::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    for (const auto& key : order_) {
        auto it = state.find(key);
        if (it != state.end()) {
            // R.18-preserving, dtype/device-adapting in-place copy (mirrors
            // Module::load_state_dict). See ParameterList::load_state_dict.
            detail::load_param_in_place(params_.at(key)->tensor(), it->second, key, "parameter");
        }
    }
}

// ============================================================================
// Hook System Implementation (Phase 2 Offload Support)
// ============================================================================

auto Module::register_forward_pre_hook(ForwardPreHook hook) -> size_t {
    auto id = next_hook_id_++;
    forward_pre_hooks_.emplace(id, std::move(hook));
    has_forward_hooks_ = true;
    return id;
}

auto Module::register_forward_post_hook(ForwardPostHook hook) -> size_t {
    auto id = next_hook_id_++;
    forward_post_hooks_.emplace(id, std::move(hook));
    has_forward_hooks_ = true;
    return id;
}

auto Module::register_forward_pre_hook_multi(ForwardPreHookMulti hook) -> size_t {
    auto id = next_hook_id_++;
    forward_pre_hooks_multi_.emplace(id, std::move(hook));
    has_forward_hooks_ = true;
    return id;
}

auto Module::register_forward_post_hook_multi(ForwardPostHookMulti hook) -> size_t {
    auto id = next_hook_id_++;
    forward_post_hooks_multi_.emplace(id, std::move(hook));
    has_forward_hooks_ = true;
    return id;
}

auto Module::call_forward_pre_hooks_multi(const std::vector<Variable>& inputs) -> void {
    for (auto& [id, hook] : forward_pre_hooks_multi_) {
        hook(this, inputs);
    }
}

auto Module::call_forward_post_hooks_multi(const std::vector<Variable>& inputs,
                                            const std::vector<Variable>& outputs) -> void {
    for (auto& [id, hook] : forward_post_hooks_multi_) {
        hook(this, inputs, outputs);
    }
}

auto Module::register_backward_pre_hook(BackwardPreHook hook) -> size_t {
    auto id = next_hook_id_++;
    backward_pre_hooks_.emplace(id, std::move(hook));
    has_backward_hooks_ = true;
    return id;
}

auto Module::register_backward_post_hook(BackwardPostHook hook) -> size_t {
    auto id = next_hook_id_++;
    backward_post_hooks_.emplace(id, std::move(hook));
    has_backward_hooks_ = true;
    return id;
}

auto Module::remove_hook(size_t hook_id) -> void {
    // Search all hook maps and erase by ID
    if (forward_pre_hooks_.erase(hook_id) ||
        forward_post_hooks_.erase(hook_id) ||
        forward_pre_hooks_multi_.erase(hook_id) ||
        forward_post_hooks_multi_.erase(hook_id)) {
        // Removed a forward hook — update flag if no forward hooks remain
        has_forward_hooks_ = !forward_pre_hooks_.empty() ||
                             !forward_post_hooks_.empty() ||
                             !forward_pre_hooks_multi_.empty() ||
                             !forward_post_hooks_multi_.empty();
        return;
    }

    if (backward_pre_hooks_.erase(hook_id) ||
        backward_post_hooks_.erase(hook_id)) {
        // Removed a backward hook — update flag if no backward hooks remain
        has_backward_hooks_ = !backward_pre_hooks_.empty() ||
                              !backward_post_hooks_.empty();
        return;
    }
}

auto Module::call_forward_pre_hooks(const Variable& input) -> void {
    for (auto& [id, hook] : forward_pre_hooks_) {
        hook(this, input);
    }
    // Recursively call hooks on submodules
    for (auto& [name, module] : submodules_) {
        module->call_forward_pre_hooks(input);
    }
}

auto Module::call_forward_pre_hooks() -> void {
    // No-argument version for modules with multiple inputs
    Variable empty;
    call_forward_pre_hooks(empty);
}

auto Module::call_forward_post_hooks(const Variable& input, const Variable& output) -> void {
    for (auto& [id, hook] : forward_post_hooks_) {
        hook(this, input, output);
    }
    // Recursively call hooks on submodules
    for (auto& [name, module] : submodules_) {
        module->call_forward_post_hooks(input, output);
    }
}

auto Module::call_forward_post_hooks() -> void {
    // No-argument version for modules with multiple inputs
    Variable empty;
    call_forward_post_hooks(empty, empty);
}

auto Module::call_backward_pre_hooks(const Variable& grad_output) -> void {
    for (auto& [id, hook] : backward_pre_hooks_) {
        hook(this, grad_output);
    }
    // Recursively call hooks on submodules
    for (auto& [name, module] : submodules_) {
        module->call_backward_pre_hooks(grad_output);
    }
}

auto Module::call_backward_post_hooks(const Variable& grad_input, const Variable& grad_output) -> void {
    for (auto& [id, hook] : backward_post_hooks_) {
        hook(this, grad_input, grad_output);
    }
    // Recursively call hooks on submodules
    for (auto& [name, module] : submodules_) {
        module->call_backward_post_hooks(grad_input, grad_output);
    }
}

// ============================================================================
// ModuleHookFunction Implementation
// ============================================================================

auto ModuleHookFunction::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    // Identity function - just pass through the input
    // Save the input for backward (to pass to hooks)
    if (!inputs.empty()) {
        save_for_backward({inputs[0].tensor()});
    }
    return inputs;
}

auto ModuleHookFunction::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    if (grad_outputs.empty() || !module_) {
        return grad_outputs;
    }

    // Create Variables from gradients for hook calls
    Variable grad_output_var(grad_outputs[0], false);

    // Call backward pre-hooks
    for (auto& [id, hook] : module_->backward_pre_hooks_) {
        hook(module_, grad_output_var);
    }

    // The gradient passes through unchanged (identity)
    Variable grad_input_var = grad_output_var;

    // Call backward post-hooks
    for (auto& [id, hook] : module_->backward_post_hooks_) {
        hook(module_, grad_input_var, grad_output_var);
    }

    // Return gradient unchanged
    return grad_outputs;
}

auto wrap_with_backward_hooks(Module* module, const Variable& input, Variable output) -> Variable {
    // Only wrap if the output requires grad (to be part of autograd graph)
    if (!output.requires_grad()) {
        return output;
    }

    // Create a ModuleHookFunction and apply it to the output
    auto hook_fn = std::make_shared<ModuleHookFunction>(module, input);

    // Apply the function to wrap the output in the computation graph
    auto result = hook_fn->forward({output});

    if (!result.empty()) {
        // Link the hook function into the computation graph
        auto& wrapped_output = result[0];

        // Set the grad_fn to our hook function so it gets called during backward
        // We need to chain it with the existing grad_fn
        auto existing_grad_fn = output.grad_fn();
        if (existing_grad_fn) {
            hook_fn->set_next_functions({existing_grad_fn});
        }
        hook_fn->set_input_variables({output});

        // Create new Variable with the hook function as grad_fn
        Variable hooked_output(wrapped_output.tensor(), true);
        hooked_output.set_grad_fn(hook_fn);

        return hooked_output;
    }

    return output;
}

} // namespace tenzor::nn
