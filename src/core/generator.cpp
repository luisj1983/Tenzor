#include "tenzor/core/generator.hpp"
#include "tenzor/ops/creation.hpp"  // tenzor::manual_seed (for global RNG restore)
#include <chrono>
#include <sstream>
#include <unordered_map>

namespace tenzor {

namespace {

// Draw a full 64-bit nondeterministic seed. std::random_device yields 32-bit
// values on most implementations, so combine two draws to cover the engine's
// 64-bit seed space.
auto random_seed_u64() -> uint64_t {
    std::random_device rd;
    return (static_cast<uint64_t>(rd()) << 32) | static_cast<uint64_t>(rd());
}

// Serialize an mt19937_64 engine to a vector of uint64. We use the standard
// stream-based serialization so we don't depend on libstdc++ internals.
auto serialize_engine_u64(const std::mt19937_64& eng) -> std::vector<uint64_t> {
    std::ostringstream oss;
    oss << eng;
    // The stream emits space-separated decimal integers: 624 (or 312 for
    // mt19937_64) state words plus the position counter.
    std::vector<uint64_t> out;
    out.reserve(313);
    std::istringstream iss(oss.str());
    uint64_t v;
    while (iss >> v) out.push_back(v);
    return out;
}

auto deserialize_engine_u64(std::mt19937_64& eng,
                            const std::vector<uint64_t>& state) -> void {
    if (state.empty()) return;
    std::ostringstream oss;
    for (size_t i = 0; i < state.size(); ++i) {
        if (i) oss << ' ';
        oss << state[i];
    }
    std::istringstream iss(oss.str());
    iss >> eng;
}

} // namespace

Generator::Generator(Device device)
    : device_(device)
    , seed_(random_seed_u64())
    , initial_seed_(seed_)
    , engine_(seed_)
{
    // seed_/initial_seed_ now match the engine's actual seed, so an unseeded
    // generator reports a value that genuinely reproduces its stream.
}

auto Generator::manual_seed(uint64_t seed) -> Generator& {
    std::lock_guard lock(mutex_);
    seed_ = seed;
    initial_seed_ = seed;
    engine_.seed(seed);
    return *this;
}

auto Generator::seed() const -> uint64_t {
    return seed_;
}

auto Generator::initial_seed() const -> uint64_t {
    return initial_seed_;
}

auto Generator::device() const -> Device {
    return device_;
}

auto Generator::clone() const -> std::unique_ptr<Generator> {
    auto g = std::make_unique<Generator>(device_);
    g->seed_ = seed_;
    g->initial_seed_ = initial_seed_;
    g->engine_ = engine_;
    return g;
}

auto Generator::engine() -> std::mt19937_64& {
    // No lock: the lock_guard here would be released before the caller uses the
    // returned reference, providing a false sense of safety. Callers needing
    // thread safety must use with_engine().
    return engine_;
}

auto Generator::next_seed() -> uint64_t {
    std::lock_guard lock(mutex_);
    return engine_();
}

auto Generator::get_state() const -> GeneratorState {
    // const_cast: the std::mt19937_64 stream-out operator isn't const on
    // every libstdc++ release, and we also need the mutex to avoid tearing
    // against a concurrent next_seed() call.
    auto& self = const_cast<Generator&>(*this);
    std::lock_guard lock(self.mutex_);
    GeneratorState s;
    s.engine_state = serialize_engine_u64(self.engine_);
    s.seed = seed_;
    s.initial_seed = initial_seed_;
    return s;
}

auto Generator::set_state(const GeneratorState& state) -> void {
    std::lock_guard lock(mutex_);
    deserialize_engine_u64(engine_, state.engine_state);
    seed_ = state.seed;
    initial_seed_ = state.initial_seed;
}

auto default_generator(Device device) -> Generator& {
    // One default generator per (device type, index), thread-local so each
    // thread has an independent RNG stream without locking. Keying only by
    // type previously aliased rocm:0/oneapi/vulkan/mps/cuda:1 all onto the
    // single cuda:0 generator, giving the wrong device() identity and a shared
    // stream across physically distinct devices.
    static thread_local std::unordered_map<uint64_t, std::unique_ptr<Generator>> gens;
    const uint64_t key = (static_cast<uint64_t>(device.type) << 32) |
                         static_cast<uint32_t>(device.index);
    auto it = gens.find(key);
    if (it == gens.end()) {
        it = gens.emplace(key, std::make_unique<Generator>(device)).first;
    }
    return *it->second;
}

namespace {

auto serialize_engine_u32(const std::mt19937& eng) -> std::vector<uint32_t> {
    std::ostringstream oss;
    oss << eng;
    std::vector<uint32_t> out;
    out.reserve(625);
    std::istringstream iss(oss.str());
    uint32_t v;
    while (iss >> v) out.push_back(v);
    return out;
}

auto deserialize_engine_u32(std::mt19937& eng,
                            const std::vector<uint32_t>& state) -> void {
    if (state.empty()) return;
    std::ostringstream oss;
    for (size_t i = 0; i < state.size(); ++i) {
        if (i) oss << ' ';
        oss << state[i];
    }
    std::istringstream iss(oss.str());
    iss >> eng;
}

} // namespace

auto save_global_rng_state() -> GlobalRngState {
    GlobalRngState s;
    s.engine_state = serialize_engine_u32(detail::get_global_rng_engine());
    s.manual_seed_set = detail::get_global_manual_seed_set();
    s.manual_seed_value = detail::get_global_manual_seed_value();
    return s;
}

auto restore_global_rng_state(const GlobalRngState& state) -> void {
    deserialize_engine_u32(detail::get_global_rng_engine(), state.engine_state);
    detail::set_global_manual_seed_set(state.manual_seed_set);
    detail::set_global_manual_seed_value(state.manual_seed_value);
}

} // namespace tenzor
