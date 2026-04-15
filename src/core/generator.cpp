#include "tenzor/core/generator.hpp"
#include <chrono>

namespace tenzor {

Generator::Generator(Device device)
    : device_(device)
    , seed_(0)
    , initial_seed_(0)
    , engine_(std::random_device{}())
{
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
    std::lock_guard lock(mutex_);
    return engine_;
}

auto Generator::next_seed() -> uint64_t {
    std::lock_guard lock(mutex_);
    return engine_();
}

auto default_generator(Device device) -> Generator& {
    // One default generator per device type + index, thread-local
    static thread_local Generator cpu_gen(Device::cpu());
    static thread_local Generator cuda_gen(Device::cuda(0));

    if (device.type == Device::Type::CPU) {
        return cpu_gen;
    }
    return cuda_gen;
}

} // namespace tenzor
