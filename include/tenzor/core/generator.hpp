/**
 * @file generator.hpp
 * @brief Per-op reproducible random number generator
 *
 * Provides a Generator class for independent RNG streams, analogous to
 * PyTorch's torch.Generator. Each Generator maintains its own state and
 * can be passed to random operations for reproducible sampling.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include "device.hpp"

namespace tenzor {

/**
 * @brief Independent random number generator with reproducible seeding.
 *
 * Generators maintain their own RNG state independent of the global seed.
 * Pass a Generator to random ops (rand, randn, randint, etc.) to get
 * reproducible, independent random streams.
 *
 * @code
 * auto g = Generator(Device::cpu());
 * g.manual_seed(42);
 * auto a = rand({3, 4}, DType::Float32, Device::cpu(), g);
 * g.manual_seed(42);
 * auto b = rand({3, 4}, DType::Float32, Device::cpu(), g);
 * // a == b
 * @endcode
 */
class Generator {
public:
    /**
     * @brief Construct a Generator for the given device.
     * @param device Device this generator is associated with (default: CPU)
     */
    explicit Generator(Device device = Device::cpu());

    /**
     * @brief Set the seed for this generator.
     * @param seed Seed value
     * @return Reference to this generator for chaining
     */
    auto manual_seed(uint64_t seed) -> Generator&;

    /**
     * @brief Get the current seed.
     * @return The seed value most recently set by manual_seed()
     */
    [[nodiscard]] auto seed() const -> uint64_t;

    /**
     * @brief Get the initial seed (first seed set on this generator).
     * @return The initial seed value
     */
    [[nodiscard]] auto initial_seed() const -> uint64_t;

    /**
     * @brief Get the device associated with this generator.
     * @return The device
     */
    [[nodiscard]] auto device() const -> Device;

    /**
     * @brief Clone this generator (copies current state).
     * @return A new Generator with identical state
     */
    [[nodiscard]] auto clone() const -> std::unique_ptr<Generator>;

    /**
     * @brief Access the underlying engine (for backend use).
     *
     * Thread-safe: acquires an internal lock. Prefer using the
     * seed() method when possible to avoid lock contention.
     *
     * @return Reference to the mt19937_64 engine
     */
    auto engine() -> std::mt19937_64&;

    /**
     * @brief Get the next seed from this generator's stream.
     *
     * Returns a deterministic seed derived from the generator's state.
     * Useful for backends that need their own RNG initialization.
     *
     * @return A deterministic seed value
     */
    auto next_seed() -> uint64_t;

private:
    Device device_;
    uint64_t seed_{0};
    uint64_t initial_seed_{0};
    std::mt19937_64 engine_;
    std::mutex mutex_;
};

/**
 * @brief Get the default generator for a device.
 *
 * Returns a thread-local default generator. The default generator is
 * seeded by manual_seed() when called globally.
 *
 * @param device Device to get the default generator for
 * @return Reference to the default generator
 */
auto default_generator(Device device = Device::cpu()) -> Generator&;

/**
 * @brief Optional generator reference type used in creation ops.
 */
using OptionalGenerator = std::optional<std::reference_wrapper<Generator>>;

} // namespace tenzor
