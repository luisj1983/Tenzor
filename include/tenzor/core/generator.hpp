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
#include <vector>
#include "device.hpp"

namespace tenzor {

/**
 * @brief Opaque RNG state snapshot for Generator save/restore.
 *
 * Captures the host-side std::mt19937_64 engine state plus seed bookkeeping.
 * All Tenzor backends derive their per-op seeds from this host-side stream
 * (CPU uses it directly; CUDA/ROCm/OneAPI/Vulkan/MPS feed `next_seed()` /
 * `get_global_seed()` into curand/hiprand/Philox UBOs), so saving/restoring
 * this is sufficient to make a recomputed forward see the same random
 * samples as the original forward across every backend.
 */
struct GeneratorState {
    /// Serialized mt19937_64 state (the 312 uint64 internal state vector plus
    /// position counter, captured via the standard ostream operator).
    std::vector<uint64_t> engine_state;
    uint64_t seed{0};
    uint64_t initial_seed{0};
};

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

    /**
     * @brief Snapshot the current RNG state for later restoration.
     *
     * Used by gradient checkpointing to ensure the recomputed forward sees
     * the same random draws as the original forward. Cheap to call.
     *
     * @return Opaque, copyable state.
     */
    [[nodiscard]] auto get_state() const -> GeneratorState;

    /**
     * @brief Restore the RNG to a previously captured state.
     *
     * After this returns, `next_seed()` and `engine()` produce the same
     * sequence as they did right after `state` was captured.
     *
     * @param state State previously returned by `get_state()`.
     */
    auto set_state(const GeneratorState& state) -> void;

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

/**
 * @brief Snapshot of the thread-local global RNG state used by all
 *        creation/stochastic ops (rand, randn, dropout, multinomial, ...).
 *
 * Backends pull seeds from `tenzor::get_global_seed()` (defined in
 * ops/creation.cpp); that helper reads three thread-local values:
 *   - the mt19937 engine
 *   - whether `manual_seed()` was set
 *   - the incrementing `manual_seed_value` counter
 *
 * Checkpoint save/restore must capture all three to keep recomputed
 * forwards bit-identical to the original on every backend.
 */
struct GlobalRngState {
    std::vector<uint32_t> engine_state;   ///< Serialized std::mt19937 state
    bool manual_seed_set{false};
    uint64_t manual_seed_value{0};
};

/// Capture the calling thread's global RNG state (for checkpoint save).
auto save_global_rng_state() -> GlobalRngState;

/// Restore the calling thread's global RNG state (for checkpoint recompute).
auto restore_global_rng_state(const GlobalRngState& state) -> void;

} // namespace tenzor
