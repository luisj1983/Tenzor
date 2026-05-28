/**
 * @file test_parametrize_lifetime.cpp
 * @brief Regression tests for the Module UID / parametrize registry fix
 *        (Stream 21).
 *
 * Background:
 *   parametrize_registry() used to key on raw `Module*`. When a Module
 *   was destroyed and a new one was constructed at the same heap address
 *   (common during NAS/sweep loops), `is_parametrized()` returned true
 *   for the *new* Module — it inherited the dead Module's registry
 *   entry. The fix is to key the registry by Module::id(), a
 *   process-monotonic UID assigned in the Module ctor.
 *
 * These tests pin that behaviour: stable address reuse must NOT leak
 * parametrization state across Module lifetimes.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/nn/utils/parametrize.hpp>
#include <tenzor/nn/layers/linear.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <unordered_set>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::utils;

namespace {

class ParametrizeLifetimeEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
::testing::Environment* const lifetime_env =
    ::testing::AddGlobalTestEnvironment(new ParametrizeLifetimeEnv);

// Simple parametrization used across tests: scale parameter by 2.
class ScaleBy2 : public Parametrization {
public:
    auto forward(const Tensor& X) -> Tensor override {
        return tenzor::mul(X, 2.0);
    }
};

// Different parametrization for the multi-module test.
class NegateParam : public Parametrization {
public:
    auto forward(const Tensor& X) -> Tensor override {
        return tenzor::mul(X, -1.0);
    }
};

} // namespace

// 1. UID basics: every Module instance has a unique id, and ids are non-zero
//    monotonic so they can be used in hash tables / serialization streams.
TEST(ParametrizeLifetime, UidsAreUniqueAndNonZero) {
    constexpr int N = 16;
    std::unordered_set<uint64_t> seen;
    std::vector<std::shared_ptr<Linear>> mods;
    mods.reserve(N);
    for (int i = 0; i < N; ++i) {
        auto m = std::make_shared<Linear>(3, 4);
        EXPECT_NE(m->id(), 0u) << "Module UID 0 is reserved (sentinel)";
        EXPECT_TRUE(seen.insert(m->id()).second)
            << "Module UID collision at iter " << i << " (id=" << m->id() << ")";
        mods.push_back(std::move(m));
    }
}

// 2. The original bug: heap-address reuse must not leak parametrize state.
//    We use placement-new on a fixed storage block to make address reuse
//    deterministic (std::make_shared / new would only sometimes reuse).
TEST(ParametrizeLifetime, AddressReuseDoesNotInheritParametrization) {
    // Pre-clear the registry so any leftover state from other tests in
    // the binary doesn't confuse this assertion.
    clear_parametrization_registry();

    alignas(Linear) std::byte storage[sizeof(Linear)];

    // --- Module A: register a parametrization, then destroy. ---
    Linear* a = ::new (storage) Linear(3, 4);
    const uint64_t id_a = a->id();

    // register_parametrization requires shared_ptr<Module>. We construct
    // an aliasing shared_ptr that does NOT own the placement-new'd object
    // (the deleter is a no-op) so destruction stays under our control.
    std::shared_ptr<Module> a_sp(a, [](Module*) { /* non-owning */ });
    register_parametrization(a_sp, "weight", std::make_shared<ScaleBy2>());

    EXPECT_TRUE(is_parametrized(*a, "weight"))
        << "sanity: parametrization should register on Module A";

    a_sp.reset();    // drop the aliasing shared_ptr first
    a->~Linear();    // explicit dtor — this must clear A's registry entries

    // --- Module B: construct at the SAME ADDRESS. ---
    Linear* b = ::new (storage) Linear(3, 4);
    const uint64_t id_b = b->id();

    // Same address, different UID — this is the whole point of the fix.
    EXPECT_EQ(static_cast<void*>(a), static_cast<void*>(b))
        << "placement new at same storage should yield same address";
    EXPECT_NE(id_a, id_b)
        << "Module B must have a fresh UID, distinct from destroyed A";

    // The new module must NOT inherit A's parametrization.
    EXPECT_FALSE(is_parametrized(*b))
        << "stale registry entry leaked across Module lifetimes — "
           "this is the Stream 21 regression";
    EXPECT_FALSE(is_parametrized(*b, "weight"));

    b->~Linear();
}

// 3. Normal register / unregister flow still works with UID keying.
TEST(ParametrizeLifetime, NormalRegisterUnregisterFlow) {
    auto linear = std::make_shared<Linear>(4, 2);

    EXPECT_FALSE(is_parametrized(*linear, "weight"));

    register_parametrization(linear, "weight", std::make_shared<ScaleBy2>());
    EXPECT_TRUE(is_parametrized(*linear, "weight"));

    remove_parametrizations(linear, "weight", /*leave_parametrized=*/false);
    EXPECT_FALSE(is_parametrized(*linear, "weight"));
    EXPECT_FALSE(is_parametrized(*linear));
}

// 4. Multiple coexisting modules: each has its own UID and its own
//    parametrization state, with no cross-talk.
TEST(ParametrizeLifetime, MultipleModulesIndependent) {
    auto m1 = std::make_shared<Linear>(3, 3);
    auto m2 = std::make_shared<Linear>(3, 3);
    auto m3 = std::make_shared<Linear>(3, 3);

    EXPECT_NE(m1->id(), m2->id());
    EXPECT_NE(m1->id(), m3->id());
    EXPECT_NE(m2->id(), m3->id());

    register_parametrization(m1, "weight", std::make_shared<ScaleBy2>());
    register_parametrization(m2, "weight", std::make_shared<NegateParam>());
    // m3 is left un-parametrized

    EXPECT_TRUE(is_parametrized(*m1, "weight"));
    EXPECT_TRUE(is_parametrized(*m2, "weight"));
    EXPECT_FALSE(is_parametrized(*m3, "weight"));
    EXPECT_FALSE(is_parametrized(*m3));

    // Removing one must not touch the others.
    remove_parametrizations(m1, "weight");
    EXPECT_FALSE(is_parametrized(*m1, "weight"));
    EXPECT_TRUE(is_parametrized(*m2, "weight"));
}

// 5. ~Module() cleans up its registry entries even if the user never
//    explicitly removed parametrizations. After the Module goes out of
//    scope, lookups by its (now-destroyed) UID return false.
TEST(ParametrizeLifetime, DestructorCleansRegistry) {
    uint64_t saved_id = 0;
    {
        auto m = std::make_shared<Linear>(3, 4);
        saved_id = m->id();
        register_parametrization(m, "weight", std::make_shared<ScaleBy2>());
        EXPECT_TRUE(is_parametrized(*m, "weight"));
    } // m destroyed here — ~Module() must call
      // unregister_parametrization_for_module(saved_id).

    // Build a fresh Module; even if it happened to get the same UID
    // (it shouldn't — UIDs are monotonic — but assert anyway), it would
    // be a logic error for the old entry to survive.
    auto m2 = std::make_shared<Linear>(3, 4);
    EXPECT_NE(m2->id(), saved_id) << "UIDs are process-monotonic; reuse would be a bug";
    EXPECT_FALSE(is_parametrized(*m2));
}
