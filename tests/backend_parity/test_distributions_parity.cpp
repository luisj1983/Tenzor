/**
 * @file test_distributions_parity.cpp
 * @brief Backend parity for closed-form distribution methods (Phase 5.1).
 *
 * Samplers are covered in test_sampling_parity.cpp (Phase 3.8). This file
 * focuses on the deterministic methods — log_prob, entropy, mean, variance —
 * where we can check exact numerical parity across backends.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributions/distribution.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class DistributionsParity : public BackendTest {};
namespace D = tenzor::distributions;

// ============================================================================
// Normal distribution
// ============================================================================

TEST_P(DistributionsParity, Normal_LogProb) {
    auto loc = randn({8}, DType::Float32, Device::cpu());
    auto scale = rand({8}, DType::Float32, Device::cpu()) + 0.5f;  // strictly positive
    auto value = randn({8}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("distributions parity");

    Tensor ref;
    try {
        D::Normal n(loc, scale);
        ref = n.log_prob(value);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Normal.log_prob CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            D::Normal n(loc.to(backends[i]), scale.to(backends[i]));
            auto got = n.log_prob(value.to(backends[i]));
            backends[i].synchronize();
            SCOPED_TRACE(std::string("Normal.log_prob on ")
                         + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, got.to(Device::cpu()), 1e-4f, 1e-6f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Normal.log_prob failed on "
                      << backend_name(backends[i]) << ": " << e.what()
                      << std::endl;
        }
    }
}

TEST_P(DistributionsParity, Normal_Entropy) {
    auto loc = randn({8}, DType::Float32, Device::cpu());
    auto scale = rand({8}, DType::Float32, Device::cpu()) + 0.5f;

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("distributions parity");

    Tensor ref;
    try {
        D::Normal n(loc, scale);
        ref = n.entropy();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Normal.entropy CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            D::Normal n(loc.to(backends[i]), scale.to(backends[i]));
            auto got = n.entropy();
            backends[i].synchronize();
            SCOPED_TRACE(std::string("Normal.entropy on ")
                         + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, got.to(Device::cpu()), 1e-4f, 1e-6f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Normal.entropy failed on "
                      << backend_name(backends[i]) << ": " << e.what()
                      << std::endl;
        }
    }
}

// ============================================================================
// Uniform distribution
// ============================================================================

TEST_P(DistributionsParity, Uniform_LogProb) {
    auto low = zeros({8}, DType::Float32, Device::cpu());
    auto high = ones({8}, DType::Float32, Device::cpu());
    auto value = rand({8}, DType::Float32, Device::cpu());

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("distributions parity");

    Tensor ref;
    try {
        D::Uniform u(low, high);
        ref = u.log_prob(value);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Uniform.log_prob CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            D::Uniform u(low.to(backends[i]), high.to(backends[i]));
            auto got = u.log_prob(value.to(backends[i]));
            backends[i].synchronize();
            SCOPED_TRACE(std::string("Uniform.log_prob on ")
                         + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, got.to(Device::cpu()), 1e-4f, 1e-6f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Uniform.log_prob failed on "
                      << backend_name(backends[i]) << ": " << e.what()
                      << std::endl;
        }
    }
}

TEST_P(DistributionsParity, Uniform_Entropy) {
    auto low = zeros({8}, DType::Float32, Device::cpu()) - 2.0f;
    auto high = ones({8}, DType::Float32, Device::cpu()) * 3.0f;

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("distributions parity");

    Tensor ref;
    try {
        D::Uniform u(low, high);
        ref = u.entropy();
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Uniform.entropy CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            D::Uniform u(low.to(backends[i]), high.to(backends[i]));
            auto got = u.entropy();
            backends[i].synchronize();
            SCOPED_TRACE(std::string("Uniform.entropy on ")
                         + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, got.to(Device::cpu()), 1e-4f, 1e-6f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Uniform.entropy failed on "
                      << backend_name(backends[i]) << ": " << e.what()
                      << std::endl;
        }
    }
}

// ============================================================================
// Exponential distribution
// ============================================================================

TEST_P(DistributionsParity, Exponential_LogProb) {
    auto rate = rand({8}, DType::Float32, Device::cpu()) + 0.5f;
    auto value = rand({8}, DType::Float32, Device::cpu()) + 0.1f;

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("distributions parity");

    Tensor ref;
    try {
        D::Exponential e(rate);
        ref = e.log_prob(value);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Exponential.log_prob CPU reference failed: " << e.what();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            D::Exponential e(rate.to(backends[i]));
            auto got = e.log_prob(value.to(backends[i]));
            backends[i].synchronize();
            SCOPED_TRACE(std::string("Exponential.log_prob on ")
                         + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, got.to(Device::cpu()), 1e-4f, 1e-6f);
        } catch (const std::exception& ex) {
            ADD_FAILURE() << "Exponential.log_prob failed on "
                      << backend_name(backends[i]) << ": " << ex.what()
                      << std::endl;
        }
    }
}

INSTANTIATE_BACKEND_TESTS(DistributionsParity);


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }
    int result = RUN_ALL_TESTS();
    try { tenzor::finalize(); } catch (...) {}
    return result;
}
