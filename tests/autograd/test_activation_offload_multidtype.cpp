/**
 * @file test_activation_offload_multidtype.cpp
 * @brief Multi-backend tests for per-function activation offload control
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/function.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/ops/creation.hpp>
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class ActivationOffloadMultiDTypeTest : public BackendTest {
protected:
    void SetUp() override {
        BackendTest::SetUp();
    }
    void TearDown() override {
        set_activation_offload(false);
    }
};

TEST_P(ActivationOffloadMultiDTypeTest, DefaultPolicyIsInherit) {
    // Create a simple backward function
    auto func = std::make_shared<AddBackward>();
    EXPECT_EQ(func->offload_policy(), OffloadPolicy::Inherit);
}

TEST_P(ActivationOffloadMultiDTypeTest, SetPolicyAlways) {
    auto func = std::make_shared<AddBackward>();
    func->set_offload_policy(OffloadPolicy::Always);
    EXPECT_EQ(func->offload_policy(), OffloadPolicy::Always);
}

TEST_P(ActivationOffloadMultiDTypeTest, SetPolicyNever) {
    auto func = std::make_shared<AddBackward>();
    func->set_offload_policy(OffloadPolicy::Never);
    EXPECT_EQ(func->offload_policy(), OffloadPolicy::Never);
}

TEST_P(ActivationOffloadMultiDTypeTest, SetPolicyWithMinBytes) {
    auto func = std::make_shared<AddBackward>();
    func->set_offload_policy(OffloadPolicy::Always, 1024);
    EXPECT_EQ(func->offload_policy(), OffloadPolicy::Always);
}

TEST_P(ActivationOffloadMultiDTypeTest, ShouldOffloadCPUTensorReturnsFalse) {
    auto func = std::make_shared<AddBackward>();
    func->set_offload_policy(OffloadPolicy::Always);

    auto tensor = tenzor::ones({10}, DType::Float32, device).to(Device::cpu());
    // CPU tensors should never be offloaded (they're already on CPU)
    EXPECT_FALSE(func->should_offload(tensor));
}

TEST_P(ActivationOffloadMultiDTypeTest, InheritPolicyFollowsGlobal) {
    auto func = std::make_shared<AddBackward>();
    // Default policy is Inherit

    auto tensor = tenzor::ones({10}, DType::Float32, device).to(Device::cpu());

    // Global offload disabled -> should not offload
    set_activation_offload(false);
    EXPECT_FALSE(func->should_offload(tensor));

    // Global offload enabled -> still false for CPU tensor
    set_activation_offload(true);
    EXPECT_FALSE(func->should_offload(tensor));
}

TEST_P(ActivationOffloadMultiDTypeTest, NeverPolicyOverridesGlobal) {
    auto func = std::make_shared<AddBackward>();
    func->set_offload_policy(OffloadPolicy::Never);

    set_activation_offload(true);
    auto tensor = tenzor::ones({10}, DType::Float32, device).to(Device::cpu());
    EXPECT_FALSE(func->should_offload(tensor));
}

INSTANTIATE_BACKEND_TESTS(ActivationOffloadMultiDTypeTest);
