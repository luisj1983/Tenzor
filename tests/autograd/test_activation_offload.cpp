/**
 * @file test_activation_offload.cpp
 * @brief Tests for per-function activation offload control
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/function.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/ops/creation.hpp>

using namespace tenzor;

class ActivationOffloadTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
    void TearDown() override {
        set_activation_offload(false);
    }
};

TEST_F(ActivationOffloadTest, DefaultPolicyIsInherit) {
    // Create a simple backward function
    auto func = std::make_shared<AddBackward>();
    EXPECT_EQ(func->offload_policy(), OffloadPolicy::Inherit);
}

TEST_F(ActivationOffloadTest, SetPolicyAlways) {
    auto func = std::make_shared<AddBackward>();
    func->set_offload_policy(OffloadPolicy::Always);
    EXPECT_EQ(func->offload_policy(), OffloadPolicy::Always);
}

TEST_F(ActivationOffloadTest, SetPolicyNever) {
    auto func = std::make_shared<AddBackward>();
    func->set_offload_policy(OffloadPolicy::Never);
    EXPECT_EQ(func->offload_policy(), OffloadPolicy::Never);
}

TEST_F(ActivationOffloadTest, SetPolicyWithMinBytes) {
    auto func = std::make_shared<AddBackward>();
    func->set_offload_policy(OffloadPolicy::Always, 1024);
    EXPECT_EQ(func->offload_policy(), OffloadPolicy::Always);
}

TEST_F(ActivationOffloadTest, ShouldOffloadCPUTensorReturnsFalse) {
    auto func = std::make_shared<AddBackward>();
    func->set_offload_policy(OffloadPolicy::Always);

    auto cpu_tensor = tenzor::ones({10}, DType::Float32);
    // CPU tensors should never be offloaded (they're already on CPU)
    EXPECT_FALSE(func->should_offload(cpu_tensor));
}

TEST_F(ActivationOffloadTest, InheritPolicyFollowsGlobal) {
    auto func = std::make_shared<AddBackward>();
    // Default policy is Inherit

    auto cpu_tensor = tenzor::ones({10}, DType::Float32);

    // Global offload disabled -> should not offload
    set_activation_offload(false);
    EXPECT_FALSE(func->should_offload(cpu_tensor));

    // Global offload enabled -> still false for CPU tensor
    set_activation_offload(true);
    EXPECT_FALSE(func->should_offload(cpu_tensor));
}

TEST_F(ActivationOffloadTest, NeverPolicyOverridesGlobal) {
    auto func = std::make_shared<AddBackward>();
    func->set_offload_policy(OffloadPolicy::Never);

    set_activation_offload(true);
    auto cpu_tensor = tenzor::ones({10}, DType::Float32);
    EXPECT_FALSE(func->should_offload(cpu_tensor));
}
