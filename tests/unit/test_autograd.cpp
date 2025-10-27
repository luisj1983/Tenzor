#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class AutogradTest : public BackendTest {};

TEST_P(AutogradTest, VariableCreation) {
    auto data = ones({2, 2}, DType::Float32, device);
    auto var = Variable(data, true);

    EXPECT_TRUE(var.requires_grad()) << "Failed on " << device.to_string();
    EXPECT_FALSE(var.has_grad()) << "Failed on " << device.to_string();
}

TEST_P(AutogradTest, Detach) {
    auto var = Variable(ones({2, 2}, DType::Float32, device), true);
    auto detached = var.detach();

    EXPECT_FALSE(detached.requires_grad()) << "Failed on " << device.to_string();
}

TEST_P(AutogradTest, SimpleAddBackward) {
    // Test: c = a + b
    // dc/da = 1, dc/db = 1
    auto a = Variable(ones({2, 2}, DType::Float32, device), true);
    auto b = Variable(ones({2, 2}, DType::Float32, device), true);
    auto c = a + b;

    // Backward with gradient of ones
    c.backward(ones({2, 2}, DType::Float32, device));

    // Check gradients exist
    ASSERT_TRUE(a.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(b.has_grad()) << "Failed on " << device.to_string();

    // Check gradient values
    auto a_grad = a.grad().value().to(Device::cpu());
    auto b_grad = b.grad().value().to(Device::cpu());

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(a_grad.data<float>()[i], 1.0f) << "Failed on " << device.to_string();
        EXPECT_FLOAT_EQ(b_grad.data<float>()[i], 1.0f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradTest, SimpleSubBackward) {
    // Test: c = a - b
    // dc/da = 1, dc/db = -1
    auto a = Variable(ones({2, 2}, DType::Float32, device) * 3.0f, true);
    auto b = Variable(ones({2, 2}, DType::Float32, device) * 2.0f, true);
    auto c = a - b;

    c.backward(ones({2, 2}, DType::Float32, device));

    ASSERT_TRUE(a.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(b.has_grad()) << "Failed on " << device.to_string();

    auto a_grad = a.grad().value().to(Device::cpu());
    auto b_grad = b.grad().value().to(Device::cpu());

    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(a_grad.data<float>()[i], 1.0f) << "Failed on " << device.to_string();
        EXPECT_FLOAT_EQ(b_grad.data<float>()[i], -1.0f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradTest, SimpleMulBackward) {
    // Test: c = a * b
    // dc/da = b, dc/db = a
    auto a_data = ones({2, 2}, DType::Float32, device) * 2.0f;
    auto b_data = ones({2, 2}, DType::Float32, device) * 3.0f;
    auto a = Variable(a_data, true);
    auto b = Variable(b_data, true);
    auto c = a * b;

    c.backward(ones({2, 2}, DType::Float32, device));

    ASSERT_TRUE(a.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(b.has_grad()) << "Failed on " << device.to_string();

    auto a_grad = a.grad().value().to(Device::cpu());
    auto b_grad = b.grad().value().to(Device::cpu());

    // da should be b (3.0), db should be a (2.0)
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(a_grad.data<float>()[i], 3.0f) << "Failed on " << device.to_string();
        EXPECT_FLOAT_EQ(b_grad.data<float>()[i], 2.0f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradTest, SimpleDivBackward) {
    // Test: c = a / b
    // dc/da = 1/b, dc/db = -a/(b^2)
    auto a_data = ones({2, 2}, DType::Float32, device) * 6.0f;
    auto b_data = ones({2, 2}, DType::Float32, device) * 2.0f;
    auto a = Variable(a_data, true);
    auto b = Variable(b_data, true);
    auto c = a / b;

    c.backward(ones({2, 2}, DType::Float32, device));

    ASSERT_TRUE(a.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(b.has_grad()) << "Failed on " << device.to_string();

    auto a_grad = a.grad().value().to(Device::cpu());
    auto b_grad = b.grad().value().to(Device::cpu());

    // da should be 1/2 = 0.5
    // db should be -6/(2^2) = -1.5
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(a_grad.data<float>()[i], 0.5f) << "Failed on " << device.to_string();
        EXPECT_FLOAT_EQ(b_grad.data<float>()[i], -1.5f) << "Failed on " << device.to_string();
    }
}

TEST_P(AutogradTest, ChainedOperations) {
    // Test: d = (a + b) * c
    auto a = Variable(ones({2, 2}, DType::Float32, device) * 2.0f, true);
    auto b = Variable(ones({2, 2}, DType::Float32, device) * 3.0f, true);
    auto c = Variable(ones({2, 2}, DType::Float32, device) * 4.0f, true);

    auto d = (a + b) * c;
    d.backward(ones({2, 2}, DType::Float32, device));

    ASSERT_TRUE(a.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(b.has_grad()) << "Failed on " << device.to_string();
    ASSERT_TRUE(c.has_grad()) << "Failed on " << device.to_string();

    auto a_grad = a.grad().value().to(Device::cpu());
    auto b_grad = b.grad().value().to(Device::cpu());
    auto c_grad = c.grad().value().to(Device::cpu());

    // d = (a + b) * c = (2 + 3) * 4 = 20
    // dd/da = c = 4
    // dd/db = c = 4
    // dd/dc = (a + b) = 5
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(a_grad.data<float>()[i], 4.0f) << "Failed on " << device.to_string();
        EXPECT_FLOAT_EQ(b_grad.data<float>()[i], 4.0f) << "Failed on " << device.to_string();
        EXPECT_FLOAT_EQ(c_grad.data<float>()[i], 5.0f) << "Failed on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(AutogradTest);
