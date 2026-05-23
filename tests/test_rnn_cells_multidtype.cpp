/**
 * @file test_rnn_cells_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for RNNCell and GRUCell layers
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/rnn.hpp>
#include "multi_backend_dtype_fixture.hpp"
#include "grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class RNNCellsMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    static constexpr int64_t batch_size = 4;
    static constexpr int64_t input_size = 8;
    static constexpr int64_t hidden_size = 16;
};

// ============================================================================
// RNNCell Tests
// ============================================================================

TEST_P(RNNCellsMultiDTypeTest, RNNCell_OutputShape) {
    auto cell = nn::RNNCell(input_size, hidden_size);
    auto in_cpu = tenzor::randn({batch_size, input_size}, DType::Float32, Device::cpu());
    auto hx_cpu = tenzor::randn({batch_size, hidden_size}, DType::Float32, Device::cpu());
    auto ref = cell.forward(Variable(in_cpu, false), Variable(hx_cpu, false));

    convert_model(cell);
    auto input = Variable(in_cpu.to(dtype_).to(device_), false);
    auto hx = Variable(hx_cpu.to(dtype_).to(device_), false);

    auto output = cell.forward(input, hx);
    expectShape(output.tensor(), {batch_size, hidden_size});
    expectDevice(output.tensor());
    expectDType(output.tensor());
    expectTensorNear(output.tensor(), ref.tensor(), std::max(atol_, 5e-2f));
}

TEST_P(RNNCellsMultiDTypeTest, RNNCell_HiddenShape) {
    auto cell = nn::RNNCell(input_size, hidden_size);
    auto in_cpu = tenzor::randn({batch_size, input_size}, DType::Float32, Device::cpu());
    auto hx_cpu = tenzor::randn({batch_size, hidden_size}, DType::Float32, Device::cpu());
    auto ref = cell.forward(Variable(in_cpu, false), Variable(hx_cpu, false));

    convert_model(cell);
    auto input = Variable(in_cpu.to(dtype_).to(device_), false);
    auto hx = Variable(hx_cpu.to(dtype_).to(device_), false);

    auto h_next = cell.forward(input, hx);
    EXPECT_EQ(h_next.tensor().shape()[0], batch_size);
    EXPECT_EQ(h_next.tensor().shape()[1], hidden_size);
    expectTensorNear(h_next.tensor(), ref.tensor(), std::max(atol_, 5e-2f));
}

TEST_P(RNNCellsMultiDTypeTest, RNNCell_GradientFlow) {
    auto cell_ref = nn::RNNCell(input_size, hidden_size);
    auto in_cpu = tenzor::randn({batch_size, input_size}, DType::Float32, Device::cpu());
    auto hx_cpu = tenzor::randn({batch_size, hidden_size}, DType::Float32, Device::cpu());
    auto in_ref = Variable(in_cpu, true);
    auto hx_ref = Variable(hx_cpu, true);
    auto out_ref = cell_ref.forward(in_ref, hx_ref);
    tenzor::sum(out_ref).backward();

    auto cell = nn::RNNCell(input_size, hidden_size);
    convert_model(cell);
    auto input = Variable(in_cpu.to(dtype_).to(device_), true);
    auto hx = Variable(hx_cpu.to(dtype_).to(device_), true);

    auto output = cell.forward(input, hx);
    auto loss_var = tenzor::sum(output);
    loss_var.backward();

    EXPECT_GRAD_FLOWS(input);
    EXPECT_GRAD_FLOWS(hx);
    expectTensorNear(output.tensor(), out_ref.tensor(), std::max(atol_, 5e-2f));
    expectTensorNear(input.grad().value(), in_ref.grad().value(),
                     std::max(atol_, 5e-2f));
    expectTensorNear(hx.grad().value(), hx_ref.grad().value(),
                     std::max(atol_, 5e-2f));
}

TEST_P(RNNCellsMultiDTypeTest, RNNCell_NoBias) {
    auto cell = nn::RNNCell(input_size, hidden_size, "tanh", /*bias=*/false);
    auto in_cpu = tenzor::randn({batch_size, input_size}, DType::Float32, Device::cpu());
    auto hx_cpu = tenzor::randn({batch_size, hidden_size}, DType::Float32, Device::cpu());
    auto ref = cell.forward(Variable(in_cpu, false), Variable(hx_cpu, false));

    convert_model(cell);
    auto input = Variable(in_cpu.to(dtype_).to(device_), false);
    auto hx = Variable(hx_cpu.to(dtype_).to(device_), false);

    auto output = cell.forward(input, hx);
    expectShape(output.tensor(), {batch_size, hidden_size});
    expectDevice(output.tensor());
    expectDType(output.tensor());
    expectTensorNear(output.tensor(), ref.tensor(), std::max(atol_, 5e-2f));
}

// ============================================================================
// GRUCell Tests
// ============================================================================

TEST_P(RNNCellsMultiDTypeTest, GRUCell_OutputShape) {
    auto cell = nn::GRUCell(input_size, hidden_size);
    auto in_cpu = tenzor::randn({batch_size, input_size}, DType::Float32, Device::cpu());
    auto hx_cpu = tenzor::randn({batch_size, hidden_size}, DType::Float32, Device::cpu());
    auto ref = cell.forward(Variable(in_cpu, false), Variable(hx_cpu, false));

    convert_model(cell);
    auto input = Variable(in_cpu.to(dtype_).to(device_), false);
    auto hx = Variable(hx_cpu.to(dtype_).to(device_), false);

    auto output = cell.forward(input, hx);
    expectShape(output.tensor(), {batch_size, hidden_size});
    expectDevice(output.tensor());
    expectDType(output.tensor());
    expectTensorNear(output.tensor(), ref.tensor(), std::max(atol_, 5e-2f));
}

TEST_P(RNNCellsMultiDTypeTest, GRUCell_HiddenShape) {
    auto cell = nn::GRUCell(input_size, hidden_size);
    auto in_cpu = tenzor::randn({batch_size, input_size}, DType::Float32, Device::cpu());
    auto hx_cpu = tenzor::randn({batch_size, hidden_size}, DType::Float32, Device::cpu());
    auto ref = cell.forward(Variable(in_cpu, false), Variable(hx_cpu, false));

    convert_model(cell);
    auto input = Variable(in_cpu.to(dtype_).to(device_), false);
    auto hx = Variable(hx_cpu.to(dtype_).to(device_), false);

    auto h_next = cell.forward(input, hx);
    EXPECT_EQ(h_next.tensor().shape()[0], batch_size);
    EXPECT_EQ(h_next.tensor().shape()[1], hidden_size);
    expectTensorNear(h_next.tensor(), ref.tensor(), std::max(atol_, 5e-2f));
}

TEST_P(RNNCellsMultiDTypeTest, GRUCell_GradientFlow) {
    auto cell_ref = nn::GRUCell(input_size, hidden_size);
    auto in_cpu = tenzor::randn({batch_size, input_size}, DType::Float32, Device::cpu());
    auto hx_cpu = tenzor::randn({batch_size, hidden_size}, DType::Float32, Device::cpu());
    auto in_ref = Variable(in_cpu, true);
    auto hx_ref = Variable(hx_cpu, true);
    auto out_ref = cell_ref.forward(in_ref, hx_ref);
    tenzor::sum(out_ref).backward();

    auto cell = nn::GRUCell(input_size, hidden_size);
    convert_model(cell);
    auto input = Variable(in_cpu.to(dtype_).to(device_), true);
    auto hx = Variable(hx_cpu.to(dtype_).to(device_), true);

    auto output = cell.forward(input, hx);
    auto loss_var = tenzor::sum(output);
    loss_var.backward();

    EXPECT_GRAD_FLOWS(input);
    EXPECT_GRAD_FLOWS(hx);
    expectTensorNear(output.tensor(), out_ref.tensor(), std::max(atol_, 5e-2f));
    expectTensorNear(input.grad().value(), in_ref.grad().value(),
                     std::max(atol_, 5e-2f));
    expectTensorNear(hx.grad().value(), hx_ref.grad().value(),
                     std::max(atol_, 5e-2f));
}

TEST_P(RNNCellsMultiDTypeTest, GRUCell_NoBias) {
    auto cell = nn::GRUCell(input_size, hidden_size, /*bias=*/false);
    auto in_cpu = tenzor::randn({batch_size, input_size}, DType::Float32, Device::cpu());
    auto hx_cpu = tenzor::randn({batch_size, hidden_size}, DType::Float32, Device::cpu());
    auto ref = cell.forward(Variable(in_cpu, false), Variable(hx_cpu, false));

    convert_model(cell);
    auto input = Variable(in_cpu.to(dtype_).to(device_), false);
    auto hx = Variable(hx_cpu.to(dtype_).to(device_), false);

    auto output = cell.forward(input, hx);
    expectShape(output.tensor(), {batch_size, hidden_size});
    expectDevice(output.tensor());
    expectDType(output.tensor());
    expectTensorNear(output.tensor(), ref.tensor(), std::max(atol_, 5e-2f));
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(RNNCellsMultiDTypeTest);
