/**
 * @file test_rnn_cells_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for RNNCell and GRUCell layers
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/rnn.hpp>
#include "multi_backend_dtype_fixture.hpp"

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
    convert_model(cell);

    auto input = createInput({batch_size, input_size}, false);
    auto hx = Variable(createRandn({batch_size, hidden_size}), false);

    auto output = cell.forward(input, hx);
    expectShape(output.tensor(), {batch_size, hidden_size});
    expectDevice(output.tensor());
    expectDType(output.tensor());
}

TEST_P(RNNCellsMultiDTypeTest, RNNCell_HiddenShape) {
    auto cell = nn::RNNCell(input_size, hidden_size);
    convert_model(cell);

    auto input = createInput({batch_size, input_size}, false);
    auto hx = Variable(createRandn({batch_size, hidden_size}), false);

    auto h_next = cell.forward(input, hx);
    EXPECT_EQ(h_next.tensor().shape()[0], batch_size);
    EXPECT_EQ(h_next.tensor().shape()[1], hidden_size);
}

TEST_P(RNNCellsMultiDTypeTest, RNNCell_GradientFlow) {
    auto cell = nn::RNNCell(input_size, hidden_size);
    convert_model(cell);

    auto input = createInput({batch_size, input_size}, true);
    auto hx = Variable(createRandn({batch_size, hidden_size}), true);

    auto output = cell.forward(input, hx);
    // Reduce to a scalar via the autograd-aware sum so the backward chain
    // stays connected to the cell's input/hx Variables.
    auto loss_var = tenzor::sum(output);
    loss_var.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_TRUE(hx.grad().has_value());
}

TEST_P(RNNCellsMultiDTypeTest, RNNCell_NoBias) {
    auto cell = nn::RNNCell(input_size, hidden_size, "tanh", /*bias=*/false);
    convert_model(cell);

    auto input = createInput({batch_size, input_size}, false);
    auto hx = Variable(createRandn({batch_size, hidden_size}), false);

    auto output = cell.forward(input, hx);
    expectShape(output.tensor(), {batch_size, hidden_size});
    expectDevice(output.tensor());
    expectDType(output.tensor());
}

// ============================================================================
// GRUCell Tests
// ============================================================================

TEST_P(RNNCellsMultiDTypeTest, GRUCell_OutputShape) {
    auto cell = nn::GRUCell(input_size, hidden_size);
    convert_model(cell);

    auto input = createInput({batch_size, input_size}, false);
    auto hx = Variable(createRandn({batch_size, hidden_size}), false);

    auto output = cell.forward(input, hx);
    expectShape(output.tensor(), {batch_size, hidden_size});
    expectDevice(output.tensor());
    expectDType(output.tensor());
}

TEST_P(RNNCellsMultiDTypeTest, GRUCell_HiddenShape) {
    auto cell = nn::GRUCell(input_size, hidden_size);
    convert_model(cell);

    auto input = createInput({batch_size, input_size}, false);
    auto hx = Variable(createRandn({batch_size, hidden_size}), false);

    auto h_next = cell.forward(input, hx);
    EXPECT_EQ(h_next.tensor().shape()[0], batch_size);
    EXPECT_EQ(h_next.tensor().shape()[1], hidden_size);
}

TEST_P(RNNCellsMultiDTypeTest, GRUCell_GradientFlow) {
    auto cell = nn::GRUCell(input_size, hidden_size);
    convert_model(cell);

    auto input = createInput({batch_size, input_size}, true);
    auto hx = Variable(createRandn({batch_size, hidden_size}), true);

    auto output = cell.forward(input, hx);
    // Reduce to a scalar via the autograd-aware sum so the backward chain
    // stays connected to the cell's input/hx Variables.
    auto loss_var = tenzor::sum(output);
    loss_var.backward();

    EXPECT_TRUE(input.grad().has_value());
    EXPECT_TRUE(hx.grad().has_value());
}

TEST_P(RNNCellsMultiDTypeTest, GRUCell_NoBias) {
    auto cell = nn::GRUCell(input_size, hidden_size, /*bias=*/false);
    convert_model(cell);

    auto input = createInput({batch_size, input_size}, false);
    auto hx = Variable(createRandn({batch_size, hidden_size}), false);

    auto output = cell.forward(input, hx);
    expectShape(output.tensor(), {batch_size, hidden_size});
    expectDevice(output.tensor());
    expectDType(output.tensor());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(RNNCellsMultiDTypeTest);
