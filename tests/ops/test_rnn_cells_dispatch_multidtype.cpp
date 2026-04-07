/**
 * @file test_rnn_cells_dispatch_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for RNN cell dispatch operations
 *
 * Covers: LSTMCell, GRUCell forward shapes and backward gradient flow,
 * LSTM/GRU multi-layer forward, BiLSTM forward.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/rnn.hpp>
#include <tenzor/autograd/variable.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class RNNCellsDispatchMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void skipIfHalf() {
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {
            GTEST_SKIP() << "RNN cells require higher precision than Float16";
        }
    }
};

// ============================================================================
// LSTMCell Tests
// ============================================================================

TEST_P(RNNCellsDispatchMultiDTypeTest, LSTMCellForwardShape) {
    skipIfHalf();
    nn::LSTMCell cell(10, 20);
    convert_model(cell);

    auto input = createInput({4, 10}, false);  // batch=4, input_size=10
    auto hx = createInput({4, 20}, false);     // hidden_size=20
    auto cx = createInput({4, 20}, false);

    auto [h_next, c_next] = cell.forward(input, hx, cx);
    expectShape(h_next.tensor(), {4, 20});
    expectShape(c_next.tensor(), {4, 20});
    expectDevice(h_next.tensor());
    expectDevice(c_next.tensor());
}

TEST_P(RNNCellsDispatchMultiDTypeTest, LSTMCellBackwardGradient) {
    skipIfHalf();
    nn::LSTMCell cell(8, 16);
    convert_model(cell);

    auto input = createInput({2, 8}, true);
    auto hx = createInput({2, 16}, true);
    auto cx = createInput({2, 16}, false);

    auto [h_next, c_next] = cell.forward(input, hx, cx);

    auto grad = tenzor::ones({2, 16}, dtype(), device());
    EXPECT_NO_THROW({ h_next.backward(grad); })
        << "LSTMCell backward threw on " << device().to_string();

    ASSERT_TRUE(input.grad().has_value())
        << "LSTMCell backward did not produce input gradient on " << device().to_string();
    expectShape(*input.grad(), {2, 8});
}

// ============================================================================
// GRUCell Tests
// ============================================================================

TEST_P(RNNCellsDispatchMultiDTypeTest, GRUCellForwardShape) {
    skipIfHalf();
    nn::GRUCell cell(10, 20);
    convert_model(cell);

    auto input = createInput({4, 10}, false);
    auto hx = createInput({4, 20}, false);

    auto h_next = cell.forward(input, hx);
    expectShape(h_next.tensor(), {4, 20});
    expectDevice(h_next.tensor());
}

TEST_P(RNNCellsDispatchMultiDTypeTest, GRUCellBackwardGradient) {
    skipIfHalf();
    nn::GRUCell cell(8, 16);
    convert_model(cell);

    auto input = createInput({2, 8}, true);
    auto hx = createInput({2, 16}, true);

    auto h_next = cell.forward(input, hx);

    auto grad = tenzor::ones({2, 16}, dtype(), device());
    EXPECT_NO_THROW({ h_next.backward(grad); })
        << "GRUCell backward threw on " << device().to_string();

    ASSERT_TRUE(input.grad().has_value())
        << "GRUCell backward did not produce input gradient on " << device().to_string();
    expectShape(*input.grad(), {2, 8});
}

// ============================================================================
// LSTM Multi-Layer Forward
// ============================================================================

TEST_P(RNNCellsDispatchMultiDTypeTest, LSTMMultiLayerForward) {
    skipIfHalf();
    nn::LSTM lstm(10, 20, 2);  // input_size=10, hidden_size=20, num_layers=2
    convert_model(lstm);

    auto input = createInput({5, 4, 10}, false);  // seq_len=5, batch=4, input=10
    auto output = lstm.forward(input);
    // Output should be (seq_len, batch, hidden_size)
    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], 5);
    EXPECT_EQ(shape[1], 4);
    EXPECT_EQ(shape[2], 20);
    expectDevice(output.tensor());
}

// ============================================================================
// GRU Multi-Layer Forward
// ============================================================================

TEST_P(RNNCellsDispatchMultiDTypeTest, GRUMultiLayerForward) {
    skipIfHalf();
    nn::GRU gru(10, 20, 2);
    convert_model(gru);

    auto input = createInput({5, 4, 10}, false);
    auto output = gru.forward(input);
    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], 5);
    EXPECT_EQ(shape[1], 4);
    EXPECT_EQ(shape[2], 20);
    expectDevice(output.tensor());
}

// ============================================================================
// BiLSTM Forward
// ============================================================================

TEST_P(RNNCellsDispatchMultiDTypeTest, BiLSTMForward) {
    skipIfHalf();
    nn::LSTM bilstm(10, 20, 1, true, true);  // bidirectional=true
    convert_model(bilstm);

    auto input = createInput({5, 4, 10}, false);
    auto output = bilstm.forward(input);
    // Bidirectional output: (seq_len, batch, 2*hidden_size)
    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], 5);
    EXPECT_EQ(shape[1], 4);
    EXPECT_EQ(shape[2], 40);  // 2 * 20
    expectDevice(output.tensor());
}

// ============================================================================
// LSTM Backward Gradient Flow
// ============================================================================

TEST_P(RNNCellsDispatchMultiDTypeTest, LSTMBackwardGradient) {
    skipIfHalf();
    nn::LSTM lstm(8, 16, 1);
    convert_model(lstm);

    auto input = createInput({3, 2, 8}, true);  // seq=3, batch=2, input=8
    auto output = lstm.forward(input);

    auto out_shape = output.shape();
    std::vector<int64_t> out_shape_vec(out_shape.begin(), out_shape.end());
    auto grad = tenzor::ones(out_shape_vec, dtype(), device());

    EXPECT_NO_THROW({ output.backward(grad); })
        << "LSTM backward threw on " << device().to_string();

    ASSERT_TRUE(input.grad().has_value())
        << "LSTM backward did not produce input gradient on " << device().to_string();
    expectShape(*input.grad(), {3, 2, 8});
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(RNNCellsDispatchMultiDTypeTest);
