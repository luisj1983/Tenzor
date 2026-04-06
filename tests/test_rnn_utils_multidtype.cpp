/**
 * @file test_rnn_utils_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for RNN utility functions
 *        (pack_padded_sequence, pad_packed_sequence)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/utils/rnn_utils.hpp>
#include "multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class RNNUtilsMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    static constexpr int64_t batch = 3;
    static constexpr int64_t max_seq_len = 5;
    static constexpr int64_t features = 8;

    /**
     * @brief Create a lengths tensor with values [5, 3, 2] on CPU (Int64).
     *
     * Lengths are always Int64 on CPU regardless of test dtype/device.
     */
    Tensor createLengths() {
        auto t = Tensor({3}, DType::Int64, Device::cpu());
        auto* d = t.data<int64_t>();
        d[0] = 5; d[1] = 3; d[2] = 2;
        return t;
    }

    /**
     * @brief Create a padded input tensor with test dtype on test device.
     *
     * Shape depends on batch_first:
     *   batch_first=true  -> (batch, max_seq_len, features)
     *   batch_first=false -> (max_seq_len, batch, features)
     */
    Tensor createPaddedInput(bool batch_first) {
        if (batch_first) {
            return createRandn({batch, max_seq_len, features});
        } else {
            return createRandn({max_seq_len, batch, features});
        }
    }
};

TEST_P(RNNUtilsMultiDTypeTest, PackRoundtrip) {
    auto input = createPaddedInput(/*batch_first=*/true);
    auto lengths = createLengths();

    auto packed = nn::pack_padded_sequence(input, lengths,
                                           /*batch_first=*/true,
                                           /*enforce_sorted=*/true);

    auto [recovered, recovered_lengths] = nn::pad_packed_sequence(
        packed, /*batch_first=*/true);

    // Recovered tensor should have the same shape as the original
    expectShape(recovered, {batch, max_seq_len, features});

    // The valid (non-padding) elements should match the original input.
    // Check the first sequence (length 5 -- full row).
    auto orig_cpu = input.to(DType::Float32).to(Device::cpu());
    auto rec_cpu = recovered.to(DType::Float32).to(Device::cpu());

    auto* orig_data = orig_cpu.data<float>();
    auto* rec_data = rec_cpu.data<float>();

    // First sequence spans all 5 timesteps
    for (int64_t i = 0; i < max_seq_len * features; ++i) {
        EXPECT_NEAR(orig_data[i], rec_data[i], atol())
            << "Mismatch at index " << i << " in first sequence";
    }
}

TEST_P(RNNUtilsMultiDTypeTest, VariableLengths) {
    auto input = createPaddedInput(/*batch_first=*/true);
    auto lengths = createLengths();

    auto packed = nn::pack_padded_sequence(input, lengths,
                                           /*batch_first=*/true,
                                           /*enforce_sorted=*/true);

    auto [recovered, recovered_lengths] = nn::pad_packed_sequence(
        packed, /*batch_first=*/true);

    // Recovered lengths should match the original lengths
    auto rl_cpu = recovered_lengths.to(Device::cpu());
    auto* rl_data = rl_cpu.data<int64_t>();
    EXPECT_EQ(rl_data[0], 5);
    EXPECT_EQ(rl_data[1], 3);
    EXPECT_EQ(rl_data[2], 2);
}

TEST_P(RNNUtilsMultiDTypeTest, BatchFirstModes) {
    auto lengths = createLengths();

    // Test batch_first=true
    {
        auto input_bf = createPaddedInput(/*batch_first=*/true);
        auto packed_bf = nn::pack_padded_sequence(input_bf, lengths,
                                                   /*batch_first=*/true,
                                                   /*enforce_sorted=*/true);
        auto [out_bf, _] = nn::pad_packed_sequence(packed_bf,
                                                    /*batch_first=*/true);
        expectShape(out_bf, {batch, max_seq_len, features});
    }

    // Test batch_first=false
    {
        auto input_sf = createPaddedInput(/*batch_first=*/false);
        auto packed_sf = nn::pack_padded_sequence(input_sf, lengths,
                                                   /*batch_first=*/false,
                                                   /*enforce_sorted=*/true);
        auto [out_sf, _] = nn::pad_packed_sequence(packed_sf,
                                                    /*batch_first=*/false);
        expectShape(out_sf, {max_seq_len, batch, features});
    }
}

TEST_P(RNNUtilsMultiDTypeTest, PackedDataShape) {
    auto input = createPaddedInput(/*batch_first=*/true);
    auto lengths = createLengths();

    auto packed = nn::pack_padded_sequence(input, lengths,
                                           /*batch_first=*/true,
                                           /*enforce_sorted=*/true);

    // Total elements = sum of lengths = 5 + 3 + 2 = 10
    // packed.data shape should be (10, features)
    expectShape(packed.data, {10, features});
    expectDevice(packed.data);
    expectDType(packed.data);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(RNNUtilsMultiDTypeTest);
