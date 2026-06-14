#include <gtest/gtest.h>

#include <tenzor/ops/op_id.hpp>

#include <string>

namespace {

struct PinnedOpName {
    tenzor::OpId op;
    std::string_view name;
};

TEST(OpIdNamesTest, RecentlyAddedPinnedOpsRoundTrip) {
    constexpr PinnedOpName pinned_ops[] = {
        {tenzor::OpId::ConvTranspose1dForward, "conv_transpose1d_forward"},
        {tenzor::OpId::FlexAttention, "flex_attention"},
        {tenzor::OpId::FlexAttentionBackward, "flex_attention_backward"},
        {tenzor::OpId::SelectScatter, "select_scatter"},
        {tenzor::OpId::SliceScatter, "slice_scatter"},
        {tenzor::OpId::DiagonalScatter, "diagonal_scatter"},
        {tenzor::OpId::LSTMCudnnTrainForward, "lstm_cudnn_train_forward"},
        {tenzor::OpId::LSTMCudnnBackward, "lstm_cudnn_backward"},
        {tenzor::OpId::GRUCudnnTrainForward, "gru_cudnn_train_forward"},
        {tenzor::OpId::GRUCudnnBackward, "gru_cudnn_backward"},
        {tenzor::OpId::SparseSoftmax, "sparse_softmax"},
        {tenzor::OpId::SparseLogSoftmax, "sparse_log_softmax"},
        {tenzor::OpId::GammaSample, "gamma_sample"},
        {tenzor::OpId::MaxUnpool1dForward, "max_unpool1d_forward"},
        {tenzor::OpId::MaxUnpool1dBackward, "max_unpool1d_backward"},
    };

    for (const auto& pinned : pinned_ops) {
        EXPECT_EQ(tenzor::op_id_to_name(pinned.op), pinned.name);
        EXPECT_EQ(tenzor::string_to_op_id(pinned.name), pinned.op)
            << "reverse lookup failed for " << pinned.name;
    }
}

}  // namespace
