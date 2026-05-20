/**
 * @file test_training_callbacks_lifecycle.cpp
 * @brief Audit G.9: NeuralNetwork::fit fires the full Callback lifecycle.
 *
 * Previously fit() only fired on_epoch_begin/on_epoch_end/on_batch_end.
 * on_train_begin, on_train_end, and on_batch_begin were declared on the
 * Callback base class but never invoked from the training loop, so any
 * tooling (logging, profiling, scheduler reset) that hooked into those
 * lifecycle points silently never ran.
 *
 * This regression test wires a recording Callback into a 1-epoch fit()
 * call and verifies every lifecycle hook fires exactly the expected
 * number of times.
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/training.hpp"
#include "tenzor/nn/callbacks.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/data/dataloader.hpp"
#include "tenzor/ops/reduction.hpp"

using namespace tenzor;
using namespace tenzor::nn;

namespace {

class LifecycleCallback : public Callback {
public:
    int train_begin = 0;
    int train_end = 0;
    int epoch_begin = 0;
    int epoch_end = 0;
    int batch_begin = 0;
    int batch_end = 0;

    auto on_train_begin() -> void override { ++train_begin; }
    auto on_train_end() -> void override { ++train_end; }
    auto on_epoch_begin(int) -> void override { ++epoch_begin; }
    auto on_epoch_end(int, float, float) -> void override { ++epoch_end; }
    auto on_batch_begin(int) -> void override { ++batch_begin; }
    auto on_batch_end(int, float) -> void override { ++batch_end; }
};

class TrainingCallbacksLifecycleTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

TEST_F(TrainingCallbacksLifecycleTest, AllSixHooksFireInFit) {
    // Tiny in-memory dataset.  4 samples, batch size 2 → 2 batches per
    // epoch.  We use 2 epochs to cross-check on_epoch_begin/end fire
    // per epoch and on_train_begin/end fire only once each.
    constexpr int kNumSamples = 4;
    constexpr int kBatchSize = 2;
    constexpr int kEpochs = 2;
    constexpr int kBatchesPerEpoch = kNumSamples / kBatchSize;
    constexpr int kInDim = 3;
    constexpr int kOutDim = 2;

    std::vector<std::pair<Tensor, Tensor>> data;
    for (int i = 0; i < kNumSamples; ++i) {
        data.emplace_back(randn({kInDim}, DType::Float32, Device::cpu()),
                          randn({kOutDim}, DType::Float32, Device::cpu()));
    }
    DataLoader loader(data, kBatchSize);

    auto model = std::make_shared<Linear>(kInDim, kOutDim);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) -> Variable {
        auto diff = pred - target;
        auto squared = diff * diff;
        return mean(squared);
    };

    NeuralNetwork nn(model, optimizer, loss_fn);
    auto cb = std::make_shared<LifecycleCallback>();

    nn.fit(loader, kEpochs, /*val_loader=*/nullptr, {cb});

    EXPECT_EQ(cb->train_begin, 1)  << "on_train_begin must fire exactly once";
    EXPECT_EQ(cb->train_end,   1)  << "on_train_end must fire exactly once";
    EXPECT_EQ(cb->epoch_begin, kEpochs)
        << "on_epoch_begin must fire once per epoch";
    EXPECT_EQ(cb->epoch_end,   kEpochs)
        << "on_epoch_end must fire once per epoch";
    EXPECT_EQ(cb->batch_begin, kEpochs * kBatchesPerEpoch)
        << "on_batch_begin must fire once per training batch";
    EXPECT_EQ(cb->batch_end,   kEpochs * kBatchesPerEpoch)
        << "on_batch_end must fire once per training batch";
}

}  // namespace
