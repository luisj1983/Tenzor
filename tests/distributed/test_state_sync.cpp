/**
 * @file test_state_sync.cpp
 * @brief Round-trip checkpoint test for distributed::elastic::StateSync.
 *
 * Covers Phase 1.7 of the vectorized-rabin audit plan: the previous
 * implementation of save_checkpoint/load_checkpoint logged to stderr without
 * ever touching disk. This test exercises the real nn::Serializer path now
 * wired into StateSync — save a module's parameters, mutate them, then
 * reload and verify the original values are restored.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/elastic/state_sync.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/ops/creation.hpp>

#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tenzor;
using namespace tenzor::distributed::elastic;

namespace {

class StateSyncEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};

::testing::Environment* const g_state_sync_env =
    ::testing::AddGlobalTestEnvironment(new StateSyncEnv);

class StateSyncCheckpointTest : public ::testing::Test {
protected:
    void SetUp() override {
        checkpoint_dir_ =
            (std::filesystem::temp_directory_path() /
             ("tenzor_state_sync_test_" +
              std::to_string(::testing::UnitTest::GetInstance()->random_seed())))
                .string();
        std::filesystem::remove_all(checkpoint_dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(checkpoint_dir_);
    }

    std::string checkpoint_dir_;
};

// Round-trip: save → mutate → load → original values restored.
TEST_F(StateSyncCheckpointTest, SaveLoadRestoresParameters) {
    constexpr int rank = 0;

    nn::Linear layer(/*in_features=*/4, /*out_features=*/3, /*bias=*/true);

    // Snapshot the initial state for later comparison.
    auto initial_state = layer.state_dict();
    ASSERT_FALSE(initial_state.empty())
        << "Linear module produced empty state_dict; cannot exercise round-trip";

    StateSync::save_checkpoint(layer, checkpoint_dir_, rank);

    // Verify the file actually exists on disk now (the old stub never wrote).
    const auto expected_path = std::filesystem::path(checkpoint_dir_) /
                               ("rank_" + std::to_string(rank) + ".pt");
    ASSERT_TRUE(std::filesystem::exists(expected_path))
        << "save_checkpoint did not write " << expected_path;
    EXPECT_GT(std::filesystem::file_size(expected_path), 0u);

    // Mutate parameters in-place so a no-op load would be detectable.
    for (auto& param : layer.parameters()) {
        auto t = param->tensor();
        if (t.dtype() == DType::Float32) {
            auto* data = t.data<float>();
            for (int64_t i = 0; i < t.numel(); ++i) {
                data[i] = 12345.0f;
            }
        }
    }

    StateSync::load_checkpoint(layer, checkpoint_dir_, rank);

    auto restored_state = layer.state_dict();
    ASSERT_EQ(restored_state.size(), initial_state.size());

    for (const auto& [name, orig] : initial_state) {
        ASSERT_TRUE(restored_state.count(name))
            << "Missing key after reload: " << name;
        const auto& reloaded = restored_state.at(name);
        ASSERT_EQ(reloaded.dtype(), orig.dtype()) << "dtype mismatch for " << name;
        const auto orig_shape = orig.shape();
        const auto reloaded_shape = reloaded.shape();
        const std::vector<int64_t> a_shape(orig_shape.begin(), orig_shape.end());
        const std::vector<int64_t> b_shape(reloaded_shape.begin(), reloaded_shape.end());
        ASSERT_EQ(a_shape, b_shape) << "shape mismatch for " << name;
        ASSERT_EQ(reloaded.numel(), orig.numel()) << "numel mismatch for " << name;

        if (orig.dtype() == DType::Float32) {
            const auto* a = orig.data<float>();
            const auto* b = reloaded.data<float>();
            for (int64_t i = 0; i < orig.numel(); ++i) {
                EXPECT_FLOAT_EQ(a[i], b[i])
                    << "Float32 mismatch at " << name << "[" << i << "]";
            }
        }
    }
}

// Rank fallback: if rank_N.pt is missing, StateSync must transparently load
// rank_0.pt (this is how elastic resize recovers ranks that didn't exist in
// the prior generation).
TEST_F(StateSyncCheckpointTest, FallsBackToRankZeroWhenRankMissing) {
    nn::Linear layer(2, 2, /*bias=*/false);
    StateSync::save_checkpoint(layer, checkpoint_dir_, /*rank=*/0);

    // Rank 7 has no checkpoint — must fall back to rank 0 and succeed.
    nn::Linear other(2, 2, /*bias=*/false);
    EXPECT_NO_THROW(StateSync::load_checkpoint(other, checkpoint_dir_, /*rank=*/7));
}

// Missing directory: must throw a specific exception, not silently no-op.
TEST_F(StateSyncCheckpointTest, MissingCheckpointThrows) {
    nn::Linear layer(2, 2, /*bias=*/false);
    const auto bogus_dir = checkpoint_dir_ + "_does_not_exist";
    EXPECT_THROW(
        StateSync::load_checkpoint(layer, bogus_dir, /*rank=*/0),
        std::runtime_error);
}

} // namespace
