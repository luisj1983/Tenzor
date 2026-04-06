/**
 * @file test_kv_cache_multidtype.cpp
 * @brief Multi-backend and multi-dtype tests for KVCache
 *
 * Tests the KVCache utility across all backends and floating-point dtypes
 * to ensure correct behavior for autoregressive inference caching.
 */

#include <gtest/gtest.h>
#include "tenzor/nn/utils/kv_cache.hpp"
#include "tenzor/tenzor.hpp"
#include "multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class KVCacheMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Cache configuration shared across tests
    static constexpr int64_t kNumLayers = 2;
    static constexpr int64_t kMaxSeqLen = 16;
    static constexpr int64_t kNumKvHeads = 4;
    static constexpr int64_t kHeadDim = 8;
    static constexpr int64_t kBatchSize = 2;

    nn::utils::KVCacheConfig makeConfig() {
        return nn::utils::KVCacheConfig{
            .num_layers = kNumLayers,
            .max_seq_len = kMaxSeqLen,
            .num_kv_heads = kNumKvHeads,
            .head_dim = kHeadDim,
            .batch_size = kBatchSize,
            .dtype = dtype(),
            .device = device(),
        };
    }

    /// Create a key or value tensor with shape (batch, num_kv_heads, seq_len, head_dim)
    Tensor makeKV(int64_t seq_len) {
        return createRandn({kBatchSize, kNumKvHeads, seq_len, kHeadDim});
    }
};

TEST_P(KVCacheMultiDTypeTest, UpdateAndRetrieve) {
    auto cache = nn::utils::KVCache(makeConfig());

    auto new_k = makeKV(1);
    auto new_v = makeKV(1);

    auto [cached_k, cached_v] = cache.update(0, new_k, new_v, 0);

    // After inserting 1 token at position 0, the returned view should cover [0, 1)
    expectShape(cached_k, {kBatchSize, kNumKvHeads, 1, kHeadDim});
    expectShape(cached_v, {kBatchSize, kNumKvHeads, 1, kHeadDim});
    expectDType(cached_k);
    expectDType(cached_v);
}

TEST_P(KVCacheMultiDTypeTest, SequentialPositions) {
    auto cache = nn::utils::KVCache(makeConfig());

    // Insert tokens at positions 0, 1, 2 sequentially
    auto k0 = makeKV(1);
    auto v0 = makeKV(1);
    cache.update(0, k0, v0, 0);

    auto k1 = makeKV(1);
    auto v1 = makeKV(1);
    cache.update(0, k1, v1, 1);

    auto k2 = makeKV(1);
    auto v2 = makeKV(1);
    auto [cached_k, cached_v] = cache.update(0, k2, v2, 2);

    // After 3 sequential inserts, the cache should span [0, 3)
    expectShape(cached_k, {kBatchSize, kNumKvHeads, 3, kHeadDim});
    expectShape(cached_v, {kBatchSize, kNumKvHeads, 3, kHeadDim});
}

TEST_P(KVCacheMultiDTypeTest, MultiLayer) {
    auto cache = nn::utils::KVCache(makeConfig());

    // Insert into layer 0 at position 0
    auto k0 = makeKV(1);
    auto v0 = makeKV(1);
    cache.update(0, k0, v0, 0);

    // Insert into layer 1 at positions 0 and 1
    auto k1a = makeKV(1);
    auto v1a = makeKV(1);
    cache.update(1, k1a, v1a, 0);

    auto k1b = makeKV(1);
    auto v1b = makeKV(1);
    auto [cached_k1, cached_v1] = cache.update(1, k1b, v1b, 1);

    // Layer 0 should have 1 token, layer 1 should have 2
    auto layer0_keys = cache.get_keys(0, 1);
    expectShape(layer0_keys, {kBatchSize, kNumKvHeads, 1, kHeadDim});

    expectShape(cached_k1, {kBatchSize, kNumKvHeads, 2, kHeadDim});
    expectShape(cached_v1, {kBatchSize, kNumKvHeads, 2, kHeadDim});
}

TEST_P(KVCacheMultiDTypeTest, Reset) {
    auto cache = nn::utils::KVCache(makeConfig());

    // Populate layer 0 with 3 tokens
    for (int64_t pos = 0; pos < 3; ++pos) {
        cache.update(0, makeKV(1), makeKV(1), pos);
    }

    // Reset all caches
    cache.reset();

    // After reset, retrieving keys should give zeroed values
    auto keys = cache.get_keys(0, 1);
    expectShape(keys, {kBatchSize, kNumKvHeads, 1, kHeadDim});

    // Verify the cache contents are zeroed
    float max_abs = compute_max_abs(keys);
    EXPECT_NEAR(max_abs, 0.0f, atol())
        << "Cache should be zeroed after reset";
}

TEST_P(KVCacheMultiDTypeTest, ConfigProperties) {
    auto config = makeConfig();
    auto cache = nn::utils::KVCache(config);

    EXPECT_EQ(cache.max_seq_len(), kMaxSeqLen);
    EXPECT_EQ(cache.num_layers(), kNumLayers);
    EXPECT_EQ(cache.config().num_kv_heads, kNumKvHeads);
    EXPECT_EQ(cache.config().head_dim, kHeadDim);
    EXPECT_EQ(cache.config().batch_size, kBatchSize);
    EXPECT_EQ(cache.config().dtype, dtype());
    EXPECT_EQ(cache.config().device.type, device().type);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(KVCacheMultiDTypeTest);
