/**
 * @file test_metrics.cpp
 * @brief Tests for MetricsRegistry and ModelMetrics
 */

#include <gtest/gtest.h>
#include <tenzor/serving/server.hpp>

using namespace tenzor::serving;

TEST(MetricsTest, RegistrySingleton) {
    auto& r1 = MetricsRegistry::instance();
    auto& r2 = MetricsRegistry::instance();
    EXPECT_EQ(&r1, &r2);
}

TEST(MetricsTest, ModelMetricsDefaults) {
    auto& registry = MetricsRegistry::instance();
    auto& m = registry.get_metrics("test_metrics_defaults");
    EXPECT_EQ(m.total_requests.load(), 0u);
    EXPECT_EQ(m.total_latency_us.load(), 0u);
    EXPECT_EQ(m.total_batch_count.load(), 0u);
    EXPECT_EQ(m.total_batch_size.load(), 0u);
    EXPECT_EQ(m.error_count.load(), 0u);
}

TEST(MetricsTest, RecordLatency) {
    auto& registry = MetricsRegistry::instance();
    auto& m = registry.get_metrics("test_record_latency");

    m.record_latency(500);
    m.record_latency(1000);
    m.record_latency(1500);

    // Verify the ring buffer was written
    EXPECT_EQ(m.latency_window[0], 500u);
    EXPECT_EQ(m.latency_window[1], 1000u);
    EXPECT_EQ(m.latency_window[2], 1500u);
    EXPECT_EQ(m.latency_idx.load(), 3u);
}

TEST(MetricsTest, LatencyWindowWraps) {
    auto& registry = MetricsRegistry::instance();
    auto& m = registry.get_metrics("test_latency_wrap");

    // Fill past the window size
    for (size_t i = 0; i < ModelMetrics::kLatencyWindowSize + 5; ++i) {
        m.record_latency(i);
    }

    // Index should have advanced past window size
    EXPECT_EQ(m.latency_idx.load(), ModelMetrics::kLatencyWindowSize + 5);

    // Slot 0 should have been overwritten by iteration kLatencyWindowSize
    EXPECT_EQ(m.latency_window[0], ModelMetrics::kLatencyWindowSize);
}

TEST(MetricsTest, MultipleModelsIndependent) {
    auto& registry = MetricsRegistry::instance();
    auto& m1 = registry.get_metrics("model_independence_a");
    auto& m2 = registry.get_metrics("model_independence_b");

    m1.total_requests.store(42);
    m2.total_requests.store(99);

    EXPECT_EQ(m1.total_requests.load(), 42u);
    EXPECT_EQ(m2.total_requests.load(), 99u);
    EXPECT_NE(&m1, &m2);
}

TEST(MetricsTest, PrometheusFormatContainsModel) {
    auto& registry = MetricsRegistry::instance();
    auto& m = registry.get_metrics("test_prom_format");
    m.total_requests.store(50);

    auto text = registry.format_prometheus();
    EXPECT_NE(text.find("test_prom_format"), std::string::npos);
    EXPECT_NE(text.find("tenzor_requests_total"), std::string::npos);
}

TEST(MetricsTest, LatencyWindowSizeConstant) {
    EXPECT_EQ(ModelMetrics::kLatencyWindowSize, 1000u);
}
