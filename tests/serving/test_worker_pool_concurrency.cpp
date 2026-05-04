/**
 * @file test_worker_pool_concurrency.cpp
 * @brief Direct tests for serving::WorkerPool (audit-2026-05-03 N4).
 *
 * Covers:
 *   - basic submit/shutdown lifecycle (every submitted task runs)
 *   - shutdown() drains in-flight work
 *   - multi-worker dispatch parallelizes
 *
 * The implementation lives at `src/serving/worker_pool.cpp` (added by
 * audit-2026-05-03 to fill the previously declared-but-unimplemented gap
 * in the serving subsystem).
 */

#include <gtest/gtest.h>
#include <tenzor/serving/worker_pool.hpp>
#include <atomic>
#include <chrono>
#include <thread>

using namespace tenzor::serving;

TEST(WorkerPoolConcurrency, ExecutesSubmittedTasks) {
    WorkerPool pool(/*num_workers=*/2, tenzor::Device::cpu());
    std::atomic<int> counter{0};
    constexpr int N = 32;
    for (int i = 0; i < N; ++i) {
        pool.submit([&counter] { ++counter; });
    }
    pool.shutdown();
    EXPECT_EQ(counter.load(), N);
}

TEST(WorkerPoolConcurrency, ShutdownDrainsInflight) {
    WorkerPool pool(/*num_workers=*/2, tenzor::Device::cpu());
    std::atomic<int> done{0};
    constexpr int N = 8;
    for (int i = 0; i < N; ++i) {
        pool.submit([&done] {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            ++done;
        });
    }
    pool.shutdown();
    EXPECT_EQ(done.load(), N);
}

TEST(WorkerPoolConcurrency, MultiWorkerParallelism) {
    // 4 workers running 4 sleeps of 100ms each should finish in ~100ms,
    // not ~400ms (the latter would mean serial dispatch).
    WorkerPool pool(/*num_workers=*/4, tenzor::Device::cpu());
    std::atomic<int> done{0};
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 4; ++i) {
        pool.submit([&done] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            ++done;
        });
    }
    pool.shutdown();
    auto elapsed = std::chrono::steady_clock::now() - t0;
    EXPECT_EQ(done.load(), 4);
    // Allow generous slack for slow CI; serial would be ~400ms, parallel <250ms.
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(),
              350);
}

TEST(WorkerPoolConcurrency, DestructorImpliesShutdown) {
    std::atomic<int> done{0};
    {
        WorkerPool pool(/*num_workers=*/2, tenzor::Device::cpu());
        for (int i = 0; i < 4; ++i) {
            pool.submit([&done] {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                ++done;
            });
        }
        // Destructor runs without explicit shutdown call — should drain.
    }
    EXPECT_EQ(done.load(), 4);
}
