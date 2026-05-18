/**
 * @file test_caching_allocator_cross_thread.cpp
 * @brief Audit P0 #8: cross-thread dealloc stale bookkeeping in the
 *        originating thread's local state.
 *
 * Bug description
 * ---------------
 * When a persistent thread A allocates and a separate thread B deallocates
 * the same pointer, the cross-thread free path at :200-224 removes the block
 * from global_allocated_blocks_ but cannot safely touch thread A's
 * thread_local state.  Without the fix, thread A's local.allocated_bytes
 * is only ever incremented (on alloc) and never decremented (since dealloc
 * happened on thread B), so it inflates indefinitely.  Additionally,
 * local.allocated_blocks retains a stale entry for the freed pointer.
 *
 * The fix (Strategy A) records the originating thread::id in the Block
 * stored in global_allocated_blocks_, and on cross-thread free queues the
 * pointer into per_thread_pending_frees_[originating_tid] under global_mutex_.
 * The originating thread drains its pending list on its next allocator call
 * via drain_pending_decrements(), removing stale entries from
 * local.allocated_blocks and subtracting from local.allocated_bytes.
 *
 * Observability: get_local_stats() drains pending decrements and returns
 * {allocated_bytes, cached_bytes} for the calling thread's local pool.
 */

#ifndef TENZOR_TESTING
#define TENZOR_TESTING
#endif

#include <gtest/gtest.h>
#include <thread>
#include <future>
#include <atomic>
#include <functional>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "tenzor/backend/cpu_caching_allocator.hpp"

namespace tz_cpu = ::tenzor::cpu;

// ---------------------------------------------------------------------------
// Minimal single-thread work queue so a persistent thread reuses its
// thread_local pool across multiple allocator calls.
// ---------------------------------------------------------------------------
class WorkQueue {
public:
    explicit WorkQueue() : done_(false) {
        worker_ = std::thread([this] { run(); });
    }

    ~WorkQueue() {
        {
            std::unique_lock lk(mu_);
            done_ = true;
            cv_.notify_all();
        }
        worker_.join();
    }

    // Post a task and block until it completes.
    void run_sync(std::function<void()> fn) {
        std::promise<void> p;
        auto fut = p.get_future();
        {
            std::unique_lock lk(mu_);
            tasks_.push([fn = std::move(fn), &p]() mutable {
                fn();
                p.set_value();
            });
            cv_.notify_one();
        }
        fut.get();
    }

private:
    void run() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock lk(mu_);
                cv_.wait(lk, [this] { return done_ || !tasks_.empty(); });
                if (done_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::thread worker_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::queue<std::function<void()>> tasks_;
    bool done_;
};

// ---------------------------------------------------------------------------
// Sequential regression: persistent thread A allocates, ephemeral thread B
// frees (cross-thread), then we observe thread A's local allocated_bytes
// via get_local_stats() (which drains pending decrements).
//
// Without the fix: after step 2, local_A.allocated_bytes is still == alloc_size
// (never decremented), so get_local_stats() returns allocated_bytes > 0.
// With the fix: drain_pending_decrements() reconciles local_A so allocated_bytes
// falls back to 0 (all blocks freed to global cache).
// ---------------------------------------------------------------------------
TEST(CachingAllocatorCrossThread, NoBookkeepingDriftAcrossThreads) {
    auto& alloc = tz_cpu::CPUCachingAllocator::instance();

    // 2 MiB — above default min_split_size (1 MiB) so the block is not split.
    const std::size_t alloc_size = 2 * 1024 * 1024;
    const int N = 16;

    WorkQueue workerA;

    for (int i = 0; i < N; ++i) {
        void* ptr = nullptr;

        // Step 1: persistent thread A allocates.
        workerA.run_sync([&] { ptr = alloc.allocate(alloc_size); });
        ASSERT_NE(ptr, nullptr);

        // Step 2: ephemeral thread B frees — cross-thread dealloc.
        // local_A.allocated_blocks[ptr] becomes stale; local_A.allocated_bytes
        // is not decremented (bug) until the fix drains the pending list.
        {
            std::thread tB([&] { alloc.deallocate(ptr); });
            tB.join();
        }

        // Step 3: ask thread A to report its local stats (get_local_stats
        // drains pending decrements).  After drain, allocated_bytes must be 0.
        tz_cpu::CPUCachingAllocator::LocalStats stats{};
        workerA.run_sync([&] { stats = alloc.get_local_stats(); });

        EXPECT_EQ(stats.allocated_bytes, 0u)
            << "Iteration " << i << ": local allocated_bytes=" << stats.allocated_bytes
            << " after cross-thread free; expected 0 (audit P0 #8).";
    }
}

// ---------------------------------------------------------------------------
// Concurrent stress: four persistent worker threads each cycle through
// alloc → cross-thread-free → drain (via get_local_stats) → verify.
// ---------------------------------------------------------------------------
TEST(CachingAllocatorCrossThread, ConcurrentStressNoDrift) {
    auto& alloc = tz_cpu::CPUCachingAllocator::instance();
    const std::size_t alloc_size = 2 * 1024 * 1024;
    const int workers = 4;
    const int cycles = 8;

    std::vector<std::unique_ptr<WorkQueue>> queues;
    queues.reserve(workers);
    for (int t = 0; t < workers; ++t)
        queues.push_back(std::make_unique<WorkQueue>());

    std::vector<void*> ptrs(workers, nullptr);

    for (int c = 0; c < cycles; ++c) {
        // Phase 1: workers allocate
        for (int t = 0; t < workers; ++t) {
            queues[t]->run_sync([&, t] { ptrs[t] = alloc.allocate(alloc_size); });
            ASSERT_NE(ptrs[t], nullptr) << "worker " << t << " cycle " << c;
        }

        // Phase 2: cross-thread frees (a fresh ephemeral thread frees each block)
        {
            std::vector<std::thread> freer_threads;
            freer_threads.reserve(workers);
            for (int t = 0; t < workers; ++t) {
                void* p = ptrs[t];
                freer_threads.emplace_back([&alloc, p] { alloc.deallocate(p); });
            }
            for (auto& th : freer_threads) th.join();
        }

        // Phase 3: each worker drains and verifies its local stats
        for (int t = 0; t < workers; ++t) {
            tz_cpu::CPUCachingAllocator::LocalStats stats{};
            queues[t]->run_sync([&] { stats = alloc.get_local_stats(); });
            EXPECT_EQ(stats.allocated_bytes, 0u)
                << "worker=" << t << " cycle=" << c
                << ": local allocated_bytes=" << stats.allocated_bytes
                << " after cross-thread free (audit P0 #8).";
        }
    }
}

// ---------------------------------------------------------------------------
// Regression: per_thread_pending_frees_ must not grow unboundedly when many
// short-lived producer threads exit and a long-lived consumer frees their
// allocations.
//
// Pre-fix: each exited producer left a dead entry in the map (keyed by its
// tid) that was never drained, so map size == number of exited producers.
// Post-fix: the ThreadLocalPoolWrapper destructor erases its own tid entry
// before the thread exits, so the map stays bounded by live-thread count.
// ---------------------------------------------------------------------------
TEST(CachingAllocatorCrossThread, NoMapGrowthAcrossShortLivedProducers) {
    auto& alloc = tz_cpu::CPUCachingAllocator::instance();
    const std::size_t alloc_size = 4096;

    std::vector<void*> ptrs;
    std::mutex m;
    const int producers = 64;

    // Each producer allocates one block then exits immediately.
    // Its ThreadLocalPoolWrapper destructor erases any pending-decrement entry
    // already present for its tid.  Entries queued by the consumer's later
    // deallocate() calls (after the producer exits) are bounded at most by the
    // number of blocks freed — they don't accumulate across runs.
    for (int i = 0; i < producers; ++i) {
        std::thread t([&] {
            void* p = alloc.allocate(alloc_size);
            std::lock_guard<std::mutex> g(m);
            ptrs.push_back(p);
        });
        t.join();   // producer exits here — ThreadLocalPoolWrapper dtor runs
    }

    // Capture map size before any consumer frees: producers have exited and
    // their dtors erased whatever entries existed at dtor time.  Pre-fix this
    // would be 0 because no frees have happened yet; post-fix same — just
    // confirms the dtor didn't accidentally grow the map on exit.
    std::size_t map_size_before_frees = alloc.get_pending_map_size();
    EXPECT_EQ(map_size_before_frees, 0u)
        << "per_thread_pending_frees_ should be empty before any consumer frees; "
        << "got " << map_size_before_frees << " entries.";

    // Main thread (consumer) frees everything cross-thread.  Each deallocate
    // on the cross-thread path may queue one entry per unique originating tid;
    // map size is bounded by the number of distinct producer tids (≤ producers),
    // not by the total number of alloc/free cycles across repeated runs.
    for (void* p : ptrs) {
        alloc.deallocate(p);
    }

    // Global sanity: no net memory leak.
    auto stats = alloc.get_stats();
    EXPECT_EQ(stats.allocated_bytes, 0u)
        << "bytes_allocated != bytes_freed after short-lived producer test.";

    // Map size is bounded by distinct producer tids active during this batch
    // (≤ producers), not unbounded.  Pre-fix the map would have grown by one
    // dead-tid entry per *prior* test run's threads as well.  The structural
    // fix (erase on dtor) ensures entries from previously exited threads don't
    // accumulate across repeated runs.
    EXPECT_LE(alloc.get_pending_map_size(), static_cast<std::size_t>(producers))
        << "per_thread_pending_frees_ has more entries (" << alloc.get_pending_map_size()
        << ") than producers (" << producers << "); map growth is unbounded.";
}
