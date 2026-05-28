#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <chrono>
#include <thread>

using namespace tenzor;

class EventAPITest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }

    static bool isBackendAvailable(const std::string& name) {
        try {
            auto& loader = backend_registry();
            auto* backend = loader.get_backend(name);
            return backend && backend->is_available();
        } catch (...) {
            return false;
        }
    }
};

// CPU events are real wall-clock timestamps (Stream 17). Same-event elapsed
// must be 0.0f, distinct events with a sleep between them must report a
// positive elapsed millisecond delta.
TEST_F(EventAPITest, CPUEventTiming) {
    auto& loader = backend_registry();
    auto* cpu = loader.get_backend("cpu");
    ASSERT_NE(cpu, nullptr);

    auto start = cpu->create_event(0, true);
    auto end = cpu->create_event(0, true);
    ASSERT_NE(start, nullptr);
    ASSERT_NE(end, nullptr);

    auto stream = cpu->create_stream(0);
    cpu->record_event(start, stream);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    cpu->record_event(end, stream);
    cpu->wait_event(end, stream);

    float elapsed = cpu->event_elapsed_ms(start, end);
    EXPECT_GE(elapsed, 0.0f);
    EXPECT_GT(elapsed, 1.0f);   // Sleep of 5 ms — give generous lower bound.
    EXPECT_LT(elapsed, 5000.0f); // Sanity ceiling.

    // Same event used as both endpoints must produce exactly 0.
    EXPECT_FLOAT_EQ(cpu->event_elapsed_ms(start, start), 0.0f);

    cpu->destroy_event(start);
    cpu->destroy_event(end);
    cpu->destroy_stream(stream);
}

// Test CUDA events if available
TEST_F(EventAPITest, CUDAEventCreateDestroy) {
    if (!isBackendAvailable("cuda")) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto& loader = backend_registry();
    auto* cuda = loader.get_backend("cuda");

    auto event = cuda->create_event(0, true);
    EXPECT_NE(event, nullptr);

    cuda->destroy_event(event);
}

TEST_F(EventAPITest, CUDAEventRecordAndWait) {
    if (!isBackendAvailable("cuda")) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto& loader = backend_registry();
    auto* cuda = loader.get_backend("cuda");

    auto stream = cuda->create_stream(0);
    auto event = cuda->create_event(0, true);

    // Record event on stream
    cuda->record_event(event, stream);

    // Wait on event (should complete immediately since stream is empty)
    cuda->wait_event(event, nullptr);

    cuda->destroy_event(event);
    cuda->destroy_stream(stream);
}

TEST_F(EventAPITest, CUDAEventElapsedTime) {
    if (!isBackendAvailable("cuda")) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto& loader = backend_registry();
    auto* cuda = loader.get_backend("cuda");

    auto stream = cuda->create_stream(0);
    auto start = cuda->create_event(0, true);
    auto end = cuda->create_event(0, true);

    // Record start, do some work, record end
    cuda->record_event(start, stream);

    // Run a matmul to generate measurable GPU work
    auto a = randn({512, 512}, DType::Float32, Device::cuda(0));
    auto b = randn({512, 512}, DType::Float32, Device::cuda(0));
    auto c = matmul(a, b);

    cuda->record_event(end, stream);

    float elapsed = cuda->event_elapsed_ms(start, end);
    // Should be non-negative (and very small for this workload)
    EXPECT_GE(elapsed, 0.0f);

    cuda->destroy_event(start);
    cuda->destroy_event(end);
    cuda->destroy_stream(stream);
}

// Test ROCm events if available
TEST_F(EventAPITest, ROCmEventCreateDestroy) {
    if (!isBackendAvailable("rocm")) {
        GTEST_SKIP() << "ROCm not available";
    }

    auto& loader = backend_registry();
    auto* rocm = loader.get_backend("rocm");

    auto event = rocm->create_event(0, true);
    EXPECT_NE(event, nullptr);

    rocm->destroy_event(event);
}

TEST_F(EventAPITest, ROCmEventRecordAndElapsed) {
    if (!isBackendAvailable("rocm")) {
        GTEST_SKIP() << "ROCm not available";
    }

    auto& loader = backend_registry();
    auto* rocm = loader.get_backend("rocm");

    auto stream = rocm->create_stream(0);
    auto start = rocm->create_event(0, true);
    auto end = rocm->create_event(0, true);

    rocm->record_event(start, stream);

    // Do some work
    auto a = randn({512, 512}, DType::Float32, Device::rocm(0));
    auto b = randn({512, 512}, DType::Float32, Device::rocm(0));
    auto c = matmul(a, b);

    rocm->record_event(end, stream);

    float elapsed = rocm->event_elapsed_ms(start, end);
    EXPECT_GE(elapsed, 0.0f);

    rocm->destroy_event(start);
    rocm->destroy_event(end);
    rocm->destroy_stream(stream);
}

// Test event with timing disabled
TEST_F(EventAPITest, CUDAEventNoTiming) {
    if (!isBackendAvailable("cuda")) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto& loader = backend_registry();
    auto* cuda = loader.get_backend("cuda");

    auto event = cuda->create_event(0, false);  // timing disabled
    EXPECT_NE(event, nullptr);

    auto stream = cuda->create_stream(0);
    cuda->record_event(event, stream);
    cuda->wait_event(event, stream);

    cuda->destroy_event(event);
    cuda->destroy_stream(stream);
}
