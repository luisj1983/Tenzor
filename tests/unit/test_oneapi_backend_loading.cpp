/**
 * @file test_oneapi_backend_loading.cpp
 * @brief Test OneAPI backend library loading and basic functionality
 *
 * Tests:
 * 1. Backend library loads without errors
 * 2. Backend can enumerate SYCL devices (may be 0 if no Intel GPU)
 * 3. Backend reports correct availability based on device count
 * 4. Backend can create tensors if devices are available
 */

#include <gtest/gtest.h>
#include "tenzor/core/device.hpp"
#include "tenzor/backend/backend.hpp"
#include <dlfcn.h>
#include <memory>
#include <string>

namespace tenzor {
namespace test {

class OneAPIBackendLoadingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Try to load the OneAPI backend library
        const char* lib_path = "./bin/tenzor_backend_oneapi.so";

        // Try to load with RTLD_NOW to catch all symbol errors
        handle_ = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);

        if (!handle_) {
            // Library load failed - this is expected if not built
            load_error_ = dlerror();
            GTEST_SKIP() << "OneAPI backend not available: " << load_error_;
            return;
        }

        // Try to get the create_backend symbol.
        //
        // IMPORTANT: the exported `extern "C" Backend* create_backend()` returns
        // a RAW `Backend*` (see backend.hpp: `using BackendFactory = Backend*(*)()`
        // and oneapi_backend.cpp). Declaring it here as returning
        // `std::unique_ptr<Backend>` is an ABI mismatch: a function returning a
        // non-trivial unique_ptr uses the hidden-sret calling convention (return
        // slot pointer in RDI) while the real factory returns the pointer in RAX.
        // Calling through the wrong signature constructed `backend_` from garbage
        // → wild pointer → SIGSEGV on `backend_->name()` and `free(): invalid
        // pointer` / stack-smashing when the bogus unique_ptr later destructs.
        // Use the correct raw-pointer signature and adopt it into the unique_ptr.
        using CreateBackendFunc = Backend* (*)();
        auto create_backend = reinterpret_cast<CreateBackendFunc>(
            dlsym(handle_, "create_backend")
        );

        if (!create_backend) {
            dlclose(handle_);
            handle_ = nullptr;
            GTEST_SKIP() << "create_backend symbol not found";
            return;
        }

        // Create the backend instance. Availability was already established
        // deterministically above (the .so loaded and create_backend resolved),
        // so a throw here is a genuine factory/construction bug in the OneAPI
        // backend — let it propagate and fail the test rather than burying it
        // as "not available". TearDown() closes handle_ on the failure path.
        backend_.reset(create_backend());
        backend_created_ = true;
    }

    void TearDown() override {
        backend_.reset();
        if (handle_) {
            dlclose(handle_);
            handle_ = nullptr;
        }
    }

    void* handle_ = nullptr;
    std::unique_ptr<Backend> backend_;
    std::string load_error_;
    bool backend_created_ = false;
};

TEST_F(OneAPIBackendLoadingTest, LibraryLoads) {
    ASSERT_NE(handle_, nullptr) << "Backend library should load: " << load_error_;
    ASSERT_TRUE(backend_created_) << "Backend should be created";
    ASSERT_NE(backend_.get(), nullptr) << "Backend pointer should be valid";
}

TEST_F(OneAPIBackendLoadingTest, BackendName) {
    ASSERT_TRUE(backend_created_);
    EXPECT_EQ(backend_->name(), "oneapi");
}

TEST_F(OneAPIBackendLoadingTest, DeviceEnumeration) {
    ASSERT_TRUE(backend_created_);

    int32_t device_count = backend_->device_count();
    std::cout << "OneAPI backend found " << device_count << " devices" << std::endl;

    // Device count can be 0 if no Intel GPUs are available
    // This is valid - the backend should still load successfully
    EXPECT_GE(device_count, 0) << "Device count should be non-negative";
}

TEST_F(OneAPIBackendLoadingTest, AvailabilityMatchesDeviceCount) {
    ASSERT_TRUE(backend_created_);

    int32_t device_count = backend_->device_count();
    bool is_available = backend_->is_available();

    if (device_count == 0) {
        EXPECT_FALSE(is_available)
            << "Backend should not be available when no devices found";
        std::cout << "NOTE: No OneAPI devices available. This is expected on systems without Intel GPUs." << std::endl;
    } else {
        EXPECT_TRUE(is_available)
            << "Backend should be available when devices found";
    }
}

TEST_F(OneAPIBackendLoadingTest, TensorCreationIfAvailable) {
    ASSERT_TRUE(backend_created_);

    if (!backend_->is_available()) {
        GTEST_SKIP() << "No OneAPI devices available for tensor creation test";
        return;
    }

    // Test tensor creation
    try {
        // Create a simple tensor
        [[maybe_unused]] Device device = Device::oneapi(0);
        std::vector<int64_t> shape = {2, 3};

        // Test allocation
        size_t bytes = 2 * 3 * sizeof(float);
        void* ptr = backend_->allocate(bytes, 0);
        ASSERT_NE(ptr, nullptr) << "Should allocate memory successfully";

        // Test deallocation
        EXPECT_NO_THROW(backend_->deallocate(ptr))
            << "Should deallocate memory without error";

        std::cout << "Successfully created and destroyed tensor on OneAPI device" << std::endl;
    } catch (const std::exception& e) {
        FAIL() << "Tensor creation failed: " << e.what();
    }
}

TEST_F(OneAPIBackendLoadingTest, GracefulHandlingOfNoDevices) {
    ASSERT_TRUE(backend_created_);

    // Even with no devices, backend operations should fail gracefully
    if (backend_->device_count() == 0) {
        // Should not crash when checking availability
        EXPECT_FALSE(backend_->is_available());

        // Should throw appropriate error when trying to use invalid device
        EXPECT_THROW(
            backend_->allocate(1024, 0),
            std::exception
        ) << "Should throw when trying to allocate on non-existent device";
    }
}

} // namespace test
} // namespace tenzor
