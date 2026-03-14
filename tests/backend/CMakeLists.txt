# =============================================================================
# tests/backend/CMakeLists.txt — Backend allocator tests
# =============================================================================

# OneAPI-specific tests
if(TENZOR_BUILD_ONEAPI)
    find_package(IntelSYCL QUIET)

    # OneAPI backend tests (comprehensive)
    add_executable(test_oneapi_backend test_oneapi_backend.cpp)
    target_link_libraries(test_oneapi_backend PRIVATE tenzor_core GTest::gtest_main)

    # OneAPI device initialization can be slow, use 60 second timeout
    tenzor_discover_tests(test_oneapi_backend DISCOVERY_TIMEOUT 60)
endif()

# Vulkan Caching Allocator Tests
if(TENZOR_BUILD_VULKAN)
    find_package(Vulkan REQUIRED)

    add_executable(test_vulkan_caching_allocator test_vulkan_caching_allocator.cpp)
    target_link_libraries(test_vulkan_caching_allocator PRIVATE
        tenzor_core tenzor_backend_vulkan ${Vulkan_LIBRARIES} GTest::gtest_main)
    target_include_directories(test_vulkan_caching_allocator PRIVATE ${Vulkan_INCLUDE_DIRS})
    tenzor_discover_tests(test_vulkan_caching_allocator DISCOVERY_TIMEOUT 60)
endif()

# OneAPI Caching Allocator Tests
if(TENZOR_BUILD_ONEAPI AND TARGET tenzor_backend_oneapi)
    add_executable(test_oneapi_caching_allocator test_oneapi_caching_allocator.cpp)

    # Link against the library (uses lib prefix and standard -l flag)
    target_link_libraries(test_oneapi_caching_allocator PRIVATE
        tenzor_core tenzor_backend_oneapi GTest::gtest_main)

    # Add the bin directory to link path so -ltenzor_backend_oneapi finds the library
    target_link_directories(test_oneapi_caching_allocator PRIVATE ${CMAKE_BINARY_DIR}/bin)

    # Ensure the oneapi backend is built before the test
    add_dependencies(test_oneapi_caching_allocator tenzor_backend_oneapi)

    tenzor_discover_tests(test_oneapi_caching_allocator DISCOVERY_TIMEOUT 60)
endif()
