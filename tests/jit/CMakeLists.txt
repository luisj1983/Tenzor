# =============================================================================
# tests/jit/CMakeLists.txt — JIT compiler tests
# =============================================================================

# JIT compiler comprehensive tests (graph, tracing, optimization, numerical verification)
add_executable(test_jit_compiler test_jit_compiler.cpp)
target_link_libraries(test_jit_compiler PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jit_compiler DISCOVERY_TIMEOUT 30)
