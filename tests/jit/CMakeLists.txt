# =============================================================================
# tests/jit/CMakeLists.txt — JIT compiler tests
# =============================================================================

# JIT compiler comprehensive tests (graph, tracing, optimization, numerical verification)
add_executable(test_jit_compiler test_jit_compiler.cpp)
target_link_libraries(test_jit_compiler PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jit_compiler DISCOVERY_TIMEOUT 60)

add_executable(test_control_flow test_control_flow.cpp)
target_link_libraries(test_control_flow PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_control_flow DISCOVERY_TIMEOUT 60)

add_executable(test_codegen test_codegen.cpp)
target_link_libraries(test_codegen PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_codegen DISCOVERY_TIMEOUT 60)
