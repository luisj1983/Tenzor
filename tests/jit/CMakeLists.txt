# =============================================================================
# tests/jit/CMakeLists.txt — JIT compiler tests
# =============================================================================

# JIT compiler comprehensive tests (graph, tracing, optimization, numerical verification)
add_executable(test_jit_compiler test_jit_compiler.cpp)
target_link_libraries(test_jit_compiler PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jit_compiler LABELS "jit" DISCOVERY_TIMEOUT 60)

add_executable(test_control_flow test_control_flow.cpp)
target_link_libraries(test_control_flow PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_control_flow LABELS "jit" DISCOVERY_TIMEOUT 60)

add_executable(test_codegen test_codegen.cpp)
target_link_libraries(test_codegen PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_codegen LABELS "jit" DISCOVERY_TIMEOUT 60)

add_executable(test_extended_codegen test_extended_codegen.cpp)
target_link_libraries(test_extended_codegen PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_extended_codegen LABELS "jit" DISCOVERY_TIMEOUT 60)

add_executable(test_compile test_compile.cpp)
target_link_libraries(test_compile PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_compile LABELS "jit" DISCOVERY_TIMEOUT 60)

add_executable(test_control_flow_compile test_control_flow_compile.cpp)
target_link_libraries(test_control_flow_compile PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_control_flow_compile LABELS "jit" DISCOVERY_TIMEOUT 60)

add_executable(test_dynamic_shapes test_dynamic_shapes.cpp)
target_link_libraries(test_dynamic_shapes PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dynamic_shapes LABELS "jit" DISCOVERY_TIMEOUT 60)
