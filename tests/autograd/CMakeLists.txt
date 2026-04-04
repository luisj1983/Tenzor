# =============================================================================
# tests/autograd/CMakeLists.txt — Autograd-specific tests
# =============================================================================

# Autograd edge cases
add_executable(test_autograd_edge_cases test_autograd_edge_cases.cpp)
target_link_libraries(test_autograd_edge_cases PRIVATE tenzor_core GTest::gtest_main)

# Higher order gradients
add_executable(test_higher_order_gradients test_higher_order_gradients.cpp)
target_link_libraries(test_higher_order_gradients PRIVATE tenzor_core GTest::gtest_main)

# Gradient checking tests
add_executable(test_gradcheck test_gradcheck.cpp)
target_link_libraries(test_gradcheck PRIVATE tenzor_core GTest::gtest_main)

# Direct gradcheck tests (non-parameterized, CPU-only, for coverage)
add_executable(test_gradcheck_direct test_gradcheck_direct.cpp)
target_link_libraries(test_gradcheck_direct PRIVATE tenzor_core GTest::gtest_main)

# Extended gradcheck tests (multi-backend)
add_executable(test_gradcheck_extended test_gradcheck_extended.cpp)
target_link_libraries(test_gradcheck_extended PRIVATE tenzor_core GTest::gtest_main)

# Additional autograd tests for increased coverage
add_executable(test_autograd_additional test_autograd_additional.cpp)
target_link_libraries(test_autograd_additional PRIVATE tenzor_core GTest::gtest_main)

# Autograd Hooks Tests
add_executable(test_hooks test_hooks.cpp)
target_link_libraries(test_hooks PRIVATE tenzor_core GTest::gtest_main)

# In-Place + Autograd Version Counter Tests
add_executable(test_inplace_autograd test_inplace_autograd.cpp)
target_link_libraries(test_inplace_autograd PRIVATE tenzor_core GTest::gtest_main)

# ============================================================================
# Test Registration
# ============================================================================
tenzor_discover_tests(test_autograd_edge_cases DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_higher_order_gradients DISCOVERY_TIMEOUT 60)
# Parameterized backend tests use PRE_TEST discovery so that test listing
# runs at ctest time rather than build time.  POST_BUILD discovery fails
# when tenzor::initialize() cannot enumerate GPUs in the build sandbox,
# causing ctest to silently register 0 tests (root cause of 0% coverage).
tenzor_discover_tests(test_gradcheck DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_gradcheck_direct DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_gradcheck_extended DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_autograd_additional DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_hooks DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_inplace_autograd DISCOVERY_TIMEOUT 60)

# Custom Op Autograd Integration Tests
add_executable(test_custom_op_autograd test_custom_op_autograd.cpp)
target_link_libraries(test_custom_op_autograd PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_custom_op_autograd DISCOVERY_TIMEOUT 60)

# Scatter-add autograd tests
add_executable(test_scatter_add_autograd test_scatter_add_autograd.cpp)
target_link_libraries(test_scatter_add_autograd PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_scatter_add_autograd DISCOVERY_TIMEOUT 60)

# Forward-mode AD (JVP) tests
add_executable(test_jvp test_jvp.cpp)
target_link_libraries(test_jvp PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jvp DISCOVERY_TIMEOUT 60)

add_executable(test_jvp_expanded test_jvp_expanded.cpp)
target_link_libraries(test_jvp_expanded PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jvp_expanded DISCOVERY_TIMEOUT 60)

# Jacobian and Hessian tests
add_executable(test_jacobian_hessian test_jacobian_hessian.cpp)
target_link_libraries(test_jacobian_hessian PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jacobian_hessian DISCOVERY_TIMEOUT 60)

# Vectorized map (vmap) tests
add_executable(test_vmap test_vmap.cpp)
target_link_libraries(test_vmap PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_vmap DISCOVERY_TIMEOUT 60)

add_executable(test_vmap_expanded test_vmap_expanded.cpp)
target_link_libraries(test_vmap_expanded PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_vmap_expanded DISCOVERY_TIMEOUT 60)

# Anomaly detection tests
add_executable(test_anomaly_detection test_anomaly_detection.cpp)
target_link_libraries(test_anomaly_detection PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_anomaly_detection DISCOVERY_TIMEOUT 60)

# HigherOrderGradMode::Warn disconnection counter and is_higher_order_stub() tests
add_executable(test_warn_mode_counter test_warn_mode_counter.cpp)
target_link_libraries(test_warn_mode_counter PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_warn_mode_counter DISCOVERY_TIMEOUT 60)

# Graph cleanup tests (verify grad_fn cleared on intermediates after backward)
add_executable(test_graph_cleanup test_graph_cleanup.cpp)
target_link_libraries(test_graph_cleanup PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_graph_cleanup DISCOVERY_TIMEOUT 60)
