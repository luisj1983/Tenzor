# =============================================================================
# tests/autograd/CMakeLists.txt — Autograd-specific tests
# =============================================================================

# Autograd edge cases
add_executable(test_autograd_edge_cases test_autograd_edge_cases.cpp)
target_link_libraries(test_autograd_edge_cases PRIVATE tenzor_core GTest::gtest_main)

# Higher order gradients
add_executable(test_higher_order_gradients test_higher_order_gradients.cpp)
target_link_libraries(test_higher_order_gradients PRIVATE tenzor_core GTest::gtest_main)

# Higher order gradients for NN layers (Conv2d, LSTM, GRU)
add_executable(test_higher_order_nn test_higher_order_nn.cpp)
target_link_libraries(test_higher_order_nn PRIVATE tenzor_core GTest::gtest_main)

# Higher-order through nonlinear activations (sigmoid, tanh) — P4.2
add_executable(test_higher_order_activations test_higher_order_activations.cpp)
target_link_libraries(test_higher_order_activations PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_higher_order_activations LABELS "autograd" DISCOVERY_TIMEOUT 60)

# Hardshrink / Threshold implementations (P2.6b)
add_executable(test_hardshrink_threshold test_hardshrink_threshold.cpp)
target_link_libraries(test_hardshrink_threshold PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_hardshrink_threshold LABELS "autograd" DISCOVERY_TIMEOUT 60)

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

# retain_grad warning (Phase 3.2) — isolated so the warn-once flag is fresh
add_executable(test_retain_grad_warning test_retain_grad_warning.cpp)
target_link_libraries(test_retain_grad_warning PRIVATE tenzor_core GTest::gtest_main)

# ============================================================================
# Test Registration
# ============================================================================
tenzor_discover_tests(test_autograd_edge_cases LABELS "autograd" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_retain_grad_warning LABELS "autograd" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_higher_order_gradients LABELS "autograd" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_higher_order_nn LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
# Parameterized backend tests use PRE_TEST discovery so that test listing
# runs at ctest time rather than build time.  POST_BUILD discovery fails
# when tenzor::initialize() cannot enumerate GPUs in the build sandbox,
# causing ctest to silently register 0 tests (root cause of 0% coverage).
tenzor_discover_tests(test_gradcheck LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_gradcheck_direct LABELS "autograd" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_gradcheck_extended LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Gradchecks for previously untested autograd functions
add_executable(test_gradcheck_missing test_gradcheck_missing.cpp)
target_link_libraries(test_gradcheck_missing PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_gradcheck_missing LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

tenzor_discover_tests(test_autograd_additional LABELS "autograd" DISCOVERY_TIMEOUT 60)
tenzor_discover_tests(test_hooks LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
tenzor_discover_tests(test_inplace_autograd LABELS "autograd" DISCOVERY_TIMEOUT 60)

# Custom Op Autograd Integration Tests
add_executable(test_custom_op_autograd test_custom_op_autograd.cpp)
target_link_libraries(test_custom_op_autograd PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_custom_op_autograd LABELS "autograd" DISCOVERY_TIMEOUT 60)

# Scatter-add autograd tests
add_executable(test_scatter_add_autograd test_scatter_add_autograd.cpp)
target_link_libraries(test_scatter_add_autograd PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_scatter_add_autograd LABELS "autograd" DISCOVERY_TIMEOUT 60)

# Forward-mode AD (JVP) tests
add_executable(test_jvp test_jvp.cpp)
target_link_libraries(test_jvp PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jvp LABELS "autograd" DISCOVERY_TIMEOUT 60)

add_executable(test_jvp_expanded test_jvp_expanded.cpp)
target_link_libraries(test_jvp_expanded PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jvp_expanded LABELS "autograd" DISCOVERY_TIMEOUT 60)

add_executable(test_jvp_expanded_multidtype test_jvp_expanded_multidtype.cpp)
target_link_libraries(test_jvp_expanded_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jvp_expanded_multidtype LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Jacobian and Hessian tests
add_executable(test_jacobian_hessian test_jacobian_hessian.cpp)
target_link_libraries(test_jacobian_hessian PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jacobian_hessian LABELS "autograd" DISCOVERY_TIMEOUT 60)

# Vectorized map (vmap) tests
add_executable(test_vmap test_vmap.cpp)
target_link_libraries(test_vmap PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_vmap LABELS "autograd" DISCOVERY_TIMEOUT 60)

add_executable(test_vmap_expanded test_vmap_expanded.cpp)
target_link_libraries(test_vmap_expanded PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_vmap_expanded LABELS "autograd" DISCOVERY_TIMEOUT 60)

# Anomaly detection tests
add_executable(test_anomaly_detection test_anomaly_detection.cpp)
target_link_libraries(test_anomaly_detection PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_anomaly_detection LABELS "autograd" DISCOVERY_TIMEOUT 60)

# HigherOrderGradMode::Warn disconnection counter and is_higher_order_stub() tests
add_executable(test_warn_mode_counter test_warn_mode_counter.cpp)
target_link_libraries(test_warn_mode_counter PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_warn_mode_counter LABELS "autograd" DISCOVERY_TIMEOUT 60)

# Graph cleanup tests (verify grad_fn cleared on intermediates after backward)
add_executable(test_graph_cleanup test_graph_cleanup.cpp)
target_link_libraries(test_graph_cleanup PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_graph_cleanup LABELS "autograd" DISCOVERY_TIMEOUT 60)

# Composable function transforms (func::grad, func::vmap, etc.)
add_executable(test_func test_func.cpp)
target_link_libraries(test_func PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_func LABELS "autograd" DISCOVERY_TIMEOUT 60)

# Activation offload per-function policy tests
add_executable(test_activation_offload test_activation_offload.cpp)
target_link_libraries(test_activation_offload PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_activation_offload LABELS "autograd" DISCOVERY_TIMEOUT 60)

# Broadcasting gradient shape tests
add_executable(test_broadcasting_gradients test_broadcasting_gradients.cpp)
target_link_libraries(test_broadcasting_gradients PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_broadcasting_gradients LABELS "autograd" DISCOVERY_TIMEOUT 60)

# ============================================================================
# Multi-backend autograd tests (Phase 3)
# ============================================================================

# Graph cleanup multi-backend
add_executable(test_graph_cleanup_multidtype test_graph_cleanup_multidtype.cpp)
target_link_libraries(test_graph_cleanup_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_graph_cleanup_multidtype LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Broadcasting gradients multi-backend
add_executable(test_broadcasting_gradients_multidtype test_broadcasting_gradients_multidtype.cpp)
target_link_libraries(test_broadcasting_gradients_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_broadcasting_gradients_multidtype LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Vmap expanded multi-backend
add_executable(test_vmap_expanded_multidtype test_vmap_expanded_multidtype.cpp)
target_link_libraries(test_vmap_expanded_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_vmap_expanded_multidtype LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Custom op autograd multi-backend
add_executable(test_custom_op_autograd_multidtype test_custom_op_autograd_multidtype.cpp)
target_link_libraries(test_custom_op_autograd_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_custom_op_autograd_multidtype LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Phase 3b: Medium autograd conversions
add_executable(test_func_multidtype test_func_multidtype.cpp)
target_link_libraries(test_func_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_func_multidtype LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_vmap_multidtype test_vmap_multidtype.cpp)
target_link_libraries(test_vmap_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_vmap_multidtype LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_scatter_add_autograd_multidtype test_scatter_add_autograd_multidtype.cpp)
target_link_libraries(test_scatter_add_autograd_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_scatter_add_autograd_multidtype LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_gradcheck_direct_multidtype test_gradcheck_direct_multidtype.cpp)
target_link_libraries(test_gradcheck_direct_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_gradcheck_direct_multidtype LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Phase 3c: Hard autograd conversions
add_executable(test_autograd_edge_cases_multidtype test_autograd_edge_cases_multidtype.cpp)
target_link_libraries(test_autograd_edge_cases_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_autograd_edge_cases_multidtype LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_inplace_autograd_multidtype test_inplace_autograd_multidtype.cpp)
target_link_libraries(test_inplace_autograd_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_inplace_autograd_multidtype LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_jvp_multidtype test_jvp_multidtype.cpp)
target_link_libraries(test_jvp_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jvp_multidtype LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

add_executable(test_jacobian_hessian_multidtype test_jacobian_hessian_multidtype.cpp)
target_link_libraries(test_jacobian_hessian_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jacobian_hessian_multidtype LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Warn mode counter multi-backend
add_executable(test_warn_mode_counter_multidtype test_warn_mode_counter_multidtype.cpp)
target_link_libraries(test_warn_mode_counter_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_warn_mode_counter_multidtype LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Activation offload multi-backend
add_executable(test_activation_offload_multidtype test_activation_offload_multidtype.cpp)
target_link_libraries(test_activation_offload_multidtype PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_activation_offload_multidtype LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Additional autograd features (13 tests using BackendTest fixture)
add_executable(test_autograd_features test_autograd_features.cpp)
target_link_libraries(test_autograd_features PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_autograd_features LABELS "autograd" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Thread safety tests for concurrent backward with shared parameters
add_executable(test_autograd_thread_safety test_autograd_thread_safety.cpp)
target_link_libraries(test_autograd_thread_safety PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_autograd_thread_safety LABELS "autograd" DISCOVERY_TIMEOUT 120 DISCOVERY_MODE PRE_TEST)
