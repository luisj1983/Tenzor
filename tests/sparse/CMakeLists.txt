# =============================================================================
# tests/sparse/CMakeLists.txt — Sparse-tensor subsystem tests
# (audit-2026-05-03 N4)
# =============================================================================

add_executable(test_sparse_triangular_solve test_sparse_triangular_solve.cpp)
target_link_libraries(test_sparse_triangular_solve PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_sparse_triangular_solve LABELS "sparse" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)
