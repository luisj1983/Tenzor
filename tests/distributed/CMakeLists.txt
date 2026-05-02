# =============================================================================
# tests/distributed/CMakeLists.txt — Distributed training tests
# =============================================================================

add_executable(test_rpc_agent test_rpc_agent.cpp)
target_link_libraries(test_rpc_agent PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_rpc_agent LABELS "distributed" DISCOVERY_TIMEOUT 60)

add_executable(test_rref test_rref.cpp)
target_link_libraries(test_rref PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_rref LABELS "distributed" DISCOVERY_TIMEOUT 60)

add_executable(test_rendezvous test_rendezvous.cpp)
target_link_libraries(test_rendezvous PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_rendezvous LABELS "distributed" DISCOVERY_TIMEOUT 60)

add_executable(test_health_monitor test_health_monitor.cpp)
target_link_libraries(test_health_monitor PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_health_monitor LABELS "distributed" DISCOVERY_TIMEOUT 60)

add_executable(test_elastic_trainer test_elastic_trainer.cpp)
target_link_libraries(test_elastic_trainer PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_elastic_trainer LABELS "distributed" DISCOVERY_TIMEOUT 60)

# Audit 2026-05-02 F.1: DTensor + DeviceMesh — public surface had zero tests.
add_executable(test_dtensor test_dtensor.cpp)
target_link_libraries(test_dtensor PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dtensor LABELS "distributed" DISCOVERY_TIMEOUT 60)

add_executable(test_device_mesh test_device_mesh.cpp)
target_link_libraries(test_device_mesh PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_device_mesh LABELS "distributed" DISCOVERY_TIMEOUT 60)

# Audit 2026-05-02 F.5: GradientCompressor C++ side (Python had test but
# the public C++ API didn't).
add_executable(test_gradient_compression test_gradient_compression.cpp)
target_link_libraries(test_gradient_compression PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_gradient_compression LABELS "distributed" DISCOVERY_TIMEOUT 60)

# Audit 2026-05-02 H.1: Gloo backend single-process smoke (no per-backend
# unit test before this — the backends were only exercised implicitly via
# the Python multirank tests).
add_executable(test_gloo_backend_smoke test_gloo_backend_smoke.cpp)
target_link_libraries(test_gloo_backend_smoke PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_gloo_backend_smoke LABELS "distributed" DISCOVERY_TIMEOUT 60)

# Audit 2026-05-02 H.2: MPI backend single-process smoke (skips at runtime
# when TENZOR_BUILD_MPI=OFF or no MPI runtime is available).
add_executable(test_mpi_backend_smoke test_mpi_backend_smoke.cpp)
target_link_libraries(test_mpi_backend_smoke PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_mpi_backend_smoke LABELS "distributed" DISCOVERY_TIMEOUT 60)

# Audit 2026-05-02 H.3: NCCL backend single-process smoke (skips at runtime
# when CUDA or NCCL isn't available).
add_executable(test_nccl_backend_smoke test_nccl_backend_smoke.cpp)
target_link_libraries(test_nccl_backend_smoke PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_nccl_backend_smoke LABELS "distributed" DISCOVERY_TIMEOUT 60)
