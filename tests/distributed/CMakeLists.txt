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

# Inf-F: native GlooProcessGroup::all_to_all_single + MPIProcessGroup
# split/all_to_all_single contract tests (single-process degenerates).
add_executable(test_process_group_inf_f test_process_group_inf_f.cpp)
target_link_libraries(test_process_group_inf_f PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_process_group_inf_f LABELS "distributed" DISCOVERY_TIMEOUT 60)

# Distributed launch helpers (audit-2026-05-03 N4)
add_executable(test_launch test_launch.cpp)
target_link_libraries(test_launch PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_launch LABELS "distributed" DISCOVERY_TIMEOUT 60)

# SequenceParallel C++ test (audit-2026-05-03 N4)
add_executable(test_sequence_parallel_cpp test_sequence_parallel.cpp)
target_link_libraries(test_sequence_parallel_cpp PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_sequence_parallel_cpp LABELS "distributed" DISCOVERY_TIMEOUT 60)

# Audit remediation A5: DistributedAllReduce autograd Function.
add_executable(test_distributed_all_reduce_autograd test_distributed_all_reduce_autograd.cpp)
target_link_libraries(test_distributed_all_reduce_autograd PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_distributed_all_reduce_autograd LABELS "distributed" DISCOVERY_TIMEOUT 60)

# Audit remediation C1: SyncBatchNorm autograd-aware all-reduce path.
add_executable(test_sync_batchnorm_autograd_aware test_sync_batchnorm_autograd_aware.cpp)
target_link_libraries(test_sync_batchnorm_autograd_aware PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_sync_batchnorm_autograd_aware LABELS "distributed" DISCOVERY_TIMEOUT 60)

# Audit remediation B2: DTensor::redistribute collective wiring.
add_executable(test_dtensor_redistribute test_dtensor_redistribute.cpp)
target_link_libraries(test_dtensor_redistribute PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dtensor_redistribute LABELS "distributed" DISCOVERY_TIMEOUT 60)

# Audit remediation B4: FSDP2 backward_hook collective wiring integration.
add_executable(test_fsdp2_collective_wiring test_fsdp2_collective_wiring.cpp)
target_link_libraries(test_fsdp2_collective_wiring PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_fsdp2_collective_wiring LABELS "distributed" DISCOVERY_TIMEOUT 60)
