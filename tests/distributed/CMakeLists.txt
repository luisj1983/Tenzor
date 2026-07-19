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

# FINDING 21 "plan K3": real 4-process multi-rank run (MultiRankDeviceMeshTest
# needs a 2x2 mesh, i.e. exactly 4 ranks, to exercise GlooProcessGroup::split()
# with two distinct sub-group colors).
add_test(
    NAME test_device_mesh_multirank
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_multirank_test.sh 4
            $<TARGET_FILE:test_device_mesh>
            --gtest_filter=MultiRankDeviceMeshTest.*
)
set_tests_properties(test_device_mesh_multirank PROPERTIES
    LABELS "distributed;multirank"
    TIMEOUT 60
)

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
# when CUDA or NCCL isn't available). The NCCLBackend implementation lives in
# the tenzor_backend_cuda DSO (not tenzor_core), so this test must link that
# DSO to resolve NCCLBackend's symbols; gated on the cuda backend target so
# CPU-only builds skip the test entirely.
if(TARGET tenzor_backend_cuda)
    add_executable(test_nccl_backend_smoke test_nccl_backend_smoke.cpp)
    target_link_libraries(test_nccl_backend_smoke PRIVATE tenzor_core tenzor_backend_cuda GTest::gtest_main)
    tenzor_discover_tests(test_nccl_backend_smoke LABELS "distributed" DISCOVERY_TIMEOUT 60)
endif()

# FINDING 4: RCCL is now wired into tenzor_backend_rocm (see
# src/backends/rocm/CMakeLists.txt) with a real create_comm_backend()
# override, but NCCLBackend is a distinct compiled type per backend DSO — see
# test_nccl_backend_smoke_rocm.cpp's file comment for why this must be a
# separate executable rather than linking tenzor_backend_rocm into the CUDA
# test above. Gated on the rocm backend target so builds without ROCm skip
# this executable entirely (mirrors the CUDA gating immediately above).
if(TARGET tenzor_backend_rocm)
    add_executable(test_nccl_backend_smoke_rocm test_nccl_backend_smoke_rocm.cpp)
    target_link_libraries(test_nccl_backend_smoke_rocm PRIVATE tenzor_core tenzor_backend_rocm GTest::gtest_main)
    tenzor_discover_tests(test_nccl_backend_smoke_rocm LABELS "distributed" DISCOVERY_TIMEOUT 60)
endif()

# FINDING 21: NCCLProcessGroup (distinct from NCCLBackend above -- the class
# DistributedContext::initialize() actually constructs for Backend::NCCL) had
# zero test coverage anywhere. Same per-backend-DSO split rationale as the
# NCCLBackend smoke tests above. Additionally gated on TENZOR_CUDA_NCCL_
# AVAILABLE / TENZOR_ROCM_RCCL_AVAILABLE (set by the respective backend
# CMakeLists.txt): NCCLProcessGroup's class declaration in process_group.hpp
# is itself wrapped in "#ifdef TENZOR_HAS_NCCL", so this test TU needs that
# define too, not just a link against the DSO -- and it must only be added
# when the DSO was ACTUALLY compiled with real NCCL/RCCL, else the class
# would be declared here but never defined (link failure) on a build without
# NCCL/RCCL installed (TENZOR_REQUIRE_NCCL/RCCL default OFF, a legitimate
# configuration).
if(TARGET tenzor_backend_cuda AND TENZOR_CUDA_NCCL_AVAILABLE)
    add_executable(test_nccl_process_group test_nccl_process_group.cpp)
    target_link_libraries(test_nccl_process_group PRIVATE tenzor_core tenzor_backend_cuda GTest::gtest_main)
    target_compile_definitions(test_nccl_process_group PRIVATE TENZOR_HAS_NCCL)
    tenzor_discover_tests(test_nccl_process_group LABELS "distributed" DISCOVERY_TIMEOUT 60)
endif()

if(TARGET tenzor_backend_rocm AND TENZOR_ROCM_RCCL_AVAILABLE)
    add_executable(test_nccl_process_group_rocm test_nccl_process_group_rocm.cpp)
    target_link_libraries(test_nccl_process_group_rocm PRIVATE tenzor_core tenzor_backend_rocm GTest::gtest_main)
    target_compile_definitions(test_nccl_process_group_rocm PRIVATE TENZOR_HAS_NCCL)
    tenzor_discover_tests(test_nccl_process_group_rocm LABELS "distributed" DISCOVERY_TIMEOUT 60)
endif()

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

# FINDING 21 "plan K3": real 2-process multi-rank run (MultiRankA5Test) via
# run_multirank_test.sh -- the multi-process C++ test job this file used to
# defer real multi-rank correctness verification to but which never existed.
add_test(
    NAME test_distributed_all_reduce_autograd_multirank
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_multirank_test.sh 2
            $<TARGET_FILE:test_distributed_all_reduce_autograd>
            --gtest_filter=MultiRankA5Test.*
)
set_tests_properties(test_distributed_all_reduce_autograd_multirank PROPERTIES
    LABELS "distributed;multirank"
    TIMEOUT 60
)

# Audit remediation C1: SyncBatchNorm autograd-aware all-reduce path.
add_executable(test_sync_batchnorm_autograd_aware test_sync_batchnorm_autograd_aware.cpp)
target_link_libraries(test_sync_batchnorm_autograd_aware PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_sync_batchnorm_autograd_aware LABELS "distributed" DISCOVERY_TIMEOUT 60)

# FINDING 21 "plan K3": real 2-process multi-rank run (MultiRankC1Test).
add_test(
    NAME test_sync_batchnorm_autograd_aware_multirank
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_multirank_test.sh 2
            $<TARGET_FILE:test_sync_batchnorm_autograd_aware>
            --gtest_filter=MultiRankC1Test.*
)
set_tests_properties(test_sync_batchnorm_autograd_aware_multirank PROPERTIES
    LABELS "distributed;multirank"
    TIMEOUT 60
)

# Audit remediation B2: DTensor::redistribute collective wiring.
add_executable(test_dtensor_redistribute test_dtensor_redistribute.cpp)
target_link_libraries(test_dtensor_redistribute PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dtensor_redistribute LABELS "distributed" DISCOVERY_TIMEOUT 60)

# FINDING 21 "plan K3": real 2-process multi-rank run (MultiRankB2Test).
add_test(
    NAME test_dtensor_redistribute_multirank
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_multirank_test.sh 2
            $<TARGET_FILE:test_dtensor_redistribute>
            --gtest_filter=MultiRankB2Test.*
)
set_tests_properties(test_dtensor_redistribute_multirank PROPERTIES
    LABELS "distributed;multirank"
    TIMEOUT 60
)

# Audit remediation B4: FSDP2 backward_hook collective wiring integration.
add_executable(test_fsdp2_collective_wiring test_fsdp2_collective_wiring.cpp)
target_link_libraries(test_fsdp2_collective_wiring PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_fsdp2_collective_wiring LABELS "distributed" DISCOVERY_TIMEOUT 60)

# FINDING 21 "plan K3": real 2-process multi-rank run (MultiRankB4Test).
add_test(
    NAME test_fsdp2_collective_wiring_multirank
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_multirank_test.sh 2
            $<TARGET_FILE:test_fsdp2_collective_wiring>
            --gtest_filter=MultiRankB4Test.*
)
set_tests_properties(test_fsdp2_collective_wiring_multirank PROPERTIES
    LABELS "distributed;multirank"
    TIMEOUT 60
)
