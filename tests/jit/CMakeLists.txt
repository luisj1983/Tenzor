# =============================================================================
# tests/jit/CMakeLists.txt — JIT compiler tests
# =============================================================================

# JIT compiler comprehensive tests (graph, tracing, optimization, numerical verification)
add_executable(test_jit_compiler test_jit_compiler.cpp)
target_link_libraries(test_jit_compiler PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jit_compiler LABELS "jit" DISCOVERY_TIMEOUT 60)

# Per-pass fusion tests (Phase 5.5 of test-coverage campaign)
add_executable(test_fusion_passes test_fusion_passes.cpp)
target_link_libraries(test_fusion_passes PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_fusion_passes LABELS "jit" DISCOVERY_TIMEOUT 60)

# PatternMatcher direct unit coverage (LayerNorm/RMSNorm affine shape guard, JIT-R111)
add_executable(test_pattern_matcher test_pattern_matcher.cpp)
target_link_libraries(test_pattern_matcher PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_pattern_matcher LABELS "jit" DISCOVERY_TIMEOUT 60)

# Fusion cost model device-aware heuristic tests (JIT-R026)
add_executable(test_fusion_cost_model test_fusion_cost_model.cpp)
target_link_libraries(test_fusion_cost_model PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_fusion_cost_model LABELS "jit" DISCOVERY_TIMEOUT 60)

add_executable(test_jit_quantization test_jit_quantization.cpp)
target_link_libraries(test_jit_quantization PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jit_quantization LABELS "jit" DISCOVERY_TIMEOUT 120)

add_executable(test_control_flow test_control_flow.cpp)
target_link_libraries(test_control_flow PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_control_flow LABELS "jit" DISCOVERY_TIMEOUT 60)

add_executable(test_codegen test_codegen.cpp)
target_link_libraries(test_codegen PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_codegen LABELS "jit" DISCOVERY_TIMEOUT 60)

# P0 / Fix-1 regression coverage: every ElemOp must produce correct output
# via execute_fused_cpu, including the 30 previously dropped via default:.
add_executable(test_codegen_elementwise_cpu test_codegen_elementwise_cpu.cpp)
target_link_libraries(test_codegen_elementwise_cpu PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_codegen_elementwise_cpu LABELS "jit" DISCOVERY_TIMEOUT 60)

# Phase P0 — JIT correctness fixes (fourth-pass audit):
#  1. CSE must not dedup stateful Dropout/BatchNorm.
#  2. Conv+BN fusion must not run on training-mode BatchNorm.
#  3. GraphReader must throw on missing input values, not silently drop.
#  4. Node::replace_input must clean up the OLD value's uses_ list.
add_executable(test_jit_correctness_fixes test_jit_correctness_fixes.cpp)
target_link_libraries(test_jit_correctness_fixes PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jit_correctness_fixes LABELS "jit" DISCOVERY_TIMEOUT 60)

add_executable(test_extended_codegen test_extended_codegen.cpp)
target_link_libraries(test_extended_codegen PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_extended_codegen LABELS "jit" DISCOVERY_TIMEOUT 60)

# Security hardening: compile_script parser (recursion/length/unroll caps) and
# GraphReader deserializer (magic/version/bounded-count) must reject malformed
# or untrusted input cleanly instead of crashing / OOM / OOB.
add_executable(test_jit_hardening test_jit_hardening.cpp)
target_link_libraries(test_jit_hardening PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jit_hardening LABELS "jit" DISCOVERY_TIMEOUT 60)

# CUDA-graph replay must copy fresh inputs into the captured buffers
# (device-to-device) instead of re-running over stale capture-time data.
# Skips when no CUDA device is present (GPU-only path).
add_executable(test_cuda_graph_replay test_cuda_graph_replay.cpp)
target_link_libraries(test_cuda_graph_replay PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_cuda_graph_replay LABELS "jit" DISCOVERY_TIMEOUT 60)

add_executable(test_compile test_compile.cpp)
target_link_libraries(test_compile PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_compile LABELS "jit" DISCOVERY_TIMEOUT 60)

# End-to-end trace/replay correctness for shape ops (transpose, flatten range)
# and elementwise ops (sin, cos, rsqrt) + RMSNorm. Guards against traced
# transpose/flatten losing their axes/range and against unmapped ops being
# frozen as constants at trace time.
add_executable(test_jit_trace_ops test_jit_trace_ops.cpp)
target_link_libraries(test_jit_trace_ops PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jit_trace_ops LABELS "jit" DISCOVERY_TIMEOUT 60)

add_executable(test_control_flow_compile test_control_flow_compile.cpp)
target_link_libraries(test_control_flow_compile PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_control_flow_compile LABELS "jit" DISCOVERY_TIMEOUT 60)

add_executable(test_dynamic_shapes test_dynamic_shapes.cpp)
target_link_libraries(test_dynamic_shapes PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_dynamic_shapes LABELS "jit" DISCOVERY_TIMEOUT 60)

# Low-level Graph/Tracer API tests (recreated as fresh test in follow-up cleanup)
add_executable(test_jit_lowlevel test_jit_lowlevel.cpp)
target_link_libraries(test_jit_lowlevel PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jit_lowlevel LABELS "jit" DISCOVERY_TIMEOUT 60)

# Strict-mode / graph-break detection (P4.1: better tracing)
add_executable(test_jit_strict_mode test_jit_strict_mode.cpp)
target_link_libraries(test_jit_strict_mode PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jit_strict_mode LABELS "jit" DISCOVERY_TIMEOUT 60)

# P4.1b: control_flow.hpp ↔ Tracer integration test
add_executable(test_control_flow_hook_integration test_control_flow_hook_integration.cpp)
target_link_libraries(test_control_flow_hook_integration PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_control_flow_hook_integration LABELS "jit" DISCOVERY_TIMEOUT 60)

# JIT backend parity tests (multi-backend forward pass comparison for composed models)
add_executable(test_jit_backend_parity test_jit_backend_parity.cpp)
target_link_libraries(test_jit_backend_parity PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jit_backend_parity LABELS "jit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Scripted-if graph-break + in-place-op value-versioning correctness fixes (M7/M8)
add_executable(test_jit_control_flow_inplace_fixes test_jit_control_flow_inplace_fixes.cpp)
target_link_libraries(test_jit_control_flow_inplace_fixes PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jit_control_flow_inplace_fixes LABELS "jit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# C4/M9: control-flow subgraph serialization round-trip + device-portable
# constants. Verifies save/load preserves If/Loop then/else/body subgraphs and
# that serialized constants replay on whatever backend the runtime inputs use.
add_executable(test_jit_serialization_control_flow test_jit_serialization_control_flow.cpp)
target_link_libraries(test_jit_serialization_control_flow PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_jit_serialization_control_flow LABELS "jit" DISCOVERY_TIMEOUT 60 DISCOVERY_MODE PRE_TEST)

# Graph-viz DOT schema validation (audit-2026-05-03 N4)
add_executable(test_graph_viz_schema test_graph_viz_schema.cpp)
target_link_libraries(test_graph_viz_schema PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_graph_viz_schema LABELS "jit" DISCOVERY_TIMEOUT 60)

# JIT-R016/JIT-R017: MemorySwapPlanner consumer-redirect consistency
# (subgraph-aware node search) and device-gated swap candidacy. Planner is
# currently unwired (no Compiler/codegen path invokes it), so this is its
# only coverage; still worth being correct per the class's own doc comment.
add_executable(test_memory_planner_swap test_memory_planner_swap.cpp)
target_link_libraries(test_memory_planner_swap PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_memory_planner_swap LABELS "jit" DISCOVERY_TIMEOUT 60)

# Phase 13 MLIR JIT tests (subdirectory; test_optype_roundtrip always built;
# dialect/runtime tests gated on TENZOR_USE_MLIR_JIT inside the subdir)
add_subdirectory(mlir)
