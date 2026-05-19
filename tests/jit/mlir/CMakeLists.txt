# Phase 13 / Group A MLIR JIT tests.
# Note: test_optype_roundtrip does NOT need TENZOR_USE_MLIR_JIT=ON since it
# only tests the tracer.hpp OpType enum which is always compiled in.
add_executable(test_optype_roundtrip test_optype_roundtrip.cpp)
target_link_libraries(test_optype_roundtrip PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_optype_roundtrip DISCOVERY_TIMEOUT 30 BACKENDS none LABELS "jit;mlir")

if(TENZOR_USE_MLIR_JIT)
    # Task A.6′ — MLIR text emission helpers.
    add_executable(test_mlir_text_emit test_mlir_text_emit.cpp)
    target_link_libraries(test_mlir_text_emit
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_mlir_text_emit
        DISCOVERY_TIMEOUT 30 BACKENDS none LABELS "jit;mlir")

    # Task A.7 — IREE Compiler embedding API integration.
    add_executable(test_iree_compile test_iree_compile.cpp)
    target_link_libraries(test_iree_compile
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_iree_compile
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir;iree")

    # Task A.8 — IREE Runtime invocation via iree-run-module subprocess.
    add_executable(test_iree_invoke test_iree_invoke.cpp)
    target_link_libraries(test_iree_invoke
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_iree_invoke
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir;iree")

    # Task A.9 — Custom_call registrar + placeholder callbacks.
    add_executable(test_iree_customcall_smoke test_iree_customcall_smoke.cpp)
    target_link_libraries(test_iree_customcall_smoke
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_iree_customcall_smoke
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir;iree")

    # Task B.1 — Graph → MLIR text lowering (Add only).
    add_executable(test_lower_add test_lower_add.cpp)
    target_link_libraries(test_lower_add
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_lower_add
        DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir;iree")

    # Task B.2–B.6 — End-to-end x+x via the MLIR backend across IREE
    # targets (llvm-cpu, cuda, vulkan-spirv, rocm), plus cache-hit check.
    add_executable(test_end_to_end_add test_end_to_end_add.cpp)
    target_link_libraries(test_end_to_end_add
        PRIVATE tenzor_core GTest::gtest_main)
    tenzor_discover_tests(test_end_to_end_add
        DISCOVERY_TIMEOUT 120 BACKENDS none LABELS "jit;mlir;iree")

    # Task C.1 — Elementwise ops lower to StableHLO primitives.
    add_executable(test_op_elementwise test_op_elementwise.cpp)
    target_link_libraries(test_op_elementwise
        PRIVATE tenzor_core GTest::gtest_main)
    tenzor_discover_tests(test_op_elementwise
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")

    # Task C.2 — Activations (ReLU / Sigmoid / SiLU / GELU).
    add_executable(test_op_activations test_op_activations.cpp)
    target_link_libraries(test_op_activations
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_activations
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")

    # Task C.3 — Reductions (Sum / Mean / Max / Softmax).
    add_executable(test_op_reductions test_op_reductions.cpp)
    target_link_libraries(test_op_reductions
        PRIVATE tenzor_core GTest::gtest_main)
    tenzor_discover_tests(test_op_reductions
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")

    # Task C.4 — Linalg (MatMul / Bmm / Linear).
    add_executable(test_op_linalg test_op_linalg.cpp)
    target_link_libraries(test_op_linalg
        PRIVATE tenzor_core GTest::gtest_main)
    tenzor_discover_tests(test_op_linalg
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")

    # Task C.5 — Shape ops (Reshape, Permute, Transpose, Slice, Cat,
    # Stack, Squeeze, Unsqueeze, Flatten, Broadcast).
    add_executable(test_op_shape test_op_shape.cpp)
    target_link_libraries(test_op_shape
        PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_op_shape
        DISCOVERY_TIMEOUT 180 BACKENDS none LABELS "jit;mlir;iree")
endif()
