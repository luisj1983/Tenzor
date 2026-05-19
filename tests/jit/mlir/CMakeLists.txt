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
endif()
