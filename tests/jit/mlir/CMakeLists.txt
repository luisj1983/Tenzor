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
endif()
