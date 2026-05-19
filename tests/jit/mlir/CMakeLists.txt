# Phase 13 / Group A MLIR JIT tests.
# Note: test_optype_roundtrip does NOT need TENZOR_USE_MLIR_JIT=ON since it
# only tests the tracer.hpp OpType enum which is always compiled in.
add_executable(test_optype_roundtrip test_optype_roundtrip.cpp)
target_link_libraries(test_optype_roundtrip PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_optype_roundtrip DISCOVERY_TIMEOUT 30 BACKENDS none LABELS "jit;mlir")

if(TENZOR_USE_MLIR_JIT)
    # Dialect smoke test (needs MLIR libs) — 1A.4
    add_executable(test_dialect_smoke test_dialect_smoke.cpp)
    target_link_libraries(test_dialect_smoke PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_dialect_smoke DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir")

    # Runtime skeleton test — 1A.7
    add_executable(test_runtime_skeleton test_runtime_skeleton.cpp)
    target_link_libraries(test_runtime_skeleton PRIVATE tenzor_core tenzor_jit_mlir GTest::gtest_main)
    tenzor_discover_tests(test_runtime_skeleton DISCOVERY_TIMEOUT 60 BACKENDS none LABELS "jit;mlir")
endif()
