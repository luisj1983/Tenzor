# Phase 13 / Group A MLIR JIT tests.
# Note: test_optype_roundtrip does NOT need TENZOR_USE_MLIR_JIT=ON since it
# only tests the tracer.hpp OpType enum which is always compiled in.
add_executable(test_optype_roundtrip test_optype_roundtrip.cpp)
target_link_libraries(test_optype_roundtrip PRIVATE tenzor_core GTest::gtest_main)
tenzor_discover_tests(test_optype_roundtrip DISCOVERY_TIMEOUT 30 BACKENDS none LABELS "jit;mlir")

if(TENZOR_USE_MLIR_JIT)
    # Per-task JIT tests (text emission, IREE compile, IREE invoke, custom_call
    # smoke) are added in tasks A.6′, A.7, A.8, A.9 of the Phase 1A amendment.
endif()
