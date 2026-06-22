# Benchmarks for Tenzor
cmake_minimum_required(VERSION 3.25)

# Benchmark executables
set(BENCHMARK_TARGETS
    benchmark_ops
    benchmark_convolutions
    benchmark_memory
    benchmark_training
    bench_fused_ops
    # New comprehensive benchmarks
    benchmark_attention
    benchmark_normalization
    benchmark_mixed_precision
    benchmark_models
    benchmark_embeddings
    benchmark_rnn
    benchmark_dispatch
    # RR.19 (audit-11): benchmark_compression now honours --device, so
    # wire it into the build so the Python runner can pick it up too.
    benchmark_compression
)

# benchmark_ops - Matrix and tensor operations
add_executable(benchmark_ops benchmark_ops.cpp)
target_link_libraries(benchmark_ops PRIVATE tenzor_core)
target_include_directories(benchmark_ops PRIVATE ${PROJECT_SOURCE_DIR}/include)

# benchmark_convolutions - Convolution operations
add_executable(benchmark_convolutions benchmark_convolutions.cpp)
target_link_libraries(benchmark_convolutions PRIVATE tenzor_core)
target_include_directories(benchmark_convolutions PRIVATE ${PROJECT_SOURCE_DIR}/include)

# benchmark_memory - Memory operations
add_executable(benchmark_memory benchmark_memory.cpp)
target_link_libraries(benchmark_memory PRIVATE tenzor_core)
target_include_directories(benchmark_memory PRIVATE ${PROJECT_SOURCE_DIR}/include)

# benchmark_training - Full training loops
add_executable(benchmark_training benchmark_training.cpp)
target_link_libraries(benchmark_training PRIVATE tenzor_core)
target_include_directories(benchmark_training PRIVATE ${PROJECT_SOURCE_DIR}/include)

# bench_fused_ops - Fused operations (existing)
add_executable(bench_fused_ops bench_fused_ops.cpp)
target_link_libraries(bench_fused_ops PRIVATE tenzor_core)
target_include_directories(bench_fused_ops PRIVATE ${PROJECT_SOURCE_DIR}/include)

# benchmark_attention - Transformer and attention mechanisms
add_executable(benchmark_attention benchmark_attention.cpp)
target_link_libraries(benchmark_attention PRIVATE tenzor_core)
target_include_directories(benchmark_attention PRIVATE ${PROJECT_SOURCE_DIR}/include)

# benchmark_normalization - BatchNorm, LayerNorm, RMSNorm
add_executable(benchmark_normalization benchmark_normalization.cpp)
target_link_libraries(benchmark_normalization PRIVATE tenzor_core)
target_include_directories(benchmark_normalization PRIVATE ${PROJECT_SOURCE_DIR}/include)

# benchmark_mixed_precision - FP16/BF16/INT8 performance
add_executable(benchmark_mixed_precision benchmark_mixed_precision.cpp)
target_link_libraries(benchmark_mixed_precision PRIVATE tenzor_core)
target_include_directories(benchmark_mixed_precision PRIVATE ${PROJECT_SOURCE_DIR}/include)

# benchmark_models - End-to-end model benchmarks
add_executable(benchmark_models benchmark_models.cpp)
target_link_libraries(benchmark_models PRIVATE tenzor_core)
target_include_directories(benchmark_models PRIVATE ${PROJECT_SOURCE_DIR}/include)

# benchmark_embeddings - Embedding layer performance
add_executable(benchmark_embeddings benchmark_embeddings.cpp)
target_link_libraries(benchmark_embeddings PRIVATE tenzor_core)
target_include_directories(benchmark_embeddings PRIVATE ${PROJECT_SOURCE_DIR}/include)

# benchmark_rnn - LSTM/GRU performance
add_executable(benchmark_rnn benchmark_rnn.cpp)
target_link_libraries(benchmark_rnn PRIVATE tenzor_core)
target_include_directories(benchmark_rnn PRIVATE ${PROJECT_SOURCE_DIR}/include)

# benchmark_dispatch - Dispatch overhead measurement
add_executable(benchmark_dispatch benchmark_dispatch.cpp)
target_link_libraries(benchmark_dispatch PRIVATE tenzor_core)
target_include_directories(benchmark_dispatch PRIVATE ${PROJECT_SOURCE_DIR}/include)

# RR.19 (audit-11): benchmark_compression — pruning / quantization
# pipelines, driven via Google Benchmark. Needs --device flag wired into
# argv via parse_device_arg(), see common.hpp. Falls back to the bundled
# benchmark library or system one (must already be findable since other
# benchmark binaries that use <benchmark/benchmark.h> compile).
find_package(benchmark QUIET)
if (benchmark_FOUND)
    add_executable(benchmark_compression benchmark_compression.cpp)
    target_link_libraries(benchmark_compression
        PRIVATE tenzor_core benchmark::benchmark)
    target_include_directories(benchmark_compression PRIVATE
        ${PROJECT_SOURCE_DIR}/include)
else()
    message(STATUS "Google Benchmark not found — skipping benchmark_compression")
    # Remove from the build list so foreach below doesn't reference an
    # un-created target.
    list(REMOVE_ITEM BENCHMARK_TARGETS benchmark_compression)
endif()

# Set C++ standard for all benchmarks
foreach(target ${BENCHMARK_TARGETS})
    set_target_properties(${target} PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
endforeach()

# Install benchmarks
install(TARGETS ${BENCHMARK_TARGETS}
    RUNTIME DESTINATION bin/benchmarks
)

# Custom target to run all benchmarks
add_custom_target(run_benchmarks
    COMMAND echo "Running Tenzor Comprehensive Benchmark Suite..."
    COMMAND echo "================================================"
    COMMAND echo ""
    COMMAND echo "1. Running Matrix Operations Benchmark..."
    COMMAND $<TARGET_FILE:benchmark_ops> || true
    COMMAND echo ""
    COMMAND echo "2. Running Convolution Operations Benchmark..."
    COMMAND $<TARGET_FILE:benchmark_convolutions> || true
    COMMAND echo ""
    COMMAND echo "3. Running Memory Operations Benchmark..."
    COMMAND $<TARGET_FILE:benchmark_memory> || true
    COMMAND echo ""
    COMMAND echo "4. Running Training Benchmark..."
    COMMAND $<TARGET_FILE:benchmark_training> || true
    COMMAND echo ""
    COMMAND echo "5. Running Fused Operations Benchmark..."
    COMMAND $<TARGET_FILE:bench_fused_ops> || true
    COMMAND echo ""
    COMMAND echo "6. Running Attention/Transformer Benchmark..."
    COMMAND $<TARGET_FILE:benchmark_attention> || true
    COMMAND echo ""
    COMMAND echo "7. Running Normalization Benchmark..."
    COMMAND $<TARGET_FILE:benchmark_normalization> || true
    COMMAND echo ""
    COMMAND echo "8. Running Mixed Precision Benchmark..."
    COMMAND $<TARGET_FILE:benchmark_mixed_precision> || true
    COMMAND echo ""
    COMMAND echo "9. Running Model-Level Benchmark..."
    COMMAND $<TARGET_FILE:benchmark_models> || true
    COMMAND echo ""
    COMMAND echo "10. Running Embedding Benchmark..."
    COMMAND $<TARGET_FILE:benchmark_embeddings> || true
    COMMAND echo ""
    COMMAND echo "11. Running RNN Benchmark..."
    COMMAND $<TARGET_FILE:benchmark_rnn> || true
    COMMAND echo ""
    COMMAND echo "================================================"
    COMMAND echo "All benchmarks completed!"
    DEPENDS ${BENCHMARK_TARGETS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running all benchmarks"
)

# Individual benchmark targets
add_custom_target(run_ops_benchmark
    COMMAND $<TARGET_FILE:benchmark_ops>
    DEPENDS benchmark_ops
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running matrix operations benchmark"
)

add_custom_target(run_conv_benchmark
    COMMAND $<TARGET_FILE:benchmark_convolutions>
    DEPENDS benchmark_convolutions
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running convolution operations benchmark"
)

add_custom_target(run_memory_benchmark
    COMMAND $<TARGET_FILE:benchmark_memory>
    DEPENDS benchmark_memory
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running memory operations benchmark"
)

add_custom_target(run_training_benchmark
    COMMAND $<TARGET_FILE:benchmark_training>
    DEPENDS benchmark_training
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running training benchmark"
)

add_custom_target(run_fused_benchmark
    COMMAND $<TARGET_FILE:bench_fused_ops>
    DEPENDS bench_fused_ops
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running fused operations benchmark"
)

add_custom_target(run_attention_benchmark
    COMMAND $<TARGET_FILE:benchmark_attention>
    DEPENDS benchmark_attention
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running attention/transformer benchmark"
)

add_custom_target(run_normalization_benchmark
    COMMAND $<TARGET_FILE:benchmark_normalization>
    DEPENDS benchmark_normalization
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running normalization benchmark"
)

add_custom_target(run_mixed_precision_benchmark
    COMMAND $<TARGET_FILE:benchmark_mixed_precision>
    DEPENDS benchmark_mixed_precision
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running mixed precision benchmark"
)

add_custom_target(run_models_benchmark
    COMMAND $<TARGET_FILE:benchmark_models>
    DEPENDS benchmark_models
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running model-level benchmark"
)

add_custom_target(run_embeddings_benchmark
    COMMAND $<TARGET_FILE:benchmark_embeddings>
    DEPENDS benchmark_embeddings
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running embedding benchmark"
)

add_custom_target(run_rnn_benchmark
    COMMAND $<TARGET_FILE:benchmark_rnn>
    DEPENDS benchmark_rnn
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running RNN benchmark"
)

message(STATUS "Configured ${CMAKE_CURRENT_LIST_DIR} with ${BENCHMARK_TARGETS}")
message(STATUS "Use 'make run_benchmarks' to run all benchmarks")
message(STATUS "Individual targets: run_ops_benchmark, run_conv_benchmark, run_attention_benchmark, etc.")

# ============================================================================
# P.8 — umbrella target wiring the DISABLED_ benchmark-style gtests into a
# convenient `tenzor_benchmarks` build/run target. These tests are prefixed
# DISABLED_ in the source so they don't run during normal `ctest` (they're
# timing-only, not correctness checks), but we still want a one-command
# entry point to exercise them when -DTENZOR_BUILD_BENCHMARKS=ON.
#
#   DISABLED_AddPerformance         — tests/unit/test_simd_ops.cpp
#   DISABLED_BaselineRegressionCheck_MatMul512
#                                   — tests/backend_parity/test_performance_regression.cpp
#   DISABLED_BenchmarkInference     — tests/unit/test_roberta.cpp,
#                                     tests/unit/test_electra.cpp (T.9)
#   DISABLED_PerformanceBenchmark   — tests/unit/test_async_ops.cpp (T.9)
#   DISABLED_BenchmarkLargeConv2d /
#   DISABLED_BenchmarkLargeMatmul   — tests/test_vulkan_complete_ops.cpp (T.9)
#
# Re-enable workflow is documented in TESTING.md (P.8 section).
# ============================================================================

# Only wire the umbrella if the underlying test executables exist (they do
# when TENZOR_BUILD_TESTS=ON, which is the default).
set(_tenzor_bench_deps "")
if(TARGET test_simd_ops)
    list(APPEND _tenzor_bench_deps test_simd_ops)
endif()
if(TARGET test_performance_regression)
    list(APPEND _tenzor_bench_deps test_performance_regression)
endif()
# T.9 — additional DISABLED_ perf tests with regression-guard ceilings.
if(TARGET test_roberta)
    list(APPEND _tenzor_bench_deps test_roberta)
endif()
if(TARGET test_electra)
    list(APPEND _tenzor_bench_deps test_electra)
endif()
if(TARGET test_async_ops)
    list(APPEND _tenzor_bench_deps test_async_ops)
endif()
if(TARGET test_vulkan_complete_ops)
    list(APPEND _tenzor_bench_deps test_vulkan_complete_ops)
endif()

if(_tenzor_bench_deps)
    add_custom_target(tenzor_benchmarks
        DEPENDS ${_tenzor_bench_deps} ${BENCHMARK_TARGETS}
        COMMENT "Built tenzor_benchmarks umbrella (including DISABLED_ perf gtests)"
    )

    # T.9 — build per-target run commands conditionally so the umbrella
    # still works when only some test executables exist.
    set(_run_disabled_perf_cmds "")
    if(TARGET test_simd_ops)
        # V.41: AddPerformance no longer carries the DISABLED_ prefix — it is
        # now a regression-guard ceiling test that runs as part of the normal
        # suite.  Keep the umbrella entry pointing at it (no
        # --gtest_also_run_disabled_tests needed) for parity with the other
        # benchmark-style tests below.
        list(APPEND _run_disabled_perf_cmds
            COMMAND $<TARGET_FILE:test_simd_ops>
                    --gtest_also_run_disabled_tests
                    --gtest_filter=*AddPerformance*:*DISABLED_BaselineRegressionCheck*)
    endif()
    if(TARGET test_performance_regression)
        list(APPEND _run_disabled_perf_cmds
            COMMAND $<TARGET_FILE:test_performance_regression>
                    --gtest_also_run_disabled_tests
                    --gtest_filter=*DISABLED_BaselineRegressionCheck_MatMul512*)
    endif()
    if(TARGET test_roberta)
        list(APPEND _run_disabled_perf_cmds
            COMMAND $<TARGET_FILE:test_roberta>
                    --gtest_also_run_disabled_tests
                    --gtest_filter=*DISABLED_BenchmarkInference*)
    endif()
    if(TARGET test_electra)
        list(APPEND _run_disabled_perf_cmds
            COMMAND $<TARGET_FILE:test_electra>
                    --gtest_also_run_disabled_tests
                    --gtest_filter=*DISABLED_BenchmarkInference*)
    endif()
    if(TARGET test_async_ops)
        list(APPEND _run_disabled_perf_cmds
            COMMAND $<TARGET_FILE:test_async_ops>
                    --gtest_also_run_disabled_tests
                    --gtest_filter=*DISABLED_PerformanceBenchmark*)
    endif()
    if(TARGET test_vulkan_complete_ops)
        list(APPEND _run_disabled_perf_cmds
            COMMAND $<TARGET_FILE:test_vulkan_complete_ops>
                    --gtest_also_run_disabled_tests
                    --gtest_filter=*DISABLED_BenchmarkLargeConv2d*:*DISABLED_BenchmarkLargeMatmul*)
    endif()

    add_custom_target(run_disabled_perf_benchmarks
        DEPENDS ${_tenzor_bench_deps}
        ${_run_disabled_perf_cmds}
        COMMENT "Running disabled-by-default benchmark/regression gtests"
        VERBATIM
    )
else()
    message(STATUS "tenzor_benchmarks umbrella: test_simd_ops / test_performance_regression "
                   "not built (TENZOR_BUILD_TESTS=OFF?); skipping.")
endif()
