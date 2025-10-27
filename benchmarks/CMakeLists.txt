# Benchmarks for Tenzor
cmake_minimum_required(VERSION 3.25)

# Benchmark executables
set(BENCHMARK_TARGETS
    benchmark_ops
    benchmark_convolutions
    benchmark_memory
    bench_fused_ops
)

# benchmark_ops - Matrix and tensor operations
add_executable(benchmark_ops benchmark_ops.cpp)
target_link_libraries(benchmark_ops PRIVATE tenzor_core)
target_include_directories(benchmark_ops PRIVATE ${CMAKE_SOURCE_DIR}/include)

# benchmark_convolutions - Convolution operations
add_executable(benchmark_convolutions benchmark_convolutions.cpp)
target_link_libraries(benchmark_convolutions PRIVATE tenzor_core)
target_include_directories(benchmark_convolutions PRIVATE ${CMAKE_SOURCE_DIR}/include)

# benchmark_memory - Memory operations
add_executable(benchmark_memory benchmark_memory.cpp)
target_link_libraries(benchmark_memory PRIVATE tenzor_core)
target_include_directories(benchmark_memory PRIVATE ${CMAKE_SOURCE_DIR}/include)

# bench_fused_ops - Fused operations (existing)
add_executable(bench_fused_ops bench_fused_ops.cpp)
target_link_libraries(bench_fused_ops PRIVATE tenzor_core)
target_include_directories(bench_fused_ops PRIVATE ${CMAKE_SOURCE_DIR}/include)

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
    COMMAND echo "Running Tenzor Benchmark Suite..."
    COMMAND echo "=================================="
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
    COMMAND echo "4. Running Fused Operations Benchmark..."
    COMMAND $<TARGET_FILE:bench_fused_ops> || true
    COMMAND echo ""
    COMMAND echo "=================================="
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

add_custom_target(run_fused_benchmark
    COMMAND $<TARGET_FILE:bench_fused_ops>
    DEPENDS bench_fused_ops
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running fused operations benchmark"
)

message(STATUS "Configured ${CMAKE_CURRENT_LIST_DIR} with ${BENCHMARK_TARGETS}")
message(STATUS "Use 'make run_benchmarks' to run all benchmarks")
