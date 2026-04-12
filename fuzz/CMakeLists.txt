# Fuzzing targets using libFuzzer (Clang only)
#
# Build: cmake -B build -G Ninja -DTENZOR_FUZZ=ON -DCMAKE_CXX_COMPILER=clang++
# Run:   ./build/fuzz/fuzz_deserialize -max_total_time=600

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(WARNING "Fuzzing requires Clang with libFuzzer support. Skipping fuzz targets.")
    return()
endif()

# Common flags for fuzzing
set(FUZZ_COMPILE_FLAGS -fsanitize=fuzzer-no-link,address -fno-omit-frame-pointer)
set(FUZZ_LINK_FLAGS -fsanitize=fuzzer,address)

# Helper function to create fuzz targets
function(add_fuzz_target name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE tenzor_core)
    target_compile_options(${name} PRIVATE ${FUZZ_COMPILE_FLAGS})
    target_link_options(${name} PRIVATE ${FUZZ_LINK_FLAGS})
    target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/include)
endfunction()

# Fuzz the native Tenzor deserialization path
add_fuzz_target(fuzz_deserialize fuzz_deserialize.cpp)

# Fuzz the SafeTensors deserialization path
add_fuzz_target(fuzz_safetensors fuzz_safetensors.cpp)

# Fuzz tensor creation with untrusted shapes
add_fuzz_target(fuzz_tensor_creation fuzz_tensor_creation.cpp)

# Fuzz reshape with untrusted dimensions
add_fuzz_target(fuzz_reshape fuzz_reshape.cpp)

# Fuzz autograd backward pass with random computation graphs
add_fuzz_target(fuzz_autograd_backward fuzz_autograd_backward.cpp)

# Fuzz matmul with edge-case shapes
add_fuzz_target(fuzz_matmul_shapes fuzz_matmul_shapes.cpp)

# Fuzz type promotion in binary operations
add_fuzz_target(fuzz_type_promotion fuzz_type_promotion.cpp)

# Fuzz the ONNX importer (hand-rolled protobuf parser — high-value target)
add_fuzz_target(fuzz_onnx_importer fuzz_onnx_importer.cpp)
