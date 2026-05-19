# Phase 13 / Group A — JIT MLIR pipeline (text-based, IREE-driven).
# Gated by TENZOR_USE_MLIR_JIT. Per the 2026-05-19 amendment, Tenzor emits MLIR
# as text and links only against IREE Compiler (embedding API) + Runtime C API.

find_package(OpenSSL REQUIRED)

add_library(tenzor_jit_mlir OBJECT
    mlir_text_emit.cpp
    iree_compile.cpp
    iree_paths.cpp
    iree_runtime.cpp
    iree_customcalls.cpp
    lowering.cpp
)
target_link_libraries(tenzor_jit_mlir PUBLIC
    iree_compiler_API_SharedImpl   # embedding_api.h
    iree_runtime_runtime           # api.h
    OpenSSL::Crypto                # SHA-256 for cache key
)
target_include_directories(tenzor_jit_mlir PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_features(tenzor_jit_mlir PUBLIC cxx_std_23)

# Propagate the CMake-found IREE binary paths into the source so the
# runtime discovery chain in iree_paths.cpp can prefer them over $PATH.
if(IREE_COMPILE_EXECUTABLE)
    target_compile_definitions(tenzor_jit_mlir PRIVATE
        TENZOR_IREE_COMPILE_PATH="${IREE_COMPILE_EXECUTABLE}")
endif()
if(IREE_RUN_MODULE_EXECUTABLE)
    target_compile_definitions(tenzor_jit_mlir PRIVATE
        TENZOR_IREE_RUN_MODULE_PATH="${IREE_RUN_MODULE_EXECUTABLE}")
endif()

# OBJECT library is spliced into the shared tenzor_core; PIC required.
set_target_properties(tenzor_jit_mlir PROPERTIES POSITION_INDEPENDENT_CODE ON)
