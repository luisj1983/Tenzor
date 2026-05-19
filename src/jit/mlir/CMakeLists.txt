# Phase 13 / Group A — JIT MLIR pipeline (text-based, IREE-driven).
# Gated by TENZOR_USE_MLIR_JIT. Per the 2026-05-19 amendment, Tenzor emits MLIR
# as text and links only against IREE Compiler (embedding API) + Runtime C API.

add_library(tenzor_jit_mlir OBJECT
    _stub.cpp
    mlir_text_emit.cpp
)
target_link_libraries(tenzor_jit_mlir PUBLIC
    iree_compiler_API_SharedImpl   # embedding_api.h
    iree_runtime_runtime           # api.h
)
target_include_directories(tenzor_jit_mlir PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_features(tenzor_jit_mlir PUBLIC cxx_std_23)
