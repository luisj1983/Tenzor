# Phase 13 / Group A — MLIR JIT pipeline build wiring
# Only entered when TENZOR_USE_MLIR_JIT=ON (gated by parent).

# TableGen wiring (op definitions land in 1A.4 / 1A.6).
include_directories(${MLIR_INCLUDE_DIRS} ${STABLEHLO_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Object library — flatter dependency graph for the JIT.
add_library(tenzor_jit_mlir OBJECT
    _stub.cpp
)
target_link_libraries(tenzor_jit_mlir PUBLIC
    MLIRIR
    MLIRSupport
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    StablehloOps
    iree_runtime_runtime
)
target_include_directories(tenzor_jit_mlir PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_features(tenzor_jit_mlir PUBLIC cxx_std_23)
