# Core library sources
set(TENZOR_CORE_SOURCES
    core/tensor.cpp
    core/storage.cpp
    core/device.cpp
    core/dtype.cpp
    core/shape.cpp
    core/init.cpp
    backend/backend.cpp
    backend/loader.cpp
    backend/registry.cpp
    backend/dispatch.cpp
    ops/creation.cpp
    ops/math.cpp
    ops/indexing.cpp
    ops/reduction.cpp
    ops/transform.cpp
    ops/fused_ops.cpp
    autograd/variable.cpp
    autograd/graph.cpp
    autograd/function.cpp
    autograd/engine.cpp
    autograd/ops.cpp
    autograd/checkpoint.cpp
    nn/module.cpp
    nn/serialize.cpp
    nn/checkpoint.cpp
    nn/layers/linear.cpp
    nn/layers/conv.cpp
    nn/layers/batchnorm.cpp
    nn/layers/dropout.cpp
    nn/layers/pooling.cpp
    nn/layers/normalization.cpp
    nn/layers/flatten.cpp
    nn/layers/embedding.cpp
    nn/layers/attention.cpp
    nn/layers/transformer.cpp
    nn/layers/rnn.cpp
    nn/layers/lstm.cpp
    nn/layers/gru.cpp
    nn/activations/activations.cpp
    nn/loss/losses.cpp
    nn/loss/losses_advanced.cpp
    nn/optim/optimizer.cpp
    nn/optim/sgd.cpp
    nn/optim/adam.cpp
    nn/optim/rmsprop.cpp
    nn/optim/adagrad.cpp
    nn/optim/adadelta.cpp
    nn/optim/scheduler.cpp
    nn/optim/scheduler_advanced.cpp
    nn/amp/grad_scaler.cpp
    nn/parallel/data_parallel.cpp
    parallel/threadpool.cpp
    data/dataloader.cpp
    utils/logging.cpp
    utils/error.cpp
    utils/config.cpp
    utils/benchmark.cpp
    backends/cpu/simd.cpp
    backends/cpu/kernels/math_simd.cpp
    backends/cpu/kernels/activations_simd.cpp
)

# Create core library
add_library(tenzor_core SHARED ${TENZOR_CORE_SOURCES})

# Set target properties
set_target_properties(tenzor_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "${CMAKE_SOURCE_DIR}/include/tenzor/tenzor.hpp"
)

# Include directories
target_include_directories(tenzor_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(tenzor_core PUBLIC
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(tenzor_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Add compiler definitions
target_compile_definitions(tenzor_core PRIVATE
    TENZOR_VERSION="${PROJECT_VERSION}"
)

# Build backends
add_subdirectory(backends)
