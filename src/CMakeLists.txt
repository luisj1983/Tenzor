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
    ops/advanced.cpp
    ops/vision.cpp
    ops/detection.cpp
    autograd/variable.cpp
    autograd/graph.cpp
    autograd/function.cpp
    autograd/engine.cpp
    autograd/ops.cpp
    autograd/checkpoint.cpp
    autograd/gradcheck.cpp
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
    nn/layers/vision.cpp
    nn/layers/mobilenet.cpp
    nn/layers/segmentation.cpp
    nn/detection/anchors.cpp
    nn/detection/roi_ops.cpp
    nn/detection/rpn.cpp
    nn/detection/roi_head.cpp
    nn/detection/mask_head.cpp
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
    nn/amp/autocast.cpp
    nn/parallel/data_parallel.cpp
    # nn/parallel/distributed_data_parallel.cpp  # Requires NCCL - disabled for now
    nn/quantization/quantize.cpp
    nn/quantization/observer.cpp
    nn/quantization/fake_quantize.cpp
    nn/quantization/qconfig.cpp
    nn/quantization/quantized_layers.cpp
    quantization/quantize_api.cpp
    parallel/threadpool.cpp
    data/dataloader.cpp
    models/hub.cpp
    models/resnet.cpp
    models/bert.cpp
    models/roberta.cpp
    models/albert.cpp
    models/electra.cpp
    models/t5.cpp
    models/gpt.cpp
    models/vgg.cpp
    models/alexnet.cpp
    models/googlenet.cpp
    models/efficientnet.cpp
    models/vit.cpp
    models/mobilenet.cpp
    models/convnext.cpp
    # models/swin_transformer.cpp  # TODO: Fix API compatibility issues
    models/unet.cpp
    models/deeplabv3plus.cpp
    models/yolo.cpp
    models/faster_rcnn.cpp
    models/mask_rcnn.cpp
    utils/logging.cpp
    utils/error.cpp
    utils/config.cpp
    utils/benchmark.cpp
    utils/tensorboard.cpp
    backends/cpu/simd.cpp
    backends/cpu/kernels/math_simd.cpp
    backends/cpu/kernels/activations_simd.cpp
    backends/cpu/kernels/quantization/quantized_linear.cpp
    backends/cpu/kernels/quantization/quantized_conv2d.cpp
    onnx/exporter.cpp
    onnx/importer.cpp
    jit/tracer.cpp
    jit/graph.cpp
    jit/compiler.cpp
    jit/serialization.cpp
    nn/compression/pruning.cpp
    nn/compression/distillation.cpp
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

# Find required dependencies for ModelHub
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)

# Link dependencies
target_link_libraries(tenzor_core PUBLIC
    Threads::Threads
    ${CMAKE_DL_LIBS}
    CURL::libcurl
    OpenSSL::SSL
    OpenSSL::Crypto
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(tenzor_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Add compiler definitions
target_compile_definitions(tenzor_core PRIVATE
    TENZOR_VERSION="${PROJECT_VERSION}"
)

# Add CUDA definition and includes if CUDA backend is enabled
if(TENZOR_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(tenzor_core PRIVATE
        TENZOR_USE_CUDA
    )
    target_include_directories(tenzor_core PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(tenzor_core PRIVATE
        CUDA::cudart
    )
endif()

# Build backends
add_subdirectory(backends)
