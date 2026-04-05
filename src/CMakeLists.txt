# Core library sources
set(TENZOR_CORE_SOURCES
    core/tensor.cpp
    core/storage.cpp
    core/device.cpp
    core/dtype.cpp
    core/shape.cpp
    core/init.cpp
    core/caching_allocator.cpp
    core/numa.cpp
    core/memory_manager.cpp
    core/transfer_engine.cpp
    core/pinned_allocator.cpp
    core/offload_engine.cpp
    core/device_guard.cpp
    backend/backend.cpp
    backend/loader.cpp
    backend/registry.cpp
    backend/dispatch.cpp
    backend/simd_dispatch.cpp
    backend/runtime_simd.cpp
    backend/dispatch_table.cpp
    backend/op_attributes.cpp
    backend/cuda_graph_stub.cpp
    ops/op_id.cpp
    ops/creation.cpp
    ops/math.cpp
    ops/type_promotion.cpp
    ops/fp8_scaling.cpp
    ops/indexing.cpp
    ops/reduction.cpp
    ops/transform.cpp
    ops/fused_ops.cpp
    ops/advanced.cpp
    ops/fft.cpp
    ops/windows.cpp
    ops/vision.cpp
    ops/detection.cpp
    ops/async_ops.cpp
    ops/linalg.cpp
    ops/custom_op.cpp
    autograd/variable.cpp
    autograd/graph.cpp
    autograd/function_base.cpp
    autograd/function_arithmetic.cpp
    autograd/function_elementwise.cpp
    autograd/function_reduction.cpp
    autograd/function_shape.cpp
    autograd/function_activations.cpp
    autograd/function_math.cpp
    autograd/function_reduction_ext.cpp
    autograd/function_shape_ext.cpp
    autograd/function_linalg.cpp
    autograd/function_fft.cpp
    autograd/function_sparse.cpp
    autograd/function_special.cpp
    autograd/function_vision.cpp
    autograd/engine.cpp
    autograd/ops.cpp
    autograd/checkpoint.cpp
    autograd/gradcheck.cpp
    autograd/graph_optimizer.cpp
    autograd/graph_viz.cpp
    autograd/dual.cpp
    autograd/jvp_rules.cpp
    autograd/functional.cpp
    autograd/vmap.cpp
    nn/module.cpp
    nn/functional.cpp
    nn/serialize.cpp
    nn/safetensors.cpp
    nn/pytorch_loader.cpp
    nn/checkpoint.cpp
    nn/layers/linear.cpp
    nn/layers/lazy_linear.cpp
    nn/layers/conv.cpp
    nn/layers/batchnorm.cpp
    nn/layers/dropout.cpp
    nn/layers/drop_path.cpp
    nn/layers/rope.cpp
    nn/layers/alibi.cpp
    nn/layers/pooling.cpp
    nn/layers/normalization.cpp
    nn/layers/flatten.cpp
    nn/layers/padding.cpp
    nn/layers/upsample.cpp
    nn/layers/embedding.cpp
    nn/layers/sparse_linear.cpp
    nn/layers/sparse_embedding.cpp
    nn/layers/attention.cpp
    nn/layers/gqa_attention.cpp
    nn/layers/transformer.cpp
    nn/layers/rnn.cpp
    nn/layers/lstm.cpp
    nn/layers/gru.cpp
    nn/layers/vision.cpp
    nn/layers/mobilenet.cpp
    nn/layers/segmentation.cpp
    nn/layers/hrm.cpp
    nn/detection/anchors.cpp
    nn/detection/roi_ops.cpp
    nn/detection/rpn.cpp
    nn/detection/roi_head.cpp
    nn/detection/mask_head.cpp
    nn/activations/activations.cpp
    nn/loss/losses.cpp
    nn/loss/losses_advanced.cpp
    nn/loss/contrastive.cpp
    nn/optim/optimizer.cpp
    nn/optim/sgd.cpp
    nn/optim/adam.cpp
    nn/optim/radam.cpp
    nn/optim/adam_atan2.cpp
    nn/optim/rmsprop.cpp
    nn/optim/adagrad.cpp
    nn/optim/adadelta.cpp
    nn/optim/lamb.cpp
    nn/optim/sparse_adam.cpp
    nn/optim/zero_optimizer.cpp
    nn/optim/gradient_utils.cpp
    nn/optim/scheduler.cpp
    nn/optim/scheduler_advanced.cpp
    nn/callbacks.cpp
    nn/training.cpp
    nn/mixed_precision.cpp
    nn/offload.cpp
    nn/amp/grad_scaler.cpp
    nn/amp/autocast.cpp
    nn/utils/clip_grad.cpp
    nn/utils/kv_cache.cpp
    nn/utils/grad_accumulation.cpp
    nn/utils/spectral_norm.cpp
    nn/utils/weight_norm.cpp
    nn/utils/parametrize.cpp
    nn/utils/rnn_utils.cpp
    nn/init.cpp
    nn/parallel/data_parallel.cpp
    # nn/parallel/distributed_data_parallel.cpp  # Disabled - use new distributed API from distributed/ instead
    nn/quantization/quantize.cpp
    nn/quantization/observer.cpp
    nn/quantization/fake_quantize.cpp
    nn/quantization/qconfig.cpp
    nn/quantization/quantized_layers.cpp
    quantization/quantize_api.cpp
    quantization/module_conversion.cpp
    parallel/threadpool.cpp
    data/dataloader.cpp
    data/sampler.cpp
    data/transforms.cpp
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
    models/swin_transformer.cpp
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
    utils/memory_profiler.cpp
    backends/cpu/kernels/quantization/quantized_linear.cpp
    backends/cpu/kernels/quantization/quantized_conv2d.cpp
    onnx/exporter.cpp
    onnx/importer.cpp
    onnx/graph_module.cpp
    jit/tracer.cpp
    jit/graph.cpp
    jit/compiler.cpp
    jit/compiled_module.cpp
    jit/memory_planner.cpp
    jit/autotune.cpp
    jit/serialization.cpp
    jit/codegen.cpp
    jit/fusion_cost_model.cpp
    nn/compression/pruning.cpp
    nn/compression/distillation.cpp
    distributed/distributed.cpp
    distributed/nccl_backend.cpp
    distributed/gloo_backend.cpp
    distributed/process_group.cpp
    distributed/ddp.cpp
    distributed/fsdp.cpp
    distributed/tensor_parallel.cpp
    distributed/launch.cpp
    distributed/gradient_compression.cpp
    sparse/sparse_tensor.cpp
    sparse/sparse_ops.cpp
)

# Create core library
add_library(tenzor_core SHARED ${TENZOR_CORE_SOURCES})

# Apply Tenzor-specific warning and optimization flags
target_compile_options(tenzor_core PRIVATE ${TENZOR_WARNING_FLAGS})

# Set target properties
set_target_properties(tenzor_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "${CMAKE_SOURCE_DIR}/include/tenzor/tenzor.hpp"
)

# Internal methods visible only to core + backends
target_compile_definitions(tenzor_core PRIVATE TENZOR_INTERNAL)

# Profiling annotations (NVTX for CUDA, ROCTX for ROCm)
if(TENZOR_ENABLE_PROFILING)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND AND TARGET CUDA::nvtx3)
        target_link_libraries(tenzor_core PRIVATE CUDA::nvtx3)
        target_compile_definitions(tenzor_core PRIVATE TENZOR_HAS_NVTX)
        message(STATUS "Profiling: NVTX3 enabled")
    else()
        find_library(ROCTX_LIBRARY roctx64 PATHS /opt/rocm/lib)
        if(ROCTX_LIBRARY)
            target_link_libraries(tenzor_core PRIVATE ${ROCTX_LIBRARY})
            target_include_directories(tenzor_core PRIVATE /opt/rocm/include)
            target_compile_definitions(tenzor_core PRIVATE TENZOR_HAS_ROCTX)
            message(STATUS "Profiling: ROCTX enabled")
        else()
            message(STATUS "Profiling: enabled but no NVTX or ROCTX found")
        endif()
    endif()
endif()

# Pre-compiled headers for faster compilation
target_precompile_headers(tenzor_core PRIVATE
    <vector>
    <memory>
    <string>
    <unordered_map>
    <cstdint>
    <stdexcept>
    <algorithm>
    <functional>
    <optional>
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

# Optional ModelHub dependencies (CURL + OpenSSL)
if(TENZOR_BUILD_MODEL_HUB)
    find_package(CURL REQUIRED)
    find_package(OpenSSL REQUIRED)
    target_sources(tenzor_core PRIVATE models/hub.cpp)
    target_link_libraries(tenzor_core PUBLIC
        CURL::libcurl
        OpenSSL::SSL
        OpenSSL::Crypto
    )
    target_compile_definitions(tenzor_core PRIVATE TENZOR_HAS_MODEL_HUB)
    message(STATUS "Model Hub: Enabled (CURL + OpenSSL)")
else()
    message(STATUS "Model Hub: Disabled")
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(tenzor_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Suppress unused parameter warnings for interface/callback functions
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tenzor_core PRIVATE -Wno-unused-parameter)
endif()

# Add compiler definitions
target_compile_definitions(tenzor_core PRIVATE
    TENZOR_VERSION="${PROJECT_VERSION}"
)

# Add MKL for linear algebra (linalg.cpp uses LAPACKE)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindMKL.cmake OPTIONAL RESULT_VARIABLE _MKL_MODULE_FOUND)
if(_MKL_MODULE_FOUND AND MKL_FOUND)
    target_link_libraries(tenzor_core PRIVATE ${MKL_LIBRARIES})
    target_include_directories(tenzor_core PRIVATE ${MKL_INCLUDE_DIRS})
    target_compile_definitions(tenzor_core PRIVATE TENZOR_USE_MKL MKL_ILP64)
else()
    if(TENZOR_REQUIRE_MKL)
        message(FATAL_ERROR "MKL required (TENZOR_REQUIRE_MKL=ON) but not found. "
                "Set MKLROOT or install Intel oneAPI MKL.")
    else()
        # Try system LAPACKE as fallback
        find_package(LAPACK QUIET)
        include(CheckIncludeFile)
        check_include_file(lapacke.h HAVE_LAPACKE_H)
        if(LAPACK_FOUND AND HAVE_LAPACKE_H)
            message(STATUS "MKL not found, using system LAPACKE for linalg operations")
            target_link_libraries(tenzor_core PRIVATE ${LAPACK_LIBRARIES} lapacke)
            target_compile_definitions(tenzor_core PRIVATE TENZOR_USE_LAPACKE)
        else()
            message(WARNING "Neither MKL nor LAPACKE found — CPU linalg operations "
                    "(det, inv, solve, svd, etc.) will be unavailable. "
                    "GPU backends can still dispatch linalg ops. "
                    "Install Intel MKL or liblapacke-dev for CPU support.")
        endif()
    endif()
endif()

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

    # Link NVRTC and CUDA Driver API for runtime kernel codegen
    if(TARGET CUDA::nvrtc)
        target_link_libraries(tenzor_core PRIVATE CUDA::nvrtc CUDA::cuda_driver)
        message(STATUS "NVRTC found: runtime kernel codegen enabled")
    else()
        message(WARNING "NVRTC not found: runtime kernel codegen will use CPU fallback")
    endif()

    # Link NCCL if available (uses cmake/FindNCCL.cmake)
    find_package(NCCL QUIET)
    if(NCCL_FOUND)
        target_link_libraries(tenzor_core PRIVATE NCCL::NCCL)
        target_compile_definitions(tenzor_core PRIVATE TENZOR_HAS_NCCL)
    else()
        message(WARNING "NCCL library not found, distributed GPU training will not be available")
    endif()
endif()

# Add ROCm/HIP definition and includes if ROCm backend is enabled
if(TENZOR_BUILD_ROCM)
    # Ensure HIP_PLATFORM is set before find_package(hip)
    if(NOT DEFINED HIP_PLATFORM AND NOT DEFINED ENV{HIP_PLATFORM})
        set(HIP_PLATFORM "amd" CACHE STRING "HIP platform (amd or nvidia)")
        set(ENV{HIP_PLATFORM} "amd")
    endif()
    find_package(hip QUIET CONFIG PATHS /opt/rocm)
    if(hip_FOUND)
        target_compile_definitions(tenzor_core PRIVATE
            TENZOR_HAS_ROCM
        )
        target_include_directories(tenzor_core PRIVATE
            /opt/rocm/include
        )
        target_link_libraries(tenzor_core PRIVATE
            hip::host
        )
    endif()
endif()

# Build backends
add_subdirectory(backends)
