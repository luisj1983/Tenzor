# Core library sources
set(TENZOR_CORE_SOURCES
    core/tensor.cpp
    core/storage.cpp
    core/device.cpp
    core/dtype.cpp
    core/cuda_config_stubs.cpp
    core/jit_hooks.cpp
    core/dlpack.cpp
    core/shape.cpp
    core/init.cpp
    core/caching_allocator.cpp
    core/numa.cpp
    core/memory_manager.cpp
    core/transfer_engine.cpp
    core/pinned_allocator.cpp
    core/offload_engine.cpp
    core/device_guard.cpp
    core/generator.cpp
    core/named_tensor.cpp
    core/masked_tensor.cpp
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
    ops/philox_dropout.cpp   # F13/F22-followup: Philox-keyed dropout mask
    ops/math.cpp
    ops/foreach.cpp
    ops/type_promotion.cpp
    ops/fp8_scaling.cpp
    ops/indexing.cpp
    ops/reduction.cpp
    ops/transform.cpp
    ops/fused_ops.cpp
    ops/fusion_optimizer.cpp
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
    autograd/function_new_ops.cpp
    autograd/function_attention.cpp
    autograd/function_distributed.cpp
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
    autograd/chrome_trace.cpp
    nn/module.cpp
    nn/functional.cpp
    nn/serialize.cpp
    nn/safetensors.cpp
    nn/pytorch_loader.cpp
    nn/checkpoint.cpp
    nn/layers/linear.cpp
    nn/layers/lazy_linear.cpp
    nn/layers/lazy_conv.cpp
    nn/layers/conv.cpp
    nn/layers/batchnorm.cpp
    nn/layers/sync_batchnorm.cpp
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
    nn/layers/flex_attention.cpp
    nn/layers/transformer.cpp
    nn/layers/rnn.cpp
    nn/layers/lstm.cpp
    nn/layers/gru.cpp
    nn/layers/vision.cpp
    nn/layers/mobilenet.cpp
    nn/layers/segmentation.cpp
    nn/layers/hrm.cpp
    nn/layers/moe.cpp
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
    nn/optim/rprop.cpp
    nn/optim/asgd.cpp
    nn/optim/adam.cpp
    nn/optim/radam.cpp
    nn/optim/nadam.cpp
    nn/optim/adamax.cpp
    nn/optim/adam_atan2.cpp
    nn/optim/rmsprop.cpp
    nn/optim/adagrad.cpp
    nn/optim/adadelta.cpp
    nn/optim/lamb.cpp
    nn/optim/sparse_adam.cpp
    nn/optim/lion.cpp
    nn/optim/lbfgs.cpp
    nn/optim/zero_optimizer.cpp
    nn/optim/gradient_utils.cpp
    nn/optim/sam.cpp
    nn/optim/swa.cpp
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
    nn/metrics.cpp
    nn/init.cpp
    nn/parallel/data_parallel.cpp
    nn/quantization/quantize.cpp
    nn/quantization/observer.cpp
    nn/quantization/fake_quantize.cpp
    nn/quantization/qconfig.cpp
    nn/quantization/quantized_layers.cpp
    nn/quantization/gptq.cpp
    nn/quantization/awq.cpp
    quantization/quantize_api.cpp
    quantization/module_conversion.cpp
    utils/threading/threadpool.cpp
    data/dataset.cpp
    data/dataloader.cpp
    data/sampler.cpp
    data/transforms.cpp
    data/datasets/mnist.cpp
    data/datasets/cifar10.cpp
    data/datasets/imagenet.cpp
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
    utils/log.cpp        # D.1: spdlog facade — tenzor::utils::logger()
    utils/error.cpp
    utils/config.cpp
    utils/benchmark.cpp
    utils/tensorboard.cpp
    utils/memory_profiler.cpp
    utils/monitor.cpp
    backends/cpu/kernels/quantization/quantized_linear.cpp
    backends/cpu/kernels/quantization/quantized_conv2d.cpp
    onnx/exporter.cpp
    onnx/importer.cpp
    onnx/graph_module.cpp
    io/image.cpp
    io/torch_pickle.cpp     # H2-followup: native PyTorch .pth loader
    jit/tracer.cpp
    jit/graph.cpp
    jit/compiler.cpp
    jit/compiled_module.cpp
    jit/memory_planner.cpp
    jit/autotune.cpp
    jit/serialization.cpp
    jit/codegen.cpp
    jit/fusion_cost_model.cpp
    jit/pattern_matcher.cpp
    jit/extended_codegen.cpp
    jit/tracing_interceptor.cpp
    jit/compile.cpp
    jit/control_flow.cpp
    jit/script.cpp
    jit/symbolic_shape_inference.cpp
    nn/compression/pruning.cpp
    nn/compression/distillation.cpp
    distributed/distributed.cpp
    distributed/pipeline_parallel.cpp
    distributed/nccl_backend.cpp
    distributed/gloo_backend.cpp
    distributed/mpi_backend.cpp
    distributed/rpc/function_registry.cpp
    distributed/rpc/rpc_agent.cpp
    distributed/rpc/rref.cpp
    distributed/rpc/rpc.cpp
    serving/server.cpp
    serving/worker_pool.cpp
    lite/runtime.cpp
    lite/lite_graph.cpp
    lite/model_format.cpp
    lite/memory_planner.cpp
    lite/tensor_bridge.cpp
    lite/dispatch_bridge.cpp
    lite/exporter.cpp
    distributed/elastic/rendezvous.cpp
    distributed/elastic/health_monitor.cpp
    distributed/elastic/elastic_trainer.cpp
    distributed/elastic/state_sync.cpp
    distributed/process_group.cpp
    distributed/ddp.cpp
    distributed/fsdp.cpp
    distributed/tensor_parallel.cpp
    distributed/launch.cpp
    distributed/gradient_compression.cpp
    distributed/device_mesh.cpp
    distributed/dtensor.cpp
    distributed/fsdp2.cpp
    distributed/dist_checkpoint.cpp
    distributed/sequence_parallel.cpp
    sparse/sparse_tensor.cpp
    sparse/sparse_ops.cpp
    nested/nested_tensor.cpp
    nested/nested_ops.cpp
    autograd/nested_autograd_ops.cpp
    lazy/lazy_tensor.cpp
    export/export.cpp
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
# NOTE: CMake's target_precompile_headers injects #pragma clang system_header
# into the generated PCH wrapper, which produces -Wunknown-pragmas on GCC.
# Use GCC's native -include mechanism instead to get the same effect cleanly.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(tenzor_core PRIVATE
        "SHELL:-include ${CMAKE_CURRENT_SOURCE_DIR}/pch.hpp"
    )
else()
    target_precompile_headers(tenzor_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/pch.hpp
    )
endif()

# Include directories
target_include_directories(tenzor_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(tenzor_core PUBLIC
    Threads::Threads
    ${CMAKE_DL_LIBS}
    spdlog::spdlog  # D.1: structured logging across distributed / io / serving
)

# ============================================================================
# ONNX protobuf bindings
# ============================================================================
# The ONNX importer / exporter used to hand-roll protobuf wire-format
# decoding, which was non-canonical (incompatible with onnxruntime /
# netron) and a security risk on untrusted inputs. We now generate real
# protobuf bindings from proto/onnx.proto and link libprotobuf. If
# protobuf isn't found, the ONNX module falls back to a non-functional
# stub so the core library still builds.
find_package(Protobuf QUIET)
if(Protobuf_FOUND)
    find_program(TENZOR_PROTOC protoc REQUIRED)

    set(ONNX_PROTO_SRC_DIR ${CMAKE_SOURCE_DIR}/proto)
    set(ONNX_PROTO_GEN_DIR ${CMAKE_BINARY_DIR}/generated/onnx_proto)
    file(MAKE_DIRECTORY ${ONNX_PROTO_GEN_DIR})

    set(ONNX_PROTO_FILE ${ONNX_PROTO_SRC_DIR}/onnx.proto)
    set(ONNX_PROTO_GEN_SRCS ${ONNX_PROTO_GEN_DIR}/onnx.pb.cc)
    set(ONNX_PROTO_GEN_HDRS ${ONNX_PROTO_GEN_DIR}/onnx.pb.h)

    add_custom_command(
        OUTPUT ${ONNX_PROTO_GEN_SRCS} ${ONNX_PROTO_GEN_HDRS}
        COMMAND ${TENZOR_PROTOC}
            --proto_path=${ONNX_PROTO_SRC_DIR}
            --cpp_out=${ONNX_PROTO_GEN_DIR}
            ${ONNX_PROTO_FILE}
        DEPENDS ${ONNX_PROTO_FILE}
        COMMENT "Generating ONNX protobuf bindings from proto/onnx.proto"
    )

    target_sources(tenzor_core PRIVATE ${ONNX_PROTO_GEN_SRCS})
    target_include_directories(tenzor_core PRIVATE ${ONNX_PROTO_GEN_DIR})
    target_link_libraries(tenzor_core PRIVATE protobuf::libprotobuf)
    target_compile_definitions(tenzor_core PRIVATE TENZOR_HAS_ONNX_PROTOBUF)
    message(STATUS "ONNX: Using generated protobuf bindings (libprotobuf ${Protobuf_VERSION})")
else()
    message(WARNING "ONNX: protobuf not found; ONNX import/export will throw at runtime. Install libprotobuf-dev to enable.")
endif()

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

# Optional serving dependencies (cpp-httplib + gRPC)
if(TENZOR_BUILD_SERVING)
    # cpp-httplib (header-only HTTP server)
    include(FetchContent)
    FetchContent_Declare(
        httplib
        GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
        GIT_TAG v0.18.7
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(httplib)
    target_link_libraries(tenzor_core PRIVATE httplib::httplib)
    target_compile_definitions(tenzor_core PRIVATE TENZOR_HAS_HTTPLIB)

    # gRPC (system package)
    find_package(gRPC CONFIG QUIET)
    if(gRPC_FOUND)
        # Generate protobuf/gRPC stubs from proto file
        find_program(PROTOC protoc REQUIRED)
        find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

        set(PROTO_SRC_DIR ${CMAKE_SOURCE_DIR}/proto)
        set(PROTO_GEN_DIR ${CMAKE_BINARY_DIR}/generated/proto)
        file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

        set(PROTO_FILE ${PROTO_SRC_DIR}/tenzor_serving.proto)
        set(PROTO_GEN_SRCS ${PROTO_GEN_DIR}/tenzor_serving.pb.cc ${PROTO_GEN_DIR}/tenzor_serving.grpc.pb.cc)
        set(PROTO_GEN_HDRS ${PROTO_GEN_DIR}/tenzor_serving.pb.h ${PROTO_GEN_DIR}/tenzor_serving.grpc.pb.h)

        add_custom_command(
            OUTPUT ${PROTO_GEN_SRCS} ${PROTO_GEN_HDRS}
            COMMAND ${PROTOC}
                --proto_path=${PROTO_SRC_DIR}
                --cpp_out=${PROTO_GEN_DIR}
                --grpc_out=${PROTO_GEN_DIR}
                --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
                ${PROTO_FILE}
            DEPENDS ${PROTO_FILE}
            COMMENT "Generating gRPC stubs for tenzor_serving.proto"
        )

        target_sources(tenzor_core PRIVATE
            serving/grpc_server.cpp
            ${PROTO_GEN_SRCS}
        )
        target_include_directories(tenzor_core PRIVATE ${PROTO_GEN_DIR})
        target_link_libraries(tenzor_core PRIVATE gRPC::grpc++ protobuf::libprotobuf)
        target_compile_definitions(tenzor_core PRIVATE TENZOR_HAS_GRPC)
        message(STATUS "Serving: HTTP + gRPC enabled")
    else()
        message(STATUS "Serving: HTTP only (gRPC not found)")
    endif()
else()
    message(STATUS "Serving: Disabled (set -DTENZOR_BUILD_SERVING=ON to enable)")
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(tenzor_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Unused parameter warnings are not suppressed — fix them with [[maybe_unused]] where needed

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
    option(TENZOR_REQUIRE_NCCL "Require NCCL library (fail if not found)" OFF)
    find_package(NCCL)
    if(NCCL_FOUND)
        target_link_libraries(tenzor_core PRIVATE NCCL::NCCL)
        target_compile_definitions(tenzor_core PRIVATE TENZOR_HAS_NCCL)
    else()
        if(TENZOR_REQUIRE_NCCL)
            message(FATAL_ERROR "NCCL library required but not found. "
                    "Set NCCL_ROOT or install NCCL.")
        else()
            message(WARNING "NCCL not found — distributed GPU training disabled. "
                    "Set -DTENZOR_REQUIRE_NCCL=ON to make this an error.")
        endif()
    endif()
endif()

# MPI backend (Phase 1.5 — OpenMPI / MPICH / Intel MPI).
#
# Opt-in via TENZOR_BUILD_MPI=ON. When enabled, we look for an MPI
# implementation on the system; the wrapper library mpi_backend.cpp
# guards its body on TENZOR_HAS_MPI, so an unsuccessful find just
# leaves the .cpp as an empty translation unit. TENZOR_REQUIRE_MPI
# turns "not found" into a hard error for CI.
option(TENZOR_BUILD_MPI "Enable MPI distributed backend" OFF)
option(TENZOR_REQUIRE_MPI "Fail the configure step if MPI is not found" OFF)
if(TENZOR_BUILD_MPI)
    find_package(MPI COMPONENTS CXX)
    if(MPI_CXX_FOUND)
        target_link_libraries(tenzor_core PRIVATE MPI::MPI_CXX)
        target_compile_definitions(tenzor_core PRIVATE TENZOR_HAS_MPI)
        message(STATUS "MPI backend: enabled (${MPI_CXX_VERSION_MAJOR}.${MPI_CXX_VERSION_MINOR})")
    else()
        if(TENZOR_REQUIRE_MPI)
            message(FATAL_ERROR "MPI library required (-DTENZOR_REQUIRE_MPI=ON) but not found. "
                    "Install OpenMPI / MPICH / Intel MPI and ensure mpicxx is on PATH.")
        else()
            message(WARNING "TENZOR_BUILD_MPI=ON but MPI not found — MPI backend will "
                    "compile to an empty stub. Install MPI or unset TENZOR_BUILD_MPI.")
        endif()
    endif()
endif()

# Add ROCm/HIP definition and includes if ROCm backend is enabled
if(TENZOR_BUILD_ROCM)
    # Ensure HIP_PLATFORM is set before find_package(hip)
    if(NOT DEFINED HIP_PLATFORM AND NOT DEFINED ENV{HIP_PLATFORM})
        set(HIP_PLATFORM "amd" CACHE STRING "HIP platform (amd or nvidia)")
        set(ENV{HIP_PLATFORM} "amd")
    endif()
    find_package(hip QUIET CONFIG PATHS ${_ROCM_ROOT})
    if(hip_FOUND)
        target_compile_definitions(tenzor_core PRIVATE
            TENZOR_HAS_ROCM
        )
        target_include_directories(tenzor_core PRIVATE
            ${_ROCM_ROOT}/include
        )
        target_link_libraries(tenzor_core PRIVATE
            hip::host
        )
    endif()
endif()

# Build backends
add_subdirectory(backends)

# Phase 13 MLIR JIT pipeline (gated on TENZOR_USE_MLIR_JIT)
if(TENZOR_USE_MLIR_JIT)
    add_subdirectory(jit/mlir)
endif()
