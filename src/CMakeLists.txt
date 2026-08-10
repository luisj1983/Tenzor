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
    core/memory_manager.cpp
    core/transfer_engine.cpp
    core/pinned_allocator.cpp
    core/offload_engine.cpp
    core/gpu_stream_ops.cpp
    core/device_guard.cpp
    core/generator.cpp
    backend/backend.cpp
    backend/loader.cpp
    backend/registry.cpp
    backend/dispatch.cpp
    backend/simd_dispatch.cpp
    backend/runtime_simd.cpp
    backend/dispatch_table.cpp
    backend/omp_thresholds.cpp
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
    autograd/jvp_dispatch.cpp
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
    PUBLIC_HEADER "${PROJECT_SOURCE_DIR}/include/tenzor/tenzor.hpp"
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
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# G.12: vendored nlohmann/json header path (third_party/nlohmann/json.hpp).
# PRIVATE — only the serving module currently uses it; not part of the public ABI.
target_link_libraries(tenzor_core PRIVATE tenzor_nlohmann_json)

# zlib: enables reading DEFLATE-compressed entries in PyTorch .pt/.pth ZIP
# archives (the default torch.save output). Optional — without it, only stored
# (uncompressed) entries can be read. Standard on every Linux/macOS toolchain.
find_package(ZLIB QUIET)
if(ZLIB_FOUND)
    target_link_libraries(tenzor_core PRIVATE ZLIB::ZLIB)
    target_compile_definitions(tenzor_core PRIVATE TENZOR_HAS_ZLIB)
    message(STATUS "tenzor_core: zlib found — compressed .pt/.pth entries supported")
else()
    message(STATUS "tenzor_core: zlib NOT found — only stored (uncompressed) .pt entries supported")
endif()

# Link dependencies
target_link_libraries(tenzor_core PUBLIC
    Threads::Threads
    ${CMAKE_DL_LIBS}
)
# spdlog is an implementation detail (used only behind utils/log.cpp), so link
# it PRIVATE. Linking it PUBLIC leaked spdlog::spdlog into the exported
# TenzorTargets while the installed TenzorConfig never find_dependency(spdlog)'d
# it (and the FetchContent'd spdlog isn't installed), so any downstream C++
# consumer's find_package(Tenzor) failed to configure.
target_link_libraries(tenzor_core PRIVATE
    spdlog::spdlog  # D.1: structured logging across distributed / io / serving
)
# fmt is propagated PUBLIC because the public header utils/logging.hpp inlines
# ::fmt::format (consumers of the TENZOR_LOG* macros need to link fmt). Guarded:
# with FetchContent's bundled header-only fmt there is no fmt::fmt target and the
# inline format needs no link. find_dependency(fmt) is emitted into TenzorConfig
# (fmt ships a CMake package config, so downstream find_package(Tenzor) resolves).
if(TARGET fmt::fmt)
    target_link_libraries(tenzor_core PUBLIC fmt::fmt)
endif()

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
    # Prefer the version-matched protobuf::protoc imported target over a
    # free-floating `protoc` on PATH, which can be a different protobuf version
    # than the one we link and emit incompatible generated headers.
    if(TARGET protobuf::protoc)
        set(TENZOR_PROTOC protobuf::protoc)
    else()
        find_program(TENZOR_PROTOC protoc REQUIRED)
    endif()

    set(ONNX_PROTO_SRC_DIR ${PROJECT_SOURCE_DIR}/proto)
    set(ONNX_PROTO_GEN_DIR ${PROJECT_BINARY_DIR}/generated/onnx_proto)
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
    # Fail-loud stubs: model factories reference ModelHub unconditionally, so
    # a hub-less core must still define the symbols (see models/hub_stub.cpp)
    # or libtenzor_core.so fails at load time with undefined references.
    target_sources(tenzor_core PRIVATE models/hub_stub.cpp)
    message(STATUS "Model Hub: Disabled (ModelHub entry points throw)")
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
        # Generate protobuf/gRPC stubs from proto file. Prefer the
        # version-matched protobuf::protoc target (see the ONNX block above).
        if(TARGET protobuf::protoc)
            set(PROTOC protobuf::protoc)
        else()
            find_program(PROTOC protoc REQUIRED)
        endif()
        find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

        set(PROTO_SRC_DIR ${PROJECT_SOURCE_DIR}/proto)
        set(PROTO_GEN_DIR ${PROJECT_BINARY_DIR}/generated/proto)
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
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindMKL.cmake OPTIONAL RESULT_VARIABLE _MKL_MODULE_FOUND)
if(_MKL_MODULE_FOUND AND MKL_FOUND)
    target_link_libraries(tenzor_core PRIVATE ${MKL_LIBRARIES})
    target_include_directories(tenzor_core PRIVATE ${MKL_INCLUDE_DIRS})
    # NOTE: must NOT define MKL_ILP64 here — FindMKL links the LP64 interface
    # (mkl_intel_lp64, 32-bit MKL_INT) and the cblas_* call sites pass plain int.
    # Defining MKL_ILP64 made <mkl_lapacke.h> declare lapack_int as 64-bit while
    # the linked LP64 library writes 32-bit pivots, silently corrupting every
    # LAPACKE pivot array (getrf/sytrf → det/lu/ldl/inv/solve/svd/qr/eigh). The
    # interface width is selected by FindMKL (MKL_INTERFACE_FULL=intel_lp64).
    target_compile_definitions(tenzor_core PRIVATE TENZOR_USE_MKL)
else()
    if(TENZOR_REQUIRE_MKL)
        message(FATAL_ERROR "MKL required (TENZOR_REQUIRE_MKL=ON) but not found. "
                "Set MKLROOT or install Intel oneAPI MKL.")
    else()
        # Try system LAPACKE as fallback
        find_package(LAPACK QUIET)
        # check_include_file() (CheckIncludeFile) compiles a .c probe, which
        # needs the C language enabled via project(LANGUAGES ...) — this
        # project only enables CXX (+ HIP conditionally), so that call fails
        # try_compile() outright ("Unknown extension '.c'") instead of just
        # reporting not-found. Use the CXX-language variant, which probes
        # with a .cxx file and works with the languages this project actually
        # enables.
        include(CheckIncludeFileCXX)
        check_include_file_cxx(lapacke.h HAVE_LAPACKE_H)
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
    # NVRTC ships no headers of its own: a runtime-generated kernel that does
    # `#include <cuda_fp16.h>` (Float16/BFloat16 fusions) fails with "could not
    # open source file" unless we hand NVRTC the CUDA include dir. Bake it in so
    # codegen.cpp can pass it as --include-path.
    list(GET CUDAToolkit_INCLUDE_DIRS 0 _TENZOR_CUDA_INC0)
    target_compile_definitions(tenzor_core PRIVATE
        TENZOR_NVRTC_INCLUDE="${_TENZOR_CUDA_INC0}"
    )

    # Link NVRTC and CUDA Driver API for runtime kernel codegen
    if(TARGET CUDA::nvrtc)
        target_link_libraries(tenzor_core PRIVATE CUDA::nvrtc CUDA::cuda_driver)
        message(STATUS "NVRTC found: runtime kernel codegen enabled")
    else()
        message(WARNING "NVRTC not found: runtime kernel codegen will use CPU fallback")
    endif()

    # NOTE: NCCL is intentionally NOT linked into the host tenzor_core library.
    # The NCCL collective implementation lives in tenzor_backend_cuda (and RCCL
    # in tenzor_backend_rocm); the distributed factory resolves the active GPU
    # backend DSO via the BackendLoader registry and asks it for a
    # CommunicationBackend. Linking libnccl into tenzor_core would pull
    # <nccl.h> into host TUs and clash with <cuda_runtime.h> on a combined
    # CUDA+ROCm build (RCCL's nccl.h drags in hip/hip_runtime.h). See
    # src/backends/cuda/CMakeLists.txt for the NCCL find_package/link.
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

# Add ROCm/HIP definition and includes if ROCm backend is enabled.
# Gated on _ROCM_AVAILABLE (not just TENZOR_BUILD_ROCM) because this block
# sets LANGUAGE HIP on a source file — the HIP language is only enabled at
# project() time when the ROCm toolchain was detected. Running it without a
# detected toolchain trips "The language HIP was requested for compilation
# but was not enabled".
if(TENZOR_BUILD_ROCM AND _ROCM_AVAILABLE)
    # Ensure HIP_PLATFORM is set before find_package(hip)
    if(NOT DEFINED HIP_PLATFORM AND NOT DEFINED ENV{HIP_PLATFORM})
        set(HIP_PLATFORM "amd" CACHE STRING "HIP platform (amd or nvidia)")
        set(ENV{HIP_PLATFORM} "amd")
    endif()
    find_package(hip QUIET CONFIG PATHS ${_ROCM_ROOT})
    if(hip_FOUND)
        # HIPRTC include dir for runtime-generated kernels that pull in
        # <hip/hip_fp16.h> / <hip/hip_bf16.h> (Float16/BFloat16 fusions).
        #
        # Do NOT hardcode ${_ROCM_ROOT}/include: the HIP runtime headers move
        # between ROCm layouts (ROCm 6.x puts them under <root>/core/include;
        # older layouts used <root>/include, and on some installs the latter
        # does not exist at all). Derive the dir from the same hip::host
        # imported target find_package(hip) set up -- it carries exactly the
        # dir the host compiler uses to find <hip/hip_runtime_api.h>, which is
        # the same dir that holds <hip/hip_fp16.h>. Fall back to the discovered
        # core/include then include if the target property is unavailable.
        get_target_property(_HIP_HOST_INCS hip::host INTERFACE_INCLUDE_DIRECTORIES)
        set(_HIPRTC_INCLUDE_DIR "")
        foreach(_d ${_HIP_HOST_INCS})
            # Imported-target include props are plain paths on ROCm, but strip
            # any $<...> generator-expression wrapper defensively.
            string(REGEX REPLACE "\\$<[^:>]+:([^>]*)>" "\\1" _d "${_d}")
            string(REGEX REPLACE "\\$<.*>" "" _d "${_d}")
            if(EXISTS "${_d}")
                set(_HIPRTC_INCLUDE_DIR "${_d}")
                break()
            endif()
        endforeach()
        if(NOT _HIPRTC_INCLUDE_DIR)
            foreach(_d "${_ROCM_ROOT}/core/include" "${_ROCM_ROOT}/include")
                if(EXISTS "${_d}")
                    set(_HIPRTC_INCLUDE_DIR "${_d}")
                    break()
                endif()
            endforeach()
        endif()
        target_compile_definitions(tenzor_core PRIVATE
            TENZOR_HAS_ROCM
            TENZOR_HIPRTC_INCLUDE="${_HIPRTC_INCLUDE_DIR}"
        )
        # NOTE (FINDING 60, resolved): ddp.cpp/fsdp.cpp/zero_optimizer.cpp used
        # to guard their dedicated comm-stream/event overlap code on
        # TENZOR_USE_ROCM, compile-time-mutually-exclusive with
        # TENZOR_USE_CUDA -- which is defined tenzor_core-wide, so the ROCm
        # branch could never be reached on this combined CUDA+ROCm build no
        # matter what. Scoping TENZOR_USE_ROCM per-file wouldn't have helped
        # either (same #elif precedence issue) and would additionally have
        # collided <hip/hip_runtime.h> against <cuda_runtime.h> in the same TU
        # (~30 make_*N/dim3 redefinition errors, proved by actually trying it).
        # Fixed by moving stream/event creation into
        # include/tenzor/core/gpu_stream_ops.hpp, which dispatches at RUNTIME
        # on the tensors' actual Device::Type: CUDA calls go straight through,
        # ROCm calls route through the isolated rocm_transfer.hip.cpp TU below
        # via its opaque void*-based API. No TENZOR_USE_ROCM define needed
        # anywhere for these three files anymore.
        target_include_directories(tenzor_core PRIVATE
            ${_ROCM_ROOT}/include
        )
        target_link_libraries(tenzor_core PRIVATE
            hip::host
        )
        # Runtime GPU kernel codegen for ROCm (src/jit/codegen.cpp) compiles HIP
        # source at runtime via HIPRTC and loads it through the HIP driver. These
        # symbols live in libhiprtc / libamdhip64 (host API only — the TU includes
        # <hip/hip_runtime_api.h>, NOT the device header, so no vector-type clash
        # with CUDA). Link them so a combined CUDA+ROCm build can target ROCm GPUs.
        find_library(TENZOR_HIPRTC_LIB hiprtc HINTS ${_ROCM_ROOT}/lib "${_ROCM_ROOT}/core/lib" REQUIRED)
        find_library(TENZOR_AMDHIP_LIB amdhip64 HINTS ${_ROCM_ROOT}/lib "${_ROCM_ROOT}/core/lib" REQUIRED)
        target_link_libraries(tenzor_core PRIVATE
            ${TENZOR_HIPRTC_LIB}
            ${TENZOR_AMDHIP_LIB}
        )
        # Real CPU<->GPU DMA lives in this isolated HIP TU (compiled by hipcc)
        # so HIP's vector-type headers never collide with CUDA's in the host TUs
        # of tenzor_core. See rocm_transfer.hpp. This restores async ROCm offload
        # transfers that were previously compiled out (the #ifdef TENZOR_USE_ROCM
        # paths in transfer_engine.cpp could not be enabled in a CUDA+HIP build).
        set_source_files_properties(core/rocm_transfer.hip.cpp PROPERTIES LANGUAGE HIP)
        target_sources(tenzor_core PRIVATE core/rocm_transfer.hip.cpp)
        set(TENZOR_ROCM_TRANSFER_REAL TRUE)
    endif()
endif()

# When the real HIP transfer TU is not built, provide a no-op stub so the
# rocm_transfer:: symbols referenced unconditionally by transfer_engine.cpp
# (wait/is_ready/~TransferState) still resolve at link time.
if(NOT TENZOR_ROCM_TRANSFER_REAL)
    target_sources(tenzor_core PRIVATE core/rocm_transfer_stub.cpp)
endif()

# Build backends
add_subdirectory(backends)

# Phase 13 MLIR JIT pipeline (gated on TENZOR_USE_MLIR_JIT)
if(TENZOR_USE_MLIR_JIT)
    add_subdirectory(jit/mlir)
    # Splice the MLIR JIT object files directly into tenzor_core (the OBJECT
    # library itself is not in tenzor_core's install/export set, which CMake
    # forbids; using $<TARGET_OBJECTS:...> embeds the .o files without
    # exporting tenzor_jit_mlir as a separate target).
    target_sources(tenzor_core PRIVATE $<TARGET_OBJECTS:tenzor_jit_mlir>)
    # OpenSSL::Crypto is required by the IREE runtime's dependency chain here,
    # independent of TENZOR_BUILD_MODEL_HUB (which has its own find_package(OpenSSL)
    # above) — find it directly rather than relying on that unrelated block
    # having already run, which left this target undefined whenever ModelHub
    # was off but MLIR JIT was on.
    find_package(OpenSSL REQUIRED)
    target_link_libraries(tenzor_core PRIVATE
        iree_compiler_API_SharedImpl
        iree_runtime_runtime
        OpenSSL::Crypto
    )
    target_compile_definitions(tenzor_core PRIVATE TENZOR_HAS_MLIR_JIT)
endif()
