# Manual performance harness comparing vendor (cuBLAS/cuDNN) vs custom CUDA
# kernels via the tenzor::cuda::force_custom_kernels() runtime knob.
#
# Built only when TENZOR_BUILD_CUDA is on (the harness calls cudaRuntime
# directly and exercises the CUDA backend). It is deliberately NOT registered
# with ctest: it is a slow, GPU-only A/B benchmark meant to be run by hand:
#   ./bin/bench_cublas_cudnn
find_package(CUDAToolkit REQUIRED)

add_executable(bench_cublas_cudnn bench_cublas_cudnn.cpp)
target_link_libraries(bench_cublas_cudnn PRIVATE
    tenzor_core
    tenzor_backend_cuda  # owns tenzor::cuda::force_custom_kernels() definitions
    GTest::gtest_main
    CUDA::cudart
)
target_include_directories(bench_cublas_cudnn PRIVATE ${CUDAToolkit_INCLUDE_DIRS})