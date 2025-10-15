# CUDA to HIP API Conversion Reference

## Quick Reference Guide

This document provides a comprehensive mapping of CUDA APIs to their HIP equivalents used in the ROCm backend implementation.

## Memory Management

### Allocation and Deallocation

| CUDA API | HIP API | Signature | Description |
|----------|---------|-----------|-------------|
| `cudaMalloc` | `hipMalloc` | `hipError_t hipMalloc(void** ptr, size_t size)` | Allocate device memory |
| `cudaFree` | `hipFree` | `hipError_t hipFree(void* ptr)` | Free device memory |
| `cudaMallocHost` | `hipHostMalloc` | `hipError_t hipHostMalloc(void** ptr, size_t size)` | Allocate pinned host memory |
| `cudaFreeHost` | `hipHostFree` | `hipError_t hipHostFree(void* ptr)` | Free pinned host memory |
| `cudaMallocManaged` | `hipMallocManaged` | `hipError_t hipMallocManaged(void** ptr, size_t size)` | Allocate unified memory |

### Memory Copy Operations

| CUDA API | HIP API | Signature |
|----------|---------|-----------|
| `cudaMemcpy` | `hipMemcpy` | `hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind)` |
| `cudaMemcpyAsync` | `hipMemcpyAsync` | `hipError_t hipMemcpyAsync(void* dst, const void* src, size_t size, hipMemcpyKind kind, hipStream_t stream)` |
| `cudaMemset` | `hipMemset` | `hipError_t hipMemset(void* ptr, int value, size_t size)` |
| `cudaMemsetAsync` | `hipMemsetAsync` | `hipError_t hipMemsetAsync(void* ptr, int value, size_t size, hipStream_t stream)` |

### Memory Copy Kinds

| CUDA | HIP | Description |
|------|-----|-------------|
| `cudaMemcpyHostToHost` | `hipMemcpyHostToHost` | CPU to CPU |
| `cudaMemcpyHostToDevice` | `hipMemcpyHostToDevice` | CPU to GPU |
| `cudaMemcpyDeviceToHost` | `hipMemcpyDeviceToHost` | GPU to CPU |
| `cudaMemcpyDeviceToDevice` | `hipMemcpyDeviceToDevice` | GPU to GPU |
| `cudaMemcpyDefault` | `hipMemcpyDefault` | Automatic direction detection |

## Device Management

### Device Control

| CUDA API | HIP API | Signature | Description |
|----------|---------|-----------|-------------|
| `cudaGetDeviceCount` | `hipGetDeviceCount` | `hipError_t hipGetDeviceCount(int* count)` | Get number of devices |
| `cudaSetDevice` | `hipSetDevice` | `hipError_t hipSetDevice(int device)` | Set active device |
| `cudaGetDevice` | `hipGetDevice` | `hipError_t hipGetDevice(int* device)` | Get current device |
| `cudaDeviceReset` | `hipDeviceReset` | `hipError_t hipDeviceReset()` | Reset device state |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` | `hipError_t hipDeviceSynchronize()` | Wait for device operations |

### Device Properties

| CUDA API | HIP API | Signature |
|----------|---------|-----------|
| `cudaGetDeviceProperties` | `hipGetDeviceProperties` | `hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int device)` |
| `cudaDeviceProp` | `hipDeviceProp_t` | Device properties structure |

### Device Properties Structure

| CUDA Field | HIP Field | Type | Description |
|------------|-----------|------|-------------|
| `name` | `name` | `char[256]` | Device name |
| `totalGlobalMem` | `totalGlobalMem` | `size_t` | Total global memory |
| `sharedMemPerBlock` | `sharedMemPerBlock` | `size_t` | Shared memory per block |
| `regsPerBlock` | `regsPerBlock` | `int` | Registers per block |
| `warpSize` | `warpSize` | `int` | Warp size |
| `maxThreadsPerBlock` | `maxThreadsPerBlock` | `int` | Max threads per block |
| `maxThreadsDim[3]` | `maxThreadsDim[3]` | `int[3]` | Max thread dimensions |
| `maxGridSize[3]` | `maxGridSize[3]` | `int[3]` | Max grid dimensions |
| `clockRate` | `clockRate` | `int` | Clock frequency (kHz) |
| `multiProcessorCount` | `multiProcessorCount` | `int` | Number of SMs/CUs |
| `major` | `major` | `int` | Compute capability major |
| `minor` | `minor` | `int` | Compute capability minor |

## Stream Management

### Stream Operations

| CUDA API | HIP API | Signature | Description |
|----------|---------|-----------|-------------|
| `cudaStream_t` | `hipStream_t` | Type | Stream handle type |
| `cudaStreamCreate` | `hipStreamCreate` | `hipError_t hipStreamCreate(hipStream_t* stream)` | Create stream |
| `cudaStreamDestroy` | `hipStreamDestroy` | `hipError_t hipStreamDestroy(hipStream_t stream)` | Destroy stream |
| `cudaStreamSynchronize` | `hipStreamSynchronize` | `hipError_t hipStreamSynchronize(hipStream_t stream)` | Wait for stream |
| `cudaStreamQuery` | `hipStreamQuery` | `hipError_t hipStreamQuery(hipStream_t stream)` | Check stream status |
| `cudaStreamWaitEvent` | `hipStreamWaitEvent` | `hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event)` | Stream waits for event |

### Stream Flags

| CUDA | HIP | Description |
|------|-----|-------------|
| `cudaStreamDefault` | `hipStreamDefault` | Default stream behavior |
| `cudaStreamNonBlocking` | `hipStreamNonBlocking` | Non-blocking stream |

## Event Management

| CUDA API | HIP API | Signature |
|----------|---------|-----------|
| `cudaEvent_t` | `hipEvent_t` | Event handle type |
| `cudaEventCreate` | `hipEventCreate` | `hipError_t hipEventCreate(hipEvent_t* event)` |
| `cudaEventDestroy` | `hipEventDestroy` | `hipError_t hipEventDestroy(hipEvent_t event)` |
| `cudaEventRecord` | `hipEventRecord` | `hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream)` |
| `cudaEventSynchronize` | `hipEventSynchronize` | `hipError_t hipEventSynchronize(hipEvent_t event)` |
| `cudaEventElapsedTime` | `hipEventElapsedTime` | `hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t end)` |

## Error Handling

### Error Types and Functions

| CUDA API | HIP API | Signature | Description |
|----------|---------|-----------|-------------|
| `cudaError_t` | `hipError_t` | Type | Error code type |
| `cudaSuccess` | `hipSuccess` | Constant | Success code |
| `cudaGetLastError` | `hipGetLastError` | `hipError_t hipGetLastError()` | Get last error |
| `cudaPeekAtLastError` | `hipPeekAtLastError` | `hipError_t hipPeekAtLastError()` | Peek at last error |
| `cudaGetErrorString` | `hipGetErrorString` | `const char* hipGetErrorString(hipError_t error)` | Get error message |
| `cudaGetErrorName` | `hipGetErrorName` | `const char* hipGetErrorName(hipError_t error)` | Get error name |

### Common Error Codes

| CUDA | HIP | Description |
|------|-----|-------------|
| `cudaSuccess` | `hipSuccess` | No error |
| `cudaErrorMemoryAllocation` | `hipErrorMemoryAllocation` | Memory allocation failed |
| `cudaErrorInvalidValue` | `hipErrorInvalidValue` | Invalid parameter |
| `cudaErrorNoDevice` | `hipErrorNoDevice` | No GPU device found |
| `cudaErrorInvalidDevice` | `hipErrorInvalidDevice` | Invalid device ID |
| `cudaErrorLaunchFailure` | `hipErrorLaunchFailure` | Kernel launch failed |
| `cudaErrorOutOfMemory` | `hipErrorOutOfMemory` | Out of device memory |

## Kernel Launch

### Launch Configuration

| CUDA Syntax | HIP Syntax | Description |
|-------------|------------|-------------|
| `kernel<<<grid, block>>>` | `hipLaunchKernelGGL(kernel, grid, block, 0, 0, ...)` | Launch kernel |
| `kernel<<<grid, block, smem, stream>>>` | `hipLaunchKernelGGL(kernel, grid, block, smem, stream, ...)` | Launch with shared mem and stream |
| `dim3` | `dim3` | 3D dimensions |

### Thread Indexing

| CUDA | HIP | Description |
|------|-----|-------------|
| `threadIdx.x/y/z` | `hipThreadIdx_x/y/z` | Thread index |
| `blockIdx.x/y/z` | `hipBlockIdx_x/y/z` | Block index |
| `blockDim.x/y/z` | `hipBlockDim_x/y/z` | Block dimensions |
| `gridDim.x/y/z` | `hipGridDim_x/y/z` | Grid dimensions |

## Pointer Attributes

| CUDA API | HIP API | Signature |
|----------|---------|-----------|
| `cudaPointerGetAttributes` | `hipPointerGetAttributes` | `hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attr, const void* ptr)` |
| `cudaPointerAttributes` | `hipPointerAttribute_t` | Pointer attributes structure |

## Math Functions

All CUDA math functions have direct HIP equivalents with the same names:

| Function Type | Examples |
|---------------|----------|
| Basic | `sin`, `cos`, `tan`, `sqrt`, `pow`, `exp`, `log` |
| Intrinsics | `__sinf`, `__cosf`, `__expf`, `__logf` |
| Special | `erf`, `erfc`, `lgamma`, `tgamma` |

## Atomic Operations

| CUDA | HIP | Description |
|------|-----|-------------|
| `atomicAdd` | `atomicAdd` | Atomic addition |
| `atomicSub` | `atomicSub` | Atomic subtraction |
| `atomicExch` | `atomicExch` | Atomic exchange |
| `atomicMin` | `atomicMin` | Atomic minimum |
| `atomicMax` | `atomicMax` | Atomic maximum |
| `atomicCAS` | `atomicCAS` | Atomic compare-and-swap |

## Synchronization

| CUDA | HIP | Description |
|------|-----|-------------|
| `__syncthreads()` | `__syncthreads()` | Block-level barrier |
| `__threadfence()` | `__threadfence()` | Global memory fence |
| `__threadfence_block()` | `__threadfence_block()` | Block memory fence |

## Memory Fence

| CUDA | HIP | Description |
|------|-----|-------------|
| `__threadfence()` | `__threadfence()` | Device memory fence |
| `__threadfence_block()` | `__threadfence_block()` | Block memory fence |
| `__threadfence_system()` | `__threadfence_system()` | System memory fence |

## Compiler Directives

| CUDA | HIP | Description |
|------|-----|-------------|
| `__device__` | `__device__` | Device function |
| `__global__` | `__global__` | Kernel function |
| `__host__` | `__host__` | Host function |
| `__constant__` | `__constant__` | Constant memory |
| `__shared__` | `__shared__` | Shared memory |

## Library Functions

### CUBLAS → rocBLAS

| CUDA Library | HIP/ROCm Library | Description |
|--------------|------------------|-------------|
| cuBLAS | rocBLAS | Basic Linear Algebra Subprograms |
| cuDNN | MIOpen | Deep Neural Network library |
| cuRAND | hipRAND | Random number generation |
| cuFFT | hipFFT / rocFFT | Fast Fourier Transform |
| cuSPARSE | hipSPARSE / rocSPARSE | Sparse linear algebra |
| cuSOLVER | rocSOLVER | Dense linear solvers |

## Implementation in ROCm Backend

### Example: Memory Allocation

**CUDA Backend:**
```cpp
void* ptr = nullptr;
cudaSetDevice(device_id);
cudaError_t err = cudaMalloc(&ptr, bytes);
if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
}
```

**ROCm Backend:**
```cpp
void* ptr = nullptr;
check_hip_error(hipSetDevice(device_id), "hipSetDevice in allocate");
hipError_t err = hipMalloc(&ptr, bytes);
if (err != hipSuccess) {
    throw std::runtime_error(hipGetErrorString(err));
}
```

### Example: Stream Management

**CUDA Backend:**
```cpp
cudaStream_t stream;
cudaSetDevice(device_id);
cudaStreamCreate(&stream);
```

**ROCm Backend:**
```cpp
hipStream_t stream;
check_hip_error(hipSetDevice(device_id), "hipSetDevice in create_stream");
check_hip_error(hipStreamCreate(&stream), "hipStreamCreate");
```

## Conversion Best Practices

1. **Use HIP instead of CUDA**: Replace all CUDA includes with HIP equivalents
   ```cpp
   // CUDA
   #include <cuda_runtime.h>

   // HIP
   #include <hip/hip_runtime.h>
   ```

2. **Check for HIP-specific differences**: Some APIs have subtle differences
   - Pointer attributes structure fields may differ
   - Stream creation flags may vary

3. **Test on actual hardware**: Always validate on AMD GPUs
   - Different architectures (CDNA, RDNA) have different capabilities
   - Use `hipGetDeviceProperties` to query capabilities

4. **Use HIP helper macros** when available
   ```cpp
   #define HIP_CHECK(call) \
       do { \
           hipError_t err = call; \
           if (err != hipSuccess) { \
               throw std::runtime_error(hipGetErrorString(err)); \
           } \
       } while(0)
   ```

5. **Maintain API parity**: Keep CUDA and HIP backends in sync
   - Use the same function signatures
   - Support the same operation set
   - Provide equivalent error handling

## Performance Considerations

### Memory Bandwidth

- AMD GPUs (especially MI200/MI300 series) have higher memory bandwidth
- Optimize for memory-bound kernels

### Compute Units

- AMD uses "Compute Units" (CUs) vs NVIDIA's "Streaming Multiprocessors" (SMs)
- Different wavefront/warp sizes (64 vs 32)
- Adjust occupancy calculations accordingly

### Shared Memory

- Different shared memory sizes per CU/SM
- Query `sharedMemPerBlock` for portable code

## References

- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/user_guide/programming_manual.html)
- [HIP API Reference](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/index.html)
- [CUDA to HIP Porting Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)
- [Hipify Tool](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/) - Automated CUDA to HIP conversion

## Automated Conversion Tool

ROCm provides `hipify-perl` for automated conversion:

```bash
# Convert single file
hipify-perl cuda_kernel.cu > hip_kernel.hip

# Convert entire directory
find . -name "*.cu" -exec hipify-perl {} > {}.hip \;
```

**Note**: Manual review is still required after automated conversion!
