#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstddef>
#include <cstdint>

// Forward declarations for Metal types
#ifdef __OBJC__
@class MTLDevice;
@class MTLCommandQueue;
@class MTLBuffer;
@class MTLLibrary;
@class MTLComputePipelineState;
@class MPSMatrixMultiplication;
@class MPSCNNConvolution;
#else
typedef void MTLDevice;
typedef void MTLCommandQueue;
typedef void MTLBuffer;
typedef void MTLLibrary;
typedef void MTLComputePipelineState;
typedef void MPSMatrixMultiplication;
typedef void MPSCNNConvolution;
#endif

namespace tenzor {
namespace backend {
namespace metal {

enum class DataType {
    Float32,
    Float16,
    Int32,
    Int8
};

struct MetalBuffer {
    MTLBuffer* buffer;
    size_t size;
    DataType dtype;
};

struct KernelParams {
    std::vector<void*> buffers;
    std::vector<size_t> buffer_sizes;
    std::vector<int> int_params;
    std::vector<float> float_params;
};

class MetalBackend {
public:
    MetalBackend();
    ~MetalBackend();

    // Initialization
    bool initialize();
    void cleanup();

    // Device management
    std::string getDeviceName() const;
    size_t getMaxMemory() const;
    size_t getUsedMemory() const;

    // Memory management
    void* allocate(size_t size);
    void deallocate(void* ptr);
    void memcpy_h2d(void* dst, const void* src, size_t size);
    void memcpy_d2h(void* dst, const void* src, size_t size);
    void memcpy_d2d(void* dst, const void* src, size_t size);
    void memset(void* ptr, int value, size_t size);

    // Synchronization
    void synchronize();

    // Kernel execution
    void executeKernel(const std::string& kernel_name,
                      const KernelParams& params,
                      uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                      uint32_t block_x, uint32_t block_y, uint32_t block_z);

    // Matrix operations (MPS-based)
    void matmul(void* a, void* b, void* c,
               int m, int n, int k,
               bool transpose_a, bool transpose_b,
               float alpha, float beta,
               DataType dtype);

    // Convolution (MPS-based)
    void conv2d(void* input, void* weight, void* bias, void* output,
               int batch, int in_channels, int out_channels,
               int in_height, int in_width,
               int kernel_h, int kernel_w,
               int stride_h, int stride_w,
               int pad_h, int pad_w,
               int dilation_h, int dilation_w,
               DataType dtype);

    // Pooling operations
    void maxpool2d(void* input, void* output, void* indices,
                  int batch, int channels, int in_h, int in_w,
                  int kernel_h, int kernel_w,
                  int stride_h, int stride_w,
                  int pad_h, int pad_w,
                  DataType dtype);

    void avgpool2d(void* input, void* output,
                  int batch, int channels, int in_h, int in_w,
                  int kernel_h, int kernel_w,
                  int stride_h, int stride_w,
                  int pad_h, int pad_w,
                  DataType dtype);

    // Batch normalization
    void batchnorm(void* input, void* output,
                  void* gamma, void* beta,
                  void* running_mean, void* running_var,
                  int batch, int channels, int height, int width,
                  float epsilon, bool training,
                  DataType dtype);

    // Activation functions
    void relu(void* input, void* output, size_t size, DataType dtype);
    void gelu(void* input, void* output, size_t size, DataType dtype);
    void sigmoid(void* input, void* output, size_t size, DataType dtype);
    void tanh(void* input, void* output, size_t size, DataType dtype);
    void leaky_relu(void* input, void* output, size_t size, float alpha, DataType dtype);
    void softmax(void* input, void* output, int batch, int dim, DataType dtype);

    // Element-wise operations
    void add(void* a, void* b, void* c, size_t size, DataType dtype);
    void sub(void* a, void* b, void* c, size_t size, DataType dtype);
    void mul(void* a, void* b, void* c, size_t size, DataType dtype);
    void div(void* a, void* b, void* c, size_t size, DataType dtype);
    void pow(void* input, void* output, float exponent, size_t size, DataType dtype);
    void sqrt(void* input, void* output, size_t size, DataType dtype);
    void exp(void* input, void* output, size_t size, DataType dtype);
    void log(void* input, void* output, size_t size, DataType dtype);

    // Reduction operations
    void sum(void* input, void* output, size_t size, DataType dtype);
    void mean(void* input, void* output, size_t size, DataType dtype);
    void max(void* input, void* output, size_t size, DataType dtype);
    void min(void* input, void* output, size_t size, DataType dtype);

    void reduce_sum_axis(void* input, void* output,
                        const int* shape, int ndim, int axis,
                        DataType dtype);

    // Transform operations
    void transpose(void* input, void* output,
                  const int* shape, const int* perm, int ndim,
                  DataType dtype);

    void reshape(void* input, void* output,
                const int* old_shape, const int* new_shape, int ndim,
                DataType dtype);

    // Indexing operations
    void gather(void* input, void* indices, void* output,
               int batch, int dim, int index_count,
               DataType dtype);

    void scatter(void* input, void* indices, void* output,
                int batch, int dim, int index_count,
                DataType dtype);

    // Utility functions
    MTLCommandQueue* getCommandQueue() { return command_queue_; }
    MTLDevice* getDevice() { return device_; }

private:
    MTLDevice* device_;
    MTLCommandQueue* command_queue_;
    MTLLibrary* library_;

    // Pipeline state cache
    std::unordered_map<std::string, MTLComputePipelineState*> pipeline_cache_;

    // MPS objects cache
    MPSMatrixMultiplication* mps_matmul_fp32_;
    MPSMatrixMultiplication* mps_matmul_fp16_;

    // Helper functions
    MTLComputePipelineState* getPipelineState(const std::string& kernel_name);
    void loadLibrary();

    // Memory tracking
    size_t used_memory_;
    std::unordered_map<void*, size_t> allocation_map_;
};

} // namespace metal
} // namespace backend
} // namespace tenzor
