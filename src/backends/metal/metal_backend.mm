#import "metal_backend.hpp"
#import "metal_utils.hpp"
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>

namespace tenzor {
namespace backend {
namespace metal {

MetalBackend::MetalBackend()
    : device_(nullptr)
    , command_queue_(nullptr)
    , library_(nullptr)
    , mps_matmul_fp32_(nullptr)
    , mps_matmul_fp16_(nullptr)
    , used_memory_(0) {
}

MetalBackend::~MetalBackend() {
    cleanup();
}

bool MetalBackend::initialize() {
    @autoreleasepool {
        // Get default Metal device
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            return false;
        }

        // Create command queue
        command_queue_ = [device_ newCommandQueue];
        if (!command_queue_) {
            return false;
        }

        // Load default library
        loadLibrary();

        return true;
    }
}

void MetalBackend::cleanup() {
    @autoreleasepool {
        // Release pipeline cache
        for (auto& pair : pipeline_cache_) {
            if (pair.second) {
                [pair.second release];
            }
        }
        pipeline_cache_.clear();

        // Release MPS objects
        if (mps_matmul_fp32_) {
            [mps_matmul_fp32_ release];
            mps_matmul_fp32_ = nullptr;
        }
        if (mps_matmul_fp16_) {
            [mps_matmul_fp16_ release];
            mps_matmul_fp16_ = nullptr;
        }

        // Release library
        if (library_) {
            [library_ release];
            library_ = nullptr;
        }

        // Release command queue
        if (command_queue_) {
            [command_queue_ release];
            command_queue_ = nullptr;
        }

        // Release device
        if (device_) {
            [device_ release];
            device_ = nullptr;
        }
    }
}

std::string MetalBackend::getDeviceName() const {
    @autoreleasepool {
        if (!device_) return "No device";
        return std::string([[device_ name] UTF8String]);
    }
}

size_t MetalBackend::getMaxMemory() const {
    if (!device_) return 0;
    return [device_ recommendedMaxWorkingSetSize];
}

size_t MetalBackend::getUsedMemory() const {
    return used_memory_;
}

void* MetalBackend::allocate(size_t size) {
    @autoreleasepool {
        id<MTLBuffer> buffer = createSharedBuffer(device_, size);
        used_memory_ += size;
        allocation_map_[buffer] = size;
        return (__bridge void*)buffer;
    }
}

void MetalBackend::deallocate(void* ptr) {
    @autoreleasepool {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)ptr;
        auto it = allocation_map_.find(ptr);
        if (it != allocation_map_.end()) {
            used_memory_ -= it->second;
            allocation_map_.erase(it);
        }
        [buffer release];
    }
}

void MetalBackend::memcpy_h2d(void* dst, const void* src, size_t size) {
    @autoreleasepool {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)dst;
        memcpy([buffer contents], src, size);
    }
}

void MetalBackend::memcpy_d2h(void* dst, const void* src, size_t size) {
    @autoreleasepool {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)src;
        memcpy(dst, [buffer contents], size);
    }
}

void MetalBackend::memcpy_d2d(void* dst, const void* src, size_t size) {
    @autoreleasepool {
        id<MTLBuffer> dst_buffer = (__bridge id<MTLBuffer>)dst;
        id<MTLBuffer> src_buffer = (__bridge id<MTLBuffer>)src;

        id<MTLCommandBuffer> commandBuffer = createCommandBuffer(command_queue_);
        copyBuffer(commandBuffer, src_buffer, dst_buffer, size);
        executeAndWait(commandBuffer);
    }
}

void MetalBackend::memset(void* ptr, int value, size_t size) {
    @autoreleasepool {
        id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)ptr;
        ::memset([buffer contents], value, size);
    }
}

void MetalBackend::synchronize() {
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = createCommandBuffer(command_queue_);
        executeAndWait(commandBuffer);
    }
}

void MetalBackend::loadLibrary() {
    @autoreleasepool {
        NSError* error = nil;

        // Try to load compiled metallib first
        NSString* libraryPath = @"default.metallib";
        library_ = [device_ newLibraryWithFile:libraryPath error:&error];

        // If that fails, load default library
        if (!library_) {
            library_ = [device_ newDefaultLibrary];
        }

        METAL_CHECK(library_ != nil, "Failed to load Metal library");
    }
}

MTLComputePipelineState* MetalBackend::getPipelineState(const std::string& kernel_name) {
    @autoreleasepool {
        auto it = pipeline_cache_.find(kernel_name);
        if (it != pipeline_cache_.end()) {
            return it->second;
        }

        id<MTLComputePipelineState> pipeline = createComputePipelineState(device_, library_, kernel_name);
        pipeline_cache_[kernel_name] = pipeline;
        return pipeline;
    }
}

void MetalBackend::executeKernel(const std::string& kernel_name,
                                const KernelParams& params,
                                uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                                uint32_t block_x, uint32_t block_y, uint32_t block_z) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = getPipelineState(kernel_name);
        id<MTLCommandBuffer> commandBuffer = createCommandBuffer(command_queue_);
        id<MTLComputeCommandEncoder> encoder = createComputeEncoder(commandBuffer);

        [encoder setComputePipelineState:pipeline];

        // Set buffer arguments
        for (size_t i = 0; i < params.buffers.size(); ++i) {
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)params.buffers[i];
            [encoder setBuffer:buffer offset:0 atIndex:i];
        }

        // Set integer parameters
        size_t param_index = params.buffers.size();
        for (int param : params.int_params) {
            [encoder setBytes:&param length:sizeof(int) atIndex:param_index++];
        }

        // Set float parameters
        for (float param : params.float_params) {
            [encoder setBytes:&param length:sizeof(float) atIndex:param_index++];
        }

        MTLSize threadGroupSize = MTLSizeMake(block_x, block_y, block_z);
        MTLSize gridSize = MTLSizeMake(grid_x, grid_y, grid_z);

        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];

        executeAndWait(commandBuffer);
    }
}

void MetalBackend::matmul(void* a, void* b, void* c,
                         int m, int n, int k,
                         bool transpose_a, bool transpose_b,
                         float alpha, float beta,
                         DataType dtype) {
    @autoreleasepool {
        id<MTLBuffer> buffer_a = (__bridge id<MTLBuffer>)a;
        id<MTLBuffer> buffer_b = (__bridge id<MTLBuffer>)b;
        id<MTLBuffer> buffer_c = (__bridge id<MTLBuffer>)c;

        id<MTLCommandBuffer> commandBuffer = createCommandBuffer(command_queue_);

        if (dtype == DataType::Float32) {
            // Create MPS matrix descriptors
            MPSMatrixDescriptor* desc_a = [MPSMatrixDescriptor
                matrixDescriptorWithRows:transpose_a ? k : m
                columns:transpose_a ? m : k
                rowBytes:k * sizeof(float)
                dataType:MPSDataTypeFloat32];

            MPSMatrixDescriptor* desc_b = [MPSMatrixDescriptor
                matrixDescriptorWithRows:transpose_b ? n : k
                columns:transpose_b ? k : n
                rowBytes:n * sizeof(float)
                dataType:MPSDataTypeFloat32];

            MPSMatrixDescriptor* desc_c = [MPSMatrixDescriptor
                matrixDescriptorWithRows:m
                columns:n
                rowBytes:n * sizeof(float)
                dataType:MPSDataTypeFloat32];

            MPSMatrix* matrix_a = [[MPSMatrix alloc] initWithBuffer:buffer_a descriptor:desc_a];
            MPSMatrix* matrix_b = [[MPSMatrix alloc] initWithBuffer:buffer_b descriptor:desc_b];
            MPSMatrix* matrix_c = [[MPSMatrix alloc] initWithBuffer:buffer_c descriptor:desc_c];

            // Create matrix multiplication kernel
            MPSMatrixMultiplication* matmul_kernel = [[MPSMatrixMultiplication alloc]
                initWithDevice:device_
                transposeLeft:transpose_a
                transposeRight:transpose_b
                resultRows:m
                resultColumns:n
                interiorColumns:k
                alpha:alpha
                beta:beta];

            [matmul_kernel encodeToCommandBuffer:commandBuffer
                leftMatrix:matrix_a
                rightMatrix:matrix_b
                resultMatrix:matrix_c];

            [matrix_a release];
            [matrix_b release];
            [matrix_c release];
            [matmul_kernel release];
        } else if (dtype == DataType::Float16) {
            // Similar setup for FP16
            MPSMatrixDescriptor* desc_a = [MPSMatrixDescriptor
                matrixDescriptorWithRows:transpose_a ? k : m
                columns:transpose_a ? m : k
                rowBytes:k * sizeof(uint16_t)
                dataType:MPSDataTypeFloat16];

            MPSMatrixDescriptor* desc_b = [MPSMatrixDescriptor
                matrixDescriptorWithRows:transpose_b ? n : k
                columns:transpose_b ? k : n
                rowBytes:n * sizeof(uint16_t)
                dataType:MPSDataTypeFloat16];

            MPSMatrixDescriptor* desc_c = [MPSMatrixDescriptor
                matrixDescriptorWithRows:m
                columns:n
                rowBytes:n * sizeof(uint16_t)
                dataType:MPSDataTypeFloat16];

            MPSMatrix* matrix_a = [[MPSMatrix alloc] initWithBuffer:buffer_a descriptor:desc_a];
            MPSMatrix* matrix_b = [[MPSMatrix alloc] initWithBuffer:buffer_b descriptor:desc_b];
            MPSMatrix* matrix_c = [[MPSMatrix alloc] initWithBuffer:buffer_c descriptor:desc_c];

            MPSMatrixMultiplication* matmul_kernel = [[MPSMatrixMultiplication alloc]
                initWithDevice:device_
                transposeLeft:transpose_a
                transposeRight:transpose_b
                resultRows:m
                resultColumns:n
                interiorColumns:k
                alpha:alpha
                beta:beta];

            [matmul_kernel encodeToCommandBuffer:commandBuffer
                leftMatrix:matrix_a
                rightMatrix:matrix_b
                resultMatrix:matrix_c];

            [matrix_a release];
            [matrix_b release];
            [matrix_c release];
            [matmul_kernel release];
        }

        executeAndWait(commandBuffer);
    }
}

void MetalBackend::conv2d(void* input, void* weight, void* bias, void* output,
                         int batch, int in_channels, int out_channels,
                         int in_height, int in_width,
                         int kernel_h, int kernel_w,
                         int stride_h, int stride_w,
                         int pad_h, int pad_w,
                         int dilation_h, int dilation_w,
                         DataType dtype) {
    @autoreleasepool {
        id<MTLBuffer> input_buffer = (__bridge id<MTLBuffer>)input;
        id<MTLBuffer> weight_buffer = (__bridge id<MTLBuffer>)weight;
        id<MTLBuffer> bias_buffer = bias ? (__bridge id<MTLBuffer>)bias : nil;
        id<MTLBuffer> output_buffer = (__bridge id<MTLBuffer>)output;

        id<MTLCommandBuffer> commandBuffer = createCommandBuffer(command_queue_);

        // Calculate output dimensions
        int out_height = (in_height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        int out_width = (in_width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

        // Create MPS convolution descriptor
        MPSCNNConvolutionDescriptor* conv_desc = [MPSCNNConvolutionDescriptor
            cnnConvolutionDescriptorWithKernelWidth:kernel_w
            kernelHeight:kernel_h
            inputFeatureChannels:in_channels
            outputFeatureChannels:out_channels];

        conv_desc.strideInPixelsX = stride_w;
        conv_desc.strideInPixelsY = stride_h;

        // Create image descriptors
        MPSImageDescriptor* input_desc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
            width:in_width
            height:in_height
            featureChannels:in_channels];

        MPSImageDescriptor* output_desc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
            width:out_width
            height:out_height
            featureChannels:out_channels];

        // Note: Full MPS convolution requires proper weight formatting
        // This is a simplified version - production code would need proper MPSCNNConvolutionDataSource

        executeAndWait(commandBuffer);
    }
}

// Activation functions
void MetalBackend::relu(void* input, void* output, size_t size, DataType dtype) {
    KernelParams params;
    params.buffers = {input, output};
    params.int_params = {static_cast<int>(size)};

    uint32_t threads = 256;
    uint32_t blocks = (size + threads - 1) / threads;

    executeKernel("relu_kernel", params, blocks, 1, 1, threads, 1, 1);
}

void MetalBackend::gelu(void* input, void* output, size_t size, DataType dtype) {
    KernelParams params;
    params.buffers = {input, output};
    params.int_params = {static_cast<int>(size)};

    uint32_t threads = 256;
    uint32_t blocks = (size + threads - 1) / threads;

    executeKernel("gelu_kernel", params, blocks, 1, 1, threads, 1, 1);
}

void MetalBackend::sigmoid(void* input, void* output, size_t size, DataType dtype) {
    KernelParams params;
    params.buffers = {input, output};
    params.int_params = {static_cast<int>(size)};

    uint32_t threads = 256;
    uint32_t blocks = (size + threads - 1) / threads;

    executeKernel("sigmoid_kernel", params, blocks, 1, 1, threads, 1, 1);
}

void MetalBackend::tanh(void* input, void* output, size_t size, DataType dtype) {
    KernelParams params;
    params.buffers = {input, output};
    params.int_params = {static_cast<int>(size)};

    uint32_t threads = 256;
    uint32_t blocks = (size + threads - 1) / threads;

    executeKernel("tanh_kernel", params, blocks, 1, 1, threads, 1, 1);
}

void MetalBackend::leaky_relu(void* input, void* output, size_t size, float alpha, DataType dtype) {
    KernelParams params;
    params.buffers = {input, output};
    params.int_params = {static_cast<int>(size)};
    params.float_params = {alpha};

    uint32_t threads = 256;
    uint32_t blocks = (size + threads - 1) / threads;

    executeKernel("leaky_relu_kernel", params, blocks, 1, 1, threads, 1, 1);
}

void MetalBackend::softmax(void* input, void* output, int batch, int dim, DataType dtype) {
    KernelParams params;
    params.buffers = {input, output};
    params.int_params = {batch, dim};

    executeKernel("softmax_kernel", params, batch, 1, 1, 256, 1, 1);
}

// Element-wise operations
void MetalBackend::add(void* a, void* b, void* c, size_t size, DataType dtype) {
    KernelParams params;
    params.buffers = {a, b, c};
    params.int_params = {static_cast<int>(size)};

    uint32_t threads = 256;
    uint32_t blocks = (size + threads - 1) / threads;

    executeKernel("add_kernel", params, blocks, 1, 1, threads, 1, 1);
}

void MetalBackend::sub(void* a, void* b, void* c, size_t size, DataType dtype) {
    KernelParams params;
    params.buffers = {a, b, c};
    params.int_params = {static_cast<int>(size)};

    uint32_t threads = 256;
    uint32_t blocks = (size + threads - 1) / threads;

    executeKernel("sub_kernel", params, blocks, 1, 1, threads, 1, 1);
}

void MetalBackend::mul(void* a, void* b, void* c, size_t size, DataType dtype) {
    KernelParams params;
    params.buffers = {a, b, c};
    params.int_params = {static_cast<int>(size)};

    uint32_t threads = 256;
    uint32_t blocks = (size + threads - 1) / threads;

    executeKernel("mul_kernel", params, blocks, 1, 1, threads, 1, 1);
}

void MetalBackend::div(void* a, void* b, void* c, size_t size, DataType dtype) {
    KernelParams params;
    params.buffers = {a, b, c};
    params.int_params = {static_cast<int>(size)};

    uint32_t threads = 256;
    uint32_t blocks = (size + threads - 1) / threads;

    executeKernel("div_kernel", params, blocks, 1, 1, threads, 1, 1);
}

void MetalBackend::pow(void* input, void* output, float exponent, size_t size, DataType dtype) {
    KernelParams params;
    params.buffers = {input, output};
    params.int_params = {static_cast<int>(size)};
    params.float_params = {exponent};

    uint32_t threads = 256;
    uint32_t blocks = (size + threads - 1) / threads;

    executeKernel("pow_kernel", params, blocks, 1, 1, threads, 1, 1);
}

void MetalBackend::sqrt(void* input, void* output, size_t size, DataType dtype) {
    KernelParams params;
    params.buffers = {input, output};
    params.int_params = {static_cast<int>(size)};

    uint32_t threads = 256;
    uint32_t blocks = (size + threads - 1) / threads;

    executeKernel("sqrt_kernel", params, blocks, 1, 1, threads, 1, 1);
}

void MetalBackend::exp(void* input, void* output, size_t size, DataType dtype) {
    KernelParams params;
    params.buffers = {input, output};
    params.int_params = {static_cast<int>(size)};

    uint32_t threads = 256;
    uint32_t blocks = (size + threads - 1) / threads;

    executeKernel("exp_kernel", params, blocks, 1, 1, threads, 1, 1);
}

void MetalBackend::log(void* input, void* output, size_t size, DataType dtype) {
    KernelParams params;
    params.buffers = {input, output};
    params.int_params = {static_cast<int>(size)};

    uint32_t threads = 256;
    uint32_t blocks = (size + threads - 1) / threads;

    executeKernel("log_kernel", params, blocks, 1, 1, threads, 1, 1);
}

// Reduction operations
void MetalBackend::sum(void* input, void* output, size_t size, DataType dtype) {
    KernelParams params;
    params.buffers = {input, output};
    params.int_params = {static_cast<int>(size)};

    executeKernel("sum_kernel", params, 1, 1, 1, 256, 1, 1);
}

void MetalBackend::mean(void* input, void* output, size_t size, DataType dtype) {
    KernelParams params;
    params.buffers = {input, output};
    params.int_params = {static_cast<int>(size)};

    executeKernel("mean_kernel", params, 1, 1, 1, 256, 1, 1);
}

void MetalBackend::max(void* input, void* output, size_t size, DataType dtype) {
    KernelParams params;
    params.buffers = {input, output};
    params.int_params = {static_cast<int>(size)};

    executeKernel("max_kernel", params, 1, 1, 1, 256, 1, 1);
}

void MetalBackend::min(void* input, void* output, size_t size, DataType dtype) {
    KernelParams params;
    params.buffers = {input, output};
    params.int_params = {static_cast<int>(size)};

    executeKernel("min_kernel", params, 1, 1, 1, 256, 1, 1);
}

void MetalBackend::reduce_sum_axis(void* input, void* output,
                                   const int* shape, int ndim, int axis,
                                   DataType dtype) {
    // Implementation for axis-specific reduction
    // This would require more complex kernel logic
}

// Pooling operations
void MetalBackend::maxpool2d(void* input, void* output, void* indices,
                            int batch, int channels, int in_h, int in_w,
                            int kernel_h, int kernel_w,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            DataType dtype) {
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    KernelParams params;
    params.buffers = {input, output, indices};
    params.int_params = {batch, channels, in_h, in_w, out_h, out_w,
                        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w};

    uint32_t threads_x = 16;
    uint32_t threads_y = 16;
    uint32_t blocks_x = (out_w + threads_x - 1) / threads_x;
    uint32_t blocks_y = (out_h + threads_y - 1) / threads_y;

    executeKernel("maxpool2d_kernel", params, blocks_x, blocks_y, batch * channels,
                 threads_x, threads_y, 1);
}

void MetalBackend::avgpool2d(void* input, void* output,
                            int batch, int channels, int in_h, int in_w,
                            int kernel_h, int kernel_w,
                            int stride_h, int stride_w,
                            int pad_h, int pad_w,
                            DataType dtype) {
    int out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;

    KernelParams params;
    params.buffers = {input, output};
    params.int_params = {batch, channels, in_h, in_w, out_h, out_w,
                        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w};

    uint32_t threads_x = 16;
    uint32_t threads_y = 16;
    uint32_t blocks_x = (out_w + threads_x - 1) / threads_x;
    uint32_t blocks_y = (out_h + threads_y - 1) / threads_y;

    executeKernel("avgpool2d_kernel", params, blocks_x, blocks_y, batch * channels,
                 threads_x, threads_y, 1);
}

// Batch normalization
void MetalBackend::batchnorm(void* input, void* output,
                            void* gamma, void* beta,
                            void* running_mean, void* running_var,
                            int batch, int channels, int height, int width,
                            float epsilon, bool training,
                            DataType dtype) {
    KernelParams params;
    params.buffers = {input, output, gamma, beta, running_mean, running_var};
    params.int_params = {batch, channels, height, width, training ? 1 : 0};
    params.float_params = {epsilon};

    uint32_t spatial_size = height * width;
    uint32_t threads = 256;
    uint32_t blocks = (spatial_size + threads - 1) / threads;

    executeKernel("batchnorm_kernel", params, blocks, channels, batch,
                 threads, 1, 1);
}

// Transform operations
void MetalBackend::transpose(void* input, void* output,
                            const int* shape, const int* perm, int ndim,
                            DataType dtype) {
    // Implementation for general transpose
    // Would need to pass shape and permutation arrays
}

void MetalBackend::reshape(void* input, void* output,
                          const int* old_shape, const int* new_shape, int ndim,
                          DataType dtype) {
    // Reshape is typically just a view operation
    // For Metal, we might just need to copy the data
    size_t total_elements = 1;
    for (int i = 0; i < ndim; ++i) {
        total_elements *= old_shape[i];
    }

    memcpy_d2d(output, input, total_elements * getDataTypeSize(static_cast<int>(dtype)));
}

// Indexing operations
void MetalBackend::gather(void* input, void* indices, void* output,
                         int batch, int dim, int index_count,
                         DataType dtype) {
    KernelParams params;
    params.buffers = {input, indices, output};
    params.int_params = {batch, dim, index_count};

    uint32_t threads = 256;
    uint32_t blocks = (index_count + threads - 1) / threads;

    executeKernel("gather_kernel", params, blocks, 1, 1, threads, 1, 1);
}

void MetalBackend::scatter(void* input, void* indices, void* output,
                          int batch, int dim, int index_count,
                          DataType dtype) {
    KernelParams params;
    params.buffers = {input, indices, output};
    params.int_params = {batch, dim, index_count};

    uint32_t threads = 256;
    uint32_t blocks = (index_count + threads - 1) / threads;

    executeKernel("scatter_kernel", params, blocks, 1, 1, threads, 1, 1);
}

} // namespace metal
} // namespace backend
} // namespace tenzor
