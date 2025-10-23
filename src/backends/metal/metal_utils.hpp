#pragma once

#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <string>
#include <stdexcept>

namespace tenzor {
namespace backend {
namespace metal {

// Error handling
class MetalException : public std::runtime_error {
public:
    explicit MetalException(const std::string& msg) : std::runtime_error(msg) {}
};

#define METAL_CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            throw MetalException(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + message); \
        } \
    } while(0)

// Thread group size calculation
inline MTLSize calculateThreadGroupSize(id<MTLComputePipelineState> pipeline,
                                       NSUInteger total_threads) {
    NSUInteger max_threads = [pipeline maxTotalThreadsPerThreadgroup];
    NSUInteger thread_width = [pipeline threadExecutionWidth];

    // Calculate optimal thread group size
    NSUInteger threads_per_group = std::min(max_threads, total_threads);

    // Round down to multiple of thread execution width
    threads_per_group = (threads_per_group / thread_width) * thread_width;

    if (threads_per_group == 0) {
        threads_per_group = thread_width;
    }

    return MTLSizeMake(threads_per_group, 1, 1);
}

inline MTLSize calculateThreadGroupSize2D(id<MTLComputePipelineState> pipeline,
                                         NSUInteger width, NSUInteger height) {
    NSUInteger max_threads = [pipeline maxTotalThreadsPerThreadgroup];
    NSUInteger thread_width = [pipeline threadExecutionWidth];

    // Try common 2D thread group sizes
    NSUInteger group_width = 16;
    NSUInteger group_height = 16;

    while (group_width * group_height > max_threads) {
        if (group_width > group_height) {
            group_width /= 2;
        } else {
            group_height /= 2;
        }
    }

    // Ensure at least thread execution width
    if (group_width < thread_width) {
        group_width = thread_width;
        group_height = max_threads / thread_width;
    }

    return MTLSizeMake(group_width, group_height, 1);
}

inline MTLSize calculateThreadGroupSize3D(id<MTLComputePipelineState> pipeline,
                                         NSUInteger width, NSUInteger height, NSUInteger depth) {
    NSUInteger max_threads = [pipeline maxTotalThreadsPerThreadgroup];

    // Try common 3D thread group sizes
    NSUInteger group_width = 8;
    NSUInteger group_height = 8;
    NSUInteger group_depth = 4;

    while (group_width * group_height * group_depth > max_threads) {
        if (group_depth > 1) {
            group_depth /= 2;
        } else if (group_height > group_width) {
            group_height /= 2;
        } else {
            group_width /= 2;
        }
    }

    return MTLSizeMake(group_width, group_height, group_depth);
}

// Grid size calculation
inline MTLSize calculateGridSize(NSUInteger total_threads, MTLSize thread_group_size) {
    NSUInteger groups = (total_threads + thread_group_size.width - 1) / thread_group_size.width;
    return MTLSizeMake(groups, 1, 1);
}

inline MTLSize calculateGridSize2D(NSUInteger width, NSUInteger height, MTLSize thread_group_size) {
    NSUInteger groups_x = (width + thread_group_size.width - 1) / thread_group_size.width;
    NSUInteger groups_y = (height + thread_group_size.height - 1) / thread_group_size.height;
    return MTLSizeMake(groups_x, groups_y, 1);
}

inline MTLSize calculateGridSize3D(NSUInteger width, NSUInteger height, NSUInteger depth,
                                  MTLSize thread_group_size) {
    NSUInteger groups_x = (width + thread_group_size.width - 1) / thread_group_size.width;
    NSUInteger groups_y = (height + thread_group_size.height - 1) / thread_group_size.height;
    NSUInteger groups_z = (depth + thread_group_size.depth - 1) / thread_group_size.depth;
    return MTLSizeMake(groups_x, groups_y, groups_z);
}

// Pipeline state creation helper
inline id<MTLComputePipelineState> createComputePipelineState(id<MTLDevice> device,
                                                              id<MTLLibrary> library,
                                                              const std::string& function_name) {
    NSString* ns_function_name = [NSString stringWithUTF8String:function_name.c_str()];
    id<MTLFunction> function = [library newFunctionWithName:ns_function_name];

    if (!function) {
        throw MetalException("Failed to find function: " + function_name);
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];

    if (!pipeline) {
        std::string error_msg = "Failed to create pipeline state for " + function_name;
        if (error) {
            error_msg += ": " + std::string([[error localizedDescription] UTF8String]);
        }
        throw MetalException(error_msg);
    }

    return pipeline;
}

// Command buffer helper
inline id<MTLCommandBuffer> createCommandBuffer(id<MTLCommandQueue> queue) {
    id<MTLCommandBuffer> buffer = [queue commandBuffer];
    METAL_CHECK(buffer != nil, "Failed to create command buffer");
    return buffer;
}

// Compute encoder helper
inline id<MTLComputeCommandEncoder> createComputeEncoder(id<MTLCommandBuffer> buffer) {
    id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
    METAL_CHECK(encoder != nil, "Failed to create compute encoder");
    return encoder;
}

// Data type size helper
inline size_t getDataTypeSize(int dtype) {
    switch (dtype) {
        case 0: return sizeof(float);    // Float32
        case 1: return sizeof(uint16_t); // Float16
        case 2: return sizeof(int32_t);  // Int32
        case 3: return sizeof(int8_t);   // Int8
        default: return sizeof(float);
    }
}

// MTLDataType conversion
inline MTLDataType toMTLDataType(int dtype) {
    switch (dtype) {
        case 0: return MTLDataTypeFloat;
        case 1: return MTLDataTypeHalf;
        case 2: return MTLDataTypeInt;
        case 3: return MTLDataTypeChar;
        default: return MTLDataTypeFloat;
    }
}

// Buffer creation helper
inline id<MTLBuffer> createBuffer(id<MTLDevice> device, size_t size, MTLResourceOptions options) {
    id<MTLBuffer> buffer = [device newBufferWithLength:size options:options];
    METAL_CHECK(buffer != nil, "Failed to allocate Metal buffer of size " + std::to_string(size));
    return buffer;
}

// Shared buffer creation (for unified memory)
inline id<MTLBuffer> createSharedBuffer(id<MTLDevice> device, size_t size) {
    return createBuffer(device, size, MTLResourceStorageModeShared);
}

// Private buffer creation (for GPU-only memory)
inline id<MTLBuffer> createPrivateBuffer(id<MTLDevice> device, size_t size) {
    return createBuffer(device, size, MTLResourceStorageModePrivate);
}

// Buffer copy helper
inline void copyBuffer(id<MTLCommandBuffer> commandBuffer,
                      id<MTLBuffer> src, id<MTLBuffer> dst,
                      size_t size, size_t src_offset = 0, size_t dst_offset = 0) {
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    [blitEncoder copyFromBuffer:src sourceOffset:src_offset
                       toBuffer:dst destinationOffset:dst_offset
                           size:size];
    [blitEncoder endEncoding];
}

// Synchronous execution helper
inline void executeAndWait(id<MTLCommandBuffer> commandBuffer) {
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    if ([commandBuffer status] == MTLCommandBufferStatusError) {
        NSError* error = [commandBuffer error];
        std::string error_msg = "Command buffer execution failed";
        if (error) {
            error_msg += ": " + std::string([[error localizedDescription] UTF8String]);
        }
        throw MetalException(error_msg);
    }
}

// Asynchronous execution helper
inline void executeAsync(id<MTLCommandBuffer> commandBuffer) {
    [commandBuffer commit];
}

} // namespace metal
} // namespace backend
} // namespace tenzor
