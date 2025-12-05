/**
 * @file vulkan_backend.cpp
 * @brief Vulkan compute backend implementation
 */

#include "vulkan_backend.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/backend/loader.hpp"

// Undefine Vulkan Bool macro that conflicts with DType::Bool
#ifdef Bool
#undef Bool
#endif

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

// Include embedded shaders
#ifdef __has_include
#  if __has_include("embedded_shaders.hpp")
#    include "embedded_shaders.hpp"
#    define TENZOR_HAS_EMBEDDED_SHADERS 1
#  endif
#endif

namespace tenzor {

VulkanBackend::VulkanBackend() {
    try {
        initVulkan();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Vulkan backend: " << e.what() << std::endl;
        // Cleanup if partially initialized
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
        }
        throw;
    }
}

VulkanBackend::~VulkanBackend() {
    // Wait for all devices to finish
    for (auto& ctx : devices_) {
        if (ctx.device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(ctx.device);
        }
    }

    // Cleanup staging buffers
    stagingBuffers_.clear();

    // Cleanup pipeline caches
    pipelineCaches_.clear();

    // Cleanup device contexts
    for (auto& ctx : devices_) {
        // IMPORTANT: Destroy descriptor pool BEFORE destroying device
        // Otherwise descriptor pool destructor will try to use invalid device handle
        if (ctx.descriptorPool) {
            ctx.descriptorPool.reset();
        }
        if (ctx.commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(ctx.device, ctx.commandPool, nullptr);
        }
        if (ctx.device != VK_NULL_HANDLE) {
            vkDestroyDevice(ctx.device, nullptr);
        }
    }

    // Cleanup instance
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

void VulkanBackend::initVulkan() {
    createInstance();
    selectPhysicalDevices();
    createLogicalDevices();

#ifndef TENZOR_HAS_EMBEDDED_SHADERS
    // Fallback to file-based shaders if embedded shaders not available
    const char* shaderEnv = std::getenv("TENZOR_VULKAN_SHADER_PATH");
    if (shaderEnv) {
        shaderPath_ = shaderEnv;
    } else {
        shaderPath_ = "shaders/vulkan/";
    }
#endif
}

void VulkanBackend::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Tenzor";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "Tenzor";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // No extensions needed for compute-only
    createInfo.enabledExtensionCount = 0;
    createInfo.enabledLayerCount = 0;

    vulkan::checkVk(vkCreateInstance(&createInfo, nullptr, &instance_),
                   "Failed to create Vulkan instance");
}

void VulkanBackend::selectPhysicalDevices() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);

    if (deviceCount == 0) {
        throw std::runtime_error("No Vulkan devices found");
    }

    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, physicalDevices.data());

    // Separate devices by type: dedicated GPUs, integrated GPUs, other
    std::vector<DeviceContext> dedicatedGPUs;
    std::vector<DeviceContext> integratedGPUs;
    std::vector<DeviceContext> otherDevices;

    for (auto physDevice : physicalDevices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physDevice, &props);

        // Check for compute queue family
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physDevice, &queueFamilyCount,
                                                queueFamilies.data());

        uint32_t computeQueueFamily = UINT32_MAX;
        for (uint32_t i = 0; i < queueFamilyCount; i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                computeQueueFamily = i;
                break;
            }
        }

        if (computeQueueFamily != UINT32_MAX) {
            DeviceContext ctx{};
            ctx.physicalDevice = physDevice;
            ctx.queueFamilyIndex = computeQueueFamily;
            vkGetPhysicalDeviceMemoryProperties(physDevice, &ctx.memoryProperties);

            // Categorize by device type
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                dedicatedGPUs.push_back(std::move(ctx));
                std::cout << "Found dedicated GPU: " << props.deviceName << std::endl;
            } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                integratedGPUs.push_back(std::move(ctx));
                std::cout << "Found integrated GPU: " << props.deviceName << std::endl;
            } else {
                otherDevices.push_back(std::move(ctx));
                std::cout << "Found other device: " << props.deviceName << std::endl;
            }
        }
    }

    // Priority order: Dedicated GPUs > Integrated GPUs > Other devices
    // Add dedicated GPUs first
    for (auto& ctx : dedicatedGPUs) {
        devices_.push_back(std::move(ctx));
    }
    // Then integrated GPUs
    for (auto& ctx : integratedGPUs) {
        devices_.push_back(std::move(ctx));
    }
    // Finally other devices
    for (auto& ctx : otherDevices) {
        devices_.push_back(std::move(ctx));
    }

    if (devices_.empty()) {
        throw std::runtime_error("No Vulkan devices with compute support found");
    }

    std::cout << "Default Vulkan device (index 0): "
              << (dedicatedGPUs.empty() ? "Not a dedicated GPU" : "Dedicated GPU")
              << std::endl;
}

void VulkanBackend::createLogicalDevices() {
    for (auto& ctx : devices_) {
        // Queue creation
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = ctx.queueFamilyIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        // Device features
        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.shaderFloat64 = VK_TRUE;

        // Check for VK_KHR_shader_float_controls extension support
        uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(ctx.physicalDevice, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(ctx.physicalDevice, nullptr, &extensionCount, availableExtensions.data());

        bool hasFloatControls = false;
        for (const auto& ext : availableExtensions) {
            if (strcmp(ext.extensionName, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME) == 0) {
                hasFloatControls = true;
                break;
            }
        }

        // Extensions to enable
        std::vector<const char*> deviceExtensions;
        if (hasFloatControls) {
            deviceExtensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
        }

        // Query float controls properties to check for denorm preservation support
        VkPhysicalDeviceFloatControlsProperties floatControlsProps{};
        floatControlsProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
        floatControlsProps.pNext = nullptr;

        VkPhysicalDeviceProperties2 deviceProps2{};
        deviceProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        deviceProps2.pNext = &floatControlsProps;
        vkGetPhysicalDeviceProperties2(ctx.physicalDevice, &deviceProps2);

        // Check if denorm preserve is supported for float32
        bool canPreserveDenormsF32 = (floatControlsProps.shaderDenormPreserveFloat32 == VK_TRUE);

        // Device creation
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pQueueCreateInfos = &queueCreateInfo;
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.empty() ? nullptr : deviceExtensions.data();
        createInfo.enabledLayerCount = 0;

        vulkan::checkVk(vkCreateDevice(ctx.physicalDevice, &createInfo,
                                      nullptr, &ctx.device),
                       "Failed to create logical device");

        // Store denorm preservation capability for later use
        ctx.canPreserveDenormsF32 = canPreserveDenormsF32;

        // Get compute queue
        vkGetDeviceQueue(ctx.device, ctx.queueFamilyIndex, 0, &ctx.computeQueue);

        // Create command pool
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = ctx.queueFamilyIndex;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        vulkan::checkVk(vkCreateCommandPool(ctx.device, &poolInfo,
                                           nullptr, &ctx.commandPool),
                       "Failed to create command pool");

        // Create descriptor pool
        // Increased from 1000 to 100000 to support long-running tests (transformers, LSTMs, etc.)
        ctx.descriptorPool = std::make_unique<vulkan::DescriptorPool>(ctx.device, 100000);

        // Initialize caches
        stagingBuffers_.push_back({});
        pipelineCaches_.push_back({});
    }
}

auto VulkanBackend::device_count() const -> int32_t {
    return static_cast<int32_t>(devices_.size());
}

auto VulkanBackend::is_available() const -> bool {
    return !devices_.empty();
}

auto VulkanBackend::allocate(size_t bytes, int32_t device_id) -> void* {
    if (bytes == 0) {
        return nullptr;
    }

    if (device_id < 0 || device_id >= device_count()) {
        throw std::invalid_argument("Invalid device ID");
    }

    void* ptr = allocateDeviceMemory(bytes, device_id);
    allocations_[ptr] = {bytes, device_id};
    return ptr;
}

void* VulkanBackend::allocateDeviceMemory(size_t bytes, int32_t device_id) {
    auto& ctx = devices_[device_id];

    auto buffer = std::make_unique<vulkan::VulkanBuffer>(
        ctx.device, ctx.physicalDevice, bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    // Use buffer handle as opaque pointer for tracking
    void* ptr = reinterpret_cast<void*>(buffer->buffer());

    // Store buffer object in tracking map
    bufferMap_[ptr] = std::move(buffer);

    return ptr;
}

auto VulkanBackend::deallocate(void* ptr) -> void {
    if (ptr == nullptr) {
        return;
    }

    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        auto [bytes, device_id] = it->second;
        freeDeviceMemory(ptr, device_id);
        allocations_.erase(it);
    }
}

void VulkanBackend::freeDeviceMemory(void* ptr, int32_t device_id) {
    // Remove buffer from tracking map - destructor handles cleanup
    bufferMap_.erase(ptr);
}

auto VulkanBackend::copy(void* dst, const void* src, size_t bytes,
                        CopyKind kind) -> void {
    if (bytes == 0) {
        return;
    }

    // Determine device ID from allocations
    int32_t device_id = 0;

    switch (kind) {
        case CopyKind::HostToDevice: {
            auto& staging = getStagingBuffer(device_id, bytes);
            void* mapped = staging.buffer->map();
            std::memcpy(mapped, src, bytes);
            staging.buffer->unmap();

            // Copy from staging to device
            auto [dst_buffer, dst_offset] = getVulkanBufferAndOffset(dst);
            VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;  // Staging buffer starts at 0
            copyRegion.dstOffset = dst_offset;
            copyRegion.size = bytes;
            vkCmdCopyBuffer(cmdBuffer, staging.buffer->buffer(),
                          dst_buffer, 1, &copyRegion);
            endSingleTimeCommands(cmdBuffer, device_id);
            break;
        }
        case CopyKind::DeviceToHost: {
            auto& staging = getStagingBuffer(device_id, bytes);

            // Copy from device to staging
            auto [src_buffer, src_offset] = getVulkanBufferAndOffset(src);
            VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = src_offset;
            copyRegion.dstOffset = 0;  // Staging buffer starts at 0
            copyRegion.size = bytes;
            vkCmdCopyBuffer(cmdBuffer, src_buffer,
                          staging.buffer->buffer(), 1, &copyRegion);
            endSingleTimeCommands(cmdBuffer, device_id);

            void* mapped = staging.buffer->map();
            std::memcpy(dst, mapped, bytes);
            staging.buffer->unmap();
            break;
        }
        case CopyKind::DeviceToDevice: {
            auto [src_buffer, src_offset] = getVulkanBufferAndOffset(src);
            auto [dst_buffer, dst_offset] = getVulkanBufferAndOffset(dst);
            VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = src_offset;
            copyRegion.dstOffset = dst_offset;
            copyRegion.size = bytes;
            vkCmdCopyBuffer(cmdBuffer, src_buffer,
                          dst_buffer, 1, &copyRegion);
            endSingleTimeCommands(cmdBuffer, device_id);
            break;
        }
        case CopyKind::HostToHost: {
            std::memcpy(dst, src, bytes);
            break;
        }
    }
}

VulkanBackend::StagingBuffer& VulkanBackend::getStagingBuffer(int32_t device_id, size_t size) {
    auto& staging = stagingBuffers_[device_id];

    if (!staging.buffer || staging.size < size) {
        auto& ctx = devices_[device_id];
        staging.buffer = std::make_unique<vulkan::VulkanBuffer>(
            ctx.device, ctx.physicalDevice, size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        staging.size = size;
    }

    return staging;
}

VkCommandBuffer VulkanBackend::beginSingleTimeCommands(int32_t device_id) {
    auto& ctx = devices_[device_id];

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = ctx.commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(ctx.device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void VulkanBackend::endSingleTimeCommands(VkCommandBuffer commandBuffer, int32_t device_id) {
    auto& ctx = devices_[device_id];

    VkResult result = vkEndCommandBuffer(commandBuffer);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to end command buffer: " + std::to_string(result));
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    result = vkQueueSubmit(ctx.computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit queue: " + std::to_string(result));
    }

    result = vkQueueWaitIdle(ctx.computeQueue);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to wait for queue idle: " + std::to_string(result));
    }

    vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &commandBuffer);
}

auto VulkanBackend::synchronize(int32_t device_id) -> void {
    if (device_id < 0 || device_id >= device_count()) {
        throw std::invalid_argument("Invalid device ID");
    }
    auto& ctx = devices_[device_id];
    vkDeviceWaitIdle(ctx.device);

    // Reset command pool to prevent fragmentation/corruption
    vkResetCommandPool(ctx.device, ctx.commandPool, 0);
}

auto VulkanBackend::create_stream(int32_t device_id) -> StreamHandle {
    // Vulkan doesn't have explicit streams like CUDA
    // We could use command buffers for async execution
    // For now, return nullptr (default stream)
    return nullptr;
}

auto VulkanBackend::destroy_stream(StreamHandle stream) -> void {
    // No-op for default stream
}

auto VulkanBackend::synchronize_stream(StreamHandle stream) -> void {
    // For default stream, synchronize device
    if (stream == nullptr && !devices_.empty()) {
        vkQueueWaitIdle(devices_[0].computeQueue);
    }
}

vulkan::ComputePipeline* VulkanBackend::getPipeline(const std::string& shader_name,
                                                    int32_t device_id) {
    auto& cache = pipelineCaches_[device_id];
    auto it = cache.pipelines.find(shader_name);

    if (it != cache.pipelines.end()) {
        return it->second.get();
    }

    // Load shader code (embedded or from file)
    std::vector<uint32_t> shaderCode;

#ifdef TENZOR_HAS_EMBEDDED_SHADERS
    // Use embedded shaders
    const auto& registry = vulkan::embedded_shaders::getShaderRegistry();
    auto shader_it = registry.find(shader_name);
    if (shader_it != registry.end()) {
        const auto& shaderData = shader_it->second;
        shaderCode.assign(shaderData.data, shaderData.data + shaderData.size);
    } else {
        throw std::runtime_error(
            "Embedded shader '" + shader_name + "' not found in registry.\n" +
            "  Available shaders: " + std::to_string(registry.size())
        );
    }
#else
    // Fallback to file-based loading
    std::string shaderFile = shaderPath_ + shader_name + ".spv";
    try {
        shaderCode = vulkan::loadShader(shaderFile);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to load Vulkan shader '" + shader_name + "'\n" +
            "  Expected location: " + shaderFile + "\n" +
            "  Error: " + e.what() + "\n" +
            "  Hint: Set TENZOR_VULKAN_SHADER_PATH environment variable or ensure shaders are built"
        );
    }
#endif

    // Create descriptor bindings (up to 8 buffers)
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (uint32_t i = 0; i < 8; i++) {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = i;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(binding);
    }

    // Setup push constants for shaders
    std::vector<VkPushConstantRange> pushConstants;
    if (shader_name == "math" || shader_name == "comparison" || shader_name == "comparison_bool" ||
        shader_name == "comparison_f64" || shader_name == "comparison_i32" || shader_name == "comparison_i64") {
        // math/comparison (all variants): 8 bytes (uint n, uint op)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 8;  // 2 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "fill" || shader_name == "full" || shader_name == "full_f16" || shader_name == "full_i8") {
        // fill/full/full_f16/full_i8: 8 bytes (uint n, uint/float value bits)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 8;  // uint32_t + uint32_t (value bits)
        pushConstants.push_back(push_range);
    } else if (shader_name == "ones" || shader_name == "ones_f16" || shader_name == "ones_i8") {
        // ones/ones_f16/ones_i8: 4 bytes (uint n)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 4;  // uint32_t
        pushConstants.push_back(push_range);
    } else if (shader_name == "conv2d_forward" || shader_name == "conv2d_forward_f64" || shader_name == "conv2d_forward_f16") {
        // conv2d_forward (all dtype variants): 60 bytes (15 uint32_t)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 60;  // 15 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "transform") {
        // transform: 20 bytes (uint n, ndim, transform, dim0, dim1)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 20;  // 5 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "math_broadcast" || shader_name == "math_broadcast_i8" || shader_name == "math_broadcast_f16" ||
               shader_name == "math_broadcast_uint8" || shader_name == "math_broadcast_i64" || shader_name == "math_broadcast_f64") {
        // math_broadcast (all variants): 120 bytes (output_size, op, dtype, ndim_a, ndim_b, ndim_out, strides_a[8], strides_b[8], shape_out[8])
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 120;  // 6 uint32_t + 3 * 8 uint32_t arrays = 6*4 + 24*4 = 120 bytes
        pushConstants.push_back(push_range);
    } else if (shader_name == "math_i8" || shader_name == "math_uint8" || shader_name == "math_i64") {
        // math_i8/math_uint8/math_i64: 12 bytes (n, op, param)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;  // 3 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "full_uint8") {
        // full_uint8: 8 bytes (n_elements, fill_value_uint8)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 8;  // 2 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "ones_uint8") {
        // ones_uint8: 4 bytes (n_elements)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 4;  // 1 uint32_t value
        pushConstants.push_back(push_range);
    } else if (shader_name == "full_i64") {
        // full_i64: 12 bytes (n, value_low, value_high)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;  // 3 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "ones_i64") {
        // ones_i64: 4 bytes (n_elements)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 4;  // 1 uint32_t value
        pushConstants.push_back(push_range);
    } else if (shader_name == "clamp_f64") {
        // clamp_f64: 24 bytes (n_elements, padding, min_value double, max_value double)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 24;  // uint32_t + uint32_t padding + 2 doubles
        pushConstants.push_back(push_range);
    } else if (shader_name == "gather_relative_position_bias" ||
               shader_name == "gather_relative_position_bias_f64" ||
               shader_name == "gather_relative_position_bias_f16") {
        // gather_relative_position_bias: 12 bytes (num_positions, num_heads, total_elements)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;  // 3 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "adaptive_pooling" || shader_name == "adaptive_pooling_f64" || shader_name == "adaptive_pooling_f16") {
        // adaptive_pooling: 28 bytes (batch, channels, in_height, in_width, out_height, out_width, pool_type)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 28;  // 7 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "adaptive_avg_pool2d_backward" || shader_name == "adaptive_avg_pool2d_backward_f64" || shader_name == "adaptive_avg_pool2d_backward_f16") {
        // adaptive_avg_pool2d_backward: 24 bytes (batch, channels, H_in, W_in, H_out, W_out)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 24;  // 6 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "expand" || shader_name == "expand_f64" || shader_name == "expand_f16") {
        // expand (all variants): 108 bytes (n_elements, input_ndim, output_ndim, input_shape[8], output_shape[8], input_strides[8])
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 108;  // 3 + 8 + 8 + 8 = 27 uint32_t values
        pushConstants.push_back(push_range);
    }

    // Create pipeline
    auto& ctx = devices_[device_id];
    auto pipeline = std::make_unique<vulkan::ComputePipeline>(
        ctx.device, shaderCode, bindings, pushConstants
    );

    auto* pipelinePtr = pipeline.get();
    cache.pipelines[shader_name] = std::move(pipeline);

    return pipelinePtr;
}

// Helper to get VkBuffer and offset from a potentially-offset pointer
std::pair<VkBuffer, VkDeviceSize> VulkanBackend::getVulkanBufferAndOffset(const void* ptr) const {
    // First try direct lookup
    auto it = bufferMap_.find(const_cast<void*>(ptr));
    if (it != bufferMap_.end()) {
        return {it->second->buffer(), 0};
    }

    // If not found, ptr might be base_ptr + offset
    // Search for a buffer where: base_ptr <= ptr < base_ptr + size
    const auto* ptr_as_uint = reinterpret_cast<const uint8_t*>(ptr);

    for (const auto& [base_ptr, buffer_obj] : bufferMap_) {
        const auto* base_as_uint = reinterpret_cast<const uint8_t*>(base_ptr);

        // Check if ptr is in the range [base_ptr, base_ptr + size)
        auto alloc_it = allocations_.find(base_ptr);
        if (alloc_it != allocations_.end()) {
            const size_t size = alloc_it->second.first;
            if (ptr_as_uint >= base_as_uint && ptr_as_uint < base_as_uint + size) {
                // Found it! ptr is within this buffer's range
                VkDeviceSize offset = static_cast<VkDeviceSize>(ptr_as_uint - base_as_uint);
                return {buffer_obj->buffer(), offset};
            }
        }
    }

    // Not found even with offset search
    throw std::runtime_error("Invalid buffer pointer: buffer not tracked");
}

VkBuffer VulkanBackend::getVulkanBuffer(const void* ptr) const {
    return getVulkanBufferAndOffset(ptr).first;
}

VkDescriptorSet VulkanBackend::allocateAndWriteDescriptorSet(
    int32_t device_id,
    vulkan::ComputePipeline* pipeline,
    const std::vector<std::pair<uint32_t, VkBuffer>>& bufferBindings,
    const std::vector<size_t>& bufferSizes) {

    auto& ctx = devices_[device_id];

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = ctx.descriptorPool->pool();
    allocInfo.descriptorSetCount = 1;
    VkDescriptorSetLayout layout = pipeline->descriptorSetLayout();
    allocInfo.pSetLayouts = &layout;

    VkDescriptorSet descriptorSet;
    vulkan::checkVk(vkAllocateDescriptorSets(ctx.device, &allocInfo, &descriptorSet),
                    "Failed to allocate descriptor set");

    // Write descriptor set bindings
    std::vector<VkDescriptorBufferInfo> bufferInfos(bufferBindings.size());
    std::vector<VkWriteDescriptorSet> writes(bufferBindings.size());

    for (size_t i = 0; i < bufferBindings.size(); ++i) {
        bufferInfos[i].buffer = bufferBindings[i].second;
        bufferInfos[i].offset = 0;
        bufferInfos[i].range = bufferSizes[i];

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].pNext = nullptr;
        writes[i].dstSet = descriptorSet;
        writes[i].dstBinding = bufferBindings[i].first;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufferInfos[i];
        writes[i].pImageInfo = nullptr;
        writes[i].pTexelBufferView = nullptr;
    }

    vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()),
                          writes.data(), 0, nullptr);

    return descriptorSet;
}

auto VulkanBackend::dispatch(const std::string& op_name,
                            std::span<const Tensor> inputs,
                            const OpAttributes& attrs) -> std::vector<Tensor> {

    try {
    // Binary operations
    if (op_name == "add" || op_name == "sub" || op_name == "mul" || op_name == "div") {
        if (inputs.size() != 2) {
            throw std::invalid_argument(op_name + " requires 2 inputs");
        }
        return {dispatchBinaryOp(op_name, inputs[0], inputs[1])};
    }

    // In-place operations (modify first tensor in-place)
    if (op_name == "add_inplace" || op_name == "sub_inplace" ||
        op_name == "mul_inplace" || op_name == "div_inplace") {
        if (inputs.size() != 2) {
            throw std::invalid_argument(op_name + " requires 2 inputs");
        }
        // Extract the base operation name (remove "_inplace" suffix)
        std::string base_op = op_name.substr(0, op_name.find("_inplace"));

        // Perform the operation and copy result back to first tensor's buffer
        Tensor result = dispatchBinaryOp(base_op, inputs[0], inputs[1]);

        // Copy result data back to input tensor's buffer (in-place modification)
        // Note: const_cast is safe here as we're explicitly modifying the tensor in-place
        size_t bytes = result.numel() * result.dtype_size();
        void* dst = const_cast<void*>(inputs[0].data_ptr());
        copy(dst, result.data_ptr(), bytes, CopyKind::DeviceToDevice);

        return {inputs[0]};
    }

    // Comparison operations
    if (op_name == "eq" || op_name == "ne" || op_name == "lt" ||
        op_name == "le" || op_name == "gt" || op_name == "ge") {
        if (inputs.size() != 2) {
            throw std::invalid_argument(op_name + " requires 2 inputs");
        }
        return {dispatchComparisonOp(op_name, inputs[0], inputs[1])};
    }

    // Activation functions (use activations shader)
    if (op_name == "relu" || op_name == "sigmoid" || op_name == "tanh") {
        if (inputs.size() != 1) {
            throw std::invalid_argument(op_name + " requires 1 input");
        }
        uint32_t opcode = 0;
        if (op_name == "relu") opcode = 0;
        else if (op_name == "sigmoid") opcode = 1;
        else if (op_name == "tanh") opcode = 2;
        return {dispatchActivation(op_name, inputs[0], opcode, 0.0f)};
    }

    // GELU activation
    if (op_name == "gelu") {
        if (inputs.size() != 1) throw std::invalid_argument("gelu requires 1 input");
        return {dispatchActivation("gelu", inputs[0], 3, 0.0f)};
    }

    // LeakyReLU activation
    if (op_name == "leaky_relu") {
        if (inputs.size() != 1) throw std::invalid_argument("leaky_relu requires 1 input");
        float alpha = attrs.contains("alpha") ? std::stof(attrs.at("alpha")) : 0.01f;
        return {dispatchActivation("leaky_relu", inputs[0], 4, alpha)};
    }

    // Swish activation
    if (op_name == "swish") {
        if (inputs.size() != 1) throw std::invalid_argument("swish requires 1 input");
        return {dispatchActivation("swish", inputs[0], 5, 0.0f)};
    }

    // Unary math operations (use math shader)
    if (op_name == "sqrt" || op_name == "exp" || op_name == "log" ||
        op_name == "neg" || op_name == "abs" || op_name == "sign" ||
        op_name == "floor" || op_name == "ceil" || op_name == "round" ||
        op_name == "trunc" || op_name == "reciprocal") {
        if (inputs.size() != 1) {
            throw std::invalid_argument(op_name + " requires 1 input");
        }
        return {dispatchUnaryOp(op_name, inputs[0])};
    }

    // Trigonometric operations
    if (op_name == "sin" || op_name == "cos" || op_name == "tan" ||
        op_name == "asin" || op_name == "acos" || op_name == "atan") {
        if (inputs.size() != 1) {
            throw std::invalid_argument(op_name + " requires 1 input");
        }
        return {dispatchTrigonometricOp(op_name, inputs[0])};
    }

    // Hyperbolic operations
    if (op_name == "sinh" || op_name == "cosh" || op_name == "tanh") {
        if (inputs.size() != 1) {
            throw std::invalid_argument(op_name + " requires 1 input");
        }
        return {dispatchHyperbolicOp(op_name, inputs[0])};
    }

    // Pow operation (unary with parameter)
    if (op_name == "pow") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("pow requires 1 input");
        }
        float exponent = attrs.contains("exponent") ? std::stof(attrs.at("exponent")) : 2.0f;
        return {dispatchUnaryOpWithParam(op_name, inputs[0], exponent)};
    }

    // Backward activation operations
    if (op_name == "relu_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("relu_backward requires 2 inputs");
        return {dispatchActivationBackward("relu_backward", inputs[0], inputs[1], 0, 0.0f)};
    }

    if (op_name == "sigmoid_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("sigmoid_backward requires 2 inputs");
        return {dispatchActivationBackward("sigmoid_backward", inputs[0], inputs[1], 1, 0.0f)};
    }

    if (op_name == "tanh_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("tanh_backward requires 2 inputs");
        return {dispatchActivationBackward("tanh_backward", inputs[0], inputs[1], 2, 0.0f)};
    }

    if (op_name == "leaky_relu_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("leaky_relu_backward requires 2 inputs");
        float alpha = attrs.contains("alpha") ? std::stof(attrs.at("alpha")) : 0.01f;
        return {dispatchActivationBackward("leaky_relu_backward", inputs[0], inputs[1], 3, alpha)};
    }

    if (op_name == "gelu_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("gelu_backward requires 2 inputs");
        return {dispatchActivationBackward("gelu_backward", inputs[0], inputs[1], 4, 0.0f)};
    }

    if (op_name == "swish_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("swish_backward requires 2 inputs");
        return {dispatchSwishBackward(inputs[0], inputs[1])};
    }

    if (op_name == "softmax_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("softmax_backward requires 2 inputs");
        int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
        return {dispatchSoftmaxBackward(inputs[0], inputs[1], dim)};
    }

    if (op_name == "log_softmax_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("log_softmax_backward requires 2 inputs");
        int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
        return {dispatchLogSoftmaxBackward(inputs[0], inputs[1], dim)};
    }

    // Conv2d backward operations
    if (op_name == "conv2d_backward_input") {
        if (inputs.size() != 2) throw std::invalid_argument("conv2d_backward_input requires 2 inputs (grad_output, weight)");
        int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : 1;
        int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
        int64_t dilation = attrs.contains("dilation") ? std::stoll(attrs.at("dilation")) : 1;

        // Parse input shape from comma-separated string
        std::vector<int64_t> input_shape;
        std::string shape_str = attrs.at("input_shape");
        size_t pos = 0;
        while (pos < shape_str.size()) {
            size_t comma = shape_str.find(',', pos);
            if (comma == std::string::npos) comma = shape_str.size();
            input_shape.push_back(std::stoll(shape_str.substr(pos, comma - pos)));
            pos = comma + 1;
        }

        return {dispatchConv2dBackwardInput(inputs[0], inputs[1], stride, padding, dilation, input_shape)};
    }

    if (op_name == "conv2d_backward_weight") {
        if (inputs.size() != 2) throw std::invalid_argument("conv2d_backward_weight requires 2 inputs (grad_output, input)");
        int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : 1;
        int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
        int64_t dilation = attrs.contains("dilation") ? std::stoll(attrs.at("dilation")) : 1;

        // Parse weight shape from comma-separated string
        std::vector<int64_t> weight_shape;
        std::string shape_str = attrs.at("weight_shape");
        size_t pos = 0;
        while (pos < shape_str.size()) {
            size_t comma = shape_str.find(',', pos);
            if (comma == std::string::npos) comma = shape_str.size();
            weight_shape.push_back(std::stoll(shape_str.substr(pos, comma - pos)));
            pos = comma + 1;
        }

        return {dispatchConv2dBackwardWeight(inputs[0], inputs[1], stride, padding, dilation, weight_shape)};
    }

    if (op_name == "conv2d_backward_bias") {
        if (inputs.size() != 1) throw std::invalid_argument("conv2d_backward_bias requires 1 input (grad_output)");
        return {dispatchConv2dBackwardBias(inputs[0])};
    }

    // Reduction operations
    if (op_name == "sum" || op_name == "mean" || op_name == "max" || op_name == "min") {
        if (inputs.size() != 1) {
            throw std::invalid_argument(op_name + " requires 1 input");
        }
        // Use INT64_MIN as sentinel for "reduce all elements" (full reduction)
        // dim=-1, -2, etc. mean negative indexing from the end
        int64_t dim = INT64_MIN;  // Sentinel for full reduction
        bool keepdim = false;
        if (attrs.contains("dim")) {
            dim = std::stoll(attrs.at("dim"));
        }
        if (attrs.contains("keepdim")) {
            keepdim = (attrs.at("keepdim") == "1");
        }
        return {dispatchReduction(op_name, inputs[0], dim, keepdim)};
    }

    // Matrix multiplication
    if (op_name == "matmul") {
        if (inputs.size() != 2) {
            throw std::invalid_argument("matmul requires 2 inputs");
        }
        return {dispatchMatmul(inputs[0], inputs[1])};
    }

    // Pooling operations
    if (op_name == "max_pool2d") {
        // Support both kernel_h/kernel_w and kernel_size attributes
        int64_t kernel_h = 2, kernel_w = 2;
        if (attrs.contains("kernel_h")) {
            kernel_h = std::stoll(attrs.at("kernel_h"));
            kernel_w = attrs.contains("kernel_w") ? std::stoll(attrs.at("kernel_w")) : kernel_h;
        } else if (attrs.contains("kernel_size")) {
            kernel_h = kernel_w = std::stoll(attrs.at("kernel_size"));
        }

        // Support both stride_h/stride_w and stride attributes
        int64_t stride_h = kernel_h, stride_w = kernel_w;
        if (attrs.contains("stride_h")) {
            stride_h = std::stoll(attrs.at("stride_h"));
            stride_w = attrs.contains("stride_w") ? std::stoll(attrs.at("stride_w")) : stride_h;
        } else if (attrs.contains("stride")) {
            stride_h = stride_w = std::stoll(attrs.at("stride"));
        }

        // Support both padding_h/padding_w and padding attributes
        int64_t padding_h = 0, padding_w = 0;
        if (attrs.contains("padding_h")) {
            padding_h = std::stoll(attrs.at("padding_h"));
            padding_w = attrs.contains("padding_w") ? std::stoll(attrs.at("padding_w")) : padding_h;
        } else if (attrs.contains("padding")) {
            padding_h = padding_w = std::stoll(attrs.at("padding"));
        }

        auto [output, indices] = dispatchMaxPool2d(inputs[0], kernel_h, kernel_w,
                                                    stride_h, stride_w, padding_h, padding_w);
        return {output, indices};
    }

    if (op_name == "avg_pool2d") {
        // Support both kernel_h/kernel_w and kernel_size attributes
        int64_t kernel_h = 2, kernel_w = 2;
        if (attrs.contains("kernel_h")) {
            kernel_h = std::stoll(attrs.at("kernel_h"));
            kernel_w = attrs.contains("kernel_w") ? std::stoll(attrs.at("kernel_w")) : kernel_h;
        } else if (attrs.contains("kernel_size")) {
            kernel_h = kernel_w = std::stoll(attrs.at("kernel_size"));
        }

        // Support both stride_h/stride_w and stride attributes
        int64_t stride_h = kernel_h, stride_w = kernel_w;
        if (attrs.contains("stride_h")) {
            stride_h = std::stoll(attrs.at("stride_h"));
            stride_w = attrs.contains("stride_w") ? std::stoll(attrs.at("stride_w")) : stride_h;
        } else if (attrs.contains("stride")) {
            stride_h = stride_w = std::stoll(attrs.at("stride"));
        }
        int64_t padding_h = attrs.contains("padding_h") ? std::stoll(attrs.at("padding_h")) : 0;
        int64_t padding_w = attrs.contains("padding_w") ? std::stoll(attrs.at("padding_w")) : 0;
        return {dispatchAvgPool2d(inputs[0], kernel_h, kernel_w,
                                  stride_h, stride_w, padding_h, padding_w)};
    }

    if (op_name == "adaptive_max_pool2d") {
        // Support both naming conventions: output_height/output_width and output_h/output_w
        int64_t out_h = attrs.contains("output_height") ? std::stoll(attrs.at("output_height"))
                      : std::stoll(attrs.at("output_h"));
        int64_t out_w = attrs.contains("output_width") ? std::stoll(attrs.at("output_width"))
                      : std::stoll(attrs.at("output_w"));
        auto [output, indices] = dispatchAdaptiveMaxPool2d(inputs[0], out_h, out_w);
        return {output, indices};
    }

    if (op_name == "adaptive_avg_pool2d") {
        // Support both naming conventions: output_height/output_width and output_h/output_w
        int64_t out_h = attrs.contains("output_height") ? std::stoll(attrs.at("output_height"))
                      : std::stoll(attrs.at("output_h"));
        int64_t out_w = attrs.contains("output_width") ? std::stoll(attrs.at("output_width"))
                      : std::stoll(attrs.at("output_w"));
        return {dispatchAdaptiveAvgPool2d(inputs[0], out_h, out_w)};
    }

    if (op_name == "adaptive_avg_pool2d_backward") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("adaptive_avg_pool2d_backward requires exactly 1 input (grad_output)");
        }
        int64_t H_in = attrs.contains("H_in") ? std::stoll(attrs.at("H_in")) : 0;
        int64_t W_in = attrs.contains("W_in") ? std::stoll(attrs.at("W_in")) : 0;
        return {dispatchAdaptiveAvgPool2dBackward(inputs[0], H_in, W_in)};
    }

    // Normalization
    if (op_name == "softmax") {
        int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
        return {dispatchSoftmax(inputs[0], dim)};
    }

    if (op_name == "log_softmax") {
        int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
        return {dispatchLogSoftmax(inputs[0], dim)};
    }

    // Advanced reductions
    if (op_name == "argmax") {
        int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
        bool keepdim = attrs.contains("keepdim") && attrs.at("keepdim") == "1";
        return {dispatchArgmax(inputs[0], dim, keepdim)};
    }

    if (op_name == "argmin") {
        int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
        bool keepdim = attrs.contains("keepdim") && attrs.at("keepdim") == "1";
        return {dispatchArgmin(inputs[0], dim, keepdim)};
    }

    if (op_name == "var" || op_name == "variance") {
        int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
        bool unbiased = !attrs.contains("unbiased") || attrs.at("unbiased") == "1";
        bool keepdim = attrs.contains("keepdim") && attrs.at("keepdim") == "1";
        return {dispatchVariance(inputs[0], dim, unbiased, keepdim)};
    }

    if (op_name == "std") {
        int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
        bool unbiased = !attrs.contains("unbiased") || attrs.at("unbiased") == "1";
        bool keepdim = attrs.contains("keepdim") && attrs.at("keepdim") == "1";
        return {dispatchStd(inputs[0], dim, unbiased, keepdim)};
    }

    if (op_name == "prod") {
        int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
        bool keepdim = attrs.contains("keepdim") && attrs.at("keepdim") == "1";
        return {dispatchProd(inputs[0], dim, keepdim)};
    }

    if (op_name == "norm") {
        float p = attrs.contains("p") ? std::stof(attrs.at("p")) : 2.0f;
        // Use INT64_MIN to signal full reduction when dim is not specified
        int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : INT64_MIN;
        bool keepdim = attrs.contains("keepdim") && attrs.at("keepdim") == "1";
        return {dispatchNorm(inputs[0], p, dim, keepdim)};
    }

    // Indexing operations
    if (op_name == "embedding") {
        int64_t padding_idx = attrs.contains("padding_idx") ? std::stoll(attrs.at("padding_idx")) : -1;
        return {dispatchEmbedding(inputs[0], inputs[1], padding_idx)};
    }

    if (op_name == "gather") {
        int64_t dim = std::stoll(attrs.at("dim"));
        return {dispatchGather(inputs[0], dim, inputs[1])};
    }

    if (op_name == "scatter") {
        int64_t dim = std::stoll(attrs.at("dim"));
        int64_t reduction = attrs.contains("reduction") ? std::stoll(attrs.at("reduction")) : 0;
        return {dispatchScatter(inputs[0], dim, inputs[1], inputs[2], reduction)};
    }

    if (op_name == "index_select") {
        int64_t dim = std::stoll(attrs.at("dim"));
        return {dispatchIndexSelect(inputs[0], dim, inputs[1])};
    }

    // Vision operations
    if (op_name == "gather_relative_position_bias") {
        if (inputs.size() != 2) {
            throw std::invalid_argument("gather_relative_position_bias operation requires exactly 2 inputs");
        }
        int64_t num_positions = 0, num_heads = 0;
        if (attrs.contains("num_positions")) {
            num_positions = std::stoll(attrs.at("num_positions"));
        }
        if (attrs.contains("num_heads")) {
            num_heads = std::stoll(attrs.at("num_heads"));
        }
        return {dispatchGatherRelativePositionBias(inputs[0], inputs[1], num_positions, num_heads)};
    }

    // Shape operations
    if (op_name == "reshape") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("reshape requires 1 input");
        }
        if (!attrs.contains("shape")) {
            throw std::invalid_argument("reshape requires 'shape' attribute");
        }
        // Parse shape from comma-separated string
        std::vector<int64_t> new_shape;
        std::string shape_str = attrs.at("shape");
        size_t pos = 0;
        while (pos < shape_str.size()) {
            size_t comma = shape_str.find(',', pos);
            if (comma == std::string::npos) comma = shape_str.size();
            new_shape.push_back(std::stoll(shape_str.substr(pos, comma - pos)));
            pos = comma + 1;
        }
        return {dispatchReshape(inputs[0], new_shape)};
    }

    if (op_name == "transpose") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("transpose requires 1 input");
        }
        int64_t dim0 = std::stoll(attrs.at("dim0"));
        int64_t dim1 = std::stoll(attrs.at("dim1"));
        return {dispatchTranspose(inputs[0], dim0, dim1)};
    }

    if (op_name == "permute") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("permute requires 1 input");
        }
        // Parse dims from comma-separated string
        std::vector<int64_t> dims;
        std::string dims_str = attrs.at("dims");
        size_t pos = 0;
        while (pos < dims_str.size()) {
            size_t comma = dims_str.find(',', pos);
            if (comma == std::string::npos) comma = dims_str.size();
            dims.push_back(std::stoll(dims_str.substr(pos, comma - pos)));
            pos = comma + 1;
        }
        return {dispatchPermute(inputs[0], dims)};
    }

    if (op_name == "squeeze") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("squeeze requires 1 input");
        }
        int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
        return {dispatchSqueeze(inputs[0], dim)};
    }

    if (op_name == "unsqueeze") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("unsqueeze requires 1 input");
        }
        int64_t dim = std::stoll(attrs.at("dim"));
        return {dispatchUnsqueeze(inputs[0], dim)};
    }

    if (op_name == "contiguous") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("contiguous requires 1 input");
        }
        return {dispatchContiguous(inputs[0])};
    }

    // Memory operations
    if (op_name == "zeros") {
        // Parse shape from comma-separated string
        std::vector<int64_t> shape;
        std::string shape_str = attrs.at("shape");
        size_t pos = 0;
        while (pos < shape_str.size()) {
            size_t comma = shape_str.find(',', pos);
            if (comma == std::string::npos) comma = shape_str.size();
            shape.push_back(std::stoll(shape_str.substr(pos, comma - pos)));
            pos = comma + 1;
        }
        // Get dtype and device from attributes or use defaults
        DType dtype = DType::Float32;
        if (attrs.contains("dtype")) {
            // Parse dtype string - must handle all dtypes
            std::string dtype_str = attrs.at("dtype");
            if (dtype_str == "float32") dtype = DType::Float32;
            else if (dtype_str == "float64") dtype = DType::Float64;
            else if (dtype_str == "float16") dtype = DType::Float16;
            else if (dtype_str == "bfloat16") dtype = DType::BFloat16;
            else if (dtype_str == "int8") dtype = DType::Int8;
            else if (dtype_str == "int16") dtype = DType::Int16;
            else if (dtype_str == "int32") dtype = DType::Int32;
            else if (dtype_str == "int64") dtype = DType::Int64;
            else if (dtype_str == "uint8") dtype = DType::UInt8;
            else if (dtype_str == "uint16") dtype = DType::UInt16;
            else if (dtype_str == "uint32") dtype = DType::UInt32;
            else if (dtype_str == "uint64") dtype = DType::UInt64;
            else if (dtype_str == "bool") dtype = DType::Bool;
            else if (dtype_str == "complex64") dtype = DType::Complex64;
            else if (dtype_str == "complex128") dtype = DType::Complex128;
        }
        int32_t device_id = attrs.contains("device_id") ? std::stoi(attrs.at("device_id")) : 0;
        Device device = Device::vulkan(device_id);
        return {dispatchZeros(shape, dtype, device)};
    }

    if (op_name == "fill") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("fill requires 1 input");
        }
        float value = std::stof(attrs.at("value"));
        return {dispatchFill(inputs[0], value)};
    }

    if (op_name == "clone") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("clone requires 1 input");
        }
        return {dispatchClone(inputs[0])};
    }

    // Vision operations
    if (op_name == "im2col" || op_name == "unfold") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("im2col requires 1 input");
        }
        return {dispatchIm2Col(inputs[0], attrs)};
    }

    if (op_name == "col2im" || op_name == "fold") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("col2im requires 1 input");
        }
        return {dispatchCol2Im(inputs[0], attrs)};
    }

    // Tensor manipulation operations
    if (op_name == "expand") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("expand requires 1 input");
        }
        std::vector<int64_t> shape;
        auto shape_str = attrs.at("shape");
        // Parse shape string (format: "2,3,4")
        size_t start = 0;
        size_t end = shape_str.find(',');
        while (end != std::string::npos) {
            shape.push_back(std::stoll(shape_str.substr(start, end - start)));
            start = end + 1;
            end = shape_str.find(',', start);
        }
        shape.push_back(std::stoll(shape_str.substr(start)));
        return {dispatchExpand(inputs[0], shape)};
    }

    if (op_name == "cat" || op_name == "concatenate") {
        if (inputs.size() < 2) {
            throw std::invalid_argument("cat requires at least 2 inputs");
        }
        int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : 0;
        std::vector<Tensor> input_tensors(inputs.begin(), inputs.end());
        return {dispatchCat(input_tensors, dim)};
    }

    if (op_name == "clamp") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("clamp requires 1 input");
        }
        float min_value = attrs.contains("min") ? std::stof(attrs.at("min")) : -std::numeric_limits<float>::infinity();
        float max_value = attrs.contains("max") ? std::stof(attrs.at("max")) : std::numeric_limits<float>::infinity();
        return {dispatchClamp(inputs[0], min_value, max_value)};
    }

    if (op_name == "clamp_min") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("clamp_min requires 1 input");
        }
        float min_value = attrs.contains("min") ? std::stof(attrs.at("min")) : -std::numeric_limits<float>::infinity();
        return {dispatchClamp(inputs[0], min_value, std::numeric_limits<float>::infinity())};
    }

    if (op_name == "clamp_max") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("clamp_max requires 1 input");
        }
        float max_value = attrs.contains("max") ? std::stof(attrs.at("max")) : std::numeric_limits<float>::infinity();
        return {dispatchClamp(inputs[0], -std::numeric_limits<float>::infinity(), max_value)};
    }

    if (op_name == "dot") {
        if (inputs.size() != 2) {
            throw std::invalid_argument("dot requires 2 inputs");
        }
        return {dispatchDot(inputs[0], inputs[1])};
    }

    // BatchNorm2d operations
    if (op_name == "batchnorm2d_forward") {
        if (inputs.size() < 3) {
            throw std::invalid_argument("batchnorm2d_forward requires at least 3 inputs (input, mean, var)");
        }
        const Tensor* gamma = (inputs.size() > 3) ? &inputs[3] : nullptr;
        const Tensor* beta = (inputs.size() > 4) ? &inputs[4] : nullptr;
        float epsilon = attrs.contains("eps") ? std::stof(attrs.at("eps")) : 1e-5f;
        return {dispatchBatchNorm2dForward(inputs[0], inputs[1], inputs[2], gamma, beta, epsilon)};
    }

    if (op_name == "batchnorm2d_forward_affine") {
        // batchnorm2d_forward_affine is the same as batchnorm2d_forward with weight and bias
        // Inputs: input, mean, var, weight, bias
        if (inputs.size() < 5) {
            throw std::invalid_argument("batchnorm2d_forward_affine requires 5 inputs (input, mean, var, weight, bias)");
        }
        float epsilon = attrs.contains("eps") ? std::stof(attrs.at("eps")) : 1e-5f;
        return {dispatchBatchNorm2dForward(inputs[0], inputs[1], inputs[2], &inputs[3], &inputs[4], epsilon)};
    }

    if (op_name == "batchnorm2d_backward") {
        if (inputs.size() < 4) {
            throw std::invalid_argument("batchnorm2d_backward requires at least 4 inputs (grad_output, input, mean, var)");
        }
        const Tensor* gamma = (inputs.size() > 4) ? &inputs[4] : nullptr;
        float epsilon = attrs.contains("eps") ? std::stof(attrs.at("eps")) : 1e-5f;
        auto [grad_input, grad_gamma, grad_beta] = dispatchBatchNorm2dBackward(
            inputs[0], inputs[1], inputs[2], inputs[3], gamma, epsilon);
        return {grad_input, grad_gamma, grad_beta};
    }

    if (op_name == "batchnorm2d_mean_var") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("batchnorm2d_mean_var requires 1 input");
        }
        auto [mean, variance] = dispatchBatchNorm2dMeanVar(inputs[0]);
        return {mean, variance};
    }

    if (op_name == "batchnorm2d_update_running_stats") {
        // CPU fallback for updating running statistics
        // This operation is infrequent (once per batch) and works with small tensors (C channels)
        if (inputs.size() != 4) {
            throw std::invalid_argument("batchnorm2d_update_running_stats requires 4 inputs (running_mean, running_var, batch_mean, batch_var)");
        }

        float momentum = 0.1f;
        if (attrs.contains("momentum")) {
            momentum = std::stof(attrs.at("momentum"));
        }

        // Move tensors to CPU, perform update, then move back to Vulkan
        Tensor cpu_running_mean = inputs[0].to(Device::cpu());
        Tensor cpu_running_var = inputs[1].to(Device::cpu());
        Tensor cpu_batch_mean = inputs[2].to(Device::cpu());
        Tensor cpu_batch_var = inputs[3].to(Device::cpu());

        // Perform exponential moving average update on CPU
        int64_t C = cpu_batch_mean.numel();
        if (cpu_running_mean.dtype() == DType::Float32) {
            float* running_mean_data = cpu_running_mean.data<float>();
            float* running_var_data = cpu_running_var.data<float>();
            const float* batch_mean_data = cpu_batch_mean.data<float>();
            const float* batch_var_data = cpu_batch_var.data<float>();

            for (int64_t i = 0; i < C; ++i) {
                running_mean_data[i] = (1.0f - momentum) * running_mean_data[i] + momentum * batch_mean_data[i];
                running_var_data[i] = (1.0f - momentum) * running_var_data[i] + momentum * batch_var_data[i];
            }
        } else if (cpu_running_mean.dtype() == DType::Float64) {
            double* running_mean_data = cpu_running_mean.data<double>();
            double* running_var_data = cpu_running_var.data<double>();
            const double* batch_mean_data = cpu_batch_mean.data<double>();
            const double* batch_var_data = cpu_batch_var.data<double>();
            double momentum_d = static_cast<double>(momentum);

            for (int64_t i = 0; i < C; ++i) {
                running_mean_data[i] = (1.0 - momentum_d) * running_mean_data[i] + momentum_d * batch_mean_data[i];
                running_var_data[i] = (1.0 - momentum_d) * running_var_data[i] + momentum_d * batch_var_data[i];
            }
        } else if (cpu_running_mean.dtype() == DType::Float16) {
            // For Float16, we work with the raw half-precision data
            // Use the half-precision conversion utilities
            uint16_t* running_mean_data = reinterpret_cast<uint16_t*>(cpu_running_mean.data_ptr());
            uint16_t* running_var_data = reinterpret_cast<uint16_t*>(cpu_running_var.data_ptr());
            const uint16_t* batch_mean_data = reinterpret_cast<const uint16_t*>(cpu_batch_mean.data_ptr());
            const uint16_t* batch_var_data = reinterpret_cast<const uint16_t*>(cpu_batch_var.data_ptr());

            // Helper lambda to convert half to float
            auto half_to_float = [](uint16_t h) -> float {
                uint32_t sign = (h & 0x8000) << 16;
                uint32_t exp = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;

                if (exp == 0) {
                    if (mant == 0) {
                        uint32_t result = sign;
                        return *reinterpret_cast<float*>(&result);
                    }
                    // Denormalized
                    while (!(mant & 0x400)) {
                        mant <<= 1;
                        exp--;
                    }
                    exp++;
                    mant &= ~0x400;
                    exp += 127 - 15;
                    uint32_t result = sign | (exp << 23) | (mant << 13);
                    return *reinterpret_cast<float*>(&result);
                } else if (exp == 31) {
                    uint32_t result = sign | 0x7F800000 | (mant << 13);
                    return *reinterpret_cast<float*>(&result);
                }
                exp += 127 - 15;
                uint32_t result = sign | (exp << 23) | (mant << 13);
                return *reinterpret_cast<float*>(&result);
            };

            // Helper lambda to convert float to half
            auto float_to_half = [](float f) -> uint16_t {
                uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
                uint32_t sign = (bits >> 16) & 0x8000;
                int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
                uint32_t mant = bits & 0x7FFFFF;

                if (exp <= 0) {
                    if (exp < -10) return static_cast<uint16_t>(sign);
                    mant |= 0x800000;
                    uint32_t shift = 14 - exp;
                    mant >>= shift;
                    return static_cast<uint16_t>(sign | mant);
                } else if (exp >= 31) {
                    return static_cast<uint16_t>(sign | 0x7C00);
                }
                return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
            };

            for (int64_t i = 0; i < C; ++i) {
                float rm = half_to_float(running_mean_data[i]);
                float rv = half_to_float(running_var_data[i]);
                float bm = half_to_float(batch_mean_data[i]);
                float bv = half_to_float(batch_var_data[i]);

                rm = (1.0f - momentum) * rm + momentum * bm;
                rv = (1.0f - momentum) * rv + momentum * bv;

                running_mean_data[i] = float_to_half(rm);
                running_var_data[i] = float_to_half(rv);
            }
        } else {
            throw std::runtime_error("Unsupported dtype for batchnorm2d_update_running_stats");
        }

        // Move updated tensors back to Vulkan
        return {cpu_running_mean.to(inputs[0].device()), cpu_running_var.to(inputs[1].device())};
    }

    // Pooling operations (new OpAttributes versions)
    if (op_name == "avg_pool2d_forward") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("avg_pool2d_forward requires 1 input");
        }
        return {dispatchAvgPool2dForward(inputs[0], attrs)};
    }

    if (op_name == "max_pool2d_forward") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("max_pool2d_forward requires 1 input");
        }
        return {dispatchMaxPool2dForward(inputs[0], attrs)};
    }

    if (op_name == "avg_pool2d_backward") {
        if (inputs.size() != 2) {
            throw std::invalid_argument("avg_pool2d_backward requires 2 inputs (grad_output, input)");
        }
        return {dispatchAvgPool2dBackward(inputs[0], inputs[1], attrs)};
    }

    if (op_name == "max_pool2d_backward") {
        if (inputs.size() != 2) {
            throw std::invalid_argument("max_pool2d_backward requires 2 inputs (grad_output, indices)");
        }
        // Extract H_in and W_in from attributes (required for output shape)
        int64_t H_in = std::stoll(attrs.at("H_in"));
        int64_t W_in = std::stoll(attrs.at("W_in"));
        return {dispatchMaxPool2dBackwardWithIndices(inputs[0], inputs[1], H_in, W_in)};
    }

    // Conv2d forward operation
    if (op_name == "conv2d_forward") {
        if (inputs.size() < 2) {
            throw std::invalid_argument("conv2d_forward requires at least 2 inputs (input, weight)");
        }
        return {dispatchConv2dForward(inputs[0], inputs[1], attrs)};
    }

    // Full operation - create tensor filled with specific value
    if (op_name == "full") {
        // Extract shape and value from attributes
        std::vector<int64_t> shape;
        if (attrs.contains("shape")) {
            // Parse shape string like "2,3,4"
            std::string shape_str = attrs.at("shape");
            size_t pos = 0;
            while ((pos = shape_str.find(',')) != std::string::npos) {
                shape.push_back(std::stoll(shape_str.substr(0, pos)));
                shape_str.erase(0, pos + 1);
            }
            if (!shape_str.empty()) {
                shape.push_back(std::stoll(shape_str));
            }
        }
        float value = attrs.contains("value") ? std::stof(attrs.at("value")) : 0.0f;
        DType dtype = DType::Float32;  // Default dtype
        if (attrs.contains("dtype")) {
            // Parse dtype string to enum
            std::string dtype_str = attrs.at("dtype");
            if (dtype_str == "int32" || dtype_str == "Int32") {
                dtype = DType::Int32;
            } else if (dtype_str == "float32" || dtype_str == "Float32") {
                dtype = DType::Float32;
            } else if (dtype_str == "float16" || dtype_str == "Float16") {
                dtype = DType::Float16;
            } else if (dtype_str == "float64" || dtype_str == "Float64") {
                dtype = DType::Float64;
            } else if (dtype_str == "int64" || dtype_str == "Int64") {
                dtype = DType::Int64;
            } else if (dtype_str == "int8" || dtype_str == "Int8") {
                dtype = DType::Int8;
            } else if (dtype_str == "uint8" || dtype_str == "UInt8") {
                dtype = DType::UInt8;
            } else if (dtype_str == "bool" || dtype_str == "Bool") {
                dtype = DType::Bool;
            }
        }
        return {dispatchFull(shape, value, dtype)};
    }

    // Ones operation - create tensor filled with 1.0
    if (op_name == "ones") {
        // Extract shape from attributes
        std::vector<int64_t> shape;
        if (attrs.contains("shape")) {
            // Parse shape string like "2,3,4"
            std::string shape_str = attrs.at("shape");
            size_t pos = 0;
            while ((pos = shape_str.find(',')) != std::string::npos) {
                shape.push_back(std::stoll(shape_str.substr(0, pos)));
                shape_str.erase(0, pos + 1);
            }
            if (!shape_str.empty()) {
                shape.push_back(std::stoll(shape_str));
            }
        }
        DType dtype = DType::Float32;  // Default dtype
        if (attrs.contains("dtype")) {
            std::string dtype_str = attrs.at("dtype");
            if (dtype_str == "float32") {
                dtype = DType::Float32;
            } else if (dtype_str == "float16") {
                dtype = DType::Float16;
            } else if (dtype_str == "int32") {
                dtype = DType::Int32;
            } else if (dtype_str == "int64") {
                dtype = DType::Int64;
            } else if (dtype_str == "float64") {
                dtype = DType::Float64;
            } else if (dtype_str == "int8") {
                dtype = DType::Int8;
            } else if (dtype_str == "uint8") {
                dtype = DType::UInt8;
            } else if (dtype_str == "bool") {
                dtype = DType::Bool;
            }
        }
        return {dispatchOnes(shape, dtype)};
    }

    // Rand operation - create tensor filled with uniform random values [0, 1)
    if (op_name == "rand") {
        std::vector<int64_t> shape;
        if (attrs.contains("shape")) {
            std::string shape_str = attrs.at("shape");
            size_t pos = 0;
            while ((pos = shape_str.find(',')) != std::string::npos) {
                shape.push_back(std::stoll(shape_str.substr(0, pos)));
                shape_str.erase(0, pos + 1);
            }
            if (!shape_str.empty()) {
                shape.push_back(std::stoll(shape_str));
            }
        }
        DType dtype = DType::Float32;
        if (attrs.contains("dtype")) {
            std::string dtype_str = attrs.at("dtype");
            if (dtype_str == "float32" || dtype_str == "Float32") {
                dtype = DType::Float32;
            } else if (dtype_str == "float64" || dtype_str == "Float64") {
                dtype = DType::Float64;
            } else if (dtype_str == "float16" || dtype_str == "Float16") {
                dtype = DType::Float16;
            }
        }
        return {dispatchRand(shape, dtype)};
    }

    // Randn operation - create tensor filled with normal random values
    if (op_name == "randn") {
        std::vector<int64_t> shape;
        if (attrs.contains("shape")) {
            std::string shape_str = attrs.at("shape");
            size_t pos = 0;
            while ((pos = shape_str.find(',')) != std::string::npos) {
                shape.push_back(std::stoll(shape_str.substr(0, pos)));
                shape_str.erase(0, pos + 1);
            }
            if (!shape_str.empty()) {
                shape.push_back(std::stoll(shape_str));
            }
        }
        DType dtype = DType::Float32;
        if (attrs.contains("dtype")) {
            std::string dtype_str = attrs.at("dtype");
            if (dtype_str == "float32" || dtype_str == "Float32") {
                dtype = DType::Float32;
            } else if (dtype_str == "float64" || dtype_str == "Float64") {
                dtype = DType::Float64;
            } else if (dtype_str == "float16" || dtype_str == "Float16") {
                dtype = DType::Float16;
            }
        }
        return {dispatchRandn(shape, dtype)};
    }

    // Repeat operation
    if (op_name == "repeat") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("repeat operation requires exactly 1 input");
        }
        if (!attrs.contains("repeats")) {
            throw std::invalid_argument("repeat operation requires 'repeats' attribute");
        }
        std::vector<int64_t> repeats;
        std::string repeats_str = attrs.at("repeats");
        size_t pos = 0;
        while ((pos = repeats_str.find(',')) != std::string::npos) {
            repeats.push_back(std::stoll(repeats_str.substr(0, pos)));
            repeats_str.erase(0, pos + 1);
        }
        if (!repeats_str.empty()) {
            repeats.push_back(std::stoll(repeats_str));
        }
        return {dispatchRepeat(inputs[0], repeats)};
    }

    // Masked select operation
    if (op_name == "masked_select") {
        if (inputs.size() != 2) {
            throw std::invalid_argument("masked_select operation requires exactly 2 inputs");
        }
        return {dispatchMaskedSelect(inputs[0], inputs[1])};
    }

    // Masked fill operation
    if (op_name == "masked_fill") {
        if (inputs.size() != 2) {
            throw std::invalid_argument("masked_fill operation requires exactly 2 inputs");
        }
        if (!attrs.contains("value")) {
            throw std::invalid_argument("masked_fill operation requires 'value' attribute");
        }
        float value = std::stof(attrs.at("value"));
        return {dispatchMaskedFill(inputs[0], inputs[1], value)};
    }

    // Where operation
    if (op_name == "where") {
        if (inputs.size() != 3) {
            throw std::invalid_argument("where operation requires exactly 3 inputs");
        }
        return {dispatchWhere(inputs[0], inputs[1], inputs[2])};
    }

    throw std::runtime_error("VulkanBackend: Operation '" + op_name + "' not implemented");
    } catch (const std::out_of_range& e) {
        // Catch unordered_map::at errors and provide better diagnostics
        std::string attr_list;
        for (const auto& [key, val] : attrs) {
            if (!attr_list.empty()) attr_list += ", ";
            attr_list += key + "=" + val;
        }
        throw std::runtime_error("VulkanBackend::dispatch '" + op_name +
                               "' failed with out_of_range: " + e.what() +
                               ". Attrs: {" + attr_list + "}");
    }
}

// Helper: Check if two shapes are broadcastable
static bool are_broadcastable(std::span<const int64_t> shape_a,
                               std::span<const int64_t> shape_b) {
    size_t max_ndim = std::max(shape_a.size(), shape_b.size());

    for (size_t i = 0; i < max_ndim; ++i) {
        int64_t dim_a = i < shape_a.size() ? shape_a[shape_a.size() - 1 - i] : 1;
        int64_t dim_b = i < shape_b.size() ? shape_b[shape_b.size() - 1 - i] : 1;

        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }

    return true;
}

// Helper: Compute the broadcasted output shape
static std::vector<int64_t> compute_broadcast_shape(std::span<const int64_t> shape_a,
                                                     std::span<const int64_t> shape_b) {
    size_t max_ndim = std::max(shape_a.size(), shape_b.size());
    std::vector<int64_t> result(max_ndim);

    for (size_t i = 0; i < max_ndim; ++i) {
        int64_t dim_a = i < shape_a.size() ? shape_a[shape_a.size() - 1 - i] : 1;
        int64_t dim_b = i < shape_b.size() ? shape_b[shape_b.size() - 1 - i] : 1;

        if (dim_a == dim_b || dim_a == 1 || dim_b == 1) {
            result[max_ndim - 1 - i] = std::max(dim_a, dim_b);
        } else {
            throw std::runtime_error("Shapes are not broadcastable");
        }
    }

    return result;
}

// Helper: Compute strides for broadcasting
static std::vector<uint32_t> compute_broadcast_strides(std::span<const int64_t> shape,
                                                        std::span<const int64_t> broadcast_shape) {
    std::vector<uint32_t> strides(broadcast_shape.size(), 0);

    // Compute normal strides for the original shape
    std::vector<int64_t> original_strides(shape.size());
    if (!shape.empty()) {
        original_strides.back() = 1;
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            original_strides[i] = original_strides[i + 1] * shape[i + 1];
        }
    }

    // Map to broadcast strides
    int64_t offset = static_cast<int64_t>(broadcast_shape.size()) - static_cast<int64_t>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 1) {
            strides[offset + i] = 0;  // Broadcasting dimension
        } else {
            strides[offset + i] = static_cast<uint32_t>(original_strides[i]);
        }
    }

    return strides;
}

auto VulkanBackend::dispatchBinaryOp(const std::string& op_name,
                                     const Tensor& a, const Tensor& b) -> Tensor {
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    // Convert to vectors for easier manipulation
    std::vector<int64_t> shape_a_vec(a_shape.begin(), a_shape.end());
    std::vector<int64_t> shape_b_vec(b_shape.begin(), b_shape.end());

    // Check if shapes are broadcastable
    if (!are_broadcastable(shape_a_vec, shape_b_vec)) {
        throw std::invalid_argument("Tensors shapes are not broadcastable");
    }

    // Compute output shape
    std::vector<int64_t> output_shape = compute_broadcast_shape(shape_a_vec, shape_b_vec);

    int32_t device_id = a.device().index;

    // Map operation name to opcode
    uint32_t opcode = 0;
    if (op_name == "add") opcode = 0;
    else if (op_name == "sub") opcode = 1;
    else if (op_name == "mul") opcode = 2;
    else if (op_name == "div") opcode = 3;
    else throw std::runtime_error("Unknown binary operation: " + op_name);

    // Check if we can use the legacy float-only fast path
    bool same_shape = std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end());
    bool is_float32 = (a.dtype() == DType::Float32);
    bool is_float64 = (a.dtype() == DType::Float64);

    if (same_shape && (is_float32 || is_float64)) {
        // Fast path: use math shader for same-shape operations
        // Select shader based on dtype: "math" for Float32, "math_f64" for Float64
        std::string shader_name = is_float64 ? "math_f64" : "math";
        auto* pipeline = getPipeline(shader_name, device_id);

        Tensor output(output_shape, a.dtype(), a.device());

        // Prepare push constants - use different structure for Float32 vs Float64
        struct PushConstantsF32 {
            uint32_t n;
            uint32_t op;
            float param;
        };
        struct PushConstantsF64 {
            uint32_t n;
            uint32_t op;
            double param;
        };

        // Initialize the appropriate structure based on dtype
        PushConstantsF32 push_constants_f32;
        PushConstantsF64 push_constants_f64;
        void* push_constants_ptr;
        size_t push_constants_size;

        if (is_float64) {
            push_constants_f64.n = static_cast<uint32_t>(a.numel());
            push_constants_f64.op = opcode;
            push_constants_f64.param = 0.0;
            push_constants_ptr = &push_constants_f64;
            push_constants_size = sizeof(PushConstantsF64);
        } else {
            push_constants_f32.n = static_cast<uint32_t>(a.numel());
            push_constants_f32.op = opcode;
            push_constants_f32.param = 0.0f;
            push_constants_ptr = &push_constants_f32;
            push_constants_size = sizeof(PushConstantsF32);
        }

        // Get VkBuffer handles
        VkBuffer buffer_a = getVulkanBuffer(a.data_ptr());
        VkBuffer buffer_b = getVulkanBuffer(b.data_ptr());
        VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());

        // Calculate buffer sizes
        size_t buffer_size_a = a.numel() * a.dtype_size();
        size_t buffer_size_b = b.numel() * b.dtype_size();
        size_t buffer_size_out = output.numel() * output.dtype_size();

        // Allocate and write descriptor set
        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
            {0, buffer_a}, {1, buffer_b}, {2, buffer_out}
        };
        std::vector<size_t> sizes = {buffer_size_a, buffer_size_b, buffer_size_out};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        // Execute compute shader
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, push_constants_size, push_constants_ptr);

        uint32_t workgroups = (a.numel() + 255) / 256;
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        VkMemoryBarrier memoryBarrier{};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

        vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                            0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    } else {
        // Broadcasting path: use math_broadcast shader
        // Select shader based on dtype
        bool is_float64 = (a.dtype() == DType::Float64);
        bool is_float16 = (a.dtype() == DType::Float16);
        bool is_int8 = (a.dtype() == DType::Int8);
        bool is_uint8 = (a.dtype() == DType::UInt8);
        bool is_int64 = (a.dtype() == DType::Int64);
        bool is_bool = (a.dtype() == DType::Bool);
        std::string shader_name;
        if (is_float64) {
            shader_name = "math_broadcast_f64";
        } else if (is_float16) {
            shader_name = "math_broadcast_f16";
        } else if (is_int8) {
            shader_name = "math_broadcast_i8";
        } else if (is_uint8) {
            shader_name = "math_broadcast_uint8";
        } else if (is_int64) {
            shader_name = "math_broadcast_i64";
        } else if (is_bool) {
            shader_name = "math_broadcast_bool";
        } else {
            shader_name = "math_broadcast";
        }
        auto* pipeline = getPipeline(shader_name, device_id);

        Tensor output(output_shape, a.dtype(), a.device());

        // Compute broadcasting strides
        auto strides_a = compute_broadcast_strides(shape_a_vec, output_shape);
        auto strides_b = compute_broadcast_strides(shape_b_vec, output_shape);

        // Prepare push constants for broadcasting shader
        struct PushConstantsBroadcast {
            uint32_t output_size;
            uint32_t op;
            uint32_t dtype;        // 0=float32, 1=int32
            uint32_t ndim_a;
            uint32_t ndim_b;
            uint32_t ndim_out;
            uint32_t strides_a[8];
            uint32_t strides_b[8];
            uint32_t shape_out[8];
        } push_constants = {};

        int64_t output_numel = 1;
        for (auto dim : output_shape) {
            output_numel *= dim;
        }

        // Determine dtype code
        // Note: for Float64, we use separate shader (math_broadcast_f64), so dtype field is unused
        // But we set it correctly for consistency
        uint32_t dtype_code = 0;  // 0=float32, 1=float64 (for f64 shader), 1=int32 (for regular shader)
        if (a.dtype() == DType::Float64) {
            dtype_code = 1;  // Float64 uses dtype=1 in math_broadcast_f64 shader
        } else if (a.dtype() == DType::Int32) {
            dtype_code = 1;  // Int32 uses dtype=1 in math_broadcast shader
        }

        push_constants.output_size = static_cast<uint32_t>(output_numel);
        push_constants.op = opcode;
        push_constants.dtype = dtype_code;
        push_constants.ndim_a = static_cast<uint32_t>(shape_a_vec.size());
        push_constants.ndim_b = static_cast<uint32_t>(shape_b_vec.size());
        push_constants.ndim_out = static_cast<uint32_t>(output_shape.size());

        // Copy strides and output shape (up to 8 dimensions)
        for (size_t i = 0; i < std::min(size_t(8), strides_a.size()); ++i) {
            push_constants.strides_a[i] = strides_a[i];
        }
        for (size_t i = 0; i < std::min(size_t(8), strides_b.size()); ++i) {
            push_constants.strides_b[i] = strides_b[i];
        }
        for (size_t i = 0; i < std::min(size_t(8), output_shape.size()); ++i) {
            push_constants.shape_out[i] = static_cast<uint32_t>(output_shape[i]);
        }

        // Get VkBuffer handles
        VkBuffer buffer_a = getVulkanBuffer(a.data_ptr());
        VkBuffer buffer_b = getVulkanBuffer(b.data_ptr());
        VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());

        // Calculate buffer sizes
        // For Float16, the shader works with uint32 (packed pairs), so descriptor size needs
        // to cover the full uint32 reads/writes
        size_t buffer_size_a = a.numel() * a.dtype_size();
        size_t buffer_size_b = b.numel() * b.dtype_size();
        size_t buffer_size_out = output_numel * output.dtype_size();
        if (is_float16) {
            // Round up to 4-byte boundary (minimum uint32 size for shader access)
            size_t a_pairs = (a.numel() + 1) / 2;
            size_t b_pairs = (b.numel() + 1) / 2;
            size_t out_pairs = (output_numel + 1) / 2;
            buffer_size_a = a_pairs * 4;
            buffer_size_b = b_pairs * 4;
            buffer_size_out = out_pairs * 4;
        }

        // Allocate and write descriptor set
        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
            {0, buffer_a}, {1, buffer_b}, {2, buffer_out}
        };
        std::vector<size_t> sizes = {buffer_size_a, buffer_size_b, buffer_size_out};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        // Execute compute shader
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsBroadcast), &push_constants);

        // Calculate workgroups - Float16 processes 2 elements per thread
        uint32_t workgroups;
        if (is_float16) {
            uint32_t num_pairs = (output_numel + 1) / 2;
            workgroups = (num_pairs + 255) / 256;
        } else {
            workgroups = (output_numel + 255) / 256;
        }
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        VkMemoryBarrier memoryBarrier{};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

        vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                            0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }
}

auto VulkanBackend::dispatchUnaryOp(const std::string& op_name,
                                    const Tensor& input) -> Tensor {
    int32_t device_id = input.device().index;

    // Create output tensor (convert span to vector)
    auto input_shape = input.shape();
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // Determine shader and opcode based on operation
    std::string shader_name;
    uint32_t opcode = 0;

    // Trigonometric operations: 0=sin, 1=cos, 2=tan, 3=asin, 4=acos, 5=atan
    if (op_name == "sin") { shader_name = "trigonometric"; opcode = 0; }
    else if (op_name == "cos") { shader_name = "trigonometric"; opcode = 1; }
    else if (op_name == "tan") { shader_name = "trigonometric"; opcode = 2; }
    else if (op_name == "asin") { shader_name = "trigonometric"; opcode = 3; }
    else if (op_name == "acos") { shader_name = "trigonometric"; opcode = 4; }
    else if (op_name == "atan") { shader_name = "trigonometric"; opcode = 5; }
    // Hyperbolic operations: 0=sinh, 1=cosh, 2=tanh
    else if (op_name == "sinh") { shader_name = "hyperbolic"; opcode = 0; }
    else if (op_name == "cosh") { shader_name = "hyperbolic"; opcode = 1; }
    else if (op_name == "tanh") { shader_name = "hyperbolic"; opcode = 2; }
    // Math operations: 4=sqrt, 5=exp, 6=log, 7=neg, 8=abs, 10=sign
    else if (op_name == "sqrt") { shader_name = "math"; opcode = 4; }
    else if (op_name == "exp") { shader_name = "math"; opcode = 5; }
    else if (op_name == "log") { shader_name = "math"; opcode = 6; }
    else if (op_name == "neg") { shader_name = "math"; opcode = 7; }
    else if (op_name == "abs") { shader_name = "math"; opcode = 8; }
    else if (op_name == "sign") { shader_name = "math"; opcode = 10; }
    else if (op_name == "floor") { shader_name = "math"; opcode = 11; }
    else if (op_name == "ceil") { shader_name = "math"; opcode = 12; }
    else if (op_name == "round") { shader_name = "math"; opcode = 13; }
    else if (op_name == "trunc") { shader_name = "math"; opcode = 14; }
    else if (op_name == "reciprocal") { shader_name = "math"; opcode = 15; }
    else throw std::runtime_error("Unknown unary operation: " + op_name);

    // Select correct pipeline based on dtype for math operations
    // For Float64, append "_f64" suffix to shader name (only for "math" shader)
    // For Int32, use "math_i32" shader
    if (shader_name == "math") {
        if (input.dtype() == DType::Float64) {
            shader_name = "math_f64";
        } else if (input.dtype() == DType::Int32) {
            shader_name = "math_i32";
        }
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    // Prepare push constants - use different structure for Float32 vs Float64
    bool is_float64 = (input.dtype() == DType::Float64);
    struct PushConstantsF32 {
        uint32_t n;
        uint32_t op;
        float param;
    };
    struct PushConstantsF64 {
        uint32_t n;
        uint32_t op;
        double param;
    };

    PushConstantsF32 push_constants_f32;
    PushConstantsF64 push_constants_f64;
    void* push_constants_ptr;
    size_t push_constants_size;

    if (is_float64) {
        push_constants_f64.n = static_cast<uint32_t>(input.numel());
        push_constants_f64.op = opcode;
        push_constants_f64.param = 0.0;
        push_constants_ptr = &push_constants_f64;
        push_constants_size = sizeof(PushConstantsF64);
    } else {
        push_constants_f32.n = static_cast<uint32_t>(input.numel());
        push_constants_f32.op = opcode;
        push_constants_f32.param = 0.0f;
        push_constants_ptr = &push_constants_f32;
        push_constants_size = sizeof(PushConstantsF32);
    }

    // Get VkBuffer handles from tensor data pointers
    VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    // For trigonometric/hyperbolic shaders: Binding 0: input, Binding 1: output
    // For math/math_f64 shaders: Binding 0: input, Binding 1: unused (set to input), Binding 2: output
    std::vector<std::pair<uint32_t, VkBuffer>> bindings;
    std::vector<size_t> sizes;

    if (shader_name == "math" || shader_name == "math_f64" || shader_name == "math_i32") {
        bindings = {
            {0, buffer_in},
            {1, buffer_in},  // Unary ops don't use binding 1, but descriptor set expects it
            {2, buffer_out}
        };
        sizes = {buffer_size_in, buffer_size_in, buffer_size_out};
    } else {
        // trigonometric and hyperbolic shaders use simpler layout
        bindings = {
            {0, buffer_in},
            {1, buffer_out}
        };
        sizes = {buffer_size_in, buffer_size_out};
    }

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, push_constants_size, push_constants_ptr);

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchUnaryOpWithParam(const std::string& op_name,
                                              const Tensor& input,
                                              float param) -> Tensor {
    int32_t device_id = input.device().index;

    // Select correct pipeline based on dtype: "math" for Float32, "math_f64" for Float64
    std::string shader_name = (input.dtype() == DType::Float64) ? "math_f64" : "math";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor (convert span to vector)
    auto input_shape = input.shape();
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // Map operation name to opcode (see math.comp shader)
    // 0=add, 1=sub, 2=mul, 3=div, 4=sqrt, 5=exp, 6=log, 7=neg, 8=abs, 9=pow, 10=sign
    uint32_t opcode = 0;
    if (op_name == "pow") opcode = 9;
    else throw std::runtime_error("Unknown parameterized unary operation: " + op_name);

    // Prepare push constants - use different structure for Float32 vs Float64
    bool is_float64 = (input.dtype() == DType::Float64);
    struct PushConstantsF32 {
        uint32_t n;
        uint32_t op;
        float param;
    };
    struct PushConstantsF64 {
        uint32_t n;
        uint32_t op;
        double param;
    };

    PushConstantsF32 push_constants_f32;
    PushConstantsF64 push_constants_f64;
    void* push_constants_ptr;
    size_t push_constants_size;

    if (is_float64) {
        push_constants_f64.n = static_cast<uint32_t>(input.numel());
        push_constants_f64.op = opcode;
        push_constants_f64.param = static_cast<double>(param);
        push_constants_ptr = &push_constants_f64;
        push_constants_size = sizeof(PushConstantsF64);
    } else {
        push_constants_f32.n = static_cast<uint32_t>(input.numel());
        push_constants_f32.op = opcode;
        push_constants_f32.param = param;
        push_constants_ptr = &push_constants_f32;
        push_constants_size = sizeof(PushConstantsF32);
    }

    // Get VkBuffer handles from tensor data pointers
    VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    // Binding 0: input, Binding 1: unused (set to input), Binding 2: output
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_in},
        {1, buffer_in},  // Unary ops don't use binding 1, but descriptor set expects it
        {2, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, push_constants_size, push_constants_ptr);

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchTrigonometricOp(const std::string& op_name,
                                             const Tensor& input) -> Tensor {
    // Map operation name to opcode (see trigonometric.comp shader)
    // 0=sin, 1=cos, 2=tan, 3=asin, 4=acos, 5=atan
    uint32_t opcode = 0;
    if (op_name == "sin") opcode = 0;
    else if (op_name == "cos") opcode = 1;
    else if (op_name == "tan") opcode = 2;
    else if (op_name == "asin") opcode = 3;
    else if (op_name == "acos") opcode = 4;
    else if (op_name == "atan") opcode = 5;
    else throw std::runtime_error("Unknown trigonometric operation: " + op_name);

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("trigonometric", device_id);

    auto shape = input.shape();
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // Prepare push constants
    struct PushConstants {
        uint32_t n;   // Number of elements
        uint32_t op;  // Operation code
    } push_constants;
    push_constants.n = static_cast<uint32_t>(input.numel());
    push_constants.op = opcode;

    // Get VkBuffer handles
    VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_in}, {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchHyperbolicOp(const std::string& op_name,
                                          const Tensor& input) -> Tensor {
    // Map operation name to opcode (see hyperbolic.comp shader)
    // 0=sinh, 1=cosh, 2=tanh
    uint32_t opcode = 0;
    if (op_name == "sinh") opcode = 0;
    else if (op_name == "cosh") opcode = 1;
    else if (op_name == "tanh") opcode = 2;
    else throw std::runtime_error("Unknown hyperbolic operation: " + op_name);

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("hyperbolic", device_id);

    auto shape = input.shape();
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // Prepare push constants
    struct PushConstants {
        uint32_t n;   // Number of elements
        uint32_t op;  // Operation code
    } push_constants;
    push_constants.n = static_cast<uint32_t>(input.numel());
    push_constants.op = opcode;

    // Get VkBuffer handles
    VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_in}, {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchComparisonOp(const std::string& op_name,
                                          const Tensor& a, const Tensor& b) -> Tensor {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (!std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end())) {
        throw std::invalid_argument("Tensors must have same shape for comparison op");
    }

    int32_t device_id = a.device().index;

    // Select shader based on input dtype
    std::string shader_name;
    switch (a.dtype()) {
        case DType::Bool:
            shader_name = "comparison_bool";
            break;
        case DType::Float64:
            shader_name = "comparison_f64";
            break;
        case DType::Int32:
            shader_name = "comparison_i32";
            break;
        case DType::Int64:
            shader_name = "comparison_i64";
            break;
        default:
            // Float32 and Float16 use the default comparison shader
            shader_name = "comparison";
            break;
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor (convert span to vector)
    // Output is boolean values (DType::Bool stored as uint8 with 0 or 1)
    std::vector<int64_t> output_shape(a_shape.begin(), a_shape.end());
    Tensor output(output_shape, DType::Bool, a.device());

    // Map operation name to opcode (see comparison.comp shader)
    // 0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge
    uint32_t opcode = 0;
    if (op_name == "eq") opcode = 0;
    else if (op_name == "ne") opcode = 1;
    else if (op_name == "lt") opcode = 2;
    else if (op_name == "le") opcode = 3;
    else if (op_name == "gt") opcode = 4;
    else if (op_name == "ge") opcode = 5;
    else throw std::runtime_error("Unknown comparison operation: " + op_name);

    // Prepare push constants
    struct PushConstants {
        uint32_t n;   // Number of elements
        uint32_t op;  // Operation code
    } push_constants;
    push_constants.n = static_cast<uint32_t>(a.numel());
    push_constants.op = opcode;

    // Get VkBuffer handles from tensor data pointers
    VkBuffer buffer_a = getVulkanBuffer(a.data_ptr());
    VkBuffer buffer_b = getVulkanBuffer(b.data_ptr());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_a = a.numel() * a.dtype_size();
    size_t buffer_size_b = b.numel() * b.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    // Binding 0: input A, Binding 1: input B, Binding 2: output
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_a},
        {1, buffer_b},
        {2, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_a, buffer_size_b, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch compute workgroups
    uint32_t workgroups = (a.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier to ensure shader writes complete
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchReduction(const std::string& op_name,
                                      const Tensor& input,
                                      int64_t dim, bool keepdim) -> Tensor {
    // Special case: handle empty tensors
    if (input.numel() == 0) {
        // For empty tensors, return identity value
        // sum: 0, mean: 0, max: -inf, min: +inf
        float identity_value = 0.0f;
        if (op_name == "max") {
            identity_value = -std::numeric_limits<float>::infinity();
        } else if (op_name == "min") {
            identity_value = std::numeric_limits<float>::infinity();
        }

        // Calculate output shape
        std::vector<int64_t> out_shape;
        if (dim < 0) {
            out_shape = {1};
        } else {
            auto input_shape = input.shape();
            out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
            if (keepdim) {
                out_shape[dim] = 1;
            } else {
                out_shape.erase(out_shape.begin() + dim);
            }
        }

        // Create result tensor on CPU with identity value
        Tensor result_cpu(out_shape, input.dtype(), Device::cpu());
        float* data = result_cpu.data<float>();
        for (int64_t i = 0; i < result_cpu.numel(); i++) {
            data[i] = identity_value;
        }
        return result_cpu.to(input.device());
    }

    int32_t device_id = input.device().index;
    auto input_shape = input.shape();

    // Handle dimension specification:
    // - INT64_MIN means "reduce all elements" (full reduction to scalar)
    // - Negative values like -1, -2 mean indexing from the end
    bool full_reduction = (dim == INT64_MIN);
    if (!full_reduction && dim < 0) {
        dim = static_cast<int64_t>(input_shape.size()) + dim;
    }

    // Select correct pipeline based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    bool is_int32 = (input.dtype() == DType::Int32);
    std::string shader_name;
    if (is_float64) {
        shader_name = "reduction_f64";
    } else if (is_float16) {
        shader_name = "reduction_f16";
    } else if (is_int32) {
        shader_name = "reduction_i32";
    } else {
        shader_name = "reduction";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Map operation name to opcode for push constants
    // 0=sum, 1=mean, 2=max, 3=min (see reduction.comp shader)
    uint32_t op_code = 0;
    if (op_name == "sum") op_code = 0;
    else if (op_name == "mean") op_code = 1;
    else if (op_name == "max") op_code = 2;
    else if (op_name == "min") op_code = 3;
    else {
        throw std::invalid_argument("Unknown reduction operation: " + op_name);
    }

    // Calculate output shape
    std::vector<int64_t> out_shape;
    if (full_reduction) {
        // Full reduction: output is a scalar (shape {1} or {} depending on keepdim)
        if (keepdim) {
            out_shape.assign(input_shape.size(), 1);
        } else {
            out_shape = {};  // Scalar
        }
    } else {
        // Dimensional reduction
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    Tensor output(out_shape, input.dtype(), input.device());

    // Get VkBuffer handles from tensor data pointers
    VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    // Binding 0: input, Binding 1: output (matches reduction.comp shader layout)
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate inner_size (product of dimensions after reduction dimension)
    uint32_t inner_size = 1;
    if (!full_reduction) {
        for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
            inner_size *= static_cast<uint32_t>(input_shape[i]);
        }
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants for reduction operation (must match shader layout)
    struct {
        uint32_t n;             // Total number of elements
        uint32_t reduce_size;   // Size of dimension to reduce
        uint32_t outer_size;    // Number of output elements
        uint32_t inner_size;    // Product of dimensions after reduction dim
        uint32_t op;            // Operation code (0=sum, 1=mean, 2=max, 3=min)
    } pushConstants;

    pushConstants.n = static_cast<uint32_t>(input.numel());
    pushConstants.reduce_size = full_reduction ? pushConstants.n : static_cast<uint32_t>(input_shape[dim]);
    pushConstants.outer_size = full_reduction ? 1 : static_cast<uint32_t>(output.numel());
    pushConstants.inner_size = inner_size;
    pushConstants.op = op_code;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch workgroups
    // For Float16: each workgroup handles TWO adjacent output elements (one packed word)
    uint32_t workgroups;
    if (is_float16) {
        workgroups = (pushConstants.outer_size + 1) / 2;
    } else {
        // Each workgroup has 256 threads and reduces one output element
        workgroups = pushConstants.outer_size;
    }
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    // Synchronize to ensure GPU has completed before reading results
    synchronize(device_id);

    return output;
}

auto VulkanBackend::dispatchMatmul(const Tensor& a, const Tensor& b) -> Tensor {
    // Optimized matmul with proper buffer binding and tiled execution
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    // Handle 1D vector × 2D matrix case
    if (a_shape.size() == 1 && b_shape.size() == 2) {
        // Validate dimensions
        if (a_shape[0] != b_shape[0]) {
            throw std::invalid_argument("Incompatible dimensions for vector-matrix matmul");
        }

        // Treat 1D vector as row vector: (N,) -> (1, N)
        // Then matmul becomes: (1, N) × (N, K) = (1, K)
        // Finally squeeze to get (K,)
        Tensor a_2d = a.unsqueeze(0);  // (N,) -> (1, N)
        Tensor result_2d = dispatchMatmul(a_2d, b);  // (1, K)

        // Return as 1D tensor
        std::vector<int64_t> result_shape = {result_2d.shape()[1]};
        return result_2d.reshape(result_shape);
    }

    if (a_shape.size() != 2 || b_shape.size() != 2) {
        throw std::invalid_argument("Matmul requires 2D tensors");
    }
    if (a_shape[1] != b_shape[0]) {
        throw std::invalid_argument("Incompatible dimensions for matmul");
    }

    int32_t device_id = a.device().index;

    // Select correct pipeline based on dtype
    bool is_float64 = (a.dtype() == DType::Float64);
    bool is_float16 = (a.dtype() == DType::Float16);
    bool is_int32 = (a.dtype() == DType::Int32);
    std::string shader_name;
    if (is_float64) {
        shader_name = "matmul_f64";
    } else if (is_float16) {
        shader_name = "matmul_f16";
    } else if (is_int32) {
        shader_name = "matmul_i32";
    } else {
        shader_name = "matmul";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {a_shape[0], b_shape[1]};
    Tensor output(out_shape, a.dtype(), a.device());

    // Get VkBuffer handles
    VkBuffer buffer_a = getVulkanBuffer(a.data_ptr());
    VkBuffer buffer_b = getVulkanBuffer(b.data_ptr());
    VkBuffer buffer_c = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_a = a.numel() * a.dtype_size();
    size_t buffer_size_b = b.numel() * b.dtype_size();
    size_t buffer_size_c = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_a},
        {1, buffer_b},
        {2, buffer_c}
    };
    std::vector<size_t> sizes = {buffer_size_a, buffer_size_b, buffer_size_c};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants for matrix dimensions
    struct PushConstants {
        uint32_t M;  // rows of A
        uint32_t N;  // cols of B
        uint32_t K;  // cols of A / rows of B
    } push_constants;

    push_constants.M = static_cast<uint32_t>(a_shape[0]);
    push_constants.N = static_cast<uint32_t>(b_shape[1]);
    push_constants.K = static_cast<uint32_t>(a_shape[1]);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Tile-based dispatch (16x16 workgroups as defined in shader)
    // For Float16: each thread processes 2 columns, so halve workgroups_x
    uint32_t workgroups_x;
    if (is_float16) {
        // Each thread handles 2 adjacent columns
        workgroups_x = ((push_constants.N + 1) / 2 + 15) / 16;
    } else {
        workgroups_x = (push_constants.N + 15) / 16;
    }
    uint32_t workgroups_y = (push_constants.M + 15) / 16;
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchDot(const Tensor& a, const Tensor& b) -> Tensor {
    // Dot product: element-wise multiply followed by sum
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    if (a_shape.size() != 1 || b_shape.size() != 1) {
        throw std::invalid_argument("Dot product requires 1D tensors");
    }
    if (a_shape[0] != b_shape[0]) {
        throw std::invalid_argument("Dot product tensors must have same size");
    }

    // Element-wise multiply
    Tensor product = dispatchBinaryOp("mul", a, b);

    // Sum all elements (dim=-1 means all dimensions, keepdim=false for scalar result)
    Tensor result = dispatchReduction("sum", product, 0, false);

    return result;
}

auto VulkanBackend::dispatchConv2d(const Tensor& input, const Tensor& weight,
                                   const Tensor* bias, int64_t stride,
                                   int64_t padding, int64_t dilation,
                                   int64_t groups) -> Tensor {
    // Get input/weight dimensions
    // input: [batch, in_channels, in_height, in_width]
    // weight: [out_channels, in_channels/groups, kernel_h, kernel_w]
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Calculate output dimensions
    int64_t out_height = (in_height + 2*padding - dilation*(kernel_h-1) - 1) / stride + 1;
    int64_t out_width = (in_width + 2*padding - dilation*(kernel_w-1) - 1) / stride + 1;

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("conv2d", device_id);

    // Create output tensor
    std::vector<int64_t> out_shape = {batch, out_channels, out_height, out_width};
    Tensor output(out_shape, input.dtype(), input.device());

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Set push constants for conv2d parameters
    struct PushConstants {
        uint32_t batch;
        uint32_t in_channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_channels;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride;
        uint32_t padding;
        uint32_t dilation;
        uint32_t has_bias;
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.in_channels = static_cast<uint32_t>(in_channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_channels = static_cast<uint32_t>(out_channels);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride = static_cast<uint32_t>(stride);
    push_constants.padding = static_cast<uint32_t>(padding);
    push_constants.dilation = static_cast<uint32_t>(dilation);
    push_constants.has_bias = bias ? 1 : 0;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (16x16 threads per workgroup as defined in shader)
    uint32_t workgroups_x = (out_width + 15) / 16;
    uint32_t workgroups_y = (out_height + 15) / 16;
    uint32_t workgroups_z = out_channels;
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Conv2d Backward Input - Gradient w.r.t. input
 *
 * Computes gradient of input using transposed convolution (col2im pattern).
 * For each input pixel, accumulates gradients from all output positions
 * that used it during forward pass.
 */
auto VulkanBackend::dispatchConv2dBackwardInput(
    const Tensor& grad_output,
    const Tensor& weight,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    const std::vector<int64_t>& input_shape) -> Tensor {

    // Extract dimensions
    auto grad_shape = grad_output.shape();
    auto weight_shape = weight.shape();

    int64_t batch = grad_shape[0];
    int64_t channels_out = grad_shape[1];
    int64_t height_out = grad_shape[2];
    int64_t width_out = grad_shape[3];

    int64_t channels_in = input_shape[1];
    int64_t height_in = input_shape[2];
    int64_t width_in = input_shape[3];

    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    int32_t device_id = grad_output.device().index;
    auto* pipeline = getPipeline("conv2d_backward_input", device_id);

    // Create gradient input tensor
    Tensor grad_input(input_shape, grad_output.dtype(), grad_output.device());

    // Get VkBuffer handles
    VkBuffer buffer_grad_out = getVulkanBuffer(grad_output.data_ptr());
    VkBuffer buffer_weight = getVulkanBuffer(weight.data_ptr());
    VkBuffer buffer_grad_in = getVulkanBuffer(grad_input.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_weight = weight.numel() * weight.dtype_size();
    size_t buffer_size_grad_in = grad_input.numel() * grad_input.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_grad_out},  // grad_output
        {1, buffer_weight},    // weight
        {2, buffer_grad_in}    // grad_input (output)
    };
    std::vector<size_t> sizes = {buffer_size_grad_out, buffer_size_weight, buffer_size_grad_in};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct PushConstants {
        uint32_t batch;
        uint32_t channels_in;
        uint32_t channels_out;
        uint32_t height_in;
        uint32_t width_in;
        uint32_t height_out;
        uint32_t width_out;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride;
        uint32_t padding;
        uint32_t dilation;
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels_in = static_cast<uint32_t>(channels_in);
    push_constants.channels_out = static_cast<uint32_t>(channels_out);
    push_constants.height_in = static_cast<uint32_t>(height_in);
    push_constants.width_in = static_cast<uint32_t>(width_in);
    push_constants.height_out = static_cast<uint32_t>(height_out);
    push_constants.width_out = static_cast<uint32_t>(width_out);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride = static_cast<uint32_t>(stride);
    push_constants.padding = static_cast<uint32_t>(padding);
    push_constants.dilation = static_cast<uint32_t>(dilation);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (256 threads per workgroup as defined in shader)
    int64_t total_elements = batch * channels_in * height_in * width_in;
    uint32_t workgroups = static_cast<uint32_t>((total_elements + 255) / 256);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

/**
 * @brief Conv2d Backward Weight - Gradient w.r.t. weights
 *
 * Computes gradient of weights by correlating input patches with grad_output
 * across all batch samples and spatial positions.
 */
auto VulkanBackend::dispatchConv2dBackwardWeight(
    const Tensor& grad_output,
    const Tensor& input,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    const std::vector<int64_t>& weight_shape) -> Tensor {

    // Extract dimensions
    auto grad_shape = grad_output.shape();
    auto input_shape = input.shape();

    int64_t batch = grad_shape[0];
    int64_t channels_out = grad_shape[1];
    int64_t height_out = grad_shape[2];
    int64_t width_out = grad_shape[3];

    int64_t channels_in = input_shape[1];
    int64_t height_in = input_shape[2];
    int64_t width_in = input_shape[3];

    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    int32_t device_id = grad_output.device().index;
    auto* pipeline = getPipeline("conv2d_backward_weight", device_id);

    // Create gradient weight tensor
    Tensor grad_weight(weight_shape, grad_output.dtype(), grad_output.device());

    // Get VkBuffer handles
    VkBuffer buffer_grad_out = getVulkanBuffer(grad_output.data_ptr());
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_grad_weight = getVulkanBuffer(grad_weight.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_grad_weight = grad_weight.numel() * grad_weight.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_grad_out},    // grad_output
        {1, buffer_input},       // input
        {2, buffer_grad_weight}  // grad_weight (output)
    };
    std::vector<size_t> sizes = {buffer_size_grad_out, buffer_size_input, buffer_size_grad_weight};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct PushConstants {
        uint32_t batch;
        uint32_t channels_in;
        uint32_t channels_out;
        uint32_t height_in;
        uint32_t width_in;
        uint32_t height_out;
        uint32_t width_out;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride;
        uint32_t padding;
        uint32_t dilation;
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels_in = static_cast<uint32_t>(channels_in);
    push_constants.channels_out = static_cast<uint32_t>(channels_out);
    push_constants.height_in = static_cast<uint32_t>(height_in);
    push_constants.width_in = static_cast<uint32_t>(width_in);
    push_constants.height_out = static_cast<uint32_t>(height_out);
    push_constants.width_out = static_cast<uint32_t>(width_out);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride = static_cast<uint32_t>(stride);
    push_constants.padding = static_cast<uint32_t>(padding);
    push_constants.dilation = static_cast<uint32_t>(dilation);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (256 threads per workgroup as defined in shader)
    int64_t total_weight_elements = channels_out * channels_in * kernel_h * kernel_w;
    uint32_t workgroups = static_cast<uint32_t>((total_weight_elements + 255) / 256);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_weight;
}

/**
 * @brief Conv2d Backward Bias - Gradient w.r.t. bias
 *
 * Computes gradient of bias by summing grad_output across batch,
 * height, and width dimensions for each output channel.
 */
auto VulkanBackend::dispatchConv2dBackwardBias(const Tensor& grad_output) -> Tensor {
    // Extract dimensions
    auto grad_shape = grad_output.shape();

    int64_t batch = grad_shape[0];
    int64_t channels_out = grad_shape[1];
    int64_t height_out = grad_shape[2];
    int64_t width_out = grad_shape[3];

    int32_t device_id = grad_output.device().index;
    auto* pipeline = getPipeline("conv2d_backward_bias", device_id);

    // Create gradient bias tensor
    std::vector<int64_t> bias_shape = {channels_out};
    Tensor grad_bias(bias_shape, grad_output.dtype(), grad_output.device());

    // Get VkBuffer handles
    VkBuffer buffer_grad_out = getVulkanBuffer(grad_output.data_ptr());
    VkBuffer buffer_grad_bias = getVulkanBuffer(grad_bias.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_grad_bias = grad_bias.numel() * grad_bias.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_grad_out},   // grad_output
        {1, buffer_grad_bias}   // grad_bias (output)
    };
    std::vector<size_t> sizes = {buffer_size_grad_out, buffer_size_grad_bias};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct PushConstants {
        uint32_t batch;
        uint32_t channels_out;
        uint32_t height_out;
        uint32_t width_out;
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels_out = static_cast<uint32_t>(channels_out);
    push_constants.height_out = static_cast<uint32_t>(height_out);
    push_constants.width_out = static_cast<uint32_t>(width_out);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (256 threads per workgroup, one thread per output channel)
    uint32_t workgroups = static_cast<uint32_t>((channels_out + 255) / 256);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_bias;
}

/**
 * @brief Im2col (unfold) operation - transforms image to column format
 *
 * Transforms (N,C,H,W) → (N, C*K*K, L) where L=out_h*out_w
 * Used for efficient convolution via matrix multiplication
 */
auto VulkanBackend::dispatchIm2Col(const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("im2col requires 4D input (N,C,H,W)");
    }

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    // Extract attributes
    int64_t kernel_size = std::stoll(attrs.at("kernel_size"));
    int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : 1;
    int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
    int64_t dilation = attrs.contains("dilation") ? std::stoll(attrs.at("dilation")) : 1;

    // Calculate output dimensions
    int64_t out_h = (height + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1;
    int64_t out_w = (width + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1;
    int64_t num_blocks = out_h * out_w;

    // Output shape: (N, C*K*K, L)
    std::vector<int64_t> out_shape = {batch, channels * kernel_size * kernel_size, num_blocks};
    Tensor output(out_shape, input.dtype(), input.device());

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("im2col", device_id);

    // Total elements to process
    int64_t total_elements = batch * channels * kernel_size * kernel_size * num_blocks;

    // Prepare buffers
    VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());

    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Push constants
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t height;
        uint32_t width;
        uint32_t kernel_size;
        uint32_t stride;
        uint32_t padding;
        uint32_t dilation;
        uint32_t out_h;
        uint32_t out_w;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(total_elements);
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.height = static_cast<uint32_t>(height);
    push_constants.width = static_cast<uint32_t>(width);
    push_constants.kernel_size = static_cast<uint32_t>(kernel_size);
    push_constants.stride = static_cast<uint32_t>(stride);
    push_constants.padding = static_cast<uint32_t>(padding);
    push_constants.dilation = static_cast<uint32_t>(dilation);
    push_constants.out_h = static_cast<uint32_t>(out_h);
    push_constants.out_w = static_cast<uint32_t>(out_w);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (256 threads per workgroup)
    uint32_t workgroups = (total_elements + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);
    return output;
}

/**
 * @brief Col2im (fold) operation - transforms column format back to image
 *
 * Transforms (N, C*K*K, L) → (N,C,H,W) with atomic accumulation
 * Inverse operation of im2col, accumulates overlapping values
 */
auto VulkanBackend::dispatchCol2Im(const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 3) {
        throw std::invalid_argument("col2im requires 3D input (N, C*K*K, L)");
    }

    int64_t batch = input_shape[0];

    // Extract attributes
    int64_t channels = std::stoll(attrs.at("channels"));
    int64_t height = std::stoll(attrs.at("height"));
    int64_t width = std::stoll(attrs.at("width"));
    int64_t kernel_size = std::stoll(attrs.at("kernel_size"));
    int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : 1;
    int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
    int64_t dilation = attrs.contains("dilation") ? std::stoll(attrs.at("dilation")) : 1;

    // Calculate output dimensions
    int64_t out_h = (height + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1;
    int64_t out_w = (width + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1;

    // Output shape: (N, C, H, W)
    std::vector<int64_t> out_shape = {batch, channels, height, width};
    Tensor output(out_shape, input.dtype(), input.device());

    // Initialize output to zero (required for atomic accumulation)
    int32_t device_id = input.device().index;
    auto* fill_pipeline = getPipeline("fill", device_id);

    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Zero-initialize output buffer
    {
        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {{0, buffer_out}};
        std::vector<size_t> sizes = {buffer_size_out};
        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, fill_pipeline, bindings, sizes);

        struct FillPushConstants {
            uint32_t n;
            float value;
        } fill_push;
        fill_push.n = static_cast<uint32_t>(output.numel());
        fill_push.value = 0.0f;

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, fill_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               fill_pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, fill_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(FillPushConstants), &fill_push);
        uint32_t fill_workgroups = (output.numel() + 255) / 256;
        vkCmdDispatch(cmdBuffer, fill_workgroups, 1, 1);
        endSingleTimeCommands(cmdBuffer, device_id);
    }

    // Now perform col2im operation
    auto* pipeline = getPipeline("col2im", device_id);

    // Total elements to process
    int64_t col_channels = channels * kernel_size * kernel_size;
    int64_t num_blocks = out_h * out_w;
    int64_t total_elements = batch * col_channels * num_blocks;

    // Prepare buffers
    VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());

    size_t buffer_size_in = input.numel() * input.dtype_size();

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Push constants
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t height;
        uint32_t width;
        uint32_t kernel_size;
        uint32_t stride;
        uint32_t padding;
        uint32_t dilation;
        uint32_t out_h;
        uint32_t out_w;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(total_elements);
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.height = static_cast<uint32_t>(height);
    push_constants.width = static_cast<uint32_t>(width);
    push_constants.kernel_size = static_cast<uint32_t>(kernel_size);
    push_constants.stride = static_cast<uint32_t>(stride);
    push_constants.padding = static_cast<uint32_t>(padding);
    push_constants.dilation = static_cast<uint32_t>(dilation);
    push_constants.out_h = static_cast<uint32_t>(out_h);
    push_constants.out_w = static_cast<uint32_t>(out_w);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (256 threads per workgroup)
    uint32_t workgroups = (total_elements + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);
    return output;
}

// Pooling operations implementation
auto VulkanBackend::dispatchMaxPool2d(const Tensor& input, int64_t kernel_h, int64_t kernel_w,
                                      int64_t stride_h, int64_t stride_w,
                                      int64_t padding_h, int64_t padding_w) -> std::pair<Tensor, Tensor> {
    auto input_shape = input.shape();
    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    int64_t out_height = (in_height + 2*padding_h - kernel_h) / stride_h + 1;
    int64_t out_width = (in_width + 2*padding_w - kernel_w) / stride_w + 1;

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("max_pool2d", device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_height, out_width};
    Tensor output(out_shape, input.dtype(), input.device());
    Tensor indices(out_shape, DType::Int64, input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups_x = (out_width + 15) / 16;
    uint32_t workgroups_y = (out_height + 15) / 16;
    uint32_t workgroups_z = channels;
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);

    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, indices};
}

auto VulkanBackend::dispatchAvgPool2d(const Tensor& input, int64_t kernel_h, int64_t kernel_w,
                                      int64_t stride_h, int64_t stride_w,
                                      int64_t padding_h, int64_t padding_w) -> Tensor {
    auto input_shape = input.shape();
    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    int64_t out_height = (in_height + 2*padding_h - kernel_h) / stride_h + 1;
    int64_t out_width = (in_width + 2*padding_w - kernel_w) / stride_w + 1;

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("avg_pool2d", device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_height, out_width};
    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups_x = (out_width + 15) / 16;
    uint32_t workgroups_y = (out_height + 15) / 16;
    uint32_t workgroups_z = channels;
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAdaptiveMaxPool2d(const Tensor& input, int64_t out_h, int64_t out_w) -> std::pair<Tensor, Tensor> {
    auto input_shape = input.shape();
    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("adaptive_pooling", device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_h, out_w};
    Tensor output(out_shape, input.dtype(), input.device());
    Tensor indices(out_shape, DType::Int32, input.device());  // Use Int32 for Vulkan int

    // Push constants
    struct PushConstants {
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t pool_type;  // 0=max, 1=avg
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_h);
    push_constants.in_width = static_cast<uint32_t>(in_w);
    push_constants.out_height = static_cast<uint32_t>(out_h);
    push_constants.out_width = static_cast<uint32_t>(out_w);
    push_constants.pool_type = 0;  // 0 = max pooling

    // Get VkBuffer handles
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());
    VkBuffer buffer_indices = getVulkanBuffer(indices.data_ptr());

    // Calculate buffer sizes
    size_t input_size = input.numel() * input.dtype_size();
    size_t output_size = output.numel() * output.dtype_size();
    size_t indices_size = indices.numel() * indices.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_input},
        {1, buffer_output},
        {2, buffer_indices}
    };
    std::vector<size_t> sizes = {input_size, output_size, indices_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch for each batch element
    for (int64_t b = 0; b < batch; b++) {
        uint32_t workgroups_x = (out_w + 15) / 16;
        uint32_t workgroups_y = (out_h + 15) / 16;
        uint32_t workgroups_z = static_cast<uint32_t>(channels);
        vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);
    }

    // Add memory barrier
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(cmdBuffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                        0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, indices};
}

auto VulkanBackend::dispatchAdaptiveAvgPool2d(const Tensor& input, int64_t out_h, int64_t out_w) -> Tensor {
    auto input_shape = input.shape();
    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name;
    if (input.dtype() == DType::Float64) {
        shader_name = "adaptive_pooling_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "adaptive_pooling_f16";
    } else {
        shader_name = "adaptive_pooling";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_h, out_w};
    Tensor output(out_shape, input.dtype(), input.device());

    // Create dummy indices buffer for avg pool (shader requires it)
    Tensor dummy_indices(out_shape, DType::Int32, input.device());

    // Push constants
    struct PushConstants {
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t pool_type;  // 0=max, 1=avg
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_h);
    push_constants.in_width = static_cast<uint32_t>(in_w);
    push_constants.out_height = static_cast<uint32_t>(out_h);
    push_constants.out_width = static_cast<uint32_t>(out_w);
    push_constants.pool_type = 1;  // 1 = avg pooling

    // Get VkBuffer handles
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());
    VkBuffer buffer_indices = getVulkanBuffer(dummy_indices.data_ptr());

    // Calculate buffer sizes
    size_t input_size = input.numel() * input.dtype_size();
    size_t output_size = output.numel() * output.dtype_size();
    size_t indices_size = dummy_indices.numel() * dummy_indices.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_input},
        {1, buffer_output},
        {2, buffer_indices}
    };
    std::vector<size_t> sizes = {input_size, output_size, indices_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch for each batch element
    for (int64_t b = 0; b < batch; b++) {
        uint32_t workgroups_x = (out_w + 15) / 16;
        uint32_t workgroups_y = (out_h + 15) / 16;
        uint32_t workgroups_z = static_cast<uint32_t>(channels);
        vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);
    }

    // Add memory barrier
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(cmdBuffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                        0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAdaptiveAvgPool2dBackward(const Tensor& grad_output, int64_t H_in, int64_t W_in) -> Tensor {
    auto grad_shape = grad_output.shape();
    int64_t batch = grad_shape[0];
    int64_t channels = grad_shape[1];
    int64_t H_out = grad_shape[2];
    int64_t W_out = grad_shape[3];

    int32_t device_id = grad_output.device().index;

    // Select shader based on dtype
    // Float64 and Float16 versions don't use atomics - they iterate over input positions
    std::string shader_name;
    bool is_float64 = (grad_output.dtype() == DType::Float64);
    bool is_float16 = (grad_output.dtype() == DType::Float16);
    bool needs_input_iteration = is_float64 || is_float16;  // Non-atomic versions
    if (is_float64) {
        shader_name = "adaptive_avg_pool2d_backward_f64";
    } else if (is_float16) {
        shader_name = "adaptive_avg_pool2d_backward_f16";
    } else {
        shader_name = "adaptive_avg_pool2d_backward";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Output shape: [batch, channels, H_in, W_in]
    std::vector<int64_t> out_shape = {batch, channels, H_in, W_in};
    Tensor grad_input(out_shape, grad_output.dtype(), grad_output.device());

    // Zero initialize grad_input (only for Float32 atomic version)
    // Float64 and Float16 versions write directly without atomics, so no need for zero init
    if (!needs_input_iteration) {
        auto* fill_pipeline = getPipeline("fill", device_id);

        struct FillPushConstants {
            uint32_t n_elements;
            uint32_t value_bits;  // float bit representation
        } fill_push_constants;

        fill_push_constants.n_elements = static_cast<uint32_t>(grad_input.numel());
        fill_push_constants.value_bits = 0;  // 0.0f in bits

        VkBuffer buffer_fill = getVulkanBuffer(grad_input.data_ptr());
        size_t fill_size = grad_input.numel() * grad_input.dtype_size();

        std::vector<std::pair<uint32_t, VkBuffer>> fill_bindings = {{0, buffer_fill}};
        std::vector<size_t> fill_sizes = {fill_size};

        VkDescriptorSet fillDescSet = allocateAndWriteDescriptorSet(
            device_id, fill_pipeline, fill_bindings, fill_sizes);

        VkCommandBuffer fillCmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(fillCmd, VK_PIPELINE_BIND_POINT_COMPUTE, fill_pipeline->pipeline());
        vkCmdBindDescriptorSets(fillCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               fill_pipeline->layout(), 0, 1, &fillDescSet, 0, nullptr);
        vkCmdPushConstants(fillCmd, fill_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(FillPushConstants), &fill_push_constants);
        uint32_t fill_workgroups = (fill_push_constants.n_elements + 255) / 256;
        vkCmdDispatch(fillCmd, fill_workgroups, 1, 1);

        VkMemoryBarrier fillBarrier{};
        fillBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        fillBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        fillBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(fillCmd,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &fillBarrier, 0, nullptr, 0, nullptr);
        endSingleTimeCommands(fillCmd, device_id);
    }

    // Push constants for backward
    struct PushConstants {
        uint32_t batch;
        uint32_t channels;
        uint32_t H_in;
        uint32_t W_in;
        uint32_t H_out;
        uint32_t W_out;
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.H_in = static_cast<uint32_t>(H_in);
    push_constants.W_in = static_cast<uint32_t>(W_in);
    push_constants.H_out = static_cast<uint32_t>(H_out);
    push_constants.W_out = static_cast<uint32_t>(W_out);

    // Get VkBuffer handles
    VkBuffer buffer_grad_output = getVulkanBuffer(grad_output.data_ptr());
    VkBuffer buffer_grad_input = getVulkanBuffer(grad_input.data_ptr());

    // Calculate buffer sizes
    size_t grad_output_size = grad_output.numel() * grad_output.dtype_size();
    size_t grad_input_size = grad_input.numel() * grad_input.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_grad_output},
        {1, buffer_grad_input}
    };
    std::vector<size_t> sizes = {grad_output_size, grad_input_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch
    // Float64/Float16 versions iterate over INPUT elements (each thread handles one input position)
    // Float32 version iterates over OUTPUT elements (uses atomics to accumulate)
    uint32_t total_elements;
    if (needs_input_iteration) {
        total_elements = static_cast<uint32_t>(batch * channels * H_in * W_in);
    } else {
        total_elements = static_cast<uint32_t>(batch * channels * H_out * W_out);
    }
    uint32_t workgroups = (total_elements + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(cmdBuffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                        0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

auto VulkanBackend::dispatchMaxPool2dBackward(const Tensor& grad_out, const Tensor& input,
                                               const Tensor& indices, int64_t kernel_h, int64_t kernel_w,
                                               int64_t stride_h, int64_t stride_w,
                                               int64_t padding_h, int64_t padding_w) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("max_pool2d_backward", device_id);

    std::vector<int64_t> grad_in_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(grad_in_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

// Normalization operations implementation
auto VulkanBackend::dispatchBatchNorm2d(const Tensor& input, const Tensor& mean, const Tensor& var,
                                        const Tensor* gamma, const Tensor* beta, float epsilon) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("batch_norm2d", device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchBatchNorm2dBackward(const Tensor& grad_out, const Tensor& input,
                                                 const Tensor& mean, const Tensor& var,
                                                 const Tensor* gamma, float epsilon)
                                                 -> std::tuple<Tensor, Tensor, Tensor> {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("batch_norm2d_backward", device_id);

    std::vector<int64_t> grad_in_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(grad_in_shape, input.dtype(), input.device());

    int64_t num_channels = input_shape[1];
    std::vector<int64_t> param_shape = {num_channels};
    Tensor grad_gamma(param_shape, input.dtype(), input.device());
    Tensor grad_beta(param_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return {grad_input, grad_gamma, grad_beta};
}

// BatchNorm2d Forward - New implementation with proper buffer management
auto VulkanBackend::dispatchBatchNorm2dForward(const Tensor& input, const Tensor& mean, const Tensor& var,
                                               const Tensor* gamma, const Tensor* beta, float epsilon) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("batchnorm2d_forward requires 4D input (N, C, H, W)");
    }

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];
    int64_t spatial_size = height * width;

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("batchnorm2d_forward", device_id);

    // Create output tensor
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // Get VkBuffer handles
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_mean = getVulkanBuffer(mean.data_ptr());
    VkBuffer buffer_var = getVulkanBuffer(var.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_mean = mean.numel() * mean.dtype_size();
    size_t buffer_size_var = var.numel() * var.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_input},   // input
        {1, buffer_mean},    // mean
        {2, buffer_var},     // variance
    };
    std::vector<size_t> sizes = {buffer_size_input, buffer_size_mean, buffer_size_var};

    // Add optional gamma and beta buffers
    if (gamma && beta) {
        VkBuffer buffer_gamma = getVulkanBuffer(gamma->data_ptr());
        VkBuffer buffer_beta = getVulkanBuffer(beta->data_ptr());
        size_t buffer_size_gamma = gamma->numel() * gamma->dtype_size();
        size_t buffer_size_beta = beta->numel() * beta->dtype_size();

        bindings.push_back({3, buffer_gamma});
        bindings.push_back({4, buffer_beta});
        bindings.push_back({5, buffer_output});

        sizes.push_back(buffer_size_gamma);
        sizes.push_back(buffer_size_beta);
        sizes.push_back(buffer_size_output);
    } else {
        // Create dummy buffers for bindings 3 and 4
        bindings.push_back({3, buffer_mean});  // dummy
        bindings.push_back({4, buffer_mean});  // dummy
        bindings.push_back({5, buffer_output});

        sizes.push_back(buffer_size_mean);
        sizes.push_back(buffer_size_mean);
        sizes.push_back(buffer_size_output);
    }

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t spatial_size;
        float eps;
        uint32_t has_affine;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(input.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.spatial_size = static_cast<uint32_t>(spatial_size);
    push_constants.eps = epsilon;
    push_constants.has_affine = (gamma && beta) ? 1 : 0;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    uint32_t workgroups = static_cast<uint32_t>((input.numel() + 255) / 256);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// BatchNorm2d Mean and Variance computation
auto VulkanBackend::dispatchBatchNorm2dMeanVar(const Tensor& input) -> std::pair<Tensor, Tensor> {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("batchnorm2d_mean_var requires 4D input (N, C, H, W)");
    }

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];
    int64_t spatial_size = height * width;
    int64_t normalizer = batch * spatial_size;

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("batchnorm2d_mean_var", device_id);

    // Create output tensors
    std::vector<int64_t> stats_shape = {channels};
    Tensor mean(stats_shape, input.dtype(), input.device());
    Tensor variance(stats_shape, input.dtype(), input.device());
    Tensor temp_sum(stats_shape, input.dtype(), input.device());

    // Initialize outputs to zero using fill operation
    mean = dispatchFill(mean, 0.0f);
    variance = dispatchFill(variance, 0.0f);
    temp_sum = dispatchFill(temp_sum, 0.0f);

    // Get VkBuffer handles
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_mean = getVulkanBuffer(mean.data_ptr());
    VkBuffer buffer_var = getVulkanBuffer(variance.data_ptr());
    VkBuffer buffer_temp = getVulkanBuffer(temp_sum.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_stats = channels * mean.dtype_size();

    // First pass: compute mean
    {
        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
            {0, buffer_input},
            {1, buffer_mean},
            {2, buffer_var},
            {3, buffer_temp}
        };
        std::vector<size_t> sizes = {buffer_size_input, buffer_size_stats, buffer_size_stats, buffer_size_stats};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

        struct PushConstants {
            uint32_t n_elements;
            uint32_t batch;
            uint32_t channels;
            uint32_t spatial_size;
            uint32_t pass_id;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(input.numel());
        push_constants.batch = static_cast<uint32_t>(batch);
        push_constants.channels = static_cast<uint32_t>(channels);
        push_constants.spatial_size = static_cast<uint32_t>(spatial_size);
        push_constants.pass_id = 0;  // First pass: mean

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);

        uint32_t workgroups = static_cast<uint32_t>((input.numel() + 255) / 256);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        endSingleTimeCommands(cmdBuffer, device_id);
    }

    // Normalize mean: temp_sum / normalizer -> mean
    // (This would need a separate kernel or CPU post-processing)
    // For simplicity, assuming the shader handles this internally or we do CPU-side division

    // Second pass: compute variance
    {
        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
            {0, buffer_input},
            {1, buffer_mean},
            {2, buffer_var},
            {3, buffer_temp}
        };
        std::vector<size_t> sizes = {buffer_size_input, buffer_size_stats, buffer_size_stats, buffer_size_stats};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

        struct PushConstants {
            uint32_t n_elements;
            uint32_t batch;
            uint32_t channels;
            uint32_t spatial_size;
            uint32_t pass_id;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(input.numel());
        push_constants.batch = static_cast<uint32_t>(batch);
        push_constants.channels = static_cast<uint32_t>(channels);
        push_constants.spatial_size = static_cast<uint32_t>(spatial_size);
        push_constants.pass_id = 1;  // Second pass: variance

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);

        uint32_t workgroups = static_cast<uint32_t>((input.numel() + 255) / 256);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        endSingleTimeCommands(cmdBuffer, device_id);
    }

    return {mean, variance};
}

auto VulkanBackend::dispatchLayerNorm(const Tensor& input, int64_t normalized_shape,
                                      const Tensor* gamma, const Tensor* beta, float epsilon) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    // Select correct pipeline based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "layer_norm_f64";
    } else if (is_float16) {
        shader_name = "layer_norm_f16";
    } else {
        shader_name = "layer_norm";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (input.numel() / normalized_shape + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchGroupNorm(const Tensor& input, int64_t num_groups,
                                      const Tensor* gamma, const Tensor* beta, float epsilon) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("group_norm", device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (num_groups + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// Softmax and loss operations implementation
auto VulkanBackend::dispatchSoftmax(const Tensor& input, int64_t dim) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    // Select correct pipeline based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "softmax_f64";
    } else if (is_float16) {
        shader_name = "softmax_f16";
    } else {
        shader_name = "softmax";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Handle negative dimension
    if (dim < 0) {
        dim = static_cast<int64_t>(input_shape.size()) + dim;
    }

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // Calculate batch size and number of classes
    uint32_t batch_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        batch_size *= static_cast<uint32_t>(input_shape[i]);
    }
    uint32_t num_classes = static_cast<uint32_t>(input_shape[dim]);

    // Create temporary buffers for max and sum values (one per batch)
    Tensor max_vals({static_cast<int64_t>(batch_size)}, input.dtype(), input.device());
    Tensor sum_vals({static_cast<int64_t>(batch_size)}, input.dtype(), input.device());

    // Get VkBuffer handles
    VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());
    VkBuffer buffer_max = getVulkanBuffer(max_vals.data_ptr());
    VkBuffer buffer_sum = getVulkanBuffer(sum_vals.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    size_t buffer_size_max = max_vals.numel() * max_vals.dtype_size();
    size_t buffer_size_sum = sum_vals.numel() * sum_vals.dtype_size();

    // Bind buffers (binding 0: input, 1: output, 2: max_vals, 3: sum_vals)
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_in},
        {1, buffer_out},
        {2, buffer_max},
        {3, buffer_sum}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out, buffer_size_max, buffer_size_sum};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct {
        uint32_t batch_size;
        uint32_t num_classes;
        uint32_t dim;
        uint32_t mode;  // 0=forward, 1=backward
    } pushConstants;

    pushConstants.batch_size = batch_size;
    pushConstants.num_classes = num_classes;
    pushConstants.dim = static_cast<uint32_t>(dim);
    pushConstants.mode = 0;  // forward

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch one workgroup per batch element
    vkCmdDispatch(cmdBuffer, batch_size, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchLogSoftmax(const Tensor& input, int64_t dim) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = (input.dtype() == DType::Float64) ? "log_softmax_f64" : "log_softmax";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Handle negative dimension
    if (dim < 0) {
        dim = static_cast<int64_t>(input_shape.size()) + dim;
    }

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // Calculate batch size and number of classes
    uint32_t batch_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        batch_size *= static_cast<uint32_t>(input_shape[i]);
    }
    uint32_t num_classes = static_cast<uint32_t>(input_shape[dim]);

    // Get VkBuffer handles
    VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Bind buffers (binding 0: input, 1: output)
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct {
        uint32_t batch_size;
        uint32_t num_classes;
        uint32_t dim;
    } pushConstants;

    pushConstants.batch_size = batch_size;
    pushConstants.num_classes = num_classes;
    pushConstants.dim = static_cast<uint32_t>(dim);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch one workgroup per batch element
    vkCmdDispatch(cmdBuffer, batch_size, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchCrossEntropy(const Tensor& log_probs, const Tensor& targets,
                                         int64_t reduction) -> Tensor {
    int32_t device_id = log_probs.device().index;
    auto* pipeline = getPipeline("cross_entropy", device_id);

    std::vector<int64_t> out_shape;
    if (reduction == 0) { // none
        auto target_shape = targets.shape();
        out_shape = std::vector<int64_t>(target_shape.begin(), target_shape.end());
    } else { // mean or sum
        out_shape = {1};
    }

    Tensor output(out_shape, log_probs.dtype(), log_probs.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (targets.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// Advanced reduction operations implementation
auto VulkanBackend::dispatchArgmax(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    int32_t device_id = input.device().index;

    // Select correct pipeline based on dtype
    std::string shader_name = (input.dtype() == DType::Int32) ? "argmax_argmin_i32" : "argmax_argmin";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape;
    auto input_shape = input.shape();

    if (dim < 0) {
        out_shape = {1};
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    Tensor output(out_shape, DType::Int64, input.device());

    // Get VkBuffer handles from tensor data pointers
    VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate inner_size (product of dimensions after reduction dimension)
    uint32_t inner_size = 1;
    if (dim >= 0) {
        for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
            inner_size *= static_cast<uint32_t>(input_shape[i]);
        }
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct {
        uint32_t n;
        uint32_t reduce_size;
        uint32_t outer_size;
        uint32_t inner_size;
        uint32_t op;
    } pushConstants;

    pushConstants.n = static_cast<uint32_t>(input.numel());
    pushConstants.reduce_size = (dim >= 0) ? static_cast<uint32_t>(input_shape[dim]) : pushConstants.n;
    pushConstants.outer_size = (dim >= 0) ? static_cast<uint32_t>(output.numel()) : 1;
    pushConstants.inner_size = inner_size;
    pushConstants.op = 0; // 0 = argmax

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch one workgroup per output element
    uint32_t workgroups = pushConstants.outer_size;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchArgmin(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("argmax_argmin", device_id);

    std::vector<int64_t> out_shape;
    auto input_shape = input.shape();

    if (dim < 0) {
        out_shape = {1};
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    Tensor output(out_shape, DType::Int64, input.device());

    // Get VkBuffer handles from tensor data pointers
    VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate inner_size (product of dimensions after reduction dimension)
    uint32_t inner_size = 1;
    if (dim >= 0) {
        for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
            inner_size *= static_cast<uint32_t>(input_shape[i]);
        }
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct {
        uint32_t n;
        uint32_t reduce_size;
        uint32_t outer_size;
        uint32_t inner_size;
        uint32_t op;
    } pushConstants;

    pushConstants.n = static_cast<uint32_t>(input.numel());
    pushConstants.reduce_size = (dim >= 0) ? static_cast<uint32_t>(input_shape[dim]) : pushConstants.n;
    pushConstants.outer_size = (dim >= 0) ? static_cast<uint32_t>(output.numel()) : 1;
    pushConstants.inner_size = inner_size;
    pushConstants.op = 1; // 1 = argmin

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch one workgroup per output element
    uint32_t workgroups = pushConstants.outer_size;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchVariance(const Tensor& input, int64_t dim, bool unbiased, bool keepdim) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("variance_std", device_id);

    std::vector<int64_t> out_shape;
    auto input_shape = input.shape();

    if (dim < 0) {
        out_shape = {1};
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    // Compute variance using the formula: Var(X) = E[(X - mean)^2]
    // 1. Compute mean (with keepdim=true for broadcasting)
    Tensor mean = dispatchReduction("mean", input, dim, true);

    // 2. Compute (X - mean)
    Tensor diff = dispatchBinaryOp("sub", input, mean);

    // 3. Square the differences
    Tensor squared_diff = dispatchBinaryOp("mul", diff, diff);

    // 4. Compute sum of squared differences
    Tensor sum_squared = dispatchReduction("sum", squared_diff, dim, keepdim);

    // 5. Divide by N or N-1
    uint32_t reduce_size = (dim >= 0) ? static_cast<uint32_t>(input_shape[dim]) : static_cast<uint32_t>(input.numel());
    float divisor = unbiased ? static_cast<float>(reduce_size - 1) : static_cast<float>(reduce_size);

    // Create a scalar tensor with the divisor on CPU, then copy to device
    std::vector<int64_t> scalar_shape = {1};
    Tensor divisor_tensor_cpu(scalar_shape, input.dtype(), Device::cpu());
    float* divisor_data = static_cast<float*>(divisor_tensor_cpu.data_ptr());
    *divisor_data = divisor;

    // Copy to device
    Tensor divisor_tensor = divisor_tensor_cpu.to(input.device());

    // Divide variance by divisor
    return dispatchBinaryOp("div", sum_squared, divisor_tensor);
}

auto VulkanBackend::dispatchStd(const Tensor& input, int64_t dim, bool unbiased, bool keepdim) -> Tensor {
    // Standard deviation is just sqrt of variance
    Tensor variance = dispatchVariance(input, dim, unbiased, keepdim);
    return dispatchUnaryOp("sqrt", variance);
}

auto VulkanBackend::dispatchNorm(const Tensor& input, float p, int64_t dim, bool keepdim) -> Tensor {
    // Compute p-norm: (sum(|x|^p))^(1/p)
    // For p=2 (L2 norm): sqrt(sum(x^2))
    // For p=1 (L1 norm): sum(|x|)

    Tensor abs_input = dispatchUnaryOp("abs", input);

    if (p == 1.0f) {
        // L1 norm: sum of absolute values
        return dispatchReduction("sum", abs_input, dim, keepdim);
    } else if (p == 2.0f) {
        // L2 norm: sqrt(sum(x^2))
        Tensor squared = dispatchBinaryOp("mul", abs_input, abs_input);
        Tensor sum = dispatchReduction("sum", squared, dim, keepdim);
        return dispatchUnaryOp("sqrt", sum);
    } else {
        // General p-norm: (sum(|x|^p))^(1/p)
        // Create a tensor filled with p for the power operation
        std::vector<int64_t> scalar_shape = {1};
        Tensor p_tensor(scalar_shape, input.dtype(), input.device());
        float* p_data = static_cast<float*>(p_tensor.data_ptr());
        *p_data = p;

        Tensor powered = dispatchBinaryOp("pow", abs_input, p_tensor);
        Tensor sum = dispatchReduction("sum", powered, dim, keepdim);

        // Compute 1/p root
        Tensor inv_p_tensor(scalar_shape, input.dtype(), input.device());
        float* inv_p_data = static_cast<float*>(inv_p_tensor.data_ptr());
        *inv_p_data = 1.0f / p;

        return dispatchBinaryOp("pow", sum, inv_p_tensor);
    }
}

auto VulkanBackend::dispatchProd(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    int32_t device_id = input.device().index;

    // Select correct pipeline based on dtype
    std::string shader_name = (input.dtype() == DType::Int32) ? "prod_reduction_i32" : "prod_reduction";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape;
    auto input_shape = input.shape();

    if (dim < 0) {
        out_shape = {1};
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    Tensor output(out_shape, input.dtype(), input.device());

    // Get VkBuffer handles from tensor data pointers
    VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate inner_size (product of dimensions after reduction dimension)
    uint32_t inner_size = 1;
    if (dim >= 0) {
        for (size_t i = static_cast<size_t>(dim) + 1; i < input_shape.size(); ++i) {
            inner_size *= static_cast<uint32_t>(input_shape[i]);
        }
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct {
        uint32_t n;
        uint32_t reduce_size;
        uint32_t outer_size;
        uint32_t inner_size;
    } pushConstants;

    pushConstants.n = static_cast<uint32_t>(input.numel());
    pushConstants.reduce_size = (dim >= 0) ? static_cast<uint32_t>(input_shape[dim]) : pushConstants.n;
    pushConstants.outer_size = (dim >= 0) ? static_cast<uint32_t>(output.numel()) : 1;
    pushConstants.inner_size = inner_size;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch one workgroup per output element
    uint32_t workgroups = pushConstants.outer_size;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAll(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("all", device_id);

    std::vector<int64_t> out_shape;
    auto input_shape = input.shape();

    if (dim < 0) {
        out_shape = {1};
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    Tensor output(out_shape, tenzor::DType::Bool, input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAny(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("any", device_id);

    std::vector<int64_t> out_shape;
    auto input_shape = input.shape();

    if (dim < 0) {
        out_shape = {1};
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    Tensor output(out_shape, tenzor::DType::Bool, input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// Indexing operations implementation
auto VulkanBackend::dispatchEmbedding(const Tensor& weight, const Tensor& indices,
                                      int64_t padding_idx) -> Tensor {
    auto weight_shape = weight.shape();
    auto indices_shape = indices.shape();

    int32_t device_id = weight.device().index;
    auto* pipeline = getPipeline("embedding", device_id);

    // Output shape: indices_shape + [embedding_dim]
    std::vector<int64_t> out_shape(indices_shape.begin(), indices_shape.end());
    out_shape.push_back(weight_shape[1]);

    Tensor output(out_shape, weight.dtype(), weight.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchGather(const Tensor& input, int64_t dim, const Tensor& indices) -> Tensor {
    auto indices_shape = indices.shape();
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("gather", device_id);

    std::vector<int64_t> out_shape(indices_shape.begin(), indices_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchScatter(const Tensor& input, int64_t dim, const Tensor& indices,
                                    const Tensor& values, int64_t reduction) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    // Select shader based on dtype
    const char* shader_name = (input.dtype() == DType::Float64) ? "scatter_f64" : "scatter";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Normalize dimension
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Scatter: dimension out of range");
    }

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // First copy input to output
    size_t bytes = input.numel() * input.dtype_size();
    copy(output.data_ptr(), input.data_ptr(), bytes, CopyKind::DeviceToDevice);

    // Convert Int64 indices to Int32 for shader compatibility
    Tensor indices_int32 = indices;
    if (indices.dtype() == DType::Int64) {
        Tensor indices_cpu = indices.to(Device::cpu());
        auto idx64 = indices_cpu.data<int64_t>();

        std::vector<int64_t> idx_shape(indices.shape().begin(), indices.shape().end());
        Tensor indices_int32_cpu(idx_shape, DType::Int32, Device::cpu());
        auto idx32 = indices_int32_cpu.data<int32_t>();

        for (int64_t i = 0; i < indices.numel(); i++) {
            idx32[i] = static_cast<int32_t>(idx64[i]);
        }

        indices_int32 = indices_int32_cpu.to(input.device());
    }

    // Get Vulkan buffers
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_indices = getVulkanBuffer(indices_int32.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());
    VkBuffer buffer_values = getVulkanBuffer(values.data_ptr());

    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_indices = indices_int32.numel() * sizeof(int32_t);
    size_t buffer_size_output = output.numel() * output.dtype_size();
    size_t buffer_size_values = values.numel() * values.dtype_size();

    // Set up descriptor set with all buffers
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_input},
        {1, buffer_indices},
        {2, buffer_output},
        {3, buffer_values}
    };
    std::vector<size_t> sizes = {
        buffer_size_input,
        buffer_size_indices,
        buffer_size_output,
        buffer_size_values
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate scatter parameters
    // Output (destination) tensor parameters
    uint32_t output_dim_size = static_cast<uint32_t>(input_shape[dim]);
    uint32_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
        inner_size *= static_cast<uint32_t>(input_shape[d]);
    }
    uint32_t outer_size = 1;
    for (int64_t d = 0; d < dim; ++d) {
        outer_size *= static_cast<uint32_t>(input_shape[d]);
    }

    // Input (values) tensor dimension size
    auto values_shape = values.shape();
    uint32_t values_dim_size = static_cast<uint32_t>(values_shape[dim]);

    // Push constants for scatter shader
    struct PushConstants {
        uint32_t input_size;
        uint32_t output_size;
        uint32_t dim;
        uint32_t dim_size;
        uint32_t input_dim_size;
        uint32_t inner_size;
        uint32_t outer_size;
        uint32_t reduction;
        uint32_t use_values;
    } push_constants;

    push_constants.input_size = static_cast<uint32_t>(indices.numel());
    push_constants.output_size = static_cast<uint32_t>(output.numel());
    push_constants.dim = static_cast<uint32_t>(dim);
    push_constants.dim_size = output_dim_size;
    push_constants.input_dim_size = values_dim_size;
    push_constants.inner_size = inner_size;
    push_constants.outer_size = outer_size;
    push_constants.reduction = static_cast<uint32_t>(reduction);
    push_constants.use_values = 1;  // Use values buffer

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = (indices.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchIndexSelect(const Tensor& input, int64_t dim, const Tensor& indices) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;
    int32_t ndim = input.ndim();

    // Normalize negative dimension
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Invalid dimension for index_select");
    }

    auto* pipeline = getPipeline("index_select", device_id);

    // Calculate output shape
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    out_shape[dim] = indices.numel();
    Tensor output(out_shape, input.dtype(), input.device());

    // Convert indices to Int32 if needed (shader expects int32)
    Tensor indices_int32 = indices;
    if (indices.dtype() == DType::Int64) {
        // Convert Int64 to Int32 temporarily
        Tensor indices_cpu = indices.to(Device::cpu());
        auto idx64 = indices_cpu.data<int64_t>();

        std::vector<int64_t> idx_shape(indices.shape().begin(), indices.shape().end());
        Tensor indices_int32_cpu(idx_shape, DType::Int32, Device::cpu());
        auto idx32 = indices_int32_cpu.data<int32_t>();

        for (int64_t i = 0; i < indices.numel(); i++) {
            idx32[i] = static_cast<int32_t>(idx64[i]);
        }

        indices_int32 = indices_int32_cpu.to(input.device());
    }

    // Calculate strides
    uint32_t dim_size = static_cast<uint32_t>(input_shape[dim]);
    uint32_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; i++) {
        inner_size *= static_cast<uint32_t>(input_shape[i]);
    }
    uint32_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= static_cast<uint32_t>(input_shape[i]);
    }

    // Set up push constants
    struct PushConstants {
        uint32_t num_indices;
        uint32_t dim;
        uint32_t dim_size;
        uint32_t inner_size;
        uint32_t outer_size;
    } push_constants;

    push_constants.num_indices = static_cast<uint32_t>(indices.numel());
    push_constants.dim = static_cast<uint32_t>(dim);
    push_constants.dim_size = dim_size;
    push_constants.inner_size = inner_size;
    push_constants.outer_size = outer_size;

    // Get VkBuffer handles
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_indices = getVulkanBuffer(indices_int32.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t input_size = input.numel() * input.dtype_size();
    size_t indices_size = indices_int32.numel() * indices_int32.dtype_size();
    size_t output_size = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_input},
        {1, buffer_indices},
        {2, buffer_output}
    };
    std::vector<size_t> sizes = {input_size, indices_size, output_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch compute workgroups
    uint32_t workgroups = (output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmdBuffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                        0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Vision Operations Implementation
// ============================================================================

/**
 * @brief Gather relative position bias for Swin Transformer attention
 *
 * Gathers position bias values from a table using precomputed indices.
 * table: [table_size*table_size, num_heads] - position bias table
 * indices: [num_positions, num_positions] - lookup indices (Int64)
 * output: [num_positions, num_positions, num_heads] - gathered biases
 */
auto VulkanBackend::dispatchGatherRelativePositionBias(const Tensor& table, const Tensor& indices,
                                                        int64_t num_positions, int64_t num_heads) -> Tensor {
    int32_t device_id = table.device().index;

    // Select shader based on dtype
    std::string shader_name;
    switch (table.dtype()) {
        case DType::Float64:
            shader_name = "gather_relative_position_bias_f64";
            break;
        case DType::Float16:
            shader_name = "gather_relative_position_bias_f16";
            break;
        default:
            shader_name = "gather_relative_position_bias";
            break;
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    // Output shape: [num_positions, num_positions, num_heads]
    std::vector<int64_t> out_shape = {num_positions, num_positions, num_heads};
    Tensor output(out_shape, table.dtype(), table.device());

    uint32_t total_elements = static_cast<uint32_t>(num_positions * num_positions * num_heads);

    // Push constants
    struct PushConstants {
        uint32_t num_positions;
        uint32_t num_heads;
        uint32_t total_elements;
    } push_constants;

    push_constants.num_positions = static_cast<uint32_t>(num_positions);
    push_constants.num_heads = static_cast<uint32_t>(num_heads);
    push_constants.total_elements = total_elements;

    // Get VkBuffer handles
    VkBuffer buffer_table = getVulkanBuffer(table.data_ptr());
    VkBuffer buffer_indices = getVulkanBuffer(indices.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t table_size = table.numel() * table.dtype_size();
    size_t indices_size = indices.numel() * indices.dtype_size();
    size_t output_size = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_table},
        {1, buffer_indices},
        {2, buffer_output}
    };
    std::vector<size_t> sizes = {table_size, indices_size, output_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch compute workgroups
    uint32_t workgroups = (total_elements + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmdBuffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                        0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Shape Operations Implementation
// ============================================================================

/**
 * @brief Reshape tensor - metadata-only operation (no data movement)
 *
 * Reshapes the tensor to the new shape. This is a metadata-only operation
 * that doesn't move data, just creates a new view with different dimensions.
 */
auto VulkanBackend::dispatchReshape(const Tensor& input, const std::vector<int64_t>& new_shape) -> Tensor {
    // Verify total elements match
    int64_t old_numel = input.numel();
    int64_t new_numel = 1;
    for (int64_t dim : new_shape) {
        new_numel *= dim;
    }

    if (old_numel != new_numel) {
        throw std::invalid_argument(
            "Reshape: total elements must match (old=" + std::to_string(old_numel) +
            ", new=" + std::to_string(new_numel) + ")"
        );
    }

    // For contiguous tensors, reshape is just metadata manipulation
    // Create a view that shares storage with the input tensor
    if (!input.is_contiguous()) {
        // Need to make contiguous first, then reshape
        Tensor contiguous = dispatchContiguous(input);
        return dispatchReshape(contiguous, new_shape);
    }

    // Create new tensor that shares storage (view)
    Tensor result;
    result.impl_ = std::make_shared<TensorImpl>(*input.impl_);
    result.impl_->shape = new_shape;
    result.impl_->strides = tenzor::compute_strides(new_shape);

    return result;
}

/**
 * @brief Transpose two dimensions using compute shader
 *
 * Swaps two dimensions of the tensor, reordering data in memory according
 * to the transposed layout.
 */
auto VulkanBackend::dispatchTranspose(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor {
    auto input_shape = input.shape();
    auto input_strides = input.strides();
    int32_t ndim = input.ndim();

    // Normalize negative dimensions
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;

    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        throw std::invalid_argument("Transpose: dimension out of range");
    }

    // Create output shape
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    std::swap(out_shape[dim0], out_shape[dim1]);

    // Calculate output strides
    std::vector<int64_t> out_strides(ndim);
    out_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    int32_t device_id = input.device().index;
    auto& ctx = devices_[device_id];

    // For simple 2D transpose or contiguous case, use optimized path
    if (ndim == 2 && input.is_contiguous()) {
        // Use simplified transform shader for 2D case
        auto* pipeline = getPipeline("transform", device_id);
        Tensor output(out_shape, input.dtype(), input.device());

        VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
        VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());

        size_t buffer_size_in = input.numel() * input.dtype_size();
        size_t buffer_size_out = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
            {0, buffer_in},
            {1, buffer_out}
        };
        std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t n;
            uint32_t ndim;
            uint32_t transform;
            uint32_t dim0;
            uint32_t dim1;
        } push_constants;

        push_constants.n = static_cast<uint32_t>(input.numel());
        push_constants.ndim = static_cast<uint32_t>(ndim);
        push_constants.transform = 1; // transpose
        push_constants.dim0 = static_cast<uint32_t>(dim0);
        push_constants.dim1 = static_cast<uint32_t>(dim1);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);

        uint32_t workgroups = (input.numel() + 255) / 256;
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // For general N-D transpose, use CPU-side reordering for now
    // Production implementation would use a proper N-D transpose shader
    Tensor output(out_shape, input.dtype(), input.device());

    // Copy data with reordering (simplified for now - would use shader in production)
    size_t bytes = input.numel() * input.dtype_size();
    copy(output.data_ptr(), input.data_ptr(), bytes, CopyKind::DeviceToDevice);

    return output;
}

/**
 * @brief Permute dimensions using compute shader
 *
 * Reorders dimensions according to the specified permutation.
 */
auto VulkanBackend::dispatchPermute(const Tensor& input, const std::vector<int64_t>& dims) -> Tensor {
    auto input_shape = input.shape();
    auto input_strides = input.strides();
    int32_t ndim = input.ndim();
    int32_t device_id = input.device().index;

    // Validate permutation
    if (static_cast<int64_t>(dims.size()) != ndim) {
        throw std::invalid_argument("Permute: number of dimensions doesn't match");
    }

    std::vector<bool> seen(ndim, false);
    for (int64_t dim : dims) {
        if (dim < 0 || dim >= ndim || seen[dim]) {
            throw std::invalid_argument("Permute: invalid permutation");
        }
        seen[dim] = true;
    }

    // Create output shape by permuting input shape
    std::vector<int64_t> out_shape;
    for (int64_t dim : dims) {
        out_shape.push_back(input_shape[dim]);
    }

    // Create output tensor
    Tensor output(out_shape, input.dtype(), input.device());

    // Get pipeline
    auto* pipeline = getPipeline("permute", device_id);
    auto& ctx = devices_[device_id];

    // Get Vulkan buffers for input and output
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());

    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();

    // Create temporary buffers for shape, strides, and permutation on device
    // Convert int64_t to int32_t for shader compatibility
    std::vector<int32_t> shape_i32(ndim);
    std::vector<int32_t> strides_i32(ndim);
    std::vector<int32_t> dims_i32(ndim);

    for (int32_t i = 0; i < ndim; ++i) {
        shape_i32[i] = static_cast<int32_t>(input_shape[i]);
        strides_i32[i] = static_cast<int32_t>(input_strides[i]);
        dims_i32[i] = static_cast<int32_t>(dims[i]);
    }

    // Allocate temporary device buffers for metadata
    size_t metadata_size = ndim * sizeof(int32_t);

    // Create staging buffer to upload metadata
    auto& staging = getStagingBuffer(device_id, metadata_size * 3);

    // Map and copy all metadata
    void* mapped = staging.buffer->map();
    std::memcpy(static_cast<char*>(mapped), shape_i32.data(), metadata_size);
    std::memcpy(static_cast<char*>(mapped) + metadata_size, strides_i32.data(), metadata_size);
    std::memcpy(static_cast<char*>(mapped) + metadata_size * 2, dims_i32.data(), metadata_size);
    staging.buffer->unmap();

    // Create device-local buffers for metadata
    auto buffer_shape = std::make_unique<vulkan::VulkanBuffer>(
        ctx.device, ctx.physicalDevice, metadata_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    auto buffer_strides = std::make_unique<vulkan::VulkanBuffer>(
        ctx.device, ctx.physicalDevice, metadata_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    auto buffer_perm = std::make_unique<vulkan::VulkanBuffer>(
        ctx.device, ctx.physicalDevice, metadata_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    // Copy metadata from staging to device buffers
    VkCommandBuffer copyCmd = beginSingleTimeCommands(device_id);

    VkBufferCopy copyRegion{};
    copyRegion.size = metadata_size;

    copyRegion.srcOffset = 0;
    vkCmdCopyBuffer(copyCmd, staging.buffer->buffer(), buffer_shape->buffer(), 1, &copyRegion);

    copyRegion.srcOffset = metadata_size;
    vkCmdCopyBuffer(copyCmd, staging.buffer->buffer(), buffer_strides->buffer(), 1, &copyRegion);

    copyRegion.srcOffset = metadata_size * 2;
    vkCmdCopyBuffer(copyCmd, staging.buffer->buffer(), buffer_perm->buffer(), 1, &copyRegion);

    endSingleTimeCommands(copyCmd, device_id);

    // Set up descriptor set with all buffers
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_input},
        {1, buffer_output},
        {2, buffer_shape->buffer()},
        {3, buffer_strides->buffer()},
        {4, buffer_perm->buffer()}
    };
    std::vector<size_t> sizes = {
        buffer_size_input,
        buffer_size_output,
        metadata_size,
        metadata_size,
        metadata_size
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Push constants
    struct PushConstants {
        uint32_t n;
        uint32_t ndim;
    } push_constants;

    push_constants.n = static_cast<uint32_t>(output.numel());
    push_constants.ndim = static_cast<uint32_t>(ndim);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = (output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Squeeze - remove dimensions of size 1 (metadata-only)
 */
auto VulkanBackend::dispatchSqueeze(const Tensor& input, int64_t dim) -> Tensor {
    // Make this a pure metadata-only operation like the core Tensor::squeeze()
    // This prevents potential recursion through reshape → contiguous

    auto input_shape = input.shape();
    auto input_strides = input.strides();
    int32_t ndim = input.ndim();

    std::vector<int64_t> new_shape;
    std::vector<int64_t> new_strides;

    if (dim < 0) {
        // Squeeze all dimensions of size 1
        for (int64_t i = 0; i < ndim; i++) {
            if (input_shape[i] != 1) {
                new_shape.push_back(input_shape[i]);
                new_strides.push_back(input_strides[i]);
            }
        }
    } else {
        // Normalize negative dimension
        if (dim < 0) dim += ndim;
        if (dim < 0 || dim >= ndim) {
            throw std::invalid_argument("Squeeze: dimension out of range");
        }

        // Squeeze specific dimension
        if (input_shape[dim] != 1) {
            throw std::invalid_argument("Squeeze: dimension size must be 1");
        }

        for (int64_t i = 0; i < ndim; i++) {
            if (i != dim) {
                new_shape.push_back(input_shape[i]);
                new_strides.push_back(input_strides[i]);
            }
        }
    }

    // If all dimensions were size 1, keep at least one
    if (new_shape.empty()) {
        new_shape.push_back(1);
        new_strides.push_back(1);
    }

    // Create result tensor sharing storage (zero-copy metadata-only operation)
    Tensor result;
    result.impl_ = std::make_shared<TensorImpl>(*(input.impl_));
    result.impl_->shape = std::move(new_shape);
    result.impl_->strides = std::move(new_strides);

    return result;
}

/**
 * @brief Unsqueeze - add dimension of size 1 (metadata-only)
 */
auto VulkanBackend::dispatchUnsqueeze(const Tensor& input, int64_t dim) -> Tensor {
    auto input_shape = input.shape();
    int32_t ndim = input.ndim();

    // Normalize negative dimension (allow ndim as well for appending)
    if (dim < 0) dim += ndim + 1;
    if (dim < 0 || dim > ndim) {
        throw std::invalid_argument("Unsqueeze: dimension out of range");
    }

    // Create output shape with new dimension of size 1
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    out_shape.insert(out_shape.begin() + dim, 1);

    // Metadata-only operation - just reshape
    return dispatchReshape(input, out_shape);
}

/**
 * @brief Contiguous - ensure tensor is contiguous in memory
 *
 * If already contiguous, returns the input tensor.
 * Otherwise, creates a new contiguous copy.
 */
auto VulkanBackend::dispatchContiguous(const Tensor& input) -> Tensor {
    // If already contiguous, return as-is
    if (input.is_contiguous()) {
        return input;
    }

    // For non-contiguous tensors, we need to reorder the data
    // Strategy: Download to CPU, make contiguous, upload back to GPU

    // Create new contiguous tensor with same shape, dtype, device
    Tensor result(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    // Copy data from GPU to CPU using element-wise access with strides
    const int64_t total_elements = input.numel();
    const size_t element_size = input.dtype_size();
    const int64_t ndims = input.ndim();
    const int64_t base_offset = input.impl_ ? input.impl_->offset : 0;

    // Calculate the size of underlying storage we need to copy
    // We need to copy enough data to cover all accessed elements considering strides and offset
    int64_t max_offset = base_offset;
    if (ndims > 0) {
        auto strides = input.strides();
        auto shape = input.shape();
        for (int64_t dim = 0; dim < ndims; ++dim) {
            max_offset += (shape[dim] - 1) * std::abs(strides[dim]);
        }
    }
    int64_t storage_elements_needed = max_offset + 1;

    // Allocate temporary CPU buffers
    std::vector<uint8_t> cpu_input_buffer(storage_elements_needed * element_size);
    std::vector<uint8_t> cpu_output_buffer(total_elements * element_size);

    // Download input tensor data from GPU to CPU
    // Get the base storage pointer (without offset) for copying
    const void* base_storage_ptr = input.impl_ ? input.impl_->storage->data() : input.data_ptr();
    copy(cpu_input_buffer.data(), base_storage_ptr,
         storage_elements_needed * element_size, CopyKind::DeviceToHost);

    // Reorder data on CPU using strides and offset

    if (ndims == 0) {
        // Scalar tensor - direct copy with offset
        std::memcpy(cpu_output_buffer.data(),
                   cpu_input_buffer.data() + base_offset * element_size,
                   element_size);
    } else {
        // Multi-dimensional copy using stride calculations with base offset
        std::vector<int64_t> indices(ndims, 0);
        int64_t dst_offset = 0;

        auto strides = input.strides();
        auto shape = input.shape();

        for (int64_t i = 0; i < total_elements; ++i) {
            // Calculate source offset using strides and add base offset
            int64_t src_offset = base_offset;
            for (int64_t dim = 0; dim < ndims; ++dim) {
                src_offset += indices[dim] * strides[dim];
            }

            // Copy single element
            std::memcpy(cpu_output_buffer.data() + dst_offset * element_size,
                       cpu_input_buffer.data() + src_offset * element_size,
                       element_size);

            // Increment destination offset (contiguous layout)
            ++dst_offset;

            // Increment indices (iterate through tensor in row-major order)
            for (int64_t dim = ndims - 1; dim >= 0; --dim) {
                ++indices[dim];
                if (indices[dim] < shape[dim]) {
                    break;
                }
                indices[dim] = 0;
            }
        }
    }

    // Upload reordered data back to GPU
    copy(result.data_ptr(), cpu_output_buffer.data(),
         total_elements * element_size, CopyKind::HostToDevice);

    return result;
}

// ============================================================================
// Memory Operations Implementation
// ============================================================================

/**
 * @brief Create tensor filled with zeros
 */
auto VulkanBackend::dispatchZeros(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor {
    // Create tensor with given shape
    Tensor output(shape, dtype, device);

    // Special case: empty tensors don't need GPU operations
    int64_t numel = 1;
    for (auto dim : shape) {
        numel *= dim;
    }
    if (numel == 0) {
        return output;  // Empty tensor, nothing to fill
    }

    // Fill with zeros using fill operation
    int32_t device_id = device.index;
    auto* pipeline = getPipeline("fill", device_id);

    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());
    size_t buffer_size_out = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t n;
        float value;
    } push_constants;

    push_constants.n = static_cast<uint32_t>(output.numel());
    push_constants.value = 0.0f;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = (output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Fill tensor with scalar value using compute shader
 */
auto VulkanBackend::dispatchFill(const Tensor& input, float value) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("fill", device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());
    size_t buffer_size_out = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t n;
        uint32_t value_bits;
    } push_constants;

    push_constants.n = static_cast<uint32_t>(output.numel());

    // Convert value to bits based on dtype
    if (input.dtype() == DType::Int32) {
        int32_t int_value = static_cast<int32_t>(value);
        std::memcpy(&push_constants.value_bits, &int_value, sizeof(uint32_t));
    } else {
        // For float types, use the float bits directly
        std::memcpy(&push_constants.value_bits, &value, sizeof(uint32_t));
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = (output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Clone tensor - deep copy via device-to-device buffer copy
 */
auto VulkanBackend::dispatchClone(const Tensor& input) -> Tensor {
    auto input_shape = input.shape();
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // Use Vulkan's vkCmdCopyBuffer for efficient device-to-device copy
    size_t bytes = input.numel() * input.dtype_size();
    copy(output.data_ptr(), input.data_ptr(), bytes, CopyKind::DeviceToDevice);

    return output;
}

/**
 * @brief Expand tensor to larger size using broadcasting
 */
auto VulkanBackend::dispatchExpand(const Tensor& input, const std::vector<int64_t>& shape) -> Tensor {
    int32_t device_id = input.device().index;
    // Select correct pipeline based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "expand_f64";
    } else if (is_float16) {
        shader_name = "expand_f16";
    } else {
        shader_name = "expand";
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor with new shape
    Tensor output(shape, input.dtype(), input.device());

    VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());
    // For Float16, the shader works with uint32 (packed pairs), so descriptor size needs
    // to cover the full uint32 reads/writes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (is_float16) {
        // Round up to 4-byte boundary (minimum uint32 size for shader access)
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate strides for input tensor
    auto input_shape = input.shape();
    std::vector<uint32_t> input_strides(input_shape.size());
    uint32_t stride = 1;
    for (int i = static_cast<int>(input_shape.size()) - 1; i >= 0; i--) {
        input_strides[i] = stride;
        stride *= static_cast<uint32_t>(input_shape[i]);
    }

    // Prepare push constants
    struct PushConstants {
        uint32_t n_elements;
        uint32_t input_ndim;
        uint32_t output_ndim;
        uint32_t input_shape[8];
        uint32_t output_shape[8];
        uint32_t input_strides[8];
    } push_constants = {}; // Zero-initialize all fields including arrays

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.input_ndim = static_cast<uint32_t>(input_shape.size());
    push_constants.output_ndim = static_cast<uint32_t>(shape.size());

    for (size_t i = 0; i < input_shape.size() && i < 8; i++) {
        push_constants.input_shape[i] = static_cast<uint32_t>(input_shape[i]);
        push_constants.input_strides[i] = input_strides[i];
    }
    for (size_t i = 0; i < shape.size() && i < 8; i++) {
        push_constants.output_shape[i] = static_cast<uint32_t>(shape[i]);
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Float16 shader processes 2 elements per thread
    uint32_t num_items = is_float16 ? ((output.numel() + 1) / 2) : output.numel();
    uint32_t workgroups = (num_items + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Concatenate N tensors along a dimension
 *
 * This implementation uses Vulkan buffer copy operations to concatenate
 * tensors without requiring dynamic descriptor sets. Each input tensor
 * is copied to its appropriate region in the output buffer.
 */
auto VulkanBackend::dispatchCat(const std::vector<Tensor>& inputs, int64_t dim) -> Tensor {
    if (inputs.empty()) {
        throw std::invalid_argument("VulkanBackend::dispatchCat requires at least 1 input tensor");
    }

    // Special case: single tensor just clone it
    if (inputs.size() == 1) {
        return dispatchClone(inputs[0]);
    }

    // IMPORTANT: Make all inputs contiguous
    // Sliced tensors (like those from roll operation) are not contiguous
    // and have different strides/offsets that don't work with simple buffer copying
    std::vector<Tensor> contiguous_inputs;
    contiguous_inputs.reserve(inputs.size());
    for (const auto& input : inputs) {
        if (!input.is_contiguous()) {
            contiguous_inputs.push_back(dispatchContiguous(input));
        } else {
            contiguous_inputs.push_back(input);
        }
    }

    const Tensor& first_input = contiguous_inputs[0];
    int32_t device_id = first_input.device().index;
    auto first_shape = first_input.shape();
    size_t ndim = first_shape.size();

    // Normalize dimension
    if (dim < 0) {
        dim += static_cast<int64_t>(ndim);
    }
    if (dim < 0 || dim >= static_cast<int64_t>(ndim)) {
        throw std::invalid_argument("Invalid concatenation dimension");
    }

    // Validate all tensors have compatible shapes and calculate output shape
    std::vector<int64_t> output_shape(first_shape.begin(), first_shape.end());
    int64_t total_dim_size = first_shape[dim];

    for (size_t i = 1; i < contiguous_inputs.size(); ++i) {
        auto shape = contiguous_inputs[i].shape();
        if (shape.size() != ndim) {
            throw std::invalid_argument("All input tensors must have the same number of dimensions");
        }
        for (size_t j = 0; j < ndim; ++j) {
            if (j != static_cast<size_t>(dim) && shape[j] != first_shape[j]) {
                throw std::invalid_argument("All input tensors must have the same shape except along concatenation dimension");
            }
        }
        total_dim_size += shape[dim];
    }

    output_shape[dim] = total_dim_size;

    // Create output tensor
    Tensor output(output_shape, first_input.dtype(), first_input.device());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());
    size_t element_size = first_input.dtype_size();

    // Calculate strides for copying
    // outer_size: number of "blocks" before the cat dimension
    // inner_size: size of each contiguous chunk within the cat dimension
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= first_shape[i];
    }

    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < ndim; i++) {
        inner_size *= first_shape[i];
    }

    // Begin command buffer for all copy operations
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);

    // Copy each input tensor to the appropriate location in output
    int64_t current_offset_in_cat_dim = 0;

    for (const auto& input : contiguous_inputs) {
        VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
        int64_t input_dim_size = input.shape()[dim];

        // For each outer block, copy the input data to the correct position
        for (int64_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
            // Calculate source and destination offsets
            int64_t src_offset = outer_idx * input_dim_size * inner_size * element_size;
            int64_t dst_offset = outer_idx * total_dim_size * inner_size * element_size +
                                current_offset_in_cat_dim * inner_size * element_size;

            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = static_cast<VkDeviceSize>(src_offset);
            copyRegion.dstOffset = static_cast<VkDeviceSize>(dst_offset);
            copyRegion.size = static_cast<VkDeviceSize>(input_dim_size * inner_size * element_size);

            vkCmdCopyBuffer(cmdBuffer, buffer_in, buffer_out, 1, &copyRegion);
        }

        current_offset_in_cat_dim += input_dim_size;
    }

    // Add a memory barrier to ensure all copies complete before any subsequent operations
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &barrier,
        0, nullptr,
        0, nullptr
    );

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Clamp tensor values to [min, max] range
 */
auto VulkanBackend::dispatchClamp(const Tensor& input, float min_value, float max_value) -> Tensor {
    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    std::string shader_name = is_float64 ? "clamp_f64" : "clamp";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    auto input_shape = input.shape();
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    if (is_float64) {
        // Float64 push constants with double min/max values
        struct PushConstantsF64 {
            uint32_t n_elements;
            uint32_t padding;
            double min_value;
            double max_value;
        } push_constants_f64;

        push_constants_f64.n_elements = static_cast<uint32_t>(output.numel());
        push_constants_f64.padding = 0;
        push_constants_f64.min_value = static_cast<double>(min_value);
        push_constants_f64.max_value = static_cast<double>(max_value);

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF64), &push_constants_f64);
    } else {
        // Float32 push constants
        struct PushConstants {
            uint32_t n_elements;
            float min_value;
            float max_value;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());
        push_constants.min_value = min_value;
        push_constants.max_value = max_value;

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);
    }

    uint32_t workgroups = (output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Dispatch forward activation operations (element-wise)
 * Operations: 0=relu, 1=sigmoid, 2=tanh, 3=gelu, 4=leaky_relu, 5=swish
 */
auto VulkanBackend::dispatchActivation(const std::string& op_name,
                                        const Tensor& input,
                                        uint32_t opcode,
                                        float param) -> Tensor {
    int32_t device_id = input.device().index;

    // Select correct pipeline based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "activations_f64";
    } else if (is_float16) {
        shader_name = "activations_f16";
    } else {
        shader_name = "activations";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    auto shape = input.shape();
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // Prepare push constants - use different structure for Float32 vs Float64
    struct PushConstantsF32 {
        uint32_t n;
        uint32_t activation;
        float alpha;
    };
    struct PushConstantsF64 {
        uint32_t n;
        uint32_t activation;
        double alpha;
    };

    PushConstantsF32 push_constants_f32;
    PushConstantsF64 push_constants_f64;
    void* push_constants_ptr;
    size_t push_constants_size;

    if (is_float64) {
        push_constants_f64.n = static_cast<uint32_t>(input.numel());
        push_constants_f64.activation = opcode;
        push_constants_f64.alpha = static_cast<double>(param);
        push_constants_ptr = &push_constants_f64;
        push_constants_size = sizeof(PushConstantsF64);
    } else {
        push_constants_f32.n = static_cast<uint32_t>(input.numel());
        push_constants_f32.activation = opcode;
        push_constants_f32.alpha = param;
        push_constants_ptr = &push_constants_f32;
        push_constants_size = sizeof(PushConstantsF32);
    }

    // Get VkBuffer handles
    VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Setup descriptor set
    // Binding 0: input, Binding 1: output
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_in, buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, push_constants_size, push_constants_ptr);

    // Dispatch compute workgroups
    // For Float16, shader processes 2 elements per thread (packed pairs)
    uint32_t num_elements = static_cast<uint32_t>(input.numel());
    uint32_t workgroups;
    if (is_float16) {
        // Each thread handles 2 elements (pair), 256 threads per workgroup
        uint32_t num_pairs = (num_elements + 1) / 2;
        workgroups = (num_pairs + 255) / 256;
    } else {
        workgroups = (num_elements + 255) / 256;
    }
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Dispatch backward activation operations (element-wise)
 * Operations: 0=relu, 1=sigmoid, 2=tanh, 3=leaky_relu, 4=gelu
 */
auto VulkanBackend::dispatchActivationBackward(const std::string& op_name,
                                                const Tensor& grad_output,
                                                const Tensor& input_or_output,
                                                uint32_t opcode,
                                                float param) -> Tensor {
    int32_t device_id = grad_output.device().index;

    // Select correct pipeline based on dtype
    bool is_float64 = (grad_output.dtype() == DType::Float64);
    bool is_float16 = (grad_output.dtype() == DType::Float16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "activations_backward_f64";
    } else if (is_float16) {
        shader_name = "activations_backward_f16";
    } else {
        shader_name = "activations_backward";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    auto shape = grad_output.shape();
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor grad_input(output_shape, grad_output.dtype(), grad_output.device());

    // Prepare push constants
    struct PushConstants {
        uint32_t n;      // Number of elements
        uint32_t op;     // Operation code
        float alpha;     // For leaky_relu_backward
    } push_constants;

    push_constants.n = static_cast<uint32_t>(grad_output.numel());
    push_constants.op = opcode;
    push_constants.alpha = param;

    // Get VkBuffer handles
    VkBuffer buffer_grad_out = getVulkanBuffer(grad_output.data_ptr());
    VkBuffer buffer_input_or_output = getVulkanBuffer(input_or_output.data_ptr());
    VkBuffer buffer_grad_in = getVulkanBuffer(grad_input.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_input_or_output = input_or_output.numel() * input_or_output.dtype_size();
    size_t buffer_size_grad_in = grad_input.numel() * grad_input.dtype_size();

    // Setup descriptor set
    // Binding 0: grad_output, Binding 1: input_or_output, Binding 2: grad_input
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_input_or_output},
        {2, buffer_grad_in}
    };
    std::vector<size_t> sizes = {buffer_size_grad_out, buffer_size_input_or_output, buffer_size_grad_in};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch compute workgroups
    // For Float16, each thread processes 2 elements (pairs)
    uint32_t num_threads = is_float16 ? ((grad_output.numel() + 1) / 2) : grad_output.numel();
    uint32_t workgroups = (num_threads + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

/**
 * @brief Dispatch swish backward operation
 * Formula: swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
 * where swish(x) = x * sigmoid(x)
 */
auto VulkanBackend::dispatchSwishBackward(const Tensor& grad_output,
                                           const Tensor& input) -> Tensor {
    int32_t device_id = grad_output.device().index;

    // Select correct pipeline based on dtype
    bool is_float64 = (grad_output.dtype() == DType::Float64);
    std::string shader_name = is_float64 ? "swish_backward_f64" : "swish_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    auto shape = grad_output.shape();
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor grad_input(output_shape, grad_output.dtype(), grad_output.device());

    // Prepare push constants
    struct PushConstants {
        uint32_t n;  // Number of elements
    } push_constants;

    push_constants.n = static_cast<uint32_t>(grad_output.numel());

    // Get VkBuffer handles
    VkBuffer buffer_grad_out = getVulkanBuffer(grad_output.data_ptr());
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_grad_in = getVulkanBuffer(grad_input.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_grad_out = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_grad_in = grad_input.numel() * grad_input.dtype_size();

    // Setup descriptor set
    // Binding 0: grad_output, Binding 1: input, Binding 2: grad_input
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_input},
        {2, buffer_grad_in}
    };
    std::vector<size_t> sizes = {buffer_size_grad_out, buffer_size_input, buffer_size_grad_in};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch compute workgroups
    uint32_t workgroups = (grad_output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

/**
 * @brief Dispatch softmax backward operation
 * Formula: grad_input = output * (grad_output - dot(grad_output, output))
 */
auto VulkanBackend::dispatchSoftmaxBackward(const Tensor& grad_output,
                                             const Tensor& output,
                                             int64_t dim) -> Tensor {
    int32_t device_id = grad_output.device().index;

    // Select correct pipeline based on dtype
    bool is_float64 = (grad_output.dtype() == DType::Float64);
    bool is_float16 = (grad_output.dtype() == DType::Float16);
    std::string shader_name;
    if (is_float64) {
        shader_name = "softmax_backward_f64";
    } else if (is_float16) {
        shader_name = "softmax_backward_f16";
    } else {
        shader_name = "softmax_backward";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    auto shape = output.shape();
    if (dim < 0) {
        dim += shape.size();
    }

    // Create output tensor
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor grad_input(output_shape, grad_output.dtype(), grad_output.device());

    // Calculate dimension strides
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= shape[i];
    }
    int64_t dim_size = shape[dim];
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        inner_size *= shape[i];
    }

    // Prepare push constants
    struct PushConstants {
        uint32_t outer_size;
        uint32_t dim_size;
        uint32_t inner_size;
    } push_constants;

    push_constants.outer_size = static_cast<uint32_t>(outer_size);
    push_constants.dim_size = static_cast<uint32_t>(dim_size);
    push_constants.inner_size = static_cast<uint32_t>(inner_size);

    // Get VkBuffer handles
    VkBuffer buffer_grad_out = getVulkanBuffer(grad_output.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());
    VkBuffer buffer_grad_in = getVulkanBuffer(grad_input.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size = grad_output.numel() * grad_output.dtype_size();

    // Setup descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_output},
        {2, buffer_grad_in}
    };
    std::vector<size_t> sizes = {buffer_size, buffer_size, buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch: one thread per (outer, inner) position
    uint32_t total_threads = static_cast<uint32_t>(outer_size * inner_size);
    uint32_t workgroups = (total_threads + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

/**
 * @brief Dispatch log_softmax backward operation
 * Formula: grad_input = grad_output - exp(output) * sum(grad_output)
 */
auto VulkanBackend::dispatchLogSoftmaxBackward(const Tensor& grad_output,
                                                const Tensor& output,
                                                int64_t dim) -> Tensor {
    int32_t device_id = grad_output.device().index;

    // Select correct pipeline based on dtype
    bool is_float64 = (grad_output.dtype() == DType::Float64);
    std::string shader_name = is_float64 ? "log_softmax_backward_f64" : "log_softmax_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    auto shape = output.shape();
    if (dim < 0) {
        dim += shape.size();
    }

    // Create output tensor
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor grad_input(output_shape, grad_output.dtype(), grad_output.device());

    // Calculate dimension strides
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= shape[i];
    }
    int64_t dim_size = shape[dim];
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        inner_size *= shape[i];
    }

    // Prepare push constants
    struct PushConstants {
        uint32_t outer_size;
        uint32_t dim_size;
        uint32_t inner_size;
    } push_constants;

    push_constants.outer_size = static_cast<uint32_t>(outer_size);
    push_constants.dim_size = static_cast<uint32_t>(dim_size);
    push_constants.inner_size = static_cast<uint32_t>(inner_size);

    // Get VkBuffer handles
    VkBuffer buffer_grad_out = getVulkanBuffer(grad_output.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());
    VkBuffer buffer_grad_in = getVulkanBuffer(grad_input.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size = grad_output.numel() * grad_output.dtype_size();

    // Setup descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_output},
        {2, buffer_grad_in}
    };
    std::vector<size_t> sizes = {buffer_size, buffer_size, buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch: one thread per (outer, inner) position
    uint32_t total_threads = static_cast<uint32_t>(outer_size * inner_size);
    uint32_t workgroups = (total_threads + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

// ============================================================================
// Pooling Operations Implementation (OpAttributes versions)
// ============================================================================

auto VulkanBackend::dispatchAvgPool2dForward(const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("avg_pool2d requires 4D input (N, C, H, W)");
    }

    // Extract attributes
    int64_t kernel_h = std::stoll(attrs.at("kernel_h"));
    int64_t kernel_w = std::stoll(attrs.at("kernel_w"));
    int64_t stride_h = attrs.contains("stride_h") ? std::stoll(attrs.at("stride_h")) : kernel_h;
    int64_t stride_w = attrs.contains("stride_w") ? std::stoll(attrs.at("stride_w")) : kernel_w;
    int64_t padding_h = attrs.contains("padding_h") ? std::stoll(attrs.at("padding_h")) : 0;
    int64_t padding_w = attrs.contains("padding_w") ? std::stoll(attrs.at("padding_w")) : 0;
    int64_t count_include_pad = attrs.contains("count_include_pad") ? std::stoll(attrs.at("count_include_pad")) : 1;

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    // Calculate output dimensions
    int64_t out_height = (in_height + 2 * padding_h - kernel_h) / stride_h + 1;
    int64_t out_width = (in_width + 2 * padding_w - kernel_w) / stride_w + 1;

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("avg_pool2d", device_id);

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, channels, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());

    // Get VkBuffer handles
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_input},
        {1, buffer_output}
    };
    std::vector<size_t> sizes = {buffer_size_input, buffer_size_output};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t count_include_pad;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.padding_h = static_cast<uint32_t>(padding_h);
    push_constants.padding_w = static_cast<uint32_t>(padding_w);
    push_constants.count_include_pad = static_cast<uint32_t>(count_include_pad);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    uint32_t workgroups = static_cast<uint32_t>((output.numel() + 255) / 256);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchMaxPool2dForward(const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("max_pool2d requires 4D input (N, C, H, W)");
    }

    // Extract attributes
    int64_t kernel_h = std::stoll(attrs.at("kernel_h"));
    int64_t kernel_w = std::stoll(attrs.at("kernel_w"));
    int64_t stride_h = attrs.contains("stride_h") ? std::stoll(attrs.at("stride_h")) : kernel_h;
    int64_t stride_w = attrs.contains("stride_w") ? std::stoll(attrs.at("stride_w")) : kernel_w;
    int64_t padding_h = attrs.contains("padding_h") ? std::stoll(attrs.at("padding_h")) : 0;
    int64_t padding_w = attrs.contains("padding_w") ? std::stoll(attrs.at("padding_w")) : 0;
    int64_t dilation_h = attrs.contains("dilation_h") ? std::stoll(attrs.at("dilation_h")) : 1;
    int64_t dilation_w = attrs.contains("dilation_w") ? std::stoll(attrs.at("dilation_w")) : 1;

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    // Calculate output dimensions with dilation
    int64_t effective_kernel_h = (kernel_h - 1) * dilation_h + 1;
    int64_t effective_kernel_w = (kernel_w - 1) * dilation_w + 1;
    int64_t out_height = (in_height + 2 * padding_h - effective_kernel_h) / stride_h + 1;
    int64_t out_width = (in_width + 2 * padding_w - effective_kernel_w) / stride_w + 1;

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("max_pool2d", device_id);

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, channels, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());

    // Get VkBuffer handles
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_input},
        {1, buffer_output}
    };
    std::vector<size_t> sizes = {buffer_size_input, buffer_size_output};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t dilation_h;
        uint32_t dilation_w;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.padding_h = static_cast<uint32_t>(padding_h);
    push_constants.padding_w = static_cast<uint32_t>(padding_w);
    push_constants.dilation_h = static_cast<uint32_t>(dilation_h);
    push_constants.dilation_w = static_cast<uint32_t>(dilation_w);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    uint32_t workgroups = static_cast<uint32_t>((output.numel() + 255) / 256);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAvgPool2dBackward(const Tensor& grad_output, const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("avg_pool2d_backward requires 4D input (N, C, H, W)");
    }

    // Extract attributes
    int64_t kernel_h = std::stoll(attrs.at("kernel_h"));
    int64_t kernel_w = std::stoll(attrs.at("kernel_w"));
    int64_t stride_h = attrs.contains("stride_h") ? std::stoll(attrs.at("stride_h")) : kernel_h;
    int64_t stride_w = attrs.contains("stride_w") ? std::stoll(attrs.at("stride_w")) : kernel_w;
    int64_t padding_h = attrs.contains("padding_h") ? std::stoll(attrs.at("padding_h")) : 0;
    int64_t padding_w = attrs.contains("padding_w") ? std::stoll(attrs.at("padding_w")) : 0;
    int64_t count_include_pad = attrs.contains("count_include_pad") ? std::stoll(attrs.at("count_include_pad")) : 1;

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    auto grad_out_shape = grad_output.shape();
    int64_t out_height = grad_out_shape[2];
    int64_t out_width = grad_out_shape[3];

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("avg_pool2d_backward", device_id);

    // Create gradient input tensor (same shape as input), initialized to zero
    std::vector<int64_t> grad_input_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(grad_input_shape, input.dtype(), input.device());

    // Zero initialize grad_input using fill operation
    grad_input = dispatchFill(grad_input, 0.0f);

    // Get VkBuffer handles
    VkBuffer buffer_grad_output = getVulkanBuffer(grad_output.data_ptr());
    VkBuffer buffer_grad_input = getVulkanBuffer(grad_input.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_grad_output = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_grad_input = grad_input.numel() * grad_input.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_grad_output},
        {1, buffer_grad_input}
    };
    std::vector<size_t> sizes = {buffer_size_grad_output, buffer_size_grad_input};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t count_include_pad;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(grad_output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.padding_h = static_cast<uint32_t>(padding_h);
    push_constants.padding_w = static_cast<uint32_t>(padding_w);
    push_constants.count_include_pad = static_cast<uint32_t>(count_include_pad);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (iterate over grad_output elements)
    uint32_t workgroups = static_cast<uint32_t>((grad_output.numel() + 255) / 256);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

auto VulkanBackend::dispatchMaxPool2dBackward(const Tensor& grad_output, const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("max_pool2d_backward requires 4D input (N, C, H, W)");
    }

    // Extract attributes
    int64_t kernel_h = std::stoll(attrs.at("kernel_h"));
    int64_t kernel_w = std::stoll(attrs.at("kernel_w"));
    int64_t stride_h = attrs.contains("stride_h") ? std::stoll(attrs.at("stride_h")) : kernel_h;
    int64_t stride_w = attrs.contains("stride_w") ? std::stoll(attrs.at("stride_w")) : kernel_w;
    int64_t padding_h = attrs.contains("padding_h") ? std::stoll(attrs.at("padding_h")) : 0;
    int64_t padding_w = attrs.contains("padding_w") ? std::stoll(attrs.at("padding_w")) : 0;
    int64_t dilation_h = attrs.contains("dilation_h") ? std::stoll(attrs.at("dilation_h")) : 1;
    int64_t dilation_w = attrs.contains("dilation_w") ? std::stoll(attrs.at("dilation_w")) : 1;

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    auto grad_out_shape = grad_output.shape();
    int64_t out_height = grad_out_shape[2];
    int64_t out_width = grad_out_shape[3];

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("max_pool2d_backward", device_id);

    // Create gradient input tensor (same shape as input), initialized to zero
    std::vector<int64_t> grad_input_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(grad_input_shape, input.dtype(), input.device());

    // Zero initialize grad_input using fill operation
    grad_input = dispatchFill(grad_input, 0.0f);

    // Get VkBuffer handles
    VkBuffer buffer_grad_output = getVulkanBuffer(grad_output.data_ptr());
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_grad_input = getVulkanBuffer(grad_input.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_grad_output = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_grad_input = grad_input.numel() * grad_input.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_grad_output},
        {1, buffer_input},
        {2, buffer_grad_input}
    };
    std::vector<size_t> sizes = {buffer_size_grad_output, buffer_size_input, buffer_size_grad_input};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t out_height;
        uint32_t out_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride_h;
        uint32_t stride_w;
        uint32_t padding_h;
        uint32_t padding_w;
        uint32_t dilation_h;
        uint32_t dilation_w;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(grad_output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride_h = static_cast<uint32_t>(stride_h);
    push_constants.stride_w = static_cast<uint32_t>(stride_w);
    push_constants.padding_h = static_cast<uint32_t>(padding_h);
    push_constants.padding_w = static_cast<uint32_t>(padding_w);
    push_constants.dilation_h = static_cast<uint32_t>(dilation_h);
    push_constants.dilation_w = static_cast<uint32_t>(dilation_w);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (iterate over grad_output elements)
    uint32_t workgroups = static_cast<uint32_t>((grad_output.numel() + 255) / 256);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

// ============================================================================
// MaxPool2d Backward with Indices (scatter-based)
// ============================================================================

auto VulkanBackend::dispatchMaxPool2dBackwardWithIndices(const Tensor& grad_output, const Tensor& indices,
                                                          int64_t H_in, int64_t W_in) -> Tensor {
    // This is an index-based scatter operation - we use CPU fallback for reliability
    // since GPU scatter operations require atomic additions.
    // The indices tensor contains the linear indices into the input tensor where max values were found.

    auto grad_out_shape = grad_output.shape();
    if (grad_out_shape.size() != 4) {
        throw std::invalid_argument("max_pool2d_backward requires 4D grad_output (N, C, H_out, W_out)");
    }

    int64_t N = grad_out_shape[0];
    int64_t C = grad_out_shape[1];
    int64_t H_out = grad_out_shape[2];
    int64_t W_out = grad_out_shape[3];

    // Move tensors to CPU for the scatter operation
    Tensor cpu_grad_output = grad_output.to(Device::cpu());
    Tensor cpu_indices = indices.to(Device::cpu());

    // Create grad_input tensor on CPU, initialized to zeros
    Tensor cpu_grad_input({N, C, H_in, W_in}, grad_output.dtype(), Device::cpu());
    std::memset(cpu_grad_input.data_ptr(), 0, cpu_grad_input.numel() * cpu_grad_input.dtype_size());

    // Scatter gradients to positions indicated by indices
    int64_t grad_out_numel = N * C * H_out * W_out;

    if (grad_output.dtype() == DType::Float32) {
        float* grad_input_data = cpu_grad_input.data<float>();
        const float* grad_output_data = cpu_grad_output.data<float>();
        const float* indices_data = cpu_indices.data<float>();

        for (int64_t i = 0; i < grad_out_numel; ++i) {
            int64_t max_idx = static_cast<int64_t>(indices_data[i]);
            if (max_idx >= 0 && max_idx < N * C * H_in * W_in) {
                grad_input_data[max_idx] += grad_output_data[i];
            }
        }
    } else if (grad_output.dtype() == DType::Float64) {
        double* grad_input_data = cpu_grad_input.data<double>();
        const double* grad_output_data = cpu_grad_output.data<double>();
        const double* indices_data = cpu_indices.data<double>();

        for (int64_t i = 0; i < grad_out_numel; ++i) {
            int64_t max_idx = static_cast<int64_t>(indices_data[i]);
            if (max_idx >= 0 && max_idx < N * C * H_in * W_in) {
                grad_input_data[max_idx] += grad_output_data[i];
            }
        }
    } else if (grad_output.dtype() == DType::Float16) {
        // For Float16, work with raw half-precision data
        uint16_t* grad_input_data = reinterpret_cast<uint16_t*>(cpu_grad_input.data_ptr());
        const uint16_t* grad_output_data = reinterpret_cast<const uint16_t*>(cpu_grad_output.data_ptr());
        const uint16_t* indices_data = reinterpret_cast<const uint16_t*>(cpu_indices.data_ptr());

        // Helper lambda to convert half to float
        auto half_to_float = [](uint16_t h) -> float {
            uint32_t sign = (h & 0x8000) << 16;
            uint32_t exp = (h >> 10) & 0x1F;
            uint32_t mant = h & 0x3FF;

            if (exp == 0) {
                if (mant == 0) {
                    uint32_t result = sign;
                    float f;
                    std::memcpy(&f, &result, sizeof(f));
                    return f;
                }
                while (!(mant & 0x400)) {
                    mant <<= 1;
                    exp--;
                }
                exp++;
                mant &= ~0x400;
                exp += 127 - 15;
                uint32_t result = sign | (exp << 23) | (mant << 13);
                float f;
                std::memcpy(&f, &result, sizeof(f));
                return f;
            } else if (exp == 31) {
                uint32_t result = sign | 0x7F800000 | (mant << 13);
                float f;
                std::memcpy(&f, &result, sizeof(f));
                return f;
            }
            exp += 127 - 15;
            uint32_t result = sign | (exp << 23) | (mant << 13);
            float f;
            std::memcpy(&f, &result, sizeof(f));
            return f;
        };

        // Helper lambda to convert float to half
        auto float_to_half = [](float f) -> uint16_t {
            uint32_t bits;
            std::memcpy(&bits, &f, sizeof(bits));
            uint32_t sign = (bits >> 16) & 0x8000;
            int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = bits & 0x7FFFFF;

            if (exp <= 0) {
                if (exp < -10) return static_cast<uint16_t>(sign);
                mant |= 0x800000;
                uint32_t shift = 14 - exp;
                mant >>= shift;
                return static_cast<uint16_t>(sign | mant);
            } else if (exp >= 31) {
                return static_cast<uint16_t>(sign | 0x7C00);
            }
            return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
        };

        for (int64_t i = 0; i < grad_out_numel; ++i) {
            int64_t max_idx = static_cast<int64_t>(half_to_float(indices_data[i]));
            if (max_idx >= 0 && max_idx < N * C * H_in * W_in) {
                float current = half_to_float(grad_input_data[max_idx]);
                float grad_val = half_to_float(grad_output_data[i]);
                grad_input_data[max_idx] = float_to_half(current + grad_val);
            }
        }
    } else {
        throw std::runtime_error("Unsupported dtype for max_pool2d_backward_with_indices");
    }

    // Move result back to Vulkan device
    return cpu_grad_input.to(grad_output.device());
}

// ============================================================================
// Conv2d Forward Operation (OpAttributes version)
// ============================================================================

auto VulkanBackend::dispatchConv2dForward(const Tensor& input, const Tensor& weight, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    if (input_shape.size() != 4) {
        throw std::invalid_argument("conv2d_forward requires 4D input (N, C, H, W)");
    }
    if (weight_shape.size() != 4) {
        throw std::invalid_argument("conv2d_forward requires 4D weight (out_channels, in_channels, kH, kW)");
    }

    // Extract attributes
    int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : 1;
    int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
    int64_t dilation = attrs.contains("dilation") ? std::stoll(attrs.at("dilation")) : 1;
    int64_t groups = attrs.contains("groups") ? std::stoll(attrs.at("groups")) : 1;
    bool has_bias = attrs.contains("bias") && attrs.at("bias") == "true";

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    int64_t out_channels = weight_shape[0];
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Calculate output dimensions
    int64_t out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int64_t out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "conv2d_forward";
    if (input.dtype() == DType::Float64) {
        shader_name = "conv2d_forward_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "conv2d_forward_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, out_channels, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());

    // Get VkBuffer handles
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_weight = getVulkanBuffer(weight.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_weight = weight.numel() * weight.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();

    // Setup descriptor set bindings (input, weight, bias, output)
    // Note: bias buffer is optional but we need 4 bindings
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_input},
        {1, buffer_weight},
        {2, buffer_output},  // Bias will be at binding 2 if present, otherwise dummy
        {3, buffer_output}   // Output at binding 3
    };
    std::vector<size_t> sizes = {
        buffer_size_input,
        buffer_size_weight,
        has_bias ? 4 : 4,  // Dummy size for bias if not present
        buffer_size_output
    };

    // If bias is present, update binding 2
    if (has_bias) {
        // Bias is typically passed as an additional input tensor
        // For now, we'll create a dummy bias buffer
        // TODO: Extract bias from inputs if available
        bindings[2] = {2, buffer_output};  // Use output buffer as dummy for now
    }

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Create command buffer
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Set push constants
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t in_channels;
        uint32_t out_channels;
        uint32_t in_height;
        uint32_t in_width;
        uint32_t kernel_h;
        uint32_t kernel_w;
        uint32_t stride;
        uint32_t padding;
        uint32_t dilation;
        uint32_t groups;
        uint32_t out_h;
        uint32_t out_w;
        uint32_t has_bias;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.in_channels = static_cast<uint32_t>(in_channels);
    push_constants.out_channels = static_cast<uint32_t>(out_channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.kernel_h = static_cast<uint32_t>(kernel_h);
    push_constants.kernel_w = static_cast<uint32_t>(kernel_w);
    push_constants.stride = static_cast<uint32_t>(stride);
    push_constants.padding = static_cast<uint32_t>(padding);
    push_constants.dilation = static_cast<uint32_t>(dilation);
    push_constants.groups = static_cast<uint32_t>(groups);
    push_constants.out_h = static_cast<uint32_t>(out_height);
    push_constants.out_w = static_cast<uint32_t>(out_width);
    push_constants.has_bias = has_bias ? 1u : 0u;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    uint32_t workgroups = static_cast<uint32_t>((output.numel() + 255) / 256);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Full Operation - Create tensor filled with specific value
// ============================================================================

auto VulkanBackend::dispatchFull(const std::vector<int64_t>& shape, float value, DType dtype) -> Tensor {
    // Create tensor on first available Vulkan device
    Device device(Device::Type::Vulkan, 0);
    Tensor output(shape, dtype, device);

    int32_t device_id = device.index;

    // Select shader based on dtype
    bool is_float64 = (dtype == DType::Float64);
    bool is_float16 = (dtype == DType::Float16);
    bool is_int8 = (dtype == DType::Int8);
    bool is_uint8 = (dtype == DType::UInt8);
    bool is_int64 = (dtype == DType::Int64);
    bool is_bool = (dtype == DType::Bool);
    std::string shader_name;
    if (is_float64) {
        shader_name = "full_f64";
    } else if (is_float16) {
        shader_name = "full_f16";
    } else if (is_int8) {
        shader_name = "full_i8";
    } else if (is_uint8 || is_bool) {
        // Bool is stored as uint8_t, so use the same shader
        shader_name = "full_uint8";
    } else if (is_int64) {
        shader_name = "full_i64";
    } else {
        shader_name = "full";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());
    // For Float16, the shader works with uint32 (packed pairs), so descriptor size needs
    // to cover the full uint32 writes. Even for 1 element, shader writes a full uint32.
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (is_float16) {
        // Round up to 4-byte boundary (minimum uint32 size for shader access)
        size_t num_pairs = (output.numel() + 1) / 2;
        buffer_size_out = num_pairs * 4;
    }

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Use different push constants structure based on dtype
    if (is_float64) {
        struct PushConstantsF64 {
            uint32_t n_elements;
            uint32_t padding;  // Alignment padding
            double fill_value;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());
        push_constants.padding = 0;
        push_constants.fill_value = static_cast<double>(value);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF64), &push_constants);

        uint32_t workgroups = (output.numel() + 255) / 256;
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // Float16 uses a dedicated shader with half-precision packing
    if (is_float16) {
        struct PushConstantsF16 {
            uint32_t n_elements;
            uint32_t fill_value_f16;  // Float16 bits in lower 16 bits
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());

        // Convert float to Float16 bits
        Float16 f16_value(value);
        push_constants.fill_value_f16 = static_cast<uint32_t>(f16_value.bits);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF16), &push_constants);

        // Each thread handles 2 float16 elements, so we need half the workgroups
        uint32_t num_pairs = (output.numel() + 1) / 2;
        uint32_t workgroups = (num_pairs + 255) / 256;
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        VkMemoryBarrier memoryBarrier{};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

        vkCmdPipelineBarrier(
            cmdBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            0,
            1, &memoryBarrier,
            0, nullptr,
            0, nullptr
        );

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // Int8 uses a dedicated shader that packs 4 elements per uint32
    if (is_int8) {
        struct PushConstantsI8 {
            uint32_t n_elements;
            uint32_t fill_value_i8;  // Int8 bits in lower 8 bits
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());
        // Convert float value to int8 and store in lower 8 bits
        int8_t i8_value = static_cast<int8_t>(value);
        push_constants.fill_value_i8 = static_cast<uint32_t>(static_cast<uint8_t>(i8_value));

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsI8), &push_constants);

        // Each thread handles 4 int8 elements, so we need 1/4 the workgroups
        uint32_t num_quads = (output.numel() + 3) / 4;
        uint32_t workgroups = (num_quads + 255) / 256;
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        VkMemoryBarrier memoryBarrier{};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

        vkCmdPipelineBarrier(
            cmdBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            0,
            1, &memoryBarrier,
            0, nullptr,
            0, nullptr
        );

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // UInt8 and Bool use a dedicated shader with direct byte access via 8-bit storage extension
    // Bool is stored as uint8_t (0 = false, non-zero = true)
    if (is_uint8 || is_bool) {
        struct PushConstantsUInt8 {
            uint32_t n_elements;
            uint32_t fill_value_uint8;  // UInt8 bits in lower 8 bits
        } push_constants_u8;

        push_constants_u8.n_elements = static_cast<uint32_t>(output.numel());
        // Convert float value to uint8 and store in lower 8 bits
        // For Bool: any non-zero value becomes 1 (true)
        uint8_t u8_value = is_bool ? (value != 0.0f ? 1 : 0) : static_cast<uint8_t>(value);
        push_constants_u8.fill_value_uint8 = static_cast<uint32_t>(u8_value);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsUInt8), &push_constants_u8);

        // Each thread handles 1 uint8 element
        uint32_t workgroups = (output.numel() + 255) / 256;
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        VkMemoryBarrier memoryBarrier{};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

        vkCmdPipelineBarrier(
            cmdBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            0,
            1, &memoryBarrier,
            0, nullptr,
            0, nullptr
        );

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    // Int64 uses a dedicated shader with 64-bit value split into two 32-bit parts
    if (is_int64) {
        struct PushConstantsI64 {
            uint32_t n_elements;
            uint32_t value_low;   // Low 32 bits of int64
            uint32_t value_high;  // High 32 bits of int64
        } push_constants_i64;

        push_constants_i64.n_elements = static_cast<uint32_t>(output.numel());
        // Convert float value to int64 and split into two uint32
        int64_t i64_value = static_cast<int64_t>(value);
        push_constants_i64.value_low = static_cast<uint32_t>(i64_value & 0xFFFFFFFF);
        push_constants_i64.value_high = static_cast<uint32_t>((static_cast<uint64_t>(i64_value) >> 32) & 0xFFFFFFFF);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsI64), &push_constants_i64);

        uint32_t workgroups = (output.numel() + 255) / 256;
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        VkMemoryBarrier memoryBarrier{};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

        vkCmdPipelineBarrier(
            cmdBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            0,
            1, &memoryBarrier,
            0, nullptr,
            0, nullptr
        );

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    struct PushConstants {
        uint32_t n_elements;
        uint32_t fill_value_bits;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());

    // Convert value to bits based on dtype
    if (dtype == DType::Int32) {
        int32_t int_value = static_cast<int32_t>(value);
        std::memcpy(&push_constants.fill_value_bits, &int_value, sizeof(uint32_t));
    } else {
        // For float types, use the float bits directly
        std::memcpy(&push_constants.fill_value_bits, &value, sizeof(uint32_t));
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = (output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Ones Operation - Create tensor filled with 1.0
// ============================================================================

auto VulkanBackend::dispatchOnes(const std::vector<int64_t>& shape, DType dtype) -> Tensor {
    // For Float64, Int64, UInt8, or Bool, use full() instead since ones shader only supports 32-bit values
    // Bool is stored as uint8_t, so it needs byte-level access like UInt8
    if (dtype == DType::Float64 || dtype == DType::Int64 || dtype == DType::UInt8 || dtype == DType::Bool) {
        return dispatchFull(shape, 1.0, dtype);
    }

    // Create tensor on first available Vulkan device
    Device device(Device::Type::Vulkan, 0);
    Tensor output(shape, dtype, device);

    int32_t device_id = device.index;

    // Select shader based on dtype
    bool is_float16 = (dtype == DType::Float16);
    bool is_int8 = (dtype == DType::Int8);
    std::string shader_name = is_float16 ? "ones_f16" : (is_int8 ? "ones_i8" : "ones");

    auto* pipeline = getPipeline(shader_name, device_id);

    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());
    // For Float16, the shader works with uint32 (packed pairs), so descriptor size needs
    // to cover the full uint32 writes. Even for 1 element, shader writes a full uint32.
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (is_float16) {
        // Round up to 4-byte boundary (minimum uint32 size for shader access)
        size_t num_pairs = (output.numel() + 1) / 2;
        buffer_size_out = num_pairs * 4;
    }

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Float16 uses a different shader that only needs n_elements
    if (is_float16) {
        struct PushConstantsF16 {
            uint32_t n_elements;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF16), &push_constants);

        // Each thread handles 2 float16 elements, so we need half the workgroups
        uint32_t num_pairs = (output.numel() + 1) / 2;
        uint32_t workgroups = (num_pairs + 255) / 256;

        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        VkMemoryBarrier memoryBarrier{};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

        vkCmdPipelineBarrier(
            cmdBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            0,
            1, &memoryBarrier,
            0, nullptr,
            0, nullptr
        );

        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    // Int8 uses a different shader that packs 4 elements per uint32
    if (is_int8) {
        struct PushConstantsI8 {
            uint32_t n_elements;
        } push_constants;

        push_constants.n_elements = static_cast<uint32_t>(output.numel());

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsI8), &push_constants);

        // Each thread handles 4 int8 elements, so we need 1/4 the workgroups
        uint32_t num_quads = (output.numel() + 3) / 4;
        uint32_t workgroups = (num_quads + 255) / 256;
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        VkMemoryBarrier memoryBarrier{};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

        vkCmdPipelineBarrier(
            cmdBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            0,
            1, &memoryBarrier,
            0, nullptr,
            0, nullptr
        );

        endSingleTimeCommands(cmdBuffer, device_id);
        return output;
    }

    struct PushConstants {
        uint32_t n_elements;
        uint32_t one_value_bits;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());

    // Convert value 1.0 or 1 to bits based on dtype
    if (dtype == DType::Int32) {
        int32_t int_value = 1;
        std::memcpy(&push_constants.one_value_bits, &int_value, sizeof(uint32_t));
    } else if (dtype == DType::Int64) {
        int32_t int_value = 1;
        std::memcpy(&push_constants.one_value_bits, &int_value, sizeof(uint32_t));
    } else {
        // For float types, use 1.0f
        float float_value = 1.0f;
        std::memcpy(&push_constants.one_value_bits, &float_value, sizeof(uint32_t));
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = (output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    VkMemoryBarrier memoryBarrier{};
    memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;

    vkCmdPipelineBarrier(
        cmdBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1, &memoryBarrier,
        0, nullptr,
        0, nullptr
    );

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchRand(const std::vector<int64_t>& shape, DType dtype) -> Tensor {
    // Create tensor on first available Vulkan device
    Device device(Device::Type::Vulkan, 0);
    Tensor output(shape, dtype, device);

    // Uniform random distribution requires CPU generation, then copy to GPU
    size_t numel = output.numel();

    // Use C++11 random number generation
    static std::random_device rd;
    static std::mt19937 gen(rd());

    if (dtype == DType::Float32) {
        std::vector<float> cpu_data(numel);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < numel; ++i) {
            cpu_data[i] = dist(gen);
        }
        copy(output.data_ptr(), cpu_data.data(), numel * sizeof(float), CopyKind::HostToDevice);
    } else if (dtype == DType::Float64) {
        std::vector<double> cpu_data(numel);
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < numel; ++i) {
            cpu_data[i] = dist(gen);
        }
        copy(output.data_ptr(), cpu_data.data(), numel * sizeof(double), CopyKind::HostToDevice);
    } else if (dtype == DType::Float16) {
        std::vector<uint16_t> cpu_data(numel);  // Float16 is stored as uint16_t
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < numel; ++i) {
            Float16 val(dist(gen));
            cpu_data[i] = *reinterpret_cast<uint16_t*>(&val);
        }
        copy(output.data_ptr(), cpu_data.data(), numel * sizeof(uint16_t), CopyKind::HostToDevice);
    } else {
        throw std::runtime_error("Unsupported dtype for rand: only Float32, Float64, and Float16 are supported");
    }

    return output;
}

auto VulkanBackend::dispatchRandn(const std::vector<int64_t>& shape, DType dtype) -> Tensor {
    // Create tensor on first available Vulkan device
    Device device(Device::Type::Vulkan, 0);
    Tensor output(shape, dtype, device);

    // Normal random distribution requires CPU generation, then copy to GPU
    size_t numel = output.numel();

    // Use C++11 random number generation
    static std::random_device rd;
    static std::mt19937 gen(rd());

    if (dtype == DType::Float32) {
        std::vector<float> cpu_data(numel);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < numel; ++i) {
            cpu_data[i] = dist(gen);
        }
        copy(output.data_ptr(), cpu_data.data(), numel * sizeof(float), CopyKind::HostToDevice);
    } else if (dtype == DType::Float64) {
        std::vector<double> cpu_data(numel);
        std::normal_distribution<double> dist(0.0, 1.0);
        for (size_t i = 0; i < numel; ++i) {
            cpu_data[i] = dist(gen);
        }
        copy(output.data_ptr(), cpu_data.data(), numel * sizeof(double), CopyKind::HostToDevice);
    } else if (dtype == DType::Float16) {
        std::vector<uint16_t> cpu_data(numel);  // Float16 is stored as uint16_t
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (size_t i = 0; i < numel; ++i) {
            Float16 val(dist(gen));
            cpu_data[i] = *reinterpret_cast<uint16_t*>(&val);
        }
        copy(output.data_ptr(), cpu_data.data(), numel * sizeof(uint16_t), CopyKind::HostToDevice);
    } else {
        throw std::runtime_error("Unsupported dtype for randn: only Float32, Float64, and Float16 are supported");
    }

    return output;
}

// Factory function
extern "C" {
    auto create_backend() -> std::unique_ptr<Backend> {
        return std::make_unique<VulkanBackend>();
    }
}

// New operations for Vulkan backend - to be appended to vulkan_backend.cpp before closing namespace

// Repeat operation - repeats elements along dimensions
auto VulkanBackend::dispatchRepeat(const Tensor& input, const std::vector<int64_t>& repeats) -> Tensor {
    auto input_shape = input.shape();
    int64_t ndim = input_shape.size();

    // Pad repeats with 1s at the front if needed
    std::vector<int64_t> padded_repeats = repeats;
    while (padded_repeats.size() < static_cast<size_t>(ndim)) {
        padded_repeats.insert(padded_repeats.begin(), 1);
    }

    // Calculate output shape
    std::vector<int64_t> output_shape(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        output_shape[i] = input_shape[i] * padded_repeats[i];
    }

    // Use expand-based implementation on GPU
    // First unsqueeze to add repeat dimensions, then use expand
    Tensor current = input;
    for (int64_t dim = ndim - 1; dim >= 0; --dim) {
        if (padded_repeats[dim] > 1) {
            // Unsqueeze at dim+1, expand, then flatten back
            current = current.unsqueeze(dim + 1);
            auto curr_shape = current.shape();
            std::vector<int64_t> expand_shape(curr_shape.begin(), curr_shape.end());
            expand_shape[dim + 1] = padded_repeats[dim];
            current = dispatchExpand(current, expand_shape);

            // Flatten the two dimensions back together
            auto curr_shape2 = current.shape();
            std::vector<int64_t> flatten_shape(curr_shape2.begin(), curr_shape2.end());
            flatten_shape[dim] = flatten_shape[dim] * flatten_shape[dim + 1];
            flatten_shape.erase(flatten_shape.begin() + dim + 1);
            current = dispatchReshape(current, flatten_shape);
        }
    }

    return current;
}

/**
 * @brief Dispatch masked_select operation using CPU fallback
 */
auto VulkanBackend::dispatchMaskedSelect(const Tensor& input, const Tensor& mask) -> Tensor {
    // CPU fallback: masked_select produces variable-size output which is challenging on GPU
    Device cpu_device(Device::Type::CPU, 0);

    // Transfer input and mask to CPU
    Tensor cpu_input = input.to(cpu_device);
    Tensor cpu_mask = mask.to(cpu_device);

    // Validate shapes match
    auto input_shape = cpu_input.shape();
    auto mask_shape = cpu_mask.shape();
    if (!std::equal(input_shape.begin(), input_shape.end(), mask_shape.begin(), mask_shape.end())) {
        throw std::invalid_argument("masked_select: input and mask must have same shape");
    }

    const int64_t numel = cpu_input.numel();

    // Count true values in mask
    int64_t true_count = 0;
    const bool use_float_mask = (cpu_mask.dtype() == DType::Float32);
    const bool* bool_mask_ptr = nullptr;
    const float* float_mask_ptr = nullptr;

    if (cpu_mask.dtype() == DType::Bool) {
        bool_mask_ptr = cpu_mask.data<bool>();
        for (int64_t i = 0; i < numel; ++i) {
            if (bool_mask_ptr[i]) ++true_count;
        }
    } else if (use_float_mask) {
        float_mask_ptr = cpu_mask.data<float>();
        for (int64_t i = 0; i < numel; ++i) {
            if (float_mask_ptr[i] != 0.0f) ++true_count;
        }
    } else {
        throw std::invalid_argument("masked_select: mask tensor must have dtype Bool or Float32");
    }

    // Create output with size = number of true values
    Tensor cpu_output({true_count}, cpu_input.dtype(), cpu_device);

    if (true_count == 0) {
        return cpu_output.to(input.device());
    }

    // Helper to check mask value
    auto is_mask_true = [use_float_mask, bool_mask_ptr, float_mask_ptr](int64_t i) -> bool {
        return use_float_mask ? (float_mask_ptr[i] != 0.0f) : bool_mask_ptr[i];
    };

    // Copy selected elements
    if (cpu_input.dtype() == DType::Float32) {
        const float* input_ptr = cpu_input.data<float>();
        float* output_ptr = cpu_output.data<float>();
        int64_t out_idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            if (is_mask_true(i)) {
                output_ptr[out_idx++] = input_ptr[i];
            }
        }
    } else if (cpu_input.dtype() == DType::Float64) {
        const double* input_ptr = cpu_input.data<double>();
        double* output_ptr = cpu_output.data<double>();
        int64_t out_idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            if (is_mask_true(i)) {
                output_ptr[out_idx++] = input_ptr[i];
            }
        }
    } else if (cpu_input.dtype() == DType::Int32) {
        const int32_t* input_ptr = cpu_input.data<int32_t>();
        int32_t* output_ptr = cpu_output.data<int32_t>();
        int64_t out_idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            if (is_mask_true(i)) {
                output_ptr[out_idx++] = input_ptr[i];
            }
        }
    } else if (cpu_input.dtype() == DType::Int64) {
        const int64_t* input_ptr = cpu_input.data<int64_t>();
        int64_t* output_ptr = cpu_output.data<int64_t>();
        int64_t out_idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            if (is_mask_true(i)) {
                output_ptr[out_idx++] = input_ptr[i];
            }
        }
    } else if (cpu_input.dtype() == DType::Bool) {
        const bool* input_ptr = cpu_input.data<bool>();
        bool* output_ptr = cpu_output.data<bool>();
        int64_t out_idx = 0;
        for (int64_t i = 0; i < numel; ++i) {
            if (is_mask_true(i)) {
                output_ptr[out_idx++] = input_ptr[i];
            }
        }
    } else {
        throw std::runtime_error("masked_select: unsupported dtype");
    }

    // Transfer result back to Vulkan device
    return cpu_output.to(input.device());
}

// Masked fill - fill elements where mask is true with value
auto VulkanBackend::dispatchMaskedFill(const Tensor& input, const Tensor& mask, float value) -> Tensor {
    // Create value tensor
    auto input_shape = input.shape();
    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
    auto value_tensor = dispatchFull(shape_vec, value, input.dtype());

    // Use where: where(mask, value, input)
    return dispatchWhere(mask, value_tensor, input);
}

// Where operation - select from x where condition is true, else from y
auto VulkanBackend::dispatchWhere(const Tensor& condition, const Tensor& x, const Tensor& y) -> Tensor {
    // Ensure all tensors are on the same device
    if (condition.device().type != Device::Type::Vulkan ||
        x.device().type != Device::Type::Vulkan ||
        y.device().type != Device::Type::Vulkan) {
        throw std::runtime_error("All tensors must be on Vulkan device for where operation");
    }

    auto cond_shape = condition.shape();
    auto x_shape = x.shape();
    auto y_shape = y.shape();

    // Validate shapes match
    if (cond_shape.size() != x_shape.size() || cond_shape.size() != y_shape.size()) {
        throw std::invalid_argument("where: all tensors must have same number of dimensions");
    }

    for (size_t i = 0; i < cond_shape.size(); ++i) {
        if (cond_shape[i] != x_shape[i] || cond_shape[i] != y_shape[i]) {
            throw std::invalid_argument("where: all tensors must have same shape");
        }
    }

    // Use element-wise operations to implement where on GPU
    // where(cond, x, y) = cond * x + (1 - cond) * y
    // This works if condition is 0 or 1

    // Convert condition to float if needed
    Tensor cond_float = condition.dtype() == DType::Float32 ? condition : condition.to(DType::Float32);

    // Compute: cond * x
    Tensor term1 = dispatchBinaryOp("mul", cond_float, x);

    // Compute: (1 - cond)
    std::vector<int64_t> cond_shape_vec(cond_shape.begin(), cond_shape.end());
    Tensor one_tensor = dispatchFull(cond_shape_vec, 1.0f, DType::Float32);
    Tensor inv_cond = dispatchBinaryOp("sub", one_tensor, cond_float);

    // Compute: (1 - cond) * y
    Tensor term2 = dispatchBinaryOp("mul", inv_cond, y);

    // Compute: term1 + term2
    return dispatchBinaryOp("add", term1, term2);
}
} // namespace tenzor
