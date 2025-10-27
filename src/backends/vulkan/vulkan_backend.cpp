/**
 * @file vulkan_backend.cpp
 * @brief Vulkan compute backend implementation
 */

#include "vulkan_backend.hpp"

// Undefine Vulkan Bool macro that conflicts with DType::Bool
#ifdef Bool
#undef Bool
#endif

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

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

    // Get shader path from environment or use default
    const char* shaderEnv = std::getenv("TENZOR_VULKAN_SHADER_PATH");
    if (shaderEnv) {
        shaderPath_ = shaderEnv;
    } else {
        // Default to build directory
        shaderPath_ = "./shaders/";
    }
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

    // Select devices with compute queue support
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
            devices_.push_back(std::move(ctx));
        }
    }

    if (devices_.empty()) {
        throw std::runtime_error("No Vulkan devices with compute support found");
    }
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

        // Device creation
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pQueueCreateInfos = &queueCreateInfo;
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = 0;
        createInfo.enabledLayerCount = 0;

        vulkan::checkVk(vkCreateDevice(ctx.physicalDevice, &createInfo,
                                      nullptr, &ctx.device),
                       "Failed to create logical device");

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
        ctx.descriptorPool = std::make_unique<vulkan::DescriptorPool>(ctx.device, 1000);

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

    // Return the buffer memory as opaque pointer
    // We need to track this mapping
    void* ptr = reinterpret_cast<void*>(buffer->buffer());

    // Store buffer in allocation tracking
    // Note: This is a simplified approach. Production code would need better tracking.
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
    // Buffer cleanup handled by VulkanBuffer destructor
    // In production, we'd maintain a map of VkBuffer -> VulkanBuffer
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
            VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
            VkBufferCopy copyRegion{};
            copyRegion.size = bytes;
            vkCmdCopyBuffer(cmdBuffer, staging.buffer->buffer(),
                          reinterpret_cast<VkBuffer>(dst), 1, &copyRegion);
            endSingleTimeCommands(cmdBuffer, device_id);
            break;
        }
        case CopyKind::DeviceToHost: {
            auto& staging = getStagingBuffer(device_id, bytes);

            // Copy from device to staging
            VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
            VkBufferCopy copyRegion{};
            copyRegion.size = bytes;
            vkCmdCopyBuffer(cmdBuffer, reinterpret_cast<VkBuffer>(const_cast<void*>(src)),
                          staging.buffer->buffer(), 1, &copyRegion);
            endSingleTimeCommands(cmdBuffer, device_id);

            void* mapped = staging.buffer->map();
            std::memcpy(dst, mapped, bytes);
            staging.buffer->unmap();
            break;
        }
        case CopyKind::DeviceToDevice: {
            VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
            VkBufferCopy copyRegion{};
            copyRegion.size = bytes;
            vkCmdCopyBuffer(cmdBuffer, reinterpret_cast<VkBuffer>(const_cast<void*>(src)),
                          reinterpret_cast<VkBuffer>(dst), 1, &copyRegion);
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

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(ctx.computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.computeQueue);

    vkFreeCommandBuffers(ctx.device, ctx.commandPool, 1, &commandBuffer);
}

auto VulkanBackend::synchronize(int32_t device_id) -> void {
    if (device_id < 0 || device_id >= device_count()) {
        throw std::invalid_argument("Invalid device ID");
    }
    vkDeviceWaitIdle(devices_[device_id].device);
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

    // Load and compile shader
    std::string shaderFile = shaderPath_ + shader_name + ".spv";
    auto shaderCode = vulkan::loadShader(shaderFile);

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

    // Create pipeline
    auto& ctx = devices_[device_id];
    auto pipeline = std::make_unique<vulkan::ComputePipeline>(
        ctx.device, shaderCode, bindings
    );

    auto* pipelinePtr = pipeline.get();
    cache.pipelines[shader_name] = std::move(pipeline);

    return pipelinePtr;
}

auto VulkanBackend::dispatch(const std::string& op_name,
                            std::span<const Tensor> inputs,
                            const OpAttributes& attrs) -> std::vector<Tensor> {

    // Binary operations
    if (op_name == "add" || op_name == "sub" || op_name == "mul" || op_name == "div") {
        if (inputs.size() != 2) {
            throw std::invalid_argument(op_name + " requires 2 inputs");
        }
        return {dispatchBinaryOp(op_name, inputs[0], inputs[1])};
    }

    // Unary operations
    if (op_name == "relu" || op_name == "sigmoid" || op_name == "tanh" ||
        op_name == "sqrt" || op_name == "exp" || op_name == "log" ||
        op_name == "neg" || op_name == "abs") {
        if (inputs.size() != 1) {
            throw std::invalid_argument(op_name + " requires 1 input");
        }
        return {dispatchUnaryOp(op_name, inputs[0])};
    }

    // Reduction operations
    if (op_name == "sum" || op_name == "mean" || op_name == "max" || op_name == "min") {
        if (inputs.size() != 1) {
            throw std::invalid_argument(op_name + " requires 1 input");
        }
        int64_t dim = -1;
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
        int64_t kernel_h = attrs.contains("kernel_h") ? std::stoll(attrs.at("kernel_h")) : 2;
        int64_t kernel_w = attrs.contains("kernel_w") ? std::stoll(attrs.at("kernel_w")) : kernel_h;
        int64_t stride_h = attrs.contains("stride_h") ? std::stoll(attrs.at("stride_h")) : kernel_h;
        int64_t stride_w = attrs.contains("stride_w") ? std::stoll(attrs.at("stride_w")) : kernel_w;
        int64_t padding_h = attrs.contains("padding_h") ? std::stoll(attrs.at("padding_h")) : 0;
        int64_t padding_w = attrs.contains("padding_w") ? std::stoll(attrs.at("padding_w")) : 0;
        auto [output, indices] = dispatchMaxPool2d(inputs[0], kernel_h, kernel_w,
                                                    stride_h, stride_w, padding_h, padding_w);
        return {output, indices};
    }

    if (op_name == "avg_pool2d") {
        int64_t kernel_h = attrs.contains("kernel_h") ? std::stoll(attrs.at("kernel_h")) : 2;
        int64_t kernel_w = attrs.contains("kernel_w") ? std::stoll(attrs.at("kernel_w")) : kernel_h;
        int64_t stride_h = attrs.contains("stride_h") ? std::stoll(attrs.at("stride_h")) : kernel_h;
        int64_t stride_w = attrs.contains("stride_w") ? std::stoll(attrs.at("stride_w")) : kernel_w;
        int64_t padding_h = attrs.contains("padding_h") ? std::stoll(attrs.at("padding_h")) : 0;
        int64_t padding_w = attrs.contains("padding_w") ? std::stoll(attrs.at("padding_w")) : 0;
        return {dispatchAvgPool2d(inputs[0], kernel_h, kernel_w,
                                  stride_h, stride_w, padding_h, padding_w)};
    }

    if (op_name == "adaptive_max_pool2d") {
        int64_t out_h = std::stoll(attrs.at("output_height"));
        int64_t out_w = std::stoll(attrs.at("output_width"));
        auto [output, indices] = dispatchAdaptiveMaxPool2d(inputs[0], out_h, out_w);
        return {output, indices};
    }

    if (op_name == "adaptive_avg_pool2d") {
        int64_t out_h = std::stoll(attrs.at("output_height"));
        int64_t out_w = std::stoll(attrs.at("output_width"));
        return {dispatchAdaptiveAvgPool2d(inputs[0], out_h, out_w)};
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

    throw std::runtime_error("VulkanBackend: Operation '" + op_name + "' not implemented");
}

auto VulkanBackend::dispatchBinaryOp(const std::string& op_name,
                                     const Tensor& a, const Tensor& b) -> Tensor {
    // Simplified implementation - production would handle broadcasting, etc.
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (!std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end())) {
        throw std::invalid_argument("Tensors must have same shape for binary op");
    }

    int32_t device_id = a.device().index;
    auto* pipeline = getPipeline(op_name, device_id);

    // Create output tensor (convert span to vector)
    std::vector<int64_t> output_shape(a_shape.begin(), a_shape.end());
    Tensor output(output_shape, a.dtype(), a.device());

    // Execute compute shader
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Bind descriptor sets and dispatch
    // (Simplified - production would properly bind buffers)
    uint32_t workgroups = (a.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchUnaryOp(const std::string& op_name,
                                    const Tensor& input) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline(op_name, device_id);

    // Create output tensor (convert span to vector)
    auto input_shape = input.shape();
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchReduction(const std::string& op_name,
                                      const Tensor& input,
                                      int64_t dim, bool keepdim) -> Tensor {
    // Simplified reduction implementation
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("reduction_" + op_name, device_id);

    // Calculate output shape
    std::vector<int64_t> out_shape;
    if (dim < 0) {
        out_shape = {1};
    } else {
        // Convert span to vector
        auto input_shape = input.shape();
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = 256; // Simplified
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchMatmul(const Tensor& a, const Tensor& b) -> Tensor {
    // Simplified matmul - production would use optimized tiled algorithms
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    if (a_shape.size() != 2 || b_shape.size() != 2) {
        throw std::invalid_argument("Matmul requires 2D tensors");
    }
    if (a_shape[1] != b_shape[0]) {
        throw std::invalid_argument("Incompatible dimensions for matmul");
    }

    int32_t device_id = a.device().index;
    auto* pipeline = getPipeline("matmul", device_id);

    std::vector<int64_t> out_shape = {a_shape[0], b_shape[1]};
    Tensor output(out_shape, a.dtype(), a.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Tile-based dispatch
    uint32_t workgroups_x = (out_shape[1] + 15) / 16;
    uint32_t workgroups_y = (out_shape[0] + 15) / 16;
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
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

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("adaptive_max_pool2d", device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_h, out_w};
    Tensor output(out_shape, input.dtype(), input.device());
    Tensor indices(out_shape, DType::Int64, input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups_x = (out_w + 15) / 16;
    uint32_t workgroups_y = (out_h + 15) / 16;
    uint32_t workgroups_z = channels;
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);

    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, indices};
}

auto VulkanBackend::dispatchAdaptiveAvgPool2d(const Tensor& input, int64_t out_h, int64_t out_w) -> Tensor {
    auto input_shape = input.shape();
    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("adaptive_avg_pool2d", device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_h, out_w};
    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups_x = (out_w + 15) / 16;
    uint32_t workgroups_y = (out_h + 15) / 16;
    uint32_t workgroups_z = channels;
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
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

auto VulkanBackend::dispatchLayerNorm(const Tensor& input, int64_t normalized_shape,
                                      const Tensor* gamma, const Tensor* beta, float epsilon) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("layer_norm", device_id);

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
    auto* pipeline = getPipeline("softmax", device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Calculate size before and after the dimension
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= input_shape[i];
    }
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < input_shape.size(); i++) {
        inner_size *= input_shape[i];
    }

    uint32_t workgroups = (outer_size * inner_size + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchLogSoftmax(const Tensor& input, int64_t dim) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("log_softmax", device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    // Calculate size before and after the dimension
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; i++) {
        outer_size *= input_shape[i];
    }
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < input_shape.size(); i++) {
        inner_size *= input_shape[i];
    }

    uint32_t workgroups = (outer_size * inner_size + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

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
    auto* pipeline = getPipeline("argmax", device_id);

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

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchArgmin(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("argmin", device_id);

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

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchVariance(const Tensor& input, int64_t dim, bool unbiased, bool keepdim) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("variance", device_id);

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

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchStd(const Tensor& input, int64_t dim, bool unbiased, bool keepdim) -> Tensor {
    // Standard deviation is just sqrt of variance
    Tensor variance = dispatchVariance(input, dim, unbiased, keepdim);
    return dispatchUnaryOp("sqrt", variance);
}

auto VulkanBackend::dispatchProd(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("prod", device_id);

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

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (output.numel() + 255) / 256;
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
    auto* pipeline = getPipeline("scatter", device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // First copy input to output
    size_t bytes = input.numel() * input.dtype_size();
    copy(output.data_ptr(), input.data_ptr(), bytes, CopyKind::DeviceToDevice);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (indices.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchIndexSelect(const Tensor& input, int64_t dim, const Tensor& indices) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("index_select", device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    out_shape[dim] = indices.numel();

    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// Shape operations implementation
auto VulkanBackend::dispatchReshape(const Tensor& input, const std::vector<int64_t>& new_shape) -> Tensor {
    // Reshape is typically a view operation, but for Vulkan we create a new tensor
    // and copy the data
    Tensor output(new_shape, input.dtype(), input.device());

    size_t bytes = input.numel() * input.dtype_size();
    copy(output.data_ptr(), input.data_ptr(), bytes, CopyKind::DeviceToDevice);

    return output;
}

auto VulkanBackend::dispatchTranspose(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor {
    auto input_shape = input.shape();
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    std::swap(out_shape[dim0], out_shape[dim1]);

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("transpose", device_id);

    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchPermute(const Tensor& input, const std::vector<int64_t>& dims) -> Tensor {
    auto input_shape = input.shape();
    std::vector<int64_t> out_shape;
    for (int64_t dim : dims) {
        out_shape.push_back(input_shape[dim]);
    }

    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("permute", device_id);

    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// Memory operations implementation
auto VulkanBackend::dispatchFill(const Tensor& input, float value) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("fill", device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchCopy(const Tensor& input) -> Tensor {
    auto input_shape = input.shape();
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    size_t bytes = input.numel() * input.dtype_size();
    copy(output.data_ptr(), input.data_ptr(), bytes, CopyKind::DeviceToDevice);

    return output;
}

// Factory function
extern "C" {
    auto create_backend() -> std::unique_ptr<Backend> {
        return std::make_unique<VulkanBackend>();
    }
}

} // namespace tenzor
