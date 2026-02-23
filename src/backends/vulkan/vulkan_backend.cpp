/**
 * @file vulkan_backend.cpp
 * @brief Vulkan compute backend implementation
 */

#include "vulkan_backend.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/backend/vulkan_caching_allocator.hpp"
#include "tenzor/backend/fast_dispatch.hpp"

// Undefine Vulkan Bool macro that conflicts with DType::Bool
#ifdef Bool
#undef Bool
#endif

#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

// Include embedded shaders
#ifdef __has_include
#  if __has_include("embedded_shaders.hpp")
#    include "embedded_shaders.hpp"
#    define TENZOR_HAS_EMBEDDED_SHADERS 1
#  endif
#endif

namespace tenzor {

// Helper function to insert transfer-to-compute barrier
// Required when a compute shader reads from a buffer that was just written by a transfer op
inline void insertTransferToComputeBarrier(VkCommandBuffer cmdBuffer) {
    if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
        VkMemoryBarrier memoryBarrier{};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmdBuffer,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
    }
    // When batching is disabled, each operation is submitted separately so no barrier needed
}

// Helper function to insert a pre-read barrier
// Required BEFORE a compute shader that reads from a buffer that may have pending writes
// from a previous compute operation in the same batch
inline void insertPreReadBarrier(VkCommandBuffer cmdBuffer) {
    if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
        VkMemoryBarrier memoryBarrier{};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmdBuffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
    }
}

// Helper function to insert compute shader memory barrier.
// Always inserts a compute-to-compute barrier (RAW hazard between consecutive
// compute dispatches). When batching is disabled, also inserts a transfer/host
// barrier for immediate readback.
inline void insertComputeBarrier(VkCommandBuffer cmdBuffer) {
    // Compute-to-compute barrier (needed in both modes for RAW hazard safety)
    VkMemoryBarrier computeBarrier{};
    computeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    computeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    computeBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmdBuffer,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 1, &computeBarrier, 0, nullptr, 0, nullptr);

    if constexpr (!vulkan_config::USE_COMMAND_BATCHING) {
        // Non-batching mode: also add transfer/host barrier for immediate readback
        VkMemoryBarrier transferBarrier{};
        transferBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        transferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        transferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmdBuffer,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                            0, 1, &transferBarrier, 0, nullptr, 0, nullptr);
    }
}

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

    // IMPORTANT: Shutdown the caching allocator BEFORE destroying the Vulkan device
    // The allocator is a singleton that may outlive this backend instance.
    // shutdown_device() releases free blocks, clears all blocks without Vulkan calls,
    // and marks the device as shutdown so future free() calls are safe.
    //
    // NOTE: During static destruction, the allocator singleton may already be destroyed
    // (due to LIFO destruction order). Check is_alive() first to avoid crash.
    if (backend::VulkanCachingAllocator::is_alive()) {
        for (size_t i = 0; i < devices_.size(); ++i) {
            if (devices_[i].device != VK_NULL_HANDLE) {
                backend::VulkanCachingAllocator::get().shutdown_device(static_cast<int>(i));
            }
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
        // Free command buffer pool before destroying command pool
        if (!ctx.commandBufferPool.empty() && ctx.commandPool != VK_NULL_HANDLE) {
            vkFreeCommandBuffers(ctx.device, ctx.commandPool,
                                static_cast<uint32_t>(ctx.commandBufferPool.size()),
                                ctx.commandBufferPool.data());
            ctx.commandBufferPool.clear();
        }
        // Destroy fences
        if (ctx.pendingFence != VK_NULL_HANDLE) {
            vkDestroyFence(ctx.device, ctx.pendingFence, nullptr);
        }
        // Destroy frame fences from ring buffer
        for (size_t i = 0; i < DeviceContext::MAX_FRAMES_IN_FLIGHT; ++i) {
            if (ctx.frameFences[i] != VK_NULL_HANDLE) {
                vkDestroyFence(ctx.device, ctx.frameFences[i], nullptr);
            }
        }
        if (ctx.commandPool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(ctx.device, ctx.commandPool, nullptr);
        }
        // Save and destroy pipeline cache
        if (ctx.pipelineCache != VK_NULL_HANDLE) {
            // Try to save cache to disk
            size_t cache_size = 0;
            if (vkGetPipelineCacheData(ctx.device, ctx.pipelineCache, &cache_size, nullptr) == VK_SUCCESS
                && cache_size > 0) {
                std::vector<char> cache_data(cache_size);
                if (vkGetPipelineCacheData(ctx.device, ctx.pipelineCache, &cache_size, cache_data.data()) == VK_SUCCESS) {
                    VkPhysicalDeviceProperties devProps;
                    vkGetPhysicalDeviceProperties(ctx.physicalDevice, &devProps);

                    std::string cache_dir;
                    if (auto* xdg = std::getenv("XDG_CACHE_HOME")) {
                        cache_dir = xdg;
                    } else if (auto* home = std::getenv("HOME")) {
                        cache_dir = std::string(home) + "/.cache";
                    }

                    if (!cache_dir.empty()) {
                        std::string dir = cache_dir + "/tenzor";
                        std::filesystem::create_directories(dir);
                        std::string path = dir + "/vulkan_pipeline_cache_"
                                         + std::to_string(devProps.deviceID) + ".bin";
                        std::ofstream out(path, std::ios::binary);
                        if (out.good()) {
                            out.write(cache_data.data(), static_cast<std::streamsize>(cache_size));
                        }
                    }
                }
            }
            vkDestroyPipelineCache(ctx.device, ctx.pipelineCache, nullptr);
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

    // Optional validation layers for debugging (set TENZOR_VULKAN_VALIDATION=1)
    std::vector<const char*> layers;
    std::vector<const char*> extensions;
    const char* validation_env = std::getenv("TENZOR_VULKAN_VALIDATION");
    bool enable_validation = validation_env && std::string(validation_env) == "1";

    if (enable_validation) {
        layers.push_back("VK_LAYER_KHRONOS_validation");
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        std::cerr << "Vulkan validation layers enabled (TENZOR_VULKAN_VALIDATION=1)\n";
    }

    createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    createInfo.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();

    vulkan::checkVk(vkCreateInstance(&createInfo, nullptr, &instance_),
                   "Failed to create Vulkan instance");

    // Register debug messenger if validation is enabled
    if (enable_validation) {
        auto vkCreateDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (vkCreateDebugUtilsMessengerEXT) {
            VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
            debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                    VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            debugInfo.pfnUserCallback = [](
                VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT types,
                const VkDebugUtilsMessengerCallbackDataEXT* data,
                [[maybe_unused]] void* user) -> VkBool32 {
                const char* level = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? "ERROR" : "WARNING";
                std::cerr << "[Vulkan " << level << "] " << data->pMessage << "\n";
                return VK_FALSE;
            };
            VkDebugUtilsMessengerEXT messenger{};
            vkCreateDebugUtilsMessengerEXT(instance_, &debugInfo, nullptr, &messenger);
        }
    }
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
    for (size_t device_idx = 0; device_idx < devices_.size(); ++device_idx) {
        auto& ctx = devices_[device_idx];
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
        bool hasAtomicFloat = false;
        bool hasAtomicInt64 = false;
        for (const auto& ext : availableExtensions) {
            if (strcmp(ext.extensionName, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME) == 0) {
                hasFloatControls = true;
            }
            if (strcmp(ext.extensionName, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) == 0) {
                hasAtomicFloat = true;
            }
            if (strcmp(ext.extensionName, "VK_KHR_shader_atomic_int64") == 0) {
                hasAtomicInt64 = true;
            }
        }

        // Extensions to enable
        std::vector<const char*> deviceExtensions;
        if (hasFloatControls) {
            deviceExtensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
        }
        if (hasAtomicFloat) {
            deviceExtensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
        }
        if (hasAtomicInt64) {
            deviceExtensions.push_back("VK_KHR_shader_atomic_int64");
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

        // Set up atomic float features if extension is available
        VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};
        atomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        atomicFloatFeatures.pNext = nullptr;

        // Query supported atomic float features
        if (hasAtomicFloat) {
            VkPhysicalDeviceFeatures2 features2{};
            features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features2.pNext = &atomicFloatFeatures;
            vkGetPhysicalDeviceFeatures2(ctx.physicalDevice, &features2);
        }

        // Set up atomic int64 features for Float64 CAS-loop atomics
        VkPhysicalDeviceShaderAtomicInt64Features atomicInt64Features{};
        atomicInt64Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES;
        atomicInt64Features.pNext = nullptr;

        if (hasAtomicInt64) {
            VkPhysicalDeviceFeatures2 features2{};
            features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features2.pNext = &atomicInt64Features;
            vkGetPhysicalDeviceFeatures2(ctx.physicalDevice, &features2);
            // Only consider it available if buffer atomics are supported
            hasAtomicInt64 = (atomicInt64Features.shaderBufferInt64Atomics == VK_TRUE);
        }

        // Device creation
        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = 1;
        createInfo.pQueueCreateInfos = &queueCreateInfo;
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.empty() ? nullptr : deviceExtensions.data();
        createInfo.enabledLayerCount = 0;

        // Chain features if any extensions need features enabled via pNext
        VkPhysicalDeviceFeatures2 features2Chain{};
        bool useFeatures2 = false;

        if (hasAtomicFloat && (atomicFloatFeatures.shaderBufferFloat32AtomicAdd ||
                              atomicFloatFeatures.shaderSharedFloat32AtomicAdd)) {
            atomicFloatFeatures.pNext = nullptr;
            useFeatures2 = true;
        }

        if (hasAtomicInt64) {
            // Chain atomic int64 features (prepend to pNext chain)
            atomicInt64Features.pNext = useFeatures2 ? static_cast<void*>(&atomicFloatFeatures) : nullptr;
            useFeatures2 = true;
        }

        if (useFeatures2) {
            features2Chain.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features2Chain.features = deviceFeatures;
            features2Chain.pNext = hasAtomicInt64 ? static_cast<void*>(&atomicInt64Features)
                                                  : static_cast<void*>(&atomicFloatFeatures);
            createInfo.pNext = &features2Chain;
            createInfo.pEnabledFeatures = nullptr;  // Must be null when using pNext chain
        }

        vulkan::checkVk(vkCreateDevice(ctx.physicalDevice, &createInfo,
                                      nullptr, &ctx.device),
                       "Failed to create logical device");

        // Store capability flags for later use
        ctx.canPreserveDenormsF32 = canPreserveDenormsF32;
        ctx.hasAtomicInt64 = hasAtomicInt64;

        // Create pipeline cache (try loading from disk)
        {
            VkPipelineCacheCreateInfo cacheCreateInfo{};
            cacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;

            // Try to load existing cache from disk
            VkPhysicalDeviceProperties devProps;
            vkGetPhysicalDeviceProperties(ctx.physicalDevice, &devProps);

            std::string cache_dir;
            if (auto* xdg = std::getenv("XDG_CACHE_HOME")) {
                cache_dir = xdg;
            } else if (auto* home = std::getenv("HOME")) {
                cache_dir = std::string(home) + "/.cache";
            }

            std::string cache_path;
            if (!cache_dir.empty()) {
                cache_path = cache_dir + "/tenzor/vulkan_pipeline_cache_"
                           + std::to_string(devProps.deviceID) + ".bin";
            }

            std::vector<char> cache_data;
            if (!cache_path.empty()) {
                std::ifstream cache_file(cache_path, std::ios::binary | std::ios::ate);
                if (cache_file.good()) {
                    auto size = cache_file.tellg();
                    if (size > 0) {
                        cache_data.resize(static_cast<size_t>(size));
                        cache_file.seekg(0);
                        cache_file.read(cache_data.data(), size);
                        cacheCreateInfo.initialDataSize = cache_data.size();
                        cacheCreateInfo.pInitialData = cache_data.data();
                    }
                }
            }

            vkCreatePipelineCache(ctx.device, &cacheCreateInfo, nullptr, &ctx.pipelineCache);
        }

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

        // Create fence for async synchronization
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = 0;  // Start unsignaled
        vulkan::checkVk(vkCreateFence(ctx.device, &fenceInfo, nullptr, &ctx.pendingFence),
                       "Failed to create synchronization fence");
        ctx.hasPendingWork = false;

        // Initialize command buffer pool
        initCommandBufferPool(ctx);

        // Initialize frame fences for batched execution
        initFrameFences(ctx);

        // Initialize VulkanCachingAllocator for this device
        backend::VulkanCachingAllocator::get().initialize(
            ctx.device, ctx.physicalDevice, static_cast<int>(device_idx));

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

auto VulkanBackend::get_device_info(int32_t device_id) const -> DeviceInfo {
    if (device_id < 0 || device_id >= device_count()) {
        throw std::out_of_range("Invalid device ID: " + std::to_string(device_id));
    }

    DeviceInfo info;
    const auto& ctx = devices_[device_id];

    // Get physical device properties
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(ctx.physicalDevice, &props);

    info.name = props.deviceName;

    // Determine vendor from vendorID
    switch (props.vendorID) {
        case 0x1002: info.vendor = "AMD"; break;
        case 0x10DE: info.vendor = "NVIDIA"; break;
        case 0x8086: info.vendor = "Intel"; break;
        case 0x13B5: info.vendor = "ARM"; break;
        case 0x5143: info.vendor = "Qualcomm"; break;
        default: info.vendor = "Unknown (0x" + std::to_string(props.vendorID) + ")"; break;
    }

    // Driver version - format varies by vendor
    uint32_t driverVersion = props.driverVersion;
    if (props.vendorID == 0x10DE) {
        // NVIDIA uses custom format
        info.driver_version = std::to_string((driverVersion >> 22) & 0x3ff) + "." +
                             std::to_string((driverVersion >> 14) & 0x0ff) + "." +
                             std::to_string((driverVersion >> 6) & 0x0ff) + "." +
                             std::to_string(driverVersion & 0x003f);
    } else {
        // Standard Vulkan version format
        info.driver_version = std::to_string(VK_VERSION_MAJOR(driverVersion)) + "." +
                             std::to_string(VK_VERSION_MINOR(driverVersion)) + "." +
                             std::to_string(VK_VERSION_PATCH(driverVersion));
    }

    // Memory info from memory properties
    const auto& memProps = ctx.memoryProperties;
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            info.total_memory = memProps.memoryHeaps[i].size;
            break;
        }
    }
    // Vulkan doesn't provide direct available memory query, set to total
    info.available_memory = info.total_memory;

    // Compute units - Vulkan doesn't expose SM count directly
    // Use maxComputeWorkGroupCount as a rough indicator
    info.compute_units = 0;  // Not directly available in Vulkan

    // Thread/workgroup limits
    info.max_threads_per_block = static_cast<int>(props.limits.maxComputeWorkGroupInvocations);
    info.max_shared_memory = static_cast<int>(props.limits.maxComputeSharedMemorySize);

    // Subgroup size (equivalent to warp size)
    VkPhysicalDeviceSubgroupProperties subgroupProps{};
    subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroupProps;
    vkGetPhysicalDeviceProperties2(ctx.physicalDevice, &props2);

    info.warp_size = static_cast<int>(subgroupProps.subgroupSize);

    // API version as compute capability equivalent
    info.major_version = VK_VERSION_MAJOR(props.apiVersion);
    info.minor_version = VK_VERSION_MINOR(props.apiVersion);

    // Feature support - check physical device features
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(ctx.physicalDevice, &features);

    info.supports_fp64 = features.shaderFloat64;
    info.supports_int8 = false;  // Would need VK_KHR_shader_integer_dot_product extension check

    // Check for FP16 support via VK_KHR_shader_float16_int8
    VkPhysicalDeviceFloat16Int8FeaturesKHR f16i8Features{};
    f16i8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &f16i8Features;
    vkGetPhysicalDeviceFeatures2(ctx.physicalDevice, &features2);

    info.supports_fp16 = f16i8Features.shaderFloat16;

    // Device type
    info.is_integrated = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
    info.is_discrete = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);

    // PCI info - requires VK_KHR_driver_properties extension
    // Try to get it if available
    VkPhysicalDevicePCIBusInfoPropertiesEXT pciInfo{};
    pciInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT;

    VkPhysicalDeviceProperties2 propsWithPci{};
    propsWithPci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    propsWithPci.pNext = &pciInfo;
    vkGetPhysicalDeviceProperties2(ctx.physicalDevice, &propsWithPci);

    // Only set if the extension filled in valid values
    if (pciInfo.pciBus != 0 || pciInfo.pciDevice != 0) {
        info.pci_bus_id = static_cast<int>(pciInfo.pciBus);
        info.pci_device_id = static_cast<int>(pciInfo.pciDevice);
    }

    return info;
}

auto VulkanBackend::allocate(size_t bytes, int32_t device_id) -> void* {
    std::lock_guard<std::recursive_mutex> lock(dispatch_mutex_);
    // Allocate minimum 4 bytes for all requests. This ensures:
    // 1. Empty tensors always have valid, tracked Vulkan buffers (no null data_ptr crashes)
    // 2. Float16 tensors with odd element counts have enough space for uint32 shader access
    //    (Float16 shaders pack 2 elements per uint32, so a single-element Float16 tensor
    //    needs 4 bytes, not just 2, to avoid out-of-bounds shader writes)
    size_t alloc_bytes = std::max(bytes, static_cast<size_t>(4));

    if (device_id < 0 || device_id >= device_count()) {
        throw std::invalid_argument("Invalid device ID");
    }

    void* ptr = allocateDeviceMemory(alloc_bytes, device_id);
    allocations_[ptr] = {alloc_bytes, device_id};
    return ptr;
}

void* VulkanBackend::allocateDeviceMemory(size_t bytes, int32_t device_id) {
    // Use caching allocator for efficient memory reuse
    auto& allocator = backend::VulkanCachingAllocator::get();

    // Allocate device-local memory for compute buffers.
    // Data transfers use staging buffers (HOST_VISIBLE), so compute tensors
    // only need DEVICE_LOCAL. This avoids per-allocation mapping overhead and
    // gives the driver maximum flexibility for memory placement.
    void* ptr = allocator.allocate(
        bytes, device_id,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    return ptr;
}

auto VulkanBackend::deallocate(void* ptr) -> void {
    std::lock_guard<std::recursive_mutex> lock(dispatch_mutex_);
    if (ptr == nullptr) {
        return;
    }

    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        auto [bytes, device_id] = it->second;
        // CRITICAL: When batching is enabled, the buffer may be referenced by
        // a command buffer that hasn't been submitted yet. We must force-submit
        // the batch and wait for it to complete before freeing the buffer.
        if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
            submitBatchIfNeeded(device_id, true);  // Force submit any pending batch
        }
        // Ensure any pending async GPU work completes before freeing memory
        ensurePendingWorkComplete(device_id);
        freeDeviceMemory(ptr, device_id);
        allocations_.erase(it);
    }
}

void VulkanBackend::freeDeviceMemory(void* ptr, int32_t device_id) {
    // Return memory to caching allocator for reuse
    backend::VulkanCachingAllocator::get().free(ptr, device_id);
}

auto VulkanBackend::copy(void* dst, const void* src, size_t bytes,
                        CopyKind kind) -> void {
    std::lock_guard<std::recursive_mutex> lock(dispatch_mutex_);
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
            // Insert barrier for subsequent compute ops that may read this buffer
            insertTransferToComputeBarrier(cmdBuffer);
            endSingleTimeCommands(cmdBuffer, device_id);

            // CRITICAL: With batching enabled, force submit now to ensure staging buffer
            // content is copied to device before staging buffer can be reused.
            // Without this, a subsequent HostToDevice copy could overwrite staging buffer
            // before our copy command is actually submitted.
            if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
                submitBatchIfNeeded(device_id, true);  // Force submit
                ensurePendingWorkComplete(device_id);   // Wait for copy to complete
            }
            break;
        }
        case CopyKind::DeviceToHost: {
            // Flush any batched commands before reading back data
            if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
                submitBatchIfNeeded(device_id, true);  // Force submit
            }
            // Ensure any pending GPU compute work is complete before copying
            ensurePendingWorkComplete(device_id);

            auto& staging = getStagingBuffer(device_id, bytes);

            // Copy from device to staging - MUST use immediate execution, not batching
            // because we need the data available right after this call
            auto [src_buffer, src_offset] = getVulkanBufferAndOffset(src);
            VkCommandBuffer cmdBuffer = acquireCommandBuffer(device_id);  // Bypass batching
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = src_offset;
            copyRegion.dstOffset = 0;  // Staging buffer starts at 0
            copyRegion.size = bytes;
            vkCmdCopyBuffer(cmdBuffer, src_buffer,
                          staging.buffer->buffer(), 1, &copyRegion);
            endSingleTimeCommandsAsync(cmdBuffer, device_id);  // Submit immediately

            // Ensure copy is complete before reading from staging buffer
            ensurePendingWorkComplete(device_id);

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
            // Insert barrier for subsequent compute ops that may read this buffer
            insertTransferToComputeBarrier(cmdBuffer);
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
    if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
        // Use batched command buffer for reduced submission overhead
        return getOrCreateBatchCommandBuffer(device_id);
    } else {
        // Legacy path: individual command buffer per operation
        return acquireCommandBuffer(device_id);
    }
}

void VulkanBackend::endSingleTimeCommands(VkCommandBuffer commandBuffer, int32_t device_id) {
    if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
        // Record operation to batch - don't submit yet
        recordOperationToBatch(device_id);
    } else {
        // Legacy path: submit immediately with fence tracking
        endSingleTimeCommandsAsync(commandBuffer, device_id);
    }
}

void VulkanBackend::initCommandBufferPool(DeviceContext& ctx) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = ctx.commandPool;
    allocInfo.commandBufferCount = static_cast<uint32_t>(DeviceContext::COMMAND_BUFFER_POOL_SIZE);

    ctx.commandBufferPool.resize(DeviceContext::COMMAND_BUFFER_POOL_SIZE);
    vulkan::checkVk(vkAllocateCommandBuffers(ctx.device, &allocInfo, ctx.commandBufferPool.data()),
                   "Failed to allocate command buffer pool");
    ctx.nextCommandBufferIndex = 0;
}

VkCommandBuffer VulkanBackend::acquireCommandBuffer(int32_t device_id) {
    auto& ctx = devices_[device_id];

    // If we've used all buffers in the pool, wait for pending work and reset
    if (ctx.nextCommandBufferIndex >= ctx.commandBufferPool.size()) {
        ensurePendingWorkComplete(device_id);
        vkResetCommandPool(ctx.device, ctx.commandPool, 0);
        ctx.nextCommandBufferIndex = 0;

        // Also reset descriptor pool since all command buffers are complete
        // and no descriptor sets are in use
        if (ctx.descriptorPool) {
            ctx.descriptorPool->reset();
        }
    }

    VkCommandBuffer cmdBuffer = ctx.commandBufferPool[ctx.nextCommandBufferIndex++];

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmdBuffer, &beginInfo);
    return cmdBuffer;
}

void VulkanBackend::releaseCommandBuffer(VkCommandBuffer cmdBuffer, int32_t device_id) {
    // Command buffers are pooled and reset together, no individual release needed
    (void)cmdBuffer;
    (void)device_id;
}

void VulkanBackend::ensurePendingWorkComplete(int32_t device_id) {
    auto& ctx = devices_[device_id];

    // Wait on all frame fences that have been submitted
    if (ctx.submittedFrames > 0) {
        // Collect all fences that might have pending work
        std::vector<VkFence> fencesToWait;
        for (size_t i = 0; i < DeviceContext::MAX_FRAMES_IN_FLIGHT; ++i) {
            if (ctx.frameFences[i] != VK_NULL_HANDLE) {
                // Check if fence is actually signaled (has pending work)
                VkResult status = vkGetFenceStatus(ctx.device, ctx.frameFences[i]);
                if (status == VK_NOT_READY) {
                    fencesToWait.push_back(ctx.frameFences[i]);
                } else if (status == VK_ERROR_DEVICE_LOST) {
                    throw std::runtime_error("Device lost before fence wait (fence status check)");
                }
            }
        }

        if (!fencesToWait.empty()) {
            // Use 30 second timeout to detect GPU hangs (often caused by memory pressure)
            constexpr uint64_t FENCE_TIMEOUT_NS = 30'000'000'000ULL;  // 30 seconds
            VkResult result = vkWaitForFences(ctx.device,
                                              static_cast<uint32_t>(fencesToWait.size()),
                                              fencesToWait.data(), VK_TRUE, FENCE_TIMEOUT_NS);
            if (result == VK_TIMEOUT) {
                throw std::runtime_error("GPU fence wait timed out after 30 seconds. "
                    "This often indicates memory pressure or a shader hang. "
                    "Try reducing batch size or model size, or use a smaller dtype.");
            }
            if (result != VK_SUCCESS) {
                std::string error_msg = "Failed to wait for fences: " + std::to_string(result);
                if (result == VK_ERROR_DEVICE_LOST) {
                    error_msg += " (VK_ERROR_DEVICE_LOST - GPU crash or timeout)";
                }
                throw std::runtime_error(error_msg);
            }
        }

        // Reset state - all work is now complete
        ctx.submittedFrames = 0;
        ctx.currentFrame = 0;
    }

    ctx.hasPendingWork = false;
}

void VulkanBackend::endSingleTimeCommandsAsync(VkCommandBuffer commandBuffer, int32_t device_id) {
    auto& ctx = devices_[device_id];

    VkResult result = vkEndCommandBuffer(commandBuffer);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to end command buffer: " + std::to_string(result));
    }

    // Use ring buffer of fences for true async execution
    // Only wait if all frames in flight are occupied
    size_t targetFrame = ctx.currentFrame;

    // Check if we need to wait for the target frame's fence
    // We only need to wait if we've submitted MAX_FRAMES_IN_FLIGHT work already
    if (ctx.submittedFrames >= DeviceContext::MAX_FRAMES_IN_FLIGHT) {
        // Wait for the oldest frame to complete before reusing its fence
        // Use 30 second timeout to detect GPU hangs
        constexpr uint64_t FENCE_TIMEOUT_NS = 30'000'000'000ULL;  // 30 seconds
        VkFence fenceToWait = ctx.frameFences[targetFrame];
        VkResult waitResult = vkWaitForFences(ctx.device, 1, &fenceToWait, VK_TRUE, FENCE_TIMEOUT_NS);
        if (waitResult == VK_TIMEOUT) {
            throw std::runtime_error("GPU fence wait timed out after 30 seconds. "
                "This often indicates memory pressure or a shader hang.");
        }
        if (waitResult != VK_SUCCESS) {
            throw std::runtime_error("Failed to wait for frame fence: " + std::to_string(waitResult));
        }
    }

    // Get the fence for this submission and reset it
    VkFence fence = ctx.frameFences[targetFrame];
    vkResetFences(ctx.device, 1, &fence);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    result = vkQueueSubmit(ctx.computeQueue, 1, &submitInfo, fence);
    if (result != VK_SUCCESS) {
        std::string error_msg = "Failed to submit queue with fence: " + std::to_string(result);
        if (result == VK_ERROR_DEVICE_LOST) {
            error_msg += " (VK_ERROR_DEVICE_LOST)";
        }
        throw std::runtime_error(error_msg);
    }

    // Move to next frame slot (ring buffer)
    ctx.currentFrame = (ctx.currentFrame + 1) % DeviceContext::MAX_FRAMES_IN_FLIGHT;
    ctx.submittedFrames = std::min(ctx.submittedFrames + 1, DeviceContext::MAX_FRAMES_IN_FLIGHT);

    // Also mark pendingFence for legacy code paths
    ctx.hasPendingWork = true;
}

auto VulkanBackend::synchronize(int32_t device_id) -> void {
    std::lock_guard<std::recursive_mutex> lock(dispatch_mutex_);
    if (device_id < 0 || device_id >= device_count()) {
        throw std::invalid_argument("Invalid device ID");
    }
    auto& ctx = devices_[device_id];

    // Submit any pending batched commands before synchronizing
    if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
        submitBatchIfNeeded(device_id, true);  // Force submit any pending work
    }

    // First ensure any fence-tracked work is complete (legacy pendingFence)
    ensurePendingWorkComplete(device_id);

    // Wait for all frame fences in the ring buffer
    if (ctx.submittedFrames > 0) {
        std::vector<VkFence> fencesToWait;
        for (size_t i = 0; i < DeviceContext::MAX_FRAMES_IN_FLIGHT; ++i) {
            if (ctx.frameFences[i] != VK_NULL_HANDLE) {
                fencesToWait.push_back(ctx.frameFences[i]);
            }
        }
        if (!fencesToWait.empty()) {
            constexpr uint64_t FENCE_TIMEOUT_NS = 30'000'000'000ULL;  // 30 seconds
            VkResult result = vkWaitForFences(ctx.device, static_cast<uint32_t>(fencesToWait.size()),
                           fencesToWait.data(), VK_TRUE, FENCE_TIMEOUT_NS);
            if (result == VK_TIMEOUT) {
                throw std::runtime_error("GPU sync fence wait timed out after 30 seconds. "
                    "This often indicates memory pressure or a shader hang.");
            }
        }
        ctx.submittedFrames = 0;
        ctx.currentFrame = 0;
    }

    // Then wait for all device operations (belt and suspenders)
    vkDeviceWaitIdle(ctx.device);

    // Reset command pool and pool index
    vkResetCommandPool(ctx.device, ctx.commandPool, 0);
    ctx.nextCommandBufferIndex = 0;

    // Reset batching state
    ctx.activeCommandBuffer = VK_NULL_HANDLE;
    ctx.operationsInBatch = 0;

    // Reset descriptor pool to reclaim descriptor sets
    // This is safe because all GPU work is complete after vkDeviceWaitIdle
    if (ctx.descriptorPool) {
        ctx.descriptorPool->reset();
    }
}

// ============================================================================
// Batched Command Execution for Performance
// ============================================================================

void VulkanBackend::initFrameFences(DeviceContext& ctx) {
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // Start signaled so first wait doesn't block

    for (size_t i = 0; i < DeviceContext::MAX_FRAMES_IN_FLIGHT; ++i) {
        vulkan::checkVk(vkCreateFence(ctx.device, &fenceInfo, nullptr, &ctx.frameFences[i]),
                       "Failed to create frame fence");
    }

    // Pre-allocate command buffers for each frame
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = ctx.commandPool;
    allocInfo.commandBufferCount = static_cast<uint32_t>(DeviceContext::MAX_FRAMES_IN_FLIGHT);

    vulkan::checkVk(vkAllocateCommandBuffers(ctx.device, &allocInfo, ctx.frameCommandBuffers.data()),
                   "Failed to allocate frame command buffers");
}

VkCommandBuffer VulkanBackend::getOrCreateBatchCommandBuffer(int32_t device_id) {
    auto& ctx = devices_[device_id];

    // If no active batch, start one
    if (ctx.activeCommandBuffer == VK_NULL_HANDLE) {
        // Wait for this frame's fence if it has pending work
        if (ctx.submittedFrames > 0) {
            waitForFrame(device_id, ctx.currentFrame);
        }

        ctx.activeCommandBuffer = ctx.frameCommandBuffers[ctx.currentFrame];

        // Reset and begin the command buffer
        vkResetCommandBuffer(ctx.activeCommandBuffer, 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(ctx.activeCommandBuffer, &beginInfo);

        ctx.operationsInBatch = 0;
    }

    return ctx.activeCommandBuffer;
}

void VulkanBackend::recordOperationToBatch(int32_t device_id) {
    auto& ctx = devices_[device_id];
    ctx.operationsInBatch++;

    // Auto-submit if batch is full (use config threshold)
    if (ctx.operationsInBatch >= vulkan_config::BATCH_SIZE_THRESHOLD) {
        submitBatchIfNeeded(device_id, true);
    }
}

void VulkanBackend::submitBatchIfNeeded(int32_t device_id, bool force) {
    auto& ctx = devices_[device_id];

    if (ctx.activeCommandBuffer == VK_NULL_HANDLE || ctx.operationsInBatch == 0) {
        return;  // Nothing to submit
    }

    if (!force && ctx.operationsInBatch < DeviceContext::MAX_OPERATIONS_PER_BATCH / 2) {
        return;  // Let more operations accumulate
    }

    // End and submit the command buffer
    vkEndCommandBuffer(ctx.activeCommandBuffer);

    VkFence fence = ctx.frameFences[ctx.currentFrame];

    // Reset fence before use
    vkResetFences(ctx.device, 1, &fence);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &ctx.activeCommandBuffer;

    VkResult result = vkQueueSubmit(ctx.computeQueue, 1, &submitInfo, fence);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit batch command buffer: " + std::to_string(result));
    }

    // Move to next frame slot
    ctx.activeCommandBuffer = VK_NULL_HANDLE;
    ctx.operationsInBatch = 0;
    ctx.currentFrame = (ctx.currentFrame + 1) % DeviceContext::MAX_FRAMES_IN_FLIGHT;
    ctx.submittedFrames = std::min(ctx.submittedFrames + 1, DeviceContext::MAX_FRAMES_IN_FLIGHT);
}

void VulkanBackend::waitForFrame(int32_t device_id, size_t frameIndex) {
    auto& ctx = devices_[device_id];
    VkFence fence = ctx.frameFences[frameIndex];

    constexpr uint64_t FENCE_TIMEOUT_NS = 30'000'000'000ULL;  // 30 seconds
    VkResult result = vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, FENCE_TIMEOUT_NS);
    if (result == VK_TIMEOUT) {
        throw std::runtime_error("GPU frame fence wait timed out after 30 seconds. "
            "This often indicates memory pressure or a shader hang.");
    }
    if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to wait for frame fence: " + std::to_string(result));
    }
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
        shader_name == "comparison_f64" || shader_name == "comparison_i32" || shader_name == "comparison_i64" ||
        shader_name == "comparison_f16" ||
        shader_name == "trigonometric" || shader_name == "trigonometric_f16" || shader_name == "trigonometric_f64" ||
        shader_name == "hyperbolic" || shader_name == "hyperbolic_f16" || shader_name == "hyperbolic_f64") {
        // math/comparison/trig/hyperbolic (all variants): 8 bytes (uint n, uint op)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 8;  // 2 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "math_f64") {
        // math_f64: 16 bytes (uint n, uint op, double param)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 16;  // uint + uint + double (aligned)
        pushConstants.push_back(push_range);
    } else if (shader_name == "fill" || shader_name == "full" || shader_name == "full_f16" || shader_name == "full_i8") {
        // fill/full/full_f16/full_i8: 8 bytes (uint n, uint/float value bits)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 8;  // uint32_t + uint32_t (value bits)
        pushConstants.push_back(push_range);
    } else if (shader_name == "full_f64") {
        // full_f64: 16 bytes (uint n_elements, [pad], double fill_value)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 16;  // uint32_t + padding + double
        pushConstants.push_back(push_range);
    } else if (shader_name == "ones" || shader_name == "ones_f16" || shader_name == "ones_i8") {
        // ones/ones_f16/ones_i8: 8 bytes (n_elements, one_value_bits)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 8;  // 2 uint32_t values
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
    } else if (shader_name == "math_i8" || shader_name == "math_uint8" || shader_name == "math_i64" ||
               shader_name == "math_f16") {
        // math_i8/math_uint8/math_i64/math_f16: 12 bytes (n, op, param)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;  // 3 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "activations" || shader_name == "activations_f16" ||
               shader_name == "activations_backward" || shader_name == "activations_backward_f16") {
        // activations (Float32/Float16): 12 bytes (n, activation, alpha)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;  // 3 uint32_t values (float fits in 4 bytes)
        pushConstants.push_back(push_range);
    } else if (shader_name == "activations_f64" || shader_name == "activations_backward_f64") {
        // activations (Float64): 16 bytes (n, activation, double alpha)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 16;  // uint + uint + double (aligned)
        pushConstants.push_back(push_range);
    } else if (shader_name == "batchnorm_update_stats" || shader_name == "batchnorm_update_stats_f64" ||
               shader_name == "batchnorm_update_stats_f16") {
        // batchnorm_update_stats: 8 bytes (n_channels, momentum)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 8;  // uint32_t + float
        pushConstants.push_back(push_range);
    } else if (shader_name == "batchnorm2d_forward" || shader_name == "batchnorm2d_backward" ||
               shader_name == "batchnorm2d_forward_f64" || shader_name == "batchnorm2d_backward_f64" ||
               shader_name == "batchnorm2d_forward_f16" || shader_name == "batchnorm2d_backward_f16") {
        // batchnorm2d_forward/backward: 24 bytes (n_elements, batch, channels, spatial_size, eps, has_affine/has_gamma)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 24;  // 6 uint32_t/float values
        pushConstants.push_back(push_range);
    } else if (shader_name == "batchnorm2d_mean_var" || shader_name == "batchnorm2d_mean_var_f64" ||
               shader_name == "batchnorm2d_mean_var_f16") {
        // batchnorm2d_mean_var: 20 bytes (n_elements, batch, channels, spatial_size, pass_id)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 20;  // 5 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "random" || shader_name == "random_f64" || shader_name == "random_f16") {
        // random: 16 bytes (n_elements, seed, offset, distribution)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 16;  // 4 uint32_t values
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
    } else if (shader_name == "reduction" || shader_name == "reduction_f64" ||
               shader_name == "reduction_f16" || shader_name == "reduction_i32") {
        // reduction (all variants): 20 bytes (n, reduce_size, outer_size, inner_size, op)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 20;  // 5 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "strided_copy" || shader_name == "strided_copy_f64" ||
               shader_name == "strided_copy_f16" || shader_name == "strided_copy_i8") {
        // strided_copy (all variants): 80 bytes (n_elements, ndims, base_offset, shape_0_3, shape_4_7, strides_0_3, strides_4_7)
        // 3 uint32_t + 4 uvec4/ivec4 = 12 + 64 = 76 bytes, aligned to 80
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 80;  // 3 uint32_t + 4 vec4 (need alignment)
        pushConstants.push_back(push_range);
    } else if (shader_name == "permute" || shader_name == "permute_f64" || shader_name == "permute_f16") {
        // permute (all variants): 8 bytes (n, ndim)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 8;  // 2 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "softmax" || shader_name == "softmax_f16" || shader_name == "softmax_f64") {
        // softmax: 16 bytes (batch_size, num_classes, dim, mode)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 16;  // 4 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "softmax_backward" || shader_name == "softmax_backward_f16" || shader_name == "softmax_backward_f64") {
        // softmax_backward: 12 bytes (outer_size, dim_size, inner_size)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;  // 3 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "log_softmax" || shader_name == "log_softmax_f64") {
        // log_softmax: 12 bytes (batch_size, num_classes, dim)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;  // 3 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "log_softmax_backward" || shader_name == "log_softmax_backward_f64") {
        // log_softmax_backward: 12 bytes (outer_size, dim_size, inner_size)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;  // 3 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "matmul" || shader_name == "matmul_f16" || shader_name == "matmul_f64" ||
               shader_name == "matmul_f64_optimized" || shader_name == "matmul_i32") {
        // matmul: 12 bytes (M, N, K)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;  // 3 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "bmm" || shader_name == "bmm_f64") {
        // bmm: 16 bytes (batch, M, N, K)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 16;  // 4 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "bmm_strided" || shader_name == "bmm_strided_f64") {
        // bmm_strided: 40 bytes (batch, M, N, K, a_stride0..2, b_stride0..2)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 40;  // 10 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "layer_norm" || shader_name == "layer_norm_f16" || shader_name == "layer_norm_f64") {
        // layer_norm: 16 bytes (batch_size, normalized_shape, epsilon, affine)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 16;  // 2 uint + 1 float + 1 uint = 16 bytes
        pushConstants.push_back(push_range);
    } else if (shader_name == "avg_pool2d" || shader_name == "avg_pool2d_f64" || shader_name == "avg_pool2d_f16" ||
               shader_name == "avg_pool2d_backward" || shader_name == "avg_pool2d_backward_f64" || shader_name == "avg_pool2d_backward_f16") {
        // avg_pool2d (all variants): 56 bytes (14 uint32_t fields)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 56;  // 14 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "max_pool2d" || shader_name == "max_pool2d_f64" || shader_name == "max_pool2d_f16") {
        // max_pool2d (all variants): 60 bytes (15 uint32_t fields including dilation)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 60;  // 15 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "bilinear_interpolate" || shader_name == "bilinear_interpolate_f64" ||
               shader_name == "bilinear_interpolate_f16" ||
               shader_name == "nearest_interpolate" || shader_name == "nearest_interpolate_f64" ||
               shader_name == "nearest_interpolate_f16") {
        // interpolation: 32 bytes (n_elements, batch, channels, in_height, in_width, out_height, out_width, align_corners)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 32;  // 8 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "roi_align" || shader_name == "roi_align_f64" || shader_name == "roi_align_f16") {
        // roi_align forward: 40 bytes (10 uint32_t values)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 40;  // 10 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "roi_align_backward" || shader_name == "roi_align_backward_f64" || shader_name == "roi_align_backward_f16") {
        // roi_align backward: 44 bytes (11 uint32_t values, includes batch_size)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 44;  // 11 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "argmax_argmin" || shader_name == "argmax_argmin_i32" ||
               shader_name == "argmax_argmin_f16" || shader_name == "argmax_argmin_f64") {
        // argmax_argmin (all variants): 20 bytes (n, reduce_size, outer_size, inner_size, op)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 20;  // 5 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "index_select" || shader_name == "index_select_f16" ||
               shader_name == "index_select_f64" || shader_name == "index_select_i64") {
        // index_select (all variants): 20 bytes (num_indices, dim, dim_size, inner_size, outer_size)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 20;  // 5 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "conv_transpose2d_forward" || shader_name == "conv_transpose2d_forward_f64" ||
               shader_name == "conv_transpose2d_forward_f16") {
        // conv_transpose2d_forward (all variants): 64 bytes (16 uint32_t values)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 64;  // 16 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "bitonic_sort") {
        // bitonic_sort: 20 bytes (n, padded_n, stage, substage, descending)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 20;  // 5 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "masked_select_count" || shader_name == "prefix_sum") {
        // masked_select_count/prefix_sum: 12 bytes (n_elements, mask_is_float, pass)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;  // 3 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "masked_select_gather") {
        // masked_select_gather: 12 bytes (n_elements, mask_is_float, output_size)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;  // 3 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "arange") {
        // arange: 12 bytes (float start, float step, uint num_elements)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;  // 2 float + 1 uint32_t
        pushConstants.push_back(push_range);
    } else if (shader_name == "arange_f64") {
        // arange_f64: 24 bytes (uint num_elements, uint _pad, double start, double step)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 24;  // 2 uint32_t + 2 double
        pushConstants.push_back(push_range);
    } else if (shader_name == "linspace") {
        // linspace: 12 bytes (float start, float end_val, uint num_steps)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;  // 2 float + 1 uint32_t
        pushConstants.push_back(push_range);
    } else if (shader_name == "linspace_f64") {
        // linspace_f64: 24 bytes (uint num_steps, uint _pad, double start, double end_val)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 24;  // 2 uint32_t + 2 double
        pushConstants.push_back(push_range);
    } else if (shader_name == "eye" || shader_name == "eye_f64") {
        // eye: 8 bytes (uint rows, uint cols)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 8;  // 2 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "variance_std" || shader_name == "variance_std_f64") {
        // variance_std: 20 bytes (n, reduce_size, outer_size, op, unbiased)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 20;  // 5 uint32_t values
        pushConstants.push_back(push_range);
    } else if (shader_name == "ones_f64") {
        // ones_f64: 4 bytes (n_elements)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 4;
        pushConstants.push_back(push_range);
    } else if (shader_name == "swish_backward" || shader_name == "swish_backward_f64") {
        // swish_backward: 4 bytes (n_elements)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 4;
        pushConstants.push_back(push_range);
    } else if (shader_name == "cast_int64_to_int32") {
        // cast_int64_to_int32: 4 bytes (n_elements)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 4;
        pushConstants.push_back(push_range);
    } else if (shader_name == "nonzero_count") {
        // nonzero_count: 8 bytes (n_elements, dtype)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 8;
        pushConstants.push_back(push_range);
    } else if (shader_name == "one_hot") {
        // one_hot: 8 bytes (n_elements, num_classes)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 8;
        pushConstants.push_back(push_range);
    } else if (shader_name == "max_pool2d_backward_indices" ||
               shader_name == "max_pool2d_backward_indices_f16" ||
               shader_name == "max_pool2d_backward_indices_f64") {
        // max_pool2d_backward_indices: 8 bytes (n_elements, channels)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 8;
        pushConstants.push_back(push_range);
    } else if (shader_name == "clamp" || shader_name == "clamp_f16") {
        // clamp/clamp_f16: 12 bytes (n, min_val, max_val)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;
        pushConstants.push_back(push_range);
    } else if (shader_name == "cross_entropy") {
        // cross_entropy: 12 bytes (batch_size, num_classes, ignore_index)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;
        pushConstants.push_back(push_range);
    } else if (shader_name == "embedding_backward" ||
               shader_name == "embedding_backward_f16" ||
               shader_name == "embedding_backward_f64") {
        // embedding_backward: 12 bytes (num_indices, embedding_dim, num_embeddings)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;
        pushConstants.push_back(push_range);
    } else if (shader_name == "indexing") {
        // indexing: 12 bytes
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;
        pushConstants.push_back(push_range);
    } else if (shader_name == "masked_select_gather_f16" ||
               shader_name == "masked_select_gather_f64") {
        // masked_select_gather f16/f64: 12 bytes (n_elements, mask_is_float, output_size)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;
        pushConstants.push_back(push_range);
    } else if (shader_name == "math_i32") {
        // math_i32: 12 bytes (n, op, param)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;
        pushConstants.push_back(push_range);
    } else if (shader_name == "matmul_bt" || shader_name == "matmul_f64_bt") {
        // matmul_bt: 12 bytes (M, N, K)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;
        pushConstants.push_back(push_range);
    } else if (shader_name == "nonzero_gather") {
        // nonzero_gather: 12 bytes
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;
        pushConstants.push_back(push_range);
    } else if (shader_name == "strided_copy_u8") {
        // strided_copy_u8: 12 bytes
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;
        pushConstants.push_back(push_range);
    } else if (shader_name == "transpose") {
        // transpose: 12 bytes (rows, cols, n_elements)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 12;
        pushConstants.push_back(push_range);
    } else if (shader_name == "boolean_reduction") {
        // boolean_reduction: 16 bytes
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 16;
        pushConstants.push_back(push_range);
    } else if (shader_name == "box_iou") {
        // box_iou: 16 bytes (num_boxes1, num_boxes2, total_pairs, mode)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 16;
        pushConstants.push_back(push_range);
    } else if (shader_name == "conv2d_backward_bias" || shader_name == "conv2d_backward_bias_f64") {
        // conv2d_backward_bias: 16 bytes (batch, out_channels, spatial_size, n_elements)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 16;
        pushConstants.push_back(push_range);
    } else if (shader_name == "embedding" || shader_name == "embedding_f16" || shader_name == "embedding_f64") {
        // embedding: 16 bytes (num_indices, embedding_dim, num_embeddings, padding_idx)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 16;
        pushConstants.push_back(push_range);
    } else if (shader_name == "fused_adagrad_step") {
        // fused_adagrad_step: 16 bytes
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 16;
        pushConstants.push_back(push_range);
    } else if (shader_name == "layer_norm_backward" ||
               shader_name == "layer_norm_backward_f16" ||
               shader_name == "layer_norm_backward_f64") {
        // layer_norm_backward: 16 bytes (batch_size, normalized_shape, epsilon, has_affine)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 16;
        pushConstants.push_back(push_range);
    } else if (shader_name == "prod_reduction" || shader_name == "prod_reduction_i32") {
        // prod_reduction: 16 bytes
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 16;
        pushConstants.push_back(push_range);
    } else if (shader_name == "rms_norm" || shader_name == "rms_norm_f16" || shader_name == "rms_norm_f64" ||
               shader_name == "rms_norm_backward" || shader_name == "rms_norm_backward_f16" ||
               shader_name == "rms_norm_backward_f64") {
        // rms_norm (all variants): 16 bytes
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 16;
        pushConstants.push_back(push_range);
    } else if (shader_name == "bitonic_sort_f64" || shader_name == "bitonic_sort_i32") {
        // bitonic_sort_f64/i32: 20 bytes (n, padded_n, stage, substage, descending)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 20;
        pushConstants.push_back(push_range);
    } else if (shader_name == "gather" || shader_name == "gather_f16" || shader_name == "gather_f64") {
        // gather: 24 bytes (n, dim, outer_size, dim_size, inner_size, num_indices)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 24;
        pushConstants.push_back(push_range);
    } else if (shader_name == "batchnorm" || shader_name == "batchnorm_backward") {
        // batchnorm/batchnorm_backward: 24 bytes
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 24;
        pushConstants.push_back(push_range);
    } else if (shader_name == "group_norm") {
        // group_norm: 28 bytes
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 28;
        pushConstants.push_back(push_range);
    } else if (shader_name == "fused_adadelta_step" || shader_name == "fused_rmsprop_step") {
        // fused_adadelta_step/fused_rmsprop_step: 32 bytes
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 32;
        pushConstants.push_back(push_range);
    } else if (shader_name == "group_norm_backward" ||
               shader_name == "group_norm_backward_f16" ||
               shader_name == "group_norm_backward_f64") {
        // group_norm_backward: 32 bytes
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 32;
        pushConstants.push_back(push_range);
    } else if (shader_name == "scatter" || shader_name == "scatter_f16" || shader_name == "scatter_f64") {
        // scatter: 36 bytes
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 36;
        pushConstants.push_back(push_range);
    } else if (shader_name == "im2col" || shader_name == "col2im") {
        // im2col/col2im: 44 bytes (11 uint32_t values)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 44;
        pushConstants.push_back(push_range);
    } else if (shader_name == "pooling_forward_with_indices") {
        // pooling_forward_with_indices: 48 bytes (12 uint32_t values)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 48;
        pushConstants.push_back(push_range);
    } else if (shader_name == "conv2d" ||
               shader_name == "conv2d_backward_input" || shader_name == "conv2d_backward_input_f64" ||
               shader_name == "conv2d_backward_weight" || shader_name == "conv2d_backward_weight_f64") {
        // conv2d/conv2d_backward_input/weight: 52 bytes (13 uint32_t values)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 52;
        pushConstants.push_back(push_range);
    } else if (shader_name == "pooling" || shader_name == "pooling_backward") {
        // pooling/pooling_backward: 52 bytes (13 uint32_t values)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 52;
        pushConstants.push_back(push_range);
    } else if (shader_name == "cat" || shader_name == "cat_f16" || shader_name == "cat_f64") {
        // cat: 60 bytes (15 uint32_t values)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 60;
        pushConstants.push_back(push_range);
    } else if (shader_name == "max_pool2d_backward") {
        // max_pool2d_backward: 60 bytes (15 uint32_t values)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 60;
        pushConstants.push_back(push_range);
    } else if (shader_name == "math_broadcast_bool") {
        // math_broadcast_bool: 120 bytes (same layout as other math_broadcast variants)
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = 120;
        pushConstants.push_back(push_range);
    }

    // Create pipeline (using persistent pipeline cache for faster creation)
    auto& ctx = devices_[device_id];
    auto pipeline = std::make_unique<vulkan::ComputePipeline>(
        ctx.device, shaderCode, bindings, pushConstants, ctx.pipelineCache
    );

    auto* pipelinePtr = pipeline.get();
    cache.pipelines[shader_name] = std::move(pipeline);

    return pipelinePtr;
}

// Helper to get VkBuffer and offset from a potentially-offset pointer
std::pair<VkBuffer, VkDeviceSize> VulkanBackend::getVulkanBufferAndOffset(const void* ptr) const {
    if (ptr == nullptr) {
        throw std::runtime_error("Invalid buffer pointer: null pointer (empty tensor?)");
    }

    auto& allocator = backend::VulkanCachingAllocator::get();

    // First try direct lookup in caching allocator
    // Find which device this allocation belongs to
    for (int32_t device_id = 0; device_id < device_count(); ++device_id) {
        try {
            VkBuffer buffer = allocator.get_buffer(const_cast<void*>(ptr), device_id);
            return {buffer, 0};
        } catch (...) {
            // Not found on this device, try next
        }
    }

    // If not found directly, ptr might be base_ptr + offset
    // Search through allocations_ to find a buffer where: base_ptr <= ptr < base_ptr + size
    const auto* ptr_as_uint = reinterpret_cast<const uint8_t*>(ptr);

    for (const auto& [base_ptr, alloc_info] : allocations_) {
        const auto* base_as_uint = reinterpret_cast<const uint8_t*>(base_ptr);
        const size_t size = alloc_info.first;
        const int32_t device_id = alloc_info.second;

        // Check if ptr is in the range [base_ptr, base_ptr + size)
        if (ptr_as_uint >= base_as_uint && ptr_as_uint < base_as_uint + size) {
            // Found it! ptr is within this buffer's range
            VkDeviceSize offset = static_cast<VkDeviceSize>(ptr_as_uint - base_as_uint);
            try {
                VkBuffer buffer = allocator.get_buffer(base_ptr, device_id);
                return {buffer, offset};
            } catch (...) {
                // Continue searching
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
    VkResult result = vkAllocateDescriptorSets(ctx.device, &allocInfo, &descriptorSet);

    // If descriptor pool is exhausted, wait for pending work, reset, and retry
    if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL) {
        // Wait for all GPU work to complete before resetting
        ensurePendingWorkComplete(device_id);
        vkDeviceWaitIdle(ctx.device);

        // Reset command pool and descriptor pool
        vkResetCommandPool(ctx.device, ctx.commandPool, 0);
        ctx.nextCommandBufferIndex = 0;
        ctx.descriptorPool->reset();

        // Retry allocation
        result = vkAllocateDescriptorSets(ctx.device, &allocInfo, &descriptorSet);
    }

    vulkan::checkVk(result, "Failed to allocate descriptor set");

    // Write descriptor set bindings
    std::vector<VkDescriptorBufferInfo> bufferInfos(bufferBindings.size());
    std::vector<VkWriteDescriptorSet> writes(bufferBindings.size());

    for (size_t i = 0; i < bufferBindings.size(); ++i) {
        bufferInfos[i].buffer = bufferBindings[i].second;
        bufferInfos[i].offset = 0;
        // Round up to 4-byte boundary (minimum uint32 size for shader access).
        // Float16 shaders pack 2 elements per uint32, so odd-element-count tensors
        // need the range rounded up to avoid out-of-bounds descriptor access.
        bufferInfos[i].range = std::max(bufferSizes[i], static_cast<size_t>(4));

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
    // Serialize all Vulkan operations. Vulkan requires external synchronization
    // for host access to queues, command pools, and descriptor pools.
    std::lock_guard<std::recursive_mutex> lock(dispatch_mutex_);

    // =========================================================================
    // Float16 handling: operations with native F16 shaders use them directly;
    // operations without F16 support fall back to CPU computation.
    // Accumulation-heavy operations (softmax, layer_norm) upcast to F32 in
    // their individual dispatch functions for numerical stability.
    // =========================================================================
    {
        bool has_float16 = false;
        Device original_device{};
        for (const auto& t : inputs) {
            if (t.dtype() == DType::Float16) {
                has_float16 = true;
                original_device = t.device();
                break;
            }
        }

        if (has_float16) {
            // Operations with native Float16 shader support
            static const std::unordered_set<std::string> f16_native_ops = {
                // Binary ops (math_broadcast_f16 shader)
                "add", "sub", "mul", "div",
                "add_inplace", "sub_inplace", "mul_inplace", "div_inplace",
                // Activation ops (activations_f16 shader)
                "relu", "sigmoid", "tanh", "gelu", "leaky_relu", "swish",
                "relu_inplace", "sigmoid_inplace", "tanh_inplace",
                "gelu_inplace", "leaky_relu_inplace",
                // Activation backward (activations_backward_f16 shader)
                "relu_backward", "sigmoid_backward", "tanh_backward",
                "gelu_backward", "leaky_relu_backward",
                // Softmax (upcasts to F32 in dispatch function)
                "softmax", "softmax_backward",
                // Layer norm (upcasts to F32 in dispatch function)
                "layer_norm", "layer_norm_backward",
                // Group norm
                "group_norm", "group_norm_backward",
                // RMSNorm
                "fused_rms_norm", "rms_norm_backward",
                // Embedding backward
                "embedding_backward",
                // Conv2d forward (conv2d_forward_f16 shader with F32 accumulation)
                "conv2d", "conv2d_forward",
                // ConvTranspose2d (conv_transpose2d_forward_f16 shader)
                "conv_transpose2d_forward",
                // Adaptive pooling (adaptive_pooling_f16 shader)
                "adaptive_avg_pool2d", "adaptive_max_pool2d",
                "adaptive_avg_pool2d_backward",
                // Max pool backward with indices
                "max_pool2d_backward_with_indices",
                // Expand/repeat (expand_f16 shader)
                "expand", "repeat",
                // Pooling (max_pool2d_f16, avg_pool2d_f16 shaders)
                "max_pool2d", "max_pool2d_forward", "max_pool2d_backward",
                "avg_pool2d", "avg_pool2d_forward", "avg_pool2d_backward",
                // Strided copy (strided_copy_f16 shader)
                "strided_copy",
                // Random (random_f16 shader)
                "uniform_random", "normal_random",
                // Unary math (math_f16 shader)
                "sqrt", "exp", "log", "neg", "abs", "sign", "pow",
                "floor", "ceil", "round", "trunc", "reciprocal",
                // Trigonometric (trigonometric_f16 shader)
                "sin", "cos", "tan", "asin", "acos", "atan",
                // Hyperbolic (hyperbolic_f16 shader)
                "sinh", "cosh",
                // Comparison (comparison_f16 shader)
                "eq", "ne", "lt", "le", "gt", "ge",
                // Matrix ops (matmul_f16 shader with F32 accumulation)
                "matmul", "bmm", "dot",
                // Fused operations
                "fused_linear_relu", "fused_batchnorm_relu", "fused_add_relu",
                "fused_conv2d_relu", "fused_gelu", "fused_layer_norm",
                "fused_softmax_cross_entropy",
                // BatchNorm (native F16 shaders)
                "argmax", "argmin", "argsort",
                "batchnorm2d_forward", "batchnorm2d_forward_affine",
                "batchnorm2d_backward",
                "batchnorm2d_mean_var", "batchnorm2d_update_running_stats",
                "index_select",
                // Reduction (reduction_f16 shader)
                "sum", "mean", "max", "min", "prod",
                // Type-agnostic operations
                "reshape", "view", "contiguous", "to", "to_dtype",
                "zeros", "ones", "full", "empty",
                // Clamp (dispatch handles F16 via upcast)
                "clamp", "clamp_min", "clamp_max",
                // Conv2d backward (F32 accumulation, safe for F16 inputs)
                "conv2d_backward_input", "conv2d_backward_weight", "conv2d_backward_bias",
                // Shape ops (type-agnostic / metadata-only)
                "transpose", "permute", "cat", "squeeze", "unsqueeze",
                // Indexing ops (dispatch handles F16)
                "gather", "scatter", "embedding", "masked_fill",
                "masked_select", "where",
                // Reduction ops (upcast to F32 internally)
                "var", "std", "norm",
                // Log softmax (dispatch upcasts to F32)
                "log_softmax", "log_softmax_backward",
                // Interpolation
                "interpolate",
                // Memory ops (type-agnostic)
                "clone", "fill", "unfold", "fold",
                // Activation forward/backward (activations_f16 / activations_backward_f16 shaders)
                "elu", "elu_backward", "selu", "selu_backward",
                "mish", "mish_backward", "softplus", "softplus_backward",
                "swish_backward",
            };

            if (!f16_native_ops.contains(op_name)) {
                // CPU fallback for any remaining ops not in the native set
                // (This map should be empty — all known F16 ops are now handled natively)
                static const std::unordered_map<std::string, OpId> op_name_to_id = {
                };

                auto it = op_name_to_id.find(op_name);
                if (it != op_name_to_id.end()) {
                    std::vector<Tensor> cpu_inputs;
                    cpu_inputs.reserve(inputs.size());
                    for (const auto& t : inputs) {
                        cpu_inputs.push_back(t.to(Device::cpu()));
                    }
                    auto cpu_results = tenzor::dispatch(it->second, cpu_inputs, attrs);
                    std::vector<Tensor> vulkan_results;
                    vulkan_results.reserve(cpu_results.size());
                    for (auto& r : cpu_results) {
                        vulkan_results.push_back(r.to(original_device));
                    }
                    return vulkan_results;
                }
            }
        }
    }

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
        if (bytes > 0) {
            void* dst = const_cast<void*>(inputs[0].data_ptr());
            copy(dst, result.data_ptr(), bytes, CopyKind::DeviceToDevice);
        }

        return {inputs[0]};
    }

    // In-place activation operations
    if (op_name == "relu_inplace" || op_name == "sigmoid_inplace" || op_name == "tanh_inplace" ||
        op_name == "leaky_relu_inplace" || op_name == "gelu_inplace") {
        if (inputs.size() < 1) {
            throw std::invalid_argument(op_name + " requires 1 input");
        }
        // Determine the base activation and its opcode/param
        std::string base_op = op_name.substr(0, op_name.find("_inplace"));
        uint32_t opcode = 0;
        float param = 0.0f;
        if (base_op == "relu") opcode = 0;
        else if (base_op == "sigmoid") opcode = 1;
        else if (base_op == "tanh") opcode = 2;
        else if (base_op == "gelu") opcode = 3;
        else if (base_op == "leaky_relu") {
            opcode = 4;
            param = attrs.contains("alpha") ? std::stof(attrs.at("alpha")) : 0.01f;
        }

        // Dispatch out-of-place activation, then copy result back
        Tensor result = dispatchActivation(base_op, inputs[0], opcode, param);
        size_t bytes = result.numel() * result.dtype_size();
        if (bytes > 0) {
            void* dst = const_cast<void*>(inputs[0].data_ptr());
            copy(dst, result.data_ptr(), bytes, CopyKind::DeviceToDevice);
        }
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

    // ELU activation
    if (op_name == "elu") {
        if (inputs.size() != 1) throw std::invalid_argument("elu requires 1 input");
        float alpha = attrs.contains("alpha") ? std::stof(attrs.at("alpha")) : 1.0f;
        return {dispatchActivation("elu", inputs[0], 6, alpha)};
    }

    // SELU activation
    if (op_name == "selu") {
        if (inputs.size() != 1) throw std::invalid_argument("selu requires 1 input");
        return {dispatchActivation("selu", inputs[0], 7, 0.0f)};
    }

    // Mish activation
    if (op_name == "mish") {
        if (inputs.size() != 1) throw std::invalid_argument("mish requires 1 input");
        return {dispatchActivation("mish", inputs[0], 8, 0.0f)};
    }

    // Softplus activation
    if (op_name == "softplus") {
        if (inputs.size() != 1) throw std::invalid_argument("softplus requires 1 input");
        float beta = attrs.contains("beta") ? std::stof(attrs.at("beta")) : 1.0f;
        return {dispatchActivation("softplus", inputs[0], 9, beta)};
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

    if (op_name == "elu_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("elu_backward requires 2 inputs");
        float alpha = attrs.contains("alpha") ? std::stof(attrs.at("alpha")) : 1.0f;
        return {dispatchActivationBackward("elu_backward", inputs[0], inputs[1], 5, alpha)};
    }

    if (op_name == "selu_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("selu_backward requires 2 inputs");
        return {dispatchActivationBackward("selu_backward", inputs[0], inputs[1], 6, 0.0f)};
    }

    if (op_name == "mish_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("mish_backward requires 2 inputs");
        return {dispatchActivationBackward("mish_backward", inputs[0], inputs[1], 7, 0.0f)};
    }

    if (op_name == "softplus_backward") {
        if (inputs.size() != 2) throw std::invalid_argument("softplus_backward requires 2 inputs");
        float beta = attrs.contains("beta") ? std::stof(attrs.at("beta")) : 1.0f;
        return {dispatchActivationBackward("softplus_backward", inputs[0], inputs[1], 8, beta)};
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

    // Unified conv2d backward: computes grad_input, grad_weight, and optionally grad_bias
    // inputs: [grad_output, input, weight]
    if (op_name == "conv2d_backward") {
        if (inputs.size() != 3) {
            throw std::invalid_argument("conv2d_backward operation requires exactly 3 inputs (grad_output, input, weight)");
        }
        const Tensor& grad_output = inputs[0];
        const Tensor& input = inputs[1];
        const Tensor& weight = inputs[2];

        int64_t stride = 1, padding = 0, dilation = 1, groups = 1;
        bool compute_grad_input = true, compute_grad_weight = true, compute_grad_bias = false;

        if (attrs.contains("stride")) stride = std::stoll(attrs.at("stride"));
        if (attrs.contains("padding")) padding = std::stoll(attrs.at("padding"));
        if (attrs.contains("dilation")) dilation = std::stoll(attrs.at("dilation"));
        if (attrs.contains("groups")) groups = std::stoll(attrs.at("groups"));
        if (attrs.contains("compute_grad_input")) compute_grad_input = (attrs.at("compute_grad_input") == "1");
        if (attrs.contains("compute_grad_weight")) compute_grad_weight = (attrs.at("compute_grad_weight") == "1");
        if (attrs.contains("compute_grad_bias")) compute_grad_bias = (attrs.at("compute_grad_bias") == "1");

        std::vector<int64_t> input_shape(input.shape().begin(), input.shape().end());
        std::vector<int64_t> weight_shape(weight.shape().begin(), weight.shape().end());

        std::vector<Tensor> results;

        // Compute grad_input
        if (compute_grad_input) {
            results.push_back(dispatchConv2dBackwardInput(grad_output, weight, stride, padding, dilation, input_shape, groups));
        }

        // Compute grad_weight
        if (compute_grad_weight) {
            results.push_back(dispatchConv2dBackwardWeight(grad_output, input, stride, padding, dilation, weight_shape, groups));
        }

        // Compute grad_bias
        if (compute_grad_bias) {
            results.push_back(dispatchConv2dBackwardBias(grad_output));
        }

        return results;
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

    // Batched matrix multiplication
    if (op_name == "bmm") {
        if (inputs.size() != 2) {
            throw std::invalid_argument("bmm requires 2 inputs");
        }
        return {dispatchBmm(inputs[0], inputs[1])};
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

        // Float16: use max_pool2d_f16 shader via dispatch path (same as Float64)
        if (inputs[0].dtype() == DType::Float16) {
            OpAttributes pool_attrs;
            pool_attrs["kernel_h"] = std::to_string(kernel_h);
            pool_attrs["kernel_w"] = std::to_string(kernel_w);
            pool_attrs["stride_h"] = std::to_string(stride_h);
            pool_attrs["stride_w"] = std::to_string(stride_w);
            pool_attrs["padding_h"] = std::to_string(padding_h);
            pool_attrs["padding_w"] = std::to_string(padding_w);
            Tensor output = dispatchMaxPool2dForward(inputs[0], pool_attrs);
            std::vector<int64_t> out_shape_vec(output.shape().begin(), output.shape().end());
            Tensor pool_indices(out_shape_vec, DType::Int32, inputs[0].device());
            pool_indices = dispatchFill(pool_indices, 0.0f);
            return {output, pool_indices};
        }

        // Float64: use max_pool2d_f64 shader via new dispatch path
        if (inputs[0].dtype() == DType::Float64) {
            OpAttributes pool_attrs;
            pool_attrs["kernel_h"] = std::to_string(kernel_h);
            pool_attrs["kernel_w"] = std::to_string(kernel_w);
            pool_attrs["stride_h"] = std::to_string(stride_h);
            pool_attrs["stride_w"] = std::to_string(stride_w);
            pool_attrs["padding_h"] = std::to_string(padding_h);
            pool_attrs["padding_w"] = std::to_string(padding_w);
            Tensor output = dispatchMaxPool2dForward(inputs[0], pool_attrs);
            // Create indices tensor (Float64 max_pool2d doesn't produce indices)
            std::vector<int64_t> out_shape_vec(output.shape().begin(), output.shape().end());
            Tensor pool_indices(out_shape_vec, DType::Int32, inputs[0].device());
            pool_indices = dispatchFill(pool_indices, 0.0f);
            return {output, pool_indices};
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

        // Float16: use avg_pool2d_f16 shader on GPU
        // (Falls through to dispatchAvgPool2d which will select the correct shader)

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

    if (op_name == "argsort") {
        int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
        bool descending = attrs.contains("descending") && (attrs.at("descending") == "1" || attrs.at("descending") == "true");
        return {dispatchArgSort(inputs[0], dim, descending)};
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
        // Autograd passes: [grad_output, input, weight(gamma), mean, invstd]
        if (inputs.size() < 5) {
            throw std::invalid_argument("batchnorm2d_backward requires 5 inputs (grad_output, input, weight, mean, invstd)");
        }
        const Tensor* gamma = &inputs[2];  // weight = gamma
        float epsilon = attrs.contains("eps") ? std::stof(attrs.at("eps")) : 1e-5f;
        // Pass mean=inputs[3], invstd=inputs[4] (shader uses invstd directly)
        auto [grad_input, grad_gamma, grad_beta] = dispatchBatchNorm2dBackward(
            inputs[0], inputs[1], inputs[3], inputs[4], gamma, epsilon);
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
        // Use GPU kernel for updating running statistics with exponential moving average
        if (inputs.size() != 4) {
            throw std::invalid_argument("batchnorm2d_update_running_stats requires 4 inputs (running_mean, running_var, batch_mean, batch_var)");
        }

        float momentum = 0.1f;
        if (attrs.contains("momentum")) {
            momentum = std::stof(attrs.at("momentum"));
        }

        const Tensor& running_mean = inputs[0];
        const Tensor& running_var = inputs[1];
        const Tensor& batch_mean = inputs[2];
        const Tensor& batch_var = inputs[3];

        int64_t n_channels = batch_mean.numel();
        if (n_channels == 0) {
            return {running_mean, running_var};
        }

        int32_t device_id = running_mean.device().index;

        // Select shader based on dtype
        std::string shader_name = "batchnorm_update_stats";
        if (running_mean.dtype() == DType::Float64) {
            shader_name = "batchnorm_update_stats_f64";
        } else if (running_mean.dtype() == DType::Float16) {
            shader_name = "batchnorm_update_stats_f16";
        } else if (running_mean.dtype() != DType::Float32) {
            throw std::runtime_error("Unsupported dtype for batchnorm2d_update_running_stats");
        }

        auto* pipeline = getPipeline(shader_name, device_id);

        VkBuffer buffer_rm = getVulkanBuffer(running_mean.data_ptr());
        VkBuffer buffer_rv = getVulkanBuffer(running_var.data_ptr());
        VkBuffer buffer_bm = getVulkanBuffer(const_cast<void*>(batch_mean.data_ptr()));
        VkBuffer buffer_bv = getVulkanBuffer(const_cast<void*>(batch_var.data_ptr()));

        size_t buffer_size = n_channels * running_mean.dtype_size();

        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
            {0, buffer_rm},
            {1, buffer_rv},
            {2, buffer_bm},
            {3, buffer_bv}
        };
        std::vector<size_t> sizes = {buffer_size, buffer_size, buffer_size, buffer_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t n_channels;
            float momentum;
        } push_constants;

        push_constants.n_channels = static_cast<uint32_t>(n_channels);
        push_constants.momentum = momentum;

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);

        uint32_t workgroups = (n_channels + 255) / 256;
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);

        // Return the updated tensors (modified in-place)
        return {running_mean, running_var};
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
        if (inputs.size() >= 2) {
            return {dispatchAvgPool2dBackward(inputs[0], inputs[1], attrs)};
        }
        if (inputs.size() == 1) {
            // Autograd path: 1 input (grad_output) + shape/pooling params in attrs
            // Build compatible attrs for dispatchAvgPool2dBackward
            OpAttributes bwd_attrs;

            // Parse input shape from "input_shape" attribute (comma-separated "N,C,H,W")
            int64_t in_n = 0, in_c = 0, in_h = 0, in_w = 0;
            if (attrs.contains("input_shape")) {
                std::string shape_str = attrs.at("input_shape");
                std::stringstream ss(shape_str);
                std::string token;
                std::vector<int64_t> dims;
                while (std::getline(ss, token, ',')) {
                    dims.push_back(std::stoll(token));
                }
                in_n = dims[0]; in_c = dims[1]; in_h = dims[2]; in_w = dims[3];
            }

            // Parse pooling parameters (support both single-value and h/w variants)
            int64_t kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
            if (attrs.contains("kernel_h")) {
                kernel_h = std::stoll(attrs.at("kernel_h"));
                kernel_w = attrs.contains("kernel_w") ? std::stoll(attrs.at("kernel_w")) : kernel_h;
            } else {
                kernel_h = kernel_w = std::stoll(attrs.at("kernel_size"));
            }
            if (attrs.contains("stride_h")) {
                stride_h = std::stoll(attrs.at("stride_h"));
                stride_w = attrs.contains("stride_w") ? std::stoll(attrs.at("stride_w")) : stride_h;
            } else if (attrs.contains("stride")) {
                stride_h = stride_w = std::stoll(attrs.at("stride"));
            } else {
                stride_h = kernel_h; stride_w = kernel_w;
            }
            if (attrs.contains("padding_h")) {
                padding_h = std::stoll(attrs.at("padding_h"));
                padding_w = attrs.contains("padding_w") ? std::stoll(attrs.at("padding_w")) : padding_h;
            } else if (attrs.contains("padding")) {
                padding_h = padding_w = std::stoll(attrs.at("padding"));
            } else {
                padding_h = padding_w = 0;
            }

            bwd_attrs["kernel_h"] = std::to_string(kernel_h);
            bwd_attrs["kernel_w"] = std::to_string(kernel_w);
            bwd_attrs["stride_h"] = std::to_string(stride_h);
            bwd_attrs["stride_w"] = std::to_string(stride_w);
            bwd_attrs["padding_h"] = std::to_string(padding_h);
            bwd_attrs["padding_w"] = std::to_string(padding_w);
            bwd_attrs["count_include_pad"] = "1";

            // Create a dummy input tensor with the right shape for dispatchAvgPool2dBackward
            Tensor dummy_input({in_n, in_c, in_h, in_w}, inputs[0].dtype(), inputs[0].device());

            return {dispatchAvgPool2dBackward(inputs[0], dummy_input, bwd_attrs)};
        }
        throw std::invalid_argument("avg_pool2d_backward requires at least 1 input");
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

    // ConvTranspose2d forward operation
    if (op_name == "conv_transpose2d_forward") {
        if (inputs.size() < 2) {
            throw std::invalid_argument("conv_transpose2d_forward requires at least 2 inputs (input, weight)");
        }
        // CPU fallback for unsupported dtypes (Int32, etc.)
        if (inputs[0].dtype() != DType::Float32 && inputs[0].dtype() != DType::Float64 && inputs[0].dtype() != DType::Float16) {
            Device original_device = inputs[0].device();
            std::vector<Tensor> cpu_inputs;
            cpu_inputs.reserve(inputs.size());
            for (const auto& t : inputs) {
                cpu_inputs.push_back(t.to(Device::cpu()));
            }
            auto cpu_results = tenzor::dispatch(OpId::ConvTranspose2dForward, cpu_inputs, attrs);
            return {cpu_results[0].to(original_device)};
        }
        const Tensor* bias_ptr = (inputs.size() >= 3) ? &inputs[2] : nullptr;
        return {dispatchConvTranspose2dForward(inputs[0], inputs[1], bias_ptr, attrs)};
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
            } else if (dtype_str == "bfloat16" || dtype_str == "BFloat16") {
                dtype = DType::BFloat16;
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
            } else if (dtype_str == "bfloat16") {
                dtype = DType::BFloat16;
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
            } else if (dtype_str == "bfloat16" || dtype_str == "BFloat16") {
                dtype = DType::BFloat16;
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
            } else if (dtype_str == "bfloat16" || dtype_str == "BFloat16") {
                dtype = DType::BFloat16;
            }
        }
        return {dispatchRandn(shape, dtype)};
    }

    // Arange operation
    if (op_name == "arange") {
        float start = attrs.contains("start") ? std::stof(attrs.at("start")) : 0.0f;
        float end_val = attrs.contains("end") ? std::stof(attrs.at("end")) : 0.0f;
        float step = attrs.contains("step") ? std::stof(attrs.at("step")) : 1.0f;
        DType dtype = DType::Float32;
        if (attrs.contains("dtype")) {
            std::string dtype_str = attrs.at("dtype");
            if (dtype_str == "float64" || dtype_str == "Float64") dtype = DType::Float64;
            else if (dtype_str == "float16" || dtype_str == "Float16") dtype = DType::Float16;
            else if (dtype_str == "bfloat16" || dtype_str == "BFloat16") dtype = DType::BFloat16;
            else if (dtype_str == "int32" || dtype_str == "Int32") dtype = DType::Int32;
            else if (dtype_str == "int64" || dtype_str == "Int64") dtype = DType::Int64;
            else if (dtype_str == "int8" || dtype_str == "Int8") dtype = DType::Int8;
            else if (dtype_str == "uint8" || dtype_str == "UInt8") dtype = DType::UInt8;
            else if (dtype_str == "bool" || dtype_str == "Bool") dtype = DType::Bool;
        }
        int32_t device_id = attrs.contains("device_id") ? std::stoi(attrs.at("device_id")) : 0;
        Device device = Device::vulkan(device_id);
        return {dispatchArange(start, end_val, step, dtype, device)};
    }

    // Linspace operation
    if (op_name == "linspace") {
        float start = attrs.contains("start") ? std::stof(attrs.at("start")) : 0.0f;
        float end_val = attrs.contains("end") ? std::stof(attrs.at("end")) : 1.0f;
        int64_t steps = attrs.contains("steps") ? std::stoll(attrs.at("steps")) : 100;
        DType dtype = DType::Float32;
        if (attrs.contains("dtype")) {
            std::string dtype_str = attrs.at("dtype");
            if (dtype_str == "float64" || dtype_str == "Float64") dtype = DType::Float64;
            else if (dtype_str == "float16" || dtype_str == "Float16") dtype = DType::Float16;
            else if (dtype_str == "bfloat16" || dtype_str == "BFloat16") dtype = DType::BFloat16;
            else if (dtype_str == "int32" || dtype_str == "Int32") dtype = DType::Int32;
            else if (dtype_str == "int64" || dtype_str == "Int64") dtype = DType::Int64;
            else if (dtype_str == "int8" || dtype_str == "Int8") dtype = DType::Int8;
            else if (dtype_str == "uint8" || dtype_str == "UInt8") dtype = DType::UInt8;
            else if (dtype_str == "bool" || dtype_str == "Bool") dtype = DType::Bool;
        }
        int32_t device_id = attrs.contains("device_id") ? std::stoi(attrs.at("device_id")) : 0;
        Device device = Device::vulkan(device_id);
        return {dispatchLinspace(start, end_val, steps, dtype, device)};
    }

    // Eye operation
    if (op_name == "eye") {
        int64_t n = attrs.contains("n") ? std::stoll(attrs.at("n")) : 0;
        int64_t m = attrs.contains("m") ? std::stoll(attrs.at("m")) : -1;
        DType dtype = DType::Float32;
        if (attrs.contains("dtype")) {
            std::string dtype_str = attrs.at("dtype");
            if (dtype_str == "float64" || dtype_str == "Float64") dtype = DType::Float64;
            else if (dtype_str == "float16" || dtype_str == "Float16") dtype = DType::Float16;
            else if (dtype_str == "bfloat16" || dtype_str == "BFloat16") dtype = DType::BFloat16;
            else if (dtype_str == "int32" || dtype_str == "Int32") dtype = DType::Int32;
            else if (dtype_str == "int64" || dtype_str == "Int64") dtype = DType::Int64;
            else if (dtype_str == "int8" || dtype_str == "Int8") dtype = DType::Int8;
            else if (dtype_str == "uint8" || dtype_str == "UInt8") dtype = DType::UInt8;
            else if (dtype_str == "bool" || dtype_str == "Bool") dtype = DType::Bool;
        }
        int32_t device_id = attrs.contains("device_id") ? std::stoi(attrs.at("device_id")) : 0;
        Device device = Device::vulkan(device_id);
        return {dispatchEye(n, m, dtype, device)};
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

    // ========================================================================
    // Fused Operations (composed from existing operations)
    // ========================================================================

    // Fused Linear + ReLU: matmul(input, weight^T) + bias + relu
    if (op_name == "fused_linear_relu") {
        if (inputs.size() < 2) {
            throw std::invalid_argument("fused_linear_relu requires at least 2 inputs (input, weight)");
        }
        bool has_bias = attrs.contains("has_bias") && attrs.at("has_bias") == "true";

        auto input_shape = inputs[0].shape();
        int64_t in_features = input_shape[input_shape.size() - 1];
        int64_t out_features = inputs[1].shape()[0];

        // Flatten input to 2D: (..., in_features) -> (batch_size, in_features)
        int64_t batch_size = 1;
        for (size_t i = 0; i < input_shape.size() - 1; ++i) {
            batch_size *= input_shape[i];
        }
        Tensor input_2d = inputs[0].reshape({batch_size, in_features});

        // Transpose weight as a view (swap shape/strides, same data)
        // dispatchMatmul will call dispatchContiguous if needed
        Tensor weight_t = inputs[1].transpose(0, 1);

        // MatMul: input_2d @ weight^T -> (batch_size, out_features)
        Tensor mm_result = dispatchMatmul(input_2d, weight_t);

        // Add bias if present
        if (has_bias && inputs.size() > 2) {
            mm_result = dispatchBinaryOp("add", mm_result, inputs[2]);
        }

        // ReLU activation
        Tensor result = dispatchActivation("relu", mm_result, 0, 0.0f);

        // Reshape back to original batch dimensions
        std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end() - 1);
        out_shape.push_back(out_features);
        result = result.reshape(out_shape);

        return {result};
    }

    // Fused BatchNorm + ReLU
    if (op_name == "fused_batchnorm_relu") {
        if (inputs.size() < 5) {
            throw std::invalid_argument("fused_batchnorm_relu requires 5 inputs (input, mean, var, weight, bias)");
        }

        float eps = attrs.contains("eps") ? std::stof(attrs.at("eps")) : 1e-5f;
        auto orig_shape = inputs[0].shape();

        // Reshape to 4D if needed: (N, C, ...) -> (N, C, H, W) where H*W = spatial_size
        Tensor input_4d = inputs[0];
        bool needs_reshape = (orig_shape.size() != 4);
        if (needs_reshape) {
            int64_t N = orig_shape[0];
            int64_t C = orig_shape[1];
            int64_t spatial = 1;
            for (size_t i = 2; i < orig_shape.size(); ++i) {
                spatial *= orig_shape[i];
            }
            input_4d = inputs[0].reshape({N, C, spatial, 1});
        }

        // BatchNorm forward
        Tensor bn_result = dispatchBatchNorm2dForward(input_4d, inputs[1], inputs[2],
                                                       &inputs[3], &inputs[4], eps);

        // Reshape back if needed
        if (needs_reshape) {
            std::vector<int64_t> shape_vec(orig_shape.begin(), orig_shape.end());
            bn_result = bn_result.reshape(shape_vec);
        }

        // ReLU activation
        Tensor result = dispatchActivation("relu", bn_result, 0, 0.0f);
        return {result};
    }

    // Fused Add + ReLU
    if (op_name == "fused_add_relu") {
        if (inputs.size() != 2) {
            throw std::invalid_argument("fused_add_relu requires 2 inputs");
        }
        // Add
        std::vector<Tensor> add_inputs = {inputs[0], inputs[1]};
        OpAttributes empty_attrs;
        Tensor add_result = dispatch("add", add_inputs, empty_attrs)[0];
        // ReLU
        std::vector<Tensor> relu_inputs = {add_result};
        return dispatch("relu", relu_inputs, empty_attrs);
    }

    // Fused GELU
    if (op_name == "fused_gelu") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("fused_gelu requires 1 input");
        }
        std::vector<Tensor> gelu_inputs = {inputs[0]};
        OpAttributes empty_attrs;
        return dispatch("gelu", gelu_inputs, empty_attrs);
    }

    // Fused Layer Norm
    if (op_name == "fused_layer_norm") {
        if (inputs.size() < 3) {
            throw std::invalid_argument("fused_layer_norm requires 3 inputs (input, weight, bias)");
        }
        // Parse normalized_shape from comma-separated string to compute total size
        std::string ns_str = attrs.at("normalized_shape");
        int64_t normalized_size = 1;
        std::stringstream ss(ns_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            normalized_size *= std::stoll(token);
        }
        float eps = attrs.contains("eps") ? std::stof(attrs.at("eps")) : 1e-5f;
        Tensor result = dispatchLayerNorm(inputs[0], normalized_size,
                                           &inputs[1], &inputs[2], eps);
        return {result};
    }

    // ========================================================================
    // Interpolation Operation
    // ========================================================================
    if (op_name == "interpolate") {
        if (inputs.size() != 1) {
            throw std::invalid_argument("interpolate requires 1 input");
        }
        return {dispatchInterpolate(inputs[0], attrs)};
    }

    // ========================================================================
    // ROI Align Operations
    // ========================================================================
    if (op_name == "roi_align_forward") {
        if (inputs.size() < 2) {
            throw std::invalid_argument("roi_align_forward requires 2 inputs (features, rois)");
        }
        return {dispatchROIAlignForward(inputs[0], inputs[1], attrs)};
    }

    if (op_name == "roi_align_backward") {
        if (inputs.size() < 2) {
            throw std::invalid_argument("roi_align_backward requires 2 inputs (grad_output, rois)");
        }
        return {dispatchROIAlignBackward(inputs[0], inputs[1], attrs)};
    }

    // ========================================================================
    // LayerNorm / GroupNorm string dispatch
    // ========================================================================
    if (op_name == "layer_norm") {
        if (inputs.size() < 1) {
            throw std::invalid_argument("layer_norm requires at least 1 input");
        }
        int64_t normalized_size = 1;
        if (attrs.contains("normalized_shape")) {
            std::string ns_str = attrs.at("normalized_shape");
            std::stringstream ss(ns_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                normalized_size *= std::stoll(token);
            }
        } else {
            normalized_size = inputs[0].shape().back();
        }
        float eps = attrs.contains("eps") ? std::stof(attrs.at("eps")) : 1e-5f;
        const Tensor* gamma = (inputs.size() > 1) ? &inputs[1] : nullptr;
        const Tensor* beta = (inputs.size() > 2) ? &inputs[2] : nullptr;
        return {dispatchLayerNorm(inputs[0], normalized_size, gamma, beta, eps)};
    }

    if (op_name == "layer_norm_backward") {
        // inputs: [grad_output, input, mean, rstd, weight]
        if (inputs.size() < 5) {
            throw std::invalid_argument("layer_norm_backward requires 5 inputs (grad_output, input, mean, rstd, weight)");
        }
        int64_t normalized_shape = inputs[0].shape().back();
        if (attrs.contains("normalized_shape")) {
            normalized_shape = std::stoll(attrs.at("normalized_shape"));
        }
        auto [grad_input, grad_weight, grad_bias] = dispatchLayerNormBackward(
            inputs[0], inputs[1], inputs[2], inputs[3], &inputs[4], normalized_shape);
        return {grad_input, grad_weight, grad_bias};
    }

    if (op_name == "group_norm") {
        if (inputs.size() < 1) {
            throw std::invalid_argument("group_norm requires at least 1 input");
        }
        int64_t num_groups = attrs.contains("num_groups") ? std::stoll(attrs.at("num_groups")) : 1;
        float eps = attrs.contains("eps") ? std::stof(attrs.at("eps")) : 1e-5f;
        const Tensor* gamma = (inputs.size() > 1) ? &inputs[1] : nullptr;
        const Tensor* beta = (inputs.size() > 2) ? &inputs[2] : nullptr;
        return dispatchGroupNorm(inputs[0], num_groups, gamma, beta, eps);
    }

    if (op_name == "group_norm_backward") {
        // inputs: [grad_output, input, mean, rstd, weight]
        if (inputs.size() < 5) {
            throw std::invalid_argument("group_norm_backward requires 5 inputs (grad_output, input, mean, rstd, weight)");
        }
        int64_t num_groups = attrs.contains("num_groups") ? std::stoll(attrs.at("num_groups")) : 1;
        auto [grad_input, grad_weight, grad_bias] = dispatchGroupNormBackward(
            inputs[0], inputs[1], inputs[2], inputs[3], &inputs[4], num_groups);
        return {grad_input, grad_weight, grad_bias};
    }

    // ========================================================================
    // Embedding Backward
    // ========================================================================
    if (op_name == "embedding_backward") {
        // inputs: [grad_output, indices]
        if (inputs.size() < 2) {
            throw std::invalid_argument("embedding_backward requires 2 inputs (grad_output, indices)");
        }
        int64_t num_embeddings = attrs.contains("num_embeddings") ? std::stoll(attrs.at("num_embeddings")) : 0;
        int64_t embedding_dim = inputs[0].shape().back();
        return {dispatchEmbeddingBackward(inputs[0], inputs[1], num_embeddings, embedding_dim)};
    }

    // ========================================================================
    // RMSNorm Forward and Backward
    // ========================================================================
    if (op_name == "fused_rms_norm") {
        if (inputs.size() < 2) {
            throw std::invalid_argument("fused_rms_norm requires 2 inputs (input, weight)");
        }
        float eps = attrs.contains("eps") ? std::stof(attrs.at("eps")) : 1e-5f;
        int64_t normalized_shape = inputs[0].shape().back();
        auto [output, rrms] = dispatchRMSNorm(inputs[0], inputs[1], normalized_shape, eps);
        return {output, rrms};
    }

    if (op_name == "rms_norm_backward") {
        // inputs: [grad_output, input, rrms, weight]
        if (inputs.size() < 4) {
            throw std::invalid_argument("rms_norm_backward requires 4 inputs (grad_output, input, rrms, weight)");
        }
        int64_t normalized_shape = inputs[0].shape().back();
        if (attrs.contains("normalized_shape")) {
            normalized_shape = std::stoll(attrs.at("normalized_shape"));
        }
        auto [grad_input, grad_weight] = dispatchRMSNormBackward(
            inputs[0], inputs[1], inputs[2], inputs[3], normalized_shape);
        return {grad_input, grad_weight};
    }

    // ========================================================================
    // Phase 3: Nonzero, OneHot, BoxIoU
    // ========================================================================
    if (op_name == "nonzero") {
        if (inputs.size() < 1) {
            throw std::invalid_argument("nonzero requires 1 input");
        }
        return {dispatchNonzero(inputs[0])};
    }

    if (op_name == "one_hot") {
        if (inputs.size() < 1) {
            throw std::invalid_argument("one_hot requires 1 input (indices)");
        }
        int64_t num_classes = attrs.contains("num_classes") ? std::stoll(attrs.at("num_classes")) : 10;
        return {dispatchOneHot(inputs[0], num_classes)};
    }

    if (op_name == "box_iou") {
        if (inputs.size() < 2) {
            throw std::invalid_argument("box_iou requires 2 inputs (boxes1, boxes2)");
        }
        int64_t iou_type = attrs.contains("iou_type") ? std::stoll(attrs.at("iou_type")) : 0;
        return {dispatchBoxIoU(inputs[0], inputs[1], iou_type)};
    }

    // ========================================================================
    // Phase 4: Fused Optimizer Steps
    // ========================================================================
    if (op_name == "fused_rmsprop_step") {
        // inputs: [grad, param, square_avg, momentum_buf, grad_avg]
        if (inputs.size() < 3) {
            throw std::invalid_argument("fused_rmsprop_step requires at least 3 inputs");
        }
        int64_t numel = inputs[0].numel();
        float lr = attrs.contains("lr") ? std::stof(attrs.at("lr")) : 0.01f;
        float alpha = attrs.contains("alpha") ? std::stof(attrs.at("alpha")) : 0.99f;
        float eps = attrs.contains("eps") ? std::stof(attrs.at("eps")) : 1e-8f;
        float weight_decay = attrs.contains("weight_decay") ? std::stof(attrs.at("weight_decay")) : 0.0f;
        float momentum = attrs.contains("momentum") ? std::stof(attrs.at("momentum")) : 0.0f;
        bool centered = attrs.contains("centered") && attrs.at("centered") == "1";

        int32_t device_id = inputs[0].device().index;
        auto* pipeline = getPipeline("fused_rmsprop_step", device_id);

        bool has_momentum = (inputs.size() > 3 && momentum > 0.0f);
        bool has_grad_avg = (inputs.size() > 4 && centered);

        VkBuffer buf_grad = getVulkanBuffer(inputs[0].data_ptr());
        VkBuffer buf_param = getVulkanBuffer(inputs[1].data_ptr());
        VkBuffer buf_sq_avg = getVulkanBuffer(inputs[2].data_ptr());

        size_t buf_size = numel * inputs[0].dtype_size();
        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
            {0, buf_grad}, {1, buf_param}, {2, buf_sq_avg},
        };
        std::vector<size_t> sizes = {buf_size, buf_size, buf_size};

        if (has_momentum) {
            bindings.push_back({3, getVulkanBuffer(inputs[3].data_ptr())});
            sizes.push_back(buf_size);
        } else {
            bindings.push_back({3, buf_param}); // dummy
            sizes.push_back(buf_size);
        }
        if (has_grad_avg) {
            bindings.push_back({4, getVulkanBuffer(inputs[4].data_ptr())});
            sizes.push_back(buf_size);
        } else {
            bindings.push_back({4, buf_param}); // dummy
            sizes.push_back(buf_size);
        }

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t numel;
            float lr;
            float alpha;
            float eps;
            float weight_decay;
            float momentum;
            uint32_t centered;
            uint32_t has_momentum;
        } pc;
        pc.numel = static_cast<uint32_t>(numel);
        pc.lr = lr;
        pc.alpha = alpha;
        pc.eps = eps;
        pc.weight_decay = weight_decay;
        pc.momentum = momentum;
        pc.centered = centered ? 1u : 0u;
        pc.has_momentum = has_momentum ? 1u : 0u;

        uint32_t workgroups = (numel + 255) / 256;
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &pc);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
        insertComputeBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return {};  // In-place update, no outputs
    }

    if (op_name == "fused_adadelta_step") {
        // inputs: [grad, param, square_avg, acc_delta]
        if (inputs.size() < 4) {
            throw std::invalid_argument("fused_adadelta_step requires 4 inputs");
        }
        int64_t numel = inputs[0].numel();
        float lr = attrs.contains("lr") ? std::stof(attrs.at("lr")) : 1.0f;
        float rho = attrs.contains("rho") ? std::stof(attrs.at("rho")) : 0.9f;
        float eps = attrs.contains("eps") ? std::stof(attrs.at("eps")) : 1e-6f;
        float weight_decay = attrs.contains("weight_decay") ? std::stof(attrs.at("weight_decay")) : 0.0f;

        int32_t device_id = inputs[0].device().index;
        auto* pipeline = getPipeline("fused_adadelta_step", device_id);

        size_t buf_size = numel * inputs[0].dtype_size();
        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
            {0, getVulkanBuffer(inputs[0].data_ptr())},
            {1, getVulkanBuffer(inputs[1].data_ptr())},
            {2, getVulkanBuffer(inputs[2].data_ptr())},
            {3, getVulkanBuffer(inputs[3].data_ptr())},
        };
        std::vector<size_t> sizes = {buf_size, buf_size, buf_size, buf_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t numel;
            float lr;
            float rho;
            float eps;
            float weight_decay;
            uint32_t padding0;
            uint32_t padding1;
            uint32_t padding2;
        } pc;
        pc.numel = static_cast<uint32_t>(numel);
        pc.lr = lr;
        pc.rho = rho;
        pc.eps = eps;
        pc.weight_decay = weight_decay;
        pc.padding0 = 0;
        pc.padding1 = 0;
        pc.padding2 = 0;

        uint32_t workgroups = (numel + 255) / 256;
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &pc);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
        insertComputeBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return {};
    }

    if (op_name == "fused_adagrad_step") {
        // inputs: [grad, param, sum_sq]
        if (inputs.size() < 3) {
            throw std::invalid_argument("fused_adagrad_step requires 3 inputs");
        }
        int64_t numel = inputs[0].numel();
        float lr = attrs.contains("lr") ? std::stof(attrs.at("lr")) : 0.01f;
        float eps = attrs.contains("eps") ? std::stof(attrs.at("eps")) : 1e-10f;
        float weight_decay = attrs.contains("weight_decay") ? std::stof(attrs.at("weight_decay")) : 0.0f;

        int32_t device_id = inputs[0].device().index;
        auto* pipeline = getPipeline("fused_adagrad_step", device_id);

        size_t buf_size = numel * inputs[0].dtype_size();
        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
            {0, getVulkanBuffer(inputs[0].data_ptr())},
            {1, getVulkanBuffer(inputs[1].data_ptr())},
            {2, getVulkanBuffer(inputs[2].data_ptr())},
        };
        std::vector<size_t> sizes = {buf_size, buf_size, buf_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t numel;
            float lr;
            float eps;
            float weight_decay;
        } pc;
        pc.numel = static_cast<uint32_t>(numel);
        pc.lr = lr;
        pc.eps = eps;
        pc.weight_decay = weight_decay;

        uint32_t workgroups = (numel + 255) / 256;
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &pc);
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);
        insertComputeBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return {};
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
        std::string err = "Tensors shapes are not broadcastable: [";
        for (size_t i = 0; i < shape_a_vec.size(); i++) {
            if (i > 0) err += ",";
            err += std::to_string(shape_a_vec[i]);
        }
        err += "] vs [";
        for (size_t i = 0; i < shape_b_vec.size(); i++) {
            if (i > 0) err += ",";
            err += std::to_string(shape_b_vec[i]);
        }
        err += "] (op=" + op_name + ")";
        throw std::runtime_error(err);
    }

    // Compute output shape
    std::vector<int64_t> output_shape = compute_broadcast_shape(shape_a_vec, shape_b_vec);

    // Handle empty tensors - no GPU work needed
    int64_t out_numel = 1;
    for (auto d : output_shape) out_numel *= d;
    if (out_numel == 0) {
        return Tensor(output_shape, a.dtype(), a.device());
    }

    int32_t device_id = a.device().index;

    // Map operation name to opcode
    uint32_t opcode = 0;
    if (op_name == "add") opcode = 0;
    else if (op_name == "sub") opcode = 1;
    else if (op_name == "mul") opcode = 2;
    else if (op_name == "div") opcode = 3;
    else throw std::runtime_error("Unknown binary operation: " + op_name);

    // BFloat16: upcast to Float32, compute, downcast back
    if (a.dtype() == DType::BFloat16) {
        auto a_f32 = a.to(DType::Float32);
        auto b_f32 = b.to(DType::Float32);
        auto result_f32 = dispatchBinaryOp(op_name, a_f32, b_f32);
        return result_f32.to(DType::BFloat16);
    }

    // Check if we can use the fast path (same-shape, no broadcasting needed)
    bool same_shape = std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end());
    bool is_float32 = (a.dtype() == DType::Float32);
    bool is_float64 = (a.dtype() == DType::Float64);
    bool is_float16 = (a.dtype() == DType::Float16);

    if (same_shape && (is_float32 || is_float64 || is_float16)) {
        // Fast path: use math shader for same-shape operations
        // Select shader based on dtype
        std::string shader_name = is_float64 ? "math_f64" : (is_float16 ? "math_f16" : "math");
        auto* pipeline = getPipeline(shader_name, device_id);

        Tensor output(output_shape, a.dtype(), a.device());

        // Prepare push constants - use different structure for Float32/Float16 vs Float64
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
            // Float32 and Float16 use the same push constants layout (n, op, param as float)
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
        // For Float16, the shader reads uint32 words (2 elements per word),
        // so descriptor ranges must be rounded up to 4-byte boundaries
        size_t buffer_size_a = a.numel() * a.dtype_size();
        size_t buffer_size_b = b.numel() * b.dtype_size();
        size_t buffer_size_out = output.numel() * output.dtype_size();
        if (is_float16) {
            size_t num_pairs_a = (a.numel() + 1) / 2;
            size_t num_pairs_b = (b.numel() + 1) / 2;
            size_t num_pairs_out = (output.numel() + 1) / 2;
            buffer_size_a = num_pairs_a * 4;
            buffer_size_b = num_pairs_b * 4;
            buffer_size_out = num_pairs_out * 4;
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
                          0, push_constants_size, push_constants_ptr);

        // Float16 shader processes pairs of elements, so we need fewer workgroups
        uint32_t workgroups;
        if (is_float16) {
            uint32_t num_pairs = (static_cast<uint32_t>(a.numel()) + 1) / 2;
            workgroups = (num_pairs + 255) / 256;
        } else {
            workgroups = (a.numel() + 255) / 256;
        }
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeBarrier(cmdBuffer);

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

        insertComputeBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);

        // Synchronize to ensure GPU has completed before using the result
        synchronize(device_id);

        return output;
    }
}

auto VulkanBackend::dispatchUnaryOp(const std::string& op_name,
                                    const Tensor& input) -> Tensor {
    // Handle empty tensors - no GPU work needed
    if (input.numel() == 0) {
        auto input_shape = input.shape();
        std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
        return Tensor(output_shape, input.dtype(), input.device());
    }

    // BFloat16: upcast to Float32, compute, downcast back
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchUnaryOp(op_name, input_f32);
        return result_f32.to(DType::BFloat16);
    }

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
    if (shader_name == "math") {
        if (input.dtype() == DType::Float64) {
            shader_name = "math_f64";
        } else if (input.dtype() == DType::Int32) {
            shader_name = "math_i32";
        } else if (input.dtype() == DType::Float16) {
            shader_name = "math_f16";
        }
    } else if (shader_name == "trigonometric") {
        if (input.dtype() == DType::Float16) {
            shader_name = "trigonometric_f16";
        } else if (input.dtype() == DType::Float64) {
            shader_name = "trigonometric_f64";
        }
    } else if (shader_name == "hyperbolic") {
        if (input.dtype() == DType::Float16) {
            shader_name = "hyperbolic_f16";
        } else if (input.dtype() == DType::Float64) {
            shader_name = "hyperbolic_f64";
        }
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    // Prepare push constants - use different structure based on shader type
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_trig_or_hyp = (shader_name == "trigonometric" || shader_name == "trigonometric_f16" || shader_name == "trigonometric_f64" ||
                           shader_name == "hyperbolic" || shader_name == "hyperbolic_f16" || shader_name == "hyperbolic_f64");
    struct PushConstantsSimple {
        uint32_t n;
        uint32_t op;
    };
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

    PushConstantsSimple push_constants_simple;
    PushConstantsF32 push_constants_f32;
    PushConstantsF64 push_constants_f64;
    void* push_constants_ptr;
    size_t push_constants_size;

    if (is_trig_or_hyp) {
        push_constants_simple.n = static_cast<uint32_t>(input.numel());
        push_constants_simple.op = opcode;
        push_constants_ptr = &push_constants_simple;
        push_constants_size = sizeof(PushConstantsSimple);
    } else if (is_float64) {
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
    if (input.dtype() == DType::Float16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

    // Allocate and write descriptor set
    // For trigonometric/hyperbolic shaders: Binding 0: input, Binding 1: output
    // For math/math_f64 shaders: Binding 0: input, Binding 1: unused (set to input), Binding 2: output
    std::vector<std::pair<uint32_t, VkBuffer>> bindings;
    std::vector<size_t> sizes;

    if (shader_name == "math" || shader_name == "math_f64" || shader_name == "math_i32" || shader_name == "math_f16") {
        bindings = {
            {0, buffer_in},
            {1, buffer_in},  // Unary ops don't use binding 1, but descriptor set expects it
            {2, buffer_out}
        };
        sizes = {buffer_size_in, buffer_size_in, buffer_size_out};
    } else {
        // trigonometric, hyperbolic, and their F16 variants use simpler layout
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

    // For F16 packed-pair shaders, each thread processes 2 elements
    bool is_f16_packed = (shader_name == "math_f16" || shader_name == "trigonometric_f16" || shader_name == "hyperbolic_f16");
    uint32_t num_work_items = is_f16_packed ? static_cast<uint32_t>((input.numel() + 1) / 2) : static_cast<uint32_t>(input.numel());
    uint32_t workgroups = (num_work_items + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    // Synchronize to ensure GPU has completed before using the result
    synchronize(device_id);

    return output;
}

auto VulkanBackend::dispatchUnaryOpWithParam(const std::string& op_name,
                                              const Tensor& input,
                                              float param) -> Tensor {
    // BFloat16: upcast to Float32, compute, downcast back
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchUnaryOpWithParam(op_name, input_f32, param);
        return result_f32.to(DType::BFloat16);
    }

    int32_t device_id = input.device().index;

    // Select correct pipeline based on dtype
    std::string shader_name;
    if (input.dtype() == DType::Float64) shader_name = "math_f64";
    else if (input.dtype() == DType::Float16) shader_name = "math_f16";
    else shader_name = "math";
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
    if (input.dtype() == DType::Float16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

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

    // For F16 packed-pair shader, each thread processes 2 elements
    uint32_t num_work_items = (shader_name == "math_f16") ? static_cast<uint32_t>((input.numel() + 1) / 2) : static_cast<uint32_t>(input.numel());
    uint32_t workgroups = (num_work_items + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeBarrier(cmdBuffer);
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
    std::string shader_name = "trigonometric";
    if (input.dtype() == DType::Float16) shader_name = "trigonometric_f16";
    else if (input.dtype() == DType::Float64) shader_name = "trigonometric_f64";
    auto* pipeline = getPipeline(shader_name, device_id);

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
    if (input.dtype() == DType::Float16) {
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

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

    // For F16, each thread processes 2 elements
    uint32_t num_work_items = (input.dtype() == DType::Float16) ?
        static_cast<uint32_t>((input.numel() + 1) / 2) : static_cast<uint32_t>(input.numel());
    uint32_t workgroups = (num_work_items + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

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
    std::string shader_name = "hyperbolic";
    if (input.dtype() == DType::Float16) shader_name = "hyperbolic_f16";
    else if (input.dtype() == DType::Float64) shader_name = "hyperbolic_f64";
    auto* pipeline = getPipeline(shader_name, device_id);

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
    if (input.dtype() == DType::Float16) {
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

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

    // For F16, each thread processes 2 elements
    uint32_t num_work_items = (input.dtype() == DType::Float16) ?
        static_cast<uint32_t>((input.numel() + 1) / 2) : static_cast<uint32_t>(input.numel());
    uint32_t workgroups = (num_work_items + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

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

    // Handle empty tensors - no GPU work needed
    if (a.numel() == 0) {
        std::vector<int64_t> output_shape(a_shape.begin(), a_shape.end());
        return Tensor(output_shape, DType::Bool, a.device());
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
        case DType::Float16:
            shader_name = "comparison_f16";
            break;
        default:
            // Float32 uses the default comparison shader
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
    if (a.dtype() == DType::Float16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t a_pairs = (a.numel() + 1) / 2;
        size_t b_pairs = (b.numel() + 1) / 2;
        buffer_size_a = a_pairs * 4;
        buffer_size_b = b_pairs * 4;
        // Output is Bool (uint8_t per element), NOT packed Float16
        // Keep buffer_size_out as output.numel() * output.dtype_size()
    }

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
    insertComputeBarrier(cmdBuffer);

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
        // Fill with identity value based on dtype
        if (input.dtype() == DType::Float64) {
            double* data = result_cpu.data<double>();
            for (int64_t i = 0; i < result_cpu.numel(); i++) {
                data[i] = static_cast<double>(identity_value);
            }
        } else if (input.dtype() == DType::Float16) {
            Float16* data = result_cpu.data<Float16>();
            for (int64_t i = 0; i < result_cpu.numel(); i++) {
                data[i] = Float16(identity_value);
            }
        } else {
            float* data = result_cpu.data<float>();
            for (int64_t i = 0; i < result_cpu.numel(); i++) {
                data[i] = identity_value;
            }
        }
        return result_cpu.to(input.device());
    }

    // BFloat16: upcast to Float32, compute, downcast back
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchReduction(op_name, input_f32, dim, keepdim);
        return result_f32.to(DType::BFloat16);
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
    if (is_float16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    // Synchronize to ensure GPU has completed before reading results
    synchronize(device_id);

    return output;
}

// Helper to check if a 2D tensor is a simple transpose of a contiguous tensor
// Returns true if the tensor's strides indicate a simple row-column swap
static bool isSimpleTranspose2D(const Tensor& t) {
    if (t.ndim() != 2) return false;
    auto strides = t.strides();
    auto shape = t.shape();
    // A transposed 2D contiguous tensor has strides [1, rows] where rows = shape[0]
    // (original was [cols, 1] before transpose)
    return strides[0] == 1 && strides[1] == static_cast<int64_t>(shape[0]);
}

auto VulkanBackend::dispatchMatmul(const Tensor& a, const Tensor& b) -> Tensor {
    // Optimized matmul with proper buffer binding and tiled execution

    // Float16/BFloat16: upcast to Float32 for numerical stability
    // The matmul_f16 shader uses F32 accumulation but outputs F16, which can
    // overflow the F16 range (±65504) for large reduction dimensions (K).
    if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
        DType orig_dtype = a.dtype();
        auto a_f32 = a.to(DType::Float32);
        auto b_f32 = b.to(DType::Float32);
        auto result_f32 = dispatchMatmul(a_f32, b_f32);
        return result_f32.to(orig_dtype);
    }

    // Make A contiguous if needed (A is usually already contiguous)
    Tensor a_contig = a.is_contiguous() ? a : dispatchContiguous(a);

    // For B, check if it's a simple transpose - we can handle that without copying
    // This is common in linear layers: weight.transpose(0, 1)
    // Check if B is a simple transpose (common for linear layers: weight.T)
    // If so, we can use the _bt shader variant instead of making a contiguous copy
    bool b_is_transposed = !b.is_contiguous() && isSimpleTranspose2D(b);
    Tensor b_for_compute = b_is_transposed ? b : (b.is_contiguous() ? b : dispatchContiguous(b));

    auto a_shape = a_contig.shape();
    auto b_shape = b_for_compute.shape();

    // Handle 1D vector × 2D matrix case
    if (a_shape.size() == 1 && b_shape.size() == 2) {
        // Validate dimensions
        if (a_shape[0] != b_shape[0]) {
            throw std::invalid_argument("Incompatible dimensions for vector-matrix matmul");
        }

        // Treat 1D vector as row vector: (N,) -> (1, N)
        // Then matmul becomes: (1, N) × (N, K) = (1, K)
        // Finally squeeze to get (K,)
        Tensor a_2d = a_contig.unsqueeze(0);  // (N,) -> (1, N)
        Tensor result_2d = dispatchMatmul(a_2d, b_for_compute);  // (1, K)

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

    int32_t device_id = a_contig.device().index;

    // Select correct pipeline based on dtype and whether B is transposed
    bool is_float64 = (a_contig.dtype() == DType::Float64);
    bool is_float16 = (a_contig.dtype() == DType::Float16);
    bool is_int32 = (a_contig.dtype() == DType::Int32);

    // For Float64, use optimized shader for larger matrices (>= 64x64)
    // Note: optimized shader doesn't have a _bt variant yet, so don't use it for transposed
    bool use_optimized_f64 = is_float64 && !b_is_transposed && (a_shape[0] >= 64 && b_shape[1] >= 64);

    std::string shader_name;
    if (is_float64) {
        if (use_optimized_f64) {
            shader_name = "matmul_f64_optimized";
        } else {
            shader_name = b_is_transposed ? "matmul_f64_bt" : "matmul_f64";
        }
    } else if (is_float16) {
        shader_name = "matmul_f16";  // TODO: add matmul_f16_bt when needed
    } else if (is_int32) {
        shader_name = "matmul_i32";  // TODO: add matmul_i32_bt when needed
    } else {
        shader_name = b_is_transposed ? "matmul_bt" : "matmul";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {a_shape[0], b_shape[1]};
    Tensor output(out_shape, a_contig.dtype(), a_contig.device());

    // Get VkBuffer handles - for transposed B, we need to access the underlying storage
    VkBuffer buffer_a = getVulkanBuffer(a_contig.data_ptr());
    void* b_data_ptr = b_is_transposed ? const_cast<void*>(b_for_compute.impl_->storage->data())
                                        : b_for_compute.data_ptr();
    VkBuffer buffer_b = getVulkanBuffer(b_data_ptr);
    VkBuffer buffer_c = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_a = a_contig.numel() * a_contig.dtype_size();
    size_t buffer_size_b = b_for_compute.numel() * b_for_compute.dtype_size();
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

    // Tile-based dispatch
    // - Standard shaders: 16x16 workgroups, 16x16 output tiles
    // - Optimized Float64: 16x16 workgroups, 32x32 output tiles (2x2 per thread)
    // - Float16: each thread processes 2 columns
    uint32_t workgroups_x, workgroups_y;
    if (use_optimized_f64) {
        // Optimized F64: 32x32 output tiles
        workgroups_x = (push_constants.N + 31) / 32;
        workgroups_y = (push_constants.M + 31) / 32;
    } else if (is_float16) {
        // Each thread handles 2 adjacent columns
        workgroups_x = ((push_constants.N + 1) / 2 + 15) / 16;
        workgroups_y = (push_constants.M + 15) / 16;
    } else {
        // Standard 16x16 tiles
        workgroups_x = (push_constants.N + 15) / 16;
        workgroups_y = (push_constants.M + 15) / 16;
    }
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, 1);

    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchBmm(const Tensor& a, const Tensor& b) -> Tensor {
    // Batched matrix multiplication: C[b, :, :] = A[b, :, :] @ B[b, :, :]
    // A: (batch, M, K), B: (batch, K, N), C: (batch, M, N)
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    if (a_shape.size() != 3 || b_shape.size() != 3) {
        throw std::invalid_argument("Bmm requires 3D tensors, got " +
            std::to_string(a_shape.size()) + "D and " +
            std::to_string(b_shape.size()) + "D");
    }

    int64_t batch = a_shape[0];
    int64_t M = a_shape[1];
    int64_t K = a_shape[2];
    int64_t N = b_shape[2];

    if (b_shape[0] != batch) {
        throw std::invalid_argument("Bmm batch dimensions must match: " +
            std::to_string(batch) + " vs " + std::to_string(b_shape[0]));
    }
    if (b_shape[1] != K) {
        throw std::invalid_argument("Bmm inner dimensions must match: " +
            std::to_string(K) + " vs " + std::to_string(b_shape[1]));
    }

    // Float16/BFloat16: upcast to Float32 BEFORE making contiguous to avoid
    // strided_copy_f16 shader issues with non-contiguous permuted tensors.
    // The .to(DType::Float32) goes through CPU which handles strides correctly.
    if (a.dtype() == DType::Float16 || a.dtype() == DType::BFloat16) {
        DType orig_dtype = a.dtype();
        Tensor a_f32 = a.to(DType::Float32);
        Tensor b_f32 = b.to(DType::Float32);
        Tensor result_f32 = dispatchBmm(a_f32, b_f32);
        return result_f32.to(orig_dtype);
    }

    int32_t device_id = a.device().index;
    bool is_float64 = (a.dtype() == DType::Float64);
    bool a_contig = a.is_contiguous();
    bool b_contig = b.is_contiguous();

    // Use strided shader when either input is non-contiguous to avoid
    // memory-wasting contiguous copies (saves ~hundreds of MB in attention backward)
    if (!a_contig || !b_contig) {
        std::string shader_name = is_float64 ? "bmm_strided_f64" : "bmm_strided";
        auto* pipeline = getPipeline(shader_name, device_id);

        std::vector<int64_t> out_shape = {batch, M, N};
        Tensor output(out_shape, a.dtype(), a.device());

        // Get VkBuffer handles - for non-contiguous tensors, data_ptr() points
        // to the underlying storage which getVulkanBuffer resolves correctly
        VkBuffer buffer_a = getVulkanBuffer(a.data_ptr());
        VkBuffer buffer_b = getVulkanBuffer(b.data_ptr());
        VkBuffer buffer_c = getVulkanBuffer(output.data_ptr());

        // Buffer sizes: for non-contiguous tensors, compute the extent
        // (max addressable byte) from strides rather than numel * dtype_size
        auto a_strides = a.strides();
        auto b_strides = b.strides();
        size_t dtype_sz = a.dtype_size();

        // Extent = (shape[i]-1)*stride[i] for each dim + 1 element
        auto compute_extent = [&](const Tensor& t) -> size_t {
            auto sh = t.shape();
            auto st = t.strides();
            size_t extent = 1;
            for (size_t i = 0; i < sh.size(); i++) {
                if (sh[i] > 1) {
                    extent += (sh[i] - 1) * std::abs(st[i]);
                }
            }
            return extent * t.dtype_size();
        };

        size_t buffer_size_a = a_contig ? a.numel() * dtype_sz : compute_extent(a);
        size_t buffer_size_b = b_contig ? b.numel() * dtype_sz : compute_extent(b);
        size_t buffer_size_c = output.numel() * dtype_sz;

        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
            {0, buffer_a}, {1, buffer_b}, {2, buffer_c}
        };
        std::vector<size_t> sizes = {buffer_size_a, buffer_size_b, buffer_size_c};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

        // Push constants with stride information
        struct StridedPushConstants {
            uint32_t batch;
            uint32_t M;
            uint32_t N;
            uint32_t K;
            uint32_t a_stride0;
            uint32_t a_stride1;
            uint32_t a_stride2;
            uint32_t b_stride0;
            uint32_t b_stride1;
            uint32_t b_stride2;
        } push_constants;

        push_constants.batch = static_cast<uint32_t>(batch);
        push_constants.M = static_cast<uint32_t>(M);
        push_constants.N = static_cast<uint32_t>(N);
        push_constants.K = static_cast<uint32_t>(K);
        push_constants.a_stride0 = static_cast<uint32_t>(a_strides[0]);
        push_constants.a_stride1 = static_cast<uint32_t>(a_strides[1]);
        push_constants.a_stride2 = static_cast<uint32_t>(a_strides[2]);
        push_constants.b_stride0 = static_cast<uint32_t>(b_strides[0]);
        push_constants.b_stride1 = static_cast<uint32_t>(b_strides[1]);
        push_constants.b_stride2 = static_cast<uint32_t>(b_strides[2]);

        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(StridedPushConstants), &push_constants);

        uint32_t workgroups_x = (push_constants.N + 15) / 16;
        uint32_t workgroups_y = (push_constants.M + 15) / 16;
        uint32_t workgroups_z = push_constants.batch;
        vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);

        insertComputeBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    // Contiguous fast path - use original shaders (no stride overhead)
    std::string shader_name = is_float64 ? "bmm_f64" : "bmm";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, M, N};
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

    // Set push constants for batched matrix dimensions
    struct PushConstants {
        uint32_t batch;  // batch size
        uint32_t M;      // rows of A
        uint32_t N;      // cols of B
        uint32_t K;      // cols of A / rows of B
    } push_constants;

    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.M = static_cast<uint32_t>(M);
    push_constants.N = static_cast<uint32_t>(N);
    push_constants.K = static_cast<uint32_t>(K);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch: each workgroup handles 16x16 output elements
    // Z dimension is batch to avoid cross-batch tiling errors
    uint32_t workgroups_x = (push_constants.N + 15) / 16;
    uint32_t workgroups_y = (push_constants.M + 15) / 16;
    uint32_t workgroups_z = push_constants.batch;
    vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);

    insertComputeBarrier(cmdBuffer);
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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

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
    const std::vector<int64_t>& input_shape,
    int64_t groups) -> Tensor {

    // Float16: upcast to Float32, compute, downcast back
    if (grad_output.dtype() == DType::Float16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        auto result = dispatchConv2dBackwardInput(grad_f32, weight_f32, stride, padding, dilation, input_shape, groups);
        return result.to(DType::Float16);
    }

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

    // Select shader based on dtype
    std::string shader_name = "conv2d_backward_input";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv2d_backward_input_f64";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

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

    // Set push constants (groups support for depthwise/grouped convolutions)
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
        uint32_t groups;
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
    push_constants.groups = static_cast<uint32_t>(groups);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (256 threads per workgroup as defined in shader)
    int64_t total_elements = batch * channels_in * height_in * width_in;
    uint32_t workgroups = static_cast<uint32_t>((total_elements + 255) / 256);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

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
    const std::vector<int64_t>& weight_shape,
    int64_t groups) -> Tensor {

    // Float16: upcast to Float32, compute, downcast back
    if (grad_output.dtype() == DType::Float16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto input_f32 = input.to(DType::Float32);
        auto result = dispatchConv2dBackwardWeight(grad_f32, input_f32, stride, padding, dilation, weight_shape, groups);
        return result.to(DType::Float16);
    }

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

    // Select shader based on dtype
    std::string shader_name = "conv2d_backward_weight";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv2d_backward_weight_f64";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

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

    // Set push constants (groups support for depthwise/grouped convolutions)
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
        uint32_t groups;
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
    push_constants.groups = static_cast<uint32_t>(groups);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups (256 threads per workgroup as defined in shader)
    // Weight shape: (C_out, C_in/groups, K_h, K_w) - total elements is the product of these
    int64_t in_channels_per_group = channels_in / groups;
    int64_t total_weight_elements = channels_out * in_channels_per_group * kernel_h * kernel_w;
    uint32_t workgroups = static_cast<uint32_t>((total_weight_elements + 255) / 256);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

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
    // Float16: upcast to Float32, compute, downcast back
    if (grad_output.dtype() == DType::Float16) {
        auto grad_f32 = grad_output.to(DType::Float32);
        auto result = dispatchConv2dBackwardBias(grad_f32);
        return result.to(DType::Float16);
    }

    // Extract dimensions
    auto grad_shape = grad_output.shape();

    int64_t batch = grad_shape[0];
    int64_t channels_out = grad_shape[1];
    int64_t height_out = grad_shape[2];
    int64_t width_out = grad_shape[3];

    int32_t device_id = grad_output.device().index;

    // Select shader based on dtype
    std::string shader_name = "conv2d_backward_bias";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "conv2d_backward_bias_f64";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

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
    // Use pooling_forward_with_indices shader which has 3 bindings and matching push constants
    auto* pipeline = getPipeline("pooling_forward_with_indices", device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_height, out_width};
    Tensor output(out_shape, input.dtype(), input.device());
    Tensor indices(out_shape, DType::Int32, input.device());  // Use Int32 for Vulkan

    // Push constants matching pooling_forward_with_indices.comp
    struct PushConstants {
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
    } push_constants;

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

    // Dispatch with (16, 16) local size shader - x = out_width, y = out_height, z = channels
    // Process each batch separately
    for (int64_t b = 0; b < batch; ++b) {
        uint32_t workgroups_x = (out_width + 15) / 16;
        uint32_t workgroups_y = (out_height + 15) / 16;
        uint32_t workgroups_z = static_cast<uint32_t>(channels);
        vkCmdDispatch(cmdBuffer, workgroups_x, workgroups_y, workgroups_z);
    }

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

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
    std::string shader_name = "avg_pool2d";
    if (input.dtype() == DType::Float64) {
        shader_name = "avg_pool2d_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "avg_pool2d_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_height, out_width};
    Tensor output(out_shape, input.dtype(), input.device());

    // Push constants matching avg_pool2d.comp shader
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
    push_constants.count_include_pad = 0;  // Default: don't include padding in average

    // Get VkBuffer handles
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t input_size = input.numel() * input.dtype_size();
    size_t output_size = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_input},
        {1, buffer_output}
    };
    std::vector<size_t> sizes = {input_size, output_size};

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

    // Shader uses local_size_x = 256, processing elements linearly
    uint32_t workgroups = (push_constants.n_elements + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAdaptiveMaxPool2d(const Tensor& input, int64_t out_h, int64_t out_w) -> std::pair<Tensor, Tensor> {
    auto cont_input = input.contiguous();
    auto input_shape = cont_input.shape();
    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];

    int32_t device_id = cont_input.device().index;
    auto* pipeline = getPipeline("adaptive_pooling", device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_h, out_w};
    Tensor output(out_shape, cont_input.dtype(), cont_input.device());
    Tensor indices(out_shape, DType::Int32, cont_input.device());

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
    VkBuffer buffer_input = getVulkanBuffer(cont_input.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());
    VkBuffer buffer_indices = getVulkanBuffer(indices.data_ptr());

    // Calculate buffer sizes
    size_t input_size = cont_input.numel() * cont_input.dtype_size();
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
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, indices};
}

auto VulkanBackend::dispatchAdaptiveAvgPool2d(const Tensor& input, int64_t out_h, int64_t out_w) -> Tensor {
    auto cont_input = input.contiguous();
    auto input_shape = cont_input.shape();
    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_h = input_shape[2];
    int64_t in_w = input_shape[3];

    int32_t device_id = cont_input.device().index;

    // Select shader based on dtype
    std::string shader_name;
    if (cont_input.dtype() == DType::Float64) {
        shader_name = "adaptive_pooling_f64";
    } else if (cont_input.dtype() == DType::Float16) {
        shader_name = "adaptive_pooling_f16";
    } else {
        shader_name = "adaptive_pooling";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape = {batch, channels, out_h, out_w};
    Tensor output(out_shape, cont_input.dtype(), cont_input.device());

    // Create dummy indices buffer for avg pool (shader requires it)
    Tensor dummy_indices(out_shape, DType::Int32, cont_input.device());

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
    VkBuffer buffer_input = getVulkanBuffer(cont_input.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());
    VkBuffer buffer_indices = getVulkanBuffer(dummy_indices.data_ptr());

    // Calculate buffer sizes
    size_t input_size = cont_input.numel() * cont_input.dtype_size();
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
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchAdaptiveAvgPool2dBackward(const Tensor& grad_output, int64_t H_in, int64_t W_in) -> Tensor {
    auto cont_grad = grad_output.contiguous();
    auto grad_shape = cont_grad.shape();
    int64_t batch = grad_shape[0];
    int64_t channels = grad_shape[1];
    int64_t H_out = grad_shape[2];
    int64_t W_out = grad_shape[3];

    int32_t device_id = cont_grad.device().index;

    // Select shader based on dtype
    // Float64 and Float16 versions don't use atomics - they iterate over input positions
    std::string shader_name;
    bool is_float64 = (cont_grad.dtype() == DType::Float64);
    bool is_float16 = (cont_grad.dtype() == DType::Float16);
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
    Tensor grad_input(out_shape, cont_grad.dtype(), cont_grad.device());

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
    VkBuffer buffer_grad_output = getVulkanBuffer(cont_grad.data_ptr());
    VkBuffer buffer_grad_input = getVulkanBuffer(grad_input.data_ptr());

    // Calculate buffer sizes
    size_t grad_output_size = cont_grad.numel() * cont_grad.dtype_size();
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
    insertComputeBarrier(cmdBuffer);

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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchBatchNorm2dBackward(const Tensor& grad_out, const Tensor& input,
                                                 const Tensor& mean, const Tensor& var,
                                                 const Tensor* gamma, float epsilon)
                                                 -> std::tuple<Tensor, Tensor, Tensor> {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("batchnorm2d_backward requires 4D input (N, C, H, W)");
    }

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];
    int64_t spatial_size = height * width;
    int64_t n_elements = input.numel();

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "batchnorm2d_backward";
    if (input.dtype() == DType::Float64) {
        shader_name = "batchnorm2d_backward_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "batchnorm2d_backward_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensors
    std::vector<int64_t> grad_in_shape(input_shape.begin(), input_shape.end());
    Tensor grad_input(grad_in_shape, input.dtype(), input.device());

    // For Float16, the backward shader accumulates grad_gamma/grad_beta in Float32
    // (mean/var are also Float32 for F16 input)
    DType stats_dtype = (input.dtype() == DType::Float16) ? DType::Float32 : input.dtype();
    std::vector<int64_t> param_shape = {channels};
    // Initialize grad_gamma and grad_beta to zeros since shader uses atomicAdd
    Tensor grad_gamma = dispatchZeros(param_shape, stats_dtype, input.device());
    Tensor grad_beta = dispatchZeros(param_shape, stats_dtype, input.device());

    // For Float16 input, cast gamma to Float32 if needed (shader expects Float32 stats)
    Tensor gamma_f32;
    const Tensor* gamma_effective = gamma;
    if (gamma && input.dtype() == DType::Float16 && gamma->dtype() == DType::Float16) {
        gamma_f32 = gamma->to(DType::Float32);
        gamma_effective = &gamma_f32;
    }

    // Get VkBuffer handles
    VkBuffer buffer_grad_out = getVulkanBuffer(grad_out.data_ptr());
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_mean = getVulkanBuffer(mean.data_ptr());
    VkBuffer buffer_var = getVulkanBuffer(var.data_ptr());
    VkBuffer buffer_grad_input = getVulkanBuffer(grad_input.data_ptr());
    VkBuffer buffer_grad_gamma = getVulkanBuffer(grad_gamma.data_ptr());
    VkBuffer buffer_grad_beta = getVulkanBuffer(grad_beta.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_input = n_elements * input.dtype_size();
    if (input.dtype() == DType::Float16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t input_pairs = (n_elements + 1) / 2;
        buffer_size_input = input_pairs * 4;
    }
    // Statistics (mean, var, gamma, grad_gamma, grad_beta) use stats_dtype
    size_t buffer_size_channel = channels * mean.dtype_size();

    // Set up descriptor bindings
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_grad_out},    // grad_output
        {1, buffer_input},       // input
        {2, buffer_mean},        // mean
        {3, buffer_var},         // variance
    };
    std::vector<size_t> sizes = {
        buffer_size_input,  // grad_output
        buffer_size_input,  // input
        buffer_size_channel, // mean
        buffer_size_channel, // variance
    };

    // Handle optional gamma
    if (gamma_effective) {
        VkBuffer buffer_gamma = getVulkanBuffer(gamma_effective->data_ptr());
        bindings.push_back({4, buffer_gamma});
        sizes.push_back(gamma_effective->numel() * gamma_effective->dtype_size());
    } else {
        // Use dummy buffer for gamma binding
        bindings.push_back({4, buffer_mean});
        sizes.push_back(buffer_size_channel);
    }

    // Add output buffers
    bindings.push_back({5, buffer_grad_input});
    bindings.push_back({6, buffer_grad_gamma});
    bindings.push_back({7, buffer_grad_beta});
    sizes.push_back(buffer_size_input);   // grad_input
    sizes.push_back(buffer_size_channel); // grad_gamma
    sizes.push_back(buffer_size_channel); // grad_beta

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Push constants
    struct PushConstants {
        uint32_t n_elements;
        uint32_t batch;
        uint32_t channels;
        uint32_t spatial_size;
        float eps;
        uint32_t has_gamma;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(n_elements);
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.spatial_size = static_cast<uint32_t>(spatial_size);
    push_constants.eps = epsilon;
    push_constants.has_gamma = (gamma != nullptr) ? 1 : 0;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // For Float16, each thread processes a word (2 elements), so dispatch half as many threads
    uint64_t dispatch_count = n_elements;
    if (input.dtype() == DType::Float16) {
        dispatch_count = (n_elements + 1) / 2;  // number of words
    }
    uint32_t workgroups = (dispatch_count + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

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

    // BFloat16: upcast to Float32, compute, downcast back
    if (input.dtype() == DType::BFloat16) {
        auto in_f32 = input.to(DType::Float32);
        auto mean_f32 = mean.to(DType::Float32);
        auto var_f32 = var.to(DType::Float32);
        Tensor gamma_f32, beta_f32;
        const Tensor* g_ptr = nullptr;
        const Tensor* b_ptr = nullptr;
        if (gamma && beta) {
            gamma_f32 = gamma->to(DType::Float32);
            beta_f32 = beta->to(DType::Float32);
            g_ptr = &gamma_f32;
            b_ptr = &beta_f32;
        }
        auto result_f32 = dispatchBatchNorm2dForward(in_f32, mean_f32, var_f32, g_ptr, b_ptr, epsilon);
        return result_f32.to(DType::BFloat16);
    }

    int32_t device_id = input.device().index;

    // Select shader based on dtype
    std::string shader_name = "batchnorm2d_forward";
    if (input.dtype() == DType::Float64) {
        shader_name = "batchnorm2d_forward_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "batchnorm2d_forward_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // For Float16 input, the shader expects mean/var as Float32 for numerical stability
    // Keep converted tensors alive in this scope so their buffers remain valid
    Tensor mean_f32, var_f32;
    const Tensor* mean_ptr = &mean;
    const Tensor* var_ptr = &var;
    if (input.dtype() == DType::Float16 && mean.dtype() == DType::Float16) {
        mean_f32 = mean.to(DType::Float32);
        var_f32 = var.to(DType::Float32);
        mean_ptr = &mean_f32;
        var_ptr = &var_f32;
    }

    // Get VkBuffer handles
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_mean = getVulkanBuffer(mean_ptr->data_ptr());
    VkBuffer buffer_var = getVulkanBuffer(var_ptr->data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_mean = mean_ptr->numel() * mean_ptr->dtype_size();
    size_t buffer_size_var = var_ptr->numel() * var_ptr->dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();
    if (input.dtype() == DType::Float16) {
        // Round up input/output to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t input_pairs = (input.numel() + 1) / 2;
        size_t output_pairs = (output.numel() + 1) / 2;
        buffer_size_input = input_pairs * 4;
        buffer_size_output = output_pairs * 4;
    }

    // Allocate and write descriptor set
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_input},   // input
        {1, buffer_mean},    // mean
        {2, buffer_var},     // variance
    };
    std::vector<size_t> sizes = {buffer_size_input, buffer_size_mean, buffer_size_var};

    // Add optional gamma and beta buffers
    // Keep cast tensors alive in this scope so their buffers remain valid
    Tensor gamma_f32, beta_f32;
    if (gamma && beta) {
        const Tensor* gamma_ptr = gamma;
        const Tensor* beta_ptr = beta;

        // For Float16 input, the shader expects gamma/beta as Float32 for numerical stability
        if (input.dtype() == DType::Float16 && gamma->dtype() == DType::Float16) {
            gamma_f32 = gamma->to(DType::Float32);
            beta_f32 = beta->to(DType::Float32);
            gamma_ptr = &gamma_f32;
            beta_ptr = &beta_f32;
        }

        VkBuffer buffer_gamma = getVulkanBuffer(gamma_ptr->data_ptr());
        VkBuffer buffer_beta = getVulkanBuffer(beta_ptr->data_ptr());
        size_t buffer_size_gamma = gamma_ptr->numel() * gamma_ptr->dtype_size();
        size_t buffer_size_beta = beta_ptr->numel() * beta_ptr->dtype_size();

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
    // For Float16, each thread processes a word (2 elements), so dispatch half as many threads
    uint64_t dispatch_count = input.numel();
    if (input.dtype() == DType::Float16) {
        dispatch_count = (input.numel() + 1) / 2;  // number of words
    }
    uint32_t workgroups = static_cast<uint32_t>((dispatch_count + 255) / 256);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

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

    // Select shader based on dtype
    std::string shader_name = "batchnorm2d_mean_var";
    if (input.dtype() == DType::Float64) {
        shader_name = "batchnorm2d_mean_var_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "batchnorm2d_mean_var_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensors - statistics are always Float32 (for F16 inputs) or same dtype
    // For F16: accumulation in Float32 for numerical stability, output as Float32
    DType stats_dtype = (input.dtype() == DType::Float16) ? DType::Float32 : input.dtype();
    std::vector<int64_t> stats_shape = {channels};
    Tensor mean(stats_shape, stats_dtype, input.device());
    Tensor variance(stats_shape, stats_dtype, input.device());
    Tensor temp_sum(stats_shape, stats_dtype, input.device());

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
    if (input.dtype() == DType::Float16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t input_pairs = (input.numel() + 1) / 2;
        buffer_size_input = input_pairs * 4;
    }
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

        // Add memory barrier
        insertComputeBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
    }

    // Normalize mean: temp_sum / normalizer -> mean
    {
        Tensor normalizer_tensor(stats_shape, stats_dtype, input.device());
        normalizer_tensor = dispatchFill(normalizer_tensor, static_cast<float>(normalizer));
        mean = dispatchBinaryOp("div", temp_sum, normalizer_tensor);
    }

    // Update buffer_mean to point to the newly computed mean tensor
    buffer_mean = getVulkanBuffer(mean.data_ptr());
    buffer_size_stats = channels * mean.dtype_size();

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

        // Add memory barrier
        insertComputeBarrier(cmdBuffer);

        endSingleTimeCommands(cmdBuffer, device_id);
    }

    // Normalize variance: var_data / normalizer -> variance
    {
        Tensor normalizer_tensor(stats_shape, stats_dtype, input.device());
        normalizer_tensor = dispatchFill(normalizer_tensor, static_cast<float>(normalizer));
        variance = dispatchBinaryOp("div", variance, normalizer_tensor);
    }

    return {mean, variance};
}

auto VulkanBackend::dispatchLayerNorm(const Tensor& input, int64_t normalized_shape,
                                      const Tensor* gamma, const Tensor* beta, float epsilon) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    // For Float16/BFloat16, upcast to Float32 for numerical stability
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        Tensor gamma_f32, beta_f32;
        const Tensor* gamma_ptr = nullptr;
        const Tensor* beta_ptr = nullptr;
        if (gamma && beta) {
            gamma_f32 = gamma->to(DType::Float32);
            beta_f32 = beta->to(DType::Float32);
            gamma_ptr = &gamma_f32;
            beta_ptr = &beta_f32;
        }
        auto result_f32 = dispatchLayerNorm(input_f32, normalized_shape, gamma_ptr, beta_ptr, epsilon);
        return result_f32.to(orig_dtype);
    }

    // Select correct pipeline based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    std::string shader_name;
    if (is_float64) {
        shader_name = "layer_norm_f64";
    } else {
        shader_name = "layer_norm";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    int64_t batch_size = input.numel() / normalized_shape;
    bool has_affine = (gamma != nullptr && beta != nullptr);

    // Get VkBuffer handles
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());

    size_t elem_size = input.dtype_size();
    size_t input_buffer_size = input.numel() * elem_size;
    size_t output_buffer_size = output.numel() * elem_size;
    size_t norm_buffer_size = normalized_shape * elem_size;

    // Build buffer bindings: input(0), output(1), gamma(2), beta(3)
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_input},
        {1, buffer_output}
    };
    std::vector<size_t> sizes = {input_buffer_size, output_buffer_size};

    if (has_affine) {
        VkBuffer buffer_gamma = getVulkanBuffer(gamma->data_ptr());
        VkBuffer buffer_beta = getVulkanBuffer(beta->data_ptr());
        bindings.push_back({2, buffer_gamma});
        bindings.push_back({3, buffer_beta});
        sizes.push_back(norm_buffer_size);
        sizes.push_back(norm_buffer_size);
    } else {
        // Bind dummy buffers (use output buffer as placeholder for unused bindings)
        bindings.push_back({2, buffer_output});
        bindings.push_back({3, buffer_output});
        sizes.push_back(output_buffer_size);
        sizes.push_back(output_buffer_size);
    }

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Push constants: batch_size, normalized_shape, epsilon, affine
    struct PushConstants {
        uint32_t batch_size;
        uint32_t normalized_shape;
        float epsilon;
        uint32_t affine;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(batch_size);
    push_constants.normalized_shape = static_cast<uint32_t>(normalized_shape);
    push_constants.epsilon = epsilon;
    push_constants.affine = has_affine ? 1u : 0u;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One workgroup per batch element
    uint32_t workgroups = static_cast<uint32_t>(batch_size);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchGroupNorm(const Tensor& input, int64_t num_groups,
                                      const Tensor* gamma, const Tensor* beta, float epsilon) -> std::vector<Tensor> {
    auto input_shape = input.shape();
    if (input_shape.size() < 2) {
        throw std::invalid_argument("group_norm requires at least 2D input");
    }

    // For non-Float32 types, convert to Float32, compute, convert back
    if (input.dtype() != DType::Float32) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        Tensor gamma_f32, beta_f32;
        const Tensor* gamma_ptr = nullptr;
        const Tensor* beta_ptr = nullptr;
        if (gamma && beta) {
            gamma_f32 = gamma->to(DType::Float32);
            beta_f32 = beta->to(DType::Float32);
            gamma_ptr = &gamma_f32;
            beta_ptr = &beta_f32;
        }
        auto results = dispatchGroupNorm(input_f32, num_groups, gamma_ptr, beta_ptr, epsilon);
        // Output in original dtype, mean/inv_std stay Float32 (used for backward)
        return {results[0].to(orig_dtype), results[1], results[2]};
    }

    int32_t device_id = input.device().index;

    // GroupNorm shader operates on Float32
    std::string shader_name = "group_norm";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = (input_shape.size() > 2) ? input_shape[2] : 1;
    int64_t W = (input_shape.size() > 3) ? input_shape[3] : 1;
    // For >4D, fold extra spatial dims into W
    for (size_t i = 4; i < input_shape.size(); ++i) {
        W *= input_shape[i];
    }

    bool has_affine = (gamma != nullptr && beta != nullptr);

    // Create output tensors
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());
    Tensor mean_out({N, num_groups}, DType::Float32, input.device());
    Tensor inv_std_out({N, num_groups}, DType::Float32, input.device());

    size_t elem_size = input.dtype_size();

    // Get VkBuffer handles
    VkBuffer buf_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buf_output = getVulkanBuffer(output.data_ptr());
    VkBuffer buf_mean = getVulkanBuffer(mean_out.data_ptr());
    VkBuffer buf_inv_std = getVulkanBuffer(inv_std_out.data_ptr());

    size_t input_buf_size = input.numel() * elem_size;
    size_t output_buf_size = output.numel() * elem_size;
    size_t stats_buf_size = N * num_groups * sizeof(float);

    // Bindings: input(0), output(1), gamma(2), beta(3), mean(4), inv_std(5)
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buf_input},
        {1, buf_output},
    };
    std::vector<size_t> sizes = {input_buf_size, output_buf_size};

    if (has_affine) {
        VkBuffer buf_gamma = getVulkanBuffer(gamma->data_ptr());
        VkBuffer buf_beta = getVulkanBuffer(beta->data_ptr());
        bindings.push_back({2, buf_gamma});
        bindings.push_back({3, buf_beta});
        sizes.push_back(C * elem_size);
        sizes.push_back(C * elem_size);
    } else {
        // Bind dummy buffers for unused bindings
        bindings.push_back({2, buf_output});
        bindings.push_back({3, buf_output});
        sizes.push_back(output_buf_size);
        sizes.push_back(output_buf_size);
    }

    bindings.push_back({4, buf_mean});
    bindings.push_back({5, buf_inv_std});
    sizes.push_back(stats_buf_size);
    sizes.push_back(stats_buf_size);

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t num_channels;
        uint32_t height;
        uint32_t width;
        uint32_t num_groups;
        float epsilon;
        uint32_t affine;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(N);
    push_constants.num_channels = static_cast<uint32_t>(C);
    push_constants.height = static_cast<uint32_t>(H);
    push_constants.width = static_cast<uint32_t>(W);
    push_constants.num_groups = static_cast<uint32_t>(num_groups);
    push_constants.epsilon = epsilon;
    push_constants.affine = has_affine ? 1u : 0u;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One workgroup per (batch, group) pair
    uint32_t workgroups = static_cast<uint32_t>(N * num_groups);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, mean_out, inv_std_out};
}

// LayerNorm Backward - GPU implementation
auto VulkanBackend::dispatchLayerNormBackward(const Tensor& grad_output, const Tensor& input,
                                               const Tensor& mean, const Tensor& rstd,
                                               const Tensor* weight, int64_t normalized_shape)
                                               -> std::tuple<Tensor, Tensor, Tensor> {
    int32_t device_id = input.device().index;

    // For Float16/BFloat16, upcast to Float32 for numerical stability
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto in_f32 = input.to(DType::Float32);
        auto mean_f32 = mean.to(DType::Float32);
        auto rstd_f32 = rstd.to(DType::Float32);
        Tensor w_f32;
        const Tensor* w_ptr = nullptr;
        if (weight) {
            w_f32 = weight->to(DType::Float32);
            w_ptr = &w_f32;
        }
        auto [gi, gw, gb] = dispatchLayerNormBackward(go_f32, in_f32, mean_f32, rstd_f32, w_ptr, normalized_shape);
        return {gi.to(orig_dtype), gw.to(orig_dtype), gb.to(orig_dtype)};
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    std::string shader_name = is_float64 ? "layer_norm_backward_f64" : "layer_norm_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t batch_size = input.numel() / normalized_shape;
    bool has_affine = (weight != nullptr);

    // Create output tensors
    Tensor grad_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                      input.dtype(), input.device());
    Tensor grad_weight = dispatchZeros({normalized_shape}, input.dtype(), input.device());
    Tensor grad_bias = dispatchZeros({normalized_shape}, input.dtype(), input.device());

    size_t elem_size = input.dtype_size();

    // Get VkBuffer handles
    VkBuffer buf_grad_out = getVulkanBuffer(grad_output.data_ptr());
    VkBuffer buf_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buf_mean = getVulkanBuffer(mean.data_ptr());
    VkBuffer buf_rstd = getVulkanBuffer(rstd.data_ptr());
    VkBuffer buf_grad_input = getVulkanBuffer(grad_input.data_ptr());
    VkBuffer buf_grad_weight = getVulkanBuffer(grad_weight.data_ptr());
    VkBuffer buf_grad_bias = getVulkanBuffer(grad_bias.data_ptr());

    size_t input_buf_size = input.numel() * elem_size;
    size_t stats_buf_size = batch_size * elem_size;
    size_t norm_buf_size = normalized_shape * elem_size;

    // Bindings: grad_output(0), input(1), mean(2), rstd(3), weight(4), grad_input(5), grad_weight(6), grad_bias(7)
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buf_grad_out},
        {1, buf_input},
        {2, buf_mean},
        {3, buf_rstd},
    };
    std::vector<size_t> sizes = {input_buf_size, input_buf_size, stats_buf_size, stats_buf_size};

    if (has_affine) {
        VkBuffer buf_weight = getVulkanBuffer(weight->data_ptr());
        bindings.push_back({4, buf_weight});
        sizes.push_back(norm_buf_size);
    } else {
        bindings.push_back({4, buf_grad_input});  // dummy
        sizes.push_back(input_buf_size);
    }

    bindings.push_back({5, buf_grad_input});
    bindings.push_back({6, buf_grad_weight});
    bindings.push_back({7, buf_grad_bias});
    sizes.push_back(input_buf_size);
    sizes.push_back(norm_buf_size);
    sizes.push_back(norm_buf_size);

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t normalized_shape;
        uint32_t affine;
        uint32_t padding;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(batch_size);
    push_constants.normalized_shape = static_cast<uint32_t>(normalized_shape);
    push_constants.affine = has_affine ? 1u : 0u;
    push_constants.padding = 0;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One workgroup per batch element
    uint32_t workgroups = static_cast<uint32_t>(batch_size);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {grad_input, grad_weight, grad_bias};
}

// GroupNorm Backward - GPU implementation
auto VulkanBackend::dispatchGroupNormBackward(const Tensor& grad_output, const Tensor& input,
                                               const Tensor& mean, const Tensor& rstd,
                                               const Tensor* weight, int64_t num_groups)
                                               -> std::tuple<Tensor, Tensor, Tensor> {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("group_norm_backward requires 4D input (N, C, H, W)");
    }

    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = input_shape[2];
    int64_t W = input_shape[3];
    int32_t device_id = input.device().index;

    // For Float16/BFloat16, upcast to Float32
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto in_f32 = input.to(DType::Float32);
        auto mean_f32 = mean.to(DType::Float32);
        auto rstd_f32 = rstd.to(DType::Float32);
        Tensor w_f32;
        const Tensor* w_ptr = nullptr;
        if (weight) {
            w_f32 = weight->to(DType::Float32);
            w_ptr = &w_f32;
        }
        auto [gi, gw, gb] = dispatchGroupNormBackward(go_f32, in_f32, mean_f32, rstd_f32, w_ptr, num_groups);
        return {gi.to(orig_dtype), gw.to(orig_dtype), gb.to(orig_dtype)};
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    std::string shader_name = is_float64 ? "group_norm_backward_f64" : "group_norm_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    bool has_affine = (weight != nullptr);

    // Create output tensors
    Tensor grad_input(std::vector<int64_t>(input_shape.begin(), input_shape.end()),
                      input.dtype(), input.device());
    Tensor grad_weight = dispatchZeros({C}, input.dtype(), input.device());
    Tensor grad_bias = dispatchZeros({C}, input.dtype(), input.device());

    size_t elem_size = input.dtype_size();
    size_t input_buf_size = input.numel() * elem_size;
    size_t stats_buf_size = N * num_groups * elem_size;
    size_t channel_buf_size = C * elem_size;

    VkBuffer buf_grad_out = getVulkanBuffer(grad_output.data_ptr());
    VkBuffer buf_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buf_mean = getVulkanBuffer(mean.data_ptr());
    VkBuffer buf_rstd = getVulkanBuffer(rstd.data_ptr());
    VkBuffer buf_grad_input = getVulkanBuffer(grad_input.data_ptr());
    VkBuffer buf_grad_weight = getVulkanBuffer(grad_weight.data_ptr());
    VkBuffer buf_grad_bias = getVulkanBuffer(grad_bias.data_ptr());

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buf_grad_out},
        {1, buf_input},
        {2, buf_mean},
        {3, buf_rstd},
    };
    std::vector<size_t> sizes = {input_buf_size, input_buf_size, stats_buf_size, stats_buf_size};

    if (has_affine) {
        VkBuffer buf_weight = getVulkanBuffer(weight->data_ptr());
        bindings.push_back({4, buf_weight});
        sizes.push_back(channel_buf_size);
    } else {
        bindings.push_back({4, buf_grad_input});  // dummy
        sizes.push_back(input_buf_size);
    }

    bindings.push_back({5, buf_grad_input});
    bindings.push_back({6, buf_grad_weight});
    bindings.push_back({7, buf_grad_bias});
    sizes.push_back(input_buf_size);
    sizes.push_back(channel_buf_size);
    sizes.push_back(channel_buf_size);

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t num_channels;
        uint32_t height;
        uint32_t width;
        uint32_t num_groups;
        float epsilon;
        uint32_t affine;
        uint32_t padding;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(N);
    push_constants.num_channels = static_cast<uint32_t>(C);
    push_constants.height = static_cast<uint32_t>(H);
    push_constants.width = static_cast<uint32_t>(W);
    push_constants.num_groups = static_cast<uint32_t>(num_groups);
    push_constants.epsilon = 0.0f;  // not used in backward, but part of push constant layout
    push_constants.affine = has_affine ? 1u : 0u;
    push_constants.padding = 0;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One workgroup per (batch, group) pair
    uint32_t workgroups = static_cast<uint32_t>(N * num_groups);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {grad_input, grad_weight, grad_bias};
}

// Embedding Backward - GPU implementation
auto VulkanBackend::dispatchEmbeddingBackward(const Tensor& grad_output, const Tensor& indices,
                                                int64_t num_embeddings, int64_t embedding_dim) -> Tensor {
    int32_t device_id = grad_output.device().index;
    int64_t num_indices = indices.numel();

    // For Float16/BFloat16, upcast to Float32
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        DType orig_dtype = grad_output.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto result_f32 = dispatchEmbeddingBackward(go_f32, indices, num_embeddings, embedding_dim);
        return result_f32.to(orig_dtype);
    }

    bool is_float64 = (grad_output.dtype() == DType::Float64);
    std::string shader_name = is_float64 ? "embedding_backward_f64" : "embedding_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Output: grad_weight of shape [num_embeddings, embedding_dim], initialized to zero
    Tensor grad_weight = dispatchZeros({num_embeddings, embedding_dim}, grad_output.dtype(), grad_output.device());

    size_t elem_size = grad_output.dtype_size();

    VkBuffer buf_grad_out = getVulkanBuffer(grad_output.data_ptr());
    VkBuffer buf_indices = getVulkanBuffer(indices.data_ptr());
    VkBuffer buf_grad_weight = getVulkanBuffer(grad_weight.data_ptr());

    size_t grad_out_size = grad_output.numel() * elem_size;
    size_t indices_size = num_indices * sizeof(int32_t);  // shader uses int (32-bit)
    size_t grad_weight_size = num_embeddings * embedding_dim * elem_size;

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buf_grad_out},
        {1, buf_indices},
        {2, buf_grad_weight},
    };
    std::vector<size_t> sizes = {grad_out_size, indices_size, grad_weight_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t num_indices;
        uint32_t embedding_dim;
        uint32_t num_embeddings;
    } push_constants;

    push_constants.num_indices = static_cast<uint32_t>(num_indices);
    push_constants.embedding_dim = static_cast<uint32_t>(embedding_dim);
    push_constants.num_embeddings = static_cast<uint32_t>(num_embeddings);

    // Total threads = num_indices * embedding_dim (one per element)
    uint64_t total_threads = static_cast<uint64_t>(num_indices) * embedding_dim;
    uint32_t workgroups = static_cast<uint32_t>((total_threads + 255) / 256);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_weight;
}

// RMSNorm Forward - GPU implementation
auto VulkanBackend::dispatchRMSNorm(const Tensor& input, const Tensor& weight,
                                     int64_t normalized_shape, float epsilon) -> std::pair<Tensor, Tensor> {
    int32_t device_id = input.device().index;

    // For Float16/BFloat16, upcast to Float32
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        auto in_f32 = input.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        auto [out_f32, rrms_f32] = dispatchRMSNorm(in_f32, w_f32, normalized_shape, epsilon);
        return {out_f32.to(orig_dtype), rrms_f32};  // rrms stays F32 for backward
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    std::string shader_name = is_float64 ? "rms_norm_f64" : "rms_norm";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t batch_size = input.numel() / normalized_shape;

    // Output tensor same shape as input
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());
    // rrms tensor: one value per batch element (always same dtype as input for F64, F32 otherwise)
    DType rrms_dtype = is_float64 ? DType::Float64 : DType::Float32;
    Tensor rrms({batch_size}, rrms_dtype, input.device());

    size_t elem_size = input.dtype_size();
    size_t rrms_elem_size = rrms.dtype_size();

    VkBuffer buf_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buf_output = getVulkanBuffer(output.data_ptr());
    VkBuffer buf_weight = getVulkanBuffer(weight.data_ptr());
    VkBuffer buf_rrms = getVulkanBuffer(rrms.data_ptr());

    size_t input_buf_size = input.numel() * elem_size;
    size_t output_buf_size = output.numel() * elem_size;
    size_t weight_buf_size = normalized_shape * elem_size;
    size_t rrms_buf_size = batch_size * rrms_elem_size;

    // Bindings: input(0), output(1), weight(2), rrms_out(3)
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buf_input},
        {1, buf_output},
        {2, buf_weight},
        {3, buf_rrms},
    };
    std::vector<size_t> sizes = {input_buf_size, output_buf_size, weight_buf_size, rrms_buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t normalized_shape;
        float epsilon;
        uint32_t padding;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(batch_size);
    push_constants.normalized_shape = static_cast<uint32_t>(normalized_shape);
    push_constants.epsilon = epsilon;
    push_constants.padding = 0;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One workgroup per batch element
    uint32_t workgroups = static_cast<uint32_t>(batch_size);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {output, rrms};
}

// RMSNorm Backward - GPU implementation
auto VulkanBackend::dispatchRMSNormBackward(const Tensor& grad_output, const Tensor& input,
                                              const Tensor& rrms, const Tensor& weight,
                                              int64_t normalized_shape)
                                              -> std::pair<Tensor, Tensor> {
    int32_t device_id = input.device().index;

    // For Float16/BFloat16, upcast to Float32
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        auto go_f32 = grad_output.to(DType::Float32);
        auto in_f32 = input.to(DType::Float32);
        auto rrms_f32 = rrms.to(DType::Float32);
        auto w_f32 = weight.to(DType::Float32);
        auto [gi, gw] = dispatchRMSNormBackward(go_f32, in_f32, rrms_f32, w_f32, normalized_shape);
        return {gi.to(orig_dtype), gw.to(orig_dtype)};
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    std::string shader_name = is_float64 ? "rms_norm_backward_f64" : "rms_norm_backward";
    auto* pipeline = getPipeline(shader_name, device_id);

    int64_t batch_size = input.numel() / normalized_shape;

    // Create output tensors
    Tensor grad_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                      input.dtype(), input.device());
    Tensor grad_weight = dispatchZeros({normalized_shape}, input.dtype(), input.device());

    size_t elem_size = input.dtype_size();
    size_t rrms_elem_size = rrms.dtype_size();

    VkBuffer buf_grad_out = getVulkanBuffer(grad_output.data_ptr());
    VkBuffer buf_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buf_rrms = getVulkanBuffer(rrms.data_ptr());
    VkBuffer buf_weight = getVulkanBuffer(weight.data_ptr());
    VkBuffer buf_grad_input = getVulkanBuffer(grad_input.data_ptr());
    VkBuffer buf_grad_weight = getVulkanBuffer(grad_weight.data_ptr());

    size_t input_buf_size = input.numel() * elem_size;
    size_t rrms_buf_size = batch_size * rrms_elem_size;
    size_t norm_buf_size = normalized_shape * elem_size;

    // Bindings: grad_output(0), input(1), rrms(2), weight(3), grad_input(4), grad_weight(5)
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buf_grad_out},
        {1, buf_input},
        {2, buf_rrms},
        {3, buf_weight},
        {4, buf_grad_input},
        {5, buf_grad_weight},
    };
    std::vector<size_t> sizes = {input_buf_size, input_buf_size, rrms_buf_size, norm_buf_size,
                                  input_buf_size, norm_buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t normalized_shape;
        uint32_t padding0;
        uint32_t padding1;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(batch_size);
    push_constants.normalized_shape = static_cast<uint32_t>(normalized_shape);
    push_constants.padding0 = 0;
    push_constants.padding1 = 0;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One workgroup per batch element
    uint32_t workgroups = static_cast<uint32_t>(batch_size);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return {grad_input, grad_weight};
}

// Phase 3: BoxIoU - GPU implementation
auto VulkanBackend::dispatchBoxIoU(const Tensor& boxes1, const Tensor& boxes2, int64_t iou_type) -> Tensor {
    int32_t device_id = boxes1.device().index;
    auto* pipeline = getPipeline("box_iou", device_id);

    int64_t N = boxes1.shape()[0];
    int64_t M = boxes2.shape()[0];

    // The box_iou shader operates on float (32-bit), so convert Float64→Float32
    // Also ensure inputs are contiguous (views/slices may have offsets into parent buffers)
    Tensor b1 = boxes1.contiguous();
    Tensor b2 = boxes2.contiguous();
    if (b1.dtype() != DType::Float32) b1 = b1.to(DType::Float32);
    if (b2.dtype() != DType::Float32) b2 = b2.to(DType::Float32);

    Tensor result({N, M}, DType::Float32, boxes1.device());

    size_t elem_size = sizeof(float);
    VkBuffer buf_boxes1 = getVulkanBuffer(b1.data_ptr());
    VkBuffer buf_boxes2 = getVulkanBuffer(b2.data_ptr());
    VkBuffer buf_result = getVulkanBuffer(result.data_ptr());

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buf_boxes1},
        {1, buf_boxes2},
        {2, buf_result},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(N * 4) * elem_size,
        static_cast<size_t>(M * 4) * elem_size,
        static_cast<size_t>(N * M) * elem_size,
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t N;
        uint32_t M;
        uint32_t iou_type;
        uint32_t padding;
    } push_constants;

    push_constants.N = static_cast<uint32_t>(N);
    push_constants.M = static_cast<uint32_t>(M);
    push_constants.iou_type = static_cast<uint32_t>(iou_type);
    push_constants.padding = 0;

    uint64_t total = static_cast<uint64_t>(N) * M;
    uint32_t workgroups = static_cast<uint32_t>((total + 255) / 256);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    // Convert back to original dtype if needed
    if (boxes1.dtype() != DType::Float32) {
        result = result.to(boxes1.dtype());
    }

    return result;
}

// Phase 3: OneHot - GPU implementation
auto VulkanBackend::dispatchOneHot(const Tensor& indices, int64_t num_classes) -> Tensor {
    int32_t device_id = indices.device().index;
    auto* pipeline = getPipeline("one_hot", device_id);

    // The one_hot shader reads int indices_data[] (32-bit), so convert Int64→Int32
    Tensor indices_i32 = (indices.dtype() == DType::Int32) ? indices : indices.to(DType::Int32);

    int64_t batch_size = indices_i32.numel();
    Tensor output({batch_size, num_classes}, DType::Float32, indices.device());

    VkBuffer buf_indices = getVulkanBuffer(indices_i32.data_ptr());
    VkBuffer buf_output = getVulkanBuffer(output.data_ptr());

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buf_indices},
        {1, buf_output},
    };
    std::vector<size_t> sizes = {
        static_cast<size_t>(batch_size) * indices_i32.dtype_size(),
        static_cast<size_t>(batch_size * num_classes) * sizeof(float),
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t batch_size;
        uint32_t num_classes;
    } push_constants;

    push_constants.batch_size = static_cast<uint32_t>(batch_size);
    push_constants.num_classes = static_cast<uint32_t>(num_classes);

    uint64_t total = static_cast<uint64_t>(batch_size) * num_classes;
    uint32_t workgroups = static_cast<uint32_t>((total + 255) / 256);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// Phase 3: Nonzero - GPU implementation (multi-pass: count, prefix_sum, gather)
auto VulkanBackend::dispatchNonzero(const Tensor& input) -> Tensor {
    auto shape = input.shape();
    int64_t ndim = shape.size();
    int64_t numel = input.numel();

    if (numel == 0) {
        return Tensor({0, ndim}, DType::Int64, input.device());
    }

    int32_t device_id = input.device().index;
    uint32_t n = static_cast<uint32_t>(numel);
    uint32_t n_workgroups = (n + 255) / 256;

    // Ensure input is Float32 for the nonzero_count shader
    Tensor input_f32 = (input.dtype() == DType::Float32) ? input : input.to(DType::Float32);

    // Allocate flags buffer (one uint per element: 1=nonzero, 0=zero)
    Tensor flags({static_cast<int64_t>(n)}, DType::Int32, input.device());
    // Allocate count buffer (one per workgroup + space for total)
    Tensor count_buf({static_cast<int64_t>(n_workgroups + 1)}, DType::Int32, input.device());
    count_buf = dispatchFill(count_buf, 0.0f);

    VkBuffer buf_input = getVulkanBuffer(input_f32.data_ptr());
    VkBuffer buf_flags = getVulkanBuffer(flags.data_ptr());
    VkBuffer buf_count = getVulkanBuffer(count_buf.data_ptr());
    size_t input_bytes = n * sizeof(float);
    size_t flags_bytes = n * sizeof(uint32_t);
    size_t count_bytes = (n_workgroups + 1) * sizeof(uint32_t);

    // ---- Pass 1a: Per-element flags + workgroup counts ----
    {
        auto* pipeline = getPipeline("nonzero_count", device_id);
        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
            {0, buf_input}, {1, buf_flags}, {2, buf_count}
        };
        std::vector<size_t> sizes = {input_bytes, flags_bytes, count_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct { uint32_t n_elements; uint32_t pass; } pc;
        pc.n_elements = n; pc.pass = 0;

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, n_workgroups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // ---- Pass 1b: Reduce workgroup counts to get total ----
    {
        auto* pipeline = getPipeline("nonzero_count", device_id);
        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
            {0, buf_input}, {1, buf_flags}, {2, buf_count}
        };
        std::vector<size_t> sizes = {input_bytes, flags_bytes, count_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct { uint32_t n_elements; uint32_t pass; } pc;
        pc.n_elements = n; pc.pass = 1;

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, 1, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Read back total count (GPU→CPU sync required for variable-size output)
    Tensor count_cpu = count_buf.to(Device::cpu());
    int64_t total_count = static_cast<int64_t>(count_cpu.data<int32_t>()[0]);

    if (total_count == 0) {
        return Tensor({0, ndim}, DType::Int64, input.device());
    }

    // ---- Pass 2: Prefix sum of flags ----
    // Reuse prefix_sum shader with flags as "mask" (mask_is_float=1 since flags are uint 0/1,
    // and uintBitsToFloat(0)==0.0, uintBitsToFloat(1)!=0.0)
    Tensor prefix_sums({static_cast<int64_t>(n)}, DType::Int32, input.device());
    prefix_sums = dispatchFill(prefix_sums, 0.0f);
    Tensor block_sums({static_cast<int64_t>(n_workgroups)}, DType::Int32, input.device());
    block_sums = dispatchFill(block_sums, 0.0f);

    VkBuffer buf_prefix = getVulkanBuffer(prefix_sums.data_ptr());
    VkBuffer buf_blocks = getVulkanBuffer(block_sums.data_ptr());
    size_t prefix_bytes = n * sizeof(uint32_t);
    size_t blocks_bytes = n_workgroups * sizeof(uint32_t);

    // Pass 2a: Local scan
    {
        auto* pipeline = getPipeline("prefix_sum", device_id);
        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
            {0, buf_flags}, {1, buf_prefix}, {2, buf_blocks}
        };
        std::vector<size_t> sizes = {flags_bytes, prefix_bytes, blocks_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct { uint32_t n_elements; uint32_t mask_is_float; uint32_t pass; } pc;
        pc.n_elements = n; pc.mask_is_float = 1; pc.pass = 0;

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, n_workgroups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Pass 2b: Add block offsets
    if (n_workgroups > 1) {
        auto* pipeline = getPipeline("prefix_sum", device_id);
        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
            {0, buf_flags}, {1, buf_prefix}, {2, buf_blocks}
        };
        std::vector<size_t> sizes = {flags_bytes, prefix_bytes, blocks_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct { uint32_t n_elements; uint32_t mask_is_float; uint32_t pass; } pc;
        pc.n_elements = n; pc.mask_is_float = 1; pc.pass = 1;

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, n_workgroups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // ---- Pass 3: Gather multi-dimensional indices ----
    // Output is (num_nonzero, ndim) Int32 on GPU, converted to Int64 at the end
    Tensor output_i32({total_count * ndim}, DType::Int32, input.device());

    // Upload shape to GPU buffer
    std::vector<uint32_t> shape_u32(ndim);
    for (int64_t d = 0; d < ndim; ++d) {
        shape_u32[d] = static_cast<uint32_t>(shape[d]);
    }
    Tensor shape_buf({ndim}, DType::Int32, input.device());
    copy(shape_buf.data_ptr(), shape_u32.data(), ndim * sizeof(uint32_t), CopyKind::HostToDevice);
    synchronize(device_id);

    {
        auto* pipeline = getPipeline("nonzero_gather", device_id);
        VkBuffer buf_output = getVulkanBuffer(output_i32.data_ptr());
        VkBuffer buf_shape = getVulkanBuffer(shape_buf.data_ptr());
        size_t output_bytes = total_count * ndim * sizeof(int32_t);
        size_t shape_bytes = ndim * sizeof(uint32_t);

        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
            {0, buf_flags}, {1, buf_prefix}, {2, buf_output}, {3, buf_shape}
        };
        std::vector<size_t> sizes = {flags_bytes, prefix_bytes, output_bytes, shape_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct { uint32_t n_elements; uint32_t ndim; uint32_t output_size; } pc;
        pc.n_elements = n; pc.ndim = static_cast<uint32_t>(ndim);
        pc.output_size = static_cast<uint32_t>(total_count);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, n_workgroups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Convert Int32 output to Int64 (standard nonzero return type)
    // Read back to CPU for int32→int64 conversion, then upload
    Tensor output_cpu = output_i32.to(Device::cpu());
    const int32_t* src = output_cpu.data<int32_t>();
    Tensor result({total_count, ndim}, DType::Int64, Device::cpu());
    int64_t* dst = result.data<int64_t>();
    for (int64_t i = 0; i < total_count * ndim; ++i) {
        dst[i] = static_cast<int64_t>(src[i]);
    }
    return result.to(input.device());
}

// Softmax and loss operations implementation
auto VulkanBackend::dispatchSoftmax(const Tensor& input, int64_t dim) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    // For Float16/BFloat16, upcast to Float32 for numerical stability
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        DType orig_dtype = input.dtype();
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchSoftmax(input_f32, dim);
        return result_f32.to(orig_dtype);
    }

    // Select correct pipeline based on dtype
    bool is_float64 = (input.dtype() == DType::Float64);
    std::string shader_name;
    if (is_float64) {
        shader_name = "softmax_f64";
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

    // NOTE: max_vals and sum_vals are computed using shared memory within the shader.
    // No separate device allocations needed - the backward pass computes from output only.

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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchLogSoftmax(const Tensor& input, int64_t dim) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    // For BFloat16, upcast to Float32 for numerical stability
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchLogSoftmax(input_f32, dim);
        return result_f32.to(DType::BFloat16);
    }

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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// Advanced reduction operations implementation
auto VulkanBackend::dispatchArgmax(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    // Compute output shape first
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

    // Handle empty tensors - no GPU work needed
    if (input.numel() == 0) {
        return Tensor(out_shape, DType::Int64, input.device());
    }

    int32_t device_id = input.device().index;

    // Select correct pipeline based on dtype
    std::string shader_name;
    if (input.dtype() == DType::Int32) shader_name = "argmax_argmin_i32";
    else if (input.dtype() == DType::Float64) shader_name = "argmax_argmin_f64";
    else if (input.dtype() == DType::Float16) shader_name = "argmax_argmin_f16";
    else shader_name = "argmax_argmin";
    auto* pipeline = getPipeline(shader_name, device_id);

    Tensor output(out_shape, DType::Int64, input.device());  // Use Int64 for consistency with other backends

    // Get VkBuffer handles from tensor data pointers
    VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (input.dtype() == DType::Float16) {
        // Round up input buffer to 4-byte boundary for uint32 shader access
        size_t in_pairs = (input.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
    }

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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchArgmin(const Tensor& input, int64_t dim, bool keepdim) -> Tensor {
    // Compute output shape first
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

    // Handle empty tensors - no GPU work needed
    if (input.numel() == 0) {
        return Tensor(out_shape, DType::Int64, input.device());
    }

    int32_t device_id = input.device().index;

    // Select correct pipeline based on dtype
    std::string shader_name;
    if (input.dtype() == DType::Int32) shader_name = "argmax_argmin_i32";
    else if (input.dtype() == DType::Float64) shader_name = "argmax_argmin_f64";
    else if (input.dtype() == DType::Float16) shader_name = "argmax_argmin_f16";
    else shader_name = "argmax_argmin";
    auto* pipeline = getPipeline(shader_name, device_id);

    Tensor output(out_shape, DType::Int64, input.device());  // Use Int64 for consistency with other backends

    // Get VkBuffer handles from tensor data pointers
    VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();
    if (input.dtype() == DType::Float16) {
        // Round up input buffer to 4-byte boundary for uint32 shader access
        size_t in_pairs = (input.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
    }

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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchVariance(const Tensor& input, int64_t dim, bool unbiased, bool keepdim) -> Tensor {
    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    std::string shader_name = is_float64 ? "variance_std_f64" : "variance_std";
    auto* pipeline = getPipeline(shader_name, device_id);

    std::vector<int64_t> out_shape;
    auto input_shape = input.shape();

    // Check if this is a full reduction (dim < 0 means reduce all elements)
    bool full_reduction = (dim < 0);

    // For dispatchReduction, use INT64_MIN to signal full reduction
    int64_t reduction_dim = full_reduction ? INT64_MIN : dim;

    if (full_reduction) {
        // Full reduction: output is scalar (empty shape) or [1,1,...] if keepdim
        if (keepdim) {
            out_shape.assign(input_shape.size(), 1);
        } else {
            out_shape = {};  // Scalar
        }
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
    Tensor mean = dispatchReduction("mean", input, reduction_dim, true);

    // 2. Compute (X - mean)
    Tensor diff = dispatchBinaryOp("sub", input, mean);

    // 3. Square the differences
    Tensor squared_diff = dispatchBinaryOp("mul", diff, diff);

    // 4. Compute sum of squared differences
    Tensor sum_squared = dispatchReduction("sum", squared_diff, reduction_dim, keepdim);

    // 5. Divide by N or N-1
    uint32_t reduce_size = full_reduction ? static_cast<uint32_t>(input.numel()) : static_cast<uint32_t>(input_shape[dim]);
    double divisor = unbiased ? static_cast<double>(reduce_size - 1) : static_cast<double>(reduce_size);

    // Create a scalar tensor with the divisor matching the output shape
    // This ensures broadcasting preserves the correct output shape
    auto sum_shape = sum_squared.shape();
    std::vector<int64_t> divisor_shape(sum_shape.begin(), sum_shape.end());
    if (divisor_shape.empty()) {
        divisor_shape = {1};  // Can't create empty tensor on CPU, use [1] for scalar
    }
    Tensor divisor_tensor_cpu(divisor_shape, input.dtype(), Device::cpu());
    if (is_float64) {
        double* divisor_data = static_cast<double*>(divisor_tensor_cpu.data_ptr());
        for (int64_t i = 0; i < divisor_tensor_cpu.numel(); i++) {
            divisor_data[i] = divisor;
        }
    } else {
        float* divisor_data = static_cast<float*>(divisor_tensor_cpu.data_ptr());
        for (int64_t i = 0; i < divisor_tensor_cpu.numel(); i++) {
            divisor_data[i] = static_cast<float>(divisor);
        }
    }

    // Copy to device
    Tensor divisor_tensor = divisor_tensor_cpu.to(input.device());

    // Divide variance by divisor and reshape to match expected output
    Tensor result = dispatchBinaryOp("div", sum_squared, divisor_tensor);

    // If the output should be a scalar but broadcast made it [1], reshape to scalar
    if (full_reduction && !keepdim && result.shape().size() > 0) {
        return result.reshape({});
    }
    return result;
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

    // Check if this is a full reduction (dim < 0 means reduce all elements)
    bool full_reduction = (dim < 0);

    if (full_reduction) {
        // Full reduction: output is scalar (empty shape) or [1,1,...] if keepdim
        if (keepdim) {
            out_shape.assign(input_shape.size(), 1);
        } else {
            out_shape = {};  // Scalar - but we need at least 1 element for the output
        }
    } else {
        out_shape = std::vector<int64_t>(input_shape.begin(), input_shape.end());
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }

    // For full reduction to scalar, we still need buffer space, so use [1] internally
    std::vector<int64_t> buffer_shape = out_shape.empty() ? std::vector<int64_t>{1} : out_shape;
    Tensor output(buffer_shape, input.dtype(), input.device());

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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    // Synchronize to ensure GPU has completed before reading results
    synchronize(device_id);

    // If output should be scalar but we used [1] internally, reshape to scalar
    if (full_reduction && !keepdim) {
        return output.reshape({});
    }
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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// Indexing operations implementation
auto VulkanBackend::dispatchEmbedding(const Tensor& weight, const Tensor& indices,
                                      int64_t padding_idx) -> Tensor {
    auto weight_shape = weight.shape();
    auto indices_shape = indices.shape();

    int32_t device_id = weight.device().index;
    bool is_float64 = (weight.dtype() == DType::Float64);
    bool is_float16 = (weight.dtype() == DType::Float16);
    std::string shader_name = is_float64 ? "embedding_f64"
                            : is_float16 ? "embedding_f16"
                            : "embedding";
    auto* pipeline = getPipeline(shader_name, device_id);

    uint32_t num_embeddings = static_cast<uint32_t>(weight_shape[0]);
    uint32_t embedding_dim = static_cast<uint32_t>(weight_shape[1]);
    uint32_t num_indices = static_cast<uint32_t>(indices.numel());

    // Output shape: indices_shape + [embedding_dim]
    std::vector<int64_t> out_shape(indices_shape.begin(), indices_shape.end());
    out_shape.push_back(weight_shape[1]);

    Tensor output(out_shape, weight.dtype(), weight.device());

    // Handle empty tensors
    if (num_indices == 0) {
        return output;
    }

    // Convert Int64 indices to Int32 for shader compatibility
    Tensor indices_i32 = indices;
    if (indices.dtype() == DType::Int64) {
        std::vector<int64_t> idx_shape(indices_shape.begin(), indices_shape.end());
        indices_i32 = Tensor(idx_shape, DType::Int32, indices.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        VkBuffer buf_in = getVulkanBuffer(indices.data_ptr());
        VkBuffer buf_out = getVulkanBuffer(indices_i32.data_ptr());
        size_t size_in = indices.numel() * sizeof(int64_t);
        size_t size_out = indices_i32.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, VkBuffer>> cast_bindings = {
            {0, buf_in}, {1, buf_out}
        };
        std::vector<size_t> cast_sizes = {size_in, size_out};
        VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cast_bindings, cast_sizes);

        struct { uint32_t n_elements; } cast_pc;
        cast_pc.n_elements = static_cast<uint32_t>(indices.numel());

        VkCommandBuffer cast_cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cast_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cast_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cast_cmd, cast_pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cast_cmd, (indices.numel() + 255) / 256, 1, 1);
        insertComputeBarrier(cast_cmd);
        endSingleTimeCommands(cast_cmd, device_id);
    }

    // Get VkBuffer handles
    VkBuffer buf_weight = getVulkanBuffer(weight.data_ptr());
    VkBuffer buf_indices = getVulkanBuffer(indices_i32.data_ptr());
    VkBuffer buf_output = getVulkanBuffer(output.data_ptr());

    size_t weight_buf_size = weight.numel() * weight.dtype_size();
    size_t indices_buf_size = indices_i32.numel() * sizeof(int32_t);
    size_t output_buf_size = output.numel() * output.dtype_size();

    // Bindings: embeddings(0), indices(1), output(2)
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buf_weight},
        {1, buf_indices},
        {2, buf_output}
    };
    std::vector<size_t> sizes = {weight_buf_size, indices_buf_size, output_buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Push constants
    struct PushConstants {
        uint32_t num_embeddings;
        uint32_t embedding_dim;
        uint32_t num_indices;
        uint32_t padding_idx;
    } push_constants;

    push_constants.num_embeddings = num_embeddings;
    push_constants.embedding_dim = embedding_dim;
    push_constants.num_indices = num_indices;
    push_constants.padding_idx = (padding_idx >= 0) ? static_cast<uint32_t>(padding_idx) : 0xFFFFFFFFu;

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // One thread per index (each thread copies embedding_dim elements)
    uint32_t workgroups = (num_indices + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchGather(const Tensor& input, int64_t dim, const Tensor& indices) -> Tensor {
    auto indices_shape = indices.shape();
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    // Handle empty tensors
    if (input.numel() == 0 || indices.numel() == 0) {
        std::vector<int64_t> out_shape(indices_shape.begin(), indices_shape.end());
        return Tensor(out_shape, input.dtype(), input.device());
    }

    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    std::string shader_name = is_float64 ? "gather_f64"
                            : is_float16 ? "gather_f16"
                            : "gather";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Normalize dimension
    int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::invalid_argument("Gather: dimension out of range");
    }

    std::vector<int64_t> out_shape(indices_shape.begin(), indices_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    // Convert Int64 indices to Int32 for shader compatibility
    Tensor indices_int32 = indices;
    if (indices.dtype() == DType::Int64) {
        std::vector<int64_t> idx_shape(indices_shape.begin(), indices_shape.end());
        indices_int32 = Tensor(idx_shape, DType::Int32, indices.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        VkBuffer buf_in = getVulkanBuffer(indices.data_ptr());
        VkBuffer buf_out = getVulkanBuffer(indices_int32.data_ptr());
        size_t size_in = indices.numel() * sizeof(int64_t);
        size_t size_out = indices_int32.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, VkBuffer>> cast_bindings = {
            {0, buf_in}, {1, buf_out}
        };
        std::vector<size_t> cast_sizes = {size_in, size_out};
        VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cast_bindings, cast_sizes);

        struct { uint32_t n_elements; } cast_pc;
        cast_pc.n_elements = static_cast<uint32_t>(indices.numel());

        uint32_t cast_groups = (cast_pc.n_elements + 255) / 256;
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cmd, cast_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cmd, cast_groups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Get Vulkan buffers
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_indices = getVulkanBuffer(indices_int32.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());

    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_indices = indices_int32.numel() * sizeof(int32_t);
    size_t buffer_size_output = output.numel() * output.dtype_size();

    // Set up descriptor set with all buffers
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_input},
        {1, buffer_indices},
        {2, buffer_output}
    };
    std::vector<size_t> sizes = {
        buffer_size_input,
        buffer_size_indices,
        buffer_size_output
    };

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Calculate gather parameters
    uint32_t dim_size = static_cast<uint32_t>(input_shape[dim]);
    uint32_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
        inner_size *= static_cast<uint32_t>(input_shape[d]);
    }
    uint32_t outer_size = 1;
    for (int64_t d = 0; d < dim; ++d) {
        outer_size *= static_cast<uint32_t>(input_shape[d]);
    }

    // Push constants matching shader layout
    struct PushConstants {
        uint32_t input_size;
        uint32_t output_size;
        uint32_t dim;
        uint32_t dim_size;
        uint32_t inner_size;
        uint32_t outer_size;
    } push_constants;

    push_constants.input_size = static_cast<uint32_t>(input.numel());
    push_constants.output_size = static_cast<uint32_t>(output.numel());
    push_constants.dim = static_cast<uint32_t>(dim);
    push_constants.dim_size = dim_size;
    push_constants.inner_size = inner_size;
    push_constants.outer_size = outer_size;

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
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchScatter(const Tensor& input, int64_t dim, const Tensor& indices,
                                    const Tensor& values, int64_t reduction) -> Tensor {
    auto input_shape = input.shape();

    // Handle empty tensors - no GPU work needed
    if (input.numel() == 0 || indices.numel() == 0) {
        std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
        return Tensor(out_shape, input.dtype(), input.device());
    }

    int32_t device_id = input.device().index;

    // Float64 scatter with reduction requires atomic int64 support for CAS-loop atomics
    if (input.dtype() == DType::Float64 && reduction != 0 &&
        !devices_[device_id].hasAtomicInt64) {
        throw std::runtime_error(
            "Scatter with Float64 reduction requires VK_KHR_shader_atomic_int64 support. "
            "Use CPU backend or reduction=0 (direct assignment) for this device.");
    }

    // Select shader based on dtype
    const char* shader_name = (input.dtype() == DType::Float64) ? "scatter_f64"
                            : (input.dtype() == DType::Float16) ? "scatter_f16"
                            : "scatter";
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

    // Convert Int64 indices to Int32 for shader compatibility (on-device)
    Tensor indices_int32 = indices;
    if (indices.dtype() == DType::Int64) {
        std::vector<int64_t> idx_shape(indices.shape().begin(), indices.shape().end());
        indices_int32 = Tensor(idx_shape, DType::Int32, indices.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        VkBuffer buf_in = getVulkanBuffer(indices.data_ptr());
        VkBuffer buf_out = getVulkanBuffer(indices_int32.data_ptr());
        size_t size_in = indices.numel() * sizeof(int64_t);
        size_t size_out = indices_int32.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, VkBuffer>> cast_bindings = {
            {0, buf_in}, {1, buf_out}
        };
        std::vector<size_t> cast_sizes = {size_in, size_out};
        VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cast_bindings, cast_sizes);

        struct { uint32_t n_elements; } cast_pc;
        cast_pc.n_elements = static_cast<uint32_t>(indices.numel());

        uint32_t cast_groups = (cast_pc.n_elements + 255) / 256;
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cmd, cast_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cmd, cast_groups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

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

    // Calculate output shape
    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    out_shape[dim] = indices.numel();
    Tensor output(out_shape, input.dtype(), input.device());

    // Handle empty tensors - no GPU work needed
    if (output.numel() == 0 || indices.numel() == 0) {
        return output;
    }

    // Select correct shader based on dtype
    std::string shader_name;
    if (input.dtype() == DType::Float64) shader_name = "index_select_f64";
    else if (input.dtype() == DType::Float16) shader_name = "index_select_f16";
    else if (input.dtype() == DType::Int64) shader_name = "index_select_i64";
    else shader_name = "index_select";

    // CPU fallback for unsupported dtypes (Int32, Bool, etc. that don't have dedicated shaders)
    if (input.dtype() != DType::Float32 && input.dtype() != DType::Float64 &&
        input.dtype() != DType::Float16 && input.dtype() != DType::Int64) {
        auto input_cpu = input.to(Device::cpu());
        auto indices_cpu = indices.to(Device::cpu());
        OpAttributes cpu_attrs;
        cpu_attrs["dim"] = std::to_string(dim);
        std::vector<Tensor> cpu_inputs = {input_cpu, indices_cpu};
        auto cpu_results = tenzor::dispatch(OpId::IndexSelect, cpu_inputs, cpu_attrs);
        return cpu_results[0].to(input.device());
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    // Convert indices to Int32 if needed (shader expects int32, on-device)
    Tensor indices_int32 = indices;
    if (indices.dtype() == DType::Int64) {
        std::vector<int64_t> idx_shape(indices.shape().begin(), indices.shape().end());
        indices_int32 = Tensor(idx_shape, DType::Int32, indices.device());

        auto* cast_pipeline = getPipeline("cast_int64_to_int32", device_id);
        VkBuffer buf_in = getVulkanBuffer(indices.data_ptr());
        VkBuffer buf_out = getVulkanBuffer(indices_int32.data_ptr());
        size_t size_in = indices.numel() * sizeof(int64_t);
        size_t size_out = indices_int32.numel() * sizeof(int32_t);

        std::vector<std::pair<uint32_t, VkBuffer>> cast_bindings = {
            {0, buf_in}, {1, buf_out}
        };
        std::vector<size_t> cast_sizes = {size_in, size_out};
        VkDescriptorSet cast_ds = allocateAndWriteDescriptorSet(device_id, cast_pipeline, cast_bindings, cast_sizes);

        struct { uint32_t n_elements; } cast_pc;
        cast_pc.n_elements = static_cast<uint32_t>(indices.numel());

        uint32_t cast_groups = (cast_pc.n_elements + 255) / 256;
        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cast_pipeline->layout(), 0, 1, &cast_ds, 0, nullptr);
        vkCmdPushConstants(cmd, cast_pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cast_pc), &cast_pc);
        vkCmdDispatch(cmd, cast_groups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
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
    insertComputeBarrier(cmdBuffer);

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
    insertComputeBarrier(cmdBuffer);

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

        // Insert pre-read barrier to ensure input data from previous ops is ready
        insertPreReadBarrier(cmdBuffer);

        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);

        uint32_t workgroups = (input.numel() + 255) / 256;
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        // Add memory barrier
        insertComputeBarrier(cmdBuffer);

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

    // Insert transfer-to-compute barrier before the compute dispatch reads this data
    insertTransferToComputeBarrier(copyCmd);

    endSingleTimeCommands(copyCmd, device_id);

    // CRITICAL: With batching enabled, force submit now to ensure staging buffer
    // content is copied to device buffers before staging buffer can be reused.
    // The staging buffer is shared and may be overwritten by any other operation
    // that uses getStagingBuffer() before our batch is submitted.
    if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
        submitBatchIfNeeded(device_id, true);  // Force submit the copy commands
        ensurePendingWorkComplete(device_id);   // Wait for copies to complete
    }

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

    // Insert pre-read barrier to ensure input data from previous ops is ready
    insertPreReadBarrier(cmdBuffer);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = (output.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    // CRITICAL: Force batch submission before temp buffers go out of scope!
    // buffer_shape, buffer_strides, and buffer_perm are unique_ptrs that will
    // be destroyed when this function returns. With batching enabled, the
    // command buffer still references these buffers. We must submit and wait
    // for completion before destroying them.
    if constexpr (vulkan_config::USE_COMMAND_BATCHING) {
        submitBatchIfNeeded(device_id, true);  // Force submit
        ensurePendingWorkComplete(device_id);   // Wait for GPU
    }

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
 * Otherwise, creates a new contiguous copy using GPU strided_copy kernel.
 */
auto VulkanBackend::dispatchContiguous(const Tensor& input) -> Tensor {
    // If already contiguous, return as-is
    if (input.is_contiguous()) {
        return input;
    }

    // For non-contiguous tensors, use GPU kernel to reorder the data
    const int64_t total_elements = input.numel();
    const int64_t ndims = input.ndim();
    const int64_t base_offset = input.impl_ ? input.impl_->offset : 0;

    // Create new contiguous tensor with same shape, dtype, device
    Tensor result(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    if (total_elements == 0) {
        return result;
    }

    int32_t device_id = input.device().index;

    // Select shader based on dtype (element size must match shader buffer layout)
    std::string shader_name = "strided_copy";
    if (input.dtype() == DType::Float64 || input.dtype() == DType::Int64) {
        shader_name = "strided_copy_f64";  // uvec2 layout works for any 8-byte type
    } else if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        shader_name = "strided_copy_f16";
    } else if (input.dtype() == DType::UInt8 || input.dtype() == DType::Bool ||
               input.dtype() == DType::Int8) {
        shader_name = "strided_copy_u8";
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    // Get Vulkan buffers - use base storage pointer for input
    const void* base_storage_ptr = input.impl_ ? input.impl_->storage->data() : input.data_ptr();
    VkBuffer buffer_in = getVulkanBuffer(const_cast<void*>(base_storage_ptr));
    VkBuffer buffer_out = getVulkanBuffer(result.data_ptr());

    // Calculate buffer sizes
    int64_t max_offset = base_offset;
    auto strides = input.strides();
    auto shape = input.shape();
    if (ndims > 0) {
        for (int64_t dim = 0; dim < ndims; ++dim) {
            max_offset += (shape[dim] - 1) * std::abs(strides[dim]);
        }
    }
    size_t input_buffer_size = (max_offset + 1) * input.dtype_size();
    size_t output_buffer_size = total_elements * input.dtype_size();

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_in},
        {1, buffer_out}
    };
    std::vector<size_t> sizes = {input_buffer_size, output_buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Push constants structure matching the shader
    // IMPORTANT: The shader uses uvec4/ivec4 which require 16-byte alignment
    // We need padding after base_offset to align the first uvec4
    struct PushConstants {
        uint32_t n_elements;
        uint32_t ndims;
        uint32_t base_offset;
        uint32_t _padding;  // Padding to align shape_0_3 to 16 bytes
        uint32_t shape_0_3[4];   // shape[0..3] as uvec4
        uint32_t shape_4_7[4];   // shape[4..7] as uvec4
        int32_t strides_0_3[4];  // strides[0..3] as ivec4
        int32_t strides_4_7[4];  // strides[4..7] as ivec4
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(total_elements);
    push_constants.ndims = static_cast<uint32_t>(ndims);
    push_constants.base_offset = static_cast<uint32_t>(base_offset);
    push_constants._padding = 0;

    // Initialize all shape/stride values to defaults
    for (int i = 0; i < 4; ++i) {
        push_constants.shape_0_3[i] = 1;
        push_constants.shape_4_7[i] = 1;
        push_constants.strides_0_3[i] = 0;
        push_constants.strides_4_7[i] = 0;
    }

    // Fill in actual shape and strides
    for (int64_t i = 0; i < ndims && i < 4; ++i) {
        push_constants.shape_0_3[i] = static_cast<uint32_t>(shape[i]);
        push_constants.strides_0_3[i] = static_cast<int32_t>(strides[i]);
    }
    for (int64_t i = 4; i < ndims && i < 8; ++i) {
        push_constants.shape_4_7[i - 4] = static_cast<uint32_t>(shape[i]);
        push_constants.strides_4_7[i - 4] = static_cast<int32_t>(strides[i]);
    }

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);

    // Insert pre-read barrier to ensure input data from previous ops is ready
    // This is critical for strided_copy which reads non-contiguous data
    insertPreReadBarrier(cmdBuffer);

    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = (total_elements + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return result;
}

// ============================================================================
// Memory Operations Implementation
// ============================================================================

/**
 * @brief Create tensor filled with zeros
 */
auto VulkanBackend::dispatchZeros(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor {
    // For Float64, Int64, UInt8, or Bool, use full() with 0.0 since the basic fill shader only handles 32-bit values
    // This is consistent with how dispatchOnes handles these types
    if (dtype == DType::Float64 || dtype == DType::Int64 || dtype == DType::UInt8 || dtype == DType::Bool) {
        return dispatchFull(shape, 0.0, dtype);
    }

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

    // Fill with zeros using fill operation (for Float32, Int32, etc.)
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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Arange - generate sequential values on GPU
 */
auto VulkanBackend::dispatchArange(float start, float end, float step, DType dtype, const Device& device) -> Tensor {
    if (step == 0.0f) {
        throw std::runtime_error("arange: step must be non-zero");
    }
    if ((step > 0 && start >= end) || (step < 0 && start <= end)) {
        return Tensor({0}, dtype, device);
    }

    int64_t numel = static_cast<int64_t>(std::ceil((end - start) / step));
    if (numel <= 0) {
        return Tensor({0}, dtype, device);
    }

    // Float64 uses dedicated arange_f64 shader for full precision
    if (dtype == DType::Float64) {
        Tensor output({numel}, DType::Float64, device);

        int32_t device_id = device.index;
        auto* pipeline = getPipeline("arange_f64", device_id);

        VkBuffer buf_out = getVulkanBuffer(output.data_ptr());
        size_t buf_size = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {{0, buf_out}};
        std::vector<size_t> sizes = {buf_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstantsF64 {
            uint32_t num_elements;
            uint32_t _pad;
            double start;
            double step;
        } push_constants;

        push_constants.num_elements = static_cast<uint32_t>(numel);
        push_constants._pad = 0;
        push_constants.start = static_cast<double>(start);
        push_constants.step = static_cast<double>(step);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF64), &push_constants);

        uint32_t workgroups = (numel + 255) / 256;
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    // For other non-Float32 dtypes, compute in Float32 then convert
    if (dtype != DType::Float32) {
        auto result_f32 = dispatchArange(start, end, step, DType::Float32, device);
        return result_f32.to(dtype);
    }

    Tensor output({numel}, DType::Float32, device);

    int32_t device_id = device.index;
    auto* pipeline = getPipeline("arange", device_id);

    VkBuffer buf_out = getVulkanBuffer(output.data_ptr());
    size_t buf_size = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {{0, buf_out}};
    std::vector<size_t> sizes = {buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        float start;
        float step;
        uint32_t num_elements;
    } push_constants;

    push_constants.start = start;
    push_constants.step = step;
    push_constants.num_elements = static_cast<uint32_t>(numel);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = (numel + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Linspace - generate linearly-spaced values on GPU
 */
auto VulkanBackend::dispatchLinspace(float start, float end, int64_t steps, DType dtype, const Device& device) -> Tensor {
    if (steps < 0) {
        throw std::runtime_error("linspace: number of steps must be non-negative");
    }
    if (steps == 0) {
        return Tensor({0}, dtype, device);
    }

    // Float64 uses dedicated linspace_f64 shader for full precision
    if (dtype == DType::Float64) {
        Tensor output({steps}, DType::Float64, device);

        int32_t device_id = device.index;
        auto* pipeline = getPipeline("linspace_f64", device_id);

        VkBuffer buf_out = getVulkanBuffer(output.data_ptr());
        size_t buf_size = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {{0, buf_out}};
        std::vector<size_t> sizes = {buf_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstantsF64 {
            uint32_t num_steps;
            uint32_t _pad;
            double start;
            double end_val;
        } push_constants;

        push_constants.num_steps = static_cast<uint32_t>(steps);
        push_constants._pad = 0;
        push_constants.start = static_cast<double>(start);
        push_constants.end_val = static_cast<double>(end);

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF64), &push_constants);

        uint32_t workgroups = (steps + 255) / 256;
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    // For other non-Float32 dtypes, compute in Float32 then convert
    if (dtype != DType::Float32) {
        auto result_f32 = dispatchLinspace(start, end, steps, DType::Float32, device);
        return result_f32.to(dtype);
    }

    Tensor output({steps}, DType::Float32, device);

    int32_t device_id = device.index;
    auto* pipeline = getPipeline("linspace", device_id);

    VkBuffer buf_out = getVulkanBuffer(output.data_ptr());
    size_t buf_size = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {{0, buf_out}};
    std::vector<size_t> sizes = {buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        float start;
        float end_val;
        uint32_t num_steps;
    } push_constants;

    push_constants.start = start;
    push_constants.end_val = end;
    push_constants.num_steps = static_cast<uint32_t>(steps);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = (steps + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Eye - generate identity matrix on GPU
 */
auto VulkanBackend::dispatchEye(int64_t n, int64_t m, DType dtype, const Device& device) -> Tensor {
    if (m < 0) m = n;

    if (n == 0 || m == 0) {
        return Tensor({n, m}, dtype, device);
    }

    // Float64 uses dedicated eye_f64 shader for proper double-precision output
    if (dtype == DType::Float64) {
        Tensor output({n, m}, DType::Float64, device);

        int32_t device_id = device.index;
        auto* pipeline = getPipeline("eye_f64", device_id);

        VkBuffer buf_out = getVulkanBuffer(output.data_ptr());
        size_t buf_size = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {{0, buf_out}};
        std::vector<size_t> sizes = {buf_size};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstants {
            uint32_t rows;
            uint32_t cols;
        } push_constants;

        push_constants.rows = static_cast<uint32_t>(n);
        push_constants.cols = static_cast<uint32_t>(m);

        int64_t total = n * m;
        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstants), &push_constants);

        uint32_t workgroups = (total + 255) / 256;
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    // For other non-Float32 dtypes, compute in Float32 then convert
    if (dtype != DType::Float32) {
        auto result_f32 = dispatchEye(n, m, DType::Float32, device);
        return result_f32.to(dtype);
    }

    Tensor output({n, m}, DType::Float32, device);

    int32_t device_id = device.index;
    auto* pipeline = getPipeline("eye", device_id);

    VkBuffer buf_out = getVulkanBuffer(output.data_ptr());
    size_t buf_size = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {{0, buf_out}};
    std::vector<size_t> sizes = {buf_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t rows;
        uint32_t cols;
    } push_constants;

    push_constants.rows = static_cast<uint32_t>(n);
    push_constants.cols = static_cast<uint32_t>(m);

    int64_t total = n * m;
    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = (total + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

/**
 * @brief Fill tensor with scalar value using compute shader
 */
auto VulkanBackend::dispatchFill(const Tensor& input, float value) -> Tensor {
    auto input_shape = input.shape();
    int32_t device_id = input.device().index;

    std::vector<int64_t> out_shape(input_shape.begin(), input_shape.end());
    Tensor output(out_shape, input.dtype(), input.device());

    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Float64 requires special handling: the generic fill shader is 32-bit only,
    // so use the full_f64 pipeline which properly writes double-precision values
    if (input.dtype() == DType::Float64) {
        auto* pipeline = getPipeline("full_f64", device_id);

        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
            {0, buffer_out}
        };
        std::vector<size_t> sizes = {buffer_size_out};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstantsF64 {
            uint32_t n_elements;
            uint32_t padding;
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

        insertComputeBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
    }

    auto* pipeline = getPipeline("fill", device_id);

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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

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

    // Handle empty tensors - no GPU work needed
    if (input.numel() == 0) {
        return output;
    }

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

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

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

    // Handle empty output tensor - no GPU work needed
    int64_t out_numel = 1;
    for (auto d : output_shape) out_numel *= d;
    if (out_numel == 0) {
        return output;
    }

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
        int64_t input_dim_size = input.shape()[dim];

        // Skip empty inputs (0-element tensors have null data_ptr)
        if (input.numel() == 0) {
            current_offset_in_cat_dim += input_dim_size;
            continue;
        }

        auto [buffer_in, buffer_in_base_offset] = getVulkanBufferAndOffset(input.data_ptr());

        // For each outer block, copy the input data to the correct position
        for (int64_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
            // Calculate source and destination offsets
            // buffer_in_base_offset accounts for slice view offset into the storage buffer
            int64_t src_offset = outer_idx * input_dim_size * inner_size * element_size +
                                static_cast<int64_t>(buffer_in_base_offset);
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
    // Handle empty tensors - no work to do
    if (input.numel() == 0) {
        auto input_shape = input.shape();
        std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
        return Tensor(output_shape, input.dtype(), input.device());
    }

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);
    std::string shader_name = is_float64 ? "clamp_f64" : (is_float16 ? "clamp_f16" : "clamp");
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
        // Float32 and Float16 push constants (same layout: n, min, max as float)
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

    // Float16 shader processes pairs of elements
    uint32_t workgroups;
    if (is_float16) {
        uint32_t num_pairs = (static_cast<uint32_t>(output.numel()) + 1) / 2;
        workgroups = (num_pairs + 255) / 256;
    } else {
        workgroups = (output.numel() + 255) / 256;
    }
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

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
    // Handle empty tensors - no GPU work needed
    if (input.numel() == 0) {
        auto input_shape = input.shape();
        std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
        return Tensor(output_shape, input.dtype(), input.device());
    }

    // BFloat16: upcast to Float32, compute, downcast back
    if (input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = dispatchActivation(op_name, input_f32, opcode, param);
        return result_f32.to(DType::BFloat16);
    }

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
    if (is_float16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t in_pairs = (input.numel() + 1) / 2;
        size_t out_pairs = (output.numel() + 1) / 2;
        buffer_size_in = in_pairs * 4;
        buffer_size_out = out_pairs * 4;
    }

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
    insertComputeBarrier(cmdBuffer);

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
    // BFloat16: upcast to Float32, compute, downcast back
    if (grad_output.dtype() == DType::BFloat16) {
        auto go_f32 = grad_output.to(DType::Float32);
        auto io_f32 = input_or_output.to(DType::Float32);
        auto result_f32 = dispatchActivationBackward(op_name, go_f32, io_f32, opcode, param);
        return result_f32.to(DType::BFloat16);
    }

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
    if (is_float16) {
        // Round up to 4-byte boundary for uint32 shader access (2 Float16 per uint32)
        size_t go_pairs = (grad_output.numel() + 1) / 2;
        size_t io_pairs = (input_or_output.numel() + 1) / 2;
        size_t gi_pairs = (grad_input.numel() + 1) / 2;
        buffer_size_grad_out = go_pairs * 4;
        buffer_size_input_or_output = io_pairs * 4;
        buffer_size_grad_in = gi_pairs * 4;
    }

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
    insertComputeBarrier(cmdBuffer);

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
    insertComputeBarrier(cmdBuffer);

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

    // For Float16/BFloat16, upcast to Float32 for numerical stability
    if (grad_output.dtype() == DType::Float16 || grad_output.dtype() == DType::BFloat16) {
        DType orig_dtype = grad_output.dtype();
        auto grad_f32 = grad_output.to(DType::Float32);
        auto out_f32 = output.to(DType::Float32);
        auto result_f32 = dispatchSoftmaxBackward(grad_f32, out_f32, dim);
        return result_f32.to(orig_dtype);
    }

    // Select correct pipeline based on dtype
    bool is_float64 = (grad_output.dtype() == DType::Float64);
    std::string shader_name;
    if (is_float64) {
        shader_name = "softmax_backward_f64";
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
    insertComputeBarrier(cmdBuffer);

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
    insertComputeBarrier(cmdBuffer);

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

    // Select shader based on dtype
    std::string shader_name = "avg_pool2d";
    if (input.dtype() == DType::Float64) {
        shader_name = "avg_pool2d_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "avg_pool2d_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

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
    insertComputeBarrier(cmdBuffer);

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

    // Select shader based on dtype
    std::string shader_name = "max_pool2d";
    if (input.dtype() == DType::Float64) {
        shader_name = "max_pool2d_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "max_pool2d_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

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
    insertComputeBarrier(cmdBuffer);

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

    // Select shader based on dtype
    std::string shader_name = "avg_pool2d_backward";
    if (input.dtype() == DType::Float64) {
        shader_name = "avg_pool2d_backward_f64";
    } else if (input.dtype() == DType::Float16) {
        shader_name = "avg_pool2d_backward_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

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
    insertComputeBarrier(cmdBuffer);

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
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

// ============================================================================
// MaxPool2d Backward with Indices (scatter-based with GPU atomicAdd)
// ============================================================================

auto VulkanBackend::dispatchMaxPool2dBackwardWithIndices(const Tensor& grad_output, const Tensor& indices,
                                                          int64_t H_in, int64_t W_in) -> Tensor {
    // Use GPU kernel with atomicAdd for scatter operation
    auto grad_out_shape = grad_output.shape();
    if (grad_out_shape.size() != 4) {
        throw std::invalid_argument("max_pool2d_backward requires 4D grad_output (N, C, H_out, W_out)");
    }

    int64_t N = grad_out_shape[0];
    int64_t C = grad_out_shape[1];
    int64_t grad_out_numel = grad_output.numel();
    int64_t grad_in_numel = N * C * H_in * W_in;

    if (grad_out_numel == 0) {
        return Tensor({N, C, H_in, W_in}, grad_output.dtype(), grad_output.device());
    }

    int32_t device_id = grad_output.device().index;

    // Create grad_input tensor initialized to zeros
    Tensor grad_input = dispatchZeros({N, C, H_in, W_in}, grad_output.dtype(), grad_output.device());

    // Select shader based on dtype
    std::string shader_name = "max_pool2d_backward_indices";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "max_pool2d_backward_indices_f64";
    } else if (grad_output.dtype() == DType::Float16) {
        shader_name = "max_pool2d_backward_indices_f16";
    } else if (grad_output.dtype() != DType::Float32) {
        throw std::runtime_error("Unsupported dtype for max_pool2d_backward_with_indices");
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    VkBuffer buffer_grad_out = getVulkanBuffer(const_cast<void*>(grad_output.data_ptr()));
    VkBuffer buffer_indices = getVulkanBuffer(const_cast<void*>(indices.data_ptr()));
    VkBuffer buffer_grad_in = getVulkanBuffer(grad_input.data_ptr());

    size_t grad_out_size = grad_out_numel * grad_output.dtype_size();
    size_t indices_size = indices.numel() * indices.dtype_size();
    size_t grad_in_size = grad_in_numel * grad_input.dtype_size();

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_grad_out},
        {1, buffer_indices},
        {2, buffer_grad_in}
    };
    std::vector<size_t> sizes = {grad_out_size, indices_size, grad_in_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    struct PushConstants {
        uint32_t n_elements;
        uint32_t grad_input_size;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(grad_out_numel);
    push_constants.grad_input_size = static_cast<uint32_t>(grad_in_numel);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = (grad_out_numel + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return grad_input;
}

// ============================================================================
// Conv2d Forward Operation (OpAttributes version)
// ============================================================================

auto VulkanBackend::dispatchConv2dForward(const Tensor& input, const Tensor& weight, const OpAttributes& attrs) -> Tensor {
    // For Float16, upcast to Float32 to avoid overflow in accumulation
    // (conv2d sums over kernel*channels elements, result can exceed F16 max 65504)
    if (input.dtype() == DType::Float16) {
        auto input_f32 = input.to(DType::Float32);
        auto weight_f32 = weight.to(DType::Float32);
        // Pass through bias handling - attrs may reference a bias tensor
        // that also needs conversion, handled by the F32 path
        auto result_f32 = dispatchConv2dForward(input_f32, weight_f32, attrs);
        // Saturating conversion: clamp to Float16 representable range to prevent
        // Inf/NaN from overflow when converting back to Float16
        result_f32 = dispatchClamp(result_f32, -65504.0f, 65504.0f);
        return result_f32.to(DType::Float16);
    }

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
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// ConvTranspose2d Forward Operation
// ============================================================================

auto VulkanBackend::dispatchConvTranspose2dForward(const Tensor& input, const Tensor& weight, const Tensor* bias, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    auto weight_shape = weight.shape();

    if (input_shape.size() != 4) {
        throw std::invalid_argument("conv_transpose2d_forward requires 4D input (N, C, H, W)");
    }
    if (weight_shape.size() != 4) {
        throw std::invalid_argument("conv_transpose2d_forward requires 4D weight (in_channels, out_channels/groups, kH, kW)");
    }

    // Extract attributes
    int64_t stride = attrs.contains("stride") ? std::stoll(attrs.at("stride")) : 1;
    int64_t padding = attrs.contains("padding") ? std::stoll(attrs.at("padding")) : 0;
    int64_t output_padding = attrs.contains("output_padding") ? std::stoll(attrs.at("output_padding")) : 0;
    int64_t dilation = attrs.contains("dilation") ? std::stoll(attrs.at("dilation")) : 1;
    int64_t groups = attrs.contains("groups") ? std::stoll(attrs.at("groups")) : 1;
    bool has_bias = (bias != nullptr);

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    // Weight shape for transposed conv: [in_channels, out_channels/groups, kH, kW]
    int64_t out_channels_per_group = weight_shape[1];
    int64_t out_channels = out_channels_per_group * groups;
    int64_t kernel_h = weight_shape[2];
    int64_t kernel_w = weight_shape[3];

    // Calculate output dimensions for transposed convolution
    // out = (in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
    int64_t out_height = (in_height - 1) * stride - 2 * padding + dilation * (kernel_h - 1) + output_padding + 1;
    int64_t out_width = (in_width - 1) * stride - 2 * padding + dilation * (kernel_w - 1) + output_padding + 1;

    if (out_height <= 0 || out_width <= 0) {
        throw std::invalid_argument("Invalid conv_transpose2d configuration: output dimensions are non-positive");
    }

    int32_t device_id = input.device().index;

    // Select pipeline based on dtype
    std::string shader_name = "conv_transpose2d_forward";
    if (input.dtype() == DType::Float64) shader_name = "conv_transpose2d_forward_f64";
    else if (input.dtype() == DType::Float16) shader_name = "conv_transpose2d_forward_f16";
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape = {batch, out_channels, out_height, out_width};
    Tensor output(output_shape, input.dtype(), input.device());

    // Get VkBuffer handles
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_weight = getVulkanBuffer(weight.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());
    VkBuffer buffer_bias = has_bias ? getVulkanBuffer(bias->data_ptr()) : buffer_output;

    // Calculate buffer sizes
    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_weight = weight.numel() * weight.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();
    size_t buffer_size_bias = has_bias ? (bias->numel() * bias->dtype_size()) : 4;

    // Setup descriptor set bindings (input, weight, bias, output)
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_input},
        {1, buffer_weight},
        {2, buffer_bias},
        {3, buffer_output}
    };
    std::vector<size_t> sizes = {
        buffer_size_input,
        buffer_size_weight,
        buffer_size_bias,
        buffer_size_output
    };

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
        uint32_t output_padding;
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
    push_constants.output_padding = static_cast<uint32_t>(output_padding);
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
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Full Operation - Create tensor filled with specific value
// ============================================================================

auto VulkanBackend::dispatchFull(const std::vector<int64_t>& shape, float value, DType dtype) -> Tensor {
    // BFloat16: generate as Float32, then convert
    if (dtype == DType::BFloat16) {
        auto result_f32 = dispatchFull(shape, value, DType::Float32);
        return result_f32.to(DType::BFloat16);
    }

    // Create tensor on first available Vulkan device
    Device device(Device::Type::Vulkan, 0);
    Tensor output(shape, dtype, device);

    // Handle empty tensors - no GPU work needed
    int64_t numel = 1;
    for (auto dim : shape) {
        numel *= dim;
    }
    if (numel == 0) {
        return output;
    }

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

        // Add memory barrier
        insertComputeBarrier(cmdBuffer);

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
        insertComputeBarrier(cmdBuffer);

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
        insertComputeBarrier(cmdBuffer);

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
        insertComputeBarrier(cmdBuffer);

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
        insertComputeBarrier(cmdBuffer);

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
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

// ============================================================================
// Ones Operation - Create tensor filled with 1.0
// ============================================================================

auto VulkanBackend::dispatchOnes(const std::vector<int64_t>& shape, DType dtype) -> Tensor {
    // For Int64, UInt8, Bool, or BFloat16, use full() instead since ones shader only supports 32-bit values
    if (dtype == DType::Int64 || dtype == DType::UInt8 || dtype == DType::Bool || dtype == DType::BFloat16) {
        return dispatchFull(shape, 1.0, dtype);
    }

    // Float64 uses dedicated ones_f64 shader
    if (dtype == DType::Float64) {
        Device device(Device::Type::Vulkan, 0);
        Tensor output(shape, dtype, device);
        int32_t device_id = device.index;

        auto* pipeline = getPipeline("ones_f64", device_id);

        VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());
        size_t buffer_size_out = output.numel() * output.dtype_size();

        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {{0, buffer_out}};
        std::vector<size_t> sizes = {buffer_size_out};

        VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
            device_id, pipeline, bindings, sizes);

        struct PushConstantsF64 {
            uint32_t n_elements;
        } push_constants;
        push_constants.n_elements = static_cast<uint32_t>(output.numel());

        VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(PushConstantsF64), &push_constants);

        uint32_t workgroups = (output.numel() + 255) / 256;
        vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

        insertComputeBarrier(cmdBuffer);
        endSingleTimeCommands(cmdBuffer, device_id);

        return output;
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
        insertComputeBarrier(cmdBuffer);

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
        insertComputeBarrier(cmdBuffer);

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
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchRand(const std::vector<int64_t>& shape, DType dtype) -> Tensor {
    // BFloat16: generate as Float32, then convert
    if (dtype == DType::BFloat16) {
        auto result_f32 = dispatchRand(shape, DType::Float32);
        return result_f32.to(DType::BFloat16);
    }

    // Create tensor on first available Vulkan device
    Device device(Device::Type::Vulkan, 0);
    Tensor output(shape, dtype, device);

    size_t numel = output.numel();
    if (numel == 0) {
        return output;
    }

    // Use GPU Philox RNG for random number generation
    int32_t device_id = device.index;

    // Select shader based on dtype
    std::string shader_name = "random";
    if (dtype == DType::Float64) {
        shader_name = "random_f64";
    } else if (dtype == DType::Float16) {
        shader_name = "random_f16";
    } else if (dtype != DType::Float32) {
        throw std::runtime_error("Unsupported dtype for rand: only Float32, Float64, Float16, and BFloat16 are supported");
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());
    size_t buffer_size = numel * output.dtype_size();

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {{0, buffer_out}};
    std::vector<size_t> sizes = {buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Generate seed from hardware random
    static std::random_device rd;
    static std::atomic<uint32_t> offset_counter{0};

    struct PushConstants {
        uint32_t n_elements;
        uint32_t seed;
        uint32_t offset;
        uint32_t distribution;  // 0 = uniform
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(numel);
    push_constants.seed = rd();
    push_constants.offset = offset_counter.fetch_add(static_cast<uint32_t>(numel));
    push_constants.distribution = 0;  // uniform

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = (numel + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchRandn(const std::vector<int64_t>& shape, DType dtype) -> Tensor {
    // BFloat16: generate as Float32, then convert
    if (dtype == DType::BFloat16) {
        auto result_f32 = dispatchRandn(shape, DType::Float32);
        return result_f32.to(DType::BFloat16);
    }

    // Create tensor on first available Vulkan device
    Device device(Device::Type::Vulkan, 0);
    Tensor output(shape, dtype, device);

    size_t numel = output.numel();
    if (numel == 0) {
        return output;
    }

    // Use GPU Philox RNG with Box-Muller transform for normal distribution
    int32_t device_id = device.index;

    // Select shader based on dtype
    std::string shader_name = "random";
    if (dtype == DType::Float64) {
        shader_name = "random_f64";
    } else if (dtype == DType::Float16) {
        shader_name = "random_f16";
    } else if (dtype != DType::Float32) {
        throw std::runtime_error("Unsupported dtype for randn: only Float32, Float64, Float16, and BFloat16 are supported");
    }

    auto* pipeline = getPipeline(shader_name, device_id);

    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());
    size_t buffer_size = numel * output.dtype_size();

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {{0, buffer_out}};
    std::vector<size_t> sizes = {buffer_size};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    // Generate seed from hardware random
    static std::random_device rd;
    static std::atomic<uint32_t> offset_counter{0};

    struct PushConstants {
        uint32_t n_elements;
        uint32_t seed;
        uint32_t offset;
        uint32_t distribution;  // 1 = normal
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(numel);
    push_constants.seed = rd();
    push_constants.offset = offset_counter.fetch_add(static_cast<uint32_t>(numel));
    push_constants.distribution = 1;  // normal

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = (numel + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

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
    // Validate shapes match
    auto input_shape = input.shape();
    auto mask_shape = mask.shape();
    if (!std::equal(input_shape.begin(), input_shape.end(), mask_shape.begin(), mask_shape.end())) {
        throw std::invalid_argument("masked_select: input and mask must have same shape");
    }

    if (mask.dtype() != DType::Bool && mask.dtype() != DType::Float32) {
        throw std::invalid_argument("masked_select: mask tensor must have dtype Bool or Float32");
    }

    const int64_t numel = input.numel();
    if (numel == 0) {
        return Tensor({0}, input.dtype(), input.device());
    }

    // Determine gather shader based on dtype
    std::string gather_shader = "masked_select_gather";
    if (input.dtype() == DType::Float64) {
        gather_shader = "masked_select_gather_f64";
    } else if (input.dtype() == DType::Float16) {
        gather_shader = "masked_select_gather_f16";
    } else if (input.dtype() != DType::Float32) {
        // CPU fallback for unsupported dtypes (Int32, Int64, Bool, etc.)
        Device cpu_device(Device::Type::CPU, 0);
        Tensor cpu_input = input.to(cpu_device);
        Tensor cpu_mask = mask.to(cpu_device);
        std::vector<Tensor> cpu_inputs = {cpu_input, cpu_mask};
        auto cpu_results = tenzor::dispatch(OpId::MaskedSelect, cpu_inputs, {});
        return cpu_results[0].to(input.device());
    }

    int32_t device_id = input.device().index;
    uint32_t n = static_cast<uint32_t>(numel);
    uint32_t mask_is_float = (mask.dtype() == DType::Float32) ? 1 : 0;
    uint32_t n_workgroups = (n + 255) / 256;

    // ---- Pass 1: Count true elements ----
    // Allocate count buffer (one per workgroup + 1 for total)
    Tensor count_buf({static_cast<int64_t>(n_workgroups + 1)}, DType::Int32, input.device());
    count_buf = dispatchFill(count_buf, 0.0f);

    VkBuffer buffer_mask = getVulkanBuffer(mask.data_ptr());
    VkBuffer buffer_count = getVulkanBuffer(count_buf.data_ptr());
    size_t mask_bytes = mask.numel() * mask.dtype_size();
    size_t count_bytes = count_buf.numel() * count_buf.dtype_size();

    // Pass 1a: Per-workgroup count
    {
        auto* pipeline = getPipeline("masked_select_count", device_id);
        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {{0, buffer_mask}, {1, buffer_count}};
        std::vector<size_t> sizes = {mask_bytes, count_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct { uint32_t n_elements; uint32_t mask_is_float; uint32_t pass; } pc;
        pc.n_elements = n; pc.mask_is_float = mask_is_float; pc.pass = 0;

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, n_workgroups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Pass 1b: Sum workgroup counts
    {
        auto* pipeline = getPipeline("masked_select_count", device_id);
        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {{0, buffer_mask}, {1, buffer_count}};
        std::vector<size_t> sizes = {mask_bytes, count_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct { uint32_t n_elements; uint32_t mask_is_float; uint32_t pass; } pc;
        pc.n_elements = n; pc.mask_is_float = mask_is_float; pc.pass = 1;

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, 1, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Read back total count (GPU→CPU sync required for variable-size output)
    Tensor count_cpu = count_buf.to(Device::cpu());
    int64_t total_count = static_cast<int64_t>(count_cpu.data<int32_t>()[0]);

    if (total_count == 0) {
        return Tensor({0}, input.dtype(), input.device());
    }

    // ---- Pass 2: Prefix sum ----
    Tensor prefix_sums({static_cast<int64_t>(n)}, DType::Int32, input.device());
    prefix_sums = dispatchFill(prefix_sums, 0.0f);
    Tensor block_sums({static_cast<int64_t>(n_workgroups)}, DType::Int32, input.device());
    block_sums = dispatchFill(block_sums, 0.0f);

    VkBuffer buffer_prefix = getVulkanBuffer(prefix_sums.data_ptr());
    VkBuffer buffer_blocks = getVulkanBuffer(block_sums.data_ptr());
    size_t prefix_bytes = prefix_sums.numel() * prefix_sums.dtype_size();
    size_t blocks_bytes = block_sums.numel() * block_sums.dtype_size();

    // Pass 2a: Local scan
    {
        auto* pipeline = getPipeline("prefix_sum", device_id);
        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {{0, buffer_mask}, {1, buffer_prefix}, {2, buffer_blocks}};
        std::vector<size_t> sizes = {mask_bytes, prefix_bytes, blocks_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct { uint32_t n_elements; uint32_t mask_is_float; uint32_t pass; } pc;
        pc.n_elements = n; pc.mask_is_float = mask_is_float; pc.pass = 0;

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, n_workgroups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // Pass 2b: Add block offsets
    if (n_workgroups > 1) {
        auto* pipeline = getPipeline("prefix_sum", device_id);
        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {{0, buffer_mask}, {1, buffer_prefix}, {2, buffer_blocks}};
        std::vector<size_t> sizes = {mask_bytes, prefix_bytes, blocks_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct { uint32_t n_elements; uint32_t mask_is_float; uint32_t pass; } pc;
        pc.n_elements = n; pc.mask_is_float = mask_is_float; pc.pass = 1;

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, n_workgroups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    // ---- Pass 3: Gather selected elements ----
    Tensor output({total_count}, input.dtype(), input.device());
    // F16 gather shader uses atomicOr on packed uint32, so output must be zeroed
    if (input.dtype() == DType::Float16) {
        output = dispatchFill(output, 0.0f);
    }

    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());
    size_t input_bytes = input.numel() * input.dtype_size();
    size_t output_bytes = output.numel() * output.dtype_size();

    {
        auto* pipeline = getPipeline(gather_shader, device_id);
        std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
            {0, buffer_input}, {1, buffer_mask}, {2, buffer_output}, {3, buffer_prefix}
        };
        std::vector<size_t> sizes = {input_bytes, mask_bytes, output_bytes, prefix_bytes};
        VkDescriptorSet ds = allocateAndWriteDescriptorSet(device_id, pipeline, bindings, sizes);

        struct { uint32_t n_elements; uint32_t mask_is_float; uint32_t output_size; } pc;
        pc.n_elements = n; pc.mask_is_float = mask_is_float; pc.output_size = static_cast<uint32_t>(total_count);

        VkCommandBuffer cmd = beginSingleTimeCommands(device_id);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->layout(), 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, pipeline->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, n_workgroups, 1, 1);
        insertComputeBarrier(cmd);
        endSingleTimeCommands(cmd, device_id);
        synchronize(device_id);
    }

    return output;
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
// ============================================================================
// Interpolation Operation Implementation
// ============================================================================

auto VulkanBackend::dispatchInterpolate(const Tensor& input, const OpAttributes& attrs) -> Tensor {
    auto input_shape = input.shape();
    if (input_shape.size() != 4) {
        throw std::invalid_argument("interpolate requires 4D input (N, C, H, W), got " +
                                    std::to_string(input_shape.size()) + "D");
    }

    // Extract attributes
    std::string mode = attrs.at("mode");
    bool align_corners = attrs.contains("align_corners") && attrs.at("align_corners") == "1";

    // Parse output size from comma-separated string
    std::string size_str = attrs.at("size");
    auto comma_pos = size_str.find(',');
    int64_t out_height = std::stoll(size_str.substr(0, comma_pos));
    int64_t out_width = std::stoll(size_str.substr(comma_pos + 1));

    int64_t batch = input_shape[0];
    int64_t channels = input_shape[1];
    int64_t in_height = input_shape[2];
    int64_t in_width = input_shape[3];

    int32_t device_id = input.device().index;
    bool is_float64 = (input.dtype() == DType::Float64);
    bool is_float16 = (input.dtype() == DType::Float16);

    // Select shader based on mode and dtype
    std::string shader_name;
    if (mode == "bilinear" || mode == "bicubic") {
        shader_name = is_float64 ? "bilinear_interpolate_f64" :
                      is_float16 ? "bilinear_interpolate_f16" : "bilinear_interpolate";
    } else {
        shader_name = is_float64 ? "nearest_interpolate_f64" :
                      is_float16 ? "nearest_interpolate_f16" : "nearest_interpolate";
    }

    auto* pipeline = getPipeline(shader_name, device_id);

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
        uint32_t align_corners;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.batch = static_cast<uint32_t>(batch);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.in_height = static_cast<uint32_t>(in_height);
    push_constants.in_width = static_cast<uint32_t>(in_width);
    push_constants.out_height = static_cast<uint32_t>(out_height);
    push_constants.out_width = static_cast<uint32_t>(out_width);
    push_constants.align_corners = align_corners ? 1u : 0u;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch workgroups
    uint32_t workgroups = static_cast<uint32_t>((output.numel() + 255) / 256);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    // Add memory barrier
    insertComputeBarrier(cmdBuffer);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchROIAlignForward(const Tensor& features, const Tensor& rois, const OpAttributes& attrs) -> Tensor {
    auto feat_shape = features.shape();
    if (feat_shape.size() != 4) {
        throw std::invalid_argument("roi_align_forward requires 4D features (N, C, H, W), got " +
                                    std::to_string(feat_shape.size()) + "D");
    }
    if (rois.ndim() != 2 || rois.shape()[1] != 5) {
        throw std::invalid_argument("roi_align_forward requires rois of shape (num_rois, 5)");
    }

    int64_t channels = feat_shape[1];
    int64_t feat_height = feat_shape[2];
    int64_t feat_width = feat_shape[3];
    int64_t num_rois = rois.shape()[0];

    int64_t output_h = std::stoll(attrs.at("output_h"));
    int64_t output_w = std::stoll(attrs.at("output_w"));
    float spatial_scale = std::stof(attrs.at("spatial_scale"));
    int64_t sampling_ratio = attrs.contains("sampling_ratio") ? std::stoll(attrs.at("sampling_ratio")) : 0;
    bool aligned = attrs.contains("aligned") && (attrs.at("aligned") == "1" || attrs.at("aligned") == "true");

    int32_t device_id = features.device().index;

    // Select shader based on dtype
    std::string shader_name = "roi_align";
    if (features.dtype() == DType::Float64) {
        shader_name = "roi_align_f64";
    } else if (features.dtype() == DType::Float16) {
        shader_name = "roi_align_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create output tensor
    std::vector<int64_t> output_shape = {num_rois, channels, output_h, output_w};
    Tensor output(output_shape, features.dtype(), features.device());

    // Get VkBuffer handles
    VkBuffer buffer_features = getVulkanBuffer(features.data_ptr());
    VkBuffer buffer_rois = getVulkanBuffer(rois.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());

    size_t buffer_size_features = features.numel() * features.dtype_size();
    size_t buffer_size_rois = rois.numel() * rois.dtype_size();
    size_t buffer_size_output = output.numel() * output.dtype_size();

    // Allocate and write descriptor set (3 bindings)
    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_features},
        {1, buffer_rois},
        {2, buffer_output}
    };
    std::vector<size_t> sizes = {buffer_size_features, buffer_size_rois, buffer_size_output};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Push constants: 10 uint32_t values = 40 bytes
    struct PushConstants {
        uint32_t n_elements;
        uint32_t num_rois;
        uint32_t channels;
        uint32_t feat_height;
        uint32_t feat_width;
        uint32_t output_h;
        uint32_t output_w;
        uint32_t spatial_scale_bits;
        uint32_t sampling_ratio;
        uint32_t aligned;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.num_rois = static_cast<uint32_t>(num_rois);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.feat_height = static_cast<uint32_t>(feat_height);
    push_constants.feat_width = static_cast<uint32_t>(feat_width);
    push_constants.output_h = static_cast<uint32_t>(output_h);
    push_constants.output_w = static_cast<uint32_t>(output_w);
    // Pass float as uint bits
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &spatial_scale, sizeof(float));
    push_constants.spatial_scale_bits = scale_bits;
    push_constants.sampling_ratio = static_cast<uint32_t>(sampling_ratio);
    push_constants.aligned = aligned ? 1u : 0u;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = static_cast<uint32_t>((output.numel() + 255) / 256);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchROIAlignBackward(const Tensor& grad_output, const Tensor& rois, const OpAttributes& attrs) -> Tensor {
    auto grad_shape = grad_output.shape();
    if (grad_shape.size() != 4) {
        throw std::invalid_argument("roi_align_backward requires 4D grad_output (num_rois, C, output_h, output_w)");
    }

    int64_t num_rois = grad_shape[0];
    int64_t channels = grad_shape[1];
    int64_t output_h = grad_shape[2];
    int64_t output_w = grad_shape[3];

    // Extract feature map dimensions from attributes
    int64_t batch_size = std::stoll(attrs.at("batch_size"));
    int64_t feat_height = std::stoll(attrs.at("feat_height"));
    int64_t feat_width = std::stoll(attrs.at("feat_width"));
    float spatial_scale = std::stof(attrs.at("spatial_scale"));
    int64_t sampling_ratio = attrs.contains("sampling_ratio") ? std::stoll(attrs.at("sampling_ratio")) : 0;
    bool aligned = attrs.contains("aligned") && (attrs.at("aligned") == "1" || attrs.at("aligned") == "true");

    int32_t device_id = grad_output.device().index;

    // Select shader based on dtype
    std::string shader_name = "roi_align_backward";
    if (grad_output.dtype() == DType::Float64) {
        shader_name = "roi_align_backward_f64";
    } else if (grad_output.dtype() == DType::Float16) {
        shader_name = "roi_align_backward_f16";
    }
    auto* pipeline = getPipeline(shader_name, device_id);

    // Create grad_features output tensor (same shape as original features)
    // For f16 backward, we accumulate in f32 then convert
    DType accum_dtype = (grad_output.dtype() == DType::Float16) ? DType::Float32 : grad_output.dtype();
    std::vector<int64_t> grad_features_shape = {batch_size, channels, feat_height, feat_width};
    Tensor grad_features(grad_features_shape, accum_dtype, grad_output.device());

    // Zero-initialize grad_features (atomicAdd accumulates into it)
    {
        std::array<Tensor, 1> fill_inputs = {grad_features};
        OpAttributes fill_attrs = {{"value", "0"}};
        auto zero_result = dispatch("fill", fill_inputs, fill_attrs);
        if (!zero_result.empty()) {
            grad_features = zero_result[0];
        }
    }

    VkBuffer buffer_grad_output = getVulkanBuffer(grad_output.data_ptr());
    VkBuffer buffer_rois = getVulkanBuffer(rois.data_ptr());
    VkBuffer buffer_grad_features = getVulkanBuffer(grad_features.data_ptr());

    size_t buffer_size_grad_output = grad_output.numel() * grad_output.dtype_size();
    size_t buffer_size_rois = rois.numel() * rois.dtype_size();
    size_t buffer_size_grad_features = grad_features.numel() * grad_features.dtype_size();

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_grad_output},
        {1, buffer_rois},
        {2, buffer_grad_features}
    };
    std::vector<size_t> sizes = {buffer_size_grad_output, buffer_size_rois, buffer_size_grad_features};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

    VkCommandBuffer cmdBuffer = beginSingleTimeCommands(device_id);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipeline->layout(), 0, 1, &descriptorSet, 0, nullptr);

    // Push constants: 11 uint32_t values = 44 bytes
    struct PushConstants {
        uint32_t n_elements;
        uint32_t num_rois;
        uint32_t channels;
        uint32_t feat_height;
        uint32_t feat_width;
        uint32_t output_h;
        uint32_t output_w;
        uint32_t spatial_scale_bits;
        uint32_t sampling_ratio;
        uint32_t aligned;
        uint32_t batch_size;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(grad_output.numel());
    push_constants.num_rois = static_cast<uint32_t>(num_rois);
    push_constants.channels = static_cast<uint32_t>(channels);
    push_constants.feat_height = static_cast<uint32_t>(feat_height);
    push_constants.feat_width = static_cast<uint32_t>(feat_width);
    push_constants.output_h = static_cast<uint32_t>(output_h);
    push_constants.output_w = static_cast<uint32_t>(output_w);
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &spatial_scale, sizeof(float));
    push_constants.spatial_scale_bits = scale_bits;
    push_constants.sampling_ratio = static_cast<uint32_t>(sampling_ratio);
    push_constants.aligned = aligned ? 1u : 0u;
    push_constants.batch_size = static_cast<uint32_t>(batch_size);

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = static_cast<uint32_t>((grad_output.numel() + 255) / 256);
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    insertComputeBarrier(cmdBuffer);
    endSingleTimeCommands(cmdBuffer, device_id);

    // Convert back from f32 accumulation buffer to f16 if needed
    if (grad_output.dtype() == DType::Float16) {
        grad_features = grad_features.to(DType::Float16);
    }

    return grad_features;
}

auto VulkanBackend::dispatchArgSort(const Tensor& input, int64_t dim, bool descending) -> Tensor {
    auto input_shape = input.shape();
    const int ndim = static_cast<int>(input_shape.size());

    // Normalize dim
    if (dim < 0) dim += ndim;

    const int64_t sort_size = input_shape[dim];

    // Determine shader based on dtype
    std::string sort_shader;
    DType work_dtype = DType::Float32;
    size_t elem_size = sizeof(float);
    if (input.dtype() == DType::Float32) {
        sort_shader = "bitonic_sort";
        work_dtype = DType::Float32;
        elem_size = sizeof(float);
    } else if (input.dtype() == DType::Float64) {
        sort_shader = "bitonic_sort_f64";
        work_dtype = DType::Float64;
        elem_size = sizeof(double);
    } else if (input.dtype() == DType::Int32) {
        sort_shader = "bitonic_sort_i32";
        work_dtype = DType::Int32;
        elem_size = sizeof(int32_t);
    } else {
        sort_shader = "";
    }

    // CPU fallback for:
    // 1. Unsupported dtype (no shader)
    // 2. Sort dimension > 2^20 (diminishing returns for bitonic sort O(n log^2 n))
    // 3. Sort not along last dimension (would need strided access)
    if (sort_shader.empty() || sort_size > (1 << 20) || dim != ndim - 1) {
        Device cpu_device(Device::Type::CPU, 0);
        Tensor cpu_input = input.to(cpu_device);
        std::vector<Tensor> cpu_inputs = {cpu_input};
        OpAttributes attrs;
        attrs["dim"] = std::to_string(dim);
        attrs["descending"] = descending ? "1" : "0";
        auto result = tenzor::dispatch(OpId::ArgSort, cpu_inputs, attrs)[0];
        return result.to(input.device());
    }

    if (sort_size <= 1) {
        std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
        Tensor result(shape_vec, DType::Int64, input.device());
        result = dispatchFill(result, 0.0f);
        return result;
    }

    int32_t device_id = input.device().index;

    // Compute padded size (next power of 2)
    uint32_t n = static_cast<uint32_t>(sort_size);
    uint32_t padded_n = 1;
    while (padded_n < n) padded_n <<= 1;

    // Number of bitonic sort stages = log2(padded_n)
    uint32_t num_stages = 0;
    { uint32_t tmp = padded_n; while (tmp > 1) { num_stages++; tmp >>= 1; } }

    // Number of independent sort slices (sort along last dim)
    int64_t num_slices = 1;
    for (int i = 0; i < ndim - 1; ++i) num_slices *= input_shape[i];

    // Output tensor (Int64 indices)
    std::vector<int64_t> shape_vec(input_shape.begin(), input_shape.end());
    Tensor output(shape_vec, DType::Int64, input.device());

    // Allocate working buffers for bitonic sort (padded to power-of-2)
    Tensor work_values({static_cast<int64_t>(padded_n)}, work_dtype, input.device());
    Tensor work_indices({static_cast<int64_t>(padded_n)}, DType::Int32, input.device());

    float pad_value = descending ? -std::numeric_limits<float>::infinity()
                                 : std::numeric_limits<float>::infinity();
    // For Int32, use max/min int as pad value
    if (work_dtype == DType::Int32) {
        pad_value = descending ? static_cast<float>(std::numeric_limits<int32_t>::min())
                               : static_cast<float>(std::numeric_limits<int32_t>::max());
    }

    size_t values_bytes = padded_n * elem_size;
    size_t indices_bytes = padded_n * sizeof(int32_t);

    auto* pipeline = getPipeline(sort_shader, device_id);
    uint32_t workgroups = (padded_n / 2 + 255) / 256;

    // Pre-build initial index array on CPU (reused for each slice)
    std::vector<int32_t> init_indices(padded_n);
    for (uint32_t i = 0; i < padded_n; ++i) {
        init_indices[i] = (i < n) ? static_cast<int32_t>(i) : static_cast<int32_t>(n);
    }

    for (int64_t slice = 0; slice < num_slices; ++slice) {
        // Step 1: Initialize working buffers
        // Fill values buffer with pad value (inf for ascending, -inf for descending)
        work_values = dispatchFill(work_values, pad_value);
        synchronize(device_id);

        // Copy this slice's data into the start of the working buffer
        size_t slice_bytes = sort_size * elem_size;
        copy(work_values.data_ptr(),
             static_cast<const char*>(input.data_ptr()) + slice * slice_bytes,
             slice_bytes, CopyKind::DeviceToDevice);
        synchronize(device_id);

        // Upload initial indices (0, 1, ..., n-1, n, n, ..., n)
        copy(work_indices.data_ptr(), init_indices.data(),
             padded_n * sizeof(int32_t), CopyKind::HostToDevice);
        synchronize(device_id);

        // Step 2: Run all bitonic sort passes in a single command buffer
        // Get VkBuffer handles after fill/copy since dispatchFill may reallocate
        {
            VkBuffer buffer_values = getVulkanBuffer(work_values.data_ptr());
            VkBuffer buffer_indices = getVulkanBuffer(work_indices.data_ptr());
            std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
                {0, buffer_values}, {1, buffer_indices}
            };
            std::vector<size_t> sizes = {values_bytes, indices_bytes};
            VkDescriptorSet ds = allocateAndWriteDescriptorSet(
                device_id, pipeline, bindings, sizes);

            VkCommandBuffer cmd = beginSingleTimeCommands(device_id);

            for (uint32_t stage = 0; stage < num_stages; ++stage) {
                for (int32_t substage = static_cast<int32_t>(stage); substage >= 0; --substage) {
                    struct {
                        uint32_t n;
                        uint32_t padded_n;
                        uint32_t stage;
                        uint32_t substage;
                        uint32_t descending;
                    } pc;
                    pc.n = n;
                    pc.padded_n = padded_n;
                    pc.stage = stage;
                    pc.substage = static_cast<uint32_t>(substage);
                    pc.descending = descending ? 1 : 0;

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipeline());
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                           pipeline->layout(), 0, 1, &ds, 0, nullptr);
                    vkCmdPushConstants(cmd, pipeline->layout(),
                                     VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                    vkCmdDispatch(cmd, workgroups, 1, 1);
                    insertComputeBarrier(cmd);
                }
            }

            endSingleTimeCommands(cmd, device_id);
            synchronize(device_id);
        }

        // Step 3: Read sorted indices and convert Int32 → Int64
        {
            Tensor cpu_indices = work_indices.to(Device::cpu());
            const int32_t* idx_src = cpu_indices.data<int32_t>();
            std::vector<int64_t> int64_indices(sort_size);
            for (int64_t i = 0; i < sort_size; ++i) {
                int64_indices[i] = static_cast<int64_t>(idx_src[i]);
            }
            void* dst_ptr = static_cast<char*>(output.data_ptr()) +
                            slice * sort_size * sizeof(int64_t);
            copy(dst_ptr, int64_indices.data(),
                 sort_size * sizeof(int64_t), CopyKind::HostToDevice);
            synchronize(device_id);
        }
    }

    return output;
}

} // namespace tenzor
