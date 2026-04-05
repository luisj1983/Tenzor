/**
 * @file vulkan_backend.cpp
 * @brief Vulkan compute backend - initialization, device queries, pipeline management, stream ops, factory
 */

#include "vulkan_helpers.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/backend/vulkan_caching_allocator.hpp"
#ifdef TENZOR_HAS_VMA
#include "tenzor/backend/vulkan_vma_allocator.hpp"
#endif

// Undefine Vulkan Bool macro that conflicts with DType::Bool
#ifdef Bool
#undef Bool
#endif

#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

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
    // Wait for all devices to finish (skip lost devices — resources are invalid)
    for (auto& ctx : devices_) {
        if (ctx.device != VK_NULL_HANDLE && !ctx.device_lost) {
            vkDeviceWaitIdle(ctx.device);
        }
    }

    // Flush any deferred frees now that all GPU work is complete
    if (is_backend_registry_alive()) {
        for (size_t i = 0; i < deferred_frees_.size(); ++i) {
            flush_deferred_frees(static_cast<int32_t>(i));
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

    // Cleanup staging buffer pools
    stagingPools_.clear();

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
                        // Atomic write: write to temp file, then rename
                        std::string tmp_path = path + ".tmp";
                        std::ofstream out(tmp_path, std::ios::binary);
                        if (out.good()) {
                            out.write(cache_data.data(), static_cast<std::streamsize>(cache_size));
                            out.close();
                            if (out.good()) {
                                if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
                                    // Rename failed — clean up temp file
                                    std::remove(tmp_path.c_str());
                                }
                            } else {
                                std::remove(tmp_path.c_str());
                            }
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

    // Cleanup debug messenger before instance
    if (debug_messenger_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE) {
        auto vkDestroyDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (vkDestroyDebugUtilsMessengerEXT) {
            vkDestroyDebugUtilsMessengerEXT(instance_, debug_messenger_, nullptr);
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
            vkCreateDebugUtilsMessengerEXT(instance_, &debugInfo, nullptr, &debug_messenger_);
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

        bool canPreserveDenormsF32 = false;
        if (hasFloatControls) {
            VkPhysicalDeviceProperties2 deviceProps2{};
            deviceProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            deviceProps2.pNext = &floatControlsProps;
            vkGetPhysicalDeviceProperties2(ctx.physicalDevice, &deviceProps2);

            // Check if denorm preserve is supported for float32
            canPreserveDenormsF32 = (floatControlsProps.shaderDenormPreserveFloat32 == VK_TRUE);
        }

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
        ctx.hasAtomicFloat = hasAtomicFloat;

        // Query subgroup properties for subgroup arithmetic support
        {
            VkPhysicalDeviceSubgroupProperties subgroupProps{};
            subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
            VkPhysicalDeviceProperties2 props2{};
            props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            props2.pNext = &subgroupProps;
            vkGetPhysicalDeviceProperties2(ctx.physicalDevice, &props2);
            ctx.subgroupSize = subgroupProps.subgroupSize;
            ctx.hasSubgroupArithmetic =
                (subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
        }

        // Determine optimal 1D workgroup size from device limits
        ctx.workgroupSize = vulkan::optimalWorkgroupSize(ctx.physicalDevice);

        // Store maximum workgroup counts for dispatch validation
        {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(ctx.physicalDevice, &props);
            ctx.maxComputeWorkGroupCount[0] = props.limits.maxComputeWorkGroupCount[0];
            ctx.maxComputeWorkGroupCount[1] = props.limits.maxComputeWorkGroupCount[1];
            ctx.maxComputeWorkGroupCount[2] = props.limits.maxComputeWorkGroupCount[2];
        }

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
                    // Validate: file must be large enough for VkPipelineCacheHeaderVersionOne
                    // Header: headerSize(4) + version(4) + vendorID(4) + deviceID(4) + UUID(16) = 32 bytes
                    constexpr size_t VK_CACHE_HEADER_SIZE = sizeof(uint32_t) * 4 + VK_UUID_SIZE;
                    if (size >= static_cast<std::streampos>(VK_CACHE_HEADER_SIZE)) {
                        cache_data.resize(static_cast<size_t>(size));
                        cache_file.seekg(0);
                        cache_file.read(cache_data.data(), size);
                        // Validate pipeline cache header: version, vendor ID, device ID, and UUID
                        uint32_t header_size = 0;
                        std::memcpy(&header_size, cache_data.data(), sizeof(uint32_t));
                        uint32_t header_version = 0;
                        std::memcpy(&header_version, cache_data.data() + sizeof(uint32_t), sizeof(uint32_t));
                        uint32_t cache_vendor_id = 0;
                        std::memcpy(&cache_vendor_id, cache_data.data() + sizeof(uint32_t) * 2, sizeof(uint32_t));
                        uint32_t cache_device_id = 0;
                        std::memcpy(&cache_device_id, cache_data.data() + sizeof(uint32_t) * 3, sizeof(uint32_t));
                        uint8_t cache_uuid[VK_UUID_SIZE];
                        std::memcpy(cache_uuid, cache_data.data() + sizeof(uint32_t) * 4, VK_UUID_SIZE);

                        bool valid = (header_version == VK_PIPELINE_CACHE_HEADER_VERSION_ONE)
                                  && (header_size <= cache_data.size())
                                  && (cache_vendor_id == devProps.vendorID)
                                  && (cache_device_id == devProps.deviceID)
                                  && (std::memcmp(cache_uuid, devProps.pipelineCacheUUID, VK_UUID_SIZE) == 0);

                        if (valid) {
                            cacheCreateInfo.initialDataSize = cache_data.size();
                            cacheCreateInfo.pInitialData = cache_data.data();
                        }
                    }
                }
            }

            VkResult cacheResult = vkCreatePipelineCache(ctx.device, &cacheCreateInfo, nullptr, &ctx.pipelineCache);
            if (cacheResult != VK_SUCCESS) {
                // Non-fatal: pipelines work without cache, just slower startup
                std::cerr << "[Vulkan] Warning: pipeline cache load failed (corrupt disk cache?). "
                          << "Shader compilation will be slower on first run.\n";
                ctx.pipelineCache = VK_NULL_HANDLE;
                // Retry with empty cache
                cacheCreateInfo.initialDataSize = 0;
                cacheCreateInfo.pInitialData = nullptr;
                VkResult retryResult = vkCreatePipelineCache(ctx.device, &cacheCreateInfo, nullptr, &ctx.pipelineCache);
                if (retryResult != VK_SUCCESS) {
                    std::cerr << "[Vulkan] Warning: pipeline cache retry also failed. "
                              << "Proceeding without pipeline cache.\n";
                    ctx.pipelineCache = VK_NULL_HANDLE;
                }
            }
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

        // Create descriptor pool — clamp to device limits
        {
            VkPhysicalDeviceProperties descPoolProps;
            vkGetPhysicalDeviceProperties(ctx.physicalDevice, &descPoolProps);
            // We need up to 100000 sets for long-running tests (transformers, LSTMs, etc.)
            // but must not exceed the device's maxBoundDescriptorSets limit.
            // Note: maxBoundDescriptorSets is typically 4-32 (simultaneous bindings),
            // but pool maxSets is about total allocatable sets — most drivers support much more.
            // Still clamp descriptorCount to reasonable fraction of device storage buffer limits.
            uint32_t maxSets = 100000u;
            uint32_t maxStorageBuffers = descPoolProps.limits.maxDescriptorSetStorageBuffers;
            if (maxStorageBuffers > 0 && maxSets * 8 > maxStorageBuffers) {
                maxSets = maxStorageBuffers / 8;
            }
            ctx.descriptorPool = std::make_unique<vulkan::DescriptorPool>(ctx.device, maxSets);
        }

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

        // Configurable fence timeout from environment (seconds → nanoseconds)
        if (const char* env = std::getenv("TENZOR_VULKAN_FENCE_TIMEOUT_S")) {
            auto secs = std::strtoul(env, nullptr, 10);
            if (secs > 0) {
                ctx.fence_timeout_ns = static_cast<uint64_t>(secs) * 1'000'000'000ULL;
            }
        }

        // Initialize VulkanCachingAllocator for this device
        backend::VulkanCachingAllocator::get().initialize(
            ctx.device, ctx.physicalDevice, static_cast<int>(device_idx));

        // Initialize caches and per-device structures
        stagingPools_.push_back({});
        pipelineCaches_.push_back({});
        deferred_frees_.push_back({});
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
        vulkan::checkVk(vkQueueWaitIdle(devices_[0].computeQueue),
                        "Failed to wait for compute queue idle");
    }
}
vulkan::ComputePipeline* VulkanBackend::getPipeline(const std::string& shader_name,
                                                    int32_t device_id) {
    // Lock per-device mutex to protect concurrent pipeline cache access.
    std::lock_guard<std::recursive_mutex> lock(devices_[device_id].mutex);

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

    // Determine push constant size via SPIR-V reflection
    std::vector<VkPushConstantRange> pushConstants;
    uint32_t pushConstantSize = vulkan::reflectPushConstantSize(shaderCode);
    if (pushConstantSize % 4 != 0) {
        throw std::runtime_error("Vulkan pipeline '" + shader_name +
            "': push constant size (" + std::to_string(pushConstantSize) +
            ") must be 4-byte aligned (SPIR-V reflection error)");
    }
    if (pushConstantSize > 256) {
        throw std::runtime_error("Vulkan pipeline '" + shader_name +
            "': push constant size (" + std::to_string(pushConstantSize) +
            ") exceeds 256-byte Vulkan limit");
    }
    if (pushConstantSize > 0) {
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = pushConstantSize;
        pushConstants.push_back(push_range);
    }

    // Create pipeline with workgroup size specialization constant.
    // Spec constant ID 0 maps to local_size_x_id = 0 in all shaders,
    // allowing runtime override of the workgroup size per device.
    auto& ctx = devices_[device_id];
    uint32_t wgSize = ctx.workgroupSize;
    VkSpecializationMapEntry specEntry{};
    specEntry.constantID = 0;
    specEntry.offset = 0;
    specEntry.size = sizeof(uint32_t);
    VkSpecializationInfo specInfo{};
    specInfo.mapEntryCount = 1;
    specInfo.pMapEntries = &specEntry;
    specInfo.dataSize = sizeof(uint32_t);
    specInfo.pData = &wgSize;

    auto pipeline = std::make_unique<vulkan::ComputePipeline>(
        ctx.device, shaderCode, bindings, pushConstants, ctx.pipelineCache, &specInfo
    );

    auto* pipelinePtr = pipeline.get();
    cache.pipelines[shader_name] = std::move(pipeline);

    return pipelinePtr;
}

vulkan::ComputePipeline* VulkanBackend::getPipelineSpecialized(
    const std::string& shader_name, int32_t device_id,
    const std::vector<VkSpecializationMapEntry>& specEntries,
    const void* specData, size_t specDataSize) {

    // Build a cache key that incorporates the specialization data so that
    // the same shader compiled with different constants gets separate entries.
    // We hash the raw specialization bytes and append to the shader name.
    size_t spec_hash = 0;
    auto* bytes = static_cast<const uint8_t*>(specData);
    for (size_t i = 0; i < specDataSize; ++i) {
        spec_hash ^= std::hash<uint8_t>{}(bytes[i]) + 0x9e3779b9 + (spec_hash << 6) + (spec_hash >> 2);
    }
    std::string cache_key = shader_name + "_spec_" + std::to_string(spec_hash);

    std::lock_guard<std::recursive_mutex> lock(devices_[device_id].mutex);

    auto& cache = pipelineCaches_[device_id];
    auto it = cache.pipelines.find(cache_key);
    if (it != cache.pipelines.end()) {
        return it->second.get();
    }

    // Load shader code (same logic as getPipeline)
    std::vector<uint32_t> shaderCode;

#ifdef TENZOR_HAS_EMBEDDED_SHADERS
    const auto& registry = vulkan::embedded_shaders::getShaderRegistry();
    auto shader_it = registry.find(shader_name);
    if (shader_it != registry.end()) {
        const auto& shaderData = shader_it->second;
        shaderCode.assign(shaderData.data, shaderData.data + shaderData.size);
    } else {
        throw std::runtime_error("Embedded shader '" + shader_name + "' not found in registry.");
    }
#else
    std::string shaderFile = shaderPath_ + shader_name + ".spv";
    try {
        shaderCode = vulkan::loadShader(shaderFile);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "Failed to load Vulkan shader '" + shader_name + "'\n"
            "  Expected location: " + shaderFile + "\n"
            "  Error: " + e.what());
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

    // Determine push constant size via SPIR-V reflection
    std::vector<VkPushConstantRange> pushConstants;
    uint32_t pushConstantSize = vulkan::reflectPushConstantSize(shaderCode);
    if (pushConstantSize > 0) {
        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.offset = 0;
        push_range.size = pushConstantSize;
        pushConstants.push_back(push_range);
    }

    // Build specialization info from caller-provided entries
    auto& ctx = devices_[device_id];
    VkSpecializationInfo specInfo{};
    specInfo.mapEntryCount = static_cast<uint32_t>(specEntries.size());
    specInfo.pMapEntries = specEntries.data();
    specInfo.dataSize = specDataSize;
    specInfo.pData = specData;

    auto pipeline = std::make_unique<vulkan::ComputePipeline>(
        ctx.device, shaderCode, bindings, pushConstants, ctx.pipelineCache, &specInfo
    );

    auto* pipelinePtr = pipeline.get();
    cache.pipelines[cache_key] = std::move(pipeline);

    return pipelinePtr;
}

// Helper to get VkBuffer and offset from a potentially-offset pointer
std::pair<VkBuffer, VkDeviceSize> VulkanBackend::getVulkanBufferAndOffset(const void* ptr) const {
    if (ptr == nullptr) {
        throw std::runtime_error("Invalid buffer pointer: null pointer (empty tensor?)");
    }

#ifdef TENZOR_HAS_VMA
    // VMA path: look up buffer via VMA allocator
    auto& vma_alloc = backend::VulkanVMAAllocator::get();
    for (int32_t device_id = 0; device_id < device_count(); ++device_id) {
        VkBuffer buffer = vma_alloc.get_buffer(const_cast<void*>(ptr), device_id);
        if (buffer != VK_NULL_HANDLE) {
            return {buffer, 0};
        }
    }
#else
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

    // If not found directly, ptr might be base_ptr + offset (e.g. a sliced tensor
    // view or an offset write into a larger tensor).  Ask the caching allocator to
    // search its block list — it knows the ACTUAL block sizes (including slab
    // sub-allocations) and can find the correct VkBuffer + byte offset.
    for (int32_t device_id = 0; device_id < device_count(); ++device_id) {
        auto [buffer, offset] = allocator.find_buffer_and_offset(ptr, device_id);
        if (buffer != VK_NULL_HANDLE) {
            return {buffer, static_cast<VkDeviceSize>(offset)};
        }
    }
#endif

    // Not found even with offset search
    throw std::runtime_error("Invalid buffer pointer: buffer not tracked");
}

VkBuffer VulkanBackend::getVulkanBuffer(const void* ptr) const {
    return getVulkanBufferAndOffset(ptr).first;
}

VkDescriptorSet VulkanBackend::allocateAndWriteDescriptorSet(
    int32_t device_id,
    vulkan::ComputePipeline* pipeline,
    const std::vector<std::pair<uint32_t, const void*>>& bufferPtrs,
    const std::vector<size_t>& bufferSizes) {

    auto& ctx = devices_[device_id];

    // Protect descriptor pool allocation and potential reset from concurrent access.
    // The pool reset path modifies submittedFrames, currentFrame, and hasPendingWork
    // which must not race with endSingleTimeCommandsAsync or other allocations.
    std::lock_guard<std::recursive_mutex> lock(ctx.mutex);

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = ctx.descriptorPool->pool();
    allocInfo.descriptorSetCount = 1;
    VkDescriptorSetLayout layout = pipeline->descriptorSetLayout();
    allocInfo.pSetLayouts = &layout;

    VkDescriptorSet descriptorSet;
    VkResult result = vkAllocateDescriptorSets(ctx.device, &allocInfo, &descriptorSet);

    // If descriptor pool is exhausted or fragmented, try reset first, then grow
    if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL) {
        bool is_fragmented = (result == VK_ERROR_FRAGMENTED_POOL);

        // Force-submit any pending batched commands before resetting pools.
        // Without this, activeCommandBuffer references descriptors from the pool
        // we're about to reset, causing use-after-reset corruption.
        submitBatchIfNeeded(device_id, true);
        ensurePendingWorkComplete(device_id);
        // Full device idle ensures no command buffer is still referencing pool descriptors.
        // ensurePendingWorkComplete only waits on fences we track, but there may be
        // in-flight commands from other code paths. vkDeviceWaitIdle is the safe barrier.
        {
            VkResult waitResult = vkDeviceWaitIdle(ctx.device);
            if (waitResult == VK_ERROR_DEVICE_LOST) {
                ctx.device_lost = true;
                throw std::runtime_error("Device lost during descriptor pool recovery");
            }
            vulkan::checkVk(waitResult, "vkDeviceWaitIdle in descriptor pool recovery");
        }

        // Reset command pool and descriptor pool
        vulkan::checkVk(vkResetCommandPool(ctx.device, ctx.commandPool, 0),
                        "Failed to reset command pool during descriptor pool recovery");
        ctx.nextCommandBufferIndex = 0;
        ctx.submittedFrames = 0;
        ctx.currentFrame = 0;
        ctx.hasPendingWork = false;
        // Invalidate the active batch command buffer.  submitBatchIfNeeded skips
        // submission when operationsInBatch == 0, leaving activeCommandBuffer
        // pointing to a handle that vkResetCommandPool just put back into
        // "initial" state.  Without this, the next getOrCreateBatchCommandBuffer
        // returns the stale handle and callers record into an un-begun buffer.
        ctx.activeCommandBuffer = VK_NULL_HANDLE;
        ctx.operationsInBatch = 0;

        // Clear pipeline cache — pipelines may reference invalidated descriptor layouts
        pipelineCaches_[device_id].pipelines.clear();

        if (is_fragmented) {
            // Fragmentation means the pool has enough total capacity but can't
            // satisfy the request due to internal fragmentation. Grow to a larger
            // pool which will be freshly allocated without fragmentation.
            std::cerr << "[Vulkan WARNING] Descriptor pool fragmentation detected "
                      << "(allocated=" << ctx.descriptorPool->allocated_sets()
                      << "/" << ctx.descriptorPool->max_sets()
                      << "). Growing pool to reduce fragmentation.\n";
            ctx.descriptorPool->grow();
        } else {
            // Out of pool memory — try a simple reset first
            ctx.descriptorPool->reset();
        }

        // Retry allocation after reset
        allocInfo.descriptorPool = ctx.descriptorPool->pool();
        result = vkAllocateDescriptorSets(ctx.device, &allocInfo, &descriptorSet);

        // If reset wasn't enough (pool capacity is genuinely too small), grow and retry
        if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL) {
            ctx.descriptorPool->grow();
            allocInfo.descriptorPool = ctx.descriptorPool->pool();
            result = vkAllocateDescriptorSets(ctx.device, &allocInfo, &descriptorSet);
        }
    }

    vulkan::checkVk(result, "Failed to allocate descriptor set");

    // Track allocation count and warn if approaching pool capacity
    ctx.descriptorPool->trackAllocation();

    // Write descriptor set bindings — resolve VkBuffer + byte offset from raw pointers
    // so tensor views (slices with non-zero storage offset) bind correctly.
    std::vector<VkDescriptorBufferInfo> bufferInfos(bufferPtrs.size());
    std::vector<VkWriteDescriptorSet> writes(bufferPtrs.size());

    for (size_t i = 0; i < bufferPtrs.size(); ++i) {
        auto [buffer, offset] = getVulkanBufferAndOffset(bufferPtrs[i].second);
        bufferInfos[i].buffer = buffer;
        bufferInfos[i].offset = offset;
        // Round up to 4-byte boundary (minimum uint32 size for shader access).
        // Float16 shaders pack 2 elements per uint32, so odd-element-count tensors
        // need the range rounded up to avoid out-of-bounds descriptor access.
        bufferInfos[i].range = std::max(bufferSizes[i], static_cast<size_t>(4));

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].pNext = nullptr;
        writes[i].dstSet = descriptorSet;
        writes[i].dstBinding = bufferPtrs[i].first;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufferInfos[i];
        writes[i].pImageInfo = nullptr;
        writes[i].pTexelBufferView = nullptr;
    }

    // vkUpdateDescriptorSets returns void per Vulkan spec — no error check needed
    vkUpdateDescriptorSets(ctx.device, static_cast<uint32_t>(writes.size()),
                          writes.data(), 0, nullptr);

    return descriptorSet;
}

/* dispatch() moved to vulkan_dispatch.cpp */
#if 0  // Old string-based dispatch removed — kept for reference only
auto VulkanBackend::dispatch(const std::string& op_name,
                            std::span<const Tensor> inputs,
                            const OpAttributes& attrs) -> std::vector<Tensor> {

    // DEBUG for add with Int32
    if (op_name == "add" && !inputs.empty() && inputs[0].dtype() == DType::Int32) {
        std::cerr << "!!! VulkanBackend::dispatch called with 'add' and Int32 inputs\n";
        std::cerr << "Number of inputs: " << inputs.size() << "\n";
    }

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

    if (op_name == "norm") {
        float p = attrs.contains("p") ? std::stof(attrs.at("p")) : 2.0f;
        int64_t dim = attrs.contains("dim") ? std::stoll(attrs.at("dim")) : -1;
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
            // Parse dtype string (simplified)
            std::string dtype_str = attrs.at("dtype");
            if (dtype_str == "float32") dtype = DType::Float32;
            else if (dtype_str == "float64") dtype = DType::Float64;
            else if (dtype_str == "int32") dtype = DType::Int32;
            else if (dtype_str == "int64") dtype = DType::Int64;
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
            throw std::invalid_argument("max_pool2d_backward requires 2 inputs (grad_output, input)");
        }
        return {dispatchMaxPool2dBackward(inputs[0], inputs[1], attrs)};
    }

    // Conv2d forward operation
    if (op_name == "conv2d_forward") {
        if (inputs.size() < 2) {
            throw std::invalid_argument("conv2d_forward requires at least 2 inputs (input, weight)");
        }
        return {dispatchConv2dForward(inputs[0], inputs[1], inputs.size() >= 3 ? &inputs[2] : nullptr, attrs)};
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
            } else if (dtype_str == "float64" || dtype_str == "Float64") {
                dtype = DType::Float64;
            } else if (dtype_str == "int64" || dtype_str == "Int64") {
                dtype = DType::Int64;
            }
            // Add more dtypes as needed
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
            // Parse dtype if provided
            // For now, assume Float32
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
            // Parse dtype if provided
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
            // Parse dtype if provided
        }
        return {dispatchRandn(shape, dtype)};
    }

    throw std::runtime_error("VulkanBackend: Operation '" + op_name + "' not implemented");
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

    if (same_shape && is_float32) {
        // Fast path: use original math shader for Float32 same-shape operations
        auto* pipeline = getPipeline("math", device_id);

        Tensor output(output_shape, a.dtype(), a.device());

        // Prepare push constants
        struct PushConstants {
            uint32_t n;   // Number of elements
            uint32_t op;  // Operation code
            float param;  // Parameter for operations like pow
        } push_constants;
        push_constants.n = static_cast<uint32_t>(a.numel());
        push_constants.op = opcode;
        push_constants.param = 0.0f;

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
                          0, sizeof(PushConstants), &push_constants);

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
        auto* pipeline = getPipeline("math_broadcast", device_id);

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
        uint32_t dtype_code = 0;  // default to float32
        if (a.dtype() == DType::Int32) {
            dtype_code = 1;
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
        size_t buffer_size_a = a.numel() * a.dtype_size();
        size_t buffer_size_b = b.numel() * b.dtype_size();
        size_t buffer_size_out = output_numel * output.dtype_size();

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

        uint32_t workgroups = (output_numel + 255) / 256;
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

    auto* pipeline = getPipeline(shader_name, device_id);

    // Prepare push constants
    struct PushConstants {
        uint32_t n;   // Number of elements
        uint32_t op;  // Operation code
        float param;  // Parameter for operations like pow (unused here)
    } push_constants;
    push_constants.n = static_cast<uint32_t>(input.numel());
    push_constants.op = opcode;
    push_constants.param = 0.0f;  // Not used for these operations

    // Get VkBuffer handles from tensor data pointers
    VkBuffer buffer_in = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());

    // Calculate buffer sizes
    size_t buffer_size_in = input.numel() * input.dtype_size();
    size_t buffer_size_out = output.numel() * output.dtype_size();

    // Allocate and write descriptor set
    // For trigonometric/hyperbolic shaders: Binding 0: input, Binding 1: output
    // For math shader: Binding 0: input, Binding 1: unused (set to input), Binding 2: output
    std::vector<std::pair<uint32_t, VkBuffer>> bindings;
    std::vector<size_t> sizes;

    if (shader_name == "math") {
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
                      0, sizeof(PushConstants), &push_constants);

    uint32_t workgroups = (input.numel() + 255) / 256;
    vkCmdDispatch(cmdBuffer, workgroups, 1, 1);

    endSingleTimeCommands(cmdBuffer, device_id);

    return output;
}

auto VulkanBackend::dispatchUnaryOpWithParam(const std::string& op_name,
                                              const Tensor& input,
                                              float param) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("math", device_id);

    // Create output tensor (convert span to vector)
    auto input_shape = input.shape();
    std::vector<int64_t> output_shape(input_shape.begin(), input_shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // Map operation name to opcode (see math.comp shader)
    // 0=add, 1=sub, 2=mul, 3=div, 4=sqrt, 5=exp, 6=log, 7=neg, 8=abs, 9=pow, 10=sign
    uint32_t opcode = 0;
    if (op_name == "pow") opcode = 9;
    else throw std::runtime_error("Unknown parameterized unary operation: " + op_name);

    // Prepare push constants
    struct PushConstants {
        uint32_t n;   // Number of elements
        uint32_t op;  // Operation code
        float param;  // Parameter for operations like pow
    } push_constants;
    push_constants.n = static_cast<uint32_t>(input.numel());
    push_constants.op = opcode;
    push_constants.param = param;

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
                      0, sizeof(PushConstants), &push_constants);

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
    auto* pipeline = getPipeline("comparison", device_id);

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
    // Use the generic "reduction" shader which handles all reduction types via push constants
    auto* pipeline = getPipeline("reduction", device_id);

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
    if (dim >= 0) {
        auto input_shape = input.shape();
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
    pushConstants.reduce_size = (dim >= 0) ? static_cast<uint32_t>(input.shape()[dim]) : pushConstants.n;
    pushConstants.outer_size = (dim >= 0) ? static_cast<uint32_t>(output.numel()) : 1;
    pushConstants.inner_size = inner_size;
    pushConstants.op = op_code;

    vkCmdPushConstants(cmdBuffer, pipeline->layout(),
                      VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    // Dispatch one workgroup per output element
    // Each workgroup has 256 threads and reduces one output element
    uint32_t workgroups = pushConstants.outer_size;
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
    uint32_t workgroups_x = (push_constants.N + 15) / 16;
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
    auto* pipeline = getPipeline("log_softmax", device_id);

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
    auto* pipeline = getPipeline("prod_reduction", device_id);

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
    auto* pipeline = getPipeline("scatter", device_id);

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

    // Get Vulkan buffers
    VkBuffer buffer_input = getVulkanBuffer(input.data_ptr());
    VkBuffer buffer_indices = getVulkanBuffer(indices.data_ptr());
    VkBuffer buffer_output = getVulkanBuffer(output.data_ptr());
    VkBuffer buffer_values = getVulkanBuffer(values.data_ptr());

    size_t buffer_size_input = input.numel() * input.dtype_size();
    size_t buffer_size_indices = indices.numel() * sizeof(int64_t);
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
    result.impl_ = make_intrusive<TensorImpl>(*input.impl_);
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
    result.impl_ = make_intrusive<TensorImpl>(*(input.impl_));
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

    // Allocate temporary CPU buffer for input data
    std::vector<uint8_t> cpu_input_buffer(total_elements * element_size);
    std::vector<uint8_t> cpu_output_buffer(total_elements * element_size);

    // Download input tensor data from GPU to CPU
    copy(cpu_input_buffer.data(), input.data_ptr(),
         total_elements * element_size, CopyKind::DeviceToHost);

    // Reorder data on CPU using strides
    const int64_t ndims = input.ndim();

    if (ndims == 0) {
        // Scalar tensor - direct copy
        std::memcpy(cpu_output_buffer.data(), cpu_input_buffer.data(), element_size);
    } else {
        // Multi-dimensional copy using stride calculations
        std::vector<int64_t> indices(ndims, 0);
        int64_t dst_offset = 0;

        auto strides = input.strides();
        auto shape = input.shape();

        for (int64_t i = 0; i < total_elements; ++i) {
            // Calculate source offset using strides
            int64_t src_offset = 0;
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
    auto* pipeline = getPipeline("expand", device_id);

    // Create output tensor with new shape
    Tensor output(shape, input.dtype(), input.device());

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
    } push_constants;

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

    uint32_t workgroups = (output.numel() + 255) / 256;
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

    const Tensor& first_input = inputs[0];
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

    for (size_t i = 1; i < inputs.size(); ++i) {
        auto shape = inputs[i].shape();
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

    for (const auto& input : inputs) {
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
    auto* pipeline = getPipeline("clamp", device_id);

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

    // Prepare push constants
    struct PushConstants {
        uint32_t n_elements;
        float min_value;
        float max_value;
    } push_constants;

    push_constants.n_elements = static_cast<uint32_t>(output.numel());
    push_constants.min_value = min_value;
    push_constants.max_value = max_value;

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
 * @brief Dispatch forward activation operations (element-wise)
 * Operations: 0=relu, 1=sigmoid, 2=tanh, 3=gelu, 4=leaky_relu, 5=swish
 */
auto VulkanBackend::dispatchActivation(const std::string& op_name,
                                        const Tensor& input,
                                        uint32_t opcode,
                                        float param) -> Tensor {
    int32_t device_id = input.device().index;
    auto* pipeline = getPipeline("activations", device_id);

    // Create output tensor
    auto shape = input.shape();
    std::vector<int64_t> output_shape(shape.begin(), shape.end());
    Tensor output(output_shape, input.dtype(), input.device());

    // Prepare push constants
    struct PushConstants {
        uint32_t n;          // Number of elements
        uint32_t activation; // Operation code
        float alpha;         // For leaky_relu
    } push_constants;

    push_constants.n = static_cast<uint32_t>(input.numel());
    push_constants.activation = opcode;
    push_constants.alpha = param;

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
                      0, sizeof(PushConstants), &push_constants);

    // Dispatch compute workgroups
    uint32_t workgroups = (input.numel() + 255) / 256;
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
    auto* pipeline = getPipeline("activations_backward", device_id);

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
 * @brief Dispatch swish backward operation
 * Formula: swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
 * where swish(x) = x * sigmoid(x)
 */
auto VulkanBackend::dispatchSwishBackward(const Tensor& grad_output,
                                           const Tensor& input) -> Tensor {
    int32_t device_id = grad_output.device().index;
    auto* pipeline = getPipeline("swish_backward", device_id);

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
    auto* pipeline = getPipeline("softmax_backward", device_id);

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
    auto* pipeline = getPipeline("log_softmax_backward", device_id);

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
// Conv2d Forward Operation (OpAttributes version)
// ============================================================================

auto VulkanBackend::dispatchConv2dForward(const Tensor& input, const Tensor& weight, const Tensor* bias, const OpAttributes& attrs) -> Tensor {
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
    bool has_bias = bias != nullptr && bias->is_valid();

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
    auto* pipeline = getPipeline("conv2d_forward", device_id);

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

    // If bias is present, bind the actual bias buffer
    if (has_bias) {
        VkBuffer buffer_bias = getVulkanBuffer(bias->data_ptr());
        size_t buffer_size_bias = bias->numel() * bias->dtype_size();
        bindings[2] = {2, buffer_bias};
        sizes[2] = buffer_size_bias;
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
    auto* pipeline = getPipeline("full", device_id);

    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());
    size_t buffer_size_out = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

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
    // Create tensor on first available Vulkan device
    Device device(Device::Type::Vulkan, 0);
    Tensor output(shape, dtype, device);

    int32_t device_id = device.index;
    auto* pipeline = getPipeline("ones", device_id);

    VkBuffer buffer_out = getVulkanBuffer(output.data_ptr());
    size_t buffer_size_out = output.numel() * output.dtype_size();

    std::vector<std::pair<uint32_t, VkBuffer>> bindings = {
        {0, buffer_out}
    };
    std::vector<size_t> sizes = {buffer_size_out};

    VkDescriptorSet descriptorSet = allocateAndWriteDescriptorSet(
        device_id, pipeline, bindings, sizes);

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
    // Generate random data on CPU
    size_t numel = output.numel();
    std::vector<float> cpu_data(numel);

    // Use C++11 random number generation
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < numel; ++i) {
        cpu_data[i] = dist(gen);
    }

    // Copy to GPU
    copy(output.data_ptr(), cpu_data.data(), numel * sizeof(float), CopyKind::HostToDevice);

    return output;
}

auto VulkanBackend::dispatchRandn(const std::vector<int64_t>& shape, DType dtype) -> Tensor {
    // Create tensor on first available Vulkan device
    Device device(Device::Type::Vulkan, 0);
    Tensor output(shape, dtype, device);

    // Normal random distribution requires CPU generation, then copy to GPU
    // Generate random data on CPU
    size_t numel = output.numel();
    std::vector<float> cpu_data(numel);

    // Use C++11 random number generation
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::normal_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < numel; ++i) {
        cpu_data[i] = dist(gen);
    }

    // Copy to GPU
    copy(output.data_ptr(), cpu_data.data(), numel * sizeof(float), CopyKind::HostToDevice);

    return output;
}

#endif  // Old dispatch code

// Factory function
extern "C" {
    auto create_backend() -> std::unique_ptr<Backend> {
        return std::make_unique<VulkanBackend>();
    }
}

} // namespace tenzor
