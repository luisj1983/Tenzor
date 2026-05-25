/**
 * @file vulkan_backend.cpp
 * @brief Vulkan compute backend - initialization, device queries, pipeline management, stream ops, factory
 */

#include "vulkan_helpers.hpp"
#include "tenzor/core/shape.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/vulkan_caching_allocator.hpp"
#include "tenzor/utils/log.hpp"
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
#include <typeinfo>
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
        // Audit I.4: unified logger.
        TENZOR_LOG_ERROR("Failed to initialize Vulkan backend: {}", e.what());
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
        // Audit I.4: unified logger.
        TENZOR_LOG_INFO("Vulkan validation layers enabled (TENZOR_VULKAN_VALIDATION=1)");
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
                // Audit I.4: unified logger; map Vulkan severity → log level.
                if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
                    TENZOR_LOG_ERROR("[Vulkan validation] {}", data->pMessage);
                } else {
                    TENZOR_LOG_WARN("[Vulkan validation] {}", data->pMessage);
                }
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
        bool hasFloat16Int8 = false;
        bool hasStorage16Bit = false;
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
            // Required for shaders that declare the Float16 SPIR-V capability.
            // Before this was enabled, F16 shaders ran with silently-undefined
            // behavior on some drivers (RADV produced wrong results for
            // odd-numel avg_pool2d_f16 outputs).
            if (strcmp(ext.extensionName, "VK_KHR_shader_float16_int8") == 0) {
                hasFloat16Int8 = true;
            }
            // 16-bit buffer storage is needed if F16 values are read from
            // SSBOs (which our packed-F16 shaders do). Strictly required
            // alongside VK_KHR_shader_float16_int8 for the full F16 path.
            if (strcmp(ext.extensionName, "VK_KHR_16bit_storage") == 0) {
                hasStorage16Bit = true;
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
        if (hasFloat16Int8) {
            deviceExtensions.push_back("VK_KHR_shader_float16_int8");
        }
        if (hasStorage16Bit) {
            deviceExtensions.push_back("VK_KHR_16bit_storage");
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

        // Query shaderFloat16 + 16bit storage buffer support. Both must be
        // enabled when the extension is available, otherwise shaders that
        // declare the Float16 SPIR-V capability (our *_f16 compute shaders)
        // run with undefined behavior on some drivers.
        VkPhysicalDeviceFloat16Int8FeaturesKHR f16i8Features{};
        f16i8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;
        f16i8Features.pNext = nullptr;
        VkPhysicalDevice16BitStorageFeaturesKHR storage16Features{};
        storage16Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR;
        storage16Features.pNext = nullptr;
        if (hasFloat16Int8 || hasStorage16Bit) {
            VkPhysicalDeviceFeatures2 features2{};
            features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            if (hasFloat16Int8) { f16i8Features.pNext = &storage16Features; features2.pNext = &f16i8Features; }
            else                { features2.pNext = &storage16Features; }
            vkGetPhysicalDeviceFeatures2(ctx.physicalDevice, &features2);
            // Disable if not actually supported by the driver
            hasFloat16Int8 = hasFloat16Int8 && (f16i8Features.shaderFloat16 == VK_TRUE);
            hasStorage16Bit = hasStorage16Bit && (storage16Features.storageBuffer16BitAccess == VK_TRUE);
            // Re-null pNext so the device-creation chain below can re-use these structs.
            f16i8Features.pNext = nullptr;
            storage16Features.pNext = nullptr;
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

        // Chain features if any extensions need features enabled via pNext.
        // Order: features2 → [int64] → [atomicFloat] → [f16i8] → [storage16].
        VkPhysicalDeviceFeatures2 features2Chain{};
        bool useFeatures2 = false;

        if (hasAtomicFloat && (atomicFloatFeatures.shaderBufferFloat32AtomicAdd ||
                              atomicFloatFeatures.shaderSharedFloat32AtomicAdd)) {
            atomicFloatFeatures.pNext = nullptr;
            useFeatures2 = true;
        }

        if (hasAtomicInt64) {
            atomicInt64Features.pNext = useFeatures2 ? static_cast<void*>(&atomicFloatFeatures) : nullptr;
            useFeatures2 = true;
        }

        // Append f16 + 16bit-storage features to the chain. Setting these to
        // VK_TRUE before device creation is the fix for the previously-silent
        // Float16 UB: shaders that declare the Float16 SPIR-V capability (our
        // *_f16 compute shaders) now run with the feature properly enabled.
        if (hasFloat16Int8) {
            f16i8Features.shaderFloat16 = VK_TRUE;
            f16i8Features.pNext = useFeatures2
                ? (hasAtomicInt64 ? static_cast<void*>(&atomicInt64Features)
                                  : static_cast<void*>(&atomicFloatFeatures))
                : nullptr;
            useFeatures2 = true;
        }
        if (hasStorage16Bit) {
            storage16Features.storageBuffer16BitAccess = VK_TRUE;
            // Chain below the f16 features if both are present, else below
            // whichever was at the top.
            void* next = nullptr;
            if (hasFloat16Int8) next = &f16i8Features;
            else if (hasAtomicInt64) next = &atomicInt64Features;
            else if (hasAtomicFloat) next = &atomicFloatFeatures;
            storage16Features.pNext = next;
            useFeatures2 = true;
        }

        if (useFeatures2) {
            features2Chain.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            features2Chain.features = deviceFeatures;
            // Head of the pNext chain is whichever struct is top-most.
            void* head = nullptr;
            if (hasStorage16Bit)       head = &storage16Features;
            else if (hasFloat16Int8)   head = &f16i8Features;
            else if (hasAtomicInt64)   head = &atomicInt64Features;
            else /* hasAtomicFloat */  head = &atomicFloatFeatures;
            features2Chain.pNext = head;
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
        // FF.8: record shaderBufferFloat64AtomicAdd separately so the F64
        // backward dispatchers that emit `atomicAdd(double)` can gate on it
        // without conflating with the F32 atomic-add feature above.
        ctx.hasAtomicFloat64 = hasAtomicFloat &&
            (atomicFloatFeatures.shaderBufferFloat64AtomicAdd == VK_TRUE);

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

        // Store maximum workgroup counts for dispatch validation + classify
        // the vendor for per-vendor workgroup tuning (Phase 2.2).
        {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(ctx.physicalDevice, &props);
            ctx.maxComputeWorkGroupCount[0] = props.limits.maxComputeWorkGroupCount[0];
            ctx.maxComputeWorkGroupCount[1] = props.limits.maxComputeWorkGroupCount[1];
            ctx.maxComputeWorkGroupCount[2] = props.limits.maxComputeWorkGroupCount[2];

            switch (props.vendorID) {
                case 0x10DE: ctx.vendor = GpuVendor::Nvidia;   break;
                case 0x1002: ctx.vendor = GpuVendor::Amd;      break;
                case 0x8086: ctx.vendor = GpuVendor::Intel;    break;
                case 0x106B: ctx.vendor = GpuVendor::Apple;    break;
                case 0x13B5: ctx.vendor = GpuVendor::Arm;      break;
                case 0x5143: ctx.vendor = GpuVendor::Qualcomm; break;
                default:     ctx.vendor = GpuVendor::Unknown;  break;
            }
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
                // Audit I.4: unified logger.
                TENZOR_LOG_WARN("[Vulkan] pipeline cache load failed (corrupt disk "
                                "cache?). Shader compilation will be slower on "
                                "first run.");
                ctx.pipelineCache = VK_NULL_HANDLE;
                // Retry with empty cache
                cacheCreateInfo.initialDataSize = 0;
                cacheCreateInfo.pInitialData = nullptr;
                VkResult retryResult = vkCreatePipelineCache(ctx.device, &cacheCreateInfo, nullptr, &ctx.pipelineCache);
                if (retryResult != VK_SUCCESS) {
                    TENZOR_LOG_WARN("[Vulkan] pipeline cache retry also failed. "
                                    "Proceeding without pipeline cache.");
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
auto VulkanBackend::create_stream([[maybe_unused]] int32_t device_id) -> StreamHandle {
    // Vulkan doesn't have explicit streams like CUDA
    // We could use command buffers for async execution
    // For now, return nullptr (default stream)
    return nullptr;
}

auto VulkanBackend::destroy_stream([[maybe_unused]] StreamHandle stream) -> void {
    // No-op for default stream
}

auto VulkanBackend::synchronize_stream(StreamHandle stream) -> void {
    // For default stream, synchronize device
    if (stream == nullptr && !devices_.empty()) {
        vulkan::checkVk(vkQueueWaitIdle(devices_[0].computeQueue),
                        "Failed to wait for compute queue idle");
    }
}

auto VulkanBackend::get_device_vendor(int32_t device_id) const -> GpuVendor {
    if (device_id < 0 || static_cast<size_t>(device_id) >= devices_.size()) {
        return GpuVendor::Unknown;
    }
    return devices_[device_id].vendor;
}

auto VulkanBackend::has_atomic_float(int32_t device_id) const -> bool {
    if (device_id < 0 || static_cast<size_t>(device_id) >= devices_.size()) {
        return false;
    }
    return devices_[device_id].hasAtomicFloat;
}

auto VulkanBackend::has_atomic_int64(int32_t device_id) const -> bool {
    if (device_id < 0 || static_cast<size_t>(device_id) >= devices_.size()) {
        return false;
    }
    return devices_[device_id].hasAtomicInt64;
}

auto VulkanBackend::has_atomic_float64(int32_t device_id) const -> bool {
    if (device_id < 0 || static_cast<size_t>(device_id) >= devices_.size()) {
        return false;
    }
    return devices_[device_id].hasAtomicFloat64;
}

auto VulkanBackend::recommended_workgroup_2d(GpuVendor vendor, OpKind op)
    -> std::pair<uint32_t, uint32_t> {
    // Defaults are chosen to match each vendor's subgroup width in the
    // inner dimension so one row of the tile fits in one warp/wave. The
    // total product is 256, matching the baked-in thread count. Matmul
    // shaders receive these values via specialization constants
    // (TILE_X / TILE_Y), which set both local_size and shared-memory
    // tile dimensions at pipeline creation time.
    switch (op) {
        case OpKind::Matmul:
            switch (vendor) {
                case GpuVendor::Nvidia:   return {32u, 8u};   // warp-width rows
                case GpuVendor::Amd:      return {64u, 4u};   // wave64 rows on GCN / RDNA compute
                case GpuVendor::Intel:    return {16u, 16u};  // SIMD8/16/32 — 16 is a safe default
                case GpuVendor::Apple:    return {32u, 8u};   // SIMD width 32
                case GpuVendor::Arm:      return {16u, 16u};
                case GpuVendor::Qualcomm: return {16u, 16u};
                case GpuVendor::Unknown:  return {16u, 16u};
            }
            break;
        case OpKind::Conv:
            switch (vendor) {
                case GpuVendor::Nvidia:   return {32u, 8u};
                case GpuVendor::Amd:      return {64u, 4u};
                default:                  return {16u, 16u};
            }
            break;
        case OpKind::ElementWise:
            // Element-wise is effectively 1D; keep a flat layout.
            switch (vendor) {
                case GpuVendor::Nvidia:   return {256u, 1u};
                case GpuVendor::Amd:      return {256u, 1u};
                default:                  return {256u, 1u};
            }
            break;
    }
    return {16u, 16u};
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

    // Create descriptor bindings (up to 12 buffers).
    // IMPORTANT: some shaders (e.g. sparse_spgemm_fill) use binding indices 0..8
    // (9 buffers). A previous value of 8 silently dropped binding 8's writes
    // (they went to an undefined binding and the buffer stayed zero-initialized,
    // producing an all-zero SpGEMM output). 12 covers all current shaders and
    // stays well under the Vulkan spec minimum maxPerStageDescriptorStorageBuffers
    // guarantee of 16.
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (uint32_t i = 0; i < 12; i++) {
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

    // Create descriptor bindings (up to 12 buffers).
    // IMPORTANT: some shaders (e.g. sparse_spgemm_fill) use binding indices 0..8
    // (9 buffers). A previous value of 8 silently dropped binding 8's writes
    // (they went to an undefined binding and the buffer stayed zero-initialized,
    // producing an all-zero SpGEMM output). 12 covers all current shaders and
    // stays well under the Vulkan spec minimum maxPerStageDescriptorStorageBuffers
    // guarantee of 16.
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (uint32_t i = 0; i < 12; i++) {
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
    // Find which device this allocation belongs to.
    //
    // S.5: catch the *typed* std::out_of_range that the allocator raises
    // for the expected "ptr lives on a different device" / "ptr not tracked"
    // miss path. Any other exception type indicates an internal allocator
    // failure (mutex poisoning, container invariant break, driver crash) —
    // log + rethrow so the diagnostic isn't silently swallowed. Mirrors L.3.
    for (int32_t device_id = 0; device_id < device_count(); ++device_id) {
        try {
            VkBuffer buffer = allocator.get_buffer(const_cast<void*>(ptr), device_id);
            return {buffer, 0};
        } catch (const std::out_of_range&) {
            // Expected: not found on this device, try next.
        } catch (const std::exception& e) {
            TENZOR_LOG_ERROR("[VulkanCachingAllocator::get_buffer] unexpected exception {} ({}); rethrowing",
                             typeid(e).name(), e.what());
            throw;
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

    // Phase 8.3: descriptor pool recovery via free-list grow. The old pool is
    // pushed onto `frozen_pools_` (still alive, still serving in-flight
    // descriptor sets); a new larger pool becomes the active target. No
    // `vkDeviceWaitIdle`, no command-pool reset, no pipeline-cache flush —
    // every existing descriptor set continues to be valid.
    if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL) {
        ctx.descriptorPool->grow();
        allocInfo.descriptorPool = ctx.descriptorPool->pool();
        result = vkAllocateDescriptorSets(ctx.device, &allocInfo, &descriptorSet);

        // If even the larger pool can't satisfy this single allocation,
        // something is wrong with the request itself (e.g. descriptor count
        // exceeding device limits) — fall through to the checkVk below.
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
        // Float16/BFloat16 shaders pack 2 elements per uint32 — a 9-element
        // Float16 tensor is 18 bytes logically, but the shader still reads/writes
        // a whole uint32 (4 bytes) at the last word. Without rounding up, the
        // descriptor range ends mid-word and the final write is OOB.
        size_t sz = std::max(bufferSizes[i], static_cast<size_t>(4));
        sz = (sz + 3) & ~size_t(3);
        bufferInfos[i].range = sz;

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


// Factory function
extern "C" {
    Backend* create_backend() {
        return new VulkanBackend();
    }
}

namespace vulkan {

// R.13: Shared FP64 capability gate for vulkan_ops_*.cpp dispatch sites.
// Mirrors the L.5 pattern in vulkan_ops_linalg.cpp / vulkan_kernel_registry.cpp.
void ensure_fp64_supported(int32_t device_id, const char* op_name) {
    auto* backend = DispatchTableRegistry::get_backend(Device::Type::Vulkan);
    if (backend == nullptr) {
        throw std::runtime_error(
            std::string("[Vulkan ") + (op_name ? op_name : "?") +
            "] Vulkan backend is not initialised; cannot query shaderFloat64");
    }
    auto* vk = static_cast<VulkanBackend*>(backend);
    DeviceInfo dev_info = vk->get_device_info(device_id);
    if (!dev_info.supports_fp64) {
        throw std::runtime_error(
            std::string("[Vulkan ") + (op_name ? op_name : "?") +
            "] requires shaderFloat64 (not supported on this device)");
    }
}

// S.4: Shared FP16 capability gate for vulkan_ops_*.cpp dispatch sites.
// Mirrors ensure_fp64_supported (R.13). BFloat16 dispatch paths reuse
// shaderFloat16 lowering on Vulkan, so they share this gate.
void ensure_fp16_supported(int32_t device_id, const char* op_name) {
    auto* backend = DispatchTableRegistry::get_backend(Device::Type::Vulkan);
    if (backend == nullptr) {
        throw std::runtime_error(
            std::string("[Vulkan ") + (op_name ? op_name : "?") +
            "] Vulkan backend is not initialised; cannot query shaderFloat16");
    }
    auto* vk = static_cast<VulkanBackend*>(backend);
    DeviceInfo dev_info = vk->get_device_info(device_id);
    if (!dev_info.supports_fp16) {
        throw std::runtime_error(
            std::string("[Vulkan ") + (op_name ? op_name : "?") +
            "] requires shaderFloat16 (VK_KHR_shader_float16_int8 not supported on this device)");
    }
}

// U.5/U.6/U.7: Shared VK_EXT_shader_atomic_float capability gate for
// vulkan_ops_*.cpp dispatchers whose compute shaders perform `atomicAdd(float)`
// on an SSBO (grid_sample_backward, affine_grid_backward, col2im / Fold /
// MaxUnpool, interpolate_bilinear_backward, …). Mirrors ensure_fp64_supported
// (R.13). Queries devices_[device_id].hasAtomicFloat via the public
// has_atomic_float() accessor.
void ensure_atomic_float_supported(int32_t device_id, const char* op_name) {
    auto* backend = DispatchTableRegistry::get_backend(Device::Type::Vulkan);
    if (backend == nullptr) {
        throw std::runtime_error(
            std::string("[Vulkan ") + (op_name ? op_name : "?") +
            "] Vulkan backend is not initialised; cannot query VK_EXT_shader_atomic_float");
    }
    auto* vk = static_cast<VulkanBackend*>(backend);
    if (!vk->has_atomic_float(device_id)) {
        throw std::runtime_error(
            std::string("[Vulkan ") + (op_name ? op_name : "?") +
            "] requires VK_EXT_shader_atomic_float (not supported on this device)");
    }
}

// Y.10: Shared VK_EXT_shader_atomic_int64 capability gate for vulkan_ops_*.cpp
// dispatchers whose compute shaders use GL_EXT_shader_atomic_int64 (typically
// F64 pooling/scatter/reduction backward paths that pack double into uint64
// for atomic CAS updates). Mirrors ensure_fp64_supported (R.13). Queries
// devices_[device_id].hasAtomicInt64 via the public has_atomic_int64() accessor.
void ensure_atomic_int64_supported(int32_t device_id, const char* op_name) {
    auto* backend = DispatchTableRegistry::get_backend(Device::Type::Vulkan);
    if (backend == nullptr) {
        throw std::runtime_error(
            std::string("[Vulkan ") + (op_name ? op_name : "?") +
            "] Vulkan backend is not initialised; cannot query VK_EXT_shader_atomic_int64");
    }
    auto* vk = static_cast<VulkanBackend*>(backend);
    if (!vk->has_atomic_int64(device_id)) {
        throw std::runtime_error(
            std::string("[Vulkan ") + (op_name ? op_name : "?") +
            "] requires VK_EXT_shader_atomic_int64 (not supported on this device)");
    }
}

// FF.8: Capability gate for shaders that perform `atomicAdd(double)` on an
// SSBO (F64 backward dispatchers using the direct double-atomicAdd path).
// Queries devices_[device_id].hasAtomicFloat64, which is set during device
// init from VkPhysicalDeviceShaderAtomicFloatFeaturesEXT::
// shaderBufferFloat64AtomicAdd. Mirrors ensure_atomic_float_supported
// (U.5/U.6/U.7).
void ensure_atomic_float64_storage_supported(int32_t device_id, const char* op_name) {
    auto* backend = DispatchTableRegistry::get_backend(Device::Type::Vulkan);
    if (backend == nullptr) {
        throw std::runtime_error(
            std::string("[Vulkan ") + (op_name ? op_name : "?") +
            "] Vulkan backend is not initialised; cannot query "
            "VK_EXT_shader_atomic_float (F64 buffer atomic-add)");
    }
    auto* vk = static_cast<VulkanBackend*>(backend);
    if (!vk->has_atomic_float64(device_id)) {
        throw std::runtime_error(
            std::string("[Vulkan ") + (op_name ? op_name : "?") +
            "] requires VK_EXT_shader_atomic_float with "
            "shaderBufferFloat64AtomicAdd (not supported on this device)");
    }
}

} // namespace vulkan

} // namespace tenzor
