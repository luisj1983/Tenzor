#!/bin/bash
# Vulkan GPU hang debug script
# Usage: ./scripts/capture_vulkan_hang.sh [--renderdoc]
#
# Options:
#   --renderdoc   Try to capture with RenderDoc (may fail due to shaderFloat64)
#   (default)     Run with Vulkan validation layers for debugging

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT/bin"

export LD_LIBRARY_PATH="$PROJECT_ROOT/bin:/opt/intel/oneapi/2025.2/lib:$LD_LIBRARY_PATH"

echo "=== Vulkan GPU Hang Debug Script ==="
echo ""
echo "Working directory: $(pwd)"
echo ""

if [ "$1" == "--renderdoc" ]; then
    echo "Mode: RenderDoc capture"
    echo "NOTE: May fail if shaderFloat64 is not supported by RenderDoc layer"
    echo ""

    if ! command -v renderdoccmd &> /dev/null; then
        echo "ERROR: renderdoccmd not found in PATH"
        exit 1
    fi

    renderdoccmd capture \
        -w \
        -d "$PROJECT_ROOT/bin" \
        -c "Tenzor Vulkan Hang Debug" \
        ./debug_vulkan_hang 2>&1 || true

    echo ""
    ls -la *.rdc 2>/dev/null || echo "No .rdc files found"
else
    echo "Mode: Vulkan Validation Layers"
    echo "Validation layer output will show any Vulkan errors/warnings"
    echo ""

    # Enable validation layers and verbose output
    export VK_INSTANCE_LAYERS="VK_LAYER_KHRONOS_validation"
    export VK_LAYER_ENABLES="VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT"

    echo "Running test..."
    echo "========================================"
    ./debug_vulkan_hang 2>&1
    echo "========================================"
    echo ""
    echo "Exit code: $?"
fi
