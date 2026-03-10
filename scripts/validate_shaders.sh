#!/usr/bin/env bash
# validate_shaders.sh - Compile all Vulkan .comp shaders and report errors
#
# Usage: ./scripts/validate_shaders.sh [--shader-dir DIR] [--glslc PATH]
#
# Finds all .comp files in the shader directory and attempts to compile them
# with glslc. Exits non-zero if any shader fails to compile.
# Useful as a CI pre-check since CMake's file(GLOB) for shaders won't detect
# new files until reconfigure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SHADER_DIR="$PROJECT_ROOT/src/backends/vulkan/kernels"
GLSLC=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --shader-dir)
            SHADER_DIR="$2"
            shift 2
            ;;
        --glslc)
            GLSLC="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--shader-dir DIR] [--glslc PATH] [--output-dir DIR]"
            echo ""
            echo "Validates all .comp shaders by compiling them with glslc."
            echo "Options:"
            echo "  --shader-dir DIR   Directory containing .comp files (default: src/backends/vulkan/kernels)"
            echo "  --glslc PATH       Path to glslc compiler (default: auto-detect)"
            echo "  --output-dir DIR   Directory for .spv output (default: temp directory)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Auto-detect glslc if not specified
if [[ -z "$GLSLC" ]]; then
    if command -v glslc &>/dev/null; then
        GLSLC="$(command -v glslc)"
    elif [[ -n "${VULKAN_SDK:-}" ]] && [[ -x "$VULKAN_SDK/bin/glslc" ]]; then
        GLSLC="$VULKAN_SDK/bin/glslc"
    else
        echo "Error: glslc not found. Install Vulkan SDK or pass --glslc PATH." >&2
        exit 1
    fi
fi

echo "Using glslc: $GLSLC"
echo "Shader dir:  $SHADER_DIR"

if [[ ! -d "$SHADER_DIR" ]]; then
    echo "Error: Shader directory '$SHADER_DIR' does not exist." >&2
    exit 1
fi

# Use temp directory for output if not specified
CLEANUP_OUTPUT=false
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$(mktemp -d)"
    CLEANUP_OUTPUT=true
fi
mkdir -p "$OUTPUT_DIR"

# Find all .comp shader files
mapfile -t SHADERS < <(find "$SHADER_DIR" -name '*.comp' -type f | sort)

if [[ ${#SHADERS[@]} -eq 0 ]]; then
    echo "Error: No .comp shader files found in $SHADER_DIR" >&2
    exit 1
fi

echo "Found ${#SHADERS[@]} shader(s) to validate."
echo ""

FAILED=0
PASSED=0

# Shaders requiring SPIR-V 1.3 (subgroup operations)
SPIRV13_SHADERS="reduction_subgroup"

for SHADER in "${SHADERS[@]}"; do
    BASENAME="$(basename "$SHADER" .comp)"
    SPV_OUT="$OUTPUT_DIR/$BASENAME.spv"

    # Determine if this shader needs SPIR-V 1.3
    EXTRA_FLAGS=""
    for S13 in $SPIRV13_SHADERS; do
        if [[ "$BASENAME" == "$S13" ]]; then
            EXTRA_FLAGS="--target-env=vulkan1.1"
            break
        fi
    done

    if "$GLSLC" -fshader-stage=compute $EXTRA_FLAGS "$SHADER" -o "$SPV_OUT" 2>&1; then
        PASSED=$((PASSED + 1))
    else
        echo "FAIL: $SHADER" >&2
        FAILED=$((FAILED + 1))
    fi
done

# Cleanup temp directory
if [[ "$CLEANUP_OUTPUT" == true ]]; then
    rm -rf "$OUTPUT_DIR"
fi

echo ""
echo "=== Shader Validation Summary ==="
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
echo "  Total:  ${#SHADERS[@]}"

if [[ "$FAILED" -ne 0 ]]; then
    echo ""
    echo "Error: $FAILED shader(s) failed to compile." >&2
    exit 1
fi

echo ""
echo "All shaders compiled successfully."
