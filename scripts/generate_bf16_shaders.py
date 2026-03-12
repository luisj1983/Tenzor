#!/usr/bin/env python3
"""Generate BFloat16 Vulkan compute shaders from Float16 templates.

BFloat16 uses the same packed-uint32 storage as Float16 (2 elements per word),
but the bit layout differs:
  F16:  1 sign + 5 exponent + 10 mantissa  -> unpackHalf2x16/packHalf2x16
  BF16: 1 sign + 8 exponent + 7 mantissa   -> bit-shift by 16 (upper 16 bits of IEEE 754 F32)
"""

import os
import re
import sys
from pathlib import Path

KERNELS_DIR = Path(__file__).parent.parent / "src" / "backends" / "vulkan" / "kernels"

# BF16 helper functions to replace F16 equivalents
READBF16_FUNC = """// Helper to read bfloat16 from packed buffer
float readBF16(uint packed, uint offset) {
    uint bits = (packed >> (offset * 16)) & 0xFFFFu;
    return uintBitsToFloat(bits << 16);
}"""

PACKBF16PAIR_FUNC = """// Helper to pack two bfloat16 values (round-to-nearest-even)
uint packBF16Pair(float v0, float v1) {
    uint b0 = floatBitsToUint(v0);
    uint b1 = floatBitsToUint(v1);
    uint bf0 = (b0 + 0x7FFFu + ((b0 >> 16) & 1u)) >> 16;
    uint bf1 = (b1 + 0x7FFFu + ((b1 >> 16) & 1u)) >> 16;
    return bf0 | (bf1 << 16);
}"""


def replace_readf16_func(content: str) -> str:
    """Replace readF16(uint, uint) function definitions with readBF16.

    Two variants:
    1. Simple: readF16(uint packed, uint offset) -> standard packed-pair reader
    2. Multi-buffer: readF16(uint buffer_idx, uint element_idx) -> keep body, fix F16 ops
    """
    # Match readF16 with 2 uint params, capturing param names
    pattern = r'(?://[^\n]*\n)*(?:float16_t|float)\s+readF16\s*\(\s*uint\s+(\w+)\s*,\s*uint\s+(\w+)\s*\)\s*\{'

    result = content
    while True:
        m = re.search(pattern, result)
        if not m:
            break
        param1, param2 = m.group(1), m.group(2)

        # Find matching closing brace
        start = m.start()
        brace_start = m.end() - 1
        brace_depth = 1
        i = brace_start + 1
        while i < len(result) and brace_depth > 0:
            if result[i] == '{':
                brace_depth += 1
            elif result[i] == '}':
                brace_depth -= 1
            i += 1
        end = i

        body = result[brace_start + 1:end - 1]

        # Check if this is a simple (packed, offset) reader or a multi-buffer reader
        is_multi_buffer = 'buffer_idx' in param1 or 'buffer_type' in param1 or \
                          'if (' in body.split('\n')[0] if body.strip() else False

        if not is_multi_buffer and (param1 == 'packed' or param1 == 'p'):
            # Simple packed reader -> replace with standard BF16 reader
            replacement = READBF16_FUNC
        else:
            # Multi-buffer reader: keep body structure, replace F16 bit ops
            new_body = body
            # Replace unpackHalf2x16(bits | (bits << 16)).x with uintBitsToFloat(bits << 16)
            new_body = re.sub(
                r'unpackHalf2x16\((\w+)\s*\|\s*\(\1\s*<<\s*16\)\)\.x',
                r'uintBitsToFloat(\1 << 16)',
                new_body
            )
            # Replace float16_t with float
            new_body = new_body.replace('float16_t', 'float')
            replacement = f"float readBF16({m.group(0).split('readF16')[1].split('{')[0].replace('readF16', '').strip().lstrip('(').rstrip(')')}"
            # Simpler: rebuild signature
            replacement = f"float readBF16(uint {param1}, uint {param2}) {{{new_body}}}"

        result = result[:start] + replacement + result[end:]

    return result


def replace_packf16pair_func(content: str) -> str:
    """Replace packF16Pair function definitions with packBF16Pair."""
    # Pattern: optional comment, packF16Pair with various param types
    pattern = r'(?://[^\n]*\n)*uint\s+packF16Pair\s*\([^)]+\)\s*\{[^}]+\}'
    return re.sub(pattern, PACKBF16PAIR_FUNC, content)


def replace_readf16element_func(content: str) -> str:
    """Replace readF16Element function definitions with BF16 equivalent.

    There are multiple variants:
    1. Returns float, reads from data_in[]
    2. Returns float16_t, reads from input_data[]
    Variable names in the body differ, so we capture and regenerate.
    """
    # Find all readF16Element definitions and replace them
    # The body can be multi-line with varying complexity
    pattern = r'(?:float16_t|float)\s+readF16Element\s*\(\s*uint\s+(\w+)\s*\)\s*\{'

    def make_bf16_readelem(match):
        param_name = match.group(1)
        # Search forward from the match to find the closing brace
        start = match.start()
        # Find the buffer name used (data_in or input_data)
        # Look in the original content after the match
        rest = content[match.end():]
        brace_depth = 1
        i = 0
        body = ""
        while i < len(rest) and brace_depth > 0:
            if rest[i] == '{':
                brace_depth += 1
            elif rest[i] == '}':
                brace_depth -= 1
            if brace_depth > 0:
                body += rest[i]
            i += 1

        # Detect buffer name from body
        buf_name = "data_in"
        if "input_data[" in body:
            buf_name = "input_data"

        return (f"float readBF16Element(uint {param_name}) {{\n"
                f"    uint pair_idx = {param_name} / 2;\n"
                f"    uint offset = {param_name} % 2;\n"
                f"    uint packed = {buf_name}[pair_idx];\n"
                f"    uint bits = (packed >> (offset * 16)) & 0xFFFFu;\n"
                f"    return uintBitsToFloat(bits << 16);\n"
                f"}}")

    # We need to do a manual replacement because the regex approach
    # with nested braces is tricky. Use a different strategy.
    result = content
    while True:
        m = re.search(pattern, result)
        if not m:
            break
        param_name = m.group(1)
        # Find matching closing brace
        start = m.start()
        brace_start = m.end() - 1  # position of opening {
        brace_depth = 1
        i = brace_start + 1
        while i < len(result) and brace_depth > 0:
            if result[i] == '{':
                brace_depth += 1
            elif result[i] == '}':
                brace_depth -= 1
            i += 1
        end = i  # position after closing }

        body = result[brace_start + 1:end - 1]
        # Extract actual buffer name from body (matches name[word_idx] or name[pair_idx])
        buf_match = re.search(r'(\w+)\[(?:word_idx|pair_idx)\]', body)
        buf_name = buf_match.group(1) if buf_match else "data_in"

        replacement = (f"float readBF16Element(uint {param_name}) {{\n"
                       f"    uint pair_idx = {param_name} / 2;\n"
                       f"    uint offset = {param_name} % 2;\n"
                       f"    uint packed = {buf_name}[pair_idx];\n"
                       f"    uint bits = (packed >> (offset * 16)) & 0xFFFFu;\n"
                       f"    return uintBitsToFloat(bits << 16);\n"
                       f"}}")
        result = result[:start] + replacement + result[end:]

    return result


def replace_writef16element_func(content: str) -> str:
    """Replace writeF16Element function definitions with BF16 equivalent."""
    pattern = r'void\s+writeF16Element\s*\(\s*uint\s+(\w+)\s*,\s*(?:float16_t|float)\s+(\w+)\s*\)\s*\{'

    result = content
    while True:
        m = re.search(pattern, result)
        if not m:
            break
        idx_param = m.group(1)
        val_param = m.group(2)
        # Find matching closing brace
        start = m.start()
        brace_start = m.end() - 1
        brace_depth = 1
        i = brace_start + 1
        while i < len(result) and brace_depth > 0:
            if result[i] == '{':
                brace_depth += 1
            elif result[i] == '}':
                brace_depth -= 1
            i += 1
        end = i

        body = result[brace_start + 1:end - 1]
        # Extract actual buffer name from body (matches name[word_idx] or name[pair_idx])
        buf_match = re.search(r'(\w+)\[(?:word_idx|pair_idx)\]', body)
        buf_name = buf_match.group(1) if buf_match else "data_out"

        # Check if the original uses atomicCompSwap (for thread-safe writes)
        uses_atomic = "atomicCompSwap" in body

        if uses_atomic:
            replacement = (
                f"void writeBF16Element(uint {idx_param}, float {val_param}) {{\n"
                f"    uint word_idx = {idx_param} / 2;\n"
                f"    uint half_offset = ({idx_param} % 2) * 16;\n"
                f"    uint b = floatBitsToUint({val_param});\n"
                f"    uint new_bits = (b + 0x7FFFu + ((b >> 16) & 1u)) >> 16;\n"
                f"    uint mask = 0xFFFFu << half_offset;\n"
                f"\n"
                f"    uint old_word = {buf_name}[word_idx];\n"
                f"    for (int attempt = 0; attempt < 1024; ++attempt) {{\n"
                f"        uint new_word = (old_word & ~mask) | (new_bits << half_offset);\n"
                f"        uint r = atomicCompSwap({buf_name}[word_idx], old_word, new_word);\n"
                f"        if (r == old_word) return;\n"
                f"        old_word = r;\n"
                f"    }}\n"
                f"}}")
        else:
            replacement = (
                f"void writeBF16Element(uint {idx_param}, float {val_param}) {{\n"
                f"    uint pair_idx = {idx_param} / 2;\n"
                f"    uint offset = {idx_param} % 2;\n"
                f"    uint b = floatBitsToUint({val_param});\n"
                f"    uint bits = (b + 0x7FFFu + ((b >> 16) & 1u)) >> 16;\n"
                f"    if (offset == 0) {{\n"
                f"        uint existing = {buf_name}[pair_idx] & 0xFFFF0000u;\n"
                f"        {buf_name}[pair_idx] = existing | bits;\n"
                f"    }} else {{\n"
                f"        uint existing = {buf_name}[pair_idx] & 0x0000FFFFu;\n"
                f"        {buf_name}[pair_idx] = existing | (bits << 16);\n"
                f"    }}\n"
                f"}}")
        result = result[:start] + replacement + result[end:]

    return result


BF16_UNPACK_HELPER = """
// BF16 utility: unpack two bfloat16 values from a uint32 word
vec2 unpackBF16x2(uint packed) {
    float lo = uintBitsToFloat((packed & 0xFFFFu) << 16);
    float hi = uintBitsToFloat(packed & 0xFFFF0000u);
    return vec2(lo, hi);
}

// BF16 utility: pack two float values into a uint32 as bfloat16 (round-to-nearest-even)
uint packBF16x2(vec2 v) {
    uint b0 = floatBitsToUint(v.x);
    uint b1 = floatBitsToUint(v.y);
    uint bf0 = (b0 + 0x7FFFu + ((b0 >> 16) & 1u)) >> 16;
    uint bf1 = (b1 + 0x7FFFu + ((b1 >> 16) & 1u)) >> 16;
    return bf0 | (bf1 << 16);
}
"""


def replace_standalone_unpack_pack(content: str) -> str:
    """Replace ALL unpackHalf2x16/packHalf2x16 calls with BF16 equivalents.

    After the helper function definitions have been replaced, any remaining
    unpackHalf2x16/packHalf2x16 calls are inline usages that also need conversion.
    """
    has_unpack = 'unpackHalf2x16' in content
    has_pack = 'packHalf2x16' in content

    if not has_unpack and not has_pack:
        return content

    # First, handle the specific pattern from readF16-style helpers:
    # unpackHalf2x16(bits | (bits << 16)).x  ->  uintBitsToFloat(bits << 16)
    content = re.sub(
        r'unpackHalf2x16\((\w+)\s*\|\s*\(\1\s*<<\s*16\)\)\.x',
        r'uintBitsToFloat(\1 << 16)',
        content
    )

    # Handle float16_t(unpackHalf2x16(...).x) -> uintBitsToFloat pattern
    # (float16_t already replaced to float by step 5, but just in case)
    content = re.sub(
        r'float\(unpackHalf2x16\((\w+)\s*\|\s*\(\1\s*<<\s*16\)\)\.x\)',
        r'uintBitsToFloat(\1 << 16)',
        content
    )

    # Handle packHalf2x16(vec2(EXPR, 0.0)) & 0xFFFFu  (extract single BF16 bits)
    def bf16_pack_single(m):
        expr = m.group(1)
        return f"((floatBitsToUint({expr}) + 0x7FFFu + ((floatBitsToUint({expr}) >> 16) & 1u)) >> 16)"

    content = re.sub(
        r'packHalf2x16\(vec2\(([^,]+),\s*0\.0\)\)(?:\.x)?\s*&\s*0xFFFFu',
        bf16_pack_single,
        content
    )

    # Handle packHalf2x16(vec2(EXPR, 0.0)) without & 0xFFFFu (used in writeF16Element)
    # For single value packing, low bits are the BF16, high 16 bits are 0
    content = re.sub(
        r'packHalf2x16\(vec2\(([^,]+),\s*0\.0\)\)',
        lambda m: f"((floatBitsToUint({m.group(1)}) + 0x7FFFu + ((floatBitsToUint({m.group(1)}) >> 16) & 1u)) >> 16)",
        content
    )

    # Check if there are still remaining calls that need the utility functions
    has_remaining_unpack = 'unpackHalf2x16' in content
    has_remaining_pack = 'packHalf2x16' in content

    if has_remaining_unpack or has_remaining_pack:
        # Insert BF16 utility functions after the version/extension block
        # Find insertion point: after last layout(...) in; or after push constants,
        # but before any function definitions. Safest: right after #version line + any extensions.
        # We'll insert after the first blank line following #version.
        insert_pos = 0
        lines = content.split('\n')
        # Find the line after #version (and any extension lines)
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#version') or line.startswith('#extension'):
                i += 1
                continue
            if line == '':
                # Insert here
                insert_pos = sum(len(l) + 1 for l in lines[:i])
                break
            break

        if insert_pos > 0:
            content = content[:insert_pos] + BF16_UNPACK_HELPER + content[insert_pos:]

        # Now replace all remaining occurrences
        content = content.replace('unpackHalf2x16(', 'unpackBF16x2(')
        content = content.replace('packHalf2x16(', 'packBF16x2(')

    return content


def transform_f16_to_bf16(content: str, filename: str) -> str:
    """Transform F16 shader content to BF16."""
    result = content

    # Step 1: Remove F16 extension (BF16 doesn't need it, uses uint bit manipulation)
    result = result.replace(
        '#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n', '')
    result = result.replace(
        '#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable\n', '')

    # Step 2: Replace helper function DEFINITIONS first (before call-site replacement)
    # Order matters: replace definitions before changing names, so regexes match original code

    # Replace readF16(uint packed, uint offset) style
    result = replace_readf16_func(result)

    # Replace packF16Pair(...) style
    result = replace_packf16pair_func(result)

    # Replace readF16Element / writeF16Element (various styles)
    result = replace_readf16element_func(result)
    result = replace_writef16element_func(result)

    # Step 3: Replace standalone unpackHalf2x16/packHalf2x16 calls
    result = replace_standalone_unpack_pack(result)

    # Step 4: Replace function CALLS (names only, definitions already replaced above)
    result = result.replace('readF16(', 'readBF16(')
    result = result.replace('packF16Pair(', 'packBF16Pair(')
    result = result.replace('readF16Element(', 'readBF16Element(')
    result = result.replace('writeF16Element(', 'writeBF16Element(')

    # Step 5: Replace float16_t type with float (BF16 has no native GLSL type)
    result = result.replace('float16_t', 'float')

    # Step 6: Update comments and identifiers
    # Be careful about replacement order to avoid double-replacement.
    # The template constants already contain "bfloat16" / "BFloat16" / "BF16" etc.,
    # so we must NOT re-replace those. Use negative lookbehinds to skip them.
    result = re.sub(r'(?<!B)Float16', 'BFloat16', result)
    result = re.sub(r'(?<!b)float16', 'bfloat16', result)
    result = re.sub(r'(?<!b)f16', 'bf16', result)
    result = re.sub(r'(?<!B)F16', 'BF16', result)

    return result


def main():
    if not KERNELS_DIR.exists():
        print(f"Error: kernels directory not found: {KERNELS_DIR}", file=sys.stderr)
        sys.exit(1)

    f16_shaders = sorted(KERNELS_DIR.glob("*_f16.comp"))
    print(f"Found {len(f16_shaders)} F16 shaders to transform")

    created = []
    skipped = []

    for f16_path in f16_shaders:
        bf16_name = f16_path.name.replace("_f16.comp", "_bf16.comp")
        bf16_path = f16_path.parent / bf16_name

        if bf16_path.exists():
            skipped.append(bf16_name)
            continue

        content = f16_path.read_text()
        bf16_content = transform_f16_to_bf16(content, f16_path.name)

        bf16_path.write_text(bf16_content)
        created.append(bf16_name)

    print(f"\nCreated {len(created)} BF16 shaders:")
    for name in sorted(created):
        print(f"  + {name}")

    if skipped:
        print(f"\nSkipped {len(skipped)} (already exist):")
        for name in sorted(skipped):
            print(f"  - {name}")

    print(f"\nTotal BF16 shaders: {len(created) + len(skipped)}")


if __name__ == "__main__":
    main()
