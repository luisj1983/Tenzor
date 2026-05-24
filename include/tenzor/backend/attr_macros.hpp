/// \file attr_macros.hpp
/// \brief Per-axis attribute query helpers shared by every GPU backend.
///
/// Backend registries pack convolution / pool / unfold / fold attributes as a
/// mix of scalar (`AttrKey::Stride`) and per-axis (`AttrKey::StrideH/W/D`)
/// keys. Prior to this header each backend reinvented the read pattern:
///
///   * CUDA used the `TENZOR_CUDA_READ_3D_PER_AXIS` macro (3D conv only).
///   * OneAPI used `TENZOR_ONEAPI_READ_TRIPLE` (3D conv only).
///   * ROCm used inline `attrs.get_int(AttrKey::StrideH, attrs.get_int(...))`
///     for some Conv2d sites and scalar-only reads for Pool/Unfold/Fold.
///
/// The audit (2026-05-17) identified ~17 P1 sites across the three backends
/// where Pool2d/Pool3d/Unfold/Fold/Depthwise/FusedConv* read only the scalar
/// `AttrKey::Stride` and silently squashed asymmetric strides to symmetric.
///
/// This header provides:
///   1. `tenzor::backend::attrs::*` inline helpers that read 1D/2D/3D
///      per-axis attribute groups into `std::array<int64_t, N>` (and
///      `std::vector<int64_t>` for backends like OneAPI that need it).
///   2. Convenience aliases (`stride_2d`, `padding_3d`, `kernel_size_2d`,
///      `output_padding_3d`, …) that bind the scalar + per-axis key triples
///      already documented in `op_attributes.hpp`.
///   3. Composite macros that mirror the existing
///      `TENZOR_CUDA_READ_3D_PER_AXIS` shape (declare local
///      `stride`/`padding`/… variables in one line) for Conv1d/2d/3d,
///      ConvTranspose1d/2d/3d, Pool1d/2d/3d, and Unfold/Fold.
///
/// Callers may choose either form. Existing backend-specific macros remain
/// in place for backwards compatibility; new call sites should prefer the
/// inline functions or the unified macros here.

#pragma once

#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/backend/backend.hpp"  // for the `OpAttributes` alias of NewOpAttributes

#include <array>
#include <cstdint>
#include <vector>

namespace tenzor::backend::attrs {

// =====================================================================
// Generic per-axis readers
// =====================================================================

/// Read a 1D attribute with scalar fallback.
[[nodiscard]] inline auto
read_1d(const OpAttributes& a,
        AttrKey scalar,
        AttrKey w,
        int64_t default_val) noexcept -> std::array<int64_t, 1> {
    const int64_t iso = a.get_int(scalar, default_val);
    return { a.get_int(w, iso) };
}

/// Read a 2D (H, W) attribute pair with scalar fallback.
[[nodiscard]] inline auto
read_2d(const OpAttributes& a,
        AttrKey scalar,
        AttrKey h,
        AttrKey w,
        int64_t default_val) noexcept -> std::array<int64_t, 2> {
    const int64_t iso = a.get_int(scalar, default_val);
    return { a.get_int(h, iso), a.get_int(w, iso) };
}

/// Read a 3D (D, H, W) attribute triple with scalar fallback.
[[nodiscard]] inline auto
read_3d(const OpAttributes& a,
        AttrKey scalar,
        AttrKey d,
        AttrKey h,
        AttrKey w,
        int64_t default_val) noexcept -> std::array<int64_t, 3> {
    const int64_t iso = a.get_int(scalar, default_val);
    return { a.get_int(d, iso), a.get_int(h, iso), a.get_int(w, iso) };
}

// Vector variants for OneAPI sites that hand a std::vector<int64_t> to the
// native conv/pool entry points.

[[nodiscard]] inline auto
read_2d_vec(const OpAttributes& a,
            AttrKey scalar,
            AttrKey h,
            AttrKey w,
            int64_t default_val) -> std::vector<int64_t> {
    const auto arr = read_2d(a, scalar, h, w, default_val);
    return { arr[0], arr[1] };
}

[[nodiscard]] inline auto
read_3d_vec(const OpAttributes& a,
            AttrKey scalar,
            AttrKey d,
            AttrKey h,
            AttrKey w,
            int64_t default_val) -> std::vector<int64_t> {
    const auto arr = read_3d(a, scalar, d, h, w, default_val);
    return { arr[0], arr[1], arr[2] };
}

// =====================================================================
// Convenience aliases — bind the canonical scalar + per-axis key groups.
// Defaults match the project-wide conventions (stride/dilation/kernel
// default to 1; padding/output_padding default to 0).
// =====================================================================

// ---- 1D (W axis only) ------------------------------------------------

[[nodiscard]] inline auto stride_1d(const OpAttributes& a) noexcept {
    return read_1d(a, AttrKey::Stride, AttrKey::StrideW, /*default*/ 1);
}
[[nodiscard]] inline auto padding_1d(const OpAttributes& a) noexcept {
    return read_1d(a, AttrKey::Padding, AttrKey::PaddingW, /*default*/ 0);
}
[[nodiscard]] inline auto dilation_1d(const OpAttributes& a) noexcept {
    return read_1d(a, AttrKey::Dilation, AttrKey::DilationW, /*default*/ 1);
}
[[nodiscard]] inline auto kernel_size_1d(const OpAttributes& a) noexcept {
    return read_1d(a, AttrKey::KernelSize, AttrKey::KernelSizeW, /*default*/ 1);
}
[[nodiscard]] inline auto output_padding_1d(const OpAttributes& a) noexcept {
    return read_1d(a, AttrKey::OutputPadding, AttrKey::OutputPaddingW, /*default*/ 0);
}

// ---- 2D (H, W) -------------------------------------------------------

[[nodiscard]] inline auto stride_2d(const OpAttributes& a) noexcept {
    return read_2d(a, AttrKey::Stride, AttrKey::StrideH, AttrKey::StrideW, 1);
}
[[nodiscard]] inline auto padding_2d(const OpAttributes& a) noexcept {
    return read_2d(a, AttrKey::Padding, AttrKey::PaddingH, AttrKey::PaddingW, 0);
}
[[nodiscard]] inline auto dilation_2d(const OpAttributes& a) noexcept {
    return read_2d(a, AttrKey::Dilation, AttrKey::DilationH, AttrKey::DilationW, 1);
}
[[nodiscard]] inline auto kernel_size_2d(const OpAttributes& a) noexcept {
    return read_2d(a, AttrKey::KernelSize, AttrKey::KernelSizeH, AttrKey::KernelSizeW, 1);
}
[[nodiscard]] inline auto output_padding_2d(const OpAttributes& a) noexcept {
    return read_2d(a, AttrKey::OutputPadding, AttrKey::OutputPaddingH, AttrKey::OutputPaddingW, 0);
}

// ---- 3D (D, H, W) ----------------------------------------------------

[[nodiscard]] inline auto stride_3d(const OpAttributes& a) noexcept {
    return read_3d(a, AttrKey::Stride, AttrKey::StrideD, AttrKey::StrideH, AttrKey::StrideW, 1);
}
[[nodiscard]] inline auto padding_3d(const OpAttributes& a) noexcept {
    return read_3d(a, AttrKey::Padding, AttrKey::PaddingD, AttrKey::PaddingH, AttrKey::PaddingW, 0);
}
[[nodiscard]] inline auto dilation_3d(const OpAttributes& a) noexcept {
    return read_3d(a, AttrKey::Dilation, AttrKey::DilationD, AttrKey::DilationH, AttrKey::DilationW, 1);
}
[[nodiscard]] inline auto kernel_size_3d(const OpAttributes& a) noexcept {
    return read_3d(a, AttrKey::KernelSize, AttrKey::KernelSizeD, AttrKey::KernelSizeH, AttrKey::KernelSizeW, 1);
}
[[nodiscard]] inline auto output_padding_3d(const OpAttributes& a) noexcept {
    return read_3d(a, AttrKey::OutputPadding, AttrKey::OutputPaddingD, AttrKey::OutputPaddingH, AttrKey::OutputPaddingW, 0);
}

// std::vector forms — for OneAPI conv-entry signatures that take vectors
[[nodiscard]] inline auto stride_2d_vec(const OpAttributes& a)        { return read_2d_vec(a, AttrKey::Stride,        AttrKey::StrideH,   AttrKey::StrideW,   1); }
[[nodiscard]] inline auto padding_2d_vec(const OpAttributes& a)       { return read_2d_vec(a, AttrKey::Padding,       AttrKey::PaddingH,  AttrKey::PaddingW,  0); }
[[nodiscard]] inline auto dilation_2d_vec(const OpAttributes& a)      { return read_2d_vec(a, AttrKey::Dilation,      AttrKey::DilationH, AttrKey::DilationW, 1); }
[[nodiscard]] inline auto kernel_size_2d_vec(const OpAttributes& a)   { return read_2d_vec(a, AttrKey::KernelSize,    AttrKey::KernelSizeH, AttrKey::KernelSizeW, 1); }
[[nodiscard]] inline auto output_padding_2d_vec(const OpAttributes& a){ return read_2d_vec(a, AttrKey::OutputPadding, AttrKey::OutputPaddingH, AttrKey::OutputPaddingW, 0); }

[[nodiscard]] inline auto stride_3d_vec(const OpAttributes& a)        { return read_3d_vec(a, AttrKey::Stride,        AttrKey::StrideD,   AttrKey::StrideH,   AttrKey::StrideW,   1); }
[[nodiscard]] inline auto padding_3d_vec(const OpAttributes& a)       { return read_3d_vec(a, AttrKey::Padding,       AttrKey::PaddingD,  AttrKey::PaddingH,  AttrKey::PaddingW,  0); }
[[nodiscard]] inline auto dilation_3d_vec(const OpAttributes& a)      { return read_3d_vec(a, AttrKey::Dilation,      AttrKey::DilationD, AttrKey::DilationH, AttrKey::DilationW, 1); }
[[nodiscard]] inline auto kernel_size_3d_vec(const OpAttributes& a)   { return read_3d_vec(a, AttrKey::KernelSize,    AttrKey::KernelSizeD, AttrKey::KernelSizeH, AttrKey::KernelSizeW, 1); }
[[nodiscard]] inline auto output_padding_3d_vec(const OpAttributes& a){ return read_3d_vec(a, AttrKey::OutputPadding, AttrKey::OutputPaddingD, AttrKey::OutputPaddingH, AttrKey::OutputPaddingW, 0); }

// =====================================================================
// Conv1d -> Conv2d attribute projection
// =====================================================================
//
// Conv1d on the GPU backends is implemented as a wrapper that unsqueezes
// the input to [N,C,1,L] and delegates to Conv2dForward. The 1D semantics
// pack stride/padding/dilation as the W axis only; the H axis is always
// neutral (stride=1, padding=0, dilation=1).
//
// Naively forwarding the caller's `attrs` to Conv2d is wrong: Conv2d
// reads scalar `AttrKey::Stride/Padding/Dilation` and applies them to
// *both* H and W. With padding=1 this produces H_out=3 instead of 1 and
// the trailing `.squeeze(2)` silently leaves a 4-D tensor of wrong
// shape; with dilation=2 Conv2d may reject because the kernel exceeds
// H=1.
//
// `conv1d_to_conv2d_attrs` projects scalar Stride/Padding/Dilation onto
// the W axis (preserving any per-axis override the caller already set)
// and pins the H axis to its neutral values, leaving all other keys
// (Groups, Stream, WeightShape, InputShape, etc.) untouched.
[[nodiscard]] inline auto
conv1d_to_conv2d_attrs(const OpAttributes& src) -> OpAttributes {
    OpAttributes dst = src;
    const auto stride   = stride_1d(src);
    const auto padding  = padding_1d(src);
    const auto dilation = dilation_1d(src);
    dst.set(AttrKey::StrideH,   int64_t{1});
    dst.set(AttrKey::StrideW,   stride[0]);
    dst.set(AttrKey::PaddingH,  int64_t{0});
    dst.set(AttrKey::PaddingW,  padding[0]);
    dst.set(AttrKey::DilationH, int64_t{1});
    dst.set(AttrKey::DilationW, dilation[0]);
    return dst;
}

}  // namespace tenzor::backend::attrs

// =====================================================================
// Backwards-compatible composite macros
// =====================================================================
//
// These mirror the shape of `TENZOR_CUDA_READ_3D_PER_AXIS` already in
// `cuda_kernel_registry.cpp` (declare local `stride`/`padding`/`dilation`
// in one line) but generalise to 1D / 2D and to non-convolution ops
// (Pool, Unfold, Fold). Use them in registry lambdas that currently read
// only scalar `AttrKey::Stride` and ignore the per-axis keys.
//
// The macros assume an `OpAttributes attrs` is in scope. They expand into
// const-declarations — drop them at the top of a registry lambda body.

// ---- Conv2d ----------------------------------------------------------
#define TENZOR_READ_CONV2D_ATTRS()                                                                  \
    const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);                               \
    const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);                              \
    const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);                             \
    const int64_t groups = attrs.get_int(::tenzor::AttrKey::Groups, 1)

// ---- ConvTranspose2d -------------------------------------------------
#define TENZOR_READ_CONVT2D_ATTRS()                                                                 \
    const auto stride         = ::tenzor::backend::attrs::stride_2d(attrs);                         \
    const auto padding        = ::tenzor::backend::attrs::padding_2d(attrs);                        \
    const auto output_padding = ::tenzor::backend::attrs::output_padding_2d(attrs);                 \
    const auto dilation       = ::tenzor::backend::attrs::dilation_2d(attrs);                       \
    const int64_t groups = attrs.get_int(::tenzor::AttrKey::Groups, 1)

// ---- Conv3d ----------------------------------------------------------
#define TENZOR_READ_CONV3D_ATTRS()                                                                  \
    const auto stride   = ::tenzor::backend::attrs::stride_3d(attrs);                               \
    const auto padding  = ::tenzor::backend::attrs::padding_3d(attrs);                              \
    const auto dilation = ::tenzor::backend::attrs::dilation_3d(attrs);                             \
    const int64_t groups = attrs.get_int(::tenzor::AttrKey::Groups, 1)

// ---- ConvTranspose3d -------------------------------------------------
#define TENZOR_READ_CONVT3D_ATTRS()                                                                 \
    const auto stride         = ::tenzor::backend::attrs::stride_3d(attrs);                         \
    const auto padding        = ::tenzor::backend::attrs::padding_3d(attrs);                        \
    const auto output_padding = ::tenzor::backend::attrs::output_padding_3d(attrs);                 \
    const auto dilation       = ::tenzor::backend::attrs::dilation_3d(attrs);                       \
    const int64_t groups = attrs.get_int(::tenzor::AttrKey::Groups, 1)

// ---- Pool1d / Pool2d / Pool3d ---------------------------------------
// (No groups, no dilation by default; AvgPool may still read CeilMode /
//  CountIncludePad as bool flags via attrs.get_int explicitly.)
#define TENZOR_READ_POOL1D_ATTRS()                                                                  \
    const auto kernel_size = ::tenzor::backend::attrs::kernel_size_1d(attrs);                       \
    const auto stride      = ::tenzor::backend::attrs::stride_1d(attrs);                            \
    const auto padding     = ::tenzor::backend::attrs::padding_1d(attrs)

#define TENZOR_READ_POOL2D_ATTRS()                                                                  \
    const auto kernel_size = ::tenzor::backend::attrs::kernel_size_2d(attrs);                       \
    const auto stride      = ::tenzor::backend::attrs::stride_2d(attrs);                            \
    const auto padding     = ::tenzor::backend::attrs::padding_2d(attrs)

#define TENZOR_READ_POOL3D_ATTRS()                                                                  \
    const auto kernel_size = ::tenzor::backend::attrs::kernel_size_3d(attrs);                       \
    const auto stride      = ::tenzor::backend::attrs::stride_3d(attrs);                            \
    const auto padding     = ::tenzor::backend::attrs::padding_3d(attrs)

// MaxPool variants also need dilation; AvgPool generally does not.
#define TENZOR_READ_MAXPOOL2D_ATTRS()                                                               \
    TENZOR_READ_POOL2D_ATTRS();                                                                     \
    const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs)

#define TENZOR_READ_MAXPOOL3D_ATTRS()                                                               \
    TENZOR_READ_POOL3D_ATTRS();                                                                     \
    const auto dilation = ::tenzor::backend::attrs::dilation_3d(attrs)

// ---- Unfold / Fold (2D) ----------------------------------------------
// Unfold and Fold pack the same four lists (kernel, stride, padding,
// dilation). Fold also takes OutputSize.
#define TENZOR_READ_UNFOLD2D_ATTRS()                                                                \
    const auto kernel_size = ::tenzor::backend::attrs::kernel_size_2d(attrs);                       \
    const auto stride      = ::tenzor::backend::attrs::stride_2d(attrs);                            \
    const auto padding     = ::tenzor::backend::attrs::padding_2d(attrs);                           \
    const auto dilation    = ::tenzor::backend::attrs::dilation_2d(attrs)

#define TENZOR_READ_FOLD2D_ATTRS()                                                                  \
    TENZOR_READ_UNFOLD2D_ATTRS();                                                                   \
    /* Fold's OutputSize is rank-2 and uses OutputSizeH/W with scalar fallback. */                  \
    const auto output_size = ::tenzor::backend::attrs::read_2d(                                     \
        attrs, ::tenzor::AttrKey::OutputSize,                                                       \
        ::tenzor::AttrKey::OutputSizeH, ::tenzor::AttrKey::OutputSizeW, /*default*/ 1)
