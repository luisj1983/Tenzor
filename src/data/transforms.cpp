#include "tenzor/data/transforms.hpp"
#include "tenzor/ops/creation.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace tenzor {
namespace data {
namespace transforms {

namespace {

// The raw-pointer vision transforms below read/write through data_ptr() with
// `static_cast<float*>`. That cast is only valid for a CPU, Float32, contiguous
// tensor:
//   * Non-CPU input: data_ptr() is a device pointer; a host dereference crashes.
//   * Non-Float32 input (Float64/Float16/BFloat16): the bytes are reinterpreted
//     as 4-byte floats, silently corrupting the data.
// These helpers normalise the input to a CPU/Float32/contiguous tensor for the
// host math and convert the result back to the caller's original dtype/device,
// mirroring the explicit host-move ColorJitter already documented.

struct TransformDomain {
    Device   device;
    DType    dtype;
};

// Capture the caller's original domain and return a CPU/Float32/contiguous view
// suitable for raw float* access.
inline auto enter_host_float32(const Tensor& input, TransformDomain& orig) -> Tensor {
    orig.device = input.device();
    orig.dtype  = input.dtype();
    Tensor host = (orig.device.type != Device::Type::CPU)
        ? input.to(Device::cpu())
        : input;
    if (host.dtype() != DType::Float32) {
        host = host.to(DType::Float32);
    }
    return host.contiguous();
}

// Convert a Float32/CPU result back to the original dtype and device.
inline auto leave_host_float32(Tensor output, const TransformDomain& orig) -> Tensor {
    if (output.dtype() != orig.dtype) {
        output = output.to(orig.dtype);
    }
    if (output.device().type != orig.device.type ||
        output.device().index != orig.device.index) {
        output = output.to(orig.device);
    }
    return output;
}

// Construct a per-call mt19937 for a vision transform.
//
// When tenzor::manual_seed() is set, pull a deterministic seed from
// get_global_seed() (which advances per call) so two runs with the same seed
// produce bit-identical augmentations.
//
// When NO manual seed is set, seed from the shared thread-local global RNG
// engine rather than re-seeding a fresh mt19937 from the wall clock. Reseeding
// from high_resolution_clock made transforms executing within the same clock
// tick (e.g. two flips in a Compose) draw from identically-seeded generators
// and make correlated/identical "random" decisions. Drawing the seed from the
// shared engine advances that engine per call, so successive transforms get
// distinct, decorrelated seeds.
inline auto make_transform_rng() -> std::mt19937 {
    if (tenzor::detail::get_global_manual_seed_set()) {
        return std::mt19937(
            static_cast<std::mt19937::result_type>(tenzor::get_global_seed()));
    }
    auto& engine = tenzor::detail::get_global_rng_engine();
    return std::mt19937(engine());
}

} // anonymous namespace

// ============================================================================
// RandomRotation
// ============================================================================

RandomRotation::RandomRotation(float min_degrees, float max_degrees)
    : min_degrees_(min_degrees), max_degrees_(max_degrees) {
    if (min_degrees > max_degrees) {
        throw std::invalid_argument(
            "min_degrees must be <= max_degrees");
    }
}

auto RandomRotation::operator()(const Tensor& input, const Tensor& target)
    -> std::pair<Tensor, Tensor> {
    const auto& shape = input.shape();
    if (shape.size() < 2) {
        throw std::invalid_argument(
            "RandomRotation requires at least 2D input (H, W)");
    }

    // Normalise to CPU/Float32/contiguous for the raw float* loop below; the
    // result is converted back to the caller's dtype/device on return.
    TransformDomain orig;
    Tensor in = enter_host_float32(input, orig);

    // Generate random angle in [min_degrees_, max_degrees_]
    // B.2: deterministic when tenzor::manual_seed() is called; otherwise the
    // seed is drawn from the shared thread-local engine (advances per call) so
    // transforms in the same Compose/clock tick are not correlated. See
    // make_transform_rng().
    std::mt19937 rng = make_transform_rng();
    std::uniform_real_distribution<float> dist(min_degrees_, max_degrees_);
    float angle_deg = dist(rng);
    float angle_rad = angle_deg * static_cast<float>(M_PI) / 180.0f;

    float cos_a = std::cos(angle_rad);
    float sin_a = std::sin(angle_rad);

    // Assume last two dims are H, W
    int64_t H = shape[shape.size() - 2];
    int64_t W = shape[shape.size() - 1];
    float cx = static_cast<float>(W) / 2.0f;
    float cy = static_cast<float>(H) / 2.0f;

    // Create output tensor filled with zeros (Float32 host buffer)
    Tensor output = zeros(std::vector<int64_t>(shape.begin(), shape.end()),
                          DType::Float32);

    // Compute total number of "images" (product of all dims except H, W)
    int64_t num_images = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        num_images *= shape[i];
    }

    // Nearest-neighbor rotation via inverse mapping
    // For each pixel (y, x) in output, find source pixel in input
    const float* src_ptr = static_cast<const float*>(in.data_ptr());
    float* dst_ptr = static_cast<float*>(output.data_ptr());
    int64_t image_size = H * W;

    for (int64_t img = 0; img < num_images; ++img) {
        const float* src = src_ptr + img * image_size;
        float* dst = dst_ptr + img * image_size;

        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                // Inverse rotation: map output (x, y) to source coordinates
                float fx = static_cast<float>(x) - cx;
                float fy = static_cast<float>(y) - cy;
                float src_x = fx * cos_a + fy * sin_a + cx;
                float src_y = -fx * sin_a + fy * cos_a + cy;

                int64_t sx = static_cast<int64_t>(std::round(src_x));
                int64_t sy = static_cast<int64_t>(std::round(src_y));

                if (sx >= 0 && sx < W && sy >= 0 && sy < H) {
                    dst[y * W + x] = src[sy * W + sx];
                }
                // Out-of-bounds pixels remain zero
            }
        }
    }

    return {leave_host_float32(std::move(output), orig), target};
}

// ============================================================================
// ColorJitter
// ============================================================================

ColorJitter::ColorJitter(float brightness, float contrast,
                         float saturation, float hue)
    : brightness_(brightness), contrast_(contrast),
      saturation_(saturation), hue_(hue) {
    if (brightness < 0 || contrast < 0 || saturation < 0 || hue < 0) {
        throw std::invalid_argument(
            "ColorJitter parameters must be non-negative");
    }
}

auto ColorJitter::operator()(const Tensor& input, const Tensor& target)
    -> std::pair<Tensor, Tensor> {
    const auto& shape = input.shape();
    if (shape.empty()) {
        return {input, target};
    }

    // B.2: deterministic when tenzor::manual_seed() is called; otherwise the
    // seed is drawn from the shared thread-local engine (advances per call) so
    // transforms in the same Compose/clock tick are not correlated. See
    // make_transform_rng().
    std::mt19937 rng = make_transform_rng();

    // Work with a copy to avoid modifying the original. The pixel math below
    // is host-side (raw float* loops), so a non-CPU input must be moved to the
    // host first (data_ptr() is a device pointer otherwise → segfault) and the
    // result moved back to the original device at the end.
    const Device orig_device = input.device();
    Tensor output = (orig_device.type != Device::Type::CPU)
        ? input.to(Device::cpu()).contiguous()
        : input;
    float* data = static_cast<float*>(output.data_ptr());
    int64_t numel = output.numel();

    // Apply brightness: scale all values by random factor in
    // [1 - brightness, 1 + brightness]
    if (brightness_ > 0) {
        std::uniform_real_distribution<float> dist(
            1.0f - brightness_, 1.0f + brightness_);
        float factor = dist(rng);
        for (int64_t i = 0; i < numel; ++i) {
            data[i] *= factor;
        }
    }

    // Apply contrast: blend each channel toward the per-image luminance
    // mean (audit item I.11 — previous code used a single global mean
    // across the entire batch / all channels, which is not what
    // torchvision.transforms.ColorJitter does).
    //
    // PyTorch contract: contrast_factor ~ U[1-contrast_, 1+contrast_];
    // output = (input - mean_grayscale) * factor + mean_grayscale.
    // For non-3-channel inputs we fall back to per-image per-channel
    // mean (no luminance is defined for grayscale/multi-channel non-RGB
    // tensors).
    if (contrast_ > 0) {
        std::uniform_real_distribution<float> dist(
            1.0f - contrast_, 1.0f + contrast_);
        const float factor = dist(rng);
        const bool is_rgb = shape.size() >= 3 && shape[shape.size() - 3] == 3;

        if (is_rgb) {
            const int64_t C = shape[shape.size() - 3];
            int64_t spatial = 1;
            for (size_t d = shape.size() - 2; d < shape.size(); ++d) {
                spatial *= shape[d];
            }
            const int64_t num_images = numel / (C * spatial);
            for (int64_t img = 0; img < num_images; ++img) {
                float* base = data + img * C * spatial;
                const float* r = base;
                const float* g = base + spatial;
                const float* b = base + 2 * spatial;
                // Compute the per-image luminance mean.
                double gray_sum = 0.0;
                for (int64_t j = 0; j < spatial; ++j) {
                    gray_sum += 0.299 * r[j] + 0.587 * g[j] + 0.114 * b[j];
                }
                const float mean = static_cast<float>(
                    gray_sum / static_cast<double>(spatial));
                for (int64_t c = 0; c < C; ++c) {
                    float* ch = base + c * spatial;
                    for (int64_t j = 0; j < spatial; ++j) {
                        ch[j] = mean + factor * (ch[j] - mean);
                    }
                }
            }
        } else {
            // Non-RGB: per-image per-channel mean (no luminance defined).
            const int64_t per_image = numel;
            int64_t image_count = 1;
            if (shape.size() >= 3) {
                int64_t spatial = 1;
                for (size_t d = shape.size() - 2; d < shape.size(); ++d) spatial *= shape[d];
                const int64_t C = shape[shape.size() - 3];
                image_count = numel / (C * spatial);
                for (int64_t img = 0; img < image_count; ++img) {
                    float* base = data + img * C * spatial;
                    for (int64_t c = 0; c < C; ++c) {
                        float* ch = base + c * spatial;
                        double s = 0.0;
                        for (int64_t j = 0; j < spatial; ++j) s += ch[j];
                        const float m = static_cast<float>(s / static_cast<double>(spatial));
                        for (int64_t j = 0; j < spatial; ++j) {
                            ch[j] = m + factor * (ch[j] - m);
                        }
                    }
                }
            } else {
                double s = 0.0;
                for (int64_t i = 0; i < per_image; ++i) s += data[i];
                const float m = static_cast<float>(s / static_cast<double>(per_image));
                for (int64_t i = 0; i < per_image; ++i) {
                    data[i] = m + factor * (data[i] - m);
                }
            }
        }
    }

    // Apply saturation: blend toward grayscale (simplified)
    // For CHW with C=3: gray = 0.299*R + 0.587*G + 0.114*B
    if (saturation_ > 0 && shape.size() >= 3 && shape[shape.size() - 3] == 3) {
        std::uniform_real_distribution<float> dist(
            1.0f - saturation_, 1.0f + saturation_);
        float factor = dist(rng);

        int64_t C = shape[shape.size() - 3];
        int64_t spatial = 1;
        for (size_t d = shape.size() - 2; d < shape.size(); ++d) {
            spatial *= shape[d];
        }

        int64_t num_images = numel / (C * spatial);

        for (int64_t img = 0; img < num_images; ++img) {
            float* base = data + img * C * spatial;
            float* r_ch = base;
            float* g_ch = base + spatial;
            float* b_ch = base + 2 * spatial;

            for (int64_t j = 0; j < spatial; ++j) {
                float gray = 0.299f * r_ch[j] + 0.587f * g_ch[j] + 0.114f * b_ch[j];
                r_ch[j] = gray + factor * (r_ch[j] - gray);
                g_ch[j] = gray + factor * (g_ch[j] - gray);
                b_ch[j] = gray + factor * (b_ch[j] - gray);
            }
        }
    }

    // Apply hue: rotate the hue channel of each pixel in HSV space.
    // Audit item I.11 — header previously noted "not applied in this
    // simplified version".  PyTorch's contract: hue_factor ~ U[-hue_,
    // hue_] and is interpreted as a hue rotation in [-0.5, 0.5] where
    // ±0.5 = ±180°.  Implementation follows torchvision's RGB↔HSV
    // formulas; only applies on a 3-channel RGB input.
    if (hue_ > 0 && shape.size() >= 3 && shape[shape.size() - 3] == 3) {
        std::uniform_real_distribution<float> dist(-hue_, hue_);
        const float hue_shift = dist(rng);  // in [-0.5, 0.5] ⇒ ±180°

        const int64_t C = shape[shape.size() - 3];
        int64_t spatial = 1;
        for (size_t d = shape.size() - 2; d < shape.size(); ++d) spatial *= shape[d];
        const int64_t num_images = numel / (C * spatial);

        for (int64_t img = 0; img < num_images; ++img) {
            float* base = data + img * C * spatial;
            float* r_ch = base;
            float* g_ch = base + spatial;
            float* b_ch = base + 2 * spatial;
            for (int64_t j = 0; j < spatial; ++j) {
                const float r = r_ch[j], g = g_ch[j], b = b_ch[j];

                // RGB → HSV (hue in [0, 1)).
                const float max_v = std::max({r, g, b});
                const float min_v = std::min({r, g, b});
                const float delta = max_v - min_v;
                float h = 0.0f;
                if (delta > 0.0f) {
                    if (max_v == r) {
                        h = std::fmod((g - b) / delta, 6.0f);
                    } else if (max_v == g) {
                        h = (b - r) / delta + 2.0f;
                    } else {
                        h = (r - g) / delta + 4.0f;
                    }
                    h /= 6.0f;
                    if (h < 0.0f) h += 1.0f;
                }
                const float s = (max_v > 0.0f) ? delta / max_v : 0.0f;
                const float v = max_v;

                // Apply hue shift, wrapping into [0, 1).
                h = std::fmod(h + hue_shift, 1.0f);
                if (h < 0.0f) h += 1.0f;

                // HSV → RGB.
                const float h6 = h * 6.0f;
                const int sector = static_cast<int>(std::floor(h6)) % 6;
                const float f = h6 - std::floor(h6);
                const float p = v * (1.0f - s);
                const float q = v * (1.0f - s * f);
                const float t = v * (1.0f - s * (1.0f - f));
                float new_r = 0.0f, new_g = 0.0f, new_b = 0.0f;
                switch (sector) {
                    case 0: new_r = v; new_g = t; new_b = p; break;
                    case 1: new_r = q; new_g = v; new_b = p; break;
                    case 2: new_r = p; new_g = v; new_b = t; break;
                    case 3: new_r = p; new_g = q; new_b = v; break;
                    case 4: new_r = t; new_g = p; new_b = v; break;
                    default: new_r = v; new_g = p; new_b = q; break;
                }
                r_ch[j] = new_r;
                g_ch[j] = new_g;
                b_ch[j] = new_b;
            }
        }
    }

    if (orig_device.type != Device::Type::CPU) {
        output = output.to(orig_device);
    }
    return {output, target};
}

// ============================================================================
// Cutout
// ============================================================================

Cutout::Cutout(int num_holes, int hole_size)
    : num_holes_(num_holes), hole_size_(hole_size) {
    if (num_holes <= 0) {
        throw std::invalid_argument("num_holes must be positive");
    }
    if (hole_size <= 0) {
        throw std::invalid_argument("hole_size must be positive");
    }
}

auto Cutout::operator()(const Tensor& input, const Tensor& target)
    -> std::pair<Tensor, Tensor> {
    const auto& shape = input.shape();
    if (shape.size() < 2) {
        throw std::invalid_argument(
            "Cutout requires at least 2D input (H, W)");
    }

    // B.2: deterministic when tenzor::manual_seed() is called; otherwise the
    // seed is drawn from the shared thread-local engine (advances per call) so
    // transforms in the same Compose/clock tick are not correlated. See
    // make_transform_rng().
    std::mt19937 rng = make_transform_rng();

    // Assume last two dims are H, W
    int64_t H = shape[shape.size() - 2];
    int64_t W = shape[shape.size() - 1];

    // Work with an independent deep copy; assignment shares storage, which would
    // mutate the caller's input tensor in place.
    Tensor output = input.clone();
    float* data = static_cast<float*>(output.data_ptr());

    // Total elements per spatial plane
    int64_t num_planes = output.numel() / (H * W);

    std::uniform_int_distribution<int64_t> dist_y(0, H - 1);
    std::uniform_int_distribution<int64_t> dist_x(0, W - 1);

    for (int hole = 0; hole < num_holes_; ++hole) {
        int64_t cy = dist_y(rng);
        int64_t cx = dist_x(rng);

        int64_t y1 = std::max(int64_t{0}, cy - hole_size_ / 2);
        int64_t y2 = std::min(H, cy + hole_size_ / 2);
        int64_t x1 = std::max(int64_t{0}, cx - hole_size_ / 2);
        int64_t x2 = std::min(W, cx + hole_size_ / 2);

        // Zero-fill the rectangular region across all planes
        for (int64_t plane = 0; plane < num_planes; ++plane) {
            float* plane_data = data + plane * H * W;
            for (int64_t y = y1; y < y2; ++y) {
                for (int64_t x = x1; x < x2; ++x) {
                    plane_data[y * W + x] = 0.0f;
                }
            }
        }
    }

    return {output, target};
}

// ============================================================================
// RandomVerticalFlip
// ============================================================================

auto RandomVerticalFlip::operator()(const Tensor& input, const Tensor& target)
    -> std::pair<Tensor, Tensor> {
    const auto& shape = input.shape();
    if (shape.size() < 2) {
        return {input, target};
    }

    // B.2: deterministic when tenzor::manual_seed() is called; otherwise the
    // seed is drawn from the shared thread-local engine (advances per call) so
    // transforms in the same Compose/clock tick are not correlated. See
    // make_transform_rng().
    std::mt19937 rng = make_transform_rng();
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    if (dist(rng) >= p_) {
        return {input, target};
    }

    // Normalise to CPU/Float32/contiguous for the raw float* loop below; the
    // result is converted back to the caller's dtype/device on return.
    TransformDomain orig;
    Tensor in = enter_host_float32(input, orig);

    int64_t H = shape[shape.size() - 2];
    int64_t W = shape[shape.size() - 1];

    Tensor output = zeros(std::vector<int64_t>(shape.begin(), shape.end()), DType::Float32);
    int64_t num_planes = output.numel() / (H * W);

    const float* src = static_cast<const float*>(in.data_ptr());
    float* dst = static_cast<float*>(output.data_ptr());

    for (int64_t plane = 0; plane < num_planes; ++plane) {
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                dst[plane * H * W + y * W + x] =
                    src[plane * H * W + (H - 1 - y) * W + x];
            }
        }
    }

    return {leave_host_float32(std::move(output), orig), target};
}

// ============================================================================
// RandomHorizontalFlip
// ============================================================================

auto RandomHorizontalFlip::operator()(const Tensor& input, const Tensor& target)
    -> std::pair<Tensor, Tensor> {
    const auto& shape = input.shape();
    if (shape.size() < 2) {
        return {input, target};
    }

    // B.2: deterministic when tenzor::manual_seed() is called; otherwise the
    // seed is drawn from the shared thread-local engine (advances per call) so
    // transforms in the same Compose/clock tick are not correlated. See
    // make_transform_rng().
    std::mt19937 rng = make_transform_rng();
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    if (dist(rng) >= p_) {
        return {input, target};
    }

    // Horizontal flip reverses the last (width) dimension. Use the tensor op
    // to stay device-agnostic (CPU/CUDA/etc.).
    int64_t last_dim = static_cast<int64_t>(shape.size()) - 1;
    Tensor output = tenzor::flip(input, std::vector<int64_t>{last_dim});
    return {output, target};
}

// ============================================================================
// CenterCrop
// ============================================================================

auto CenterCrop::operator()(const Tensor& input, const Tensor& target)
    -> std::pair<Tensor, Tensor> {
    const auto& shape = input.shape();
    if (shape.size() < 2) {
        throw std::invalid_argument("CenterCrop requires at least 2D input");
    }

    int64_t H = shape[shape.size() - 2];
    int64_t W = shape[shape.size() - 1];

    if (height_ > H || width_ > W) {
        throw std::invalid_argument("CenterCrop size exceeds input dimensions");
    }

    int64_t top = (H - height_) / 2;
    int64_t left = (W - width_) / 2;

    // Normalise to CPU/Float32/contiguous for the raw float* loop below; the
    // result is converted back to the caller's dtype/device on return.
    TransformDomain orig;
    Tensor in = enter_host_float32(input, orig);

    // Build output shape
    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[shape.size() - 2] = height_;
    out_shape[shape.size() - 1] = width_;

    Tensor output = zeros(out_shape, DType::Float32);
    int64_t num_planes = output.numel() / (height_ * width_);

    const float* src = static_cast<const float*>(in.data_ptr());
    float* dst = static_cast<float*>(output.data_ptr());

    for (int64_t plane = 0; plane < num_planes; ++plane) {
        for (int64_t y = 0; y < height_; ++y) {
            for (int64_t x = 0; x < width_; ++x) {
                dst[plane * height_ * width_ + y * width_ + x] =
                    src[plane * H * W + (top + y) * W + (left + x)];
            }
        }
    }

    return {leave_host_float32(std::move(output), orig), target};
}

// ============================================================================
// RandomCrop
// ============================================================================

auto RandomCrop::operator()(const Tensor& input, const Tensor& target)
    -> std::pair<Tensor, Tensor> {
    const auto& shape = input.shape();
    if (shape.size() < 2) {
        throw std::invalid_argument("RandomCrop requires at least 2D input");
    }

    int64_t H = shape[shape.size() - 2] + 2 * padding_;
    int64_t W = shape[shape.size() - 1] + 2 * padding_;

    if (height_ > H || width_ > W) {
        throw std::invalid_argument("RandomCrop size exceeds (padded) input dimensions");
    }

    // B.2: deterministic when tenzor::manual_seed() is called; otherwise the
    // seed is drawn from the shared thread-local engine (advances per call) so
    // transforms in the same Compose/clock tick are not correlated. See
    // make_transform_rng().
    std::mt19937 rng = make_transform_rng();
    std::uniform_int_distribution<int64_t> dist_y(0, H - height_);
    std::uniform_int_distribution<int64_t> dist_x(0, W - width_);

    int64_t top = dist_y(rng) - padding_;
    int64_t left = dist_x(rng) - padding_;

    int64_t orig_H = shape[shape.size() - 2];
    int64_t orig_W = shape[shape.size() - 1];

    // Normalise to CPU/Float32/contiguous for the raw float* loop below; the
    // result is converted back to the caller's dtype/device on return.
    TransformDomain orig;
    Tensor in = enter_host_float32(input, orig);

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[shape.size() - 2] = height_;
    out_shape[shape.size() - 1] = width_;

    Tensor output = zeros(out_shape, DType::Float32);
    int64_t num_planes = output.numel() / (height_ * width_);

    const float* src = static_cast<const float*>(in.data_ptr());
    float* dst = static_cast<float*>(output.data_ptr());

    for (int64_t plane = 0; plane < num_planes; ++plane) {
        for (int64_t y = 0; y < height_; ++y) {
            for (int64_t x = 0; x < width_; ++x) {
                int64_t sy = top + y;
                int64_t sx = left + x;
                if (sy >= 0 && sy < orig_H && sx >= 0 && sx < orig_W) {
                    dst[plane * height_ * width_ + y * width_ + x] =
                        src[plane * orig_H * orig_W + sy * orig_W + sx];
                }
                // Out of bounds stays zero (padding)
            }
        }
    }

    return {leave_host_float32(std::move(output), orig), target};
}

// ============================================================================
// Resize (nearest-neighbor)
// ============================================================================

auto Resize::operator()(const Tensor& input, const Tensor& target)
    -> std::pair<Tensor, Tensor> {
    const auto& shape = input.shape();
    if (shape.size() < 2) {
        throw std::invalid_argument("Resize requires at least 2D input");
    }

    int64_t H = shape[shape.size() - 2];
    int64_t W = shape[shape.size() - 1];

    // Normalise to CPU/Float32/contiguous for the raw float* loop below; the
    // result is converted back to the caller's dtype/device on return.
    TransformDomain orig;
    Tensor in = enter_host_float32(input, orig);

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[shape.size() - 2] = height_;
    out_shape[shape.size() - 1] = width_;

    Tensor output = zeros(out_shape, DType::Float32);
    int64_t num_planes = output.numel() / (height_ * width_);

    const float* src = static_cast<const float*>(in.data_ptr());
    float* dst = static_cast<float*>(output.data_ptr());

    float scale_y = static_cast<float>(H) / static_cast<float>(height_);
    float scale_x = static_cast<float>(W) / static_cast<float>(width_);

    for (int64_t plane = 0; plane < num_planes; ++plane) {
        for (int64_t y = 0; y < height_; ++y) {
            for (int64_t x = 0; x < width_; ++x) {
                int64_t sy = std::min(static_cast<int64_t>(y * scale_y), H - 1);
                int64_t sx = std::min(static_cast<int64_t>(x * scale_x), W - 1);
                dst[plane * height_ * width_ + y * width_ + x] =
                    src[plane * H * W + sy * W + sx];
            }
        }
    }

    return {leave_host_float32(std::move(output), orig), target};
}

// ============================================================================
// RandomResizedCrop
// ============================================================================

auto RandomResizedCrop::operator()(const Tensor& input, const Tensor& target)
    -> std::pair<Tensor, Tensor> {
    const auto& shape = input.shape();
    if (shape.size() < 2) {
        throw std::invalid_argument("RandomResizedCrop requires at least 2D input");
    }

    int64_t H = shape[shape.size() - 2];
    int64_t W = shape[shape.size() - 1];
    float area = static_cast<float>(H * W);

    // Normalise to CPU/Float32/contiguous for the raw float* loop below; the
    // result is converted back to the caller's dtype/device on return.
    TransformDomain orig;
    Tensor in = enter_host_float32(input, orig);

    // B.2: deterministic when tenzor::manual_seed() is called; otherwise the
    // seed is drawn from the shared thread-local engine (advances per call) so
    // transforms in the same Compose/clock tick are not correlated. See
    // make_transform_rng().
    std::mt19937 rng = make_transform_rng();
    std::uniform_real_distribution<float> scale_dist(scale_min_, scale_max_);
    std::uniform_real_distribution<float> ratio_dist(std::log(ratio_min_), std::log(ratio_max_));

    int64_t crop_h = H, crop_w = W, top = 0, left = 0;

    // Try up to 10 times to find valid crop
    for (int attempt = 0; attempt < 10; ++attempt) {
        float target_area = area * scale_dist(rng);
        float aspect_ratio = std::exp(ratio_dist(rng));

        crop_w = static_cast<int64_t>(std::round(std::sqrt(target_area * aspect_ratio)));
        crop_h = static_cast<int64_t>(std::round(std::sqrt(target_area / aspect_ratio)));

        if (crop_w > 0 && crop_w <= W && crop_h > 0 && crop_h <= H) {
            std::uniform_int_distribution<int64_t> dy(0, H - crop_h);
            std::uniform_int_distribution<int64_t> dx(0, W - crop_w);
            top = dy(rng);
            left = dx(rng);
            break;
        }
    }

    // Clamp to valid region
    crop_h = std::min(crop_h, H);
    crop_w = std::min(crop_w, W);
    top = std::min(top, H - crop_h);
    left = std::min(left, W - crop_w);

    // Extract crop and resize
    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[shape.size() - 2] = height_;
    out_shape[shape.size() - 1] = width_;

    Tensor output = zeros(out_shape, DType::Float32);
    int64_t num_planes = output.numel() / (height_ * width_);

    const float* src = static_cast<const float*>(in.data_ptr());
    float* dst = static_cast<float*>(output.data_ptr());

    float scale_y = static_cast<float>(crop_h) / static_cast<float>(height_);
    float scale_x = static_cast<float>(crop_w) / static_cast<float>(width_);

    for (int64_t plane = 0; plane < num_planes; ++plane) {
        for (int64_t y = 0; y < height_; ++y) {
            for (int64_t x = 0; x < width_; ++x) {
                int64_t sy = top + std::min(static_cast<int64_t>(y * scale_y), crop_h - 1);
                int64_t sx = left + std::min(static_cast<int64_t>(x * scale_x), crop_w - 1);
                dst[plane * height_ * width_ + y * width_ + x] =
                    src[plane * H * W + sy * W + sx];
            }
        }
    }

    return {leave_host_float32(std::move(output), orig), target};
}

// ============================================================================
// GaussianBlur
// ============================================================================

auto GaussianBlur::operator()(const Tensor& input, const Tensor& target)
    -> std::pair<Tensor, Tensor> {
    const auto& shape = input.shape();
    if (shape.size() < 2) {
        throw std::invalid_argument("GaussianBlur requires at least 2D input");
    }

    // Normalise to CPU/Float32/contiguous for the raw float* loop below; the
    // result is converted back to the caller's dtype/device on return.
    TransformDomain orig;
    Tensor in = enter_host_float32(input, orig);

    // B.2: deterministic when tenzor::manual_seed() is called; otherwise the
    // seed is drawn from the shared thread-local engine (advances per call) so
    // transforms in the same Compose/clock tick are not correlated. See
    // make_transform_rng().
    std::mt19937 rng = make_transform_rng();
    std::uniform_real_distribution<float> sigma_dist(sigma_min_, sigma_max_);
    float sigma = sigma_dist(rng);

    int half = kernel_size_ / 2;

    // Build 1D Gaussian kernel
    std::vector<float> kernel(kernel_size_);
    float sum = 0.0f;
    for (int i = 0; i < kernel_size_; ++i) {
        float x = static_cast<float>(i - half);
        kernel[i] = std::exp(-x * x / (2.0f * sigma * sigma));
        sum += kernel[i];
    }
    for (auto& k : kernel) k /= sum;

    int64_t H = shape[shape.size() - 2];
    int64_t W = shape[shape.size() - 1];

    Tensor output = zeros(std::vector<int64_t>(shape.begin(), shape.end()), DType::Float32);
    Tensor temp = zeros(std::vector<int64_t>(shape.begin(), shape.end()), DType::Float32);
    int64_t num_planes = output.numel() / (H * W);

    const float* src = static_cast<const float*>(in.data_ptr());
    float* tmp = static_cast<float*>(temp.data_ptr());
    float* dst = static_cast<float*>(output.data_ptr());

    // Separable blur: horizontal pass
    for (int64_t plane = 0; plane < num_planes; ++plane) {
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                float val = 0.0f;
                for (int k = 0; k < kernel_size_; ++k) {
                    int64_t sx = x + k - half;
                    sx = std::max(int64_t{0}, std::min(sx, W - 1));  // clamp
                    val += src[plane * H * W + y * W + sx] * kernel[k];
                }
                tmp[plane * H * W + y * W + x] = val;
            }
        }
    }

    // Vertical pass
    for (int64_t plane = 0; plane < num_planes; ++plane) {
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                float val = 0.0f;
                for (int k = 0; k < kernel_size_; ++k) {
                    int64_t sy = y + k - half;
                    sy = std::max(int64_t{0}, std::min(sy, H - 1));  // clamp
                    val += tmp[plane * H * W + sy * W + x] * kernel[k];
                }
                dst[plane * H * W + y * W + x] = val;
            }
        }
    }

    return {leave_host_float32(std::move(output), orig), target};
}

// ============================================================================
// RandomAffine
// ============================================================================

auto RandomAffine::operator()(const Tensor& input, const Tensor& target)
    -> std::pair<Tensor, Tensor> {
    const auto& shape = input.shape();
    if (shape.size() < 2) {
        throw std::invalid_argument("RandomAffine requires at least 2D input (H, W)");
    }

    // Normalise to CPU/Float32/contiguous for the raw float* loop below; the
    // result is converted back to the caller's dtype/device on return.
    TransformDomain orig;
    Tensor in = enter_host_float32(input, orig);

    // B.2: deterministic when tenzor::manual_seed() is called; otherwise the
    // seed is drawn from the shared thread-local engine (advances per call) so
    // transforms in the same Compose/clock tick are not correlated. See
    // make_transform_rng().
    std::mt19937 rng = make_transform_rng();

    int64_t H = shape[shape.size() - 2];
    int64_t W = shape[shape.size() - 1];
    float cx = static_cast<float>(W) / 2.0f;
    float cy = static_cast<float>(H) / 2.0f;

    // Random parameters
    std::uniform_real_distribution<float> angle_dist(-degrees_, degrees_);
    std::uniform_real_distribution<float> tx_dist(-translate_x_ * W, translate_x_ * W);
    std::uniform_real_distribution<float> ty_dist(-translate_y_ * H, translate_y_ * H);
    std::uniform_real_distribution<float> scale_dist(scale_min_, scale_max_);
    std::uniform_real_distribution<float> shear_dist(-shear_, shear_);

    float angle_rad = angle_dist(rng) * static_cast<float>(M_PI) / 180.0f;
    float tx = (translate_x_ > 0) ? tx_dist(rng) : 0.0f;
    float ty = (translate_y_ > 0) ? ty_dist(rng) : 0.0f;
    float s = scale_dist(rng);
    float shear_rad = shear_dist(rng) * static_cast<float>(M_PI) / 180.0f;

    // Affine matrix: scale * rotation * shear, then translate
    // M = [[s*(cos - sin*shear),  s*(-sin - cos*shear), tx],
    //      [s*(sin + cos*shear),  s*(cos - sin*shear),  ty]]
    float cos_a = std::cos(angle_rad);
    float sin_a = std::sin(angle_rad);
    float sh = std::tan(shear_rad);

    float m00 = s * (cos_a + sin_a * sh);
    float m01 = s * (-sin_a + cos_a * sh);
    float m10 = s * sin_a;
    float m11 = s * cos_a;

    Tensor output = zeros(std::vector<int64_t>(shape.begin(), shape.end()), DType::Float32);
    int64_t num_planes = output.numel() / (H * W);

    const float* src = static_cast<const float*>(in.data_ptr());
    float* dst = static_cast<float*>(output.data_ptr());
    int64_t image_size = H * W;

    for (int64_t plane = 0; plane < num_planes; ++plane) {
        const float* s_ptr = src + plane * image_size;
        float* d_ptr = dst + plane * image_size;

        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                // Inverse mapping: output(x,y) <- input(sx,sy)
                float fx = static_cast<float>(x) - cx - tx;
                float fy = static_cast<float>(y) - cy - ty;

                // Inverse of affine matrix
                float det = m00 * m11 - m01 * m10;
                if (std::abs(det) < 1e-8f) continue;
                float inv_det = 1.0f / det;

                float src_x = (m11 * fx - m01 * fy) * inv_det + cx;
                float src_y = (-m10 * fx + m00 * fy) * inv_det + cy;

                int64_t sx = static_cast<int64_t>(std::round(src_x));
                int64_t sy = static_cast<int64_t>(std::round(src_y));

                if (sx >= 0 && sx < W && sy >= 0 && sy < H) {
                    d_ptr[y * W + x] = s_ptr[sy * W + sx];
                }
            }
        }
    }

    return {leave_host_float32(std::move(output), orig), target};
}

// ============================================================================
// RandomErasing
// ============================================================================

RandomErasing::RandomErasing(float p, float scale_min, float scale_max,
                             float ratio_min, float ratio_max, float value)
    : p_(p), scale_min_(scale_min), scale_max_(scale_max),
      ratio_min_(ratio_min), ratio_max_(ratio_max), value_(value) {
    if (p < 0.0f || p > 1.0f) {
        throw std::invalid_argument("Probability must be in [0, 1]");
    }
    if (scale_min < 0.0f || scale_max > 1.0f || scale_min > scale_max) {
        throw std::invalid_argument("Scale must satisfy 0 <= scale_min <= scale_max <= 1");
    }
    if (ratio_min <= 0.0f || ratio_min > ratio_max) {
        throw std::invalid_argument("Ratio must satisfy 0 < ratio_min <= ratio_max");
    }
}

auto RandomErasing::operator()(const Tensor& input, const Tensor& target)
    -> std::pair<Tensor, Tensor> {
    const auto& shape = input.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("RandomErasing requires 3D input (C, H, W)");
    }

    // B.2: deterministic when tenzor::manual_seed() is called; otherwise the
    // seed is drawn from the shared thread-local engine (advances per call) so
    // transforms in the same Compose/clock tick are not correlated. See
    // make_transform_rng().
    std::mt19937 rng = make_transform_rng();
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);

    if (coin(rng) >= p_) {
        return {input, target};
    }

    int64_t C = shape[0];
    int64_t H = shape[1];
    int64_t W = shape[2];
    float area = static_cast<float>(H * W);

    // Independent deep copy: assignment shares storage and would erase regions of
    // the caller's input tensor in place.
    Tensor output = input.clone();
    float* data = static_cast<float*>(output.data_ptr());

    std::uniform_real_distribution<float> scale_dist(scale_min_, scale_max_);
    std::uniform_real_distribution<float> log_ratio_dist(std::log(ratio_min_),
                                                         std::log(ratio_max_));

    for (int attempt = 0; attempt < 10; ++attempt) {
        float erase_area = area * scale_dist(rng);
        float aspect_ratio = std::exp(log_ratio_dist(rng));

        int64_t eh = static_cast<int64_t>(std::round(std::sqrt(erase_area / aspect_ratio)));
        int64_t ew = static_cast<int64_t>(std::round(std::sqrt(erase_area * aspect_ratio)));

        if (eh <= 0 || ew <= 0 || eh > H || ew > W) {
            continue;
        }

        std::uniform_int_distribution<int64_t> dy(0, H - eh);
        std::uniform_int_distribution<int64_t> dx(0, W - ew);
        int64_t top = dy(rng);
        int64_t left = dx(rng);

        for (int64_t c = 0; c < C; ++c) {
            for (int64_t y = top; y < top + eh; ++y) {
                for (int64_t x = left; x < left + ew; ++x) {
                    data[c * H * W + y * W + x] = value_;
                }
            }
        }
        break;
    }

    return {output, target};
}

// ============================================================================
// RandomPerspective
// ============================================================================

RandomPerspective::RandomPerspective(float distortion_scale, float p)
    : distortion_scale_(distortion_scale), p_(p) {
    if (p < 0.0f || p > 1.0f) {
        throw std::invalid_argument("Probability must be in [0, 1]");
    }
    if (distortion_scale < 0.0f) {
        throw std::invalid_argument("Distortion scale must be non-negative");
    }
}

auto RandomPerspective::operator()(const Tensor& input, const Tensor& target)
    -> std::pair<Tensor, Tensor> {
    const auto& shape = input.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("RandomPerspective requires 3D input (C, H, W)");
    }

    // Normalise to CPU/Float32/contiguous for the raw float* loop below; the
    // result is converted back to the caller's dtype/device on return.
    TransformDomain orig;
    Tensor in = enter_host_float32(input, orig);

    // B.2: deterministic when tenzor::manual_seed() is called; otherwise the
    // seed is drawn from the shared thread-local engine (advances per call) so
    // transforms in the same Compose/clock tick are not correlated. See
    // make_transform_rng().
    std::mt19937 rng = make_transform_rng();
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);

    if (coin(rng) >= p_) {
        return {input, target};
    }

    int64_t C = shape[0];
    int64_t H = shape[1];
    int64_t W = shape[2];

    float half_h = static_cast<float>(H) * distortion_scale_;
    float half_w = static_cast<float>(W) * distortion_scale_;
    std::uniform_real_distribution<float> disp_h(0.0f, half_h);
    std::uniform_real_distribution<float> disp_w(0.0f, half_w);

    // Source corners: top-left, top-right, bottom-right, bottom-left
    float sx[4] = {0.0f, static_cast<float>(W), static_cast<float>(W), 0.0f};
    float sy[4] = {0.0f, 0.0f, static_cast<float>(H), static_cast<float>(H)};

    // Destination corners with random displacement
    float dx[4], dy_arr[4];
    dx[0] = disp_w(rng);         dy_arr[0] = disp_h(rng);
    dx[1] = W - disp_w(rng);     dy_arr[1] = disp_h(rng);
    dx[2] = W - disp_w(rng);     dy_arr[2] = H - disp_h(rng);
    dx[3] = disp_w(rng);         dy_arr[3] = H - disp_h(rng);

    // Compute 3x3 perspective matrix from src->dst mapping
    // We need the inverse mapping (dst->src) for sampling
    // Solve for the 8-parameter perspective transform:
    //   x' = (a*x + b*y + c) / (g*x + h*y + 1)
    //   y' = (d*x + e*y + f) / (g*x + h*y + 1)
    // Set up linear system from 4 point correspondences (8 equations, 8 unknowns)
    // Here we solve dst->src (inverse mapping)

    // Build 8x8 system: for each corner i, we have:
    //   dst_x[i]*a + dst_y[i]*b + c - dst_x[i]*src_x[i]*g - dst_y[i]*src_x[i]*h = src_x[i]
    //   dst_x[i]*d + dst_y[i]*e + f - dst_x[i]*src_y[i]*g - dst_y[i]*src_y[i]*h = src_y[i]
    float A[8][8] = {};
    float B[8] = {};

    for (int i = 0; i < 4; ++i) {
        int r0 = 2 * i;
        int r1 = 2 * i + 1;
        A[r0][0] = dx[i]; A[r0][1] = dy_arr[i]; A[r0][2] = 1.0f;
        A[r0][3] = 0.0f;  A[r0][4] = 0.0f;      A[r0][5] = 0.0f;
        A[r0][6] = -dx[i] * sx[i]; A[r0][7] = -dy_arr[i] * sx[i];
        B[r0] = sx[i];

        A[r1][0] = 0.0f;  A[r1][1] = 0.0f;      A[r1][2] = 0.0f;
        A[r1][3] = dx[i]; A[r1][4] = dy_arr[i]; A[r1][5] = 1.0f;
        A[r1][6] = -dx[i] * sy[i]; A[r1][7] = -dy_arr[i] * sy[i];
        B[r1] = sy[i];
    }

    // Gaussian elimination with partial pivoting
    float coeffs[8];
    for (int col = 0; col < 8; ++col) {
        // Pivot
        int max_row = col;
        float max_val = std::abs(A[col][col]);
        for (int row = col + 1; row < 8; ++row) {
            if (std::abs(A[row][col]) > max_val) {
                max_val = std::abs(A[row][col]);
                max_row = row;
            }
        }
        if (max_row != col) {
            std::swap(B[col], B[max_row]);
            for (int k = 0; k < 8; ++k) std::swap(A[col][k], A[max_row][k]);
        }
        if (std::abs(A[col][col]) < 1e-10f) {
            // Degenerate, return unchanged
            return {input, target};
        }
        for (int row = col + 1; row < 8; ++row) {
            float factor = A[row][col] / A[col][col];
            for (int k = col; k < 8; ++k) A[row][k] -= factor * A[col][k];
            B[row] -= factor * B[col];
        }
    }
    // Back substitution
    for (int row = 7; row >= 0; --row) {
        coeffs[row] = B[row];
        for (int k = row + 1; k < 8; ++k) coeffs[row] -= A[row][k] * coeffs[k];
        coeffs[row] /= A[row][row];
    }

    float a = coeffs[0], b = coeffs[1], c = coeffs[2];
    float d = coeffs[3], e = coeffs[4], f = coeffs[5];
    float g = coeffs[6], h = coeffs[7];

    Tensor output = zeros(std::vector<int64_t>(shape.begin(), shape.end()), DType::Float32);
    const float* src_ptr = static_cast<const float*>(in.data_ptr());
    float* dst_ptr = static_cast<float*>(output.data_ptr());

    for (int64_t ch = 0; ch < C; ++ch) {
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                float px = static_cast<float>(x);
                float py = static_cast<float>(y);
                float denom = g * px + h * py + 1.0f;
                if (std::abs(denom) < 1e-8f) continue;

                float src_x = (a * px + b * py + c) / denom;
                float src_y = (d * px + e * py + f) / denom;

                // Bilinear interpolation
                if (src_x < 0.0f || src_x >= W - 1.0f || src_y < 0.0f || src_y >= H - 1.0f) {
                    continue; // out of bounds stays zero
                }

                int64_t x0 = static_cast<int64_t>(std::floor(src_x));
                int64_t y0 = static_cast<int64_t>(std::floor(src_y));
                int64_t x1 = x0 + 1;
                int64_t y1 = y0 + 1;
                float wx = src_x - static_cast<float>(x0);
                float wy = src_y - static_cast<float>(y0);

                x0 = std::clamp(x0, int64_t{0}, W - 1);
                x1 = std::clamp(x1, int64_t{0}, W - 1);
                y0 = std::clamp(y0, int64_t{0}, H - 1);
                y1 = std::clamp(y1, int64_t{0}, H - 1);

                float v00 = src_ptr[ch * H * W + y0 * W + x0];
                float v10 = src_ptr[ch * H * W + y0 * W + x1];
                float v01 = src_ptr[ch * H * W + y1 * W + x0];
                float v11 = src_ptr[ch * H * W + y1 * W + x1];

                float val = (1.0f - wy) * ((1.0f - wx) * v00 + wx * v10)
                          + wy * ((1.0f - wx) * v01 + wx * v11);
                dst_ptr[ch * H * W + y * W + x] = val;
            }
        }
    }

    return {leave_host_float32(std::move(output), orig), target};
}

// ============================================================================
// ElasticTransform
// ============================================================================

ElasticTransform::ElasticTransform(float alpha, float sigma, float p)
    : alpha_(alpha), sigma_(sigma), p_(p) {
    if (p < 0.0f || p > 1.0f) {
        throw std::invalid_argument("Probability must be in [0, 1]");
    }
    if (sigma <= 0.0f) {
        throw std::invalid_argument("Sigma must be positive");
    }
}

auto ElasticTransform::operator()(const Tensor& input, const Tensor& target)
    -> std::pair<Tensor, Tensor> {
    const auto& shape = input.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("ElasticTransform requires 3D input (C, H, W)");
    }

    // Normalise to CPU/Float32/contiguous for the raw float* loop below; the
    // result is converted back to the caller's dtype/device on return.
    TransformDomain orig;
    Tensor in = enter_host_float32(input, orig);

    // B.2: deterministic when tenzor::manual_seed() is called; otherwise the
    // seed is drawn from the shared thread-local engine (advances per call) so
    // transforms in the same Compose/clock tick are not correlated. See
    // make_transform_rng().
    std::mt19937 rng = make_transform_rng();
    std::uniform_real_distribution<float> coin(0.0f, 1.0f);

    if (coin(rng) >= p_) {
        return {input, target};
    }

    int64_t C = shape[0];
    int64_t H = shape[1];
    int64_t W = shape[2];

    // Generate random displacement fields in [-1, 1]
    std::uniform_real_distribution<float> disp_dist(-1.0f, 1.0f);
    std::vector<float> dx_field(H * W);
    std::vector<float> dy_field(H * W);
    for (int64_t i = 0; i < H * W; ++i) {
        dx_field[i] = disp_dist(rng);
        dy_field[i] = disp_dist(rng);
    }

    // Gaussian smoothing of displacement fields (separable)
    int kernel_size = static_cast<int>(std::ceil(sigma_ * 6.0f)) | 1; // ensure odd
    int half = kernel_size / 2;

    // Build 1D Gaussian kernel
    std::vector<float> kernel(kernel_size);
    float ksum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        float v = static_cast<float>(i - half);
        kernel[i] = std::exp(-v * v / (2.0f * sigma_ * sigma_));
        ksum += kernel[i];
    }
    for (auto& k : kernel) k /= ksum;

    // Smooth dx and dy with separable Gaussian
    auto smooth_field = [&](std::vector<float>& field) {
        std::vector<float> temp(H * W);

        // Horizontal pass
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                float val = 0.0f;
                for (int k = 0; k < kernel_size; ++k) {
                    int64_t sx = std::clamp(x + k - half, int64_t{0}, W - 1);
                    val += field[y * W + sx] * kernel[k];
                }
                temp[y * W + x] = val;
            }
        }

        // Vertical pass
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                float val = 0.0f;
                for (int k = 0; k < kernel_size; ++k) {
                    int64_t sy = std::clamp(y + k - half, int64_t{0}, H - 1);
                    val += temp[sy * W + x] * kernel[k];
                }
                field[y * W + x] = val;
            }
        }
    };

    smooth_field(dx_field);
    smooth_field(dy_field);

    // Scale by alpha
    for (int64_t i = 0; i < H * W; ++i) {
        dx_field[i] *= alpha_;
        dy_field[i] *= alpha_;
    }

    // Apply displacements with bilinear interpolation
    Tensor output = zeros(std::vector<int64_t>(shape.begin(), shape.end()), DType::Float32);
    const float* src_ptr = static_cast<const float*>(in.data_ptr());
    float* dst_ptr = static_cast<float*>(output.data_ptr());

    for (int64_t ch = 0; ch < C; ++ch) {
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                float src_x = static_cast<float>(x) + dx_field[y * W + x];
                float src_y = static_cast<float>(y) + dy_field[y * W + x];

                if (src_x < 0.0f || src_x >= W - 1.0f || src_y < 0.0f || src_y >= H - 1.0f) {
                    continue;
                }

                int64_t x0 = static_cast<int64_t>(std::floor(src_x));
                int64_t y0 = static_cast<int64_t>(std::floor(src_y));
                int64_t x1 = x0 + 1;
                int64_t y1 = y0 + 1;
                float wx = src_x - static_cast<float>(x0);
                float wy = src_y - static_cast<float>(y0);

                x0 = std::clamp(x0, int64_t{0}, W - 1);
                x1 = std::clamp(x1, int64_t{0}, W - 1);
                y0 = std::clamp(y0, int64_t{0}, H - 1);
                y1 = std::clamp(y1, int64_t{0}, H - 1);

                float v00 = src_ptr[ch * H * W + y0 * W + x0];
                float v10 = src_ptr[ch * H * W + y0 * W + x1];
                float v01 = src_ptr[ch * H * W + y1 * W + x0];
                float v11 = src_ptr[ch * H * W + y1 * W + x1];

                float val = (1.0f - wy) * ((1.0f - wx) * v00 + wx * v10)
                          + wy * ((1.0f - wx) * v01 + wx * v11);
                dst_ptr[ch * H * W + y * W + x] = val;
            }
        }
    }

    return {leave_host_float32(std::move(output), orig), target};
}

// ============================================================================
// MixUp
// ============================================================================

MixUp::MixUp(float alpha) : alpha_(alpha) {
    if (alpha <= 0.0f) {
        throw std::invalid_argument("Alpha must be positive");
    }
}

auto MixUp::operator()(const Tensor& input1, const Tensor& target1,
                        const Tensor& input2, const Tensor& target2)
    -> std::pair<Tensor, Tensor> {
    // B.2: deterministic when tenzor::manual_seed() is called; otherwise the
    // seed is drawn from the shared thread-local engine (advances per call) so
    // transforms in the same Compose/clock tick are not correlated. See
    // make_transform_rng().
    std::mt19937 rng = make_transform_rng();

    // Sample lambda from Beta(alpha, alpha) using gamma variates
    std::gamma_distribution<float> gamma_dist(alpha_, 1.0f);
    float x = gamma_dist(rng);
    float y = gamma_dist(rng);
    float lambda = x / (x + y);

    // mixed_input = lambda * input1 + (1 - lambda) * input2
    int64_t numel = input1.numel();
    Tensor mixed_input = zeros(
        std::vector<int64_t>(input1.shape().begin(), input1.shape().end()),
        input1.dtype());
    const float* src1 = static_cast<const float*>(input1.data_ptr());
    const float* src2 = static_cast<const float*>(input2.data_ptr());
    float* dst = static_cast<float*>(mixed_input.data_ptr());

    for (int64_t i = 0; i < numel; ++i) {
        dst[i] = lambda * src1[i] + (1.0f - lambda) * src2[i];
    }

    // mixed_target = lambda * target1 + (1 - lambda) * target2
    int64_t tgt_numel = target1.numel();
    Tensor mixed_target = zeros(
        std::vector<int64_t>(target1.shape().begin(), target1.shape().end()),
        target1.dtype());
    const float* tsrc1 = static_cast<const float*>(target1.data_ptr());
    const float* tsrc2 = static_cast<const float*>(target2.data_ptr());
    float* tdst = static_cast<float*>(mixed_target.data_ptr());

    for (int64_t i = 0; i < tgt_numel; ++i) {
        tdst[i] = lambda * tsrc1[i] + (1.0f - lambda) * tsrc2[i];
    }

    return {mixed_input, mixed_target};
}

// ============================================================================
// CutMix
// ============================================================================

CutMix::CutMix(float alpha) : alpha_(alpha) {
    if (alpha <= 0.0f) {
        throw std::invalid_argument("Alpha must be positive");
    }
}

auto CutMix::operator()(const Tensor& input1, const Tensor& target1,
                         const Tensor& input2, const Tensor& target2)
    -> std::pair<Tensor, Tensor> {
    const auto& shape = input1.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("CutMix requires 3D input (C, H, W)");
    }

    // B.2: deterministic when tenzor::manual_seed() is called; otherwise the
    // seed is drawn from the shared thread-local engine (advances per call) so
    // transforms in the same Compose/clock tick are not correlated. See
    // make_transform_rng().
    std::mt19937 rng = make_transform_rng();

    // Sample lambda from Beta(alpha, alpha)
    std::gamma_distribution<float> gamma_dist(alpha_, 1.0f);
    float x = gamma_dist(rng);
    float y = gamma_dist(rng);
    float lambda = x / (x + y);

    int64_t C = shape[0];
    int64_t H = shape[1];
    int64_t W = shape[2];

    // Compute cut rectangle dimensions: sqrt(1 - lambda) fraction of each side
    float cut_ratio = std::sqrt(1.0f - lambda);
    int64_t cut_h = static_cast<int64_t>(std::round(static_cast<float>(H) * cut_ratio));
    int64_t cut_w = static_cast<int64_t>(std::round(static_cast<float>(W) * cut_ratio));
    cut_h = std::clamp(cut_h, int64_t{1}, H);
    cut_w = std::clamp(cut_w, int64_t{1}, W);

    // Random center
    std::uniform_int_distribution<int64_t> cy_dist(0, H - 1);
    std::uniform_int_distribution<int64_t> cx_dist(0, W - 1);
    int64_t cy = cy_dist(rng);
    int64_t cx = cx_dist(rng);

    int64_t y1 = std::clamp(cy - cut_h / 2, int64_t{0}, H - cut_h);
    int64_t x1 = std::clamp(cx - cut_w / 2, int64_t{0}, W - cut_w);
    int64_t y2 = y1 + cut_h;
    int64_t x2 = x1 + cut_w;

    // Copy input1, paste region from input2
    // Independent deep copy of input1: assignment shares storage and the patch
    // write below would corrupt the caller's input1 tensor in place.
    Tensor mixed_input = input1.clone();
    float* dst = static_cast<float*>(mixed_input.data_ptr());
    const float* src2 = static_cast<const float*>(input2.data_ptr());

    for (int64_t c = 0; c < C; ++c) {
        for (int64_t row = y1; row < y2; ++row) {
            for (int64_t col = x1; col < x2; ++col) {
                dst[c * H * W + row * W + col] = src2[c * H * W + row * W + col];
            }
        }
    }

    // Actual lambda based on cut area
    float actual_lambda = 1.0f - static_cast<float>((y2 - y1) * (x2 - x1))
                                 / static_cast<float>(H * W);

    // Mixed target
    int64_t tgt_numel = target1.numel();
    Tensor mixed_target = zeros(
        std::vector<int64_t>(target1.shape().begin(), target1.shape().end()),
        target1.dtype());
    const float* tsrc1 = static_cast<const float*>(target1.data_ptr());
    const float* tsrc2 = static_cast<const float*>(target2.data_ptr());
    float* tdst = static_cast<float*>(mixed_target.data_ptr());

    for (int64_t i = 0; i < tgt_numel; ++i) {
        tdst[i] = actual_lambda * tsrc1[i] + (1.0f - actual_lambda) * tsrc2[i];
    }

    return {mixed_input, mixed_target};
}

// ============================================================================
// Augmentation pool helpers (shared by RandAugment, TrivialAugmentWide, AugMix)
// ============================================================================

namespace {

// Bilinear sample helper for spatial transforms
inline float bilinear_sample(const float* plane, int64_t H, int64_t W,
                              float src_x, float src_y) {
    if (src_x < 0.0f || src_x >= W - 1.0f || src_y < 0.0f || src_y >= H - 1.0f) {
        // Nearest-neighbor at edges
        int64_t nx = std::clamp(static_cast<int64_t>(std::round(src_x)), int64_t{0}, W - 1);
        int64_t ny = std::clamp(static_cast<int64_t>(std::round(src_y)), int64_t{0}, H - 1);
        return plane[ny * W + nx];
    }
    int64_t x0 = static_cast<int64_t>(std::floor(src_x));
    int64_t y0 = static_cast<int64_t>(std::floor(src_y));
    int64_t x1 = x0 + 1;
    int64_t y1 = y0 + 1;
    float wx = src_x - static_cast<float>(x0);
    float wy = src_y - static_cast<float>(y0);
    return (1.0f - wy) * ((1.0f - wx) * plane[y0 * W + x0] + wx * plane[y0 * W + x1])
         + wy * ((1.0f - wx) * plane[y1 * W + x0] + wx * plane[y1 * W + x1]);
}

// Apply an affine transform (2x3 matrix) to a CHW tensor in-place via inverse mapping
void apply_affine_chw(const float* src, float* dst, int64_t C, int64_t H, int64_t W,
                      float m00, float m01, float m02,
                      float m10, float m11, float m12) {
    // Inverse of 2x2 part
    float det = m00 * m11 - m01 * m10;
    if (std::abs(det) < 1e-8f) return;
    float inv = 1.0f / det;
    float i00 = m11 * inv, i01 = -m01 * inv;
    float i10 = -m10 * inv, i11 = m00 * inv;

    float cx = static_cast<float>(W) * 0.5f;
    float cy = static_cast<float>(H) * 0.5f;

    for (int64_t ch = 0; ch < C; ++ch) {
        const float* s = src + ch * H * W;
        float* d = dst + ch * H * W;
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                float fx = static_cast<float>(x) - cx - m02;
                float fy = static_cast<float>(y) - cy - m12;
                float sx = i00 * fx + i01 * fy + cx;
                float sy = i10 * fx + i11 * fy + cy;
                d[y * W + x] = bilinear_sample(s, H, W, sx, sy);
            }
        }
    }
}

enum class AugOp {
    Identity, Rotate, ShearX, ShearY, TranslateX, TranslateY,
    Brightness, Contrast, Sharpness, Solarize, Posterize,
    Equalize, AutoContrast,
    kCount
};

constexpr int kNumAugOps = static_cast<int>(AugOp::kCount);

// Apply a single augmentation op to a CHW float tensor
// magnitude is in [0, 1] range (normalized from magnitude level)
void apply_aug_op(AugOp op, float magnitude, float* data, int64_t C, int64_t H, int64_t W,
                  std::mt19937& rng) {
    int64_t spatial = H * W;
    int64_t total = C * spatial;

    // Random sign for geometric transforms
    std::uniform_int_distribution<int> sign_dist(0, 1);
    float sign = sign_dist(rng) ? 1.0f : -1.0f;

    switch (op) {
    case AugOp::Identity:
        break;

    case AugOp::Rotate: {
        float angle = sign * magnitude * 30.0f; // up to +-30 degrees
        float rad = angle * static_cast<float>(M_PI) / 180.0f;
        float cos_a = std::cos(rad), sin_a = std::sin(rad);
        std::vector<float> tmp(total);
        std::copy(data, data + total, tmp.data());
        // Rotation matrix centered on image
        apply_affine_chw(tmp.data(), data, C, H, W,
                         cos_a, -sin_a, 0.0f,
                         sin_a, cos_a, 0.0f);
        break;
    }

    case AugOp::ShearX: {
        float shear = sign * magnitude * 0.3f; // up to +-0.3
        std::vector<float> tmp(total);
        std::copy(data, data + total, tmp.data());
        apply_affine_chw(tmp.data(), data, C, H, W,
                         1.0f, shear, 0.0f,
                         0.0f, 1.0f, 0.0f);
        break;
    }

    case AugOp::ShearY: {
        float shear = sign * magnitude * 0.3f;
        std::vector<float> tmp(total);
        std::copy(data, data + total, tmp.data());
        apply_affine_chw(tmp.data(), data, C, H, W,
                         1.0f, 0.0f, 0.0f,
                         shear, 1.0f, 0.0f);
        break;
    }

    case AugOp::TranslateX: {
        float tx = sign * magnitude * static_cast<float>(W) * 0.33f;
        std::vector<float> tmp(total);
        std::copy(data, data + total, tmp.data());
        apply_affine_chw(tmp.data(), data, C, H, W,
                         1.0f, 0.0f, tx,
                         0.0f, 1.0f, 0.0f);
        break;
    }

    case AugOp::TranslateY: {
        float ty = sign * magnitude * static_cast<float>(H) * 0.33f;
        std::vector<float> tmp(total);
        std::copy(data, data + total, tmp.data());
        apply_affine_chw(tmp.data(), data, C, H, W,
                         1.0f, 0.0f, 0.0f,
                         0.0f, 1.0f, ty);
        break;
    }

    case AugOp::Brightness: {
        float factor = 1.0f + sign * magnitude * 0.9f; // [0.1, 1.9]
        for (int64_t i = 0; i < total; ++i) {
            data[i] *= factor;
        }
        break;
    }

    case AugOp::Contrast: {
        float factor = 1.0f + sign * magnitude * 0.9f;
        // Compute mean
        float mean = 0.0f;
        for (int64_t i = 0; i < total; ++i) mean += data[i];
        mean /= static_cast<float>(total);
        for (int64_t i = 0; i < total; ++i) {
            data[i] = mean + factor * (data[i] - mean);
        }
        break;
    }

    case AugOp::Sharpness: {
        // Simple 3x3 unsharp mask blend
        float factor = 1.0f + sign * magnitude * 0.9f;
        std::vector<float> blurred(total);
        // Box blur each channel
        for (int64_t ch = 0; ch < C; ++ch) {
            const float* src = data + ch * spatial;
            float* dst = blurred.data() + ch * spatial;
            for (int64_t y = 0; y < H; ++y) {
                for (int64_t x = 0; x < W; ++x) {
                    float sum = 0.0f;
                    int count = 0;
                    for (int64_t dy = -1; dy <= 1; ++dy) {
                        for (int64_t dx = -1; dx <= 1; ++dx) {
                            int64_t ny = std::clamp(y + dy, int64_t{0}, H - 1);
                            int64_t nx = std::clamp(x + dx, int64_t{0}, W - 1);
                            sum += src[ny * W + nx];
                            ++count;
                        }
                    }
                    dst[y * W + x] = sum / static_cast<float>(count);
                }
            }
        }
        // Blend: result = blurred + factor * (original - blurred)
        for (int64_t i = 0; i < total; ++i) {
            data[i] = blurred[i] + factor * (data[i] - blurred[i]);
        }
        break;
    }

    case AugOp::Solarize: {
        // Invert pixels above threshold; threshold decreases with magnitude
        float threshold = 1.0f - magnitude; // assumes [0,1] pixel range
        for (int64_t i = 0; i < total; ++i) {
            if (data[i] >= threshold) {
                data[i] = 1.0f - data[i];
            }
        }
        break;
    }

    case AugOp::Posterize: {
        // Reduce bit depth; fewer bits at higher magnitude
        int bits = std::max(1, 8 - static_cast<int>(magnitude * 7.0f));
        float levels = static_cast<float>(1 << bits);
        for (int64_t i = 0; i < total; ++i) {
            float v = std::clamp(data[i], 0.0f, 1.0f);
            data[i] = std::floor(v * (levels - 1.0f) + 0.5f) / (levels - 1.0f);
        }
        break;
    }

    case AugOp::Equalize: {
        // Per-channel histogram equalization (256 bins, assumes [0,1])
        for (int64_t ch = 0; ch < C; ++ch) {
            float* ch_data = data + ch * spatial;
            int hist[256] = {};
            for (int64_t i = 0; i < spatial; ++i) {
                int bin = std::clamp(static_cast<int>(ch_data[i] * 255.0f), 0, 255);
                hist[bin]++;
            }
            // CDF
            int cdf[256];
            cdf[0] = hist[0];
            for (int b = 1; b < 256; ++b) cdf[b] = cdf[b - 1] + hist[b];
            int cdf_min = 0;
            for (int b = 0; b < 256; ++b) {
                if (cdf[b] > 0) { cdf_min = cdf[b]; break; }
            }
            float denom = static_cast<float>(spatial - cdf_min);
            if (denom < 1.0f) break;
            for (int64_t i = 0; i < spatial; ++i) {
                int bin = std::clamp(static_cast<int>(ch_data[i] * 255.0f), 0, 255);
                ch_data[i] = static_cast<float>(cdf[bin] - cdf_min) / denom;
            }
        }
        break;
    }

    case AugOp::AutoContrast: {
        // Per-channel linear stretch to [0, 1]
        for (int64_t ch = 0; ch < C; ++ch) {
            float* ch_data = data + ch * spatial;
            float lo = ch_data[0], hi = ch_data[0];
            for (int64_t i = 1; i < spatial; ++i) {
                lo = std::min(lo, ch_data[i]);
                hi = std::max(hi, ch_data[i]);
            }
            float range = hi - lo;
            if (range < 1e-8f) continue;
            float inv_range = 1.0f / range;
            for (int64_t i = 0; i < spatial; ++i) {
                ch_data[i] = (ch_data[i] - lo) * inv_range;
            }
        }
        break;
    }

    default:
        break;
    }
}

} // anonymous namespace

// ============================================================================
// RandAugment
// ============================================================================

RandAugment::RandAugment(int num_ops, int magnitude, float magnitude_std)
    : num_ops_(num_ops), magnitude_(magnitude), magnitude_std_(magnitude_std) {
    if (num_ops < 0) {
        throw std::invalid_argument("num_ops must be non-negative");
    }
    if (magnitude < 0 || magnitude > 30) {
        throw std::invalid_argument("magnitude must be in [0, 30]");
    }
}

auto RandAugment::operator()(const Tensor& input, const Tensor& target)
    -> std::pair<Tensor, Tensor> {
    const auto& shape = input.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("RandAugment requires 3D input (C, H, W)");
    }

    // Normalise to CPU/Float32/contiguous for the raw float* loop below; the
    // result is converted back to the caller's dtype/device on return.
    TransformDomain orig;
    Tensor in = enter_host_float32(input, orig);

    // B.2: deterministic when tenzor::manual_seed() is called; otherwise the
    // seed is drawn from the shared thread-local engine (advances per call) so
    // transforms in the same Compose/clock tick are not correlated. See
    // make_transform_rng().
    std::mt19937 rng = make_transform_rng();

    int64_t C = shape[0];
    int64_t H = shape[1];
    int64_t W = shape[2];
    int64_t total = C * H * W;

    // Copy input data
    Tensor output = zeros(std::vector<int64_t>(shape.begin(), shape.end()), DType::Float32);
    const float* src = static_cast<const float*>(in.data_ptr());
    float* dst = static_cast<float*>(output.data_ptr());
    std::copy(src, src + total, dst);

    std::uniform_int_distribution<int> op_dist(0, kNumAugOps - 1);
    std::normal_distribution<float> mag_noise(0.0f, magnitude_std_);

    for (int i = 0; i < num_ops_; ++i) {
        auto op = static_cast<AugOp>(op_dist(rng));

        // Randomize magnitude around the set level
        float mag = static_cast<float>(magnitude_) + mag_noise(rng);
        mag = std::clamp(mag, 0.0f, 30.0f);
        float normalized_mag = mag / 30.0f; // normalize to [0, 1]

        apply_aug_op(op, normalized_mag, dst, C, H, W, rng);
    }

    return {leave_host_float32(std::move(output), orig), target};
}

// ============================================================================
// TrivialAugmentWide
// ============================================================================

TrivialAugmentWide::TrivialAugmentWide(int num_magnitude_bins)
    : num_magnitude_bins_(num_magnitude_bins) {
    if (num_magnitude_bins < 1) {
        throw std::invalid_argument("num_magnitude_bins must be positive");
    }
}

auto TrivialAugmentWide::operator()(const Tensor& input, const Tensor& target)
    -> std::pair<Tensor, Tensor> {
    const auto& shape = input.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("TrivialAugmentWide requires 3D input (C, H, W)");
    }

    // Normalise to CPU/Float32/contiguous for the raw float* loop below; the
    // result is converted back to the caller's dtype/device on return.
    TransformDomain orig;
    Tensor in = enter_host_float32(input, orig);

    // B.2: deterministic when tenzor::manual_seed() is called; otherwise the
    // seed is drawn from the shared thread-local engine (advances per call) so
    // transforms in the same Compose/clock tick are not correlated. See
    // make_transform_rng().
    std::mt19937 rng = make_transform_rng();

    int64_t C = shape[0];
    int64_t H = shape[1];
    int64_t W = shape[2];
    int64_t total = C * H * W;

    // Copy input data
    Tensor output = zeros(std::vector<int64_t>(shape.begin(), shape.end()), DType::Float32);
    const float* src = static_cast<const float*>(in.data_ptr());
    float* dst = static_cast<float*>(output.data_ptr());
    std::copy(src, src + total, dst);

    // Select one random op
    std::uniform_int_distribution<int> op_dist(0, kNumAugOps - 1);
    auto op = static_cast<AugOp>(op_dist(rng));

    // Uniform magnitude
    std::uniform_int_distribution<int> mag_dist(0, num_magnitude_bins_ - 1);
    float normalized_mag = static_cast<float>(mag_dist(rng))
                           / static_cast<float>(num_magnitude_bins_ - 1);

    apply_aug_op(op, normalized_mag, dst, C, H, W, rng);

    return {leave_host_float32(std::move(output), orig), target};
}

// ============================================================================
// AugMix
// ============================================================================

AugMix::AugMix(int width, int depth, float severity, float alpha)
    : width_(width), depth_(depth), severity_(severity), alpha_(alpha) {
    if (width < 1) {
        throw std::invalid_argument("width must be positive");
    }
    if (severity < 0.0f || severity > 10.0f) {
        throw std::invalid_argument("severity must be in [0, 10]");
    }
    if (alpha <= 0.0f) {
        throw std::invalid_argument("alpha must be positive");
    }
}

auto AugMix::operator()(const Tensor& input, const Tensor& target)
    -> std::pair<Tensor, Tensor> {
    const auto& shape = input.shape();
    if (shape.size() != 3) {
        throw std::invalid_argument("AugMix requires 3D input (C, H, W)");
    }

    // Normalise to CPU/Float32/contiguous for the raw float* loop below; the
    // result is converted back to the caller's dtype/device on return.
    TransformDomain orig;
    Tensor in = enter_host_float32(input, orig);

    // B.2: deterministic when tenzor::manual_seed() is called; otherwise the
    // seed is drawn from the shared thread-local engine (advances per call) so
    // transforms in the same Compose/clock tick are not correlated. See
    // make_transform_rng().
    std::mt19937 rng = make_transform_rng();

    int64_t C = shape[0];
    int64_t H = shape[1];
    int64_t W = shape[2];
    int64_t total = C * H * W;

    const float* src = static_cast<const float*>(in.data_ptr());

    // Normalized magnitude from severity
    float normalized_mag = severity_ / 10.0f;

    // Sample Dirichlet weights (width_ values)
    std::gamma_distribution<float> gamma_dist(alpha_, 1.0f);
    std::vector<float> weights(width_);
    float weight_sum = 0.0f;
    for (int i = 0; i < width_; ++i) {
        weights[i] = gamma_dist(rng);
        weight_sum += weights[i];
    }
    for (auto& w : weights) w /= weight_sum;

    // Sample mixing coefficient m from Beta(alpha, alpha)
    float gx = gamma_dist(rng);
    float gy = gamma_dist(rng);
    float m = gx / (gx + gy);

    // Create augmentation chains and accumulate weighted result
    std::vector<float> mixed(total, 0.0f);

    std::uniform_int_distribution<int> op_dist(0, kNumAugOps - 1);
    std::uniform_int_distribution<int> depth_dist(1, 3);

    for (int chain = 0; chain < width_; ++chain) {
        // Start from a copy of original
        std::vector<float> chain_data(src, src + total);

        int chain_depth = (depth_ > 0) ? depth_ : depth_dist(rng);

        for (int d = 0; d < chain_depth; ++d) {
            auto op = static_cast<AugOp>(op_dist(rng));
            apply_aug_op(op, normalized_mag, chain_data.data(), C, H, W, rng);
        }

        // Accumulate weighted chain
        for (int64_t i = 0; i < total; ++i) {
            mixed[i] += weights[chain] * chain_data[i];
        }
    }

    // Final mix: output = m * original + (1 - m) * mixed
    Tensor output = zeros(std::vector<int64_t>(shape.begin(), shape.end()), DType::Float32);
    float* dst = static_cast<float*>(output.data_ptr());

    for (int64_t i = 0; i < total; ++i) {
        dst[i] = m * src[i] + (1.0f - m) * mixed[i];
    }

    return {leave_host_float32(std::move(output), orig), target};
}

} // namespace transforms
} // namespace data
} // namespace tenzor
