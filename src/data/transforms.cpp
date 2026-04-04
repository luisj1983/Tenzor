#include "tenzor/data/transforms.hpp"
#include "tenzor/ops/creation.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

namespace tenzor {
namespace data {
namespace transforms {

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

    // Generate random angle in [min_degrees_, max_degrees_]
    static thread_local std::mt19937 rng{std::random_device{}()};
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

    // Create output tensor filled with zeros (same shape and dtype)
    Tensor output = zeros(std::vector<int64_t>(shape.begin(), shape.end()),
                          input.dtype());

    // Compute total number of "images" (product of all dims except H, W)
    int64_t num_images = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        num_images *= shape[i];
    }

    // Nearest-neighbor rotation via inverse mapping
    // For each pixel (y, x) in output, find source pixel in input
    const float* src_ptr = static_cast<const float*>(input.data_ptr());
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

    return {output, target};
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

    static thread_local std::mt19937 rng{std::random_device{}()};

    // Work with a copy to avoid modifying the original
    Tensor output = input;
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

    // Apply contrast: blend toward the per-channel mean
    // result = mean + contrast_factor * (value - mean)
    if (contrast_ > 0) {
        std::uniform_real_distribution<float> dist(
            1.0f - contrast_, 1.0f + contrast_);
        float factor = dist(rng);

        // Compute global mean (simplified)
        float sum = 0.0f;
        for (int64_t i = 0; i < numel; ++i) {
            sum += data[i];
        }
        float mean = sum / static_cast<float>(numel);

        for (int64_t i = 0; i < numel; ++i) {
            data[i] = mean + factor * (data[i] - mean);
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

    static thread_local std::mt19937 rng{std::random_device{}()};

    // Assume last two dims are H, W
    int64_t H = shape[shape.size() - 2];
    int64_t W = shape[shape.size() - 1];

    // Work with a copy
    Tensor output = input;
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

    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    if (dist(rng) >= p_) {
        return {input, target};
    }

    int64_t H = shape[shape.size() - 2];
    int64_t W = shape[shape.size() - 1];

    Tensor output = zeros(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype());
    int64_t num_planes = output.numel() / (H * W);

    const float* src = static_cast<const float*>(input.data_ptr());
    float* dst = static_cast<float*>(output.data_ptr());

    for (int64_t plane = 0; plane < num_planes; ++plane) {
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                dst[plane * H * W + y * W + x] =
                    src[plane * H * W + (H - 1 - y) * W + x];
            }
        }
    }

    return {output, target};
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

    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    if (dist(rng) >= p_) {
        return {input, target};
    }

    int64_t H = shape[shape.size() - 2];
    int64_t W = shape[shape.size() - 1];

    Tensor output = zeros(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype());
    int64_t num_planes = output.numel() / (H * W);

    const float* src = static_cast<const float*>(input.data_ptr());
    float* dst = static_cast<float*>(output.data_ptr());

    for (int64_t plane = 0; plane < num_planes; ++plane) {
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                dst[plane * H * W + y * W + x] =
                    src[plane * H * W + y * W + (W - 1 - x)];
            }
        }
    }

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

    // Build output shape
    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[shape.size() - 2] = height_;
    out_shape[shape.size() - 1] = width_;

    Tensor output = zeros(out_shape, input.dtype());
    int64_t num_planes = output.numel() / (height_ * width_);

    const float* src = static_cast<const float*>(input.data_ptr());
    float* dst = static_cast<float*>(output.data_ptr());

    for (int64_t plane = 0; plane < num_planes; ++plane) {
        for (int64_t y = 0; y < height_; ++y) {
            for (int64_t x = 0; x < width_; ++x) {
                dst[plane * height_ * width_ + y * width_ + x] =
                    src[plane * H * W + (top + y) * W + (left + x)];
            }
        }
    }

    return {output, target};
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

    static thread_local std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int64_t> dist_y(0, H - height_);
    std::uniform_int_distribution<int64_t> dist_x(0, W - width_);

    int64_t top = dist_y(rng) - padding_;
    int64_t left = dist_x(rng) - padding_;

    int64_t orig_H = shape[shape.size() - 2];
    int64_t orig_W = shape[shape.size() - 1];

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[shape.size() - 2] = height_;
    out_shape[shape.size() - 1] = width_;

    Tensor output = zeros(out_shape, input.dtype());
    int64_t num_planes = output.numel() / (height_ * width_);

    const float* src = static_cast<const float*>(input.data_ptr());
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

    return {output, target};
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

    std::vector<int64_t> out_shape(shape.begin(), shape.end());
    out_shape[shape.size() - 2] = height_;
    out_shape[shape.size() - 1] = width_;

    Tensor output = zeros(out_shape, input.dtype());
    int64_t num_planes = output.numel() / (height_ * width_);

    const float* src = static_cast<const float*>(input.data_ptr());
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

    return {output, target};
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

    static thread_local std::mt19937 rng{std::random_device{}()};
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

    Tensor output = zeros(out_shape, input.dtype());
    int64_t num_planes = output.numel() / (height_ * width_);

    const float* src = static_cast<const float*>(input.data_ptr());
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

    return {output, target};
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

    static thread_local std::mt19937 rng{std::random_device{}()};
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

    Tensor output = zeros(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype());
    Tensor temp = zeros(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype());
    int64_t num_planes = output.numel() / (H * W);

    const float* src = static_cast<const float*>(input.data_ptr());
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

    return {output, target};
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

    static thread_local std::mt19937 rng{std::random_device{}()};

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

    Tensor output = zeros(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype());
    int64_t num_planes = output.numel() / (H * W);

    const float* src = static_cast<const float*>(input.data_ptr());
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

    return {output, target};
}

} // namespace transforms
} // namespace data
} // namespace tenzor
