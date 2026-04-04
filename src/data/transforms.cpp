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

} // namespace transforms
} // namespace data
} // namespace tenzor
