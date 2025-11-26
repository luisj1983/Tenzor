/**
 * @file vit.cpp
 * @brief Implementation of Vision Transformer model family
 */

#include "tenzor/models/vit.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/transform.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <filesystem>

namespace tenzor {
namespace models {

// ============================================================================
// PatchEmbedding Implementation
// ============================================================================

PatchEmbedding::PatchEmbedding(int64_t image_size, int64_t patch_size,
                               int64_t num_channels, int64_t hidden_size)
    : image_size_(image_size)
    , patch_size_(patch_size)
    , num_channels_(num_channels)
    , hidden_size_(hidden_size) {

    // Calculate number of patches
    if (image_size % patch_size != 0) {
        throw std::invalid_argument(
            "Image size (" + std::to_string(image_size) +
            ") must be divisible by patch size (" + std::to_string(patch_size) + ")");
    }

    int64_t patches_per_dim = image_size / patch_size;
    num_patches_ = patches_per_dim * patches_per_dim;

    // Create convolutional projection
    // Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    projection_ = std::make_shared<nn::Conv2d>(
        num_channels,      // in_channels
        hidden_size,       // out_channels
        patch_size,        // kernel_size (same as patch_size)
        patch_size,        // stride (same as patch_size for non-overlapping)
        0                  // padding (no padding needed)
    );

    register_module("projection", projection_);
}

auto PatchEmbedding::forward_impl(const Variable& x) -> Variable {
    // Input shape: [batch, channels, height, width]
    auto shape = x.shape();
    int64_t batch_size = shape[0];
    int64_t channels = shape[1];
    int64_t height = shape[2];
    int64_t width = shape[3];

    // Validate input
    if (channels != num_channels_) {
        throw std::runtime_error(
            "Expected " + std::to_string(num_channels_) + " channels, got " +
            std::to_string(channels));
    }
    if (height != image_size_ || width != image_size_) {
        throw std::runtime_error(
            "Expected image size " + std::to_string(image_size_) + "x" +
            std::to_string(image_size_) + ", got " + std::to_string(height) +
            "x" + std::to_string(width));
    }

    // Apply convolutional projection
    // Output shape: [batch, hidden_size, height/patch_size, width/patch_size]
    auto patches = projection_->forward(x);

    auto proj_shape = patches.shape();
    int64_t out_h = proj_shape[2];
    int64_t out_w = proj_shape[3];

    // Reshape to [batch, hidden_size, num_patches]
    // num_patches = out_h * out_w
    patches = tenzor::reshape(patches, {batch_size, hidden_size_, out_h * out_w});

    // Transpose to [batch, num_patches, hidden_size]
    patches = tenzor::transpose(patches, 1, 2);

    return patches;
}

// ============================================================================
// ViTEmbeddings Implementation
// ============================================================================

ViTEmbeddings::ViTEmbeddings(const ViTConfig& config)
    : config_(config) {

    // Patch embeddings
    patch_embeddings_ = std::make_shared<PatchEmbedding>(
        config.image_size,
        config.patch_size,
        config.num_channels,
        config.hidden_size
    );
    register_module("patch_embeddings", patch_embeddings_);

    // Dropout
    dropout_ = std::make_shared<nn::Dropout>(config.hidden_dropout_prob);
    register_module("dropout", dropout_);

    // Initialize learnable parameters
    initialize_parameters();
}

auto ViTEmbeddings::initialize_parameters() -> void {
    // [CLS] token: [1, 1, hidden_size]
    // Initialize to zeros (standard practice for ViT)
    Tensor cls_tensor({1, 1, config_.hidden_size}, DType::Float32, Device::cpu());
    cls_tensor.zero_();
    cls_token_ = Variable(cls_tensor, true);  // Requires gradient
    register_parameter("cls_token", cls_token_);

    // Position embeddings: [1, seq_len, hidden_size]
    // seq_len = num_patches + 1 (for [CLS] token)
    int64_t seq_len = config_.seq_length();
    Tensor pos_tensor({1, seq_len, config_.hidden_size}, DType::Float32, Device::cpu());

    // Initialize with truncated normal distribution
    // Standard deviation = 0.02 (standard for ViT)
    float* pos_data = pos_tensor.data<float>();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f);

    for (int64_t i = 0; i < pos_tensor.numel(); ++i) {
        pos_data[i] = dist(gen);
    }

    position_embeddings_ = Variable(pos_tensor, true);  // Requires gradient
    register_parameter("position_embeddings", position_embeddings_);
}

auto ViTEmbeddings::forward_impl(const Variable& pixel_values) -> Variable {
    auto shape = pixel_values.shape();
    int64_t batch_size = shape[0];

    // Get patch embeddings: [batch, num_patches, hidden_size]
    auto embeddings = patch_embeddings_->forward(pixel_values);

    // Expand [CLS] token for batch: [1, 1, hidden_size] -> [batch, 1, hidden_size]
    // We need to repeat the [CLS] token for each sample in the batch
    auto cls_shape = cls_token_.shape();
    Tensor cls_expanded({batch_size, 1, config_.hidden_size},
                        DType::Float32, pixel_values.tensor().device());

    // Copy [CLS] token data for each batch
    const float* cls_data = cls_token_.tensor().data<float>();
    float* expanded_data = cls_expanded.data<float>();
    for (int64_t b = 0; b < batch_size; ++b) {
        std::copy(cls_data, cls_data + config_.hidden_size,
                  expanded_data + b * config_.hidden_size);
    }

    Variable cls_tokens(cls_expanded, true);
    cls_tokens.set_grad_fn(cls_token_.grad_fn());  // Share gradient function

    // Concatenate [CLS] token with patch embeddings using autograd cat
    // This maintains the gradient chain for proper backpropagation
    // Result: [batch, num_patches + 1, hidden_size]
    std::vector<Variable> to_concat = {cls_tokens, embeddings};
    embeddings = cat(to_concat, 1);

    // Add position embeddings
    // Position embeddings are [1, seq_len, hidden_size], broadcast to [batch, seq_len, hidden_size]
    embeddings = embeddings + position_embeddings_;

    // Apply dropout
    embeddings = dropout_->forward(embeddings);

    return embeddings;
}

// ============================================================================
// ViTEncoder Implementation
// ============================================================================

ViTEncoder::ViTEncoder(const ViTConfig& config)
    : config_(config) {

    // Create a single encoder layer with ViT-specific settings
    auto encoder_layer = std::make_shared<nn::TransformerEncoderLayer>(
        config.hidden_size,
        config.num_attention_heads,
        config.intermediate_size,
        config.attention_probs_dropout_prob,
        config.hidden_act,
        true  // batch_first=true (ViT uses batch-first format)
    );

    // No final layer norm in the encoder stack
    // ViT applies layer norm after the encoder in the main model
    encoder_ = std::make_shared<nn::TransformerEncoder>(
        encoder_layer,
        config.num_hidden_layers,
        nullptr  // No norm layer in encoder stack
    );

    register_module("encoder", encoder_);
}

auto ViTEncoder::forward_impl(const Variable& hidden_states) -> Variable {
    // Pass through encoder layers
    // No attention mask needed for ViT (all patches attend to all patches)
    return encoder_->forward(hidden_states, Tensor{}, Tensor{});
}

// ============================================================================
// ViT Implementation
// ============================================================================

ViT::ViT(const ViTConfig& config, bool add_pooling_layer)
    : config_(config)
    , add_pooling_layer_(add_pooling_layer) {

    // Initialize components
    embeddings_ = std::make_shared<ViTEmbeddings>(config);
    encoder_ = std::make_shared<ViTEncoder>(config);
    layernorm_ = std::make_shared<nn::LayerNorm>(
        std::vector<int64_t>{config.hidden_size},
        config.layer_norm_eps
    );

    register_module("embeddings", embeddings_);
    register_module("encoder", encoder_);
    register_module("layernorm", layernorm_);

    // Optional pooler (for classification)
    if (add_pooling_layer) {
        pooler_ = std::make_shared<nn::Linear>(config.hidden_size, config.hidden_size);
        register_module("pooler", pooler_);
    }
}

auto ViT::forward_vit(const Variable& pixel_values) -> ViTOutput {
    // Get embeddings
    auto embedding_output = embeddings_->forward(pixel_values);

    // Encode
    auto encoder_output = encoder_->forward(embedding_output);

    // Apply final layer norm
    auto sequence_output = layernorm_->forward(encoder_output);

    // Extract [CLS] token and pool
    Variable pooled_output;
    if (add_pooling_layer_) {
        // Extract [CLS] token (first token in sequence)
        auto shape = sequence_output.shape();
        int64_t batch_size = shape[0];
        int64_t seq_len = shape[1];
        int64_t hidden_size = shape[2];

        // Reshape and extract first token using matrix multiplication trick
        auto reshaped = tenzor::reshape(sequence_output, {batch_size * seq_len, hidden_size});

        // Create selection matrix to extract first tokens
        Tensor selection_matrix({batch_size, batch_size * seq_len},
                               DType::Float32, sequence_output.tensor().device());
        selection_matrix.zero_();
        float* sel_data = selection_matrix.data<float>();
        for (int64_t b = 0; b < batch_size; ++b) {
            sel_data[b * (batch_size * seq_len) + b * seq_len] = 1.0f;
        }

        Variable selection_var(selection_matrix, false);
        auto cls_token = tenzor::matmul(selection_var, reshaped);  // [batch, hidden_size]

        // Apply pooler (linear + tanh)
        pooled_output = pooler_->forward(cls_token);
        pooled_output = nn::tanh(pooled_output);
    } else {
        // Return empty variable if no pooling
        pooled_output = Variable();
    }

    return ViTOutput{sequence_output, pooled_output};
}

// ============================================================================
// ViTForImageClassification Implementation
// ============================================================================

ViTForImageClassification::ViTForImageClassification(
    const ViTConfig& config, int64_t num_labels)
    : config_(config)
    , num_labels_(num_labels) {

    // Base ViT model (without pooling layer since we'll extract [CLS] ourselves)
    vit_ = std::make_shared<ViT>(config, false);
    register_module("vit", vit_);

    // Classification head
    classifier_ = std::make_shared<nn::Linear>(config.hidden_size, num_labels);
    register_module("classifier", classifier_);
}

auto ViTForImageClassification::forward_impl(const Variable& pixel_values) -> Variable {
    // Get ViT outputs
    auto outputs = vit_->forward_vit(pixel_values);
    auto sequence_output = outputs.last_hidden_state;

    // Extract [CLS] token (first token)
    auto shape = sequence_output.shape();
    int64_t batch_size = shape[0];
    int64_t seq_len = shape[1];
    int64_t hidden_size = shape[2];

    // Use matrix multiplication to extract first token
    auto reshaped = tenzor::reshape(sequence_output, {batch_size * seq_len, hidden_size});

    Tensor selection_matrix({batch_size, batch_size * seq_len},
                           DType::Float32, sequence_output.tensor().device());
    selection_matrix.zero_();
    float* sel_data = selection_matrix.data<float>();
    for (int64_t b = 0; b < batch_size; ++b) {
        sel_data[b * (batch_size * seq_len) + b * seq_len] = 1.0f;
    }

    Variable selection_var(selection_matrix, false);
    auto cls_output = tenzor::matmul(selection_var, reshaped);  // [batch, hidden_size]

    // Classify
    auto logits = classifier_->forward(cls_output);

    return logits;
}

// ============================================================================
// Factory Functions Implementation
// ============================================================================

auto ViT_Base_Patch16(int64_t num_classes, bool pretrained, int64_t img_size)
    -> std::shared_ptr<ViTForImageClassification> {

    auto config = ViTConfig::base_patch16(img_size);
    auto model = std::make_shared<ViTForImageClassification>(config, num_classes);

    if (pretrained) {
        auto path = ViTModelHub::download_pretrained("vit_base_patch16_224");
        ViTModelHub::load_pretrained_weights(*model, path);
    }

    return model;
}

auto ViT_Base_Patch32(int64_t num_classes, bool pretrained, int64_t img_size)
    -> std::shared_ptr<ViTForImageClassification> {

    auto config = ViTConfig::base_patch32(img_size);
    auto model = std::make_shared<ViTForImageClassification>(config, num_classes);

    if (pretrained) {
        auto path = ViTModelHub::download_pretrained("vit_base_patch32_224");
        ViTModelHub::load_pretrained_weights(*model, path);
    }

    return model;
}

auto ViT_Large_Patch16(int64_t num_classes, bool pretrained, int64_t img_size)
    -> std::shared_ptr<ViTForImageClassification> {

    auto config = ViTConfig::large_patch16(img_size);
    auto model = std::make_shared<ViTForImageClassification>(config, num_classes);

    if (pretrained) {
        auto path = ViTModelHub::download_pretrained("vit_large_patch16_224");
        ViTModelHub::load_pretrained_weights(*model, path);
    }

    return model;
}

auto ViT_Large_Patch32(int64_t num_classes, bool pretrained, int64_t img_size)
    -> std::shared_ptr<ViTForImageClassification> {

    auto config = ViTConfig::large_patch32(img_size);
    auto model = std::make_shared<ViTForImageClassification>(config, num_classes);

    if (pretrained) {
        auto path = ViTModelHub::download_pretrained("vit_large_patch32_224");
        ViTModelHub::load_pretrained_weights(*model, path);
    }

    return model;
}

auto ViT_Huge_Patch14(int64_t num_classes, bool pretrained, int64_t img_size)
    -> std::shared_ptr<ViTForImageClassification> {

    auto config = ViTConfig::huge_patch14(img_size);
    auto model = std::make_shared<ViTForImageClassification>(config, num_classes);

    if (pretrained) {
        auto path = ViTModelHub::download_pretrained("vit_huge_patch14_224");
        ViTModelHub::load_pretrained_weights(*model, path);
    }

    return model;
}

auto ViT_Huge_Patch16(int64_t num_classes, bool pretrained, int64_t img_size)
    -> std::shared_ptr<ViTForImageClassification> {

    auto config = ViTConfig::huge_patch16(img_size);
    auto model = std::make_shared<ViTForImageClassification>(config, num_classes);

    if (pretrained) {
        auto path = ViTModelHub::download_pretrained("vit_huge_patch16_224");
        ViTModelHub::load_pretrained_weights(*model, path);
    }

    return model;
}

// ============================================================================
// ViTModelHub Implementation
// ============================================================================

auto ViTModelHub::download_pretrained(const std::string& model_name) -> std::string {
    // Determine cache directory
    std::string cache_path;
    const char* home = std::getenv("HOME");
    if (home) {
        cache_path = std::string(home) + "/.cache/tenzor/vit";
    } else {
        cache_path = "/tmp/tenzor_cache/vit";
    }

    // Create cache directory if it doesn't exist
    std::filesystem::create_directories(cache_path);

    // Model checkpoint path
    auto model_dir = cache_path + "/" + model_name;
    auto checkpoint_file = model_dir + "/pytorch_model.bin";

    // Check if already downloaded
    if (std::filesystem::exists(checkpoint_file)) {
        return checkpoint_file;
    }

    // In a real implementation, this would download from a model hub
    // For now, we'll throw an error and let the user manually download
    throw std::runtime_error(
        "Model '" + model_name + "' not found in cache. "
        "Please download the model from a model hub (e.g., Hugging Face) to: " + model_dir);
}

auto ViTModelHub::load_pretrained_weights(nn::Module& model,
                                         const std::string& checkpoint_path) -> void {
    // In a real implementation, this would:
    // 1. Load PyTorch checkpoint file
    // 2. Map parameter names (e.g., "vit.encoder.layer.0" -> "vit.encoder.layers.0")
    // 3. Load weights into model

    // For now, we'll throw a not-implemented error
    throw std::runtime_error(
        "Pretrained weight loading not yet implemented. "
        "This requires integration with PyTorch C++ API or custom checkpoint format. "
        "Checkpoint path: " + checkpoint_path);
}

} // namespace models
} // namespace tenzor
