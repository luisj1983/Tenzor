/**
 * @file vit.cpp
 * @brief Implementation of Vision Transformer model family
 */

#include "tenzor/models/vit.hpp"
#include "tenzor/models/hub.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/ops/transform.hpp"
#include <cmath>
#include <random>
#include <stdexcept>
#include <filesystem>
#include <mutex>

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
    DType dtype = pixel_values.tensor().dtype();
    Device device = pixel_values.tensor().device();

    // Get patch embeddings: [batch, num_patches, hidden_size]
    auto embeddings = patch_embeddings_->forward(pixel_values);

    // Get cls_token from parameters map (this gets the dtype-converted version)
    auto& cls_token = *parameters_["cls_token"];

    // Expand [CLS] token for batch: [1, 1, hidden_size] -> [batch, 1, hidden_size]
    // We must keep this in the autograd graph so the learnable cls_token_ parameter
    // actually receives gradients. cls_token is a registered leaf (grad_fn() == nullptr),
    // so a tensor-level repeat + Variable(...) leaf would disconnect it from backward.
    Variable cls_var = cls_token;

    // Ensure cls_token is on the same device and dtype as the input. Module::to()
    // normally keeps these in sync, so these branches are usually no-ops; when they
    // do fire we rewrap (device/dtype conversion is not part of the differentiable
    // path here, the parameter itself is the leaf that must receive gradients).
    if (cls_var.tensor().device() != device) {
        cls_var = Variable(cls_var.tensor().to(device), cls_var.requires_grad());
    }
    if (cls_var.tensor().dtype() != dtype) {
        cls_var = Variable(cls_var.tensor().to(dtype), cls_var.requires_grad());
    }

    // Expand the [CLS] token to batch size using the autograd-aware repeat so the
    // grad_fn chain back to cls_token_ is preserved.
    // [1, 1, hidden_size] -> [batch, 1, hidden_size]
    Variable cls_tokens = tenzor::repeat(cls_var, {batch_size, 1, 1});

    // Concatenate [CLS] token with patch embeddings using autograd cat
    // This maintains the gradient chain for proper backpropagation
    // Result: [batch, num_patches + 1, hidden_size]
    std::vector<Variable> to_concat = {cls_tokens, embeddings};
    embeddings = cat(to_concat, 1);

    // Get position_embeddings from parameters map (this gets the dtype-converted version)
    auto& pos_embeddings = *parameters_["position_embeddings"];

    // Ensure position embeddings are on the same device and dtype as input
    Variable pos_var = pos_embeddings;
    if (pos_var.tensor().device() != device) {
        Tensor pos_tensor = pos_var.tensor().to(device);
        pos_var = Variable(pos_tensor, pos_var.requires_grad());
    }
    if (pos_var.tensor().dtype() != dtype) {
        Tensor pos_tensor = pos_var.tensor().to(dtype);
        pos_var = Variable(pos_tensor, pos_var.requires_grad());
    }

    // Add position embeddings
    // Position embeddings are [1, seq_len, hidden_size], broadcast to [batch, seq_len, hidden_size]
    embeddings = embeddings + pos_var;

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
        // Extract [CLS] token (first token in sequence) directly via
        // autograd-aware slice + squeeze on the sequence dim. This replaces the
        // old O(batch^2 * seq_len) [batch, batch*seq_len] selection-matrix
        // matmul, preserves the grad_fn chain, and works for every dtype.
        // sequence_output: [batch, seq_len, hidden] -> [batch, hidden]
        auto cls_token = tenzor::squeeze(tenzor::slice(sequence_output, 1, 0, 1), 1);

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

    // Extract [CLS] token (first token) directly via autograd-aware slice +
    // squeeze on the sequence dim. This replaces the old O(batch^2 * seq_len)
    // [batch, batch*seq_len] selection-matrix matmul, preserves the grad_fn
    // chain, and works for every dtype.
    // sequence_output: [batch, seq_len, hidden] -> [batch, hidden]
    auto cls_output = tenzor::squeeze(tenzor::slice(sequence_output, 1, 0, 1), 1);

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

// ----------------------------------------------------------------------------
// ViT pretrained-model registry — HuggingFace timm + Google JAX-port URLs.
// We lazily register these with ModelHub on first download so the rest of
// the hub plumbing (libcurl fetch, SHA verification, on-disk caching,
// safetensors/.bin dispatch, key remapping) is reused as-is.
// ----------------------------------------------------------------------------

namespace {

struct ViTRegistry {
    std::once_flag flag;

    void register_once() {
        std::call_once(flag, []() {
            using MWI = ModelWeightInfo;

            // timm safetensors mirrors — preferred (no pickle, faster).
            // Field order: { name, url, sha256, size, description, safetensors_url }.
            const std::vector<MWI> entries = {
                MWI{ "vit_base_patch16_224",
                     "https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/resolve/main/pytorch_model.bin",
                     /*sha256=*/"",
                     /*size=*/0,
                     /*description=*/"ViT-Base/16, 224px (timm augreg2 IN21k→IN1k)",
                     /*safetensors_url=*/"https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/resolve/main/model.safetensors" },
                MWI{ "vit_base_patch32_224",
                     "https://huggingface.co/timm/vit_base_patch32_224.augreg_in21k_ft_in1k/resolve/main/pytorch_model.bin",
                     "", 0, "ViT-Base/32, 224px (timm augreg IN21k→IN1k)",
                     "https://huggingface.co/timm/vit_base_patch32_224.augreg_in21k_ft_in1k/resolve/main/model.safetensors" },
                MWI{ "vit_large_patch16_224",
                     "https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k_ft_in1k/resolve/main/pytorch_model.bin",
                     "", 0, "ViT-Large/16, 224px (timm augreg IN21k→IN1k)",
                     "https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k_ft_in1k/resolve/main/model.safetensors" },
                MWI{ "vit_large_patch32_224",
                     "https://huggingface.co/timm/vit_large_patch32_224.orig_in21k_ft_in1k/resolve/main/pytorch_model.bin",
                     "", 0, "ViT-Large/32, 224px (timm orig IN21k→IN1k)",
                     "https://huggingface.co/timm/vit_large_patch32_224.orig_in21k_ft_in1k/resolve/main/model.safetensors" },
                MWI{ "vit_huge_patch14_224",
                     "https://huggingface.co/timm/vit_huge_patch14_224.orig_in21k/resolve/main/pytorch_model.bin",
                     "", 0, "ViT-Huge/14, 224px (timm orig IN21k)",
                     "https://huggingface.co/timm/vit_huge_patch14_224.orig_in21k/resolve/main/model.safetensors" },
                MWI{ "vit_huge_patch16_224",
                     "https://huggingface.co/timm/vit_huge_patch16_224.orig_in21k/resolve/main/pytorch_model.bin",
                     "", 0, "ViT-Huge/16, 224px (timm orig IN21k)",
                     "https://huggingface.co/timm/vit_huge_patch16_224.orig_in21k/resolve/main/model.safetensors" },
            };
            ModelHub::register_models(entries);
        });
    }
};

ViTRegistry& vit_registry() {
    static ViTRegistry r;
    return r;
}

} // namespace

auto ViTModelHub::download_pretrained(const std::string& model_name) -> std::string {
    // Ensure the canonical ViT entries are visible to ModelHub. Registration
    // is idempotent and safe to call from any thread (std::call_once).
    vit_registry().register_once();

    // Delegate to ModelHub: handles cache lookup, libcurl fetch, retry,
    // SHA verification (when populated), and progress reporting. Prefer
    // the safetensors mirror when registered — faster and pickle-free.
    return ModelHub::download_pretrained_safetensors(
        model_name, /*prefer_safetensors=*/true);
}

auto ViTModelHub::load_pretrained_weights(nn::Module& model,
                                          const std::string& checkpoint_path) -> void {
    // ModelHub::load_pretrained_weights dispatches by file extension:
    //   .safetensors        -> SafeTensorsSerializer
    //   .pth / .pt / .bin   -> tenzor::io::load_torch_pickle (native C++ pickle reader)
    //   *                   -> Tenzor native format
    // It also applies the timm/torchvision → tenzor key remap (Sequential
    // `module_N` etc.). strict=false so partial loads (e.g. head pruned)
    // succeed cleanly with a warning.
    ModelHub::load_pretrained_weights(model, checkpoint_path, /*strict=*/false);
}

} // namespace models
} // namespace tenzor
