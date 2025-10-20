# Phase 9: Model Zoo & Pretrained Models - SPARC Specification

**Version:** 1.0
**Date:** 2025-10-17
**Estimated Effort:** 220 hours
**Status:** Draft

---

## Table of Contents

1. [SPARC Overview](#sparc-overview)
2. [Specification Phase](#specification-phase)
3. [Pseudocode Phase](#pseudocode-phase)
4. [Architecture Phase](#architecture-phase)
5. [Refinement Phase](#refinement-phase)
6. [Completion Phase](#completion-phase)
7. [Testing Requirements](#testing-requirements)
8. [Documentation Requirements](#documentation-requirements)
9. [Dependencies](#dependencies)
10. [Timeline and Priorities](#timeline-and-priorities)

---

## SPARC Overview

### Purpose

Phase 9 establishes Tenzor as a production-ready deep learning framework by implementing a comprehensive model zoo with pretrained weights. This phase delivers:

1. **Computer Vision Models**: ResNet family, VGG, AlexNet, GoogLeNet, EfficientNet, ViT, Swin, ConvNeXt, MobileNet, detection/segmentation models
2. **NLP Models**: BERT family, GPT-2/3, RoBERTa, ALBERT, T5, ELECTRA with Hugging Face weight compatibility
3. **Pretrained Weight Management**: Robust download, caching, verification, and loading system

### Success Criteria

- All models achieve parity with PyTorch/TensorFlow implementations (bit-accurate where possible)
- Pretrained weights load correctly and produce expected outputs on reference inputs
- 100% test coverage for model forward passes and gradient computation
- Python API matches PyTorch ergonomics: `model = tenzor.models.resnet50(pretrained=True)`
- Zero stubs, placeholders, or workarounds
- Performance within 10% of reference implementations

### Key Principles

1. **Accuracy First**: Bit-accurate implementations where feasible
2. **PyTorch Compatibility**: API design mirrors PyTorch for easy adoption
3. **Hugging Face Integration**: Direct loading of Hugging Face pretrained weights for NLP
4. **Production Quality**: Robust error handling, validation, and testing
5. **Performance**: Optimized implementations leveraging Phase 8 fused ops

---

## Specification Phase

### 1. Functional Requirements

#### 1.1 Computer Vision Models

**ResNet Family (20h)**
- ResNet-18, 34, 50, 101, 152 architectures
- ResNeXt-50-32x4d, ResNeXt-101-32x8d variants
- Wide ResNet-50-2, Wide ResNet-101-2
- ImageNet-1k pretrained weights for all variants
- Support for custom input sizes and output classes
- Batch normalization with momentum and epsilon parameters
- Optional final fully-connected layer for feature extraction

**VGG/AlexNet/GoogleNet (15h)**
- VGG-11, VGG-13, VGG-16, VGG-19 (with/without batch normalization)
- AlexNet original architecture
- GoogLeNet (Inception v1) with auxiliary classifiers
- ImageNet-1k pretrained weights
- Dropout configuration options
- Feature extraction modes

**Modern Architectures (30h)**
- EfficientNet-B0 through B7 (compound scaling)
- Vision Transformer (ViT): ViT-B/16, ViT-B/32, ViT-L/16, ViT-L/32
- Swin Transformer: Swin-T, Swin-S, Swin-B, Swin-L
- ConvNeXt: ConvNeXt-T, ConvNeXt-S, ConvNeXt-B, ConvNeXt-L
- MobileNetV2, MobileNetV3-Small, MobileNetV3-Large
- ImageNet-1k/21k pretrained weights where applicable
- Patch embedding for transformers
- Shifted window attention for Swin

**Detection & Segmentation (40h)**
- Faster R-CNN with ResNet-50/101 backbones
- Mask R-CNN with ResNet-50/101 backbones
- YOLOv3, YOLOv5s/m/l/x architectures
- U-Net for medical image segmentation
- DeepLabV3/V3+ with various backbones
- Region Proposal Networks (RPN)
- ROI pooling and ROI Align
- Feature Pyramid Networks (FPN)
- Non-Maximum Suppression (NMS)
- COCO pretrained weights for detection models

#### 1.2 NLP Models (HIGH PRIORITY)

**BERT Family (30h)**
- BertModel: Base transformer encoder
- BertForSequenceClassification: Sequence-level classification
- BertForTokenClassification: Token-level classification (NER, POS tagging)
- BertForQuestionAnswering: Span prediction for QA
- BertForMaskedLM: Masked language modeling
- BertForNextSentencePrediction: NSP task
- Support for BERT-base-uncased, BERT-large-uncased, BERT-base-cased, BERT-large-cased
- Direct loading of Hugging Face pretrained weights
- WordPiece tokenization integration
- Positional embeddings (learned, absolute)
- Segment embeddings for sentence pairs

**GPT Family (25h)**
- GPT2Model: Transformer decoder architecture
- GPT2LMHeadModel: Language modeling head
- GPT3Architecture: Scaled GPT-2 with configurable parameters
- Text generation utilities: greedy, beam search, top-k, top-p sampling
- Support for GPT-2 small/medium/large/xl
- Hugging Face weight loading for GPT-2
- Byte-pair encoding (BPE) tokenization
- Position embeddings (learned)
- Key-value caching for efficient generation

**Other Transformers (30h)**
- RoBERTa: Robustly optimized BERT
- ALBERT: A Lite BERT with parameter sharing
- T5: Text-to-Text Transfer Transformer (encoder-decoder)
- ELECTRA: Efficiently Learning an Encoder that Classifies Token Replacements Accurately
- Hugging Face weight compatibility for all models
- Model-specific tokenization (SentencePiece for T5, WordPiece for others)
- Relative position biases (T5)
- Factorized embeddings (ALBERT)
- Discriminator head (ELECTRA)

#### 1.3 Pretrained Weight Management (15h)

**Download System**
- HTTP/HTTPS URL downloads with resume capability
- Progress bars with speed and ETA
- Concurrent downloads with connection pooling
- Retry logic with exponential backoff
- Proxy support via environment variables
- Timeout configuration

**Caching System**
- XDG-compliant cache directory (`~/.cache/tenzor/hub/`)
- File locking for concurrent access safety
- Atomic file writes (download to temp, rename on success)
- Cache invalidation based on ETags/Last-Modified headers
- Manual cache clearing API
- Configurable cache size limits with LRU eviction

**Verification System**
- SHA256 checksum verification
- Manifest files with model metadata (architecture, input size, classes, etc.)
- Version tracking for weight updates
- Corrupted file detection and re-download

**Python API**
```python
# High-level model loading
model = tenzor.models.resnet50(pretrained=True, progress=True)

# Granular control
model = tenzor.models.resnet50(pretrained=False)
weights = tenzor.hub.load_weights('resnet50-imagenet1k', cache_dir='/custom/path')
model.load_state_dict(weights)

# Cache management
tenzor.hub.set_cache_dir('/custom/cache')
tenzor.hub.clear_cache()
tenzor.hub.list_cached_models()
```

### 2. Non-Functional Requirements

#### 2.1 Performance

- Model forward pass within 10% of PyTorch performance
- Weight loading < 5 seconds for models up to 500MB
- Memory-mapped weight loading for models > 1GB
- Multi-threaded weight deserialization
- Zero-copy weight transfers where possible

#### 2.2 Compatibility

- Support for PyTorch state dict format (`.pth`, `.pt`)
- Support for Hugging Face safetensors format (`.safetensors`)
- Backward compatibility with future weight format versions
- Cross-platform weight loading (Linux, macOS, Windows)

#### 2.3 Robustness

- Graceful handling of network failures
- Disk space checking before downloads
- Memory validation before weight loading
- Informative error messages with troubleshooting hints
- Model architecture validation against loaded weights

#### 2.4 Usability

- Zero configuration for default use cases
- Sensible defaults matching PyTorch behavior
- Clear documentation with runnable examples
- Type hints in Python API
- Progress feedback for long operations

---

## Pseudocode Phase

### 1. Model Base Classes

```cpp
// Base class for all vision models
class VisionModel : public nn::Module {
public:
    // Load pretrained weights from URL or local path
    virtual void load_pretrained(const std::string& weights_name,
                                 bool progress = true,
                                 const std::string& cache_dir = "");

    // Forward pass with optional feature extraction
    virtual Tensor forward(const Tensor& x, bool extract_features = false) = 0;

    // Get model-specific metadata
    virtual ModelMetadata metadata() const = 0;

protected:
    // Subclass hook for weight loading customization
    virtual void _load_state_dict(const StateDict& state_dict);

    // Subclass hook for architecture validation
    virtual bool _validate_architecture() const = 0;
};

// Base class for all NLP models
class NLPModel : public nn::Module {
public:
    // Load pretrained weights with tokenizer
    virtual void load_pretrained(const std::string& model_name,
                                 bool include_tokenizer = true,
                                 bool progress = true);

    // Forward pass with attention mask and token type IDs
    virtual ModelOutput forward(const Tensor& input_ids,
                               const Tensor& attention_mask = {},
                               const Tensor& token_type_ids = {}) = 0;

    // Get tokenizer for this model
    virtual std::shared_ptr<Tokenizer> tokenizer() const;

protected:
    std::shared_ptr<Tokenizer> tokenizer_;
};
```

### 2. Weight Management System

```cpp
class WeightManager {
public:
    // Download weights with progress tracking
    std::string download(const std::string& url,
                        const std::string& checksum,
                        bool progress = true,
                        int max_retries = 3);

    // Load weights from cache or download if missing
    StateDict load_weights(const std::string& model_name,
                          bool progress = true,
                          const std::string& cache_dir = "");

    // Cache management
    void set_cache_dir(const std::string& dir);
    std::string get_cache_dir() const;
    void clear_cache();
    std::vector<std::string> list_cached_models() const;

private:
    // HTTP download with resume support
    void download_file(const std::string& url,
                      const std::string& dest,
                      ProgressCallback callback);

    // Verify checksum
    bool verify_checksum(const std::string& file,
                        const std::string& expected_checksum);

    // Deserialize state dict from file
    StateDict deserialize_state_dict(const std::string& file,
                                    FileFormat format);

    std::string cache_dir_;
    std::unique_ptr<HTTPClient> http_client_;
};

// Model registry for pretrained weights
class ModelRegistry {
public:
    // Register a model with its weight URLs
    void register_model(const std::string& name,
                       const ModelSpec& spec);

    // Get model specification
    ModelSpec get_spec(const std::string& name) const;

    // List available models
    std::vector<std::string> list_models(const std::string& filter = "") const;

private:
    std::unordered_map<std::string, ModelSpec> registry_;
};

struct ModelSpec {
    std::string name;
    std::string architecture;
    std::map<std::string, WeightSpec> weights; // variant -> WeightSpec
    ModelMetadata metadata;
};

struct WeightSpec {
    std::string url;
    std::string checksum;
    size_t size_bytes;
    FileFormat format;
    std::string version;
};
```

### 3. ResNet Implementation

```cpp
class ResNet : public VisionModel {
public:
    ResNet(const ResNetConfig& config);

    Tensor forward(const Tensor& x, bool extract_features = false) override;

    static std::shared_ptr<ResNet> resnet18(bool pretrained = false,
                                           int num_classes = 1000,
                                           bool progress = true);
    static std::shared_ptr<ResNet> resnet50(bool pretrained = false,
                                           int num_classes = 1000,
                                           bool progress = true);
    // ... other variants

private:
    Tensor _make_layer(int planes, int blocks, int stride);

    nn::Conv2d conv1;
    nn::BatchNorm2d bn1;
    nn::ReLU relu;
    nn::MaxPool2d maxpool;
    nn::Sequential layer1, layer2, layer3, layer4;
    nn::AdaptiveAvgPool2d avgpool;
    nn::Linear fc;
};

class BasicBlock : public nn::Module {
public:
    BasicBlock(int inplanes, int planes, int stride = 1,
              nn::Sequential downsample = nullptr);

    Tensor forward(const Tensor& x) override;

    static constexpr int expansion = 1;

private:
    nn::Conv2d conv1, conv2;
    nn::BatchNorm2d bn1, bn2;
    nn::ReLU relu;
    nn::Sequential downsample;
};

class Bottleneck : public nn::Module {
public:
    Bottleneck(int inplanes, int planes, int stride = 1,
              nn::Sequential downsample = nullptr,
              int groups = 1, int base_width = 64);

    Tensor forward(const Tensor& x) override;

    static constexpr int expansion = 4;

private:
    nn::Conv2d conv1, conv2, conv3;
    nn::BatchNorm2d bn1, bn2, bn3;
    nn::ReLU relu;
    nn::Sequential downsample;
};
```

### 4. BERT Implementation

```cpp
class BertModel : public NLPModel {
public:
    BertModel(const BertConfig& config);

    BertOutput forward(const Tensor& input_ids,
                      const Tensor& attention_mask = {},
                      const Tensor& token_type_ids = {}) override;

    static std::shared_ptr<BertModel> from_pretrained(
        const std::string& model_name,
        bool progress = true);

private:
    BertEmbeddings embeddings;
    BertEncoder encoder;
    BertPooler pooler;
    BertConfig config_;
};

class BertEmbeddings : public nn::Module {
public:
    BertEmbeddings(const BertConfig& config);

    Tensor forward(const Tensor& input_ids,
                  const Tensor& token_type_ids = {},
                  const Tensor& position_ids = {});

private:
    nn::Embedding word_embeddings;
    nn::Embedding position_embeddings;
    nn::Embedding token_type_embeddings;
    nn::LayerNorm layer_norm;
    nn::Dropout dropout;
};

class BertEncoder : public nn::Module {
public:
    BertEncoder(const BertConfig& config);

    Tensor forward(const Tensor& hidden_states,
                  const Tensor& attention_mask = {},
                  bool output_attentions = false,
                  bool output_hidden_states = false);

private:
    std::vector<BertLayer> layers;
};

class BertLayer : public nn::Module {
public:
    BertLayer(const BertConfig& config);

    Tensor forward(const Tensor& hidden_states,
                  const Tensor& attention_mask = {});

private:
    BertAttention attention;
    BertIntermediate intermediate;
    BertOutput output;
};

class BertForSequenceClassification : public NLPModel {
public:
    BertForSequenceClassification(const BertConfig& config,
                                 int num_labels);

    SequenceClassifierOutput forward(const Tensor& input_ids,
                                    const Tensor& attention_mask = {},
                                    const Tensor& token_type_ids = {},
                                    const Tensor& labels = {}) override;

    static std::shared_ptr<BertForSequenceClassification> from_pretrained(
        const std::string& model_name,
        int num_labels,
        bool progress = true);

private:
    BertModel bert;
    nn::Dropout dropout;
    nn::Linear classifier;
    int num_labels_;
};
```

### 5. GPT-2 Implementation

```cpp
class GPT2Model : public NLPModel {
public:
    GPT2Model(const GPT2Config& config);

    GPT2Output forward(const Tensor& input_ids,
                      const Tensor& attention_mask = {},
                      const Tensor& past_key_values = {},
                      bool use_cache = false) override;

    static std::shared_ptr<GPT2Model> from_pretrained(
        const std::string& model_name,
        bool progress = true);

private:
    nn::Embedding wte; // word token embeddings
    nn::Embedding wpe; // position embeddings
    nn::Dropout drop;
    std::vector<GPT2Block> h; // transformer blocks
    nn::LayerNorm ln_f;
    GPT2Config config_;
};

class GPT2LMHeadModel : public NLPModel {
public:
    GPT2LMHeadModel(const GPT2Config& config);

    CausalLMOutput forward(const Tensor& input_ids,
                          const Tensor& attention_mask = {},
                          const Tensor& past_key_values = {},
                          const Tensor& labels = {},
                          bool use_cache = false) override;

    // Text generation methods
    Tensor generate(const Tensor& input_ids,
                   int max_length = 50,
                   const GenerationConfig& config = {});

    static std::shared_ptr<GPT2LMHeadModel> from_pretrained(
        const std::string& model_name,
        bool progress = true);

private:
    GPT2Model transformer;
    nn::Linear lm_head;

    // Generation strategies
    Tensor _greedy_search(const Tensor& input_ids, int max_length);
    Tensor _beam_search(const Tensor& input_ids, int max_length, int num_beams);
    Tensor _sample(const Tensor& input_ids, int max_length,
                  float temperature, int top_k, float top_p);
};

struct GenerationConfig {
    int max_length = 50;
    int min_length = 0;
    bool do_sample = false;
    float temperature = 1.0;
    int top_k = 50;
    float top_p = 1.0;
    float repetition_penalty = 1.0;
    int num_beams = 1;
    int num_return_sequences = 1;
};
```

### 6. Vision Transformer Implementation

```cpp
class VisionTransformer : public VisionModel {
public:
    VisionTransformer(const ViTConfig& config);

    Tensor forward(const Tensor& x, bool extract_features = false) override;

    static std::shared_ptr<VisionTransformer> vit_b_16(
        bool pretrained = false,
        int num_classes = 1000,
        bool progress = true);

private:
    PatchEmbedding patch_embed;
    nn::Parameter class_token;
    nn::Parameter position_embeddings;
    nn::Dropout dropout;
    TransformerEncoder encoder;
    nn::LayerNorm norm;
    nn::Linear head;
};

class PatchEmbedding : public nn::Module {
public:
    PatchEmbedding(int img_size, int patch_size, int in_channels, int embed_dim);

    Tensor forward(const Tensor& x);

private:
    nn::Conv2d proj;
    int num_patches_;
};
```

### 7. YOLO Implementation

```cpp
class YOLOv5 : public VisionModel {
public:
    YOLOv5(const YOLOConfig& config);

    DetectionOutput forward(const Tensor& x,
                           float conf_threshold = 0.25,
                           float iou_threshold = 0.45) override;

    static std::shared_ptr<YOLOv5> yolov5s(bool pretrained = false,
                                          int num_classes = 80,
                                          bool progress = true);

private:
    YOLOBackbone backbone;
    YOLONeck neck; // FPN + PAN
    YOLOHead head;

    // Post-processing
    Tensor _non_max_suppression(const Tensor& predictions,
                                float conf_threshold,
                                float iou_threshold);
};

struct DetectionOutput {
    Tensor boxes;      // [N, 4] (x1, y1, x2, y2)
    Tensor scores;     // [N]
    Tensor class_ids;  // [N]
};
```

---

## Architecture Phase

### 1. Directory Structure

```
include/tenzor/models/
├── vision/
│   ├── resnet.hpp
│   ├── vgg.hpp
│   ├── alexnet.hpp
│   ├── googlenet.hpp
│   ├── efficientnet.hpp
│   ├── vision_transformer.hpp
│   ├── swin_transformer.hpp
│   ├── convnext.hpp
│   ├── mobilenet.hpp
│   ├── detection/
│   │   ├── faster_rcnn.hpp
│   │   ├── mask_rcnn.hpp
│   │   ├── yolo.hpp
│   │   └── rpn.hpp
│   ├── segmentation/
│   │   ├── unet.hpp
│   │   └── deeplab.hpp
│   └── common/
│       ├── vision_model.hpp
│       ├── patch_embedding.hpp
│       └── fpn.hpp
├── nlp/
│   ├── bert.hpp
│   ├── gpt2.hpp
│   ├── roberta.hpp
│   ├── albert.hpp
│   ├── t5.hpp
│   ├── electra.hpp
│   └── common/
│       ├── nlp_model.hpp
│       ├── transformer_layer.hpp
│       └── embeddings.hpp
├── hub/
│   ├── weight_manager.hpp
│   ├── model_registry.hpp
│   ├── download.hpp
│   └── cache.hpp
└── models.hpp (main header)

src/models/
├── vision/
│   ├── resnet.cpp
│   ├── vgg.cpp
│   ├── alexnet.cpp
│   ├── googlenet.cpp
│   ├── efficientnet.cpp
│   ├── vision_transformer.cpp
│   ├── swin_transformer.cpp
│   ├── convnext.cpp
│   ├── mobilenet.cpp
│   ├── detection/
│   │   ├── faster_rcnn.cpp
│   │   ├── mask_rcnn.cpp
│   │   ├── yolo.cpp
│   │   ├── rpn.cpp
│   │   └── roi_ops.cpp
│   ├── segmentation/
│   │   ├── unet.cpp
│   │   └── deeplab.cpp
│   └── common/
│       ├── vision_model.cpp
│       ├── patch_embedding.cpp
│       └── fpn.cpp
├── nlp/
│   ├── bert.cpp
│   ├── gpt2.cpp
│   ├── roberta.cpp
│   ├── albert.cpp
│   ├── t5.cpp
│   ├── electra.cpp
│   └── common/
│       ├── nlp_model.cpp
│       ├── transformer_layer.cpp
│       └── embeddings.cpp
├── hub/
│   ├── weight_manager.cpp
│   ├── model_registry.cpp
│   ├── download.cpp
│   ├── cache.cpp
│   └── registry_data.cpp (hardcoded model URLs)
└── CMakeLists.txt

tests/unit/models/
├── vision/
│   ├── test_resnet.cpp
│   ├── test_vgg.cpp
│   ├── test_vision_transformer.cpp
│   ├── test_efficientnet.cpp
│   └── test_yolo.cpp
├── nlp/
│   ├── test_bert.cpp
│   ├── test_gpt2.cpp
│   ├── test_roberta.cpp
│   └── test_t5.cpp
├── hub/
│   ├── test_weight_manager.cpp
│   ├── test_download.cpp
│   └── test_cache.cpp
└── test_model_zoo.cpp

tests/integration/models/
├── test_pretrained_loading.cpp
├── test_inference_accuracy.cpp
├── test_bert_huggingface.cpp
└── test_generation.cpp

python/models/
├── __init__.py
├── vision.py
├── nlp.py
└── hub.py

examples/models/
├── resnet_imagenet_inference.cpp
├── bert_sequence_classification.cpp
├── gpt2_text_generation.cpp
├── yolo_object_detection.cpp
└── vit_transfer_learning.cpp
```

### 2. Component Architecture

#### 2.1 Model Base Classes Hierarchy

```
nn::Module
├── VisionModel
│   ├── ResNet
│   ├── VGG
│   ├── AlexNet
│   ├── GoogLeNet
│   ├── EfficientNet
│   ├── VisionTransformer
│   ├── SwinTransformer
│   ├── ConvNeXt
│   ├── MobileNet
│   ├── DetectionModel
│   │   ├── FasterRCNN
│   │   ├── MaskRCNN
│   │   └── YOLO
│   └── SegmentationModel
│       ├── UNet
│       └── DeepLab
└── NLPModel
    ├── BertModel
    ├── BertForSequenceClassification
    ├── BertForTokenClassification
    ├── BertForQuestionAnswering
    ├── GPT2Model
    ├── GPT2LMHeadModel
    ├── RoBERTaModel
    ├── ALBERTModel
    ├── T5Model
    └── ELECTRAModel
```

#### 2.2 Weight Management Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Python API Layer                      │
│  tenzor.models.resnet50(pretrained=True)                │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  Model Registry                          │
│  - Model specifications (URLs, checksums, metadata)     │
│  - Model factory functions                              │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  Weight Manager                          │
│  - Load/download coordination                           │
│  - Format detection and dispatch                        │
└──────┬──────────────────────┬──────────────────────────┘
       │                      │
┌──────▼──────┐      ┌────────▼────────┐
│   Download  │      │      Cache       │
│   Manager   │      │     Manager      │
│             │      │                  │
│ - HTTP      │      │ - XDG dirs       │
│ - Progress  │      │ - File locking   │
│ - Retry     │      │ - LRU eviction   │
│ - Resume    │      │ - Atomic writes  │
└──────┬──────┘      └────────┬────────┘
       │                      │
       └──────────┬───────────┘
                  │
┌─────────────────▼────────────────────────────────────────┐
│              State Dict Deserializer                      │
│  - PyTorch format (.pth, .pt) via LibTorch API           │
│  - Safetensors format (.safetensors) via safetensors-cpp│
│  - Checksum verification                                 │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│                 Model Instance                            │
│  - load_state_dict() validates and loads weights         │
│  - Architecture compatibility checking                   │
└──────────────────────────────────────────────────────────┘
```

#### 2.3 Model Forward Pass Pipeline

```
Input Tensor
     │
     ▼
[Preprocessing] (normalization, resize if needed)
     │
     ▼
[Model Forward Pass]
     │
     ├─ Vision Models:
     │  └─ Conv/Pooling → ResBlocks → Global Pool → FC
     │
     └─ NLP Models:
        └─ Embeddings → Transformer Layers → Task Head
     │
     ▼
[Postprocessing] (softmax, argmax, NMS for detection)
     │
     ▼
Output (predictions, logits, features, etc.)
```

### 3. Data Flow Diagrams

#### 3.1 Pretrained Weight Loading Flow

```
User calls: model = tenzor.models.bert_base_uncased(pretrained=True)
     │
     ▼
Model constructor creates architecture
     │
     ▼
load_pretrained() method invoked
     │
     ├─ Look up "bert-base-uncased" in ModelRegistry
     │  └─ Get URL, checksum, format
     │
     ├─ Check cache: ~/.cache/tenzor/hub/bert-base-uncased-*.safetensors
     │  │
     │  ├─ Cache hit:
     │  │  └─ Verify checksum → Load from cache
     │  │
     │  └─ Cache miss:
     │     ├─ Download from URL with progress bar
     │     ├─ Verify checksum
     │     ├─ Atomic write to cache
     │     └─ Load from cache
     │
     ├─ Deserialize StateDict based on format
     │
     ├─ Validate architecture compatibility
     │  └─ Check layer names, shapes, dtypes
     │
     └─ Load weights into model
        └─ Copy tensors to appropriate devices
     │
     ▼
Return initialized model
```

#### 3.2 BERT Inference Flow

```
Input: "Hello [MASK] world"
     │
     ▼
Tokenizer: tokenize → ["[CLS]", "hello", "[MASK]", "world", "[SEP]"]
     │
     ▼
Convert to IDs: [101, 7592, 103, 2088, 102]
     │
     ▼
Create attention mask: [1, 1, 1, 1, 1]
     │
     ▼
BertEmbeddings:
  - Word embeddings: [5, 768]
  - Position embeddings: [5, 768]
  - Token type embeddings: [5, 768]
  - Sum and normalize: [5, 768]
     │
     ▼
BertEncoder (12 layers):
  For each layer:
    - Multi-head self-attention with mask
    - Add & Norm
    - Feed-forward network
    - Add & Norm
     │
     ▼
Output: [5, 768] hidden states
     │
     ▼
Task-specific head (e.g., masked LM):
  - Linear projection: [5, 30522] (vocab size)
  - Predict token at position 2 (mask)
     │
     ▼
Output: "beautiful" (predicted token)
```

### 4. Technology Stack

#### 4.1 Core Dependencies

- **LibTorch C++ API**: For loading PyTorch `.pth` files
- **safetensors-cpp**: For loading Hugging Face `.safetensors` files
- **libcurl**: HTTP/HTTPS downloads
- **OpenSSL**: SHA256 checksum computation
- **nlohmann/json**: Manifest file parsing
- **spdlog**: Logging
- **indicators**: Progress bars (header-only)

#### 4.2 Optional Dependencies

- **cuDNN/cuBLAS**: Accelerated vision model inference on CUDA
- **oneDNN**: Optimized CPU inference for Intel architectures
- **SentencePiece**: Tokenization for T5 and other models
- **Hugging Face Tokenizers**: Fast tokenization library

#### 4.3 Build System Integration

```cmake
# Find LibTorch
find_package(Torch REQUIRED)

# Find curl
find_package(CURL REQUIRED)

# Find OpenSSL
find_package(OpenSSL REQUIRED)

# Add safetensors-cpp as submodule or FetchContent
FetchContent_Declare(
  safetensors
  GIT_REPOSITORY https://github.com/huggingface/safetensors.git
  GIT_TAG main
)

# Add nlohmann/json
find_package(nlohmann_json REQUIRED)

# Models library
add_library(tenzor_models
  src/models/vision/resnet.cpp
  src/models/nlp/bert.cpp
  src/models/hub/weight_manager.cpp
  # ... more sources
)

target_link_libraries(tenzor_models
  PUBLIC
    tenzor_core
    tenzor_nn
    torch
    ${CURL_LIBRARIES}
    OpenSSL::Crypto
    nlohmann_json::nlohmann_json
)
```

### 5. Configuration and Extensibility

#### 5.1 Model Configuration Objects

```cpp
struct BertConfig {
    int vocab_size = 30522;
    int hidden_size = 768;
    int num_hidden_layers = 12;
    int num_attention_heads = 12;
    int intermediate_size = 3072;
    std::string hidden_act = "gelu";
    float hidden_dropout_prob = 0.1;
    float attention_probs_dropout_prob = 0.1;
    int max_position_embeddings = 512;
    int type_vocab_size = 2;
    float layer_norm_eps = 1e-12;

    static BertConfig bert_base_uncased();
    static BertConfig bert_large_uncased();
    static BertConfig from_json(const std::string& json_path);
    nlohmann::json to_json() const;
};

struct ResNetConfig {
    std::vector<int> layers; // blocks per layer
    int num_classes = 1000;
    bool zero_init_residual = false;
    int groups = 1;
    int width_per_group = 64;
    bool replace_stride_with_dilation = false;
    std::string norm_layer = "batch_norm";

    static ResNetConfig resnet50();
    static ResNetConfig resnext50_32x4d();
};
```

#### 5.2 Plugin System for Custom Models

```cpp
// Users can register custom models
class ModelFactory {
public:
    using CreateFn = std::function<std::shared_ptr<nn::Module>(const nlohmann::json&)>;

    static void register_model(const std::string& name, CreateFn create_fn);
    static std::shared_ptr<nn::Module> create(const std::string& name,
                                             const nlohmann::json& config);
};

// Registration example
REGISTER_MODEL("custom-bert", [](const nlohmann::json& config) {
    return std::make_shared<CustomBertModel>(BertConfig::from_json(config));
});
```

---

## Refinement Phase

### 1. Implementation Priorities

#### Priority 1: Foundation (Week 1-2, 40h)

**Goal**: Establish weight management infrastructure and one reference model (ResNet-50)

1. **Weight Manager** (15h)
   - Implement download manager with libcurl
   - Implement cache manager with XDG directory support
   - SHA256 checksum verification
   - Progress bars with indicators library
   - Unit tests for all edge cases

2. **Model Registry** (5h)
   - Define ModelSpec and WeightSpec structures
   - Hardcode initial registry with ResNet-50, BERT-base
   - Implement lookup and listing functions

3. **State Dict Deserialization** (10h)
   - Integrate LibTorch for `.pth` loading
   - Integrate safetensors-cpp for `.safetensors` loading
   - Format auto-detection
   - Validation utilities

4. **ResNet-50** (10h)
   - Implement BasicBlock and Bottleneck
   - Implement ResNet class with all configurations
   - Load ImageNet-1k pretrained weights
   - Validation on ImageNet samples (top-1/top-5 accuracy)

#### Priority 2: BERT Family (Week 3-4, 45h)

**Goal**: Complete BERT implementation with Hugging Face compatibility

1. **BERT Core** (20h)
   - BertEmbeddings with all embedding types
   - BertLayer with multi-head attention
   - BertEncoder with 12/24 layers
   - BertPooler
   - Load Hugging Face weights for bert-base-uncased
   - Validation on GLUE benchmark samples

2. **BERT Task Heads** (15h)
   - BertForSequenceClassification
   - BertForTokenClassification
   - BertForQuestionAnswering
   - BertForMaskedLM
   - Fine-tuning API with frozen/unfrozen layers

3. **Tokenization** (10h)
   - WordPiece tokenizer implementation or integration
   - Load vocabulary from Hugging Face
   - Special token handling ([CLS], [SEP], [MASK], [PAD])
   - Batch encoding utilities

#### Priority 3: Additional Vision Models (Week 5-6, 35h)

1. **VGG Family** (8h)
   - VGG-16, VGG-19 with batch norm variants
   - Load ImageNet weights

2. **EfficientNet** (12h)
   - Implement compound scaling
   - MBConv blocks with squeeze-excitation
   - EfficientNet-B0 through B4 (B5-B7 optional)
   - Load ImageNet weights

3. **Vision Transformer** (15h)
   - Patch embedding layer
   - Transformer encoder with positional embeddings
   - Classification head
   - ViT-B/16, ViT-L/16
   - Load ImageNet-21k pretrained weights

#### Priority 4: GPT-2 and Generation (Week 7-8, 35h)

1. **GPT-2 Core** (15h)
   - GPT2Block with causal attention
   - GPT2Model with position embeddings
   - Load Hugging Face GPT-2 weights (small, medium)
   - Validation on perplexity benchmarks

2. **GPT-2 Generation** (15h)
   - Greedy decoding
   - Beam search (beam size 1-10)
   - Top-k sampling
   - Top-p (nucleus) sampling
   - Temperature scaling
   - Repetition penalty

3. **BPE Tokenization** (5h)
   - Byte-pair encoding implementation or integration
   - Load GPT-2 vocabulary and merges
   - Encode/decode utilities

#### Priority 5: Other Transformers (Week 9, 30h)

1. **RoBERTa** (8h)
   - Subclass BERT with dynamic masking
   - Load Hugging Face RoBERTa weights

2. **ALBERT** (10h)
   - Factorized embeddings
   - Cross-layer parameter sharing
   - Load Hugging Face ALBERT weights

3. **T5** (12h)
   - Encoder-decoder architecture
   - Relative position biases
   - SentencePiece tokenization
   - Load Hugging Face T5-small weights
   - Text-to-text format utilities

#### Priority 6: Modern CV Architectures (Week 10-11, 35h)

1. **Swin Transformer** (15h)
   - Shifted window attention
   - Patch merging
   - Swin-T, Swin-S
   - Load ImageNet weights

2. **ConvNeXt** (10h)
   - Modernized ResNet blocks
   - ConvNeXt-T, ConvNeXt-S
   - Load ImageNet weights

3. **MobileNet** (10h)
   - Depthwise separable convolutions
   - Inverted residuals
   - MobileNetV2, MobileNetV3
   - Load ImageNet weights

#### Priority 7: Detection & Segmentation (Week 12-14, 60h)

1. **Foundational Components** (15h)
   - Region Proposal Network (RPN)
   - ROI pooling and ROI Align
   - Feature Pyramid Network (FPN)
   - Non-Maximum Suppression (NMS)
   - Bounding box utilities (IoU, encoding/decoding)

2. **Faster R-CNN** (20h)
   - Two-stage detection pipeline
   - ResNet-50 backbone with FPN
   - RPN for proposal generation
   - ROI head for classification and box regression
   - Load COCO pretrained weights
   - Validation on COCO samples (mAP)

3. **YOLO** (15h)
   - Single-stage detection
   - YOLOv5s architecture (backbone, neck, head)
   - Grid-based predictions
   - NMS post-processing
   - Load COCO pretrained weights

4. **U-Net** (10h)
   - Encoder-decoder with skip connections
   - Segmentation head
   - Medical imaging pretrained weights (optional)

### 2. Testing Strategy

#### 2.1 Unit Tests

**Model Architecture Tests**
- Each model class has dedicated test file
- Test forward pass with random inputs (check output shapes)
- Test backward pass with gradient checking
- Test model.train() and model.eval() mode switching
- Test layer freezing/unfreezing
- Test state dict save/load

**Weight Loading Tests**
- Test loading from local file
- Test loading from cache
- Test checksum verification (pass and fail cases)
- Test corrupt file handling
- Test architecture mismatch detection

**Download Tests**
- Mock HTTP server for controlled testing
- Test successful download
- Test resume capability after partial download
- Test retry on failure
- Test progress callback
- Test timeout handling

**Cache Tests**
- Test cache hit/miss logic
- Test atomic file writes
- Test concurrent access (file locking)
- Test LRU eviction
- Test cache clearing

#### 2.2 Integration Tests

**Pretrained Weight Loading**
- Download real pretrained weights (ResNet-50, BERT-base) in CI
- Verify forward pass produces expected outputs on reference inputs
- Compare with PyTorch outputs (tolerance: 1e-5 for FP32, 1e-3 for FP16)

**Inference Accuracy**
- ResNet-50 on ImageNet validation set (100 images): achieve 76% top-1 accuracy
- BERT on GLUE tasks (SST-2, MNLI subsets): match Hugging Face accuracy
- GPT-2 perplexity on WikiText-2: within 1% of reference

**Hugging Face Compatibility**
- Load bert-base-uncased from Hugging Face Hub
- Load roberta-base from Hugging Face Hub
- Load gpt2 from Hugging Face Hub
- Verify output parity with transformers library

**Text Generation**
- GPT-2 generates coherent text (qualitative check)
- Beam search produces diverse outputs
- Sampling strategies produce expected distributions

#### 2.3 Performance Tests

**Inference Latency**
- ResNet-50 forward pass: < 50ms on GPU, < 200ms on CPU (batch size 1)
- BERT-base forward pass: < 100ms on GPU, < 500ms on CPU (sequence length 128)
- GPT-2 generation (50 tokens): < 2s on GPU, < 10s on CPU

**Memory Usage**
- ResNet-50 inference: < 500MB GPU memory
- BERT-base inference: < 1GB GPU memory
- Weight loading: < 5s for 500MB model

**Throughput**
- ResNet-50: > 100 images/s on GPU (batch size 32)
- BERT-base: > 50 sequences/s on GPU (batch size 32, seq len 128)

#### 2.4 Regression Tests

- Golden output tests: save outputs for reference inputs, compare on every commit
- Checksum tests: verify pretrained weights haven't changed unexpectedly
- API compatibility tests: ensure public API remains stable

### 3. Optimization Opportunities

**Inference Optimizations**
- Use Phase 8 fused operations (fused_attention, fused_mlp)
- Kernel fusion for common patterns (conv-bn-relu)
- Memory-efficient attention for long sequences (Flash Attention)
- KV cache for autoregressive generation
- Int8 quantization for deployment (leverage Phase 8.2 quantization)

**Weight Loading Optimizations**
- Memory-mapped weight loading for large models (> 1GB)
- Parallel deserialization (multi-threaded tensor creation)
- Zero-copy transfers when possible
- Lazy weight loading (load on first forward pass)

**Model Optimizations**
- TorchScript compilation for production deployment
- ONNX export for cross-framework compatibility
- Model pruning and distillation utilities
- Mixed-precision training/inference (FP16, BF16)

### 4. Error Handling

**Download Failures**
- Network errors: retry with exponential backoff
- Disk full: clear error message, suggest cache directory change
- Invalid URL: immediate failure with suggestion to check registry

**Weight Loading Failures**
- Checksum mismatch: delete cached file, retry download
- Format error: suggest checking file format, provide supported formats
- Architecture mismatch: detailed error showing missing/extra/mismatched layers

**Model Errors**
- Invalid input shapes: clear error with expected vs. actual shapes
- Missing tokenizer: prompt to download or provide tokenizer path
- Out of memory: suggest reducing batch size or sequence length

---

## Completion Phase

### 1. Integration with Existing Codebase

#### 1.1 Dependencies on Prior Phases

**Phase 7: RNN & Transformer Layers**
- BERT, GPT-2, T5 rely on TransformerEncoderLayer/TransformerDecoderLayer
- Ensure multi-head attention supports causal masking (GPT-2)
- Ensure layer normalization supports eps parameter (BERT, T5)
- Ensure positional encoding supports learned embeddings (BERT, GPT-2)

**Phase 8.1: Fused Operations**
- Vision models leverage fused_conv_bn_relu
- Transformers leverage fused_attention and fused_mlp
- Ensure fused ops support both training and inference modes

**Phase 8.2: Quantization**
- Detection models benefit from int8 quantization for deployment
- Ensure quantization preserves accuracy (within 1% of FP32)

**Python Bindings**
- All models exposed via Python API
- Type hints for all public methods
- Docstrings with parameter descriptions and examples

#### 1.2 CMake Integration

```cmake
# src/models/CMakeLists.txt
add_library(tenzor_models
  # Vision models
  vision/resnet.cpp
  vision/vgg.cpp
  vision/efficientnet.cpp
  vision/vision_transformer.cpp
  vision/swin_transformer.cpp
  vision/convnext.cpp
  vision/mobilenet.cpp
  vision/detection/faster_rcnn.cpp
  vision/detection/yolo.cpp
  vision/segmentation/unet.cpp

  # NLP models
  nlp/bert.cpp
  nlp/gpt2.cpp
  nlp/roberta.cpp
  nlp/albert.cpp
  nlp/t5.cpp

  # Hub
  hub/weight_manager.cpp
  hub/model_registry.cpp
  hub/download.cpp
  hub/cache.cpp
  hub/registry_data.cpp
)

target_link_libraries(tenzor_models
  PUBLIC
    tenzor_core
    tenzor_nn
    torch
    ${CURL_LIBRARIES}
    OpenSSL::Crypto
    nlohmann_json::nlohmann_json
)

target_include_directories(tenzor_models
  PUBLIC
    ${CMAKE_SOURCE_DIR}/include
  PRIVATE
    ${CURL_INCLUDE_DIRS}
)
```

#### 1.3 Python Bindings

```cpp
// python/models.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tenzor/models/models.hpp>

namespace py = pybind11;

PYBIND11_MODULE(models, m) {
    m.doc() = "Tenzor Model Zoo";

    // Vision models
    py::module vision = m.def_submodule("vision", "Computer vision models");

    vision.def("resnet18", &tenzor::models::resnet18,
               py::arg("pretrained") = false,
               py::arg("num_classes") = 1000,
               py::arg("progress") = true,
               "ResNet-18 model");

    vision.def("resnet50", &tenzor::models::resnet50,
               py::arg("pretrained") = false,
               py::arg("num_classes") = 1000,
               py::arg("progress") = true,
               "ResNet-50 model");

    // ... more vision models

    // NLP models
    py::module nlp = m.def_submodule("nlp", "Natural language processing models");

    py::class_<tenzor::models::BertModel, std::shared_ptr<tenzor::models::BertModel>>(nlp, "BertModel")
        .def(py::init<const tenzor::models::BertConfig&>())
        .def("forward", &tenzor::models::BertModel::forward,
             py::arg("input_ids"),
             py::arg("attention_mask") = py::none(),
             py::arg("token_type_ids") = py::none())
        .def_static("from_pretrained", &tenzor::models::BertModel::from_pretrained,
                    py::arg("model_name"),
                    py::arg("progress") = true);

    // ... more NLP models

    // Hub
    py::module hub = m.def_submodule("hub", "Pretrained weight management");

    hub.def("set_cache_dir", &tenzor::hub::set_cache_dir);
    hub.def("get_cache_dir", &tenzor::hub::get_cache_dir);
    hub.def("clear_cache", &tenzor::hub::clear_cache);
    hub.def("list_cached_models", &tenzor::hub::list_cached_models);
}
```

```python
# python/tenzor/models/__init__.py
"""Tenzor Model Zoo and Pretrained Weights"""

from .vision import (
    resnet18, resnet50, resnet101,
    vgg16, vgg19,
    efficientnet_b0, efficientnet_b4,
    vit_b_16, vit_l_16,
    swin_t, swin_s,
)

from .nlp import (
    BertModel,
    BertForSequenceClassification,
    BertForTokenClassification,
    GPT2Model,
    GPT2LMHeadModel,
)

from .hub import (
    set_cache_dir,
    get_cache_dir,
    clear_cache,
    list_cached_models,
)

__all__ = [
    # Vision
    'resnet18', 'resnet50', 'resnet101',
    'vgg16', 'vgg19',
    'efficientnet_b0', 'efficientnet_b4',
    'vit_b_16', 'vit_l_16',
    'swin_t', 'swin_s',

    # NLP
    'BertModel',
    'BertForSequenceClassification',
    'BertForTokenClassification',
    'GPT2Model',
    'GPT2LMHeadModel',

    # Hub
    'set_cache_dir',
    'get_cache_dir',
    'clear_cache',
    'list_cached_models',
]
```

### 2. Documentation

#### 2.1 API Documentation

**Sphinx Documentation Structure**
```
docs/api/models/
├── index.rst
├── vision.rst
│   ├── resnet.rst
│   ├── vgg.rst
│   ├── efficientnet.rst
│   ├── vision_transformer.rst
│   └── detection.rst
├── nlp.rst
│   ├── bert.rst
│   ├── gpt2.rst
│   ├── roberta.rst
│   └── t5.rst
└── hub.rst
```

**Example Documentation (BERT)**

```markdown
# BERT

BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model pretrained on masked language modeling and next sentence prediction tasks.

## Models

### BertModel

Base BERT model without task-specific head.

**C++ API:**
```cpp
#include <tenzor/models/nlp/bert.hpp>

auto config = tenzor::models::BertConfig::bert_base_uncased();
auto model = std::make_shared<tenzor::models::BertModel>(config);
model->load_pretrained("bert-base-uncased");

auto output = model->forward(input_ids, attention_mask, token_type_ids);
// output.last_hidden_state: [batch_size, seq_length, 768]
// output.pooler_output: [batch_size, 768]
```

**Python API:**
```python
import tenzor.models.nlp as nlp

model = nlp.BertModel.from_pretrained("bert-base-uncased")
output = model(input_ids, attention_mask, token_type_ids)
# output.last_hidden_state: [batch_size, seq_length, 768]
# output.pooler_output: [batch_size, 768]
```

**Parameters:**
- `input_ids`: Tensor of shape [batch_size, seq_length] with token IDs
- `attention_mask`: Optional tensor of shape [batch_size, seq_length] (1 for real tokens, 0 for padding)
- `token_type_ids`: Optional tensor of shape [batch_size, seq_length] (0 for sentence A, 1 for sentence B)

**Returns:**
- `last_hidden_state`: Tensor of shape [batch_size, seq_length, hidden_size]
- `pooler_output`: Tensor of shape [batch_size, hidden_size] (CLS token representation)

### BertForSequenceClassification

BERT model with sequence classification head.

[Similar documentation pattern...]

## Pretrained Weights

Available pretrained models:
- `bert-base-uncased`: 12-layer, 768-hidden, 12-heads, 110M parameters
- `bert-large-uncased`: 24-layer, 1024-hidden, 16-heads, 340M parameters
- `bert-base-cased`: 12-layer, 768-hidden, 12-heads (preserves case)
- `bert-large-cased`: 24-layer, 1024-hidden, 16-heads (preserves case)

Weights are automatically downloaded from Hugging Face Hub and cached locally.

## Example: Sequence Classification

[Complete working example with code...]
```

#### 2.2 Usage Examples

**examples/models/resnet_imagenet_inference.cpp**
```cpp
#include <tenzor/tenzor.hpp>
#include <tenzor/models/vision/resnet.hpp>

int main() {
    // Load pretrained ResNet-50
    auto model = tenzor::models::resnet50(true, 1000, true);
    model->eval();

    // Load and preprocess image
    auto image = tenzor::load_image("cat.jpg");
    image = tenzor::resize(image, {224, 224});
    image = tenzor::normalize(image, {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
    image = image.unsqueeze(0); // Add batch dimension

    // Inference
    auto output = model->forward(image);
    auto probs = tenzor::softmax(output, 1);
    auto top5 = tenzor::topk(probs, 5);

    // Print results
    std::cout << "Top 5 predictions:\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "  Class " << top5.indices[i].item<int>()
                  << ": " << top5.values[i].item<float>() << "\n";
    }

    return 0;
}
```

**examples/models/bert_sequence_classification.py**
```python
import tenzor
import tenzor.models.nlp as nlp

# Load pretrained BERT for sequence classification
model = nlp.BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
model.eval()

# Tokenize input
tokenizer = model.tokenizer()
inputs = tokenizer("This movie is great!", return_tensors="tenzor")

# Inference
with tenzor.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = logits.argmax(-1)

print(f"Sentiment: {'positive' if prediction == 1 else 'negative'}")
```

#### 2.3 Model Cards

Each pretrained model has a model card documenting:
- Architecture details
- Training data and procedure
- Evaluation metrics
- Intended use and limitations
- Bias and fairness considerations
- Citation information

### 3. Deployment Considerations

#### 3.1 Production Checklist

- [ ] All models pass accuracy validation tests
- [ ] Performance benchmarks meet targets
- [ ] Documentation complete and reviewed
- [ ] Python type stubs generated
- [ ] Example code tested and working
- [ ] CI/CD pipeline includes model tests
- [ ] Pretrained weights hosted reliably (CDN, mirrors)
- [ ] License compliance verified (model weights, code)

#### 3.2 Versioning Strategy

**Model Versions:**
- Semantic versioning for model zoo: `v1.0.0`, `v1.1.0`, etc.
- Pretrained weight versions: `resnet50-imagenet1k-v1`, `resnet50-imagenet1k-v2`
- Deprecation policy: 6 months notice before removing models/weights

**API Stability:**
- Public API guaranteed stable within major version
- Breaking changes only in major version bumps
- Deprecation warnings added one minor version before removal

### 4. Future Extensibility

#### 4.1 Planned Additions (Post-Phase 9)

**Vision Models:**
- DeiT (Data-efficient Image Transformers)
- BEiT (BERT Pre-Training of Image Transformers)
- CLIP (Contrastive Language-Image Pre-Training)
- Segment Anything Model (SAM)

**NLP Models:**
- LLaMA (Large Language Model Meta AI)
- Mistral (Mistral AI models)
- Gemma (Google models)
- Flan-T5 (Instruction-tuned T5)

**Multimodal Models:**
- CLIP for vision-language tasks
- BLIP for image captioning
- Flamingo for few-shot learning

**Model Compression:**
- DistilBERT, TinyBERT, MobileBERT
- Pruned model variants
- Quantized model variants

#### 4.2 Plugin API Design

```cpp
// Allow users to add custom models to registry
namespace tenzor::models {

class CustomModelRegistry {
public:
    static void register_model(
        const std::string& name,
        ModelFactory::CreateFn factory,
        const std::vector<WeightSpec>& weights
    );

    static void register_from_json(const std::string& config_path);
};

}

// User code:
TENZOR_REGISTER_MODEL("my-custom-resnet",
    [](const nlohmann::json& config) {
        return std::make_shared<MyCustomResNet>(config);
    },
    {
        {"v1", {
            .url = "https://my-server.com/weights.pth",
            .checksum = "abc123...",
            .format = FileFormat::PyTorch
        }}
    }
);
```

---

## Testing Requirements

### 1. Unit Test Coverage

**Target: 95% code coverage for models module**

#### 1.1 Vision Models

**ResNet Tests** (tests/unit/models/vision/test_resnet.cpp)
- Test BasicBlock forward/backward
- Test Bottleneck forward/backward
- Test ResNet-18, 50, 101 construction
- Test pretrained weight loading
- Test custom num_classes
- Test feature extraction mode
- Test gradient flow through skip connections
- Test batch norm in train vs. eval mode

**VGG Tests**
- Test VGG-16, VGG-19 construction
- Test with/without batch norm
- Test pretrained weight loading
- Test classifier dropout

**EfficientNet Tests**
- Test compound scaling (width, depth, resolution)
- Test MBConv blocks with squeeze-excitation
- Test Swish activation
- Test pretrained weight loading

**Vision Transformer Tests**
- Test patch embedding (image to patches)
- Test positional encoding addition
- Test class token concatenation
- Test transformer encoder
- Test classification head
- Test pretrained weight loading

**Detection Tests**
- Test RPN proposal generation
- Test ROI pooling and ROI Align
- Test FPN construction
- Test NMS with various IoU thresholds
- Test bounding box encoding/decoding
- Test end-to-end Faster R-CNN
- Test YOLO grid predictions

#### 1.2 NLP Models

**BERT Tests** (tests/unit/models/nlp/test_bert.cpp)
- Test BertEmbeddings (word + position + token type)
- Test BertSelfAttention with mask
- Test BertLayer forward/backward
- Test BertEncoder (12 layers)
- Test BertPooler
- Test BertModel end-to-end
- Test BertForSequenceClassification
- Test BertForTokenClassification
- Test BertForQuestionAnswering
- Test pretrained weight loading from Hugging Face
- Test gradient checkpointing for memory efficiency

**GPT-2 Tests**
- Test GPT2Block with causal attention
- Test GPT2Model end-to-end
- Test GPT2LMHeadModel
- Test key-value caching
- Test greedy generation
- Test beam search (beam size 1, 5, 10)
- Test top-k sampling
- Test top-p sampling
- Test temperature scaling
- Test pretrained weight loading

**RoBERTa Tests**
- Test RoBERTa model construction (BERT subclass)
- Test pretrained weight loading
- Test dynamic masking (if implemented)

**T5 Tests**
- Test T5 encoder-decoder architecture
- Test relative position bias
- Test text-to-text formatting
- Test pretrained weight loading

#### 1.3 Hub Tests

**Download Manager Tests** (tests/unit/models/hub/test_download.cpp)
- Test successful download
- Test resume from partial download
- Test retry on network failure
- Test timeout handling
- Test progress callback invocation
- Test proxy support
- Mock HTTP server for controlled testing

**Cache Manager Tests**
- Test cache directory creation (XDG compliance)
- Test cache hit (file exists, checksum matches)
- Test cache miss (file doesn't exist)
- Test cache invalidation (checksum mismatch)
- Test concurrent access (file locking)
- Test atomic file writes
- Test LRU eviction
- Test manual cache clearing

**Weight Manager Tests**
- Test load_weights (download + cache + deserialize)
- Test checksum verification (pass and fail)
- Test format auto-detection
- Test state dict deserialization (PyTorch, safetensors)
- Test architecture validation (layer name/shape mismatch)

**Model Registry Tests**
- Test model registration
- Test model lookup
- Test listing models with filter
- Test missing model error handling

### 2. Integration Tests

#### 2.1 Pretrained Weight Loading

**tests/integration/models/test_pretrained_loading.cpp**

```cpp
TEST(PretrainedLoading, ResNet50ImageNet) {
    auto model = tenzor::models::resnet50(true, 1000, false);
    ASSERT_NE(model, nullptr);

    // Load reference input and output
    auto input = load_tensor("testdata/imagenet_sample.pt");
    auto expected_output = load_tensor("testdata/resnet50_output.pt");

    model->eval();
    auto output = model->forward(input);

    EXPECT_TRUE(allclose(output, expected_output, 1e-5, 1e-5));
}

TEST(PretrainedLoading, BertBaseUncased) {
    auto model = tenzor::models::BertModel::from_pretrained("bert-base-uncased");
    ASSERT_NE(model, nullptr);

    auto input_ids = load_tensor("testdata/bert_input_ids.pt");
    auto attention_mask = load_tensor("testdata/bert_attention_mask.pt");
    auto expected_output = load_tensor("testdata/bert_output.pt");

    model->eval();
    auto output = model->forward(input_ids, attention_mask);

    EXPECT_TRUE(allclose(output.last_hidden_state, expected_output, 1e-5, 1e-5));
}
```

#### 2.2 Inference Accuracy

**tests/integration/models/test_inference_accuracy.cpp**

```cpp
TEST(InferenceAccuracy, ResNet50ImageNet100) {
    auto model = tenzor::models::resnet50(true);
    model->eval();

    auto dataset = load_imagenet_val_subset(100);
    int correct_top1 = 0, correct_top5 = 0;

    for (auto [image, label] : dataset) {
        auto output = model->forward(image.unsqueeze(0));
        auto probs = tenzor::softmax(output, 1);
        auto top5 = tenzor::topk(probs, 5);

        if (top5.indices[0].item<int>() == label) correct_top1++;
        for (int i = 0; i < 5; i++) {
            if (top5.indices[i].item<int>() == label) {
                correct_top5++;
                break;
            }
        }
    }

    float top1_acc = correct_top1 / 100.0;
    float top5_acc = correct_top5 / 100.0;

    EXPECT_GT(top1_acc, 0.70); // ResNet-50 achieves ~76% on full ImageNet
    EXPECT_GT(top5_acc, 0.90);
}

TEST(InferenceAccuracy, BertSST2) {
    auto model = tenzor::models::BertForSequenceClassification::from_pretrained(
        "textattack/bert-base-uncased-SST-2"
    );
    model->eval();

    auto dataset = load_sst2_subset(100);
    int correct = 0;

    for (auto [text, label] : dataset) {
        auto inputs = model->tokenizer()->encode(text);
        auto output = model->forward(inputs.input_ids, inputs.attention_mask);
        auto prediction = tenzor::argmax(output.logits, -1).item<int>();

        if (prediction == label) correct++;
    }

    float accuracy = correct / 100.0;
    EXPECT_GT(accuracy, 0.90); // Fine-tuned BERT achieves ~93% on SST-2
}
```

#### 2.3 Hugging Face Compatibility

**tests/integration/models/test_bert_huggingface.cpp**

```cpp
TEST(HuggingFaceCompat, BertOutputParity) {
    // Load same model in Tenzor and PyTorch
    auto tenzor_model = tenzor::models::BertModel::from_pretrained("bert-base-uncased");
    auto torch_model = load_torch_model("bert-base-uncased"); // via LibTorch

    tenzor_model->eval();
    torch_model->eval();

    auto input_ids = random_tensor({1, 128}, tenzor::kInt64);
    auto attention_mask = ones_tensor({1, 128}, tenzor::kInt64);

    auto tenzor_output = tenzor_model->forward(input_ids, attention_mask);
    auto torch_output = torch_model->forward(input_ids, attention_mask);

    // Outputs should match within numerical precision
    EXPECT_TRUE(allclose(tenzor_output.last_hidden_state,
                         torch_output.last_hidden_state,
                         1e-5, 1e-5));
}
```

#### 2.4 Text Generation

**tests/integration/models/test_generation.cpp**

```cpp
TEST(TextGeneration, GPT2GreedyCoherence) {
    auto model = tenzor::models::GPT2LMHeadModel::from_pretrained("gpt2");
    model->eval();

    auto prompt = "The quick brown fox";
    auto input_ids = model->tokenizer()->encode(prompt);

    auto output_ids = model->generate(input_ids, 20); // Generate 20 tokens
    auto output_text = model->tokenizer()->decode(output_ids);

    std::cout << "Generated: " << output_text << "\n";

    // Basic coherence check: no repetition of same token 5 times
    auto tokens = split(output_text, " ");
    EXPECT_FALSE(has_excessive_repetition(tokens, 5));
}

TEST(TextGeneration, GPT2BeamSearch) {
    auto model = tenzor::models::GPT2LMHeadModel::from_pretrained("gpt2");
    model->eval();

    GenerationConfig config;
    config.num_beams = 5;
    config.max_length = 30;

    auto prompt = "Once upon a time";
    auto input_ids = model->tokenizer()->encode(prompt);
    auto output_ids = model->generate(input_ids, config);
    auto output_text = model->tokenizer()->decode(output_ids);

    std::cout << "Generated (beam search): " << output_text << "\n";
    EXPECT_GT(output_text.length(), prompt.length());
}
```

### 3. Performance Tests

**tests/performance/bench_models.cpp**

```cpp
BENCHMARK(ResNet50_Forward_GPU) {
    auto model = tenzor::models::resnet50(true);
    model->to(tenzor::kCUDA);
    model->eval();

    auto input = random_tensor({32, 3, 224, 224}, tenzor::kCUDA);

    for (auto _ : state) {
        auto output = model->forward(input);
        tenzor::cuda::synchronize();
    }
}

BENCHMARK(BERT_Forward_GPU) {
    auto model = tenzor::models::BertModel::from_pretrained("bert-base-uncased");
    model->to(tenzor::kCUDA);
    model->eval();

    auto input_ids = random_tensor({32, 128}, tenzor::kCUDA, tenzor::kInt64);
    auto attention_mask = ones_tensor({32, 128}, tenzor::kCUDA, tenzor::kInt64);

    for (auto _ : state) {
        auto output = model->forward(input_ids, attention_mask);
        tenzor::cuda::synchronize();
    }
}

BENCHMARK(GPT2_Generation) {
    auto model = tenzor::models::GPT2LMHeadModel::from_pretrained("gpt2");
    model->to(tenzor::kCUDA);
    model->eval();

    auto input_ids = random_tensor({1, 10}, tenzor::kCUDA, tenzor::kInt64);

    for (auto _ : state) {
        auto output = model->generate(input_ids, 50);
    }
}
```

### 4. Continuous Integration

**CI Pipeline Steps:**

1. **Build Models Module** (5 min)
   - Compile C++ model code
   - Build Python bindings
   - Check for compilation warnings

2. **Run Unit Tests** (10 min)
   - Execute all unit tests
   - Generate coverage report
   - Fail if coverage < 95%

3. **Download Pretrained Weights** (10 min)
   - Download ResNet-50, BERT-base, GPT-2 to CI cache
   - Verify checksums
   - Cache for subsequent runs

4. **Run Integration Tests** (20 min)
   - Pretrained loading tests
   - Inference accuracy tests (subset of validation data)
   - Hugging Face compatibility tests
   - Generation tests

5. **Run Performance Tests** (15 min)
   - Execute benchmarks on GPU runner
   - Compare against baseline (fail if > 20% slower)
   - Upload results to dashboard

6. **Build Documentation** (5 min)
   - Generate Sphinx docs
   - Check for broken links
   - Deploy to docs site (on main branch)

**CI Configuration (.github/workflows/phase9.yml)**

```yaml
name: Phase 9 - Model Zoo

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest-gpu

    steps:
    - uses: actions/checkout@v3

    - name: Setup dependencies
      run: |
        sudo apt-get install libcurl4-openssl-dev libssl-dev
        pip install torch transformers

    - name: Build
      run: |
        mkdir build && cd build
        cmake -DBUILD_MODELS=ON -DBUILD_TESTS=ON ..
        make -j$(nproc)

    - name: Run unit tests
      run: |
        cd build
        ctest --output-on-failure -R "models_unit_*"

    - name: Download pretrained weights
      run: |
        python scripts/download_test_weights.py

    - name: Run integration tests
      run: |
        cd build
        ctest --output-on-failure -R "models_integration_*"

    - name: Run benchmarks
      run: |
        cd build
        ./bench_models --benchmark_format=json > bench_results.json

    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmarks
        path: build/bench_results.json
```

---

## Documentation Requirements

### 1. User Documentation

#### 1.1 Quickstart Guide

**docs/quickstart/model_zoo.md**

```markdown
# Getting Started with Tenzor Model Zoo

## Installation

Ensure Tenzor is built with model zoo support:

```bash
cmake -DBUILD_MODELS=ON ..
make install
```

## Loading a Pretrained Model (C++)

```cpp
#include <tenzor/models/vision/resnet.hpp>

int main() {
    // Load pretrained ResNet-50
    auto model = tenzor::models::resnet50(
        /*pretrained=*/true,
        /*num_classes=*/1000,
        /*progress=*/true
    );

    // Model is ready for inference
    model->eval();

    return 0;
}
```

## Loading a Pretrained Model (Python)

```python
import tenzor.models as models

# Load pretrained ResNet-50
model = models.vision.resnet50(pretrained=True)
model.eval()

# Model is ready for inference
```

## Next Steps

- [Vision Models Tutorial](tutorials/vision_models.md)
- [NLP Models Tutorial](tutorials/nlp_models.md)
- [Fine-tuning Guide](tutorials/fine_tuning.md)
- [API Reference](api/models/index.md)
```

#### 1.2 Tutorials

**docs/tutorials/vision_models.md**
- Image classification with ResNet
- Transfer learning (freezing layers, fine-tuning)
- Object detection with YOLO
- Image segmentation with U-Net

**docs/tutorials/nlp_models.md**
- Sequence classification with BERT
- Token classification (NER) with BERT
- Question answering with BERT
- Text generation with GPT-2

**docs/tutorials/fine_tuning.md**
- Loading pretrained weights
- Freezing/unfreezing layers
- Replacing classification head
- Training loop with optimizer
- Saving fine-tuned model

**docs/tutorials/deployment.md**
- Exporting to ONNX
- Quantization for mobile deployment
- Serving with TorchServe
- Performance optimization tips

#### 1.3 API Reference

**Automatically generated from Doxygen comments in C++ headers**

Example Doxygen comment:

```cpp
/**
 * @brief ResNet-50 model from "Deep Residual Learning for Image Recognition"
 *
 * ResNet-50 is a 50-layer residual network with bottleneck blocks.
 *
 * @param pretrained If true, loads ImageNet-1k pretrained weights
 * @param num_classes Number of output classes (default: 1000)
 * @param progress If true, displays download progress bar
 *
 * @return Shared pointer to ResNet model
 *
 * @example
 * ```cpp
 * auto model = tenzor::models::resnet50(true);
 * model->eval();
 * auto output = model->forward(input_image);
 * ```
 *
 * @see https://arxiv.org/abs/1512.03385
 */
std::shared_ptr<ResNet> resnet50(bool pretrained = false,
                                 int num_classes = 1000,
                                 bool progress = true);
```

### 2. Developer Documentation

#### 2.1 Architecture Documentation

**docs/architecture/model_zoo.md**
- High-level architecture diagram
- Component interaction flow
- Weight loading pipeline
- Extension points for custom models

#### 2.2 Contributing Guide

**docs/contributing/adding_models.md**

```markdown
# Adding a New Model to the Zoo

## Step 1: Implement Model Class

Create header file in `include/tenzor/models/vision/` or `include/tenzor/models/nlp/`:

```cpp
class YourModel : public VisionModel {
public:
    YourModel(const YourModelConfig& config);
    Tensor forward(const Tensor& x, bool extract_features = false) override;

    static std::shared_ptr<YourModel> from_pretrained(
        const std::string& variant,
        bool progress = true
    );

private:
    // Model layers
};
```

## Step 2: Implement Model Source

Create source file in `src/models/vision/your_model.cpp`:

```cpp
YourModel::YourModel(const YourModelConfig& config) {
    // Initialize layers
}

Tensor YourModel::forward(const Tensor& x, bool extract_features) {
    // Implement forward pass
}
```

## Step 3: Register Pretrained Weights

Add entry to `src/models/hub/registry_data.cpp`:

```cpp
registry["your-model-v1"] = ModelSpec{
    .name = "your-model-v1",
    .architecture = "YourModel",
    .weights = {
        {"default", {
            .url = "https://download.tenzor.ai/models/your-model-v1.pth",
            .checksum = "sha256:abc123...",
            .size_bytes = 102400000,
            .format = FileFormat::PyTorch,
            .version = "1.0"
        }}
    },
    .metadata = {
        .input_size = {224, 224},
        .num_classes = 1000,
        .description = "Your model trained on ImageNet-1k"
    }
};
```

## Step 4: Add Python Bindings

Update `python/models.cpp`:

```cpp
vision.def("your_model", &tenzor::models::your_model,
           py::arg("pretrained") = false,
           "Your Model");
```

## Step 5: Write Tests

Create `tests/unit/models/vision/test_your_model.cpp`:

```cpp
TEST(YourModel, ForwardPass) {
    auto model = tenzor::models::your_model(false);
    auto input = random_tensor({1, 3, 224, 224});
    auto output = model->forward(input);
    EXPECT_EQ(output.sizes(), std::vector<int64_t>({1, 1000}));
}
```

## Step 6: Add Documentation

Create model card in `docs/models/your_model.md` with:
- Architecture description
- Training details
- Performance metrics
- Usage examples
- Citations

## Step 7: Submit PR

- Ensure all tests pass
- Include benchmark results
- Update CHANGELOG.md
```

### 3. Model Cards

Each model has a dedicated model card following the structure:

```markdown
# ResNet-50

## Model Description

ResNet-50 is a 50-layer deep convolutional neural network from the paper "Deep Residual Learning for Image Recognition" (He et al., 2015). It introduces residual connections that enable training of very deep networks.

## Architecture

- **Layers**: 50 (1 conv + 48 in residual blocks + 1 fc)
- **Parameters**: 25.6M
- **FLOPs**: 4.1B (for 224x224 input)
- **Input**: RGB image, 224x224 (can handle arbitrary sizes)
- **Output**: 1000 classes (ImageNet-1k)

## Training

- **Dataset**: ImageNet ILSVRC-2012 (1.28M training images, 1000 classes)
- **Augmentation**: Random crop, horizontal flip, color jitter
- **Optimizer**: SGD with momentum (0.9)
- **Learning rate**: 0.1, decayed by 10 at epochs 30, 60, 90
- **Batch size**: 256
- **Epochs**: 90
- **Weight decay**: 1e-4

## Performance

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 76.13% |
| Top-5 Accuracy | 92.86% |
| Inference (GPU) | 15ms |
| Inference (CPU) | 120ms |

## Usage

**C++:**
```cpp
auto model = tenzor::models::resnet50(true);
model->eval();
auto output = model->forward(image);
```

**Python:**
```python
import tenzor.models as models
model = models.vision.resnet50(pretrained=True)
model.eval()
output = model(image)
```

## Limitations

- Trained on ImageNet, may not generalize to significantly different domains
- Performs poorly on very small objects (<32x32)
- Biased towards ImageNet class distribution

## Citation

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={CVPR},
  year={2016}
}
```

## License

Weights: MIT License
Code: Apache 2.0
```

---

## Dependencies

### 1. Internal Dependencies

**Phase 7: RNN & Transformer Layers** (REQUIRED)
- TransformerEncoderLayer for BERT, RoBERTa, ALBERT
- TransformerDecoderLayer for GPT-2, T5
- Multi-head attention with masking
- Positional encoding (learned and sinusoidal)
- Layer normalization

**Phase 8.1: Fused Operations** (RECOMMENDED)
- fused_conv_bn_relu for vision models
- fused_attention for transformers
- fused_mlp for transformer feed-forward

**Phase 8.2: Quantization** (OPTIONAL)
- Int8 quantization for deployment
- Quantization-aware training utilities

**Python Bindings** (REQUIRED)
- pybind11 infrastructure
- Tensor conversion utilities
- Exception handling

### 2. External Dependencies

**Required:**
- **LibTorch** (PyTorch C++ API): Loading .pth files
- **libcurl**: HTTP/HTTPS downloads
- **OpenSSL**: SHA256 checksum computation
- **nlohmann/json**: JSON parsing for manifests
- **spdlog**: Logging framework

**Optional:**
- **safetensors-cpp**: Loading Hugging Face .safetensors files
- **SentencePiece**: Tokenization for T5
- **Hugging Face Tokenizers**: Fast tokenization library
- **indicators**: Progress bars (header-only)

### 3. Dependency Installation

**Ubuntu/Debian:**
```bash
sudo apt-get install \
    libcurl4-openssl-dev \
    libssl-dev \
    nlohmann-json3-dev
```

**macOS:**
```bash
brew install curl openssl nlohmann-json
```

**LibTorch:**
```bash
# Download from pytorch.org
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```

**CMake Find Scripts:**
```cmake
# CMakeLists.txt
find_package(Torch REQUIRED)
find_package(CURL REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json REQUIRED)

# Optional: safetensors
FetchContent_Declare(
  safetensors
  GIT_REPOSITORY https://github.com/huggingface/safetensors.git
)
FetchContent_MakeAvailable(safetensors)
```

### 4. Version Requirements

| Dependency | Minimum Version | Recommended |
|------------|----------------|-------------|
| LibTorch | 2.0.0 | 2.1.0+ |
| CURL | 7.68.0 | 8.0.0+ |
| OpenSSL | 1.1.1 | 3.0.0+ |
| nlohmann/json | 3.9.0 | 3.11.0+ |
| CMake | 3.18 | 3.25+ |
| C++ Standard | C++17 | C++20 |

---

## Timeline and Priorities

### Overview (14 weeks, 220 hours)

| Priority | Component | Duration | Dependencies |
|----------|-----------|----------|--------------|
| P0 | Foundation & ResNet | 2 weeks (40h) | Phase 8 |
| P0 | BERT Family | 2 weeks (45h) | Phase 7 |
| P1 | Additional Vision | 2 weeks (35h) | P0 Foundation |
| P1 | GPT-2 & Generation | 2 weeks (35h) | Phase 7 |
| P2 | Other Transformers | 1 week (30h) | P0 BERT |
| P2 | Modern CV | 2 weeks (35h) | P0 Foundation |
| P3 | Detection & Segmentation | 3 weeks (60h) | P1 Vision |

### Detailed Timeline

#### Week 1-2: Foundation & ResNet (P0, 40h)

**Week 1:**
- [ ] Weight Manager implementation (15h)
  - Download manager with libcurl
  - Cache manager with XDG directories
  - Checksum verification
  - Progress bars
- [ ] Model Registry (5h)
  - Registry data structure
  - Initial entries (ResNet, BERT)

**Week 2:**
- [ ] State Dict Deserialization (10h)
  - LibTorch integration
  - Safetensors integration
  - Format auto-detection
- [ ] ResNet-50 (10h)
  - BasicBlock, Bottleneck
  - ResNet class
  - Load pretrained weights
  - Validation tests

**Deliverables:**
- Working weight download/caching system
- ResNet-50 with pretrained ImageNet weights
- Unit tests for all components
- Integration test: ResNet inference accuracy

#### Week 3-4: BERT Family (P0, 45h)

**Week 3:**
- [ ] BERT Core (20h)
  - BertEmbeddings
  - BertLayer (attention + FFN)
  - BertEncoder
  - BertModel
  - Load Hugging Face weights

**Week 4:**
- [ ] BERT Task Heads (15h)
  - BertForSequenceClassification
  - BertForTokenClassification
  - BertForQuestionAnswering
- [ ] Tokenization (10h)
  - WordPiece tokenizer
  - Vocabulary loading
  - Batch encoding

**Deliverables:**
- Complete BERT implementation
- Hugging Face weight compatibility
- Task-specific heads
- Integration test: BERT on GLUE tasks

#### Week 5-6: Additional Vision Models (P1, 35h)

**Week 5:**
- [ ] VGG Family (8h)
  - VGG-16, VGG-19
  - Batch norm variants
  - Pretrained weights
- [ ] EfficientNet (12h)
  - MBConv blocks
  - Compound scaling
  - EfficientNet-B0 to B4

**Week 6:**
- [ ] Vision Transformer (15h)
  - Patch embedding
  - Transformer encoder
  - ViT-B/16, ViT-L/16
  - ImageNet-21k weights

**Deliverables:**
- VGG, EfficientNet, ViT implementations
- Pretrained weights for all models
- Inference accuracy tests

#### Week 7-8: GPT-2 & Generation (P1, 35h)

**Week 7:**
- [ ] GPT-2 Core (15h)
  - GPT2Block with causal attention
  - GPT2Model
  - GPT2LMHeadModel
  - Load Hugging Face weights

**Week 8:**
- [ ] Text Generation (15h)
  - Greedy decoding
  - Beam search
  - Sampling strategies (top-k, top-p)
  - Temperature scaling
- [ ] BPE Tokenization (5h)
  - Byte-pair encoding
  - GPT-2 vocabulary

**Deliverables:**
- Complete GPT-2 implementation
- All generation strategies
- Coherent text generation demo

#### Week 9: Other Transformers (P2, 30h)

- [ ] RoBERTa (8h)
  - Subclass BERT
  - Load Hugging Face weights
- [ ] ALBERT (10h)
  - Factorized embeddings
  - Parameter sharing
  - Load weights
- [ ] T5 (12h)
  - Encoder-decoder
  - Relative position bias
  - SentencePiece tokenization
  - Load T5-small

**Deliverables:**
- RoBERTa, ALBERT, T5 implementations
- Hugging Face compatibility
- Inference tests

#### Week 10-11: Modern CV Architectures (P2, 35h)

**Week 10:**
- [ ] Swin Transformer (15h)
  - Shifted window attention
  - Patch merging
  - Swin-T, Swin-S
  - ImageNet weights

**Week 11:**
- [ ] ConvNeXt (10h)
  - Modernized ResNet blocks
  - ConvNeXt-T, ConvNeXt-S
- [ ] MobileNet (10h)
  - Depthwise separable convolutions
  - MobileNetV2, MobileNetV3

**Deliverables:**
- Swin, ConvNeXt, MobileNet implementations
- Pretrained weights
- Mobile deployment examples

#### Week 12-14: Detection & Segmentation (P3, 60h)

**Week 12:**
- [ ] Foundational Components (15h)
  - RPN (Region Proposal Network)
  - ROI pooling/align
  - FPN (Feature Pyramid Network)
  - NMS (Non-Maximum Suppression)

**Week 13:**
- [ ] Faster R-CNN (20h)
  - Two-stage pipeline
  - ResNet-50 backbone
  - COCO pretrained weights
  - mAP evaluation

**Week 14:**
- [ ] YOLO (15h)
  - YOLOv5s architecture
  - Grid predictions
  - NMS post-processing
- [ ] U-Net (10h)
  - Encoder-decoder
  - Skip connections
  - Segmentation head

**Deliverables:**
- Faster R-CNN, YOLO, U-Net implementations
- COCO pretrained weights
- Object detection/segmentation demos

### Priority Justification

**P0 (Critical Path):**
- Foundation infrastructure required for all models
- BERT and ResNet are most requested models
- Establishes patterns for future models

**P1 (High Value):**
- GPT-2 demonstrates generative capabilities
- Additional vision models provide model variety
- High user demand

**P2 (Medium Value):**
- Other transformers round out NLP offerings
- Modern CV architectures (Swin, ConvNeXt) show SOTA performance

**P3 (Specialized):**
- Detection/segmentation for specialized use cases
- Can be deferred to post-Phase 9 if timeline pressure

### Risk Mitigation

**Risks:**
1. **LibTorch integration complexity**: Mitigate by allocating extra time for P0
2. **Hugging Face weight format changes**: Use stable safetensors format
3. **Performance issues**: Leverage Phase 8 fused ops early
4. **Download infrastructure failures**: Implement robust retry and fallback logic

**Contingency Plan:**
- If timeline slips, defer P3 (detection/segmentation) to Phase 9.1
- Focus on quality over quantity: fewer models with perfect implementation beats many models with bugs

---

## Completion Criteria

### Functional Completeness

- [ ] All P0 models implemented and tested (ResNet, BERT)
- [ ] All P1 models implemented and tested (VGG, EfficientNet, ViT, GPT-2)
- [ ] Weight management system fully functional
- [ ] Pretrained weights load correctly for all models
- [ ] Python bindings complete for all models

### Quality Standards

- [ ] 95%+ code coverage for models module
- [ ] All integration tests pass (accuracy validation)
- [ ] Performance benchmarks meet targets (within 10% of PyTorch)
- [ ] No memory leaks detected (valgrind clean)
- [ ] Documentation complete (API docs, tutorials, model cards)

### Validation Metrics

**Inference Accuracy (vs. reference implementations):**
- [ ] ResNet-50: Top-1 accuracy > 75% on ImageNet subset
- [ ] BERT: Accuracy within 1% of Hugging Face on GLUE tasks
- [ ] GPT-2: Perplexity within 5% of reference on WikiText-2
- [ ] ViT: Top-1 accuracy > 80% on ImageNet subset

**Performance Targets:**
- [ ] ResNet-50 forward pass: < 50ms on GPU (batch=1)
- [ ] BERT forward pass: < 100ms on GPU (seq_len=128, batch=1)
- [ ] GPT-2 generation: < 2s for 50 tokens on GPU
- [ ] Weight loading: < 5s for models up to 500MB

**Code Quality:**
- [ ] Zero compiler warnings
- [ ] Clang-tidy clean
- [ ] Memory leak free (valgrind)
- [ ] Thread-safe (TSAN clean)

### Production Readiness

- [ ] CI/CD pipeline includes all model tests
- [ ] Pretrained weights hosted on reliable CDN
- [ ] Documentation deployed and accessible
- [ ] Python package includes models module
- [ ] Release notes prepared
- [ ] Migration guide from PyTorch drafted

---

## Appendix

### A. Model Specifications

#### Computer Vision Models

| Model | Params | FLOPs | Top-1 Acc | ImageNet Weights |
|-------|--------|-------|-----------|------------------|
| ResNet-18 | 11.7M | 1.8B | 69.76% | ✓ |
| ResNet-50 | 25.6M | 4.1B | 76.13% | ✓ |
| ResNet-101 | 44.5M | 7.8B | 77.37% | ✓ |
| VGG-16 | 138M | 15.5B | 71.59% | ✓ |
| VGG-19 | 144M | 19.6B | 72.38% | ✓ |
| EfficientNet-B0 | 5.3M | 0.4B | 77.1% | ✓ |
| EfficientNet-B4 | 19.0M | 4.2B | 82.9% | ✓ |
| ViT-B/16 | 86M | 17.6B | 81.1% | ✓ (ImageNet-21k) |
| Swin-T | 28M | 4.5B | 81.3% | ✓ |
| ConvNeXt-T | 28M | 4.5B | 82.1% | ✓ |

#### NLP Models

| Model | Params | Hidden Size | Layers | Heads | Weights Available |
|-------|--------|-------------|--------|-------|-------------------|
| BERT-base-uncased | 110M | 768 | 12 | 12 | ✓ |
| BERT-large-uncased | 340M | 1024 | 24 | 16 | ✓ |
| GPT-2 | 117M | 768 | 12 | 12 | ✓ |
| GPT-2 Medium | 345M | 1024 | 24 | 16 | ✓ |
| RoBERTa-base | 125M | 768 | 12 | 12 | ✓ |
| ALBERT-base-v2 | 12M | 768 | 12 | 12 | ✓ |
| T5-small | 60M | 512 | 6/6 | 8 | ✓ |

### B. Weight URLs

Pretrained weights will be hosted at:

**Primary:** `https://download.tenzor.ai/models/`
**Mirror:** `https://cdn.tenzor.ai/models/`
**Fallback:** Hugging Face Hub via API

URL format: `{base_url}/{model_name}-{variant}-{version}.{format}`

Example:
- `https://download.tenzor.ai/models/resnet50-imagenet1k-v1.pth`
- `https://download.tenzor.ai/models/bert-base-uncased-v1.safetensors`

### C. File Formats

**PyTorch Format (.pth, .pt):**
- Pickle-based serialization
- Loaded via LibTorch C++ API
- Supports arbitrary Python objects (requires care with security)

**Safetensors Format (.safetensors):**
- Zero-copy tensor deserialization
- Memory-safe (no arbitrary code execution)
- Preferred for production use
- Loaded via safetensors-cpp

**Conversion Script:**
```python
# scripts/convert_weights.py
import torch
from safetensors.torch import save_file

# Load PyTorch weights
weights = torch.load("model.pth")

# Convert to safetensors
save_file(weights, "model.safetensors")
```

### D. Tokenization Details

**WordPiece (BERT):**
- Subword tokenization
- Vocabulary size: 30,522 (BERT-base-uncased)
- Special tokens: [CLS], [SEP], [MASK], [PAD], [UNK]

**BPE (GPT-2):**
- Byte-level byte-pair encoding
- Vocabulary size: 50,257
- No special tokens in vocabulary (added dynamically)

**SentencePiece (T5):**
- Unigram language model
- Vocabulary size: 32,000 (T5-small)
- Special tokens: <pad>, </s>, <unk>

### E. References

**Papers:**
1. ResNet: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
2. VGG: Simonyan & Zisserman, "Very Deep Convolutional Networks", ICLR 2015
3. EfficientNet: Tan & Le, "EfficientNet: Rethinking Model Scaling", ICML 2019
4. ViT: Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
5. Swin: Liu et al., "Swin Transformer: Hierarchical Vision Transformer", ICCV 2021
6. BERT: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", NAACL 2019
7. GPT-2: Radford et al., "Language Models are Unsupervised Multitask Learners", 2019
8. RoBERTa: Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach", 2019
9. T5: Raffel et al., "Exploring the Limits of Transfer Learning", JMLR 2020

**Code References:**
- PyTorch torchvision.models: https://github.com/pytorch/vision/tree/main/torchvision/models
- Hugging Face transformers: https://github.com/huggingface/transformers

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-17 | System Architect | Initial specification |

---

**END OF SPECIFICATION**
