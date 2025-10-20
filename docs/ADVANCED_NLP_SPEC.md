# Advanced NLP Models Implementation Specification

**Status**: Research & Specification Phase
**Date**: 2025-10-18
**Purpose**: Detailed architectural specifications for implementing RoBERTa, ALBERT, T5, and ELECTRA models in C++

---

## Table of Contents

1. [Overview](#overview)
2. [RoBERTa (Robustly Optimized BERT)](#roberta)
3. [ALBERT (A Lite BERT)](#albert)
4. [T5 (Text-to-Text Transfer Transformer)](#t5)
5. [ELECTRA (Efficiently Learning an Encoder)](#electra)
6. [Code Reuse Strategy](#code-reuse-strategy)
7. [Implementation Priorities](#implementation-priorities)
8. [Reference Papers](#reference-papers)

---

## Overview

This document provides detailed architectural specifications for implementing four advanced NLP models based on the existing Tenzor BERT and GPT implementations located at:
- `/home/lee/Projects/Tenzor/include/tenzor/models/bert.hpp`
- `/home/lee/Projects/Tenzor/include/tenzor/models/gpt.hpp`
- `/home/lee/Projects/Tenzor/src/models/bert.cpp`
- `/home/lee/Projects/Tenzor/src/models/gpt.cpp`

All models extend or modify the base BERT/GPT architectures with specific optimizations and training objectives.

---

## RoBERTa

### Overview
RoBERTa (Robustly Optimized BERT Approach) improves upon BERT through modified training procedures and architectural tweaks, achieving better performance without changing the core transformer architecture.

### Key Differences from BERT

#### 1. Dynamic Masking
**BERT Approach:**
- Static masking: masks are generated once during preprocessing
- Same mask used for all training epochs

**RoBERTa Approach:**
- Dynamic masking: new masks generated each time a sequence is fed to model
- Different mask patterns across training epochs
- Prevents model from memorizing specific masked positions

**Implementation Impact:**
```cpp
// No architectural change needed - this is a data preprocessing difference
// Masking should be done in data loader, not model architecture
// Each epoch generates new random masks for the same sequence
```

#### 2. Removed Next Sentence Prediction (NSP)
**BERT Architecture:**
```cpp
class BertModel {
    BertEmbeddings embeddings_;  // Includes token_type_embeddings for NSP
    BertEncoder encoder_;
    BertPooler pooler_;          // Used for NSP classification
};
```

**RoBERTa Architecture:**
```cpp
class RobertaModel {
    RobertaEmbeddings embeddings_;  // No token_type_embeddings (or single segment)
    BertEncoder encoder_;           // Same as BERT
    // No pooler needed for pre-training (only MLM objective)
};
```

**Code Changes:**
- Remove or make optional `token_type_embeddings` in embeddings layer
- Simplify pre-training to use only Masked Language Model (MLM) objective
- Pooler still needed for downstream tasks (classification, etc.)

#### 3. Larger Batch Sizes & Longer Training
**BERT Training:**
- Batch size: 256 sequences
- Training steps: 1M steps
- Learning rate: 1e-4

**RoBERTa Training:**
- Batch size: 8K sequences (using gradient accumulation)
- Training steps: 500K steps
- Learning rate: adaptive with warmup

**Implementation Impact:**
- No architectural changes
- Affects training loop and optimizer configuration
- May require gradient accumulation if GPU memory limited

#### 4. Byte-Level BPE Tokenization
**BERT Tokenization:**
- WordPiece tokenization
- Vocabulary size: 30,522

**RoBERTa Tokenization:**
- Byte-level Byte-Pair Encoding (BPE)
- Vocabulary size: 50,265
- Universal encoding (no unknown tokens for any Unicode character)

**Implementation Impact:**
```cpp
struct RobertaConfig {
    int64_t vocab_size = 50265;  // Changed from BERT's 30522
    // ... rest same as BERT
};
```

#### 5. Training on Full Sentences
**BERT Training:**
- Combines segments from multiple documents
- Uses NSP to learn cross-document relationships

**RoBERTa Training:**
- Packs full sentences from single document until max length
- No cross-document segments
- Better context coherence

**Implementation Impact:**
- Data loading strategy, not architectural change

### Configuration Parameters

```cpp
struct RobertaConfig {
    // Vocabulary
    int64_t vocab_size = 50265;              // Byte-level BPE vocab

    // Architecture (same as BERT)
    int64_t hidden_size = 768;               // Embedding dimension
    int64_t num_hidden_layers = 12;          // Encoder layers
    int64_t num_attention_heads = 12;        // Attention heads
    int64_t intermediate_size = 3072;        // FFN dimension (4 * hidden_size)

    // Regularization
    double hidden_dropout_prob = 0.1;
    double attention_probs_dropout_prob = 0.1;

    // Position & Segments
    int64_t max_position_embeddings = 514;   // Slightly larger than BERT (512 + 2 special tokens)
    int64_t type_vocab_size = 1;             // Only one segment type (no NSP)

    // Layer Norm
    double layer_norm_eps = 1e-5;            // Changed from BERT's 1e-12

    // Activation
    std::string hidden_act = "gelu";

    // Static factory methods
    static RobertaConfig base() {
        return RobertaConfig{};  // 125M params
    }

    static RobertaConfig large() {
        RobertaConfig config;
        config.hidden_size = 1024;
        config.num_hidden_layers = 24;
        config.num_attention_heads = 16;
        config.intermediate_size = 4096;
        return config;  // 355M params
    }
};
```

### Leveraging Existing BERT Code

**High Reusability:**
- `BertEncoder`: 100% reusable (identical transformer stack)
- `BertEmbeddings`: 95% reusable (modify to handle single segment type)
- `TransformerEncoderLayer`: 100% reusable
- `MultiheadAttention`: 100% reusable

**Modifications Needed:**
```cpp
class RobertaEmbeddings : public nn::Module {
    // Inherit most of BertEmbeddings implementation
    // Key difference: token_type_embeddings defaults to zeros or single type

    auto forward(const Variable& input_ids,
                const Variable& token_type_ids = Variable{},  // Usually unused
                const Variable& position_ids = Variable{}) -> Variable {
        // If token_type_ids not provided, use all zeros (single segment)
        // Rest identical to BERT
    }
};
```

### Pre-training Objectives

**RoBERTa uses only MLM (no NSP):**
```cpp
class RobertaForMaskedLM : public nn::Module {
    std::shared_ptr<RobertaModel> roberta_;
    std::shared_ptr<nn::Linear> lm_head_;

    auto forward(const Variable& input_ids,
                const Tensor& attention_mask = Tensor{}) -> Variable {
        auto outputs = roberta_->forward(input_ids, attention_mask);
        auto sequence_output = outputs.sequence_output;
        return lm_head_->forward(sequence_output);  // Predict masked tokens
    }
};
```

### Reference Implementation
- **Paper**: "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (Liu et al., 2019)
- **Official Code**: https://github.com/pytorch/fairseq/tree/master/examples/roberta
- **Hugging Face**: transformers.RobertaModel

---

## ALBERT

### Overview
ALBERT (A Lite BERT) achieves parameter reduction through factorized embeddings and cross-layer parameter sharing while maintaining or improving performance.

### Key Architectural Differences from BERT

#### 1. Factorized Embedding Parameterization

**BERT Embeddings:**
```
Vocabulary size (V) × Hidden size (H) = 30,522 × 768 = 23.4M parameters
```

**ALBERT Embeddings:**
```
Factorization: V × E → E × H
Where E (embedding size) << H (hidden size)

Example: 30,000 × 128 → 128 × 768 = 3.84M + 0.098M = 3.94M parameters
Reduction: 83% fewer parameters in embedding layer
```

**Implementation:**
```cpp
class AlbertEmbeddings : public nn::Module {
public:
    AlbertEmbeddings(const AlbertConfig& config)
        : config_(config) {
        // Small embedding matrix
        word_embeddings_ = std::make_shared<nn::Embedding>(
            config.vocab_size,
            config.embedding_size  // Much smaller than hidden_size
        );

        position_embeddings_ = std::make_shared<nn::Embedding>(
            config.max_position_embeddings,
            config.embedding_size
        );

        token_type_embeddings_ = std::make_shared<nn::Embedding>(
            config.type_vocab_size,
            config.embedding_size
        );

        // Projection layer to expand to hidden_size
        embedding_projection_ = std::make_shared<nn::Linear>(
            config.embedding_size,
            config.hidden_size
        );

        layer_norm_ = std::make_shared<nn::LayerNorm>(
            std::vector<int64_t>{config.hidden_size},
            config.layer_norm_eps
        );
        dropout_ = std::make_shared<nn::Dropout>(config.hidden_dropout_prob);
    }

    auto forward(const Variable& input_ids,
                const Variable& token_type_ids = Variable{},
                const Variable& position_ids = Variable{}) -> Variable {
        // Get embeddings (size E)
        auto embeddings = word_embeddings_->forward(input_ids);
        auto position_embeddings = get_position_embeddings(input_ids, position_ids);
        auto token_type_embeddings = get_token_type_embeddings(input_ids, token_type_ids);

        // Combine (still size E)
        embeddings = embeddings + position_embeddings + token_type_embeddings;

        // Project to hidden_size H
        embeddings = embedding_projection_->forward(embeddings);

        // Layer norm and dropout
        embeddings = layer_norm_->forward(embeddings);
        embeddings = dropout_->forward(embeddings);

        return embeddings;
    }

private:
    AlbertConfig config_;
    std::shared_ptr<nn::Embedding> word_embeddings_;      // V × E
    std::shared_ptr<nn::Embedding> position_embeddings_;   // P × E
    std::shared_ptr<nn::Embedding> token_type_embeddings_; // T × E
    std::shared_ptr<nn::Linear> embedding_projection_;     // E × H
    std::shared_ptr<nn::LayerNorm> layer_norm_;
    std::shared_ptr<nn::Dropout> dropout_;
};
```

#### 2. Cross-Layer Parameter Sharing

**BERT Approach:**
```cpp
class BertEncoder {
    std::vector<std::shared_ptr<TransformerEncoderLayer>> layers_;  // Each layer unique

    BertEncoder(config) {
        for (int i = 0; i < num_layers; ++i) {
            layers_.push_back(std::make_shared<TransformerEncoderLayer>(...));
        }
    }
};
```

**ALBERT Approach - Share ALL parameters:**
```cpp
class AlbertEncoder : public nn::Module {
public:
    AlbertEncoder(const AlbertConfig& config)
        : config_(config) {
        // Create a SINGLE transformer layer
        shared_layer_ = std::make_shared<nn::TransformerEncoderLayer>(
            config.hidden_size,
            config.num_attention_heads,
            config.intermediate_size,
            config.attention_probs_dropout_prob,
            config.hidden_act,
            true  // batch_first
        );

        register_module("shared_layer", shared_layer_);
    }

    auto forward(const Variable& hidden_states,
                const Tensor& attention_mask = Tensor{}) -> Variable {
        auto output = hidden_states;

        // Apply the SAME layer multiple times
        for (int64_t i = 0; i < config_.num_hidden_layers; ++i) {
            output = shared_layer_->forward(output, attention_mask, Tensor{});
        }

        return output;
    }

private:
    AlbertConfig config_;
    std::shared_ptr<nn::TransformerEncoderLayer> shared_layer_;  // Single shared layer
};
```

**Parameter Reduction:**
- BERT-base: 12 unique layers × 7M params/layer = 84M params
- ALBERT-base: 1 layer × 7M params = 7M params
- **Reduction: 91% fewer encoder parameters**

**Alternative: Share only attention or only FFN:**
```cpp
class AlbertEncoderPartialSharing : public nn::Module {
    // Option 1: Share attention, unique FFN
    std::shared_ptr<nn::MultiheadAttention> shared_attention_;
    std::vector<std::shared_ptr<FeedForward>> unique_ffn_layers_;

    // Option 2: Share FFN, unique attention
    std::vector<std::shared_ptr<nn::MultiheadAttention>> unique_attention_layers_;
    std::shared_ptr<FeedForward> shared_ffn_;
};
```

#### 3. Sentence-Order Prediction (SOP)

**BERT uses NSP (Next Sentence Prediction):**
- Positive: consecutive sentences
- Negative: random sentences from different documents
- Too easy - model relies on topic shift rather than coherence

**ALBERT uses SOP:**
- Positive: sentences in correct order
- Negative: sentences in SWAPPED order (from same document)
- Harder task - model must understand inter-sentence coherence

**Implementation:**
```cpp
struct AlbertSOPOutput {
    Variable sop_logits;  // [batch, 2] - binary classification
};

class AlbertForPreTraining : public nn::Module {
public:
    AlbertForPreTraining(const AlbertConfig& config, int64_t num_labels = 2)
        : config_(config) {
        albert_ = std::make_shared<AlbertModel>(config);

        // MLM head
        mlm_head_ = std::make_shared<nn::Linear>(config.hidden_size, config.vocab_size);

        // SOP head (sentence order prediction)
        sop_classifier_ = std::make_shared<nn::Linear>(config.hidden_size, 2);

        register_module("albert", albert_);
        register_module("mlm_head", mlm_head_);
        register_module("sop_classifier", sop_classifier_);
    }

    auto forward(const Variable& input_ids,
                const Tensor& attention_mask = Tensor{},
                const Variable& token_type_ids = Variable{})
        -> std::tuple<Variable, Variable> {
        auto outputs = albert_->forward(input_ids, attention_mask, token_type_ids);

        // MLM predictions
        auto mlm_logits = mlm_head_->forward(outputs.sequence_output);

        // SOP predictions (from pooled [CLS] token)
        auto sop_logits = sop_classifier_->forward(outputs.pooled_output);

        return {mlm_logits, sop_logits};
    }

private:
    AlbertConfig config_;
    std::shared_ptr<AlbertModel> albert_;
    std::shared_ptr<nn::Linear> mlm_head_;
    std::shared_ptr<nn::Linear> sop_classifier_;
};
```

### Configuration Parameters

```cpp
struct AlbertConfig {
    // Vocabulary
    int64_t vocab_size = 30000;

    // Factorized Embeddings
    int64_t embedding_size = 128;            // NEW: Small embedding dimension
    int64_t hidden_size = 768;               // Hidden dimension (larger than embedding)

    // Architecture
    int64_t num_hidden_layers = 12;          // How many times to apply shared layer
    int64_t num_attention_heads = 12;
    int64_t intermediate_size = 3072;

    // Regularization
    double hidden_dropout_prob = 0.1;        // Often higher in ALBERT (0.1)
    double attention_probs_dropout_prob = 0.1;

    // Position & Segments
    int64_t max_position_embeddings = 512;
    int64_t type_vocab_size = 2;

    // Layer Norm
    double layer_norm_eps = 1e-12;

    // Activation
    std::string hidden_act = "gelu";

    // Static factory methods
    static AlbertConfig base() {
        return AlbertConfig{};  // 12M params (vs BERT-base 110M)
    }

    static AlbertConfig large() {
        AlbertConfig config;
        config.hidden_size = 1024;
        config.num_hidden_layers = 24;
        config.num_attention_heads = 16;
        config.intermediate_size = 4096;
        return config;  // 18M params (vs BERT-large 340M)
    }

    static AlbertConfig xlarge() {
        AlbertConfig config;
        config.hidden_size = 2048;
        config.num_hidden_layers = 24;
        config.num_attention_heads = 16;
        config.intermediate_size = 8192;
        return config;  // 60M params
    }

    static AlbertConfig xxlarge() {
        AlbertConfig config;
        config.hidden_size = 4096;
        config.num_hidden_layers = 12;
        config.num_attention_heads = 64;
        config.intermediate_size = 16384;
        return config;  // 233M params
    }
};
```

### Comparison: ALBERT vs BERT Parameters

| Model | Params | Embedding | Encoder | Speed |
|-------|--------|-----------|---------|-------|
| BERT-base | 110M | 23M | 85M | 1.0× |
| ALBERT-base | 12M | 4M | 7M | 1.7× faster |
| BERT-large | 340M | 31M | 303M | 0.4× |
| ALBERT-large | 18M | 5M | 11M | 1.2× faster |
| ALBERT-xlarge | 60M | 8M | 46M | 0.9× |
| ALBERT-xxlarge | 233M | 21M | 209M | 0.3× |

### Leveraging Existing BERT Code

**Reusable Components:**
- `TransformerEncoderLayer`: 100% reusable (just shared, not unique per layer)
- `MultiheadAttention`: 100% reusable
- `BertPooler`: 100% reusable
- `BertForSequenceClassification` structure: 95% reusable

**New Components Needed:**
- `AlbertEmbeddings`: Factorized embeddings with projection layer
- `AlbertEncoder`: Single shared transformer layer applied N times
- `AlbertForPreTraining`: MLM + SOP objectives

**Code Sharing Strategy:**
```cpp
// 1. Embeddings: Extend BERT embeddings with factorization
class AlbertEmbeddings : public nn::Module {
    // Add embedding_projection_ layer
    // Rest similar to BertEmbeddings
};

// 2. Encoder: Simplify to single shared layer
class AlbertEncoder : public nn::Module {
    // Single shared_layer_ instead of vector of layers_
};

// 3. Base model: Reuse BERT structure
class AlbertModel : public nn::Module {
    std::shared_ptr<AlbertEmbeddings> embeddings_;  // Modified
    std::shared_ptr<AlbertEncoder> encoder_;        // Modified
    std::shared_ptr<BertPooler> pooler_;            // Reused!
};

// 4. Task-specific heads: Inherit BERT implementations
class AlbertForSequenceClassification : public nn::Module {
    // Nearly identical to BertForSequenceClassification
    // Just swap BertModel for AlbertModel
};
```

### Reference Implementation
- **Paper**: "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" (Lan et al., 2019)
- **Official Code**: https://github.com/google-research/albert
- **Hugging Face**: transformers.AlbertModel

---

## T5

### Overview
T5 (Text-to-Text Transfer Transformer) reformulates all NLP tasks as text-to-text problems using a unified encoder-decoder architecture. Unlike BERT (encoder-only) and GPT (decoder-only), T5 uses the full Transformer architecture.

### Key Architectural Differences

#### 1. Encoder-Decoder Architecture

**BERT:** Encoder-only (bidirectional attention)
**GPT:** Decoder-only (causal attention)
**T5:** Full Transformer (encoder + decoder)

```
Input: "translate English to German: Hello"
Encoder: Processes full input bidirectionally
Decoder: Generates "Hallo" autoregressively
```

**Architecture Diagram:**
```
Input Text
    |
    v
[Token Embeddings + Position Embeddings]
    |
    v
[Encoder Stack - Bidirectional]
    |  (Cross-attention connection)
    |  |
    v  v
[Decoder Stack - Causal + Cross-attention]
    |
    v
[LM Head]
    |
    v
Output Text
```

#### 2. Relative Position Embeddings

**BERT/GPT Position Embeddings:**
```cpp
// Absolute learned positions
position_embeddings_ = std::make_shared<nn::Embedding>(
    max_seq_len,    // 512 or 1024
    hidden_size
);

// Added to token embeddings
embeddings = token_embeds + position_embeds;
```

**T5 Relative Position Embeddings:**
```cpp
// Relative position bias added to attention scores
// Not added to embeddings, but to attention computation

class T5Attention : public nn::Module {
public:
    T5Attention(const T5Config& config, bool has_relative_attention_bias)
        : config_(config),
          has_relative_attention_bias_(has_relative_attention_bias) {

        // Standard Q, K, V projections
        q_proj_ = std::make_shared<nn::Linear>(config.d_model, config.d_kv * config.num_heads);
        k_proj_ = std::make_shared<nn::Linear>(config.d_model, config.d_kv * config.num_heads);
        v_proj_ = std::make_shared<nn::Linear>(config.d_model, config.d_kv * config.num_heads);
        o_proj_ = std::make_shared<nn::Linear>(config.d_kv * config.num_heads, config.d_model);

        // Relative position bias (only in first layer of each stack)
        if (has_relative_attention_bias_) {
            relative_attention_bias_ = std::make_shared<nn::Embedding>(
                config.relative_attention_num_buckets,
                config.num_heads
            );
        }
    }

    auto compute_bias(int64_t query_length, int64_t key_length) -> Tensor {
        // Compute relative position bucket indices
        // query_length × key_length grid of relative positions

        Tensor position_bias({config_.num_heads, query_length, key_length},
                            DType::Float32, Device::cpu());

        for (int64_t i = 0; i < query_length; ++i) {
            for (int64_t j = 0; j < key_length; ++j) {
                int64_t relative_position = i - j;
                int64_t bucket = relative_position_bucket(
                    relative_position,
                    config_.relative_attention_num_buckets,
                    config_.relative_attention_max_distance
                );

                // Look up bias for this bucket
                auto bias_values = relative_attention_bias_->forward(
                    Variable(Tensor({1}, DType::Int64, Device::cpu()).fill_(bucket), false)
                ).tensor();

                // Assign to all heads
                for (int64_t h = 0; h < config_.num_heads; ++h) {
                    position_bias.data<float>()[h * query_length * key_length + i * key_length + j]
                        = bias_values.data<float>()[h];
                }
            }
        }

        return position_bias;
    }

    // Compute which bucket a relative position falls into
    auto relative_position_bucket(int64_t relative_position,
                                  int64_t num_buckets,
                                  int64_t max_distance) -> int64_t {
        int64_t bucket = 0;
        int64_t n = -relative_position;  // Flip sign

        // Half buckets for exact positions (small distances)
        // Half for logarithmic buckets (large distances)
        int64_t num_exact_buckets = num_buckets / 2;
        int64_t max_exact = num_buckets / 2;
        bool is_small = n < max_exact;

        if (is_small) {
            bucket = n;
        } else {
            // Logarithmic bucketing for larger distances
            int64_t scale = (num_buckets - num_exact_buckets);
            bucket = num_exact_buckets + static_cast<int64_t>(
                std::log(static_cast<double>(n) / max_exact) /
                std::log(static_cast<double>(max_distance) / max_exact) * scale
            );
            bucket = std::min(bucket, num_buckets - 1);
        }

        // Add offset for negative positions
        if (relative_position < 0) {
            bucket += num_buckets;
        }

        return bucket;
    }

    auto forward(const Variable& hidden_states,
                const Variable& key_value_states = Variable{},  // For cross-attention
                const Tensor& attention_mask = Tensor{},
                const Tensor& position_bias = Tensor{}) -> std::pair<Variable, Tensor> {

        bool is_cross_attention = key_value_states.is_initialized();
        int64_t seq_len = hidden_states.shape()[1];

        // Compute Q, K, V
        auto q = q_proj_->forward(hidden_states);
        auto k = is_cross_attention ? k_proj_->forward(key_value_states)
                                     : k_proj_->forward(hidden_states);
        auto v = is_cross_attention ? v_proj_->forward(key_value_states)
                                     : v_proj_->forward(hidden_states);

        // Reshape for multi-head attention
        // ... (similar to BERT/GPT)

        // Compute attention scores: Q @ K^T
        auto scores = tenzor::matmul(q, k.transpose(-2, -1));

        // Add relative position bias
        Tensor bias = position_bias;
        if (!position_bias.numel() && has_relative_attention_bias_) {
            bias = compute_bias(seq_len, is_cross_attention ? key_value_states.shape()[1] : seq_len);
        }

        if (bias.numel() > 0) {
            scores = scores + Variable(bias, false);
        }

        // Apply mask and softmax
        if (attention_mask.numel() > 0) {
            scores = scores + Variable(attention_mask, false);
        }

        auto attn_weights = nn::softmax(scores, -1);

        // Apply attention to values
        auto context = tenzor::matmul(attn_weights, v);

        // Project output
        auto output = o_proj_->forward(context);

        return {output, bias};  // Return bias for reuse in subsequent layers
    }

private:
    T5Config config_;
    bool has_relative_attention_bias_;
    std::shared_ptr<nn::Linear> q_proj_;
    std::shared_ptr<nn::Linear> k_proj_;
    std::shared_ptr<nn::Linear> v_proj_;
    std::shared_ptr<nn::Linear> o_proj_;
    std::shared_ptr<nn::Embedding> relative_attention_bias_;  // Only in first layer
};
```

**Key Points:**
- Relative position bias is computed once in first layer
- Subsequent layers reuse the same bias
- Bias is added to attention scores, not embeddings
- Logarithmic bucketing for efficiency

#### 3. Pre-Layer Normalization

**BERT/GPT (Post-Norm):**
```cpp
// Add & Norm: x = LayerNorm(x + Sublayer(x))
auto residual = x;
x = sublayer(x);
x = layer_norm(residual + x);
```

**T5 (Pre-Norm):**
```cpp
// Norm & Add: x = x + Sublayer(LayerNorm(x))
auto residual = x;
x = layer_norm(x);
x = sublayer(x);
x = residual + x;
```

**Implementation:**
```cpp
class T5Block : public nn::Module {
    auto forward(const Variable& hidden_states,
                const Variable& encoder_hidden_states = Variable{}) -> Variable {
        // Self-attention with pre-norm
        auto residual = hidden_states;
        auto normed = layer_norm_1_->forward(hidden_states);
        auto attn_output = self_attention_->forward(normed);
        hidden_states = residual + attn_output;

        // Cross-attention (decoder only) with pre-norm
        if (is_decoder_ && encoder_hidden_states.is_initialized()) {
            residual = hidden_states;
            normed = layer_norm_2_->forward(hidden_states);
            auto cross_attn = cross_attention_->forward(normed, encoder_hidden_states);
            hidden_states = residual + cross_attn;
        }

        // Feed-forward with pre-norm
        residual = hidden_states;
        normed = layer_norm_ffn_->forward(hidden_states);
        auto ffn_output = feed_forward_->forward(normed);
        hidden_states = residual + ffn_output;

        return hidden_states;
    }
};
```

#### 4. No Dropout in Pre-training

T5 omits dropout during pre-training, only using it during fine-tuning. This simplifies training.

#### 5. Text-to-Text Framework

**All tasks converted to text-to-text format:**

```
Task: Translation
Input:  "translate English to German: Hello world"
Output: "Hallo Welt"

Task: Classification
Input:  "sentiment: This movie is great!"
Output: "positive"

Task: Summarization
Input:  "summarize: [long text]"
Output: "[summary]"

Task: Question Answering
Input:  "question: What is the capital? context: France's capital is Paris."
Output: "Paris"
```

**Implementation:**
```cpp
class T5ForConditionalGeneration : public nn::Module {
public:
    auto forward(const Variable& input_ids,
                const Variable& decoder_input_ids,
                const Tensor& attention_mask = Tensor{},
                const Tensor& decoder_attention_mask = Tensor{}) -> Variable {

        // Encode input
        auto encoder_outputs = encoder_->forward(input_ids, attention_mask);

        // Decode to generate output
        auto decoder_outputs = decoder_->forward(
            decoder_input_ids,
            encoder_outputs,
            decoder_attention_mask,
            Tensor{}  // encoder attention mask
        );

        // Project to vocabulary
        auto lm_logits = lm_head_->forward(decoder_outputs);

        return lm_logits;
    }

private:
    std::shared_ptr<T5Encoder> encoder_;
    std::shared_ptr<T5Decoder> decoder_;
    std::shared_ptr<nn::Linear> lm_head_;
};
```

### Configuration Parameters

```cpp
struct T5Config {
    // Vocabulary
    int64_t vocab_size = 32128;              // SentencePiece vocab

    // Architecture
    int64_t d_model = 512;                   // Hidden size (different naming convention)
    int64_t d_kv = 64;                       // Key/value dimension per head
    int64_t d_ff = 2048;                     // FFN intermediate size
    int64_t num_layers = 6;                  // Encoder + decoder layers (each)
    int64_t num_heads = 8;                   // Attention heads

    // Relative Position
    int64_t relative_attention_num_buckets = 32;
    int64_t relative_attention_max_distance = 128;

    // Regularization
    double dropout_rate = 0.1;               // Used in fine-tuning, not pre-training
    double layer_norm_epsilon = 1e-6;

    // Activation
    std::string dense_act_fn = "relu";       // T5 uses ReLU (not GELU)

    // Feed-forward
    bool is_gated_act = false;               // T5.1.1 uses gated activations

    // Static factory methods
    static T5Config small() {
        T5Config config;
        config.d_model = 512;
        config.d_kv = 64;
        config.d_ff = 2048;
        config.num_layers = 6;
        config.num_heads = 8;
        return config;  // 60M params
    }

    static T5Config base() {
        T5Config config;
        config.d_model = 768;
        config.d_kv = 64;
        config.d_ff = 3072;
        config.num_layers = 12;
        config.num_heads = 12;
        return config;  // 220M params
    }

    static T5Config large() {
        T5Config config;
        config.d_model = 1024;
        config.d_kv = 64;
        config.d_ff = 4096;
        config.num_layers = 24;
        config.num_heads = 16;
        return config;  // 770M params
    }

    static T5Config xl() {
        T5Config config;
        config.d_model = 2048;
        config.d_kv = 128;
        config.d_ff = 5120;
        config.num_layers = 24;
        config.num_heads = 32;
        return config;  // 3B params
    }

    static T5Config xxl() {
        T5Config config;
        config.d_model = 4096;
        config.d_kv = 128;
        config.d_ff = 10240;
        config.num_layers = 24;
        config.num_heads = 64;
        return config;  // 11B params
    }
};
```

### Pre-training Objective: Span Corruption

**Unlike BERT's masked tokens, T5 uses span masking:**

```
Original: "The quick brown fox jumps over the lazy dog"
Masked:   "The quick <X> jumps over <Y> dog"
Targets:  "<X> brown fox <Y> the lazy </S>"
```

**Implementation:**
```cpp
// Span corruption masks consecutive tokens
// Average span length: 3 tokens
// Masks ~15% of tokens (similar to BERT)

struct SpanCorruptionExample {
    std::vector<int64_t> input_ids;   // With <X>, <Y>, etc. sentinels
    std::vector<int64_t> target_ids;  // Masked spans with sentinels
};

// Masking function
auto create_span_corruption_example(const std::vector<int64_t>& tokens,
                                    double corruption_rate = 0.15,
                                    int64_t mean_span_length = 3)
    -> SpanCorruptionExample {
    // Randomly select spans to mask
    // Replace with sentinel tokens <X>, <Y>, etc.
    // Create target sequence with masked content
}
```

### Leveraging Existing Code

**From BERT:**
- `LayerNorm`: 100% reusable
- `FeedForward` (Linear layers): 100% reusable
- Token embeddings: Reusable (but remove position embeddings)

**From GPT:**
- Decoder architecture: 80% reusable (add cross-attention)
- Causal masking: Reusable

**From Transformer base:**
- `TransformerEncoder`: 60% reusable (modify for pre-norm and relative positions)
- `TransformerDecoder`: 60% reusable (modify for pre-norm and relative positions)

**New Components:**
- `T5Attention`: New implementation with relative position bias
- `T5Block`: Pre-norm variant of transformer block
- `T5Encoder`: Stack of T5Blocks
- `T5Decoder`: Stack of T5Blocks with cross-attention

```cpp
class T5Model : public nn::Module {
public:
    T5Model(const T5Config& config)
        : config_(config) {
        // Shared token embeddings (encoder and decoder share)
        shared_embeddings_ = std::make_shared<nn::Embedding>(
            config.vocab_size,
            config.d_model
        );

        encoder_ = std::make_shared<T5Encoder>(config, shared_embeddings_);
        decoder_ = std::make_shared<T5Decoder>(config, shared_embeddings_);

        register_module("shared", shared_embeddings_);
        register_module("encoder", encoder_);
        register_module("decoder", decoder_);
    }

    auto forward(const Variable& input_ids,
                const Variable& decoder_input_ids,
                const Tensor& attention_mask = Tensor{},
                const Tensor& decoder_attention_mask = Tensor{})
        -> std::pair<Variable, Variable> {

        // Encode
        auto encoder_outputs = encoder_->forward(input_ids, attention_mask);

        // Decode
        auto decoder_outputs = decoder_->forward(
            decoder_input_ids,
            encoder_outputs,
            decoder_attention_mask,
            attention_mask  // Pass encoder mask for cross-attention
        );

        return {encoder_outputs, decoder_outputs};
    }

private:
    T5Config config_;
    std::shared_ptr<nn::Embedding> shared_embeddings_;
    std::shared_ptr<T5Encoder> encoder_;
    std::shared_ptr<T5Decoder> decoder_;
};
```

### Reference Implementation
- **Paper**: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (Raffel et al., 2019)
- **Official Code**: https://github.com/google-research/text-to-text-transfer-transformer
- **Hugging Face**: transformers.T5Model

---

## ELECTRA

### Overview
ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately) uses a generator-discriminator architecture similar to GANs. It's more sample-efficient than BERT's MLM, training on all tokens rather than just masked ones.

### Key Architectural Differences

#### 1. Generator-Discriminator Architecture

**BERT Architecture:**
```
Input: "The [MASK] is blue"
BERT Model → Predict masked token
Training signal: Only from masked positions (~15% of tokens)
```

**ELECTRA Architecture:**
```
Input: "The [MASK] is blue"
    ↓
Generator (small BERT-like model)
    ↓
Generated: "The cat is blue"  (replace [MASK] with "cat")
    ↓
Discriminator (BERT-sized model)
    ↓
Classify EACH token as real/fake:
    [real, fake, real, real]

Training signal: From ALL tokens (100% of tokens)
```

**Implementation:**
```cpp
// Generator: Small masked language model
class ElectraGenerator : public nn::Module {
public:
    ElectraGenerator(const ElectraConfig& config)
        : config_(config) {
        // Smaller BERT-like model (e.g., 1/4 size of discriminator)
        BertConfig gen_config;
        gen_config.hidden_size = config.generator_hidden_size;  // e.g., 256
        gen_config.num_hidden_layers = config.generator_layers;  // e.g., 12
        gen_config.num_attention_heads = config.generator_heads;  // e.g., 4
        gen_config.intermediate_size = gen_config.hidden_size * 4;
        gen_config.vocab_size = config.vocab_size;

        generator_ = std::make_shared<BertModel>(gen_config);
        lm_head_ = std::make_shared<nn::Linear>(gen_config.hidden_size, config.vocab_size);

        register_module("generator", generator_);
        register_module("lm_head", lm_head_);
    }

    // Predict tokens for masked positions
    auto forward(const Variable& input_ids,
                const Tensor& attention_mask = Tensor{}) -> Variable {
        auto outputs = generator_->forward(input_ids, attention_mask);
        auto logits = lm_head_->forward(outputs.sequence_output);
        return logits;  // [batch, seq_len, vocab_size]
    }

private:
    ElectraConfig config_;
    std::shared_ptr<BertModel> generator_;
    std::shared_ptr<nn::Linear> lm_head_;
};

// Discriminator: Full-sized model
class ElectraDiscriminator : public nn::Module {
public:
    ElectraDiscriminator(const ElectraConfig& config)
        : config_(config) {
        // Full BERT-sized model
        BertConfig disc_config;
        disc_config.hidden_size = config.hidden_size;  // e.g., 768
        disc_config.num_hidden_layers = config.num_hidden_layers;  // e.g., 12
        disc_config.num_attention_heads = config.num_attention_heads;  // e.g., 12
        disc_config.intermediate_size = config.intermediate_size;
        disc_config.vocab_size = config.vocab_size;

        discriminator_ = std::make_shared<BertModel>(disc_config);

        // Binary classification head (real vs. replaced)
        classifier_ = std::make_shared<nn::Linear>(config.hidden_size, 1);

        register_module("discriminator", discriminator_);
        register_module("classifier", classifier_);
    }

    // Classify each token as real (0) or replaced (1)
    auto forward(const Variable& input_ids,
                const Tensor& attention_mask = Tensor{}) -> Variable {
        auto outputs = discriminator_->forward(input_ids, attention_mask);
        auto logits = classifier_->forward(outputs.sequence_output);
        return logits.squeeze(-1);  // [batch, seq_len]
    }

private:
    ElectraConfig config_;
    std::shared_ptr<BertModel> discriminator_;
    std::shared_ptr<nn::Linear> classifier_;
};
```

#### 2. Training Process

**Step-by-step ELECTRA training:**

```cpp
class ElectraForPreTraining : public nn::Module {
public:
    ElectraForPreTraining(const ElectraConfig& config)
        : config_(config) {
        generator_ = std::make_shared<ElectraGenerator>(config);
        discriminator_ = std::make_shared<ElectraDiscriminator>(config);

        register_module("generator", generator_);
        register_module("discriminator", discriminator_);
    }

    auto forward(const Variable& input_ids,
                const Tensor& masked_positions,  // Which positions are masked
                const Tensor& original_tokens)   // Original tokens (for labels)
        -> std::tuple<Variable, Variable, Tensor> {

        int64_t batch_size = input_ids.shape()[0];
        int64_t seq_len = input_ids.shape()[1];

        // Step 1: Generator predicts masked tokens
        auto gen_logits = generator_->forward(input_ids);  // [batch, seq_len, vocab]

        // Step 2: Sample tokens from generator predictions (at masked positions)
        auto gen_probs = nn::softmax(gen_logits, -1);

        // For each masked position, sample a token
        Tensor generated_tokens = input_ids.tensor();  // Copy original
        Tensor is_replaced(input_ids.shape(), DType::Float32, input_ids.tensor().device());
        is_replaced.zero_();

        const int64_t* mask_data = masked_positions.data<int64_t>();
        const int64_t* orig_data = original_tokens.data<int64_t>();
        int64_t* gen_data = generated_tokens.data<int64_t>();
        float* repl_data = is_replaced.data<float>();

        for (int64_t b = 0; b < batch_size; ++b) {
            for (int64_t s = 0; s < seq_len; ++s) {
                int64_t idx = b * seq_len + s;

                if (mask_data[idx] == 1) {  // This position was masked
                    // Sample from generator's distribution
                    // (In practice, use gumbel-softmax or actual sampling)
                    int64_t sampled_token = sample_from_distribution(
                        gen_probs.tensor().data<float>() + idx * config_.vocab_size,
                        config_.vocab_size
                    );

                    gen_data[idx] = sampled_token;

                    // Mark as replaced if different from original
                    repl_data[idx] = (sampled_token != orig_data[idx]) ? 1.0f : 0.0f;
                }
            }
        }

        // Step 3: Discriminator classifies all tokens as real/replaced
        Variable gen_input(generated_tokens, false);
        auto disc_logits = discriminator_->forward(gen_input);  // [batch, seq_len]

        return {gen_logits, disc_logits, is_replaced};
    }

    // Loss computation
    auto compute_loss(const Variable& gen_logits,
                     const Variable& disc_logits,
                     const Tensor& is_replaced,
                     const Tensor& masked_positions,
                     const Tensor& original_tokens) -> Variable {

        // Generator loss: MLM loss on masked tokens only
        auto gen_loss = masked_lm_loss(gen_logits, masked_positions, original_tokens);

        // Discriminator loss: Binary cross-entropy on ALL tokens
        auto disc_loss = binary_cross_entropy_with_logits(disc_logits, is_replaced);

        // Combined loss (generator loss typically scaled down)
        auto total_loss = config_.gen_loss_weight * gen_loss + disc_loss;

        return total_loss;
    }

private:
    ElectraConfig config_;
    std::shared_ptr<ElectraGenerator> generator_;
    std::shared_ptr<ElectraDiscriminator> discriminator_;

    auto sample_from_distribution(const float* probs, int64_t size) -> int64_t {
        // Sample token index from probability distribution
        // Can use gumbel-softmax for differentiability
        std::discrete_distribution<int64_t> dist(probs, probs + size);
        static std::mt19937 gen{42};
        return dist(gen);
    }
};
```

#### 3. Replaced Token Detection (RTD) Objective

**Key differences from BERT's MLM:**

| Aspect | BERT MLM | ELECTRA RTD |
|--------|----------|-------------|
| Masked tokens | 15% | 15% (same) |
| Training signal | Only masked positions | ALL positions |
| Task | Predict original token | Binary classification (real/fake) |
| Sample efficiency | Low (85% of tokens ignored) | High (100% of tokens used) |
| Difficulty | Easier (multi-class) | Harder (subtle replacements) |

**Why RTD is harder:**
```
Original: "The dog is brown"
BERT: "The [MASK] is brown" → Predict "dog" (from 30K vocab)
ELECTRA: "The cat is brown" → Detect "cat" is wrong (binary, but subtle)
```

#### 4. Weight Tying

**Generator and discriminator can share token embeddings:**

```cpp
class ElectraModel : public nn::Module {
public:
    ElectraModel(const ElectraConfig& config)
        : config_(config) {
        // Shared token embeddings
        if (config.tie_embeddings) {
            shared_embeddings_ = std::make_shared<nn::Embedding>(
                config.vocab_size,
                config.generator_hidden_size  // Use generator size
            );

            // Create models with shared embeddings
            generator_ = std::make_shared<ElectraGenerator>(
                config, shared_embeddings_
            );

            // Discriminator needs to project from generator embedding size
            discriminator_ = std::make_shared<ElectraDiscriminator>(
                config, shared_embeddings_
            );
        } else {
            generator_ = std::make_shared<ElectraGenerator>(config);
            discriminator_ = std::make_shared<ElectraDiscriminator>(config);
        }
    }

private:
    ElectraConfig config_;
    std::shared_ptr<nn::Embedding> shared_embeddings_;
    std::shared_ptr<ElectraGenerator> generator_;
    std::shared_ptr<ElectraDiscriminator> discriminator_;
};
```

### Configuration Parameters

```cpp
struct ElectraConfig {
    // Vocabulary
    int64_t vocab_size = 30522;

    // Discriminator (main model, same as BERT)
    int64_t hidden_size = 768;
    int64_t num_hidden_layers = 12;
    int64_t num_attention_heads = 12;
    int64_t intermediate_size = 3072;

    // Generator (smaller model)
    int64_t generator_hidden_size = 256;     // Typically 1/4 of discriminator
    int64_t generator_layers = 12;           // Same depth as discriminator
    int64_t generator_heads = 4;             // Fewer heads
    int64_t generator_intermediate_size = 1024;  // 4 * generator_hidden_size

    // Training
    double gen_loss_weight = 1.0;            // Weight for generator loss
    double disc_loss_weight = 50.0;          // Weight for discriminator loss
    double mask_prob = 0.15;                 // Masking probability
    bool tie_embeddings = true;              // Share token embeddings

    // Regularization
    double hidden_dropout_prob = 0.1;
    double attention_probs_dropout_prob = 0.1;

    // Position & Segments
    int64_t max_position_embeddings = 512;
    int64_t type_vocab_size = 2;

    // Layer Norm
    double layer_norm_eps = 1e-12;

    // Activation
    std::string hidden_act = "gelu";

    // Static factory methods
    static ElectraConfig small() {
        ElectraConfig config;
        config.hidden_size = 256;
        config.num_hidden_layers = 12;
        config.num_attention_heads = 4;
        config.intermediate_size = 1024;
        config.generator_hidden_size = 128;
        config.generator_heads = 2;
        config.generator_intermediate_size = 512;
        return config;  // 14M params (discriminator)
    }

    static ElectraConfig base() {
        return ElectraConfig{};  // 110M params (discriminator)
    }

    static ElectraConfig large() {
        ElectraConfig config;
        config.hidden_size = 1024;
        config.num_hidden_layers = 24;
        config.num_attention_heads = 16;
        config.intermediate_size = 4096;
        config.generator_hidden_size = 256;
        config.generator_heads = 4;
        config.generator_intermediate_size = 1024;
        return config;  // 335M params (discriminator)
    }
};
```

### Parameter Counts

**ELECTRA-Small:**
- Generator: 3M parameters
- Discriminator: 14M parameters
- **Total: 17M parameters**

**ELECTRA-Base:**
- Generator: 33M parameters (1/3 size)
- Discriminator: 110M parameters
- **Total: 143M parameters**

**ELECTRA-Large:**
- Generator: 80M parameters (1/4 size)
- Discriminator: 335M parameters
- **Total: 415M parameters**

### After Pre-training

**Only discriminator is used for downstream tasks:**
```cpp
// During fine-tuning, discard generator
class ElectraForSequenceClassification : public nn::Module {
public:
    ElectraForSequenceClassification(const ElectraConfig& config, int64_t num_labels)
        : config_(config), num_labels_(num_labels) {
        // Use ONLY the discriminator
        // It's essentially a BERT model at this point
        discriminator_ = std::make_shared<ElectraDiscriminator>(config);

        // Standard classification head
        classifier_ = std::make_shared<nn::Linear>(config.hidden_size, num_labels);

        register_module("discriminator", discriminator_);
        register_module("classifier", classifier_);
    }

    auto forward(const Variable& input_ids,
                const Tensor& attention_mask = Tensor{}) -> Variable {
        auto outputs = discriminator_->forward(input_ids, attention_mask);

        // Use [CLS] token representation (first token)
        auto cls_output = extract_cls_token(outputs);

        return classifier_->forward(cls_output);
    }

private:
    ElectraConfig config_;
    int64_t num_labels_;
    std::shared_ptr<ElectraDiscriminator> discriminator_;
    std::shared_ptr<nn::Linear> classifier_;
};
```

### Leveraging Existing BERT Code

**Extremely high reusability:**
- **Generator**: 100% reuse of `BertModel` + `nn::Linear` for LM head
- **Discriminator**: 100% reuse of `BertModel` + `nn::Linear` for binary classification
- **Embeddings**: 100% reuse of `BertEmbeddings`
- **Encoder**: 100% reuse of `BertEncoder`

**Only new component:**
- Training loop that coordinates generator and discriminator
- Sampling mechanism for replaced token detection

```cpp
// ELECTRA is essentially two BERT models working together
class ElectraForPreTraining : public nn::Module {
    // Generator = Small BertModel + LM head
    std::shared_ptr<BertModel> generator_bert_;
    std::shared_ptr<nn::Linear> generator_lm_head_;

    // Discriminator = Full BertModel + binary classification head
    std::shared_ptr<BertModel> discriminator_bert_;
    std::shared_ptr<nn::Linear> discriminator_classifier_;

    // Training coordination
    auto training_step(const Variable& input_ids) {
        // 1. Mask input
        // 2. Generator predicts
        // 3. Sample replacements
        // 4. Discriminator detects
        // 5. Compute combined loss
    }
};
```

### Reference Implementation
- **Paper**: "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators" (Clark et al., 2020)
- **Official Code**: https://github.com/google-research/electra
- **Hugging Face**: transformers.ElectraModel

---

## Code Reuse Strategy

### Existing Components (Fully Reusable)

| Component | Location | Used By |
|-----------|----------|---------|
| `MultiheadAttention` | `/home/lee/Projects/Tenzor/include/tenzor/nn/layers/attention.hpp` | All models |
| `TransformerEncoderLayer` | `/home/lee/Projects/Tenzor/include/tenzor/nn/layers/transformer.hpp` | RoBERTa, ALBERT, ELECTRA |
| `TransformerDecoderLayer` | `/home/lee/Projects/Tenzor/include/tenzor/nn/layers/transformer.hpp` | T5 (with modifications) |
| `LayerNorm` | `/home/lee/Projects/Tenzor/include/tenzor/nn/layers/normalization.hpp` | All models |
| `Linear` | `/home/lee/Projects/Tenzor/include/tenzor/nn/layers/linear.hpp` | All models |
| `Embedding` | `/home/lee/Projects/Tenzor/include/tenzor/nn/layers/embedding.hpp` | All models |
| `Dropout` | `/home/lee/Projects/Tenzor/include/tenzor/nn/layers/dropout.hpp` | All models |
| `GELU/ReLU` | `/home/lee/Projects/Tenzor/include/tenzor/nn/activations/activations.hpp` | All models |

### Component Reusability Matrix

| Component | RoBERTa | ALBERT | T5 | ELECTRA |
|-----------|---------|--------|-----|---------|
| BERT Embeddings | 95% | 50% | 30% | 100% |
| BERT Encoder | 100% | 80% | 60% | 100% |
| BERT Pooler | 100% | 100% | 0% | 100% |
| GPT Decoder | 0% | 0% | 70% | 0% |
| Transformer Layers | 100% | 100% | 70% | 100% |

### Implementation Order (Recommended)

**Phase 1: RoBERTa (Easiest)**
- Estimated effort: 1-2 days
- Requires: Minimal changes to `BertEmbeddings` and `BertConfig`
- Benefit: Quick win, validates training pipeline changes

**Phase 2: ELECTRA (Medium)**
- Estimated effort: 3-4 days
- Requires: New training loop, generator-discriminator coordination
- Benefit: High sample efficiency, interesting architecture

**Phase 3: ALBERT (Medium-Hard)**
- Estimated effort: 4-5 days
- Requires: New `AlbertEmbeddings` with factorization, new `AlbertEncoder` with sharing
- Benefit: Parameter efficiency, good for resource-constrained environments

**Phase 4: T5 (Hardest)**
- Estimated effort: 7-10 days
- Requires: New attention mechanism (relative positions), encoder-decoder architecture, text-to-text framework
- Benefit: Unified framework for all NLP tasks, state-of-the-art performance

### Shared Implementation Utilities

**Create common utilities for all models:**

```cpp
// File: /home/lee/Projects/Tenzor/include/tenzor/models/model_utils.hpp

namespace tenzor {
namespace models {

// Masking utilities
class MaskingUtils {
public:
    // Static masking (BERT-style)
    static auto static_masking(const std::vector<int64_t>& tokens,
                              double mask_prob = 0.15,
                              int64_t mask_token_id = 103,
                              int64_t vocab_size = 30522)
        -> std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>;

    // Dynamic masking (RoBERTa-style)
    static auto dynamic_masking(const std::vector<int64_t>& tokens,
                               double mask_prob = 0.15,
                               int64_t mask_token_id = 103,
                               int64_t vocab_size = 30522)
        -> std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>;

    // Span masking (T5-style)
    static auto span_masking(const std::vector<int64_t>& tokens,
                            double corruption_rate = 0.15,
                            int64_t mean_span_length = 3,
                            int64_t sentinel_start_id = 32000)
        -> std::tuple<std::vector<int64_t>, std::vector<int64_t>>;
};

// Attention mask utilities
class AttentionMaskUtils {
public:
    // Causal mask for autoregressive models
    static auto create_causal_mask(int64_t seq_len, Device device = Device::cpu())
        -> Tensor;

    // Padding mask
    static auto create_padding_mask(const Tensor& input_ids, int64_t pad_token_id = 0)
        -> Tensor;

    // Combined mask (causal + padding)
    static auto create_combined_mask(const Tensor& input_ids,
                                     int64_t pad_token_id = 0,
                                     bool causal = true)
        -> Tensor;
};

// Position encoding utilities
class PositionEncodingUtils {
public:
    // Sinusoidal position encoding (original Transformer)
    static auto sinusoidal_encoding(int64_t max_len, int64_t d_model, Device device)
        -> Tensor;

    // Learned position encoding (BERT/GPT)
    static auto learned_encoding(int64_t max_len, int64_t d_model)
        -> std::shared_ptr<nn::Embedding>;

    // Relative position encoding (T5)
    static auto relative_position_bucket(int64_t relative_position,
                                        int64_t num_buckets = 32,
                                        int64_t max_distance = 128)
        -> int64_t;
};

} // namespace models
} // namespace tenzor
```

### Directory Structure

```
include/tenzor/models/
├── bert.hpp                 # Existing
├── gpt.hpp                  # Existing
├── roberta.hpp              # New
├── albert.hpp               # New
├── t5.hpp                   # New
├── electra.hpp              # New
├── model_utils.hpp          # New - Shared utilities
└── hub.hpp                  # Existing - Model loading

src/models/
├── bert.cpp                 # Existing
├── gpt.cpp                  # Existing
├── roberta.cpp              # New
├── albert.cpp               # New
├── t5.cpp                   # New
├── electra.cpp              # New
├── model_utils.cpp          # New - Shared utilities
└── hub.cpp                  # Existing

tests/unit/
├── test_bert.cpp            # Existing
├── test_gpt.cpp             # Existing
├── test_roberta.cpp         # New
├── test_albert.cpp          # New
├── test_t5.cpp              # New
└── test_electra.cpp         # New

examples/nlp/
├── bert_example.cpp         # Existing
├── gpt_text_generation.cpp  # Existing
├── roberta_classification.cpp  # New
├── albert_ner.cpp           # New
├── t5_translation.cpp       # New
└── electra_pretraining.cpp  # New
```

---

## Implementation Priorities

### Priority 1: RoBERTa
**Reason:** Minimal code changes, validates training improvements
**Benefits:**
- Better performance than BERT with same architecture
- Dynamic masking improves robustness
- Easy to implement and test

### Priority 2: ELECTRA
**Reason:** High sample efficiency, novel training approach
**Benefits:**
- 4x more sample-efficient than BERT
- Interesting generator-discriminator architecture
- Good learning experience for adversarial training

### Priority 3: ALBERT
**Reason:** Parameter efficiency important for deployment
**Benefits:**
- 18x fewer parameters than BERT-large
- Good for mobile/edge deployment
- Teaches parameter sharing techniques

### Priority 4: T5
**Reason:** Most complex, but most general
**Benefits:**
- Unified text-to-text framework
- State-of-the-art on many benchmarks
- Good for sequence-to-sequence tasks

---

## Reference Papers

### RoBERTa
- **Title**: "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
- **Authors**: Yinhan Liu, Myle Ott, Naman Goyal, et al.
- **Year**: 2019
- **Link**: https://arxiv.org/abs/1907.11692

### ALBERT
- **Title**: "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"
- **Authors**: Zhenzhong Lan, Mingda Chen, Sebastian Goodman, et al.
- **Year**: 2019
- **Link**: https://arxiv.org/abs/1909.11942

### T5
- **Title**: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
- **Authors**: Colin Raffel, Noam Shazeer, Adam Roberts, et al.
- **Year**: 2019
- **Link**: https://arxiv.org/abs/1910.10683

### ELECTRA
- **Title**: "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators"
- **Authors**: Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning
- **Year**: 2020
- **Link**: https://arxiv.org/abs/2003.10555

### Original Papers (for reference)
- **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018)
- **GPT-2**: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- **Attention**: "Attention Is All You Need" (Vaswani et al., 2017)

---

## Next Steps

1. **Review this specification** with the team
2. **Choose implementation priority** based on project needs
3. **Set up test infrastructure** for new models
4. **Implement model utilities** (`model_utils.hpp/cpp`)
5. **Start with RoBERTa** as proof of concept
6. **Iterate to other models** based on priority

---

**Document Version**: 1.0
**Last Updated**: 2025-10-18
**Status**: Ready for Implementation
