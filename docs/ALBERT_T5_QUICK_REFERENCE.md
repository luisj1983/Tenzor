# ALBERT and T5 Quick Reference Guide

**Quick start guide for using ALBERT and T5 models in Tenzor**

---

## ALBERT Models

### Basic Usage

```cpp
#include "tenzor/models/albert.hpp"
using namespace tenzor::models;

// Create base model (12M parameters)
auto config = AlbertConfig::base();
auto albert = AlbertModel(config);

// Forward pass
Variable input_ids(Tensor({2, 128}, DType::Int64, Device::cpu()), false);
// ... fill input_ids ...

auto outputs = albert.forward(input_ids);
auto sequence_output = outputs.sequence_output;  // [2, 128, 768]
auto pooled_output = outputs.pooled_output;      // [2, 768]
```

### Model Variants

```cpp
// ALBERT-base: 12M params (H=768, L=12, E=128)
auto config = AlbertConfig::base();

// ALBERT-large: 18M params (H=1024, L=24, E=128)
auto config = AlbertConfig::large();

// ALBERT-xlarge: 60M params (H=2048, L=24, E=128)
auto config = AlbertConfig::xlarge();

// ALBERT-xxlarge: 233M params (H=4096, L=12, E=128)
auto config = AlbertConfig::xxlarge();
```

### Task-Specific Models

#### Sequence Classification

```cpp
// Text classification, sentiment analysis
auto classifier = AlbertForSequenceClassification(config, 2);  // 2 classes

Variable input_ids;  // [batch, seq_len]
auto logits = classifier.forward(input_ids);  // [batch, 2]
```

#### Token Classification

```cpp
// NER, POS tagging
auto tagger = AlbertForTokenClassification(config, 9);  // 9 NER tags

Variable input_ids;  // [batch, seq_len]
auto logits = tagger.forward(input_ids);  // [batch, seq_len, 9]
```

#### Pre-training

```cpp
// MLM + SOP objectives
auto pretraining = AlbertForPreTraining(config);

Variable input_ids;  // [batch, seq_len]
auto [mlm_logits, sop_logits] = pretraining.forward(input_ids);
// mlm_logits: [batch, seq_len, vocab_size]
// sop_logits: [batch, 2] - binary classification
```

### Key Features

**1. Factorized Embeddings**
```cpp
// Configuration
config.embedding_size = 128;  // Small embedding dimension
config.hidden_size = 768;     // Large hidden dimension

// Result: 83% fewer embedding parameters
// BERT: 30,000 × 768 = 23M
// ALBERT: (30,000 × 128) + (128 × 768) = 3.94M
```

**2. Cross-Layer Parameter Sharing**
```cpp
// Single transformer layer applied N times
// BERT: 12 unique layers × 7M = 84M encoder params
// ALBERT: 1 shared layer × 7M = 7M encoder params
// Result: 91% fewer encoder parameters
```

**3. Sentence-Order Prediction (SOP)**
```cpp
// Better than NSP (Next Sentence Prediction)
// Positive: sentences in correct order
// Negative: sentences in swapped order (same document)
```

---

## T5 Models

### Basic Usage

```cpp
#include "tenzor/models/t5.hpp"
using namespace tenzor::models;

// Create base model (220M parameters)
auto config = T5Config::base();
auto t5 = T5Model(config);

// Forward pass (encoder-decoder)
Variable input_ids(Tensor({2, 64}, DType::Int64, Device::cpu()), false);
Variable decoder_input_ids(Tensor({2, 32}, DType::Int64, Device::cpu()), false);
// ... fill inputs ...

auto outputs = t5.forward(input_ids, decoder_input_ids);
auto encoder_output = outputs.encoder_output;  // [2, 64, 768]
auto decoder_output = outputs.decoder_output;  // [2, 32, 768]
```

### Model Variants

```cpp
// T5-Small: 60M params (d_model=512, L=6, H=8)
auto config = T5Config::small();

// T5-Base: 220M params (d_model=768, L=12, H=12)
auto config = T5Config::base();

// T5-Large: 770M params (d_model=1024, L=24, H=16)
auto config = T5Config::large();

// T5-XL: 3B params (d_model=2048, L=24, H=32)
auto config = T5Config::xl();

// T5-XXL: 11B params (d_model=4096, L=24, H=64)
auto config = T5Config::xxl();
```

### Conditional Generation

```cpp
// For text-to-text tasks
auto gen_model = T5ForConditionalGeneration(config);

Variable input_ids;          // [batch, source_seq_len]
Variable decoder_input_ids;  // [batch, target_seq_len]

// Training: teacher forcing
auto logits = gen_model.forward(input_ids, decoder_input_ids);
// logits: [batch, target_seq_len, vocab_size]

// Generation: autoregressive
auto generated = gen_model.generate(input_ids, max_length=50);
// generated: [batch, 50]
```

### Text-to-Text Tasks

```cpp
// All tasks are text-to-text in T5

// Translation
// Input: "translate English to German: Hello world"
// Output: "Hallo Welt"

// Summarization
// Input: "summarize: [long article text]"
// Output: "[summary text]"

// Classification (as text generation)
// Input: "sentiment: This movie is great!"
// Output: "positive"

// Question Answering
// Input: "question: What is the capital? context: France's capital is Paris."
// Output: "Paris"
```

### Key Features

**1. Relative Position Bias**
```cpp
// No absolute position embeddings
// Instead, adds bias to attention scores based on relative position
// - First layer computes bias
// - Subsequent layers reuse it
// - Logarithmic bucketing for efficiency
```

**2. Pre-Layer Normalization**
```cpp
// Different from BERT/GPT (post-norm)
// BERT: x = LayerNorm(x + Sublayer(x))
// T5:   x = x + Sublayer(LayerNorm(x))

// Benefits:
// - Better gradient flow
// - Easier training
// - More stable for deep models
```

**3. Encoder-Decoder Architecture**
```cpp
// Full Transformer (not encoder-only like BERT or decoder-only like GPT)
class T5Model {
    T5Encoder encoder_;      // Bidirectional attention
    T5Decoder decoder_;      // Causal + cross-attention
    // Shared embeddings between encoder and decoder
};
```

**4. Cross-Attention**
```cpp
// Decoder attends to encoder outputs
// - Self-attention: within decoder sequence (causal)
// - Cross-attention: decoder → encoder outputs
// - Feed-forward network
```

---

## Configuration Options

### ALBERT Config

```cpp
struct AlbertConfig {
    int64_t vocab_size = 30000;
    int64_t embedding_size = 128;        // Key: factorized embedding
    int64_t hidden_size = 768;
    int64_t num_hidden_layers = 12;      // Applied to shared layer
    int64_t num_attention_heads = 12;
    int64_t intermediate_size = 3072;
    double hidden_dropout_prob = 0.1;
    double attention_probs_dropout_prob = 0.1;
    int64_t max_position_embeddings = 512;
    int64_t type_vocab_size = 2;
    double layer_norm_eps = 1e-12;
    std::string hidden_act = "gelu";
};
```

### T5 Config

```cpp
struct T5Config {
    int64_t vocab_size = 32128;
    int64_t d_model = 512;               // Hidden size
    int64_t d_kv = 64;                   // Key/value dim per head
    int64_t d_ff = 2048;                 // FFN intermediate size
    int64_t num_layers = 6;              // Encoder + decoder layers
    int64_t num_heads = 8;
    int64_t relative_attention_num_buckets = 32;
    int64_t relative_attention_max_distance = 128;
    double dropout_rate = 0.1;
    double layer_norm_epsilon = 1e-6;
    std::string dense_act_fn = "relu";   // T5 uses ReLU (not GELU)
    bool is_gated_act = false;
};
```

---

## Parameter Counts

### ALBERT vs BERT

| Model | Total | Embedding | Encoder |
|-------|-------|-----------|---------|
| BERT-base | 110M | 23M | 85M |
| ALBERT-base | 12M | 4M | 7M |
| Reduction | 89% | 83% | 91% |

### T5 Model Sizes

| Variant | Parameters | d_model | Layers | Heads |
|---------|------------|---------|--------|-------|
| Small | 60M | 512 | 6 | 8 |
| Base | 220M | 768 | 12 | 12 |
| Large | 770M | 1024 | 24 | 16 |
| XL | 3B | 2048 | 24 | 32 |
| XXL | 11B | 4096 | 24 | 64 |

---

## Performance Tips

### ALBERT

1. **Memory Efficiency**: Cross-layer sharing reduces memory by 91%
2. **Training Speed**: 1.7× faster than BERT-base
3. **Batch Size**: Can use larger batches due to smaller model
4. **Best Use**: When parameter efficiency is important

### T5

1. **Generation**: Use greedy or beam search for better quality
2. **Caching**: Cache encoder outputs when generating
3. **Mixed Precision**: FP16 can speed up large variants
4. **Best Use**: Text-to-text tasks, multi-task learning

---

## Common Patterns

### ALBERT Classification Example

```cpp
// Create model
auto config = AlbertConfig::base();
auto model = AlbertForSequenceClassification(config, 3);  // 3 classes

// Prepare input
Variable input_ids;  // [32, 128] - batch of 32, seq_len 128
Tensor attention_mask;  // [32, 128] - 1 for real tokens, 0 for padding

// Forward pass
auto logits = model.forward(input_ids, attention_mask);  // [32, 3]

// Compute loss
auto loss = nn::cross_entropy(logits, labels);

// Backward pass
loss.backward();
```

### T5 Translation Example

```cpp
// Create model
auto config = T5Config::base();
auto model = T5ForConditionalGeneration(config);

// Prepare input
Variable src_ids;  // [8, 64] - source sentences
Variable tgt_ids;  // [8, 64] - target sentences

// Training
auto logits = model.forward(src_ids, tgt_ids);  // [8, 64, vocab_size]
auto loss = nn::cross_entropy(logits.view({-1, vocab_size}), tgt_ids.view({-1}));

// Generation
auto generated = model.generate(src_ids, max_length=64, temperature=0.9);
// generated: [8, 64] - generated translations
```

---

## Integration with Tenzor

### Device Compatibility

```cpp
// CPU
auto model = AlbertModel(config);
Variable input(tensor.to(Device::cpu()), true);

// CUDA
auto model_cuda = AlbertModel(config).to(Device::cuda(0));
Variable input_cuda(tensor.to(Device::cuda(0)), true);

// ROCm
auto model_rocm = T5Model(config).to(Device::rocm(0));

// oneAPI
auto model_oneapi = T5Model(config).to(Device::oneapi(0));
```

### Optimization

```cpp
// Mixed Precision Training
nn::amp::GradScaler scaler;
nn::amp::autocast_context ctx(true);

// Forward pass with autocast
auto outputs = model.forward(input_ids);
auto loss = compute_loss(outputs, labels);

// Backward with gradient scaling
scaler.scale(loss).backward();
scaler.step(optimizer);
scaler.update();
```

### Checkpointing

```cpp
// Save model
nn::save(model, "albert_base.pt");

// Load model
auto loaded_model = AlbertModel(config);
nn::load(loaded_model, "albert_base.pt");
```

---

## File Locations

- **ALBERT Header**: `/home/lee/Projects/Tenzor/include/tenzor/models/albert.hpp`
- **ALBERT Source**: `/home/lee/Projects/Tenzor/src/models/albert.cpp`
- **T5 Header**: `/home/lee/Projects/Tenzor/include/tenzor/models/t5.hpp`
- **T5 Source**: `/home/lee/Projects/Tenzor/src/models/t5.cpp`
- **Spec**: `/home/lee/Projects/Tenzor/docs/ADVANCED_NLP_SPEC.md`
- **Summary**: `/home/lee/Projects/Tenzor/docs/ALBERT_T5_IMPLEMENTATION_SUMMARY.md`

---

## Next Steps

1. **Testing**: Write unit tests for key features
2. **Benchmarking**: Compare with reference implementations
3. **Pre-training**: Train on large corpus
4. **Fine-tuning**: Adapt to specific tasks
5. **Optimization**: Profile and optimize critical paths

---

**Quick Reference Complete** ✅
