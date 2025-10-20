# RoBERTa and ELECTRA Quick Reference

## File Locations

### RoBERTa
- **Header**: `/home/lee/Projects/Tenzor/include/tenzor/models/roberta.hpp` (366 lines)
- **Implementation**: `/home/lee/Projects/Tenzor/src/models/roberta.cpp` (232 lines)
- **Tests**: `/home/lee/Projects/Tenzor/tests/unit/test_roberta.cpp` (329 lines)
- **Total**: 927 lines of code

### ELECTRA
- **Header**: `/home/lee/Projects/Tenzor/include/tenzor/models/electra.hpp` (487 lines)
- **Implementation**: `/home/lee/Projects/Tenzor/src/models/electra.cpp` (376 lines)
- **Tests**: `/home/lee/Projects/Tenzor/tests/unit/test_electra.cpp` (518 lines)
- **Total**: 1,381 lines of code

## Quick Usage

### RoBERTa Base Model

```cpp
#include <tenzor/models/roberta.hpp>
using namespace tenzor::models;

// Create base model (125M params)
auto config = RobertaConfig::base();
RobertaModel model(config);

// Forward pass
Tensor input({2, 128}, DType::Int64, Device::cpu());
Variable input_var(input, true);
auto outputs = model.forward(input_var);
// outputs.sequence_output: [2, 128, 768]
// outputs.pooled_output: [2, 768]
```

### RoBERTa Classification

```cpp
// Binary classification
auto config = RobertaConfig::base();
RobertaForSequenceClassification classifier(config, 2);

auto logits = classifier.forward(input_var);  // [batch, 2]
```

### ELECTRA Generator

```cpp
#include <tenzor/models/electra.hpp>
using namespace tenzor::models;

auto config = ElectraConfig::base();
ElectraGenerator generator(config);

auto logits = generator.forward(input_var);  // [batch, seq_len, vocab_size]
```

### ELECTRA Discriminator

```cpp
auto config = ElectraConfig::base();
ElectraDiscriminator discriminator(config);

auto logits = discriminator.forward(input_var);  // [batch, seq_len]
// Binary classification per token (real vs replaced)
```

### ELECTRA Pre-Training

```cpp
auto config = ElectraConfig::base();
ElectraForPreTraining model(config);

// Prepare inputs
Tensor input_ids({32, 128}, DType::Int64, Device::cpu());
Tensor masked_positions({32, 128}, DType::Int64, Device::cpu());
Tensor original_tokens({32, 128}, DType::Int64, Device::cpu());

Variable input_var(input_ids, true);

// Forward
auto outputs = model.forward(input_var, masked_positions, original_tokens);

// Compute loss
auto loss = model.compute_loss(
    outputs.gen_logits,
    outputs.disc_logits,
    outputs.is_replaced,
    masked_positions,
    original_tokens
);

loss.backward();
```

## Model Configurations

### RoBERTa Variants

```cpp
// Base: 125M params, 12 layers, 768 hidden, 12 heads
auto base = RobertaConfig::base();

// Large: 355M params, 24 layers, 1024 hidden, 16 heads
auto large = RobertaConfig::large();
```

### ELECTRA Variants

```cpp
// Small: 17M params total (14M disc + 3M gen)
auto small = ElectraConfig::small();

// Base: 143M params total (110M disc + 33M gen)
auto base = ElectraConfig::base();

// Large: 415M params total (335M disc + 80M gen)
auto large = ElectraConfig::large();
```

## Key Differences

### RoBERTa vs BERT

| Feature | BERT | RoBERTa |
|---------|------|---------|
| Vocab Size | 30,522 | 50,265 |
| Tokenization | WordPiece | Byte-level BPE |
| NSP Task | Yes | No |
| Segments | 2 | 1 |
| LayerNorm ε | 1e-12 | 1e-5 |
| Max Positions | 512 | 514 |

### ELECTRA vs BERT

| Feature | BERT | ELECTRA |
|---------|------|---------|
| Architecture | Encoder only | Generator + Discriminator |
| Pre-training | MLM (15% tokens) | RTD (100% tokens) |
| Sample Efficiency | 1x | 4x |
| Task | Predict masked | Detect replaced |
| Fine-tuning | Full model | Discriminator only |

## Available Task Heads

### RoBERTa
- `RobertaForSequenceClassification` - Text classification, sentiment
- `RobertaForTokenClassification` - NER, POS tagging
- `RobertaForQuestionAnswering` - Extractive QA

### ELECTRA
- `ElectraForSequenceClassification` - Text classification
- `ElectraForTokenClassification` - NER, POS tagging
- `ElectraForQuestionAnswering` - Extractive QA
- `ElectraForPreTraining` - Pre-training (both models)

## Testing

```bash
# Build
cd build
cmake ..
make test_roberta test_electra

# Run
./tests/test_roberta
./tests/test_electra

# With CTest
ctest -R roberta -V
ctest -R electra -V
```

## Common Patterns

### Creating Models

```cpp
// Always specify config explicitly
auto config = RobertaConfig::base();
RobertaModel model(config);

// Customize config
auto custom_config = RobertaConfig::base();
custom_config.num_hidden_layers = 6;  // Smaller model
custom_config.hidden_dropout_prob = 0.2;  // More regularization
RobertaModel small_model(custom_config);
```

### Gradient Flow

```cpp
// Enable gradients on input
Variable input_var(input_tensor, true);

// Forward
auto outputs = model.forward(input_var);

// Compute loss
auto loss = criterion(outputs, labels);

// Backward
loss.backward();

// Access gradients
auto input_grad = input_var.grad();
```

### Memory Management

```cpp
// Models use shared_ptr for submodules
// No manual memory management needed

{
    RobertaModel model(config);
    // ... use model ...
}  // Automatically cleaned up
```

## Performance Tips

1. **Batch Size**: Larger batches = better GPU utilization
   - Typical: 16-32 for training, 64-128 for inference

2. **Sequence Length**: Shorter sequences = faster
   - Pad to multiple of 8 for tensor cores

3. **Mixed Precision**: Use AMP for faster training
   - ELECTRA benefits more (larger models)

4. **Gradient Checkpointing**: For large models
   - Trade compute for memory

5. **DataParallel**: Multi-GPU training
   - RoBERTa-large: 2-4 GPUs recommended
   - ELECTRA-large: 4-8 GPUs recommended

## Integration with Existing Code

### Using with BERT Code

```cpp
// RoBERTa can often be used as drop-in replacement for BERT
// Just change the model type:

// Before:
BertForSequenceClassification bert_classifier(bert_config, 2);

// After:
RobertaForSequenceClassification roberta_classifier(roberta_config, 2);

// Same API, same usage
```

### Converting Configs

```cpp
// RoBERTa to BERT config
auto roberta_config = RobertaConfig::base();
auto bert_config = roberta_config.to_bert_config();

// ELECTRA to BERT configs
auto electra_config = ElectraConfig::base();
auto disc_config = electra_config.to_discriminator_config();
auto gen_config = electra_config.to_generator_config();
```

## Error Handling

```cpp
// Models throw std::runtime_error on invalid inputs
try {
    auto outputs = model.forward(input_ids);
} catch (const std::runtime_error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
}

// Common errors:
// - Shape mismatch (batch size, seq length)
// - Invalid token IDs (>= vocab_size)
// - Missing required inputs
```

## Best Practices

1. **Configuration**:
   - Always use factory methods (`base()`, `large()`)
   - Modify config before model creation
   - Don't change config after model creation

2. **Input Preparation**:
   - Validate token IDs are within vocab range
   - Use proper padding and attention masks
   - Batch similar-length sequences together

3. **Training**:
   - Use learning rate warmup (1-10% of training)
   - Gradient clipping (max_norm=1.0)
   - Regular checkpointing
   - Monitor validation loss

4. **Fine-tuning**:
   - Lower learning rate than pre-training (1e-5 to 5e-5)
   - Few epochs (2-4)
   - Task-specific hyperparameters

## Code Reuse

### Inheriting from BERT

```cpp
// RoBERTa components inherit from BERT
class RobertaEmbeddings : public BertEmbeddings {
    // Same implementation, different config
};

class RobertaEncoder : public BertEncoder {
    // 100% reuse
};
```

### Composing BERT Models

```cpp
// ELECTRA composes BERT models
class ElectraGenerator {
    std::shared_ptr<BertModel> generator_;  // Small BERT
    std::shared_ptr<nn::Linear> lm_head_;
};

class ElectraDiscriminator {
    std::shared_ptr<BertModel> discriminator_;  // Full BERT
    std::shared_ptr<nn::Linear> classifier_;
};
```

## Summary

- **RoBERTa**: Drop-in BERT replacement with better performance
- **ELECTRA**: Novel pre-training approach, 4x more efficient
- **Code Reuse**: 95%+ from existing BERT implementation
- **Testing**: 33 test cases, comprehensive coverage
- **Documentation**: Full Doxygen comments, usage examples
- **Integration**: Seamless with existing Tenzor infrastructure

For detailed implementation notes, see `ROBERTA_ELECTRA_IMPLEMENTATION.md`.
