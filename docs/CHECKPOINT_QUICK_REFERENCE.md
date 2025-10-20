# Checkpoint Quick Reference Guide

Quick reference for using gradient checkpointing and model checkpointing in Tenzor.

---

## Gradient Checkpointing

### Basic Usage

```cpp
#include "tenzor/autograd/checkpoint.hpp"

// Single input/output
auto y = checkpoint([](const Variable& x) {
    return layer->forward(x);
}, input);

// Multiple inputs/outputs
auto outputs = checkpoint([](const std::vector<Variable>& inputs) {
    return {compute(inputs[0], inputs[1])};
}, {x, y});

// For leaf variables, use TENZOR_CHECKPOINT macro
Variable x(tensor, true);
auto y = TENZOR_CHECKPOINT([](const Variable& in) {
    return in * 2.0f;
}, x);
```

### Nested Checkpointing

```cpp
auto y = checkpoint([&](const Variable& input) {
    // Inner checkpoint
    auto intermediate = checkpoint([](const Variable& in) {
        return in * 3.0f;
    }, input);
    return intermediate + 1.0f;
}, x);
```

### Statistics and Monitoring

```cpp
// Context manager
{
    CheckpointContext ctx;
    // ... checkpointed code ...
    auto stats = ctx.get_stats();
    std::cout << "Checkpoints: " << stats.num_checkpoints << "\n";
    std::cout << "Memory saved: " << stats.saved_memory_bytes << " bytes\n";
}

// Memory tracking
MemoryTracker::start_tracking();
// ... code ...
std::cout << "Peak: " << MemoryTracker::peak_memory() << " bytes\n";
MemoryTracker::stop_tracking();

// Global stats
auto& stats = get_checkpoint_stats();
reset_checkpoint_stats();
```

### Enable/Disable

```cpp
set_checkpoint_enabled(false);  // Disable checkpointing
set_checkpoint_enabled(true);   // Enable checkpointing
bool enabled = is_checkpoint_enabled();
```

---

## Model Checkpointing

### Basic Save/Load

```cpp
#include "tenzor/nn/checkpoint.hpp"

ModelCheckpoint checkpoint;

// Save model only
checkpoint.save_model("model.pt", model);

// Load model only
auto state = checkpoint.load_model("model.pt");
model.load_state_dict(state);
```

### Full Checkpoint (Model + Optimizer + Scheduler)

```cpp
// Save everything
checkpoint.save(
    "checkpoint.pt",
    model,
    &optimizer,
    &scheduler,
    metadata
);

// Load everything
auto loaded = checkpoint.load("checkpoint.pt");
model.load_state_dict(loaded.model_state);
optimizer.load_state_dict(loaded.optimizer_state);
// scheduler->load_state_dict(loaded.scheduler_state);
```

### Training Metadata

```cpp
TrainingMetadata metadata;
metadata.epoch = 10;
metadata.global_step = 1000;
metadata.learning_rate = 0.001;
metadata.train_loss = 0.25;
metadata.val_loss = 0.30;
metadata.custom_metrics["f1_score"] = 0.81;

checkpoint.save_model("model.pt", model, metadata);

// Read metadata without loading full checkpoint
auto meta = checkpoint.get_metadata("model.pt");
std::cout << "Epoch: " << meta.epoch << "\n";
```

### Checkpoint Verification

```cpp
// Verify integrity
if (!checkpoint.verify_checkpoint("model.pt")) {
    std::cerr << "Corrupted checkpoint\n";
    return;
}

// Check version
uint32_t version = checkpoint.get_version("model.pt");
if (!checkpoint.is_compatible("model.pt")) {
    std::cerr << "Incompatible version\n";
    return;
}
```

### Configuration Options

```cpp
CheckpointConfig config;
config.compression = CompressionType::None;  // or LZ4, Zstd, Gzip
config.save_optimizer = true;
config.save_scheduler = true;
config.verify_checksum = true;
config.atomic_save = true;  // Use atomic writes (recommended)

ModelCheckpoint checkpoint(config);
```

---

## AutoCheckpoint

### Setup

```cpp
// Keep top 5 checkpoints, save every 1 epoch
AutoCheckpoint auto_checkpoint("./checkpoints", 5, 1);

// Minimize metric (for loss)
auto_checkpoint.set_metric_mode("min");

// Maximize metric (for accuracy)
auto_checkpoint.set_metric_mode("max");
```

### Training Loop

```cpp
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    double train_loss = train_epoch(model, optimizer);
    double val_loss = validate(model);

    // Automatically save and manage checkpoints
    bool saved = auto_checkpoint.step(
        model,
        optimizer,
        epoch,
        val_loss,
        "val_loss",
        &scheduler  // optional
    );

    if (saved) {
        std::cout << "Checkpoint saved\n";
    }
}
```

### Retrieve Best Checkpoint

```cpp
// Get best checkpoint path
std::string best = auto_checkpoint.best_checkpoint_path();
double best_metric = auto_checkpoint.best_metric_value();

// Load best checkpoint
ModelCheckpoint checkpoint;
auto loaded = checkpoint.load(best);
model.load_state_dict(loaded.model_state);

// List all checkpoint paths
auto paths = auto_checkpoint.checkpoint_paths();
```

### Manual Cleanup

```cpp
auto_checkpoint.cleanup();  // Remove old checkpoints
```

---

## Common Patterns

### Training with Checkpointing

```cpp
#include "tenzor/autograd/checkpoint.hpp"
#include "tenzor/nn/checkpoint.hpp"

// Setup
Linear model(128, 64);
auto params = model.parameters();
optim::Adam optimizer(params, 0.001);
AutoCheckpoint auto_checkpoint("./checkpoints", 5);
auto_checkpoint.set_metric_mode("min");

// Training loop
for (int epoch = 0; epoch < 100; ++epoch) {
    // Forward pass with gradient checkpointing
    for (auto& batch : train_loader) {
        auto output = checkpoint([&](const Variable& x) {
            return model.forward(x);
        }, batch.input);

        auto loss = criterion(output, batch.target);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    // Validation
    double val_loss = validate(model);

    // Save checkpoint
    auto_checkpoint.step(model, optimizer, epoch, val_loss, "val_loss");
}

// Load best model
std::string best = auto_checkpoint.best_checkpoint_path();
ModelCheckpoint checkpoint;
auto loaded = checkpoint.load(best);
model.load_state_dict(loaded.model_state);
```

### Memory-Efficient Deep Model

```cpp
// Checkpoint each layer block
Variable hidden = input;
for (int i = 0; i < num_layers; ++i) {
    hidden = checkpoint([&, i](const Variable& x) {
        auto attn = attention_layers[i]->forward(x);
        auto ffn = ffn_layers[i]->forward(attn);
        return ffn;
    }, hidden);
}
```

### Checkpoint with Statistics

```cpp
CheckpointContext ctx;

// Training epoch with checkpointing
for (auto& batch : train_loader) {
    auto output = checkpoint([&](const Variable& x) {
        return model.forward(x);
    }, batch.input);
    // ... backward pass ...
}

// Print statistics
auto stats = ctx.get_stats();
std::cout << "Checkpoints: " << stats.num_checkpoints << "\n";
std::cout << "Recomputations: " << stats.num_recomputations << "\n";
std::cout << "Memory saved: " << stats.saved_memory_bytes / 1024 / 1024 << " MB\n";
std::cout << "Recompute time: " << stats.total_recompute_time_ms << " ms\n";
```

### Resume Training from Checkpoint

```cpp
ModelCheckpoint checkpoint;
std::string resume_path = "checkpoint_epoch_50.pt";

if (std::filesystem::exists(resume_path)) {
    auto loaded = checkpoint.load(resume_path);

    // Restore model
    model.load_state_dict(loaded.model_state);

    // Restore optimizer
    optimizer.load_state_dict(loaded.optimizer_state);

    // Restore training state
    int start_epoch = loaded.metadata.epoch + 1;
    double best_loss = loaded.metadata.best_val_loss;

    std::cout << "Resuming from epoch " << start_epoch << "\n";

    // Continue training
    for (int epoch = start_epoch; epoch < num_epochs; ++epoch) {
        // ... training loop ...
    }
}
```

---

## Performance Tips

### Gradient Checkpointing

1. **Checkpoint large blocks**: ResNet blocks, transformer layers
2. **Don't checkpoint small ops**: Element-wise operations, activations alone
3. **Use nested checkpoints**: For very deep models (100+ layers)
4. **Monitor overhead**: Use `CheckpointStats` to track compute cost
5. **Disable during inference**: `set_checkpoint_enabled(false)`

### Model Checkpointing

1. **Use atomic saves**: Prevents corruption on crashes
2. **Save less frequently**: Every N epochs, not every step
3. **Use AutoCheckpoint**: Automatic cleanup and best model tracking
4. **Verify critical checkpoints**: Use `verify_checkpoint()` before deployment
5. **Compress for storage**: Use compression for archival (when implemented)

---

## Error Handling

### Gradient Checkpointing

```cpp
try {
    auto y = checkpoint(fn, x);
    loss.backward();
} catch (const std::runtime_error& e) {
    std::cerr << "Checkpoint error: " << e.what() << "\n";
}
```

### Model Checkpointing

```cpp
try {
    checkpoint.save("model.pt", model);
} catch (const std::runtime_error& e) {
    std::cerr << "Save failed: " << e.what() << "\n";
}

try {
    auto loaded = checkpoint.load("model.pt");
} catch (const std::runtime_error& e) {
    std::cerr << "Load failed: " << e.what() << "\n";
}
```

---

## Testing Your Checkpoints

### Gradient Checkpointing

```cpp
// Test checkpoint correctness
Variable x(randn({3, 3}), true);

// Without checkpoint
auto y1 = fn(x);
auto loss1 = sum(y1);
loss1.backward();
auto grad1 = x.grad().value();

x.zero_grad();

// With checkpoint
auto y2 = checkpoint(fn, x);
auto loss2 = sum(y2);
loss2.backward();
auto grad2 = x.grad().value();

// Gradients should match
assert(allclose(grad1, grad2));
```

### Model Checkpointing

```cpp
// Test save/load roundtrip
ModelCheckpoint checkpoint;
checkpoint.save("test.pt", model);
auto loaded_state = checkpoint.load_model("test.pt");

// Verify parameter values match
auto original_params = model.parameters();
for (const auto& [name, loaded_param] : loaded_state) {
    auto original_param = model.state_dict()[name];
    assert(allclose(original_param, loaded_param));
}
```

---

## API Reference

### Gradient Checkpointing Functions

```cpp
// Free functions
auto checkpoint(fn, inputs) -> std::vector<Variable>;
auto checkpoint(fn, input) -> Variable;
auto checkpoint_with_originals(fn, inputs, originals) -> std::vector<Variable>;
auto checkpoint_with_original(fn, input, original) -> Variable;

// Macro
TENZOR_CHECKPOINT(fn, input)

// Statistics
auto get_checkpoint_stats() -> CheckpointStats&;
auto reset_checkpoint_stats() -> void;
auto is_checkpoint_enabled() -> bool;
auto set_checkpoint_enabled(bool) -> void;

// Classes
CheckpointFunction
CheckpointContext
MemoryTracker
CheckpointSegment
```

### Model Checkpointing Classes

```cpp
// Main class
ModelCheckpoint
  - save()
  - load()
  - save_model()
  - load_model()
  - verify_checkpoint()
  - get_metadata()
  - get_version()
  - is_compatible()

// Auto checkpoint
AutoCheckpoint
  - step()
  - set_metric_mode()
  - best_checkpoint_path()
  - best_metric_value()
  - checkpoint_paths()
  - cleanup()

// Data structures
CheckpointConfig
TrainingMetadata
Checkpoint
CompressionType
```

---

## Troubleshooting

### Gradient Checkpointing

**Problem**: Gradients don't accumulate correctly for leaf variables
**Solution**: Use `checkpoint_with_original()` or `TENZOR_CHECKPOINT` macro

**Problem**: High memory usage despite checkpointing
**Solution**: Check checkpoint placement, ensure large activations are checkpointed

**Problem**: Slow backward pass
**Solution**: Reduce checkpoint frequency, checkpoint only expensive operations

### Model Checkpointing

**Problem**: Checkpoint file corrupted
**Solution**: Enable `atomic_save` and `verify_checksum` in config

**Problem**: Load fails with version error
**Solution**: Check compatibility with `is_compatible()` before loading

**Problem**: Missing optimizer state after load
**Solution**: Ensure `save_optimizer=true` in config when saving

---

## Additional Resources

- Full implementation report: `docs/CHECKPOINT_IMPLEMENTATION_REPORT.md`
- Test examples: `tests/unit/test_gradient_checkpoint.cpp`, `tests/unit/test_model_checkpoint.cpp`
- Header files: `include/tenzor/autograd/checkpoint.hpp`, `include/tenzor/nn/checkpoint.hpp`

---

Last Updated: 2025-10-16
Tenzor Version: 1.0.0
