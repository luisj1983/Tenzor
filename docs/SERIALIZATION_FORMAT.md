# Tenzor Serialization Format

## Overview

Tenzor uses a simple binary format for serializing model weights and optimizer state. The format is designed to be efficient, portable, and easy to implement.

## File Format Specification

### File Header

```
Offset | Size (bytes) | Type     | Description
-------|--------------|----------|----------------------------------
0x00   | 4            | uint32_t | Magic number (0x544E5A52 "TNZR")
0x04   | 4            | uint32_t | Format version (currently 1)
0x08   | 4            | uint32_t | Number of tensors (N)
```

### Tensor Entry (repeated N times)

For each tensor in the state dictionary:

```
Offset | Size (bytes) | Type     | Description
-------|--------------|----------|----------------------------------
0x00   | 4            | uint32_t | Name length (L)
0x04   | L            | char[]   | Tensor name (UTF-8)
0x04+L | 1            | uint8_t  | Data type (DType enum value)
0x05+L | 4            | uint32_t | Number of dimensions (D)
0x09+L | 8*D          | int64_t[]| Shape dimensions
0x09+L+8*D | variable | bytes    | Raw tensor data
```

### Data Types

The DType enum values are serialized as uint8_t:

```cpp
enum class DType : uint8_t {
    Float32 = 0,
    Float64 = 1,
    Float16 = 2,
    BFloat16 = 3,
    Int8 = 4,
    Int16 = 5,
    Int32 = 6,
    Int64 = 7,
    UInt8 = 8,
    UInt16 = 9,
    UInt32 = 10,
    UInt64 = 11,
    Bool = 12,
    Complex64 = 13,
    Complex128 = 14
};
```

## Example File Structure

For a simple Linear layer with weight and bias:

```
[Magic: 0x544E5A52]
[Version: 1]
[Num Tensors: 2]

[Name Length: 6]
[Name: "weight"]
[DType: 0 (Float32)]
[NDim: 2]
[Shape: [3, 4]]
[Data: 48 bytes (3*4*4 bytes)]

[Name Length: 4]
[Name: "bias"]
[DType: 0 (Float32)]
[NDim: 1]
[Shape: [3]]
[Data: 12 bytes (3*4 bytes)]
```

## Usage Examples

### Saving a Model

```cpp
#include "tenzor/nn/layers/linear.hpp"

auto model = std::make_shared<Linear>(4, 3);

// Train your model...

// Save model weights
model->save("model.bin");

// Or use state dict
auto state = model->state_dict();
Serializer::save(state, "model.bin");
```

### Loading a Model

```cpp
#include "tenzor/nn/layers/linear.hpp"

auto model = std::make_shared<Linear>(4, 3);

// Load model weights
model->load("model.bin");

// Or use state dict
auto state = Serializer::load("model.bin");
model->load_state_dict(state);
```

### Saving Optimizer State

```cpp
#include "tenzor/nn/optim/adam.hpp"

auto model = std::make_shared<Linear>(4, 3);
auto params = model->parameters();
optim::Adam optimizer(params, 0.001);

// Train for some epochs...

// Save optimizer state
optimizer.save_state("optimizer.bin");
```

### Loading Optimizer State

```cpp
auto model = std::make_shared<Linear>(4, 3);
auto params = model->parameters();
optim::Adam optimizer(params, 0.001);

// Load optimizer state
optimizer.load_state("optimizer.bin");

// Continue training...
```

## Design Decisions

### 1. Simple Binary Format

We chose a simple binary format over more complex formats like HDF5 or Protocol Buffers for several reasons:

- **No external dependencies**: The implementation uses only standard C++ library
- **Small file size**: No metadata overhead or compression
- **Fast I/O**: Sequential reads and writes with minimal parsing
- **Easy to implement**: Simple format is easier to debug and extend

### 2. CPU-Only Serialization

Tensors are always moved to CPU before serialization:

- **Portability**: Files can be loaded on any device
- **Cross-platform**: No GPU-specific format concerns
- **Simplicity**: No need to handle device-specific memory layouts

### 3. Endianness

The current implementation uses native endianness. For cross-platform compatibility between systems with different endianness, you may need to:

- Add byte-swapping logic
- Store endianness flag in header
- Convert to little-endian (most common) on save

### 4. State Dictionary Format

State dictionaries use hierarchical naming:

```
weight                  # Top-level parameter
bias                    # Top-level parameter
layer1.weight          # Submodule parameter
layer1.bias            # Submodule parameter
layer2.sublayer.weight # Nested submodule parameter
```

## Optimizer State Format

Optimizers save additional metadata as scalar tensors:

```
step_count      # int64_t scalar tensor
lr              # float64 scalar tensor
beta1           # float64 scalar tensor (Adam)
beta2           # float64 scalar tensor (Adam)
eps             # float64 scalar tensor (Adam)
weight_decay    # float64 scalar tensor (Adam)
exp_avg_0       # First moment estimate for parameter 0
exp_avg_sq_0    # Second moment estimate for parameter 0
exp_avg_1       # First moment estimate for parameter 1
exp_avg_sq_1    # Second moment estimate for parameter 1
...
```

## Error Handling

The serialization system handles several error cases:

- **Invalid magic number**: File is not a valid Tenzor file
- **Unsupported version**: File was created with incompatible version
- **Shape mismatch**: Cannot load tensor with different shape
- **DType mismatch**: Cannot load tensor with different data type
- **Missing file**: File does not exist
- **Corrupted file**: Unexpected end of file or invalid data

## Future Enhancements

Potential improvements for future versions:

1. **Compression**: Add optional zlib/lz4 compression
2. **Checksums**: Add CRC32 or SHA256 for corruption detection
3. **Versioning**: Support loading older format versions
4. **Partial Loading**: Load only specific parameters
5. **Memory Mapping**: Support mmap for large models
6. **Sparse Tensors**: Add support for sparse tensor formats
7. **Metadata**: Store training metadata (epoch, accuracy, etc.)

## Compatibility

- **Format Version**: 1
- **Minimum Tenzor Version**: 0.1.0
- **Supported DTypes**: Float32, Float64, Int32, Int64, UInt8, Bool
- **Maximum Dimensions**: Unlimited (limited by memory)
- **Maximum Name Length**: 4,294,967,295 bytes (uint32_t max)
- **Maximum Tensors**: 4,294,967,295 (uint32_t max)

## Performance Characteristics

- **Save Speed**: ~1-2 GB/s (sequential write)
- **Load Speed**: ~1-2 GB/s (sequential read)
- **File Size**: Sum of tensor sizes + ~20 bytes overhead per tensor
- **Memory Usage**: Peak usage is 2x tensor size during save/load

## Testing

Comprehensive tests are provided in `tests/nn/test_serialization.cpp`:

- Basic save/load functionality
- Different data types
- Shape and dtype validation
- Error handling
- Large tensors
- Sequential models
- Optimizer state
- Round-trip computation

Run tests with:

```bash
cd build
ctest -R test_serialization -V
```
