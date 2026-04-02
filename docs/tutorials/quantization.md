# Quantization

Reduce model size and improve inference speed with INT8/INT4 quantization.

## Post-Training Quantization (PTQ)

Quantize a trained model without retraining:

```python
import tenzor as tz
Q = tz.nn.quantization

# Create observer to calibrate quantization parameters
obs = Q.MinMaxObserver(Q.QuantDType.INT8, Q.QuantizationScheme.PerTensorSymmetric)

# Run calibration data through the observer
for batch in calibration_loader:
    obs.observe(model(batch).data)

# Get quantization parameters
qparams = obs.calculate_qparams()
```

## Quantization-Aware Training (QAT)

Insert fake-quantize nodes during training for better accuracy:

```python
# Prepare model for QAT
qat_helper = Q.QATHelper()
qat_model = qat_helper.prepare_qat(model, dtype=Q.QuantDType.INT8)

# Train with fake quantization
for epoch in range(num_epochs):
    for data, target in train_loader:
        output = qat_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Freeze BN stats and convert to quantized model
qat_helper.freeze_bn_stats()
quantized_model = qat_helper.convert_to_quantized(qat_model)
```

## Quantized Layers

Use pre-built quantized layers for efficient inference:

```python
# Quantized linear layer
ql = Q.QuantizedLinear(in_features=512, out_features=256)

# Quantized convolution
qc = Q.QuantizedConv2d(in_channels=3, out_channels=64, kernel_size=3)

# Fused quantized operations
qcr = Q.QuantizedConv2dReLU(in_channels=3, out_channels=64, kernel_size=3)
```

## Supported Schemes

| Scheme | Description | Best For |
|--------|-------------|----------|
| PerTensorSymmetric | Single scale, zero_point=0 | Weights |
| PerTensorAsymmetric | Scale + zero_point per tensor | Activations |
| PerChannelSymmetric | Scale per output channel | Conv weights |
| PerChannelAsymmetric | Scale + zero_point per channel | Fine-grained |

## Supported Data Types

- **INT8**: Standard 8-bit quantization (most hardware support)
- **UINT8**: Unsigned 8-bit (activations)
- **INT4**: 4-bit quantization (LLM weight compression)
- **UINT4**: Unsigned 4-bit
