// tenzor.quantization Python bindings. Extracted from python/bindings.cpp
// as part of P3.4 (incremental split of the ~10k-line monolith).
//
// Covers: quantization schemes, QuantDType, QuantizationParams,
// QuantizedTensor, observers, fake-quantize (QAT), QConfig / QConfigMapping,
// quantized layers (Linear/Conv/BN/Embedding/LSTM/GRU), QATHelper, and the
// fake_quantize_with_grad / fold_bn utility functions.

#include "register.hpp"

#include <pybind11/stl.h>

#include <tenzor/core/tensor.hpp>
#include <tenzor/nn/layers/batchnorm.hpp>
#include <tenzor/nn/layers/conv.hpp>
#include <tenzor/nn/layers/embedding.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/layers/rnn.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/nn/quantization.hpp>

namespace py = pybind11;

namespace tenzor::python {

void register_quantization(py::module_& m) {
    using tenzor::nn::Module;
    using tenzor::Tensor;
    using namespace tenzor::nn::quantization;

    auto quant = m.def_submodule("quantization", "Neural network quantization");

    py::enum_<QuantizationScheme>(quant, "QuantizationScheme")
        .value("PerTensorSymmetric", QuantizationScheme::PerTensorSymmetric)
        .value("PerTensorAsymmetric", QuantizationScheme::PerTensorAsymmetric)
        .value("PerChannelSymmetric", QuantizationScheme::PerChannelSymmetric)
        .value("PerChannelAsymmetric", QuantizationScheme::PerChannelAsymmetric);

    py::enum_<QuantDType>(quant, "QuantDType")
        .value("INT8", QuantDType::INT8, "Signed 8-bit integer [-128, 127]")
        .value("UINT8", QuantDType::UINT8, "Unsigned 8-bit integer [0, 255]")
        .value("INT4", QuantDType::INT4, "Signed 4-bit integer [-8, 7], packed 2 per byte")
        .value("UINT4", QuantDType::UINT4, "Unsigned 4-bit integer [0, 15], packed 2 per byte");

    py::class_<QuantizationParams>(quant, "QuantizationParams")
        .def(py::init<Tensor, Tensor, QuantDType, QuantizationScheme, int64_t>(),
             py::arg("scale"), py::arg("zero_point"), py::arg("dtype"),
             py::arg("scheme"), py::arg("axis") = -1)
        .def_readonly("scale", &QuantizationParams::scale)
        .def_readonly("zero_point", &QuantizationParams::zero_point)
        .def_readonly("dtype", &QuantizationParams::dtype)
        .def_readonly("scheme", &QuantizationParams::scheme)
        .def_readonly("axis", &QuantizationParams::axis);

    py::class_<QuantizedTensor>(quant, "QuantizedTensor")
        .def("dequantize", &QuantizedTensor::dequantize,
             "Dequantize tensor back to floating point")
        .def("data", &QuantizedTensor::data, "Get quantized integer data")
        .def("params", &QuantizedTensor::params, "Get quantization parameters")
        .def("shape", &QuantizedTensor::shape)
        .def("device", &QuantizedTensor::device);

    quant.def("compute_quantization_params", &compute_quantization_params,
        py::arg("min"), py::arg("max"), py::arg("dtype"), py::arg("scheme"),
        "Compute quantization parameters from min/max values");

    quant.def("quantize_tensor", &quantize_tensor,
        py::arg("input"), py::arg("params"),
        "Quantize tensor using specified parameters");

    quant.def("quantize_per_tensor_symmetric", &quantize_per_tensor_symmetric,
        py::arg("input"), py::arg("dtype") = QuantDType::INT8,
        "Symmetric per-tensor quantization (zero-point = 0)");

    quant.def("quantize_per_tensor_asymmetric", &quantize_per_tensor_asymmetric,
        py::arg("input"), py::arg("dtype") = QuantDType::INT8,
        "Asymmetric per-tensor quantization (learnable zero-point)");

    quant.def("quantize_per_channel_symmetric", &quantize_per_channel_symmetric,
        py::arg("input"), py::arg("axis"), py::arg("dtype") = QuantDType::INT8,
        "Symmetric per-channel quantization");

    quant.def("quantize_per_channel_asymmetric", &quantize_per_channel_asymmetric,
        py::arg("input"), py::arg("axis"), py::arg("dtype") = QuantDType::INT8,
        "Asymmetric per-channel quantization");

    quant.def("dequantize_tensor", &dequantize_tensor,
        py::arg("quantized"),
        "Dequantize tensor back to floating point");

    quant.def("compute_quantization_error", &compute_quantization_error,
        py::arg("original"), py::arg("quantized"),
        "Compute quantization error (MSE) between original and quantized tensors");

    quant.def("calibrate_quantization_params", &calibrate_quantization_params,
        py::arg("activations"), py::arg("scheme"),
        py::arg("dtype") = QuantDType::INT8, py::arg("axis") = -1,
        "Calibrate quantization parameters from activation statistics");

    py::class_<Observer, std::shared_ptr<Observer>>(quant, "Observer")
        .def("observe", &Observer::observe, "Observe tensor statistics")
        .def("calculate_qparams", &Observer::calculate_qparams,
             py::arg("dtype"), py::arg("scheme"))
        .def("reset", &Observer::reset);

    py::class_<MinMaxObserver, Observer, std::shared_ptr<MinMaxObserver>>(quant, "MinMaxObserver")
        .def(py::init<>());

    py::class_<MovingAverageMinMaxObserver, Observer,
                std::shared_ptr<MovingAverageMinMaxObserver>>(quant, "MovingAverageMinMaxObserver")
        .def(py::init<float>(), py::arg("averaging_constant") = 0.01f);

    py::class_<HistogramObserver, Observer, std::shared_ptr<HistogramObserver>>(quant, "HistogramObserver")
        .def(py::init<int>(), py::arg("bins") = 2048);

    py::class_<PerChannelHistogramObserver, Observer,
                std::shared_ptr<PerChannelHistogramObserver>>(quant, "PerChannelHistogramObserver")
        .def(py::init<int64_t, int>(), py::arg("axis") = 0, py::arg("bins") = 2048);

    quant.def("make_observer", &make_observer,
        py::arg("scheme"), py::arg("use_histogram") = false, py::arg("axis") = 0,
        "Create observer instance");

    py::class_<FakeQuantize, Module, std::shared_ptr<FakeQuantize>>(quant, "FakeQuantize")
        .def(py::init<QuantDType, QuantizationScheme, bool, bool, int64_t>(),
             py::arg("dtype") = QuantDType::INT8,
             py::arg("scheme") = QuantizationScheme::PerTensorSymmetric,
             py::arg("learnable") = false,
             py::arg("observer_enabled") = true,
             py::arg("axis") = -1,
             "Fake quantization module for QAT")
        .def("enable_observer", &FakeQuantize::enable_observer, py::arg("enabled") = true)
        .def("disable_observer", &FakeQuantize::disable_observer)
        .def("enable_fake_quant", &FakeQuantize::enable_fake_quant, py::arg("enabled") = true)
        .def("disable_fake_quant", &FakeQuantize::disable_fake_quant)
        .def("set_qparams", &FakeQuantize::set_qparams, py::arg("params"));

    py::class_<LearnableFakeQuantize, FakeQuantize,
                std::shared_ptr<LearnableFakeQuantize>>(quant, "LearnableFakeQuantize")
        .def(py::init<QuantDType, QuantizationScheme, int64_t>(),
             py::arg("dtype") = QuantDType::INT8,
             py::arg("scheme") = QuantizationScheme::PerTensorSymmetric,
             py::arg("axis") = -1);

    py::class_<QConfig>(quant, "QConfig");

    py::class_<DefaultQConfigs>(quant, "DefaultQConfigs")
        .def_static("default_qconfig", &DefaultQConfigs::default_qconfig, "Default quantization config")
        .def_static("high_accuracy_qconfig", &DefaultQConfigs::high_accuracy_qconfig, "High accuracy QConfig")
        .def_static("fast_qconfig", &DefaultQConfigs::fast_qconfig, "Fast calibration QConfig")
        .def_static("qat_qconfig", &DefaultQConfigs::qat_qconfig, "QAT QConfig")
        .def_static("uint8_activation_qconfig", &DefaultQConfigs::uint8_activation_qconfig, "UINT8 activation config");

    py::class_<QConfigMapping>(quant, "QConfigMapping")
        .def(py::init<>())
        .def("set_global", &QConfigMapping::set_global)
        .def("set_layer_qconfig", &QConfigMapping::set_layer_qconfig)
        .def("set_type_qconfig", &QConfigMapping::set_type_qconfig)
        .def("get_qconfig", &QConfigMapping::get_qconfig);

    py::class_<QuantizedLinear, Module, std::shared_ptr<QuantizedLinear>>(quant, "QuantizedLinear")
        .def(py::init<int64_t, int64_t, QuantizationParams, float>(),
             py::arg("in_features"), py::arg("out_features"),
             py::arg("weight_qparams"), py::arg("bias_scale") = 1.0f,
             "INT8 quantized linear layer")
        .def("forward_quantized", &QuantizedLinear::forward_quantized,
             py::arg("input"), "Forward pass with quantized input")
        .def("set_weight", &QuantizedLinear::set_weight, py::arg("weights"))
        .def("set_bias", &QuantizedLinear::set_bias, py::arg("bias"))
        .def_static("from_float", &QuantizedLinear::from_float,
             py::arg("fp_linear"), py::arg("qconfig"),
             "Create quantized linear from floating-point layer");

    py::class_<QuantizedConv2d, Module, std::shared_ptr<QuantizedConv2d>>(quant, "QuantizedConv2d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                       QuantizationParams, float>(),
             py::arg("in_channels"), py::arg("out_channels"),
             py::arg("kernel_size"), py::arg("stride") = 1,
             py::arg("padding") = 0, py::arg("dilation") = 1,
             py::arg("groups") = 1, py::arg("weight_qparams"),
             py::arg("bias_scale") = 1.0f,
             "INT8 quantized 2D convolution layer")
        .def("forward_quantized", &QuantizedConv2d::forward_quantized,
             py::arg("input"), "Forward pass with quantized input")
        .def("set_weight", &QuantizedConv2d::set_weight, py::arg("weights"))
        .def("set_bias", &QuantizedConv2d::set_bias, py::arg("bias"))
        .def_static("from_float", &QuantizedConv2d::from_float,
             py::arg("fp_conv"), py::arg("qconfig"),
             "Create quantized conv2d from floating-point layer");

    py::class_<QuantizedConv1d, Module, std::shared_ptr<QuantizedConv1d>>(quant, "QuantizedConv1d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                       QuantizationParams, float>(),
             py::arg("in_channels"), py::arg("out_channels"),
             py::arg("kernel_size"), py::arg("stride") = 1,
             py::arg("padding") = 0, py::arg("dilation") = 1,
             py::arg("groups") = 1, py::arg("weight_qparams"),
             py::arg("bias_scale") = 1.0f,
             "INT8 quantized 1D convolution layer")
        .def("forward_quantized", &QuantizedConv1d::forward_quantized,
             py::arg("input"), "Forward pass with quantized input")
        .def("set_weight", &QuantizedConv1d::set_weight, py::arg("weights"))
        .def("set_bias", &QuantizedConv1d::set_bias, py::arg("bias"))
        .def_static("from_float", &QuantizedConv1d::from_float,
             py::arg("fp_conv"), py::arg("qconfig"),
             "Create quantized conv1d from floating-point layer");

    py::class_<QuantizedConvTranspose2d, Module,
               std::shared_ptr<QuantizedConvTranspose2d>>(quant, "QuantizedConvTranspose2d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                       QuantizationParams, float>(),
             py::arg("in_channels"), py::arg("out_channels"),
             py::arg("kernel_size"), py::arg("stride") = 1,
             py::arg("padding") = 0, py::arg("output_padding") = 0,
             py::arg("groups") = 1, py::arg("weight_qparams"),
             py::arg("bias_scale") = 1.0f,
             "INT8 quantized transposed 2D convolution layer")
        .def("forward_quantized", &QuantizedConvTranspose2d::forward_quantized,
             py::arg("input"), "Forward pass with quantized input")
        .def("set_weight", &QuantizedConvTranspose2d::set_weight, py::arg("weights"))
        .def("set_bias", &QuantizedConvTranspose2d::set_bias, py::arg("bias"))
        .def_static("from_float", &QuantizedConvTranspose2d::from_float,
             py::arg("fp_conv"), py::arg("qconfig"),
             "Create quantized transposed conv2d from floating-point layer");

    py::class_<QuantizedBatchNorm2d, Module,
               std::shared_ptr<QuantizedBatchNorm2d>>(quant, "QuantizedBatchNorm2d")
        .def(py::init<int64_t, Tensor, Tensor>(),
             py::arg("num_features"), py::arg("scale"), py::arg("bias"),
             "Quantized batch normalization with folded parameters")
        .def("forward_quantized", &QuantizedBatchNorm2d::forward_quantized,
             py::arg("input"), "Forward pass with quantized input")
        .def_static("from_float", &QuantizedBatchNorm2d::from_float,
             py::arg("fp_bn"), py::arg("qconfig"),
             "Create quantized batchnorm from floating-point layer");

    py::class_<QuantizedEmbedding, Module,
               std::shared_ptr<QuantizedEmbedding>>(quant, "QuantizedEmbedding")
        .def(py::init<int64_t, int64_t, QuantizationParams, int64_t>(),
             py::arg("num_embeddings"), py::arg("embedding_dim"),
             py::arg("weight_qparams"), py::arg("padding_idx") = -1,
             "INT8 quantized embedding table")
        .def("forward_quantized", &QuantizedEmbedding::forward_quantized,
             py::arg("indices"), "Look up and dequantize embeddings")
        .def("set_weight", &QuantizedEmbedding::set_weight, py::arg("weights"))
        .def_property_readonly("num_embeddings", &QuantizedEmbedding::num_embeddings)
        .def_property_readonly("embedding_dim", &QuantizedEmbedding::embedding_dim)
        .def_static("from_float", &QuantizedEmbedding::from_float,
             py::arg("fp_embedding"), py::arg("qconfig"),
             "Create quantized embedding from floating-point layer");

    py::class_<QuantizedLSTMCell, Module,
               std::shared_ptr<QuantizedLSTMCell>>(quant, "QuantizedLSTMCell")
        .def(py::init<int64_t, int64_t, bool, QuantizationParams>(),
             py::arg("input_size"), py::arg("hidden_size"),
             py::arg("bias") = true,
             py::arg("weight_qparams") = QuantizationParams(
                 Tensor(), Tensor(), QuantDType::INT8, QuantizationScheme::PerTensorSymmetric),
             "INT8 quantized LSTM cell")
        .def("forward_cell", &QuantizedLSTMCell::forward_cell,
             py::arg("input"), py::arg("hx"), py::arg("cx"),
             "Single-step LSTM cell forward with explicit states")
        .def_property_readonly("input_size", &QuantizedLSTMCell::input_size)
        .def_property_readonly("hidden_size", &QuantizedLSTMCell::hidden_size)
        .def_static("from_float", &QuantizedLSTMCell::from_float,
             py::arg("fp_lstm_cell"), py::arg("qconfig"),
             "Create quantized LSTM cell from floating-point layer");

    py::class_<QuantizedLSTM, Module,
               std::shared_ptr<QuantizedLSTM>>(quant, "QuantizedLSTM")
        .def(py::init<int64_t, int64_t, int64_t, bool, bool, bool, float, QuantizationParams>(),
             py::arg("input_size"), py::arg("hidden_size"),
             py::arg("num_layers") = 1, py::arg("bias") = true,
             py::arg("batch_first") = true, py::arg("bidirectional") = false,
             py::arg("dropout") = 0.0f,
             py::arg("weight_qparams") = QuantizationParams(
                 Tensor(), Tensor(), QuantDType::INT8, QuantizationScheme::PerTensorSymmetric),
             "INT8 quantized LSTM")
        .def("forward_with_state", &QuantizedLSTM::forward_with_state,
             py::arg("input"), py::arg("h0"), py::arg("c0"),
             "Forward with explicit initial states")
        .def_static("from_float", &QuantizedLSTM::from_float,
             py::arg("fp_lstm"), py::arg("qconfig"),
             "Create quantized LSTM from floating-point layer");

    py::class_<QuantizedGRU, Module,
               std::shared_ptr<QuantizedGRU>>(quant, "QuantizedGRU")
        .def(py::init<int64_t, int64_t, int64_t, bool, bool, bool, QuantizationParams>(),
             py::arg("input_size"), py::arg("hidden_size"),
             py::arg("num_layers") = 1, py::arg("bias") = true,
             py::arg("batch_first") = true, py::arg("bidirectional") = false,
             py::arg("weight_qparams") = QuantizationParams(
                 Tensor(), Tensor(), QuantDType::INT8, QuantizationScheme::PerTensorSymmetric),
             "INT8 quantized GRU")
        .def("forward_with_state", &QuantizedGRU::forward_with_state,
             py::arg("input"), py::arg("h0"),
             "Forward with explicit initial hidden state")
        .def_static("from_float", &QuantizedGRU::from_float,
             py::arg("fp_gru"), py::arg("qconfig"),
             "Create quantized GRU from floating-point layer");

    py::class_<QATHelper>(quant, "QATHelper",
        "Quantization-aware training helper for model preparation and conversion")
        .def(py::init<>())
        .def("prepare_qat", &QATHelper::prepare_qat,
             py::arg("model"),
             py::arg("dtype") = QuantDType::INT8,
             py::arg("scheme") = QuantizationScheme::PerTensorSymmetric,
             py::arg("learnable") = false,
             "Prepare model for quantization-aware training")
        .def("enable_observer", &QATHelper::enable_observer,
             "Enable observers for all fake quantize modules")
        .def("disable_observer", &QATHelper::disable_observer,
             "Disable observers and fix quantization parameters")
        .def("freeze_bn_stats", &QATHelper::freeze_bn_stats,
             "Freeze BN statistics and calculate final qparams")
        .def("convert_to_quantized", &QATHelper::convert_to_quantized,
             py::arg("model"),
             "Convert QAT model to quantized inference model");

    quant.def("fake_quantize_with_grad", &fake_quantize_with_grad,
        py::arg("input"), py::arg("scale"), py::arg("zero_point"),
        py::arg("quant_min"), py::arg("quant_max"),
        "Apply fake quantization with autograd support (STE backward)");

    quant.def("fold_bn", &fold_bn,
        py::arg("model"),
        "Fold BatchNorm2d into preceding Conv2d layers");
}

} // namespace tenzor::python
