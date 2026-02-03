/**
 * @file op_id.cpp
 * @brief Implementation of OpId string conversion for error messages
 */

#include "tenzor/ops/op_id.hpp"
#include <array>

namespace tenzor {

namespace {

// Static array of operation names indexed by OpId
// Used only for error messages and debugging
constexpr std::array<std::string_view, OP_COUNT> op_names = []() {
    std::array<std::string_view, OP_COUNT> names{};

    // Initialize all to "unknown" first
    for (auto& name : names) {
        name = "unknown";
    }

    // Arithmetic
    names[static_cast<size_t>(OpId::Add)] = "add";
    names[static_cast<size_t>(OpId::Sub)] = "sub";
    names[static_cast<size_t>(OpId::Mul)] = "mul";
    names[static_cast<size_t>(OpId::Div)] = "div";
    names[static_cast<size_t>(OpId::MatMul)] = "matmul";
    names[static_cast<size_t>(OpId::Bmm)] = "bmm";
    names[static_cast<size_t>(OpId::Dot)] = "dot";
    names[static_cast<size_t>(OpId::AddInplace)] = "add_inplace";
    names[static_cast<size_t>(OpId::SubInplace)] = "sub_inplace";
    names[static_cast<size_t>(OpId::MulInplace)] = "mul_inplace";
    names[static_cast<size_t>(OpId::DivInplace)] = "div_inplace";

    // Reductions
    names[static_cast<size_t>(OpId::Sum)] = "sum";
    names[static_cast<size_t>(OpId::Mean)] = "mean";
    names[static_cast<size_t>(OpId::Max)] = "max";
    names[static_cast<size_t>(OpId::Min)] = "min";
    names[static_cast<size_t>(OpId::ArgMax)] = "argmax";
    names[static_cast<size_t>(OpId::ArgMin)] = "argmin";
    names[static_cast<size_t>(OpId::Prod)] = "prod";
    names[static_cast<size_t>(OpId::Var)] = "var";
    names[static_cast<size_t>(OpId::Std)] = "std";
    names[static_cast<size_t>(OpId::Norm)] = "norm";
    names[static_cast<size_t>(OpId::ArgSort)] = "argsort";

    // Element-wise math
    names[static_cast<size_t>(OpId::Sqrt)] = "sqrt";
    names[static_cast<size_t>(OpId::Neg)] = "neg";
    names[static_cast<size_t>(OpId::Abs)] = "abs";
    names[static_cast<size_t>(OpId::Sign)] = "sign";
    names[static_cast<size_t>(OpId::Log)] = "log";
    names[static_cast<size_t>(OpId::Exp)] = "exp";
    names[static_cast<size_t>(OpId::Pow)] = "pow";
    names[static_cast<size_t>(OpId::Clamp)] = "clamp";
    names[static_cast<size_t>(OpId::ClampMin)] = "clamp_min";
    names[static_cast<size_t>(OpId::ClampMax)] = "clamp_max";
    names[static_cast<size_t>(OpId::Reciprocal)] = "reciprocal";
    names[static_cast<size_t>(OpId::Floor)] = "floor";
    names[static_cast<size_t>(OpId::Ceil)] = "ceil";
    names[static_cast<size_t>(OpId::Round)] = "round";

    // Trigonometric
    names[static_cast<size_t>(OpId::Sin)] = "sin";
    names[static_cast<size_t>(OpId::Cos)] = "cos";
    names[static_cast<size_t>(OpId::Tan)] = "tan";
    names[static_cast<size_t>(OpId::Asin)] = "asin";
    names[static_cast<size_t>(OpId::Acos)] = "acos";
    names[static_cast<size_t>(OpId::Atan)] = "atan";
    names[static_cast<size_t>(OpId::Sinh)] = "sinh";
    names[static_cast<size_t>(OpId::Cosh)] = "cosh";
    names[static_cast<size_t>(OpId::Tanh)] = "tanh";

    // Activations
    names[static_cast<size_t>(OpId::ReLU)] = "relu";
    names[static_cast<size_t>(OpId::ReLUBackward)] = "relu_backward";
    names[static_cast<size_t>(OpId::Sigmoid)] = "sigmoid";
    names[static_cast<size_t>(OpId::SigmoidBackward)] = "sigmoid_backward";
    names[static_cast<size_t>(OpId::TanhActivation)] = "tanh_activation";
    names[static_cast<size_t>(OpId::TanhBackward)] = "tanh_backward";
    names[static_cast<size_t>(OpId::Gelu)] = "gelu";
    names[static_cast<size_t>(OpId::GeluBackward)] = "gelu_backward";
    names[static_cast<size_t>(OpId::Swish)] = "swish";
    names[static_cast<size_t>(OpId::SwishBackward)] = "swish_backward";
    names[static_cast<size_t>(OpId::LeakyReLU)] = "leaky_relu";
    names[static_cast<size_t>(OpId::LeakyReLUBackward)] = "leaky_relu_backward";
    names[static_cast<size_t>(OpId::Elu)] = "elu";
    names[static_cast<size_t>(OpId::EluBackward)] = "elu_backward";
    names[static_cast<size_t>(OpId::Selu)] = "selu";
    names[static_cast<size_t>(OpId::SeluBackward)] = "selu_backward";
    names[static_cast<size_t>(OpId::Mish)] = "mish";
    names[static_cast<size_t>(OpId::MishBackward)] = "mish_backward";
    names[static_cast<size_t>(OpId::Softplus)] = "softplus";
    names[static_cast<size_t>(OpId::SoftplusBackward)] = "softplus_backward";
    names[static_cast<size_t>(OpId::Softmax)] = "softmax";
    names[static_cast<size_t>(OpId::SoftmaxBackward)] = "softmax_backward";
    names[static_cast<size_t>(OpId::LogSoftmax)] = "log_softmax";
    names[static_cast<size_t>(OpId::LogSoftmaxBackward)] = "log_softmax_backward";
    // In-place activations
    names[static_cast<size_t>(OpId::ReLUInplace)] = "relu_inplace";
    names[static_cast<size_t>(OpId::SigmoidInplace)] = "sigmoid_inplace";
    names[static_cast<size_t>(OpId::TanhInplace)] = "tanh_inplace";
    names[static_cast<size_t>(OpId::LeakyReLUInplace)] = "leaky_relu_inplace";
    names[static_cast<size_t>(OpId::GeluInplace)] = "gelu_inplace";

    // Shape operations
    names[static_cast<size_t>(OpId::Reshape)] = "reshape";
    names[static_cast<size_t>(OpId::Transpose)] = "transpose";
    names[static_cast<size_t>(OpId::Permute)] = "permute";
    names[static_cast<size_t>(OpId::Squeeze)] = "squeeze";
    names[static_cast<size_t>(OpId::Unsqueeze)] = "unsqueeze";
    names[static_cast<size_t>(OpId::Flatten)] = "flatten";
    names[static_cast<size_t>(OpId::Contiguous)] = "contiguous";
    names[static_cast<size_t>(OpId::Clone)] = "clone";
    names[static_cast<size_t>(OpId::Fill)] = "fill";
    names[static_cast<size_t>(OpId::Repeat)] = "repeat";
    names[static_cast<size_t>(OpId::Tile)] = "tile";
    names[static_cast<size_t>(OpId::Expand)] = "expand";
    names[static_cast<size_t>(OpId::Stack)] = "stack";
    names[static_cast<size_t>(OpId::Split)] = "split";
    names[static_cast<size_t>(OpId::Chunk)] = "chunk";
    names[static_cast<size_t>(OpId::ToMemoryFormat)] = "to_memory_format";

    // Indexing
    names[static_cast<size_t>(OpId::IndexSelect)] = "index_select";
    names[static_cast<size_t>(OpId::Gather)] = "gather";
    names[static_cast<size_t>(OpId::Scatter)] = "scatter";
    names[static_cast<size_t>(OpId::MaskedSelect)] = "masked_select";
    names[static_cast<size_t>(OpId::MaskedFill)] = "masked_fill";
    names[static_cast<size_t>(OpId::Where)] = "where";
    names[static_cast<size_t>(OpId::Slice)] = "slice";
    names[static_cast<size_t>(OpId::Cat)] = "cat";
    names[static_cast<size_t>(OpId::Take)] = "take";
    names[static_cast<size_t>(OpId::Put)] = "put";
    names[static_cast<size_t>(OpId::Nonzero)] = "nonzero";
    names[static_cast<size_t>(OpId::OneHot)] = "one_hot";

    // Comparison
    names[static_cast<size_t>(OpId::Eq)] = "eq";
    names[static_cast<size_t>(OpId::Ne)] = "ne";
    names[static_cast<size_t>(OpId::Lt)] = "lt";
    names[static_cast<size_t>(OpId::Le)] = "le";
    names[static_cast<size_t>(OpId::Gt)] = "gt";
    names[static_cast<size_t>(OpId::Ge)] = "ge";

    // Normalization
    names[static_cast<size_t>(OpId::BatchNorm2dMeanVar)] = "batchnorm2d_mean_var";
    names[static_cast<size_t>(OpId::BatchNorm2dForward)] = "batchnorm2d_forward";
    names[static_cast<size_t>(OpId::BatchNorm2dForwardAffine)] = "batchnorm2d_forward_affine";
    names[static_cast<size_t>(OpId::BatchNorm2dUpdateRunningStats)] = "batchnorm2d_update_running_stats";
    names[static_cast<size_t>(OpId::BatchNorm2dBackward)] = "batchnorm2d_backward";
    names[static_cast<size_t>(OpId::LayerNorm)] = "layer_norm";
    names[static_cast<size_t>(OpId::LayerNormBackward)] = "layer_norm_backward";
    names[static_cast<size_t>(OpId::GroupNorm)] = "group_norm";
    names[static_cast<size_t>(OpId::GroupNormBackward)] = "group_norm_backward";
    names[static_cast<size_t>(OpId::InstanceNorm)] = "instance_norm";
    names[static_cast<size_t>(OpId::InstanceNormBackward)] = "instance_norm_backward";
    names[static_cast<size_t>(OpId::RMSNormBackward)] = "rms_norm_backward";

    // Convolution
    names[static_cast<size_t>(OpId::Conv2dForward)] = "conv2d_forward";
    names[static_cast<size_t>(OpId::Conv2dBackwardInput)] = "conv2d_backward_input";
    names[static_cast<size_t>(OpId::Conv2dBackwardWeight)] = "conv2d_backward_weight";
    names[static_cast<size_t>(OpId::Conv2dBackwardBias)] = "conv2d_backward_bias";
    names[static_cast<size_t>(OpId::ConvTranspose2dForward)] = "conv_transpose2d_forward";
    names[static_cast<size_t>(OpId::DepthwiseConv2d)] = "depthwise_conv2d";

    // Pooling
    names[static_cast<size_t>(OpId::MaxPool2dForward)] = "maxpool2d_forward";
    names[static_cast<size_t>(OpId::MaxPool2dBackward)] = "maxpool2d_backward";
    names[static_cast<size_t>(OpId::AvgPool2dForward)] = "avgpool2d_forward";
    names[static_cast<size_t>(OpId::AvgPool2dBackward)] = "avgpool2d_backward";
    names[static_cast<size_t>(OpId::AdaptiveAvgPool2d)] = "adaptive_avgpool2d";
    names[static_cast<size_t>(OpId::AdaptiveAvgPool2dBackward)] = "adaptive_avgpool2d_backward";
    names[static_cast<size_t>(OpId::AdaptiveMaxPool2d)] = "adaptive_maxpool2d";
    names[static_cast<size_t>(OpId::AdaptiveMaxPool2dBackward)] = "adaptive_maxpool2d_backward";

    // Vision
    names[static_cast<size_t>(OpId::Unfold)] = "unfold";
    names[static_cast<size_t>(OpId::Fold)] = "fold";
    names[static_cast<size_t>(OpId::Interpolate)] = "interpolate";
    names[static_cast<size_t>(OpId::ROIAlignForward)] = "roi_align_forward";
    names[static_cast<size_t>(OpId::ROIAlignBackward)] = "roi_align_backward";
    names[static_cast<size_t>(OpId::BoxIoU)] = "box_iou";

    // Fused operations
    names[static_cast<size_t>(OpId::FusedLinearReLU)] = "fused_linear_relu";
    names[static_cast<size_t>(OpId::FusedConv2dReLU)] = "fused_conv2d_relu";
    names[static_cast<size_t>(OpId::FusedBatchNormReLU)] = "fused_batchnorm_relu";
    names[static_cast<size_t>(OpId::FusedSoftmaxCrossEntropy)] = "fused_softmax_cross_entropy";
    names[static_cast<size_t>(OpId::FusedAddReLU)] = "fused_add_relu";
    names[static_cast<size_t>(OpId::FusedGelu)] = "fused_gelu";
    names[static_cast<size_t>(OpId::FusedLayerNorm)] = "fused_layer_norm";
    names[static_cast<size_t>(OpId::FusedRMSNorm)] = "fused_rms_norm";
    names[static_cast<size_t>(OpId::FusedAttention)] = "fused_attention";
    names[static_cast<size_t>(OpId::FusedSGDStep)] = "fused_sgd_step";
    names[static_cast<size_t>(OpId::FusedAdamStep)] = "fused_adam_step";
    names[static_cast<size_t>(OpId::FusedRMSPropStep)] = "fused_rmsprop_step";
    names[static_cast<size_t>(OpId::FusedAdadeltaStep)] = "fused_adadelta_step";
    names[static_cast<size_t>(OpId::FusedAdagradStep)] = "fused_adagrad_step";
    names[static_cast<size_t>(OpId::FusedConv2dSigmoid)] = "fused_conv2d_sigmoid";
    names[static_cast<size_t>(OpId::FusedConv2dTanh)] = "fused_conv2d_tanh";
    names[static_cast<size_t>(OpId::FusedConv2dSwish)] = "fused_conv2d_swish";

    // Creation
    names[static_cast<size_t>(OpId::Zeros)] = "zeros";
    names[static_cast<size_t>(OpId::Ones)] = "ones";
    names[static_cast<size_t>(OpId::Full)] = "full";
    names[static_cast<size_t>(OpId::Rand)] = "rand";
    names[static_cast<size_t>(OpId::Randn)] = "randn";
    names[static_cast<size_t>(OpId::Arange)] = "arange";
    names[static_cast<size_t>(OpId::Linspace)] = "linspace";
    names[static_cast<size_t>(OpId::Eye)] = "eye";

    // RNN
    names[static_cast<size_t>(OpId::LSTMCellForward)] = "lstm_cell_forward";
    names[static_cast<size_t>(OpId::LSTMCellBackward)] = "lstm_cell_backward";
    names[static_cast<size_t>(OpId::GRUCellForward)] = "gru_cell_forward";
    names[static_cast<size_t>(OpId::GRUCellBackward)] = "gru_cell_backward";

    // Embedding
    names[static_cast<size_t>(OpId::Embedding)] = "embedding";
    names[static_cast<size_t>(OpId::EmbeddingBackward)] = "embedding_backward";

    // Linear
    names[static_cast<size_t>(OpId::Linear)] = "linear";
    names[static_cast<size_t>(OpId::LinearBackward)] = "linear_backward";

    // Dropout
    names[static_cast<size_t>(OpId::Dropout)] = "dropout";
    names[static_cast<size_t>(OpId::DropoutBackward)] = "dropout_backward";

    // JIT markers
    names[static_cast<size_t>(OpId::Constant)] = "constant";
    names[static_cast<size_t>(OpId::Input)] = "input";
    names[static_cast<size_t>(OpId::Output)] = "output";

    return names;
}();

} // anonymous namespace

auto op_id_to_name(OpId id) noexcept -> std::string_view {
    auto idx = static_cast<size_t>(id);
    if (idx < OP_COUNT) {
        return op_names[idx];
    }
    return "unknown";
}

} // namespace tenzor
