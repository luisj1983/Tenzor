/**
 * @file op_id.cpp
 * @brief Implementation of OpId string conversion for error messages
 */

#include "tenzor/ops/op_id.hpp"
#include <array>
#include <string>
#include <unordered_map>

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
    names[static_cast<size_t>(OpId::Any)] = "any";
    names[static_cast<size_t>(OpId::All)] = "all";
    names[static_cast<size_t>(OpId::Median)] = "median";
    names[static_cast<size_t>(OpId::Mode)] = "mode";
    names[static_cast<size_t>(OpId::CountNonzero)] = "count_nonzero";
    names[static_cast<size_t>(OpId::Nansum)] = "nansum";
    names[static_cast<size_t>(OpId::Nanmean)] = "nanmean";
    names[static_cast<size_t>(OpId::Aminmax)] = "aminmax";

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
    names[static_cast<size_t>(OpId::Trunc)] = "trunc";
    names[static_cast<size_t>(OpId::Frac)] = "frac";
    names[static_cast<size_t>(OpId::Heaviside)] = "heaviside";
    names[static_cast<size_t>(OpId::NanToNum)] = "nan_to_num";

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
    names[static_cast<size_t>(OpId::LogSigmoid)] = "log_sigmoid";
    names[static_cast<size_t>(OpId::LogSigmoidBackward)] = "log_sigmoid_backward";
    names[static_cast<size_t>(OpId::RReLU)] = "rrelu";
    names[static_cast<size_t>(OpId::RReLUBackward)] = "rrelu_backward";

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
    names[static_cast<size_t>(OpId::AdvancedIndex)] = "advanced_index";
    names[static_cast<size_t>(OpId::AdvancedIndexPut)] = "advanced_index_put";

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
    names[static_cast<size_t>(OpId::RMSNorm)] = "rms_norm";
    names[static_cast<size_t>(OpId::RMSNormBackward)] = "rms_norm_backward";
    names[static_cast<size_t>(OpId::BatchNorm2dFusedTraining)] = "batchnorm2d_fused_training";

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
    names[static_cast<size_t>(OpId::GatherRelativePositionBias)] = "gather_relative_position_bias";
    names[static_cast<size_t>(OpId::NMS)] = "nms";
    names[static_cast<size_t>(OpId::GridSample)] = "grid_sample";
    names[static_cast<size_t>(OpId::AffineGrid)] = "affine_grid";

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
    names[static_cast<size_t>(OpId::FusedConv2dBnReLU)] = "fused_conv2d_bn_relu";
    names[static_cast<size_t>(OpId::FusedLayerNormBackward)] = "fused_layer_norm_backward";
    names[static_cast<size_t>(OpId::FusedAdamAtan2Step)] = "fused_adam_atan2_step";

    // Creation
    names[static_cast<size_t>(OpId::Zeros)] = "zeros";
    names[static_cast<size_t>(OpId::Ones)] = "ones";
    names[static_cast<size_t>(OpId::Full)] = "full";
    names[static_cast<size_t>(OpId::Rand)] = "rand";
    names[static_cast<size_t>(OpId::Randn)] = "randn";
    names[static_cast<size_t>(OpId::Arange)] = "arange";
    names[static_cast<size_t>(OpId::Linspace)] = "linspace";
    names[static_cast<size_t>(OpId::Eye)] = "eye";
    names[static_cast<size_t>(OpId::Randint)] = "randint";

    // RNN
    names[static_cast<size_t>(OpId::LSTMCellForward)] = "lstm_cell_forward";
    names[static_cast<size_t>(OpId::LSTMCellBackward)] = "lstm_cell_backward";
    names[static_cast<size_t>(OpId::GRUCellForward)] = "gru_cell_forward";
    names[static_cast<size_t>(OpId::GRUCellBackward)] = "gru_cell_backward";
    names[static_cast<size_t>(OpId::LSTMForward)] = "lstm_forward";
    names[static_cast<size_t>(OpId::GRUForward)] = "gru_forward";
    names[static_cast<size_t>(OpId::LSTMMultiLayerForward)] = "lstm_multi_layer_forward";
    names[static_cast<size_t>(OpId::GRUMultiLayerForward)] = "gru_multi_layer_forward";
    names[static_cast<size_t>(OpId::BiLSTMForward)] = "bilstm_forward";

    // Embedding
    names[static_cast<size_t>(OpId::Embedding)] = "embedding";
    names[static_cast<size_t>(OpId::EmbeddingBackward)] = "embedding_backward";

    // Linear
    names[static_cast<size_t>(OpId::Linear)] = "linear";
    names[static_cast<size_t>(OpId::LinearBackward)] = "linear_backward";

    // Dropout
    names[static_cast<size_t>(OpId::Dropout)] = "dropout";
    names[static_cast<size_t>(OpId::DropoutBackward)] = "dropout_backward";

    // Advanced operations
    names[static_cast<size_t>(OpId::TopK)] = "topk";
    names[static_cast<size_t>(OpId::Sort)] = "sort";
    names[static_cast<size_t>(OpId::CumSum)] = "cumsum";
    names[static_cast<size_t>(OpId::CumProd)] = "cumprod";
    names[static_cast<size_t>(OpId::Unique)] = "unique";
    names[static_cast<size_t>(OpId::FlashAttention)] = "flash_attention";
    names[static_cast<size_t>(OpId::FlashAttentionBackward)] = "flash_attention_backward";

    // 3D Convolution and Pooling
    names[static_cast<size_t>(OpId::Conv3dForward)] = "conv3d_forward";
    names[static_cast<size_t>(OpId::Conv3dBackwardInput)] = "conv3d_backward_input";
    names[static_cast<size_t>(OpId::Conv3dBackwardWeight)] = "conv3d_backward_weight";
    names[static_cast<size_t>(OpId::Conv3dBackwardBias)] = "conv3d_backward_bias";
    names[static_cast<size_t>(OpId::MaxPool3dForward)] = "maxpool3d_forward";
    names[static_cast<size_t>(OpId::MaxPool3dBackward)] = "maxpool3d_backward";
    names[static_cast<size_t>(OpId::AvgPool3dForward)] = "avgpool3d_forward";
    names[static_cast<size_t>(OpId::AvgPool3dBackward)] = "avgpool3d_backward";
    names[static_cast<size_t>(OpId::AdaptiveMaxPool3d)] = "adaptive_maxpool3d";
    names[static_cast<size_t>(OpId::AdaptiveMaxPool3dBackward)] = "adaptive_maxpool3d_backward";
    names[static_cast<size_t>(OpId::AdaptiveAvgPool3d)] = "adaptive_avgpool3d";
    names[static_cast<size_t>(OpId::AdaptiveAvgPool3dBackward)] = "adaptive_avgpool3d_backward";

    // 3D Transposed Convolution
    names[static_cast<size_t>(OpId::ConvTranspose3dForward)] = "conv_transpose3d_forward";
    names[static_cast<size_t>(OpId::ConvTranspose3dBackwardInput)] = "conv_transpose3d_backward_input";
    names[static_cast<size_t>(OpId::ConvTranspose3dBackwardWeight)] = "conv_transpose3d_backward_weight";
    names[static_cast<size_t>(OpId::ConvTranspose3dBackwardBias)] = "conv_transpose3d_backward_bias";
    names[static_cast<size_t>(OpId::Conv1dForward)] = "conv1d_forward";
    names[static_cast<size_t>(OpId::Conv1dBackwardInput)] = "conv1d_backward_input";
    names[static_cast<size_t>(OpId::Conv1dBackwardWeight)] = "conv1d_backward_weight";
    names[static_cast<size_t>(OpId::Conv1dBackwardBias)] = "conv1d_backward_bias";

    // Type Conversion
    names[static_cast<size_t>(OpId::Cast)] = "cast";

    // Extended Math
    names[static_cast<size_t>(OpId::Log2)] = "log2";
    names[static_cast<size_t>(OpId::Log10)] = "log10";
    names[static_cast<size_t>(OpId::Log1p)] = "log1p";
    names[static_cast<size_t>(OpId::Exp2)] = "exp2";
    names[static_cast<size_t>(OpId::Expm1)] = "expm1";
    names[static_cast<size_t>(OpId::Erf)] = "erf";
    names[static_cast<size_t>(OpId::Erfc)] = "erfc";
    names[static_cast<size_t>(OpId::IsNan)] = "isnan";
    names[static_cast<size_t>(OpId::IsInf)] = "isinf";
    names[static_cast<size_t>(OpId::IsFinite)] = "isfinite";
    names[static_cast<size_t>(OpId::Atan2)] = "atan2";
    names[static_cast<size_t>(OpId::Fmod)] = "fmod";
    names[static_cast<size_t>(OpId::Remainder)] = "remainder";
    names[static_cast<size_t>(OpId::Lerp)] = "lerp";

    // Tensor Manipulation
    names[static_cast<size_t>(OpId::Triu)] = "triu";
    names[static_cast<size_t>(OpId::Tril)] = "tril";
    names[static_cast<size_t>(OpId::Diag)] = "diag";
    names[static_cast<size_t>(OpId::Trace)] = "trace";
    names[static_cast<size_t>(OpId::Flip)] = "flip";
    names[static_cast<size_t>(OpId::LogicalAnd)] = "logical_and";
    names[static_cast<size_t>(OpId::LogicalOr)] = "logical_or";
    names[static_cast<size_t>(OpId::LogicalNot)] = "logical_not";
    names[static_cast<size_t>(OpId::LogicalXor)] = "logical_xor";

    names[static_cast<size_t>(OpId::Minimum)] = "minimum";
    names[static_cast<size_t>(OpId::Maximum)] = "maximum";
    names[static_cast<size_t>(OpId::Cross)] = "cross";

    // 1D Pooling
    names[static_cast<size_t>(OpId::MaxPool1dForward)] = "maxpool1d_forward";
    names[static_cast<size_t>(OpId::MaxPool1dBackward)] = "maxpool1d_backward";
    names[static_cast<size_t>(OpId::AvgPool1dForward)] = "avgpool1d_forward";
    names[static_cast<size_t>(OpId::AvgPool1dBackward)] = "avgpool1d_backward";
    names[static_cast<size_t>(OpId::AdaptiveMaxPool1d)] = "adaptive_maxpool1d";
    names[static_cast<size_t>(OpId::AdaptiveMaxPool1dBackward)] = "adaptive_maxpool1d_backward";
    names[static_cast<size_t>(OpId::AdaptiveAvgPool1d)] = "adaptive_avgpool1d";
    names[static_cast<size_t>(OpId::AdaptiveAvgPool1dBackward)] = "adaptive_avgpool1d_backward";

    // Tensor Manipulation (continued)
    names[static_cast<size_t>(OpId::Roll)] = "roll";

    // Search and Sampling
    names[static_cast<size_t>(OpId::SearchSorted)] = "searchsorted";
    names[static_cast<size_t>(OpId::GumbelSoftmax)] = "gumbel_softmax";

    // FFT Operations
    names[static_cast<size_t>(OpId::FFT)] = "fft";
    names[static_cast<size_t>(OpId::IFFT)] = "ifft";
    names[static_cast<size_t>(OpId::RFFT)] = "rfft";
    names[static_cast<size_t>(OpId::IRFFT)] = "irfft";
    names[static_cast<size_t>(OpId::FFT2)] = "fft2";
    names[static_cast<size_t>(OpId::IFFT2)] = "ifft2";
    names[static_cast<size_t>(OpId::FFTN)] = "fftn";
    names[static_cast<size_t>(OpId::IFFTN)] = "ifftn";

    // Indexing Operations
    names[static_cast<size_t>(OpId::ScatterAdd)] = "scatter_add";
    names[static_cast<size_t>(OpId::IndexAdd)] = "index_add";
    names[static_cast<size_t>(OpId::IndexCopy)] = "index_copy";
    names[static_cast<size_t>(OpId::IndexFill)] = "index_fill";

    // Linear Algebra Operations
    names[static_cast<size_t>(OpId::LinalgDet)] = "linalg_det";
    names[static_cast<size_t>(OpId::LinalgInv)] = "linalg_inv";
    names[static_cast<size_t>(OpId::LinalgSolve)] = "linalg_solve";
    names[static_cast<size_t>(OpId::LinalgSVD)] = "linalg_svd";
    names[static_cast<size_t>(OpId::LinalgQR)] = "linalg_qr";
    names[static_cast<size_t>(OpId::LinalgEigh)] = "linalg_eigh";
    names[static_cast<size_t>(OpId::LinalgEig)] = "linalg_eig";
    names[static_cast<size_t>(OpId::LinalgCholesky)] = "linalg_cholesky";
    names[static_cast<size_t>(OpId::StridedFill)] = "strided_fill";
    names[static_cast<size_t>(OpId::QuantizedLinear)] = "quantized_linear";
    names[static_cast<size_t>(OpId::QuantizedConv2d)] = "quantized_conv2d";
    names[static_cast<size_t>(OpId::EmbeddingWithBoundsCheck)] = "embedding_with_bounds_check";
    names[static_cast<size_t>(OpId::LogSumExp)] = "logsumexp";
    names[static_cast<size_t>(OpId::HasInfNan)] = "has_inf_nan";
    names[static_cast<size_t>(OpId::EmbeddingBagForward)] = "embedding_bag_forward";
    names[static_cast<size_t>(OpId::EmbeddingBagBackward)] = "embedding_bag_backward";

    // Complex operations
    names[static_cast<size_t>(OpId::Conj)] = "conj";
    names[static_cast<size_t>(OpId::Real)] = "real";
    names[static_cast<size_t>(OpId::Imag)] = "imag";
    names[static_cast<size_t>(OpId::Angle)] = "angle";
    names[static_cast<size_t>(OpId::Polar)] = "polar";

    // Signal Processing Operations
    names[static_cast<size_t>(OpId::STFT)] = "stft";
    names[static_cast<size_t>(OpId::ISTFT)] = "istft";
    names[static_cast<size_t>(OpId::CDist)] = "cdist";

    // Sampling and Statistics Operations
    names[static_cast<size_t>(OpId::Multinomial)] = "multinomial";
    names[static_cast<size_t>(OpId::Bernoulli)] = "bernoulli";
    names[static_cast<size_t>(OpId::Histogram)] = "histogram";
    names[static_cast<size_t>(OpId::Bucketize)] = "bucketize";
    names[static_cast<size_t>(OpId::NormalSample)] = "normal_sample";
    names[static_cast<size_t>(OpId::PoissonSample)] = "poisson_sample";
    names[static_cast<size_t>(OpId::ExponentialSample)] = "exponential_sample";
    names[static_cast<size_t>(OpId::Histogramdd)] = "histogramdd";

    // Special Math Functions
    names[static_cast<size_t>(OpId::Gamma)] = "gamma";
    names[static_cast<size_t>(OpId::Lgamma)] = "lgamma";
    names[static_cast<size_t>(OpId::Digamma)] = "digamma";
    names[static_cast<size_t>(OpId::Polygamma)] = "polygamma";
    names[static_cast<size_t>(OpId::Beta)] = "beta";
    names[static_cast<size_t>(OpId::BetaInc)] = "betainc";
    names[static_cast<size_t>(OpId::BesselJ0)] = "bessel_j0";
    names[static_cast<size_t>(OpId::BesselJ1)] = "bessel_j1";
    names[static_cast<size_t>(OpId::BesselY0)] = "bessel_y0";
    names[static_cast<size_t>(OpId::BesselY1)] = "bessel_y1";
    names[static_cast<size_t>(OpId::BesselI0)] = "bessel_i0";
    names[static_cast<size_t>(OpId::BesselI1)] = "bessel_i1";
    names[static_cast<size_t>(OpId::ErfInv)] = "erfinv";
    names[static_cast<size_t>(OpId::Sinc)] = "sinc";
    names[static_cast<size_t>(OpId::Zeta)] = "zeta";

    // Sparse Tensor Operations
    names[static_cast<size_t>(OpId::SparseSpMM)] = "sparse_spmm";
    names[static_cast<size_t>(OpId::SparseSpMV)] = "sparse_spmv";
    names[static_cast<size_t>(OpId::SparseToDense)] = "sparse_to_dense";
    names[static_cast<size_t>(OpId::DenseToSparse)] = "dense_to_sparse";
    names[static_cast<size_t>(OpId::SparseAdd)] = "sparse_add";
    names[static_cast<size_t>(OpId::SparseSpGEMM)] = "sparse_spgemm";
    names[static_cast<size_t>(OpId::SparseTrsv)] = "sparse_trsv";
    names[static_cast<size_t>(OpId::SparseTrsm)] = "sparse_trsm";

    // Bitwise operations
    names[static_cast<size_t>(OpId::BitwiseAnd)] = "bitwise_and";
    names[static_cast<size_t>(OpId::BitwiseOr)] = "bitwise_or";
    names[static_cast<size_t>(OpId::BitwiseXor)] = "bitwise_xor";
    names[static_cast<size_t>(OpId::BitwiseNot)] = "bitwise_not";
    names[static_cast<size_t>(OpId::BitwiseLeftShift)] = "bitwise_left_shift";
    names[static_cast<size_t>(OpId::BitwiseRightShift)] = "bitwise_right_shift";

    // Scatter-reduce
    names[static_cast<size_t>(OpId::ScatterReduce)] = "scatter_reduce";

    // Fused GEMM operations
    names[static_cast<size_t>(OpId::Addmm)] = "addmm";
    names[static_cast<size_t>(OpId::Addmv)] = "addmv";
    names[static_cast<size_t>(OpId::Baddbmm)] = "baddbmm";
    names[static_cast<size_t>(OpId::Trapezoid)] = "trapezoid";
    names[static_cast<size_t>(OpId::CumulativeTrapezoid)] = "cumulative_trapezoid";
    names[static_cast<size_t>(OpId::NumericalGradient)] = "gradient";
    names[static_cast<size_t>(OpId::PairwiseDistance)] = "pairwise_distance";
    names[static_cast<size_t>(OpId::Pdist)] = "pdist";

    // Repeat interleave
    names[static_cast<size_t>(OpId::RepeatInterleave)] = "repeat_interleave";

    // Cumulative scan / histogram
    names[static_cast<size_t>(OpId::Logcumsumexp)] = "logcumsumexp";
    names[static_cast<size_t>(OpId::Bincount)] = "bincount";
    names[static_cast<size_t>(OpId::SegmentReduce)] = "segment_reduce";
    names[static_cast<size_t>(OpId::Ndtr)] = "ndtr";
    names[static_cast<size_t>(OpId::LogNdtr)] = "log_ndtr";
    names[static_cast<size_t>(OpId::Multigammaln)] = "multigammaln";

    // New element-wise math (Phase 3)
    names[static_cast<size_t>(OpId::Rsqrt)] = "rsqrt";
    names[static_cast<size_t>(OpId::Square)] = "square";
    names[static_cast<size_t>(OpId::Asinh)] = "asinh";
    names[static_cast<size_t>(OpId::Acosh)] = "acosh";
    names[static_cast<size_t>(OpId::Atanh)] = "atanh";
    names[static_cast<size_t>(OpId::Hypot)] = "hypot";
    names[static_cast<size_t>(OpId::Copysign)] = "copysign";
    names[static_cast<size_t>(OpId::Nextafter)] = "nextafter";
    names[static_cast<size_t>(OpId::Gcd)] = "gcd";
    names[static_cast<size_t>(OpId::Lcm)] = "lcm";
    names[static_cast<size_t>(OpId::Addcmul)] = "addcmul";
    names[static_cast<size_t>(OpId::Addcdiv)] = "addcdiv";
    names[static_cast<size_t>(OpId::Igamma)] = "igamma";
    names[static_cast<size_t>(OpId::Igammac)] = "igammac";

    // New reduction operations (Phase 4)
    names[static_cast<size_t>(OpId::CumMax)] = "cummax";
    names[static_cast<size_t>(OpId::CumMin)] = "cummin";
    names[static_cast<size_t>(OpId::Isin)] = "isin";
    names[static_cast<size_t>(OpId::Kthvalue)] = "kthvalue";
    names[static_cast<size_t>(OpId::Fmax)] = "fmax";
    names[static_cast<size_t>(OpId::Fmin)] = "fmin";
    names[static_cast<size_t>(OpId::Quantile)] = "quantile";
    names[static_cast<size_t>(OpId::Nanquantile)] = "nanquantile";
    names[static_cast<size_t>(OpId::Nanmedian)] = "nanmedian";
    names[static_cast<size_t>(OpId::Histc)] = "histc";
    names[static_cast<size_t>(OpId::UniqueConsecutive)] = "unique_consecutive";

    // New linear algebra (Phase 5)
    names[static_cast<size_t>(OpId::DiagEmbed)] = "diag_embed";
    names[static_cast<size_t>(OpId::Diagflat)] = "diagflat";
    names[static_cast<size_t>(OpId::SolveTriangular)] = "solve_triangular";
    names[static_cast<size_t>(OpId::CholeskyInverse)] = "cholesky_inverse";
    names[static_cast<size_t>(OpId::TensorInv)] = "tensorinv";
    names[static_cast<size_t>(OpId::TensorSolve)] = "tensorsolve";
    names[static_cast<size_t>(OpId::Ormqr)] = "ormqr";
    names[static_cast<size_t>(OpId::Geqrf)] = "geqrf";

    // Additional linear algebra (510-517)
    names[static_cast<size_t>(OpId::LinalgLU)] = "linalg_lu";
    names[static_cast<size_t>(OpId::LinalgLUSolve)] = "linalg_lu_solve";
    names[static_cast<size_t>(OpId::LinalgHouseholder)] = "linalg_householder";
    names[static_cast<size_t>(OpId::LinalgLDLFactor)] = "linalg_ldl_factor";
    names[static_cast<size_t>(OpId::LinalgLDLSolve)] = "linalg_ldl_solve";
    names[static_cast<size_t>(OpId::LinalgVectorNorm)] = "linalg_vector_norm";
    names[static_cast<size_t>(OpId::LinalgMatrixNorm)] = "linalg_matrix_norm";
    names[static_cast<size_t>(OpId::LinalgVecdot)] = "linalg_vecdot";

    // New shape/indexing (Phase 6)
    names[static_cast<size_t>(OpId::TakeAlongDim)] = "take_along_dim";
    names[static_cast<size_t>(OpId::MaskedScatter)] = "masked_scatter";
    names[static_cast<size_t>(OpId::TrilIndices)] = "tril_indices";
    names[static_cast<size_t>(OpId::TriuIndices)] = "triu_indices";
    names[static_cast<size_t>(OpId::AsStrided)] = "as_strided";
    names[static_cast<size_t>(OpId::ComplexTensor)] = "complex";

    // New pooling (Phase 9)
    names[static_cast<size_t>(OpId::FractionalMaxPool2dForward)] = "fractional_max_pool2d_forward";
    names[static_cast<size_t>(OpId::FractionalMaxPool2dBackward)] = "fractional_max_pool2d_backward";
    names[static_cast<size_t>(OpId::FractionalMaxPool3dForward)] = "fractional_max_pool3d_forward";
    names[static_cast<size_t>(OpId::FractionalMaxPool3dBackward)] = "fractional_max_pool3d_backward";
    names[static_cast<size_t>(OpId::MaxUnpool2dForward)] = "max_unpool2d_forward";
    names[static_cast<size_t>(OpId::MaxUnpool2dBackward)] = "max_unpool2d_backward";
    names[static_cast<size_t>(OpId::MaxUnpool3dForward)] = "max_unpool3d_forward";
    names[static_cast<size_t>(OpId::MaxUnpool3dBackward)] = "max_unpool3d_backward";
    names[static_cast<size_t>(OpId::NanVar)] = "nanvar";
    names[static_cast<size_t>(OpId::NanStd)] = "nanstd";
    names[static_cast<size_t>(OpId::Deg2Rad)] = "deg2rad";
    names[static_cast<size_t>(OpId::Rad2Deg)] = "rad2deg";
    names[static_cast<size_t>(OpId::Logit)] = "logit";
    names[static_cast<size_t>(OpId::Signbit)] = "signbit";
    names[static_cast<size_t>(OpId::FloatPower)] = "float_power";
    names[static_cast<size_t>(OpId::Xlog1py)] = "xlog1py";
    names[static_cast<size_t>(OpId::Ldexp)] = "ldexp";
    names[static_cast<size_t>(OpId::IsReal)] = "isreal";
    names[static_cast<size_t>(OpId::IsPosInf)] = "isposinf";
    names[static_cast<size_t>(OpId::IsNegInf)] = "isneginf";
    names[static_cast<size_t>(OpId::Frexp)] = "frexp";

    // Nested Tensor Operations
    names[static_cast<size_t>(OpId::NestedSoftmax)] = "nested_softmax";
    names[static_cast<size_t>(OpId::NestedLogSoftmax)] = "nested_log_softmax";
    names[static_cast<size_t>(OpId::NestedLayerNorm)] = "nested_layer_norm";
    names[static_cast<size_t>(OpId::NestedSum)] = "nested_sum";
    names[static_cast<size_t>(OpId::NestedMean)] = "nested_mean";
    names[static_cast<size_t>(OpId::NestedAttention)] = "nested_attention";
    names[static_cast<size_t>(OpId::NestedAttentionBackward)] = "nested_attention_backward";
    names[static_cast<size_t>(OpId::NestedToPadded)] = "nested_to_padded";
    names[static_cast<size_t>(OpId::NestedFromPadded)] = "nested_from_padded";
    names[static_cast<size_t>(OpId::NestedLinear)] = "nested_linear";

    // Numerically stable math / special functions
    names[static_cast<size_t>(OpId::LogAddExp)] = "logaddexp";
    names[static_cast<size_t>(OpId::LogAddExp2)] = "logaddexp2";
    names[static_cast<size_t>(OpId::XLogY)] = "xlogy";
    names[static_cast<size_t>(OpId::CosineSimilarity)] = "cosine_similarity";
    names[static_cast<size_t>(OpId::Renorm)] = "renorm";
    names[static_cast<size_t>(OpId::I0e)] = "i0e";
    names[static_cast<size_t>(OpId::I1e)] = "i1e";
    names[static_cast<size_t>(OpId::Entr)] = "entr";
    names[static_cast<size_t>(OpId::SphericalBesselJ0)] = "spherical_bessel_j0";

    return names;
}();

// Compile-time check: count named ops to catch forgotten entries when adding new OpIds
constexpr size_t count_named_ops() {
    size_t count = 0;
    for (const auto& name : op_names) {
        if (name != "unknown") ++count;
    }
    return count;
}

// Count of actual OpId enum values (excluding gap slots).
// Update this when adding new OpIds to catch missing name entries at compile time.
inline constexpr size_t EXPECTED_NAMED_OPS = 449;  // 448 previous + 1 new op

// If this fires, a new OpId was added without a corresponding name in op_names above
static_assert(count_named_ops() == EXPECTED_NAMED_OPS,
    "Mismatch: update EXPECTED_NAMED_OPS and add name entry when adding new OpIds");

} // anonymous namespace

auto op_id_to_name(OpId id) noexcept -> std::string_view {
    auto idx = static_cast<size_t>(id);
    if (idx < OP_COUNT) {
        return op_names[idx];
    }
    return "unknown";
}

auto string_to_op_id(std::string_view name) noexcept -> OpId {
    // Build the reverse map once on first call
    static const auto& reverse_map = *[]() {
        auto* m = new std::unordered_map<std::string, OpId>();
        m->reserve(EXPECTED_NAMED_OPS * 2);  // low load factor for speed
        for (size_t i = 0; i < OP_COUNT; ++i) {
            if (op_names[i] != "unknown") {
                m->emplace(std::string(op_names[i]), static_cast<OpId>(i));
            }
        }
        return m;
    }();

    auto it = reverse_map.find(std::string(name));
    if (it != reverse_map.end()) {
        return it->second;
    }
    return OpId::OP_COUNT;
}

} // namespace tenzor
