#include "tenzor/backend/op_attributes.hpp"

namespace tenzor {

auto attr_key_name(AttrKey key) -> std::string_view {
    switch (key) {
        case AttrKey::Dim: return "dim";
        case AttrKey::Dim0: return "dim0";
        case AttrKey::Dim1: return "dim1";
        case AttrKey::StartDim: return "start_dim";
        case AttrKey::EndDim: return "end_dim";
        case AttrKey::NormalizedShape: return "normalized_shape";
        case AttrKey::Stride: return "stride";
        case AttrKey::StrideH: return "stride_h";
        case AttrKey::StrideW: return "stride_w";
        case AttrKey::StrideD: return "stride_d";
        case AttrKey::Padding: return "padding";
        case AttrKey::PaddingH: return "padding_h";
        case AttrKey::PaddingW: return "padding_w";
        case AttrKey::PaddingD: return "padding_d";
        case AttrKey::Dilation: return "dilation";
        case AttrKey::DilationH: return "dilation_h";
        case AttrKey::DilationW: return "dilation_w";
        case AttrKey::Groups: return "groups";
        case AttrKey::KernelSize: return "kernel_size";
        case AttrKey::KernelSizeH: return "kernel_size_h";
        case AttrKey::KernelSizeW: return "kernel_size_w";
        case AttrKey::KernelSizeD: return "kernel_size_d";
        case AttrKey::OutputPadding: return "output_padding";
        case AttrKey::OutputPaddingH: return "output_padding_h";
        case AttrKey::OutputPaddingW: return "output_padding_w";
        case AttrKey::Eps: return "eps";
        case AttrKey::Momentum: return "momentum";
        case AttrKey::Alpha: return "alpha";
        case AttrKey::Beta: return "beta";
        case AttrKey::Tau: return "tau";
        case AttrKey::Value: return "value";
        case AttrKey::Lr: return "lr";
        case AttrKey::WeightDecay: return "weight_decay";
        case AttrKey::Rho: return "rho";
        case AttrKey::Keepdim: return "keepdim";
        case AttrKey::Training: return "training";
        case AttrKey::CeilMode: return "ceil_mode";
        case AttrKey::CountIncludePad: return "count_include_pad";
        case AttrKey::Right: return "right";
        case AttrKey::Hard: return "hard";
        case AttrKey::Centered: return "centered";
        case AttrKey::Accumulate: return "accumulate";
        case AttrKey::Shape: return "shape";
        case AttrKey::Repeats: return "repeats";
        case AttrKey::OutputSize: return "output_size";
        case AttrKey::OutputSizeH: return "output_size_h";
        case AttrKey::OutputSizeW: return "output_size_w";
        case AttrKey::OutputSizeD: return "output_size_d";
        case AttrKey::Dtype: return "dtype";
        case AttrKey::Device: return "device";
        case AttrKey::NumEmbeddings: return "num_embeddings";
        case AttrKey::EmbeddingDim: return "embedding_dim";
        case AttrKey::NumClasses: return "num_classes";
        case AttrKey::Shift: return "shift";
        case AttrKey::Start: return "start";
        case AttrKey::End: return "end";
        case AttrKey::Step: return "step";
        case AttrKey::MemoryFormat: return "memory_format";
        case AttrKey::Algorithm: return "algorithm";
        case AttrKey::WorkspaceLimit: return "workspace_limit";
        case AttrKey::Negative_slope: return "negative_slope";
        case AttrKey::P: return "p";
        case AttrKey::Norm: return "norm";
        case AttrKey::N: return "n";
        case AttrKey::Stream: return "stream";
        case AttrKey::_Count: return "_count";
    }
    return "unknown";
}

} // namespace tenzor
