// Vision, detection, async_ops, and fused submodule Python bindings.
// Extracted from python/bindings.cpp as part of P3.4. These four
// submodules are self-contained (no shared lambdas with the root
// module) and total ~170 lines — a natural extraction target.

#include "register.hpp"

#include <pybind11/stl.h>

#include <tenzor/ops/vision.hpp>
#include <tenzor/ops/detection.hpp>
#include <tenzor/ops/async_ops.hpp>
#include <tenzor/ops/fused_ops.hpp>
#include <tenzor/core/tensor.hpp>

namespace py = pybind11;

namespace tenzor::python {

void register_vision_detection(py::module_& m) {
    // =========================================================================
    // Vision
    // =========================================================================
    auto vision = m.def_submodule("vision", "Vision operations");

    vision.def("unfold", &tenzor::ops::unfold,
               py::arg("input"),
               py::arg("kernel_size"),
               py::arg("stride") = 1,
               py::arg("padding") = 0,
               py::arg("dilation") = 1,
               "Extract sliding local blocks (im2col)");

    vision.def("fold", &tenzor::ops::fold,
               py::arg("input"),
               py::arg("output_size"),
               py::arg("kernel_size"),
               py::arg("stride") = 1,
               py::arg("padding") = 0,
               py::arg("dilation") = 1,
               "Fold tensor back to spatial dimensions (col2im)");

    vision.def("interpolate", &tenzor::ops::interpolate,
               py::arg("input"),
               py::arg("size"),
               py::arg("mode") = "bilinear",
               py::arg("align_corners") = false,
               "Resize tensor using interpolation");

    vision.def("grid_sample", &tenzor::ops::grid_sample,
               py::arg("input"),
               py::arg("grid"),
               py::arg("mode") = "bilinear",
               py::arg("padding_mode") = "zeros",
               py::arg("align_corners") = false,
               "Sample from input using grid coordinates (spatial transformer)");

    vision.def("affine_grid", &tenzor::ops::affine_grid,
               py::arg("theta"),
               py::arg("size"),
               py::arg("align_corners") = false,
               "Generate 2D affine grid for grid_sample");

    // =========================================================================
    // Detection
    // =========================================================================
    auto detection = m.def_submodule("detection", "Object detection operations");

    py::enum_<tenzor::ops::IoUType>(detection, "IoUType")
        .value("IoU", tenzor::ops::IoUType::IoU)
        .value("GIoU", tenzor::ops::IoUType::GIoU)
        .value("DIoU", tenzor::ops::IoUType::DIoU)
        .value("CIoU", tenzor::ops::IoUType::CIoU);

    detection.def("box_iou", &tenzor::ops::box_iou,
                  py::arg("boxes1"), py::arg("boxes2"),
                  py::arg("iou_type") = tenzor::ops::IoUType::IoU,
                  "Compute IoU between box sets");

    detection.def("nms", &tenzor::ops::nms,
                  py::arg("boxes"), py::arg("scores"),
                  py::arg("iou_threshold") = 0.5,
                  "Non-Maximum Suppression");

    detection.def("batched_nms", &tenzor::ops::batched_nms,
                  py::arg("boxes"), py::arg("scores"),
                  py::arg("iou_threshold") = 0.5,
                  py::arg("score_threshold") = 0.05,
                  py::arg("max_output_boxes") = 100,
                  "Batched NMS for multiple classes");

    detection.def("encode_boxes", &tenzor::ops::encode_boxes,
                  py::arg("boxes"), py::arg("anchors"),
                  py::arg("weights") = std::vector<double>{1.0, 1.0, 1.0, 1.0},
                  "Encode boxes relative to anchors");

    detection.def("decode_boxes", &tenzor::ops::decode_boxes,
                  py::arg("deltas"), py::arg("anchors"),
                  py::arg("weights") = std::vector<double>{1.0, 1.0, 1.0, 1.0},
                  "Decode boxes from deltas and anchors");

    detection.def("clip_boxes_to_image", &tenzor::ops::clip_boxes_to_image,
                  py::arg("boxes"), py::arg("height"), py::arg("width"),
                  "Clip boxes to image boundaries");

    detection.def("remove_small_boxes", &tenzor::ops::remove_small_boxes,
                  py::arg("boxes"), py::arg("scores"), py::arg("min_size"),
                  "Remove boxes smaller than min_size");

    // =========================================================================
    // Async ops
    // =========================================================================
    auto async_ops = m.def_submodule("async_ops", "Asynchronous tensor operations");

    // Each async op returns Future<Tensor>. Python consumers get a blocking
    // Tensor return here — the future is awaited on the C++ side. A true
    // Future<T> binding is deferred until there's a Python user that actually
    // wants to overlap compute; today every call site just needs the result.
    async_ops.def("async_matmul", [](const Tensor& a, const Tensor& b) {
        return tenzor::async_matmul(a, b).wait();
    }, py::arg("a"), py::arg("b"), "Asynchronous matrix multiplication (blocks on completion)");
    async_ops.def("async_add", [](const Tensor& a, const Tensor& b) {
        return tenzor::async_add(a, b).wait();
    }, py::arg("a"), py::arg("b"), "Asynchronous element-wise addition (blocks on completion)");
    async_ops.def("async_mul", [](const Tensor& a, const Tensor& b) {
        return tenzor::async_mul(a, b).wait();
    }, py::arg("a"), py::arg("b"), "Asynchronous element-wise multiplication (blocks on completion)");
    async_ops.def("async_sub", [](const Tensor& a, const Tensor& b) {
        return tenzor::async_sub(a, b).wait();
    }, py::arg("a"), py::arg("b"), "Asynchronous element-wise subtraction (blocks on completion)");
    async_ops.def("async_div", [](const Tensor& a, const Tensor& b) {
        return tenzor::async_div(a, b).wait();
    }, py::arg("a"), py::arg("b"), "Asynchronous element-wise division (blocks on completion)");
    async_ops.def("async_relu", [](const Tensor& input) {
        return tenzor::async_relu(input).wait();
    }, py::arg("input"), "Asynchronous ReLU activation (blocks on completion)");
    async_ops.def("async_sigmoid", [](const Tensor& input) {
        return tenzor::async_sigmoid(input).wait();
    }, py::arg("input"), "Asynchronous sigmoid activation (blocks on completion)");
    async_ops.def("async_tanh", [](const Tensor& input) {
        return tenzor::async_tanh(input).wait();
    }, py::arg("input"), "Asynchronous tanh activation (blocks on completion)");
    async_ops.def("async_softmax", [](const Tensor& input, int64_t dim) {
        return tenzor::async_softmax(input, dim).wait();
    }, py::arg("input"), py::arg("dim") = -1, "Asynchronous softmax (blocks on completion)");

    // =========================================================================
    // Fused ops
    // =========================================================================
    auto fused = m.def_submodule("fused", "Fused kernel operations");

    fused.def("fused_linear_relu", &tenzor::ops::fused_linear_relu,
              py::arg("input"), py::arg("weight"), py::arg("bias") = nullptr,
              "Fused linear + ReLU (1.5-2x faster)");
    fused.def("fused_conv2d_relu", &tenzor::ops::fused_conv2d_relu,
              py::arg("input"), py::arg("weight"), py::arg("bias") = nullptr,
              py::arg("stride") = 1, py::arg("padding") = 0,
              "Fused conv2d + ReLU (1.8-2.5x faster)");
    fused.def("fused_batchnorm_relu", &tenzor::ops::fused_batchnorm_relu,
              py::arg("input"), py::arg("running_mean"), py::arg("running_var"),
              py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5f,
              "Fused batchnorm + ReLU (1.6-2.2x faster)");
    fused.def("fused_softmax_cross_entropy", &tenzor::ops::fused_softmax_cross_entropy,
              py::arg("logits"), py::arg("targets"), py::arg("reduction") = "mean",
              "Fused softmax + cross-entropy (2-3x faster, 50% less memory)");
    fused.def("fused_add_relu", &tenzor::ops::fused_add_relu,
              py::arg("a"), py::arg("b"),
              "Fused add + ReLU for residual connections");
    fused.def("fused_gelu", &tenzor::ops::fused_gelu,
              py::arg("input"),
              "Fused GELU activation (1.5x faster)");
    fused.def("fused_layer_norm", &tenzor::ops::fused_layer_norm,
              py::arg("input"), py::arg("normalized_shape"),
              py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5f,
              "Fused layer normalization (1.4-2x faster)");
}

} // namespace tenzor::python
