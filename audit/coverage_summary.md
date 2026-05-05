# Tenzor Coverage Summary

Build directory: `build-cov`
Source files measured: 489
`.gcda` files (executed objects): 896
`.gcno` files (compiled objects): 1009

## Overall

**Line coverage: 61190 / 119577 = 51.17%**

## Per-subsystem

| Subsystem | Files | Executable | Executed | Coverage |
|---|---:|---:|---:|---:|
| `(other)` | 181 | 8806 | 4101 | 46.6% |
| `src/autograd` | 31 | 10327 | 5211 | 50.5% |
| `src/backend` | 7 | 927 | 330 | 35.6% |
| `src/backends/cpu` | 51 | 33129 | 13023 | 39.3% |
| `src/core` | 17 | 6366 | 3094 | 48.6% |
| `src/data` | 7 | 1542 | 247 | 16.0% |
| `src/distributed` | 23 | 3149 | 895 | 28.4% |
| `src/export` | 1 | 181 | 158 | 87.3% |
| `src/io` | 1 | 133 | 108 | 81.2% |
| `src/jit` | 16 | 6037 | 2843 | 47.1% |
| `src/lazy` | 1 | 241 | 138 | 57.3% |
| `src/lite` | 3 | 232 | 94 | 40.5% |
| `src/models` | 21 | 5695 | 3397 | 59.6% |
| `src/nested` | 2 | 562 | 223 | 39.7% |
| `src/nn` | 93 | 26566 | 17998 | 67.7% |
| `src/onnx` | 3 | 3585 | 990 | 27.6% |
| `src/ops` | 18 | 9000 | 6215 | 69.1% |
| `src/quantization` | 2 | 402 | 172 | 42.8% |
| `src/serving` | 2 | 226 | 177 | 78.3% |
| `src/sparse` | 2 | 1673 | 1219 | 72.9% |
| `src/utils` | 7 | 798 | 557 | 69.8% |

## Top 30 files by uncovered-line count

| File | Executable | Executed | Coverage | Uncovered |
|---|---:|---:|---:|---:|
| `src/backends/cpu/kernels/broadcast.hpp` | 7500 | 697 | 9.3% | 6803 |
| `src/backends/cpu/kernels/math.cpp` | 5296 | 2508 | 47.4% | 2788 |
| `include/tenzor/io/stb/stb_image.h` | 3473 | 947 | 27.3% | 2526 |
| `src/core/tensor.cpp` | 3879 | 1533 | 39.5% | 2346 |
| `src/autograd/ops.cpp` | 2240 | 340 | 15.2% | 1900 |
| `src/onnx/exporter.cpp` | 2409 | 595 | 24.7% | 1814 |
| `src/backends/cpu/cpu_kernel_registry.cpp` | 2256 | 602 | 26.7% | 1654 |
| `src/nn/quantization/quantized_layers.cpp` | 1267 | 165 | 13.0% | 1102 |
| `src/backends/cpu/kernels/reduction.cpp` | 2371 | 1279 | 53.9% | 1092 |
| `src/jit/graph.cpp` | 1323 | 345 | 26.1% | 978 |
| `src/data/transforms.cpp` | 928 | 11 | 1.2% | 917 |
| `src/backends/cpu/kernels/indexing.cpp` | 1521 | 684 | 45.0% | 837 |
| `src/nn/optim/zero_optimizer.cpp` | 1430 | 601 | 42.0% | 829 |
| `src/backends/cpu/kernels/advanced.cpp` | 1284 | 486 | 37.9% | 798 |
| `src/jit/compiler.cpp` | 1374 | 625 | 45.5% | 749 |
| `src/onnx/importer.cpp` | 1126 | 395 | 35.1% | 731 |
| `src/backends/cpu/kernels/activations.cpp` | 1223 | 521 | 42.6% | 702 |
| `src/backends/cpu/cpu_backend.cpp` | 696 | 44 | 6.3% | 652 |
| `include/tenzor/distributions/distribution.hpp` | 1199 | 627 | 52.3% | 572 |
| `src/backends/cpu/kernels/nn_kernels.cpp` | 1008 | 442 | 43.8% | 566 |
| `src/ops/transform.cpp` | 1304 | 777 | 59.6% | 527 |
| `src/models/mask_rcnn.cpp` | 499 | 0 | 0.0% | 499 |
| `src/distributed/gloo_backend.cpp` | 572 | 115 | 20.1% | 457 |
| `src/ops/creation.cpp` | 768 | 312 | 40.6% | 456 |
| `src/backends/cpu/kernels/rnn_onednn.hpp` | 598 | 144 | 24.1% | 454 |
| `src/backends/cpu/kernels/pooling.cpp` | 1246 | 792 | 63.6% | 454 |
| `src/nn/functional.cpp` | 1125 | 677 | 60.2% | 448 |
| `src/autograd/function_linalg.cpp` | 679 | 259 | 38.1% | 420 |
| `src/nn/layers/conv.cpp` | 1370 | 958 | 69.9% | 412 |
| `src/autograd/function_new_ops.cpp` | 947 | 539 | 56.9% | 408 |

## Files with 0% line coverage (72)

| File | Executable lines |
|---|---:|
| `src/models/mask_rcnn.cpp` | 499 |
| `src/backends/cpu/kernels/fused_lstm.hpp` | 261 |
| `src/jit/symbolic_shape_inference.cpp` | 189 |
| `src/distributed/dist_checkpoint.cpp` | 184 |
| `src/backend/op_attributes.cpp` | 183 |
| `src/core/numa.cpp` | 151 |
| `src/jit/autotune.cpp` | 148 |
| `src/nn/detection/mask_head.cpp` | 148 |
| `src/distributed/ddp.cpp` | 147 |
| `src/models/deeplabv3plus.cpp` | 145 |
| `src/core/named_tensor.cpp` | 131 |
| `src/backend/runtime_simd.cpp` | 131 |
| `src/nn/parallel/data_parallel.cpp` | 130 |
| `src/models/unet.cpp` | 110 |
| `src/utils/monitor.cpp` | 92 |
| `src/nn/training.cpp` | 90 |
| `src/data/sampler.cpp` | 90 |
| `src/core/masked_tensor.cpp` | 88 |
| `src/autograd/function_helpers.hpp` | 83 |
| `src/data/datasets/cifar10.cpp` | 81 |
| `src/distributed/rpc/rpc.cpp` | 68 |
| `src/data/datasets/mnist.cpp` | 62 |
| `src/distributed/sequence_parallel.cpp` | 57 |
| `include/tenzor/models/electra.hpp` | 56 |
| `include/tenzor/nn/layers/conv.hpp` | 53 |
| `src/data/datasets/imagenet.cpp` | 53 |
| `src/onnx/graph_module.cpp` | 50 |
| `src/jit/fusion_cost_model.cpp` | 45 |
| `include/tenzor/backend/fast_dispatch.hpp` | 42 |
| `src/backends/cpu/kernels/fused_conv_bn_relu.hpp` | 39 |
| _… and 42 more_ |  |

## Compiled objects with no execution data (113)

These have a `.gcno` from compilation but no `.gcda` — the linker pulled them in but no test exercised the code.

- `src/CMakeFiles/tenzor_core.dir/backend/backend.cpp.gcno`
- `src/CMakeFiles/tenzor_core.dir/backend/registry.cpp.gcno`
- `src/CMakeFiles/tenzor_core.dir/core/shape.cpp.gcno`
- `src/CMakeFiles/tenzor_core.dir/distributed/nccl_backend.cpp.gcno`
- `src/CMakeFiles/tenzor_core.dir/utils/error.cpp.gcno`
- `tests/CMakeFiles/benchmark_backends.dir/benchmarks/benchmark_backends.cpp.gcno`
- `tests/CMakeFiles/benchmark_suite.dir/benchmarks/benchmark_suite.cpp.gcno`
- `tests/CMakeFiles/check_float32_gradients.dir/check_float32_gradients.cpp.gcno`
- `tests/CMakeFiles/debug_dataloader.dir/debug_dataloader.cpp.gcno`
- `tests/CMakeFiles/minimal_swin_debug.dir/minimal_swin_debug.cpp.gcno`
- `tests/CMakeFiles/tenzor_simd_benchmark.dir/unit/benchmark_simd.cpp.gcno`
- `tests/CMakeFiles/vulkan_add_debug.dir/vulkan_add_debug.cpp.gcno`
- `tests/CMakeFiles/vulkan_diagnostic.dir/vulkan_diagnostic.cpp.gcno`
- `tests/CMakeFiles/vulkan_tensor_test.dir/vulkan_tensor_test.cpp.gcno`
- `tests/backend_parity/CMakeFiles/test_advanced_index_parity.dir/test_advanced_index_parity.cpp.gcno`
- `tests/backend_parity/CMakeFiles/test_amp_parity.dir/test_amp_parity.cpp.gcno`
- `tests/backend_parity/CMakeFiles/test_backend_stress.dir/test_backend_stress.cpp.gcno`
- `tests/backend_parity/CMakeFiles/test_bf16_parity.dir/test_bf16_parity.cpp.gcno`
- `tests/backend_parity/CMakeFiles/test_bitwise_parity.dir/test_bitwise_parity.cpp.gcno`
- `tests/backend_parity/CMakeFiles/test_comparison_parity.dir/test_comparison_parity.cpp.gcno`
- `tests/backend_parity/CMakeFiles/test_complex_parity.dir/test_complex_parity.cpp.gcno`
- `tests/backend_parity/CMakeFiles/test_cross_backend_pairs.dir/test_cross_backend_pairs.cpp.gcno`
- `tests/backend_parity/CMakeFiles/test_cumulative_parity.dir/test_cumulative_parity.cpp.gcno`
- `tests/backend_parity/CMakeFiles/test_custom_op_parity.dir/test_custom_op_parity.cpp.gcno`
- `tests/backend_parity/CMakeFiles/test_deformable_conv2d_backward_parity.dir/test_deformable_conv2d_backward_parity.cpp.gcno`
- `tests/backend_parity/CMakeFiles/test_distributions_parity.dir/test_distributions_parity.cpp.gcno`
- `tests/backend_parity/CMakeFiles/test_dtype_parity.dir/test_dtype_parity.cpp.gcno`
- `tests/backend_parity/CMakeFiles/test_embedding_bag_backward_parity.dir/test_embedding_bag_backward_parity.cpp.gcno`
- `tests/backend_parity/CMakeFiles/test_extended_math_parity.dir/test_extended_math_parity.cpp.gcno`
- `tests/backend_parity/CMakeFiles/test_fft_parity.dir/test_fft_parity.cpp.gcno`
- _… and 83 more_
