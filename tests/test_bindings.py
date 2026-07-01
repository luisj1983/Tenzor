#!/usr/bin/env python3
"""
Test Python bindings for activations, losses, and tensor operations.
Tests all newly added bindings to ensure they work correctly.
"""

import sys
import os
import numpy as np

# tenzor_core.so lives under build/python/tenzor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'python', 'tenzor'))

# Import tenzor
try:
    import tenzor_core as tz
except ImportError:
    print("Error: Could not import tenzor_core module")
    print("Please build the Python bindings first")
    sys.exit(1)

def test_tensor_operations():
    """Test additional tensor operations."""
    print("\n=== Testing Tensor Operations ===")

    # Initialize library
    tz.initialize()

    # Test shape manipulation
    print("Testing shape manipulation...")
    t = tz.randn([2, 3, 4])

    # Transpose
    t_trans = t.transpose(0, 2)
    print(f"  transpose(0, 2): {list(t.shape)} -> {list(t_trans.shape)}")
    assert list(t_trans.shape) == [4, 3, 2], "Transpose failed"

    # Permute
    t_perm = t.permute([2, 0, 1])
    print(f"  permute([2, 0, 1]): {list(t.shape)} -> {list(t_perm.shape)}")
    assert list(t_perm.shape) == [4, 2, 3], "Permute failed"

    # Squeeze
    t_sq = tz.randn([1, 3, 1, 4])
    t_squeezed = t_sq.squeeze()
    print(f"  squeeze(): {list(t_sq.shape)} -> {list(t_squeezed.shape)}")
    assert list(t_squeezed.shape) == [3, 4], "Squeeze all failed"

    t_squeezed_dim = t_sq.squeeze(0)
    print(f"  squeeze(0): {list(t_sq.shape)} -> {list(t_squeezed_dim.shape)}")
    assert list(t_squeezed_dim.shape) == [3, 1, 4], "Squeeze dim failed"

    # Unsqueeze
    t_unsq = t.unsqueeze(0)
    print(f"  unsqueeze(0): {list(t.shape)} -> {list(t_unsq.shape)}")
    assert list(t_unsq.shape) == [1, 2, 3, 4], "Unsqueeze failed"

    # Flatten
    t_flat = t.flatten()
    print(f"  flatten(): {list(t.shape)} -> {list(t_flat.shape)}")
    assert list(t_flat.shape) == [24], "Flatten all failed"

    t_flat_partial = t.flatten(1, 2)
    print(f"  flatten(1, 2): {list(t.shape)} -> {list(t_flat_partial.shape)}")
    assert list(t_flat_partial.shape) == [2, 12], "Flatten partial failed"

    # Memory operations
    print("Testing memory operations...")
    t_clone = t.clone()
    print(f"  clone(): shape {list(t_clone.shape)}")

    t_detach = t.detach()
    print(f"  detach(): shape {list(t_detach.shape)}")

    t_contig = t.contiguous()
    print(f"  contiguous(): is_contiguous = {t_contig.is_contiguous}")

    # Item extraction
    print("Testing item extraction...")
    scalar = tz.randn([1])
    scalar_val = scalar.item()
    print(f"  item(): extracted {scalar_val} (type: {type(scalar_val).__name__})")

    print("Tensor operations: PASSED")

def test_math_operations():
    """Test math operations."""
    print("\n=== Testing Math Operations ===")

    t = tz.randn([3, 4])

    print("Testing element-wise math...")
    t_exp = tz.exp(t)
    print(f"  exp(): shape {list(t_exp.shape)}")

    t_abs = tz.abs(t)
    print(f"  abs(): shape {list(t_abs.shape)}")

    t_sqrt = tz.sqrt(tz.abs(t))  # sqrt of abs to avoid negative
    print(f"  sqrt(): shape {list(t_sqrt.shape)}")

    t_log = tz.log(tz.abs(t) + 1.0)  # log of abs+1 to avoid negative/zero
    print(f"  log(): shape {list(t_log.shape)}")

    print("Math operations: PASSED")

def test_reduction_operations():
    """Test reduction operations."""
    print("\n=== Testing Reduction Operations ===")

    t = tz.randn([3, 4, 5])

    print("Testing reductions...")
    t_sum = tz.sum(t)
    print(f"  sum(all): shape {list(t_sum.shape)}")

    t_sum_dim = tz.sum(t, dim=1, keepdim=True)
    print(f"  sum(dim=1, keepdim=True): {list(t.shape)} -> {list(t_sum_dim.shape)}")

    t_mean = tz.mean(t, dim=0)
    print(f"  mean(dim=0): {list(t.shape)} -> {list(t_mean.shape)}")
    assert list(t_mean.shape) == [4, 5], "Mean reduction failed"

    t_max = tz.max(t, dim=2, keepdim=False)
    print(f"  max(dim=2): {list(t.shape)} -> {list(t_max.shape)}")
    assert list(t_max.shape) == [3, 4], "Max reduction failed"

    t_min = tz.min(t)
    print(f"  min(all): shape {list(t_min.shape)}")

    print("Reduction operations: PASSED")

def test_activation_functions():
    """Test activation function classes and functional forms."""
    print("\n=== Testing Activation Functions ===")

    # Create dummy input
    x = tz.Variable(tz.randn([2, 10]), requires_grad=True)

    # Test module-based activations
    print("Testing activation modules...")

    relu = tz.nn.ReLU()
    out_relu = relu(x)
    print(f"  ReLU: input {list(x.data.shape)} -> output {list(out_relu.data.shape)}")

    leaky_relu = tz.nn.LeakyReLU(negative_slope=0.01)
    out_lrelu = leaky_relu(x)
    print(f"  LeakyReLU(0.01): output shape {list(out_lrelu.data.shape)}")

    elu = tz.nn.ELU(alpha=1.0)
    out_elu = elu(x)
    print(f"  ELU(alpha=1.0): output shape {list(out_elu.data.shape)}")

    gelu = tz.nn.GELU()
    out_gelu = gelu(x)
    print(f"  GELU: output shape {list(out_gelu.data.shape)}")

    sigmoid = tz.nn.Sigmoid()
    out_sigmoid = sigmoid(x)
    print(f"  Sigmoid: output shape {list(out_sigmoid.data.shape)}")

    tanh = tz.nn.Tanh()
    out_tanh = tanh(x)
    print(f"  Tanh: output shape {list(out_tanh.data.shape)}")

    softmax = tz.nn.Softmax(dim=-1)
    out_softmax = softmax(x)
    print(f"  Softmax(dim=-1): output shape {list(out_softmax.data.shape)}")

    log_softmax = tz.nn.LogSoftmax(dim=-1)
    out_log_softmax = log_softmax(x)
    print(f"  LogSoftmax(dim=-1): output shape {list(out_log_softmax.data.shape)}")

    selu = tz.nn.SELU()
    out_selu = selu(x)
    print(f"  SELU: output shape {list(out_selu.data.shape)}")

    swish = tz.nn.Swish()
    out_swish = swish(x)
    print(f"  Swish: output shape {list(out_swish.data.shape)}")

    mish = tz.nn.Mish()
    out_mish = mish(x)
    print(f"  Mish: output shape {list(out_mish.data.shape)}")

    # Test functional activations
    print("Testing functional activations...")

    out_f_relu = tz.nn.relu(x)
    print(f"  nn.relu(): output shape {list(out_f_relu.data.shape)}")

    out_f_leaky = tz.nn.leaky_relu(x, negative_slope=0.02)
    print(f"  nn.leaky_relu(negative_slope=0.02): output shape {list(out_f_leaky.data.shape)}")

    out_f_gelu = tz.nn.gelu(x)
    print(f"  nn.gelu(): output shape {list(out_f_gelu.data.shape)}")

    out_f_sigmoid = tz.nn.sigmoid(x)
    print(f"  nn.sigmoid(): output shape {list(out_f_sigmoid.data.shape)}")

    out_f_softmax = tz.nn.softmax(x, dim=1)
    print(f"  nn.softmax(dim=1): output shape {list(out_f_softmax.data.shape)}")

    print("Activation functions: PASSED")

def test_loss_functions():
    """Test loss function classes and functional forms."""
    print("\n=== Testing Loss Functions ===")

    # Create dummy data
    pred = tz.Variable(tz.randn([4, 10]), requires_grad=True)
    target_reg = tz.Variable(tz.randn([4, 10]), requires_grad=False)
    target_cls = tz.zeros([4], dtype=tz.dtype.int64)

    # Test loss modules
    print("Testing loss modules...")

    # MSE Loss
    mse_loss = tz.nn.MSELoss(reduction=tz.nn.Reduction.mean)
    loss_mse = mse_loss(pred, target_reg)
    print(f"  MSELoss(mean): loss shape {list(loss_mse.data.shape)}")

    mse_loss_sum = tz.nn.MSELoss(reduction=tz.nn.Reduction.sum)
    loss_mse_sum = mse_loss_sum(pred, target_reg)
    print(f"  MSELoss(sum): loss shape {list(loss_mse_sum.data.shape)}")

    # L1 Loss
    l1_loss = tz.nn.L1Loss(reduction=tz.nn.Reduction.mean)
    loss_l1 = l1_loss(pred, target_reg)
    print(f"  L1Loss(mean): loss shape {list(loss_l1.data.shape)}")

    # Smooth L1 Loss
    smooth_l1 = tz.nn.SmoothL1Loss(reduction=tz.nn.Reduction.mean, beta=1.0)
    loss_smooth = smooth_l1(pred, target_reg)
    print(f"  SmoothL1Loss(beta=1.0): loss shape {list(loss_smooth.data.shape)}")

    # Cross Entropy Loss
    ce_loss = tz.nn.CrossEntropyLoss(reduction=tz.nn.Reduction.mean)
    loss_ce = ce_loss(pred, target_cls)
    print(f"  CrossEntropyLoss(mean): loss shape {list(loss_ce.data.shape)}")

    # NLL Loss
    log_probs = tz.nn.log_softmax(pred, dim=1)
    nll_loss = tz.nn.NLLLoss(reduction=tz.nn.Reduction.mean)
    loss_nll = nll_loss(log_probs, target_cls)
    print(f"  NLLLoss(mean): loss shape {list(loss_nll.data.shape)}")

    # BCE Loss
    pred_binary = tz.nn.sigmoid(pred)
    target_binary = tz.Variable(tz.ones([4, 10]) * 0.5, requires_grad=False)
    bce_loss = tz.nn.BCELoss(reduction=tz.nn.Reduction.mean)
    loss_bce = bce_loss(pred_binary, target_binary)
    print(f"  BCELoss(mean): loss shape {list(loss_bce.data.shape)}")

    # BCE with Logits Loss
    bce_logits = tz.nn.BCEWithLogitsLoss(reduction=tz.nn.Reduction.mean)
    loss_bce_logits = bce_logits(pred, target_binary)
    print(f"  BCEWithLogitsLoss(mean): loss shape {list(loss_bce_logits.data.shape)}")

    # Test functional losses
    print("Testing functional losses...")

    loss_f_mse = tz.nn.mse_loss(pred, target_reg, reduction=tz.nn.Reduction.mean)
    print(f"  nn.mse_loss(): loss shape {list(loss_f_mse.data.shape)}")

    loss_f_l1 = tz.nn.l1_loss(pred, target_reg, reduction=tz.nn.Reduction.mean)
    print(f"  nn.l1_loss(): loss shape {list(loss_f_l1.data.shape)}")

    loss_f_ce = tz.nn.cross_entropy(pred, target_cls, reduction=tz.nn.Reduction.mean)
    print(f"  nn.cross_entropy(): loss shape {list(loss_f_ce.data.shape)}")

    loss_f_nll = tz.nn.nll_loss(log_probs, target_cls, reduction=tz.nn.Reduction.mean)
    print(f"  nn.nll_loss(): loss shape {list(loss_f_nll.data.shape)}")

    loss_f_bce = tz.nn.bce_loss(pred_binary, target_binary, reduction=tz.nn.Reduction.mean)
    print(f"  nn.bce_loss(): loss shape {list(loss_f_bce.data.shape)}")

    print("Loss functions: PASSED")

def test_optimizers():
    """Test optimizer enhancements."""
    print("\n=== Testing Optimizers ===")

    # Create dummy parameters
    w1 = tz.Variable(tz.randn([10, 5]), requires_grad=True)
    w2 = tz.Variable(tz.randn([5, 3]), requires_grad=True)
    params = [w1, w2]

    print("Testing SGD optimizer...")
    sgd = tz.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
    print(f"  Initial LR: {sgd.get_lr()}")

    sgd.set_lr(0.001)
    print(f"  After set_lr(0.001): {sgd.get_lr()}")

    state = sgd.state_dict()
    print(f"  state_dict() returned {len(state)} items")

    sgd.load_state_dict(state)
    print(f"  load_state_dict() successful")

    print("Testing Adam optimizer...")
    adam = tz.optim.Adam(params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)
    print(f"  Initial LR: {adam.get_lr()}")

    adam.set_lr(0.0001)
    print(f"  After set_lr(0.0001): {adam.get_lr()}")

    state_adam = adam.state_dict()
    print(f"  state_dict() returned {len(state_adam)} items")

    adam.load_state_dict(state_adam)
    print(f"  load_state_dict() successful")

    print("Testing AdamW optimizer...")
    adamw = tz.optim.AdamW(params, lr=0.001, beta1=0.9, beta2=0.999,
                           eps=1e-8, weight_decay=0.01)
    print(f"  Initial LR: {adamw.get_lr()}")

    adamw.set_lr(0.0005)
    print(f"  After set_lr(0.0005): {adamw.get_lr()}")

    state_adamw = adamw.state_dict()
    print(f"  state_dict() returned {len(state_adamw)} items")

    adamw.load_state_dict(state_adamw)
    print(f"  load_state_dict() successful")

    print("Optimizers: PASSED")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Tenzor Python Bindings")
    print("=" * 60)

    try:
        test_tensor_operations()
        test_math_operations()
        test_reduction_operations()
        test_activation_functions()
        test_loss_functions()
        test_optimizers()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
