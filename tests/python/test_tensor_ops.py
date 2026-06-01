#!/usr/bin/env python3
"""
Test Python bindings for tensor creation, math, reductions, indexing, and dtype casting.
"""

import sys
import os

build_python_dir = os.path.join(os.path.dirname(__file__), '../../build/python')
sys.path.insert(0, build_python_dir)

import tenzor.tenzor_core as tz


def test_creation_zeros():
    """Test zeros creation."""
    print("Testing zeros...")
    t = tz.zeros([2, 3])
    assert t.shape == [2, 3], f"Wrong shape: {t.shape}"
    assert t.dtype == tz.dtype.float32, f"Wrong dtype: {t.dtype}"

    t64 = tz.zeros([4], tz.dtype.float64)
    assert t64.dtype == tz.dtype.float64
    print("  zeros OK")


def test_creation_ones():
    """Test ones creation."""
    print("Testing ones...")
    t = tz.ones([3, 4])
    assert t.shape == [3, 4]
    print("  ones OK")


def test_creation_full():
    """Test full creation."""
    print("Testing full...")
    t = tz.full([2, 2], 3.14)
    assert t.shape == [2, 2]
    print("  full OK")


def test_creation_randn():
    """Test randn creation."""
    print("Testing randn...")
    tz.manual_seed(42)
    t = tz.randn([5, 5])
    assert t.shape == [5, 5]
    print("  randn OK")


def test_creation_rand():
    """Test rand creation."""
    print("Testing rand...")
    t = tz.rand([3, 3])
    assert t.shape == [3, 3]
    print("  rand OK")


def test_creation_eye():
    """Test eye creation."""
    print("Testing eye...")
    t = tz.eye(4)
    assert t.shape == [4, 4]
    print("  eye OK")


def test_creation_arange():
    """Test arange creation."""
    print("Testing arange...")
    t = tz.arange(0, 10, 1)
    assert t.shape == [10]
    print("  arange OK")


def test_creation_linspace():
    """Test linspace creation."""
    print("Testing linspace...")
    t = tz.linspace(0, 1, 5)
    assert t.shape == [5]
    print("  linspace OK")


def test_creation_empty():
    """Test empty creation."""
    print("Testing empty...")
    t = tz.empty([2, 3])
    assert t.shape == [2, 3]
    print("  empty OK")


def test_creation_randperm():
    """Test randperm creation."""
    import numpy as np
    print("Testing randperm...")
    t = tz.randperm(10)
    assert t.shape == [10]
    # Must be a valid permutation of [0, 10), not the unshuffled identity.
    arr = np.sort(np.asarray(t.tensor() if hasattr(t, "tensor") else t).astype(np.int64).ravel())
    assert np.array_equal(arr, np.arange(10)), "randperm is not a valid permutation"

    # GPU randperm must produce a real on-device permutation (regression: the
    # old non-CPU path silently returned the unshuffled arange identity).
    for dev in ("vulkan", "cuda", "rocm", "oneapi"):
        try:
            g = tz.randperm(64, dev)
        except Exception:
            continue  # backend not built/available in this environment
        garr = np.asarray((g.tensor() if hasattr(g, "tensor") else g).cpu()).astype(np.int64).ravel()
        assert np.array_equal(np.sort(garr), np.arange(64)), f"{dev} randperm not a permutation"
        assert not np.array_equal(garr, np.arange(64)), f"{dev} randperm returned identity"
        print(f"  randperm {dev} OK")
    print("  randperm OK")


def test_creation_meshgrid():
    """meshgrid must match numpy and work on every device (regression: the old
    host memcpy loop dereferenced device pointers and crashed on GPU tensors)."""
    import numpy as np
    print("Testing meshgrid...")
    x = tz.arange(0, 3, 1)
    y = tz.arange(0, 4, 1)
    for indexing in ("ij", "xy"):
        gx, gy = tz.meshgrid([x, y], indexing)
        nx, ny = np.meshgrid(np.arange(3), np.arange(4), indexing=indexing)
        assert np.array_equal(np.asarray(gx).astype(int), nx)
        assert np.array_equal(np.asarray(gy).astype(int), ny)
    # On-device meshgrid: must not crash and must match the CPU/numpy result.
    refx, refy = np.meshgrid(np.arange(3), np.arange(4), indexing="ij")
    for dev in ("vulkan", "cuda", "rocm", "oneapi"):
        try:
            xg = tz.arange(0, 3, 1).to(dev)
            yg = tz.arange(0, 4, 1).to(dev)
        except Exception:
            continue
        gx, gy = tz.meshgrid([xg, yg], "ij")
        assert np.array_equal(np.asarray(gx.cpu()).astype(int), refx), f"{dev} meshgrid x"
        assert np.array_equal(np.asarray(gy.cpu()).astype(int), refy), f"{dev} meshgrid y"
        print(f"  meshgrid {dev} OK")
    print("  meshgrid OK")


def test_math_basic():
    """Test basic math operations."""
    print("Testing basic math ops...")
    a = tz.ones([2, 3])
    b = tz.ones([2, 3])

    c = tz.add(a, b)
    assert c.shape == [2, 3]

    d = tz.sub(a, b)
    assert d.shape == [2, 3]

    e = tz.mul(a, b)
    assert e.shape == [2, 3]

    f = tz.div(a, b)
    assert f.shape == [2, 3]
    print("  basic math OK")


def test_math_unary():
    """Test unary math operations."""
    print("Testing unary math ops...")
    t = tz.ones([3, 3])

    for op_name in ['sqrt', 'exp', 'log', 'abs', 'neg', 'sign',
                     'sigmoid', 'reciprocal', 'sin', 'cos', 'tanh',
                     'floor', 'ceil', 'round']:
        op = getattr(tz, op_name)
        result = op(t)
        assert result.shape == [3, 3], f"{op_name} wrong shape"
    print("  unary math OK")


def test_math_matmul():
    """Test matrix multiplication."""
    print("Testing matmul...")
    a = tz.randn([2, 3])
    b = tz.randn([3, 4])
    c = tz.matmul(a, b)
    assert c.shape == [2, 4], f"Wrong shape: {c.shape}"
    print("  matmul OK")


def test_reductions():
    """Test reduction operations."""
    print("Testing reductions...")
    t = tz.randn([4, 5])

    s = tz.sum(t)
    m = tz.mean(t)

    mx = tz.max(t)
    mn = tz.min(t)
    print("  reductions OK")


def test_argmax_argmin():
    """Test argmax/argmin."""
    print("Testing argmax/argmin...")
    t = tz.randn([3, 4])
    am = tz.argmax(t)
    ai = tz.argmin(t)
    print("  argmax/argmin OK")


def test_comparison_ops():
    """Test comparison operations."""
    print("Testing comparison ops...")
    a = tz.ones([2, 3])
    b = tz.zeros([2, 3])

    for op_name in ['eq', 'ne', 'lt', 'le', 'gt', 'ge']:
        op = getattr(tz, op_name)
        result = op(a, b)
        assert result.shape == [2, 3], f"{op_name} wrong shape"
    print("  comparison ops OK")


def test_logical_ops():
    """Test logical operations."""
    print("Testing logical ops...")
    a = tz.ones([2, 3])
    b = tz.zeros([2, 3])

    r1 = tz.logical_and(a, b)
    r2 = tz.logical_or(a, b)
    r3 = tz.logical_not(a)
    r4 = tz.logical_xor(a, b)
    print("  logical ops OK")


def test_classification_ops():
    """Test isnan/isinf/isfinite."""
    print("Testing isnan/isinf/isfinite...")
    t = tz.ones([2, 3])
    r1 = tz.isnan(t)
    r2 = tz.isinf(t)
    r3 = tz.isfinite(t)
    print("  classification ops OK")


def test_shape_ops():
    """Test shape manipulation operations."""
    print("Testing shape ops...")
    t = tz.randn([2, 3, 4])

    r = tz.reshape(t, [6, 4])
    assert r.shape == [6, 4]

    tr = tz.transpose(t, 0, 1)
    assert tr.shape == [3, 2, 4]

    f = tz.flatten(t)
    assert f.shape == [24]

    u = tz.unsqueeze(t, 0)
    assert u.shape == [1, 2, 3, 4]

    s = tz.squeeze(u, 0)
    assert s.shape == [2, 3, 4]
    print("  shape ops OK")


def test_cat_stack():
    """Test cat and stack."""
    print("Testing cat/stack...")
    a = tz.randn([2, 3])
    b = tz.randn([2, 3])

    c = tz.cat([a, b], 0)
    assert c.shape == [4, 3]

    s = tz.stack([a, b], 0)
    assert s.shape == [2, 2, 3]
    print("  cat/stack OK")


def test_clamp():
    """Test clamp operations."""
    print("Testing clamp...")
    t = tz.randn([3, 3])
    c = tz.clamp(t, -0.5, 0.5)
    assert c.shape == [3, 3]
    print("  clamp OK")


def test_cumulative_ops():
    """Test cumsum and cumprod."""
    print("Testing cumulative ops...")
    t = tz.ones([3, 4])
    cs = tz.cumsum(t, 0)
    assert cs.shape == [3, 4]
    cp = tz.cumprod(t, 0)
    assert cp.shape == [3, 4]
    print("  cumulative ops OK")


def test_indexing_ops():
    """Test indexing operations."""
    print("Testing indexing ops...")
    t = tz.randn([4, 5])

    # index_select
    idx = tz.Tensor([3], tz.dtype.int64)
    r = tz.index_select(t, 0, idx)
    assert r.shape[1] == 5

    # where
    cond = tz.gt(t, tz.zeros([4, 5]))
    w = tz.where(cond, t, tz.zeros([4, 5]))
    assert w.shape == [4, 5]
    print("  indexing ops OK")


def test_triu_tril_diag():
    """Test triangle and diagonal ops."""
    print("Testing triu/tril/diag...")
    t = tz.ones([4, 4])
    u = tz.triu(t)
    l = tz.tril(t)
    assert u.shape == [4, 4]
    assert l.shape == [4, 4]

    d = tz.diag(tz.ones([4]))
    assert d.shape == [4, 4]
    print("  triu/tril/diag OK")


def test_sort_topk():
    """Test sort and topk."""
    print("Testing sort/topk...")
    t = tz.randn([3, 5])
    s = tz.sort(t, -1)
    tk = tz.topk(t, 2)
    print("  sort/topk OK")


def test_dtype_casting():
    """Test dtype conversions."""
    print("Testing dtype casting...")
    t = tz.ones([2, 3], tz.dtype.float32)

    t64 = t.to(tz.dtype.float64)
    assert t64.dtype == tz.dtype.float64

    t32 = t64.to(tz.dtype.float32)
    assert t32.dtype == tz.dtype.float32
    print("  dtype casting OK")


def main():
    print("=" * 60)
    print("Testing Tensor Operations Bindings")
    print("=" * 60)

    try:
        tz.initialize()

        test_creation_zeros()
        test_creation_ones()
        test_creation_full()
        test_creation_randn()
        test_creation_rand()
        test_creation_eye()
        test_creation_arange()
        test_creation_linspace()
        test_creation_empty()
        test_creation_randperm()
        test_creation_meshgrid()
        test_math_basic()
        test_math_unary()
        test_math_matmul()
        test_reductions()
        test_argmax_argmin()
        test_comparison_ops()
        test_logical_ops()
        test_classification_ops()
        test_shape_ops()
        test_cat_stack()
        test_clamp()
        test_cumulative_ops()
        test_indexing_ops()
        test_triu_tril_diag()
        test_sort_topk()
        test_dtype_casting()

        print("\n" + "=" * 60)
        print("ALL TENSOR OPS TESTS PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
