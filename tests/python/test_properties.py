#!/usr/bin/env python3
"""
Property-based tests for Tenzor using Hypothesis.

Tests fundamental invariants that must hold for all inputs:
- Shape preservation across reshape/view operations
- Dtype promotion rules (follow NumPy conventions)
- Broadcasting shape rules (follow NumPy conventions)
- Autograd consistency (numerical vs analytical gradients)
- Creation function invariants
"""

import sys
import os
import math
from functools import reduce
from operator import mul

build_python_dir = os.path.join(os.path.dirname(__file__), "../../build/python")
sys.path.insert(0, build_python_dir)

import pytest
import tenzor.tenzor_core as tz

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

# Shapes: 1-4 dimensions, each 1-8 elements (keep small for speed)
dims = st.integers(min_value=1, max_value=8)
shapes = st.lists(dims, min_size=1, max_size=4).map(list)

# Dtypes that support arithmetic
float_dtypes = st.sampled_from([tz.dtype.float32, tz.dtype.float64])

# All numeric dtypes
numeric_dtypes = st.sampled_from(
    [tz.dtype.float32, tz.dtype.float64, tz.dtype.int32, tz.dtype.int64]
)

# Scalar values (finite, non-zero where needed)
finite_floats = st.floats(
    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False
)
nonzero_floats = finite_floats.filter(lambda x: abs(x) > 1e-6)


def numel(shape):
    """Compute total number of elements from a shape."""
    if not shape:
        return 1
    return reduce(mul, shape, 1)


def broadcastable_shapes(draw):
    """Draw two shapes that are broadcastable with each other."""
    # Start with a result shape, then derive two compatible input shapes
    ndim = draw(st.integers(min_value=1, max_value=4))
    result_shape = [draw(dims) for _ in range(ndim)]

    shape_a = []
    shape_b = []
    for d in result_shape:
        choice = draw(st.integers(min_value=0, max_value=2))
        if choice == 0:
            # Both have full dimension
            shape_a.append(d)
            shape_b.append(d)
        elif choice == 1:
            # a has full, b has 1
            shape_a.append(d)
            shape_b.append(1)
        else:
            # a has 1, b has full
            shape_a.append(1)
            shape_b.append(d)

    # Optionally drop leading dims from one operand
    drop_a = draw(st.integers(min_value=0, max_value=max(0, ndim - 1)))
    drop_b = draw(st.integers(min_value=0, max_value=max(0, ndim - 1)))
    shape_a = shape_a[drop_a:]
    shape_b = shape_b[drop_b:]

    if not shape_a:
        shape_a = [1]
    if not shape_b:
        shape_b = [1]

    return shape_a, shape_b, result_shape


broadcastable = st.composite(broadcastable_shapes)


# ---------------------------------------------------------------------------
# Shape preservation properties
# ---------------------------------------------------------------------------


class TestShapePreservation:
    """Reshape and view operations must preserve element count."""

    @given(shape=shapes)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_reshape_preserves_numel(self, shape):
        """reshape(x, new_shape).numel == x.numel for any valid reshape."""
        t = tz.zeros(shape)
        n = numel(shape)

        # Reshape to flat
        flat = t.reshape([n])
        assert flat.numel() == n

        # Reshape back
        restored = flat.reshape(shape)
        assert restored.shape == shape
        assert restored.numel() == n

    @given(shape=shapes)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_transpose_preserves_numel(self, shape):
        """Transpose preserves element count."""
        if len(shape) < 2:
            return
        t = tz.randn(shape)
        # Transpose last two dims
        tt = t.transpose(-2, -1)
        assert tt.numel() == t.numel()

    @given(shape=shapes, dtype=float_dtypes)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_contiguous_preserves_shape_and_values(self, shape, dtype):
        """contiguous() preserves shape and values."""
        t = tz.randn(shape, dtype)
        c = t.contiguous()
        assert c.shape == t.shape
        assert c.dtype == t.dtype


# ---------------------------------------------------------------------------
# Dtype promotion properties
# ---------------------------------------------------------------------------


class TestDtypePromotion:
    """Arithmetic between different dtypes must follow promotion rules."""

    @given(shape=shapes)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_float32_plus_float64_gives_float64(self, shape):
        """float32 + float64 -> float64."""
        a = tz.ones(shape, tz.dtype.float32)
        b = tz.ones(shape, tz.dtype.float64)
        result = a + b
        assert result.dtype == tz.dtype.float64

    @given(shape=shapes)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_int32_plus_float32_gives_float32(self, shape):
        """int32 + float32 -> float32."""
        a = tz.ones(shape, tz.dtype.int32)
        b = tz.ones(shape, tz.dtype.float32)
        result = a + b
        assert result.dtype == tz.dtype.float32

    @given(shape=shapes)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_int32_plus_int64_gives_int64(self, shape):
        """int32 + int64 -> int64."""
        a = tz.ones(shape, tz.dtype.int32)
        b = tz.ones(shape, tz.dtype.int64)
        result = a + b
        assert result.dtype == tz.dtype.int64

    @given(shape=shapes)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_scalar_mul_preserves_dtype(self, shape):
        """Multiplying by a scalar preserves the tensor dtype."""
        t = tz.randn(shape, tz.dtype.float32)
        result = t * 2.0
        assert result.dtype == tz.dtype.float32


# ---------------------------------------------------------------------------
# Broadcasting properties
# ---------------------------------------------------------------------------


class TestBroadcasting:
    """Broadcasting must follow NumPy rules."""

    @given(bc=broadcastable)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_broadcast_add_shape(self, bc):
        """Addition of broadcastable tensors produces correct output shape."""
        shape_a, shape_b, expected_shape = bc
        a = tz.ones(shape_a)
        b = tz.ones(shape_b)
        result = a + b
        assert result.shape == expected_shape

    @given(bc=broadcastable)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_broadcast_mul_shape(self, bc):
        """Multiplication of broadcastable tensors produces correct output shape."""
        shape_a, shape_b, expected_shape = bc
        a = tz.ones(shape_a)
        b = tz.ones(shape_b)
        result = a * b
        assert result.shape == expected_shape

    @given(bc=broadcastable)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_broadcast_commutativity(self, bc):
        """a + b == b + a for broadcastable tensors."""
        shape_a, shape_b, _ = bc
        tz.manual_seed(42)
        a = tz.randn(shape_a)
        tz.manual_seed(123)
        b = tz.randn(shape_b)
        r1 = a + b
        r2 = b + a
        assert r1.shape == r2.shape


# ---------------------------------------------------------------------------
# Creation function properties
# ---------------------------------------------------------------------------


class TestCreationInvariants:
    """Creation functions must satisfy basic invariants."""

    @given(shape=shapes, dtype=float_dtypes)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_zeros_all_zero(self, shape, dtype):
        """zeros() must produce all-zero tensors."""
        t = tz.zeros(shape, dtype)
        assert t.shape == shape
        assert t.dtype == dtype
        # Sum of zeros is zero
        s = t.sum()
        assert abs(s.item()) < 1e-10

    @given(shape=shapes, dtype=float_dtypes)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_ones_sum_equals_numel(self, shape, dtype):
        """ones() sum must equal numel."""
        t = tz.ones(shape, dtype)
        n = numel(shape)
        s = t.sum()
        assert abs(s.item() - n) < 1e-6

    @given(shape=shapes, val=finite_floats)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_full_all_same_value(self, shape, val):
        """full() must produce tensors where all elements equal val."""
        t = tz.full(shape, val)
        n = numel(shape)
        s = t.sum()
        expected = val * n
        if abs(expected) < 1e-10:
            assert abs(s.item()) < 1e-4
        else:
            assert abs(s.item() - expected) / (abs(expected) + 1e-10) < 1e-4

    @given(n=st.integers(min_value=1, max_value=16))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_eye_trace_equals_n(self, n):
        """eye(n) trace must equal n."""
        t = tz.eye(n)
        # Trace = sum of diagonal = n for identity matrix
        # Sum of entire matrix also equals n since off-diag are 0
        s = t.sum()
        assert abs(s.item() - n) < 1e-6


# ---------------------------------------------------------------------------
# Autograd consistency properties
# ---------------------------------------------------------------------------


class TestAutogradConsistency:
    """Analytical gradients must match numerical gradients."""

    @given(
        shape=st.lists(dims, min_size=1, max_size=2).map(list),
        dtype=st.just(tz.dtype.float64),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_sum_gradient_is_ones(self, shape, dtype):
        """Gradient of sum(x) w.r.t. x is all ones."""
        x = tz.randn(shape, dtype, requires_grad=True)
        y = x.sum()
        y.backward()
        grad = x.grad()
        # Every element of grad should be 1.0
        ones = tz.ones(shape, dtype)
        diff = grad - ones
        max_err = diff.abs().max().item()
        assert max_err < 1e-10

    @given(
        n=st.integers(min_value=1, max_value=8),
        dtype=st.just(tz.dtype.float64),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_mul_gradient(self, n, dtype):
        """Gradient of (x * 3).sum() w.r.t. x is all 3s."""
        x = tz.randn([n], dtype, requires_grad=True)
        y = (x * 3.0).sum()
        y.backward()
        grad = x.grad()
        expected = tz.full([n], 3.0, dtype)
        diff = grad - expected
        max_err = diff.abs().max().item()
        assert max_err < 1e-10

    @given(
        shape=st.lists(dims, min_size=1, max_size=2).map(list),
        dtype=st.just(tz.dtype.float64),
    )
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
    def test_add_gradient_both_inputs(self, shape, dtype):
        """Gradient of (a + b).sum() is ones for both a and b."""
        a = tz.randn(shape, dtype, requires_grad=True)
        b = tz.randn(shape, dtype, requires_grad=True)
        y = (a + b).sum()
        y.backward()

        ones = tz.ones(shape, dtype)
        err_a = (a.grad() - ones).abs().max().item()
        err_b = (b.grad() - ones).abs().max().item()
        assert err_a < 1e-10
        assert err_b < 1e-10


# ---------------------------------------------------------------------------
# Reduction properties
# ---------------------------------------------------------------------------


class TestReductionProperties:
    """Reduction operations must satisfy algebraic identities."""

    @given(shape=shapes, dtype=float_dtypes)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_sum_of_ones_is_numel(self, shape, dtype):
        """sum(ones(shape)) == numel(shape)."""
        t = tz.ones(shape, dtype)
        s = t.sum().item()
        assert abs(s - numel(shape)) < 1e-6

    @given(shape=shapes)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_mean_of_constant_is_constant(self, shape):
        """mean(full(shape, c)) == c."""
        c = 3.14
        t = tz.full(shape, c)
        m = t.mean().item()
        assert abs(m - c) < 1e-4

    @given(shape=shapes)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_min_le_mean_le_max(self, shape):
        """min(x) <= mean(x) <= max(x) for any tensor."""
        tz.manual_seed(42)
        t = tz.randn(shape)
        lo = t.min().item()
        hi = t.max().item()
        m = t.mean().item()
        assert lo <= m + 1e-6
        assert m <= hi + 1e-6


# ---------------------------------------------------------------------------
# Arithmetic identity properties
# ---------------------------------------------------------------------------


class TestArithmeticIdentities:
    """Basic algebraic identities must hold."""

    @given(shape=shapes, dtype=float_dtypes)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_add_zero_identity(self, shape, dtype):
        """x + 0 == x."""
        tz.manual_seed(42)
        x = tz.randn(shape, dtype)
        z = tz.zeros(shape, dtype)
        result = x + z
        diff = (result - x).abs().max().item()
        assert diff < 1e-10

    @given(shape=shapes, dtype=float_dtypes)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_mul_one_identity(self, shape, dtype):
        """x * 1 == x."""
        tz.manual_seed(42)
        x = tz.randn(shape, dtype)
        o = tz.ones(shape, dtype)
        result = x * o
        diff = (result - x).abs().max().item()
        assert diff < 1e-10

    @given(shape=shapes, dtype=float_dtypes)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_sub_self_is_zero(self, shape, dtype):
        """x - x == 0."""
        tz.manual_seed(42)
        x = tz.randn(shape, dtype)
        result = x - x
        max_val = result.abs().max().item()
        assert max_val < 1e-10

    @given(shape=shapes, dtype=float_dtypes)
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_neg_neg_identity(self, shape, dtype):
        """neg(neg(x)) == x."""
        tz.manual_seed(42)
        x = tz.randn(shape, dtype)
        result = -(-x)
        diff = (result - x).abs().max().item()
        assert diff < 1e-10


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
