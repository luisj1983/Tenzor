"""S22 — Python frontend gap fixes.

Coverage:
  1. ``Tensor.__dlpack__(stream=...)`` accepts protocol kwargs without raising.
  2. ``Tensor.__dlpack__(dl_device=("cuda", 0))`` transfers across devices
     (CUDA-only — skipped when CUDA isn't built).
  3. ``Tensor.__dlpack__(copy=True)`` allocates a distinct storage.
  4. Tuple bool-mask of rank ≥ 2 is supported in tuple indexing path.
  5. ``F.avg_pool2d(..., count_include_pad=False)`` averages with the in-bounds
     divisor, not the full kernel area.
  6. ``F.avg_pool1d / avg_pool3d`` honour ``count_include_pad=False`` too.
"""

from __future__ import annotations

import math

import pytest

# conftest.py prepends build/python to sys.path and runs tz.initialize().
import tenzor as tz  # noqa: E402
import tenzor.nn.functional as F  # noqa: E402
import tenzor.tenzor_core as _core  # noqa: E402


f32 = _core.dtype.float32
bool_ = _core.dtype.bool


# ----------------------------------------------------------------------------
# Fix 1 — DLPack stream / dl_device / copy / max_version protocol kwargs.
# ----------------------------------------------------------------------------


def test_dlpack_stream_kwarg_accepted() -> None:
    """``stream=None`` and integer streams must be accepted without raising."""
    t = tz.zeros((4,), dtype=f32)

    # Every combination the v0.8 / v1.0 spec allows.
    for stream in (None, -1, 0, 1, 2):
        capsule = t.__dlpack__(stream=stream)
        # Capsule must be importable back into Tenzor (round-trip sanity).
        roundtrip = tz.from_dlpack(capsule)
        assert tuple(roundtrip.shape) == (4,)


def test_dlpack_max_version_kwarg_accepted() -> None:
    """``max_version`` is accepted (we always emit the v0.7 capsule)."""
    t = tz.ones((3,), dtype=f32)
    for mv in (None, (0, 7), (1, 0), (2, 0)):
        capsule = t.__dlpack__(max_version=mv)
        assert capsule is not None


def test_dlpack_copy_true_distinct_storage() -> None:
    """``copy=True`` must allocate fresh storage (different data_ptr)."""
    t = tz.full((6,), 3.5, dtype=f32)
    original_ptr = t.data_ptr()

    capsule = t.__dlpack__(copy=True)
    copied = tz.from_dlpack(capsule)

    # Copy must have its own storage.
    assert copied.data_ptr() != original_ptr
    # Values preserved.
    assert math.isclose(float(copied.sum().item()), 6 * 3.5, rel_tol=1e-6)


def test_dlpack_copy_false_or_none_zero_copy() -> None:
    """``copy=None`` / ``copy=False`` does not allocate new storage."""
    t = tz.full((6,), 1.25, dtype=f32)
    expected_ptr = t.data_ptr()

    for copy_val in (None, False):
        capsule = t.__dlpack__(copy=copy_val)
        view = tz.from_dlpack(capsule)
        assert view.data_ptr() == expected_ptr


def test_dlpack_dl_device_same_device_noop() -> None:
    """``dl_device`` pointing at the source device must be accepted."""
    t = tz.arange(0, 8, dtype=f32)
    # kDLCPU == 1 per dlpack.h
    capsule = t.__dlpack__(dl_device=(1, 0))
    roundtrip = tz.from_dlpack(capsule)
    assert tuple(roundtrip.shape) == (8,)


def test_dlpack_dl_device_cross_device_transfer() -> None:
    """Cross-device DLPack export via ``dl_device`` (CUDA-only)."""
    if not getattr(tz, "cuda_is_available", lambda: False)():
        pytest.skip("CUDA backend not available")
    t = tz.arange(0, 4, dtype=f32)
    # kDLCUDA == 2 per dlpack.h
    capsule = t.__dlpack__(dl_device=(2, 0))
    on_gpu = tz.from_dlpack(capsule)
    # The DLPack consumer view should now report a CUDA device.
    assert "cuda" in str(on_gpu.device).lower()


def test_dlpack_all_kwargs_combined() -> None:
    """All four kwargs at once must compose cleanly on the CPU path."""
    t = tz.eye(3, dtype=f32)
    capsule = t.__dlpack__(
        stream=None,
        max_version=(1, 0),
        dl_device=(1, 0),  # kDLCPU
        copy=False,
    )
    rt = tz.from_dlpack(capsule)
    assert tuple(rt.shape) == (3, 3)


# ----------------------------------------------------------------------------
# Fix 2 — Tuple bool-mask N-D support.
# ----------------------------------------------------------------------------


def test_tuple_bool_mask_2d_full_match() -> None:
    """``x[mask, ...]`` with a 2-D mask covering the leading dims of ``x``.

    Mirrors PyTorch: result has shape ``[num_true, *trailing]`` where the
    trailing dims here are empty since the mask consumes both dims of ``x``.
    """
    x = tz.tensor([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=f32)
    mask = x > 0  # 2-D bool mask
    selected = x[mask, ...]
    # Expected: the three positive values [1.0, 3.0, 5.0]
    assert tuple(selected.shape) == (3,)
    vals = sorted(selected.numpy().tolist())
    assert vals == [1.0, 3.0, 5.0]


def test_tuple_bool_mask_2d_with_trailing_dim() -> None:
    """3-D tensor, 2-D bool mask on the leading two dims, full slice on last."""
    x = tz.arange(0, 24, dtype=f32).reshape((2, 3, 4))
    # 2-D mask of shape (2, 3) — selects 4 rows of the trailing dim.
    mask_data = [[True, False, True], [False, True, False]]
    mask = tz.tensor(mask_data, dtype=bool_)
    sel = x[mask, :]
    # Expected shape: (num_true, 4) == (3, 4)
    assert tuple(sel.shape) == (3, 4)
    # The selected rows are x[0,0], x[0,2], x[1,1] in flattened-mask order.
    expected_rows = [
        x[0, 0, :].numpy().tolist(),
        x[0, 2, :].numpy().tolist(),
        x[1, 1, :].numpy().tolist(),
    ]
    assert sel.numpy().tolist() == expected_rows


def test_tuple_bool_mask_1d_unchanged() -> None:
    """1-D mask path (the historical fast path) must still work."""
    x = tz.arange(0, 12, dtype=f32).reshape((3, 4))
    mask = tz.tensor([True, False, True], dtype=bool_)
    sel = x[mask, :]
    assert tuple(sel.shape) == (2, 4)
    assert sel.numpy().tolist() == [x[0, :].numpy().tolist(), x[2, :].numpy().tolist()]


# ----------------------------------------------------------------------------
# Fix 3 — count_include_pad=False for avg_pool{1,2,3}d.
# ----------------------------------------------------------------------------


def test_avg_pool2d_count_include_pad_false_corner() -> None:
    """Boundary pixels divide by the in-bounds count, not the full window.

    Construct a 1x1x2x2 input of all ones with kernel=2, stride=1, padding=1.
    The output is 1x1x3x3. The four corner outputs each see exactly one
    in-bounds element, so with count_include_pad=False they should be 1.0,
    not 0.25 (the count_include_pad=True divisor would be 4).
    """
    x = tz.Variable(tz.ones((1, 1, 2, 2), dtype=f32), False)
    y = F.avg_pool2d(x, kernel_size=2, stride=1, padding=1,
                     count_include_pad=False)
    assert tuple(y.shape) == (1, 1, 3, 3)
    # Corners: 1 valid element each -> divisor=1, value=1.0
    corners = [y[0, 0, 0, 0].item(), y[0, 0, 0, 2].item(),
               y[0, 0, 2, 0].item(), y[0, 0, 2, 2].item()]
    for v in corners:
        assert math.isclose(v, 1.0, rel_tol=1e-6)

    # And the count_include_pad=True path should differ: corners see divisor=4.
    x_true = tz.Variable(tz.ones((1, 1, 2, 2), dtype=f32), False)
    y_true = F.avg_pool2d(x_true, kernel_size=2, stride=1, padding=1,
                          count_include_pad=True)
    corners_true = [y_true[0, 0, 0, 0].item(), y_true[0, 0, 0, 2].item()]
    for v in corners_true:
        assert math.isclose(v, 0.25, rel_tol=1e-6)


def test_avg_pool1d_count_include_pad_false() -> None:
    """1-D analogue: a length-2 input with kernel=2, padding=1.

    With padding=1 and kernel=2, the output length is (2 + 2 - 2)/2 + 1 = 2
    (stride defaults to kernel=2). The first window sees 1 in-bounds element,
    the second sees 1 in-bounds element. So with count_include_pad=False
    every output is the in-bounds value itself (divisor=1).
    """
    x = tz.Variable(tz.ones((1, 1, 2), dtype=f32), False)
    y = F.avg_pool1d(x, kernel_size=2, padding=1, count_include_pad=False)
    # First window: [pad, x0] -> avg with divisor=1 -> 1.0
    # Second window: [x1, pad] -> avg with divisor=1 -> 1.0
    assert tuple(y.shape) == (1, 1, 2)
    for i in range(2):
        assert math.isclose(y[0, 0, i].item(), 1.0, rel_tol=1e-6)
    # Compare to count_include_pad=True: divisor=2 -> 0.5 each.
    x_true = tz.Variable(tz.ones((1, 1, 2), dtype=f32), False)
    y_true = F.avg_pool1d(x_true, kernel_size=2, padding=1, count_include_pad=True)
    for i in range(2):
        assert math.isclose(y_true[0, 0, i].item(), 0.5, rel_tol=1e-6)


def test_avg_pool3d_count_include_pad_false() -> None:
    """3-D analogue: corner voxel of a 1x1x2x2x2 input with kernel=2, pad=1."""
    x = tz.Variable(tz.ones((1, 1, 2, 2, 2), dtype=f32), False)
    y = F.avg_pool3d(x, kernel_size=2, stride=1, padding=1,
                     count_include_pad=False)
    # Output spatial dims: (2 + 2 - 2)/1 + 1 = 3 in each axis.
    assert tuple(y.shape) == (1, 1, 3, 3, 3)
    # The eight corner voxels of the output see exactly one in-bounds input
    # element each -> divisor=1 -> value=1.0.
    for d in (0, 2):
        for h in (0, 2):
            for w in (0, 2):
                v = y[0, 0, d, h, w].item()
                assert math.isclose(v, 1.0, rel_tol=1e-6)
    # count_include_pad=True divisor=8 -> 1/8 at the corners.
    x_true = tz.Variable(tz.ones((1, 1, 2, 2, 2), dtype=f32), False)
    y_true = F.avg_pool3d(x_true, kernel_size=2, stride=1, padding=1,
                          count_include_pad=True)
    assert math.isclose(y_true[0, 0, 0, 0, 0].item(), 0.125, rel_tol=1e-6)


def test_avg_pool2d_default_unchanged() -> None:
    """``count_include_pad=True`` (default) must still produce 1/(k*k) at the
    boundary, so the new wiring is the only thing that changes behaviour."""
    x = tz.Variable(tz.ones((1, 1, 2, 2), dtype=f32), False)
    y = F.avg_pool2d(x, kernel_size=2, stride=1, padding=1)
    # Default count_include_pad=True -> divisor=4 -> corner=0.25.
    assert math.isclose(y[0, 0, 0, 0].item(), 0.25, rel_tol=1e-6)
