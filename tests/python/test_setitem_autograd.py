"""
S8: Variable.__setitem__ autograd-safety guard tests.

Background
----------
``Variable.__setitem__`` cannot register a ``CopySlices`` autograd Function
yet, so an in-place mutation on a Variable that participates in an autograd
graph would silently corrupt saved-for-backward tensors that alias its
storage. The bindings refuse the unsafe cases up front:

  * leaf  + requires_grad=True  -> ValueError  (PyTorch parity)
  * non-leaf + requires_grad=True -> RuntimeError (S8 fix; was a
                                                   UserWarning)
  * leaf  + requires_grad=False -> mutation succeeds
  * non-leaf + requires_grad=False -> mutation succeeds (no graph to
                                                         corrupt)

These tests pin down all four routes so the bindings cannot regress back to
the silently-wrong-gradient behavior.
"""

import os
import sys

import pytest

# Match the conftest.py convention used by every other test in this dir.
build_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
sys.path.insert(0, build_python_dir)

tz = pytest.importorskip("tenzor.tenzor_core", reason="Tenzor Python module not built")


@pytest.fixture(scope="module", autouse=True)
def _init_tenzor():
    tz.initialize()
    tz.manual_seed(0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zeros4_var(requires_grad: bool) -> "tz.Variable":
    """Construct a 1D length-4 Float32 zero Variable."""
    return tz.Variable(tz.zeros([4], tz.dtype.float32), requires_grad)


# ---------------------------------------------------------------------------
# Case 1: leaf + requires_grad=False -- must succeed
# ---------------------------------------------------------------------------

def test_setitem_leaf_no_grad_succeeds():
    x = _zeros4_var(requires_grad=False)
    assert x.is_leaf
    x[0] = 1.0  # must NOT raise
    val = float(x.tensor().numpy()[0])
    assert val == 1.0, f"expected 1.0 at index 0, got {val}"


# ---------------------------------------------------------------------------
# Case 2: leaf + requires_grad=True -- existing AA.9 guard, must raise
# (Included so the four-case routing is exhaustively pinned.)
# ---------------------------------------------------------------------------

def test_setitem_leaf_with_grad_raises():
    x = _zeros4_var(requires_grad=True)
    assert x.is_leaf
    assert x.requires_grad
    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        x[0] = 1.0
    # AA.9 message includes 'leaf' and 'in-place'.
    msg = str(excinfo.value)
    assert "leaf" in msg.lower()
    assert "in-place" in msg.lower() or "in place" in msg.lower()


# ---------------------------------------------------------------------------
# Case 3: non-leaf + requires_grad=False -- must succeed
# ---------------------------------------------------------------------------

def test_setitem_nonleaf_no_grad_succeeds():
    # Build a non-leaf with requires_grad=False: take a graph-free input,
    # produce a result via an op, then assign. ``x + 1`` on a no-grad
    # Variable yields a no-grad Variable; is_leaf may still be True because
    # there is no grad_fn to attach. We verify the actual routing properties
    # rather than asserting is_leaf semantics that PyTorch and Tenzor define
    # slightly differently for no-grad inputs.
    x = _zeros4_var(requires_grad=False)
    y = x + 1.0
    # Whatever leaf/non-leaf status y has, requires_grad is False here, so
    # the binding must permit the mutation either way (case 3 OR case 1).
    assert not y.requires_grad
    y[0] = 7.0  # must NOT raise
    val = float(y.tensor().numpy()[0])
    assert val == 7.0, f"expected 7.0 at index 0, got {val}"


# ---------------------------------------------------------------------------
# Case 4: non-leaf + requires_grad=True -- the bug case, must raise
# ---------------------------------------------------------------------------

def test_setitem_nonleaf_with_grad_raises():
    x = _zeros4_var(requires_grad=True)
    y = x + 1.0  # non-leaf, inherits requires_grad
    assert y.requires_grad, "y should require grad (inherits from x)"
    assert not y.is_leaf, "y should be a non-leaf (has grad_fn)"
    with pytest.raises(RuntimeError) as excinfo:
        y[0] = 1.0
    msg = str(excinfo.value)
    # S8 diagnostic must mention non-leaf and CopySlices / detach guidance.
    assert "non-leaf" in msg.lower()
    assert ("copyslices" in msg.lower()) or ("detach" in msg.lower())


# ---------------------------------------------------------------------------
# Workaround: detach to bypass the guard
# ---------------------------------------------------------------------------

def test_setitem_rebuild_as_leaf_workaround_succeeds():
    x = _zeros4_var(requires_grad=True)
    y = x + 1.0
    # Workaround: re-wrap the computed tensor as a fresh leaf without grad.
    # This is the documented escape hatch when the user wants in-place
    # mutation on a value derived from an autograd graph.
    z = tz.Variable(y.tensor(), requires_grad=False)
    assert not z.requires_grad, "rebuilt Variable must have requires_grad=False"
    assert z.is_leaf, "rebuilt Variable must be a leaf"
    z[0] = 1.0  # must NOT raise
    val = float(z.tensor().numpy()[0])
    assert val == 1.0, f"expected 1.0 at index 0, got {val}"


# ---------------------------------------------------------------------------
# Direct-execution shim: the ctest harness runs this file as
# ``python3 test_setitem_autograd.py``. Without ``pytest.main`` the test
# functions defined above would never execute.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
