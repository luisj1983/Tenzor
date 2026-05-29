"""Frontend regression tests for the release audit (E1-E4, F2).

Covers:
  E1  default_collate stacks scalar int/float labels into tensors.
  E2  multi-worker map-style DataLoader yields in submission order.
  E3  Categorical.log_prob / entropy are autograd-aware (return Variables and
      propagate gradient to probs).
  E4  Gamma.rsample(()) returns a scalar-shaped sample; random_split accepts
      fractional lengths.
  F2  isinstance(nn.MSELoss(), nn.Loss) is True at runtime.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build/python'))
import tenzor as tz  # noqa: E402
tz.initialize()

import pytest  # noqa: E402


# ---------------------------------------------------------------------------
# E1 — default_collate scalar stacking
# ---------------------------------------------------------------------------

def test_default_collate_int_labels_become_tensor():
    from tenzor.data import default_collate
    out = default_collate([1, 2, 3])
    # Must be a tensor/variable, not a Python list.
    assert hasattr(out, "shape"), f"expected a tensor, got {type(out)}"
    assert list(out.shape) == [3]


def test_default_collate_float_labels_become_tensor():
    from tenzor.data import default_collate
    out = default_collate([1.0, 2.0, 3.0, 4.0])
    assert hasattr(out, "shape"), f"expected a tensor, got {type(out)}"
    assert list(out.shape) == [4]


def test_default_collate_tuple_sample_pattern():
    # The canonical (tensor, int_label) dataset pattern: labels must collate
    # into a tensor, not a list.
    from tenzor.data import default_collate
    batch = [(tz.zeros([2]), 0), (tz.zeros([2]), 1), (tz.zeros([2]), 2)]
    feats, labels = default_collate(batch)
    assert list(feats.shape) == [3, 2]
    assert hasattr(labels, "shape"), f"labels should be a tensor, got {type(labels)}"
    assert list(labels.shape) == [3]


# ---------------------------------------------------------------------------
# E2 — multi-worker DataLoader preserves submission order
# ---------------------------------------------------------------------------

class _IndexDataset(tz.data.Dataset):
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return i


def test_multiworker_dataloader_preserves_order():
    ds = _IndexDataset(48)
    loader = tz.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)
    got = [int(b.item() if hasattr(b, "item") else b[0]) for b in loader]
    assert got == list(range(48)), f"multi-worker order not preserved: {got}"


# ---------------------------------------------------------------------------
# E3 — Categorical autograd
# ---------------------------------------------------------------------------

def test_categorical_log_prob_is_autograd_aware():
    probs = tz.tensor([0.1, 0.2, 0.3, 0.4], tz.dtype.float32)
    probs = tz.Variable(probs, True) if hasattr(tz, "Variable") else probs
    dist = tz.distributions.Categorical(probs)
    lp = dist.log_prob(2)
    # Must be a Variable (gradient-carrying), not a bare Tensor.
    assert lp.__class__.__name__ in ("Variable",), f"log_prob returned {type(lp)}"
    lp.backward()
    g = probs.grad
    assert g is not None, "gradient did not flow to probs through log_prob"


def test_categorical_entropy_is_autograd_aware():
    probs = tz.tensor([0.25, 0.25, 0.5], tz.dtype.float32)
    probs = tz.Variable(probs, True) if hasattr(tz, "Variable") else probs
    dist = tz.distributions.Categorical(probs)
    h = dist.entropy()
    assert h.__class__.__name__ in ("Variable",), f"entropy returned {type(h)}"


# ---------------------------------------------------------------------------
# E4 — Gamma rsample scalar shape + random_split fractional lengths
# ---------------------------------------------------------------------------

def test_gamma_rsample_scalar_shape():
    g = tz.distributions.Gamma(2.0, 1.0)
    s = g.rsample(())
    # Scalar params + empty sample_shape -> scalar () sample (not [1]).
    assert tuple(s.shape) == (), f"expected scalar (), got shape {tuple(s.shape)}"


def test_random_split_accepts_fractions():
    ds = _IndexDataset(100)
    train, val = tz.data.random_split(ds, [0.8, 0.2])
    assert len(train) == 80
    assert len(val) == 20
    assert len(train) + len(val) == len(ds)


# ---------------------------------------------------------------------------
# F2 — isinstance(loss, nn.Loss)
# ---------------------------------------------------------------------------

def test_mseloss_isinstance_loss():
    assert isinstance(tz.nn.MSELoss(), tz.nn.Loss)


def test_crossentropyloss_isinstance_loss():
    assert isinstance(tz.nn.CrossEntropyLoss(), tz.nn.Loss)


def test_non_loss_module_is_not_loss():
    assert not isinstance(tz.nn.Linear(2, 2), tz.nn.Loss)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
