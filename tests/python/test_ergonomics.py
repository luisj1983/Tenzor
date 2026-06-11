#!/usr/bin/env python3
"""Ergonomics regression tests (2026-06 release polish).

Covers the PyTorch-parity ergonomics layer in python/tenzor/__init__.py:
  - auto-initialization on import (and TENZOR_AUTO_INIT=0 opt-out)
  - requires_grad= kwarg on tensor-creation factories
  - variadic shape sugar: tz.randn(3, 4) == tz.randn([3, 4])
  - method-style reductions on Variable: (v * v).sum().backward()
  - string device specs: .to("cpu"), Device("cuda:1")
  - nn.Module subclasses overriding forward() (PyTorch convention)
"""

import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_BUILD_PYTHON = os.path.join(_ROOT, "build", "python")
sys.path.insert(0, _BUILD_PYTHON)

import tenzor as tz  # noqa: E402  (auto-initializes on import)

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


# 1. Auto-init: import alone is enough — no explicit tz.initialize() call
#    has happened in this process, yet ops must work.
x = tz.ones([2, 3])
check("auto-init: ops work right after import", float((x + x).numpy().max()) == 2.0)

# 1b. initialize() stays a valid idempotent no-op.
tz.initialize()
check("explicit initialize() still callable (idempotent)", True)

# 1c. TENZOR_AUTO_INIT=0 defers init (verified in a subprocess).
_probe = subprocess.run(
    [sys.executable, "-c",
     "import sys; sys.path.insert(0, %r)\n"
     "import tenzor as tz\n"
     "tz.initialize()\n"
     "x = tz.ones([2])\n"
     "assert float((x + x).numpy().max()) == 2.0\n"
     "print('DEFERRED_OK')" % _BUILD_PYTHON],
    env={**os.environ, "TENZOR_AUTO_INIT": "0"},
    capture_output=True, text=True,
)
check("TENZOR_AUTO_INIT=0 defers, manual initialize() works",
      "DEFERRED_OK" in _probe.stdout)

# 2. requires_grad= on factories returns Variable with working autograd.
v = tz.randn([4, 4], requires_grad=True)
check("randn(requires_grad=True) -> Variable", isinstance(v, tz.Variable))
loss = tz.sum(v * v)
loss.backward()
check("factory Variable: backward populates grad", v.grad is not None)

t = tz.zeros([2, 2], requires_grad=False)
check("requires_grad=False -> Tensor-compatible (facade isinstance)",
      isinstance(t, tz.Tensor))
check("requires_grad=False -> dormant Variable (merge)",
      isinstance(t, tz.Variable) and not t.requires_grad)

w = tz.tensor([1.0, 2.0, 3.0], requires_grad=True)
check("tensor(data, requires_grad=True) -> Variable", isinstance(w, tz.Variable))

f = tz.full([2, 2], 7.0, requires_grad=True)
check("full(..., requires_grad=True) -> Variable", isinstance(f, tz.Variable))

# 3. Variadic shape sugar for pure-shape factories.
a = tz.randn(3, 4)
check("randn(3, 4) variadic shape", list(a.shape) == [3, 4])
b = tz.zeros(5)
check("zeros(5) single-int shape", list(b.shape) == [5])
c = tz.ones(2, 3, requires_grad=True)
check("ones(2, 3, requires_grad=True) -> Variable shape [2,3]",
      isinstance(c, tz.Variable) and list(c.shape) == [2, 3])
d = tz.rand([2, 2])
check("list shape still works", list(d.shape) == [2, 2])

# 4. Method-style reductions on Variable.
v2 = tz.randn([3, 3], requires_grad=True)
s = (v2 * v2).sum()
check("(v * v).sum() returns Variable", isinstance(s, tz.Variable))
s.backward()
check("method-style sum: backward populates grad", v2.grad is not None)
m = tz.randn([3, 3], requires_grad=True).mean()
check("Variable.mean() works", isinstance(m, tz.Variable))

# 5. String device specs.
y = tz.ones([2, 2]).to("cpu")
check('Tensor.to("cpu") string device', str(y.device) == "cpu")
dev = tz.Device("cuda:1")
check('Device("cuda:1") parses', str(dev) == "cuda:1")
vv = tz.Variable(tz.ones([2]), requires_grad=False).to("cpu")
check('Variable.to("cpu") string device', True)

# 6. nn.Module subclass overriding forward() (PyTorch convention).
class Net(tz.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = tz.nn.Linear(3, 2)

    def forward(self, x):
        return self.fc(x)


net = Net()
out = net(tz.Variable(tz.randn([4, 3]), requires_grad=False))
check("Module.forward() override dispatches", list(out.shape) == [4, 2])


# 6b. forward_impl still supported (back-compat).
class NetImpl(tz.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = tz.nn.Linear(3, 2)

    def forward_impl(self, x):
        return self.fc(x)


out2 = NetImpl()(tz.Variable(tz.randn([4, 3]), requires_grad=False))
check("Module.forward_impl() still works", list(out2.shape) == [4, 2])

# 7. Tensor/Variable merge.
x = tz.randn(3, 3)
check("merge: factory output is both Tensor and Variable",
      isinstance(x, tz.Tensor) and isinstance(x, tz.Variable))

# The headline: PyTorch-0.4 semantics with identity preserved.
x.requires_grad = True
(x * x).sum().backward()
check("merge: x.requires_grad = True; backward; x.grad", x.grad is not None)

x2 = tz.randn(2, 3)
check("merge: Variable.numpy() delegation", x2.numpy().shape == (2, 3))
r = x2.reshape([3, 2])
check("merge: Variable.reshape() delegation wraps to Variable",
      isinstance(r, tz.Variable) and list(r.shape) == [3, 2])

raw = tz.Tensor([2, 2])
check("merge: raw Tensor.requires_grad is False", raw.requires_grad is False)
try:
    raw.requires_grad = True
    check("merge: raw Tensor requires_grad setter fails loud", False)
except RuntimeError:
    check("merge: raw Tensor requires_grad setter fails loud", True)

# Implicit Variable->Tensor: dormant variables flow into tensor-only ops.
cc = tz.cat([tz.randn(2, 2), tz.randn(2, 2)], 0)
check("merge: cat(factory outputs) via implicit conversion",
      isinstance(cc, tz.Tensor) and list(cc.shape) == [4, 2])

# Fail-loud demotion: tracking variables refuse tensor-only ops...
xg = tz.randn(2, 2, requires_grad=True)
try:
    tz.flatten(xg)
    check("merge: tensor-only op on tracking Variable fails loud", False)
except (RuntimeError, TypeError):
    check("merge: tensor-only op on tracking Variable fails loud", True)
# ...unless explicitly under no_grad (intentional graph exit).
with tz.no_grad():
    _ = tz.flatten(xg)
check("merge: no_grad() permits intentional demotion", True)

# In-place init pattern: blocked while tracking, allowed under no_grad.
p = tz.randn(3, requires_grad=True)
try:
    p.fill_(0.0)
    check("merge: in-place on tracking Variable fails loud", False)
except RuntimeError:
    check("merge: in-place on tracking Variable fails loud", True)
with tz.no_grad():
    p.fill_(0.0)
check("merge: in-place under no_grad works (param init)",
      float(p.tensor().numpy().max()) == 0.0)


# Module attribute semantics: plain (non-tracking) tensors stay plain
# attributes; parameters register as before.
class AttrNet(tz.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = tz.nn.Linear(3, 2)
        self.scale = tz.ones(2)  # plain attribute, NOT a buffer/param

    def forward(self, x):
        return self.fc(x)


anet = AttrNet()
pnames = [n for n, _ in anet.named_parameters()]
check("merge: plain tensor attr not auto-registered as param",
      not any("scale" in n for n in pnames))
check("merge: plain tensor attr readable", list(anet.scale.shape) == [2])
out3 = anet(tz.Variable(tz.randn([4, 3]), requires_grad=False))
check("merge: module with plain attr still runs", list(out3.shape) == [4, 2])

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("ALL ERGONOMICS TESTS PASSED")
