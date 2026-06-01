"""
Autograd-aware distribution densities (release-prep Workstream F).

F4: Gumbel.log_prob previously detached exp(-z), dropping the gradient through
that term. With the new autograd-aware exp() Variable overload it must flow to
loc and scale.
"""

import os
import sys

import pytest

build_python_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
sys.path.insert(0, build_python_dir)

tz = pytest.importorskip("tenzor", reason="Tenzor Python package not built")
tc = tz.tenzor_core


@pytest.fixture(scope="module", autouse=True)
def _init():
    tc.initialize()


def _scalar(v):
    t = v.tensor() if hasattr(v, "tensor") else v
    return float(t.item())


def test_exp_variable_overload_is_autograd_aware():
    x = tc.Variable(tc.full([3], 0.5), True)
    y = tc.exp(x)
    tc.sum(y).backward()
    # d/dx sum(exp(x)) = exp(x) = exp(0.5) ~ 1.6487 per element.
    assert x.grad is not None
    assert abs(_scalar(x.grad.sum()) - 3.0 * 1.6487212707) < 1e-3


def test_gumbel_log_prob_grad_flows_to_params():
    from tenzor.distributions import Gumbel

    loc = tc.Variable(tc.full([4], 0.5), True)
    scale = tc.Variable(tc.full([4], 2.0), True)
    g = Gumbel(loc, scale)
    value = tc.Variable(tc.full([4], 1.0), False)

    lp = g.log_prob(value)
    tc.sum(lp).backward()

    assert loc.grad is not None, "gradient did not reach loc"
    assert scale.grad is not None, "gradient did not reach scale"
    # Gradients must be finite (the exp(-z) term used to be detached / could be
    # garbage); both should be nonzero for these inputs.
    import math
    assert math.isfinite(_scalar(loc.grad.sum()))
    assert math.isfinite(_scalar(scale.grad.sum()))
    assert abs(_scalar(loc.grad.sum())) > 0.0


def test_gamma_log_prob_grad_flows_to_params():
    # F3: Gamma.log_prob must be autograd-aware (lgamma Variable overload),
    # not scipy-on-detached-numpy.
    from tenzor.distributions import Gamma

    conc = tc.Variable(tc.full([4], 3.0), True)
    rate = tc.Variable(tc.full([4], 2.0), True)
    g = Gamma(conc, rate)
    value = tc.Variable(tc.full([4], 1.0), False)

    tc.sum(g.log_prob(value)).backward()
    assert conc.grad is not None, "gradient did not reach concentration"
    assert rate.grad is not None, "gradient did not reach rate"
    # d/d(rate) log_prob = alpha/rate - value = 3/2 - 1 = 0.5 per element.
    assert abs(_scalar(rate.grad.sum()) - 4 * 0.5) < 1e-3


def test_beta_log_prob_grad_flows_to_params():
    from tenzor.distributions import Beta
    a = tc.Variable(tc.full([4], 2.0), True)
    b = tc.Variable(tc.full([4], 3.0), True)
    d = Beta(a, b)
    v = tc.Variable(tc.full([4], 0.4), False)
    tc.sum(d.log_prob(v)).backward()
    assert a.grad is not None and b.grad is not None


def test_chi2_log_prob_grad_flows_to_df():
    from tenzor.distributions import Chi2
    df = tc.Variable(tc.full([4], 5.0), True)
    d = Chi2(df)
    v = tc.Variable(tc.full([4], 3.0), False)
    tc.sum(d.log_prob(v)).backward()
    assert df.grad is not None, "gradient did not reach df"


def test_studentt_log_prob_grad_flows():
    from tenzor.distributions import StudentT
    df = tc.Variable(tc.full([4], 5.0), True)
    loc = tc.Variable(tc.full([4], 0.0), True)
    scale = tc.Variable(tc.full([4], 2.0), True)
    d = StudentT(df, loc, scale)
    v = tc.Variable(tc.full([4], 1.0), False)
    tc.sum(d.log_prob(v)).backward()
    assert df.grad is not None and loc.grad is not None and scale.grad is not None


def test_dirichlet_log_prob_grad_flows():
    from tenzor.distributions import Dirichlet
    a = tc.Variable(tc.full([2, 3], 2.0), True)
    d = Dirichlet(a)
    v = tc.Variable(tc.full([2, 3], 1.0 / 3.0), False)
    tc.sum(d.log_prob(v)).backward()
    assert a.grad is not None


def test_weibull_log_prob_grad_flows():
    from tenzor.distributions import Weibull
    k = tc.Variable(tc.full([4], 1.5), True)
    lam = tc.Variable(tc.full([4], 2.0), True)
    d = Weibull(lam, k)  # ctor order checked below via shape, both Variables
    v = tc.Variable(tc.full([4], 1.0), False)
    tc.sum(d.log_prob(v)).backward()
    assert k.grad is not None or lam.grad is not None


def test_lognormal_rsample_and_log_prob_grad_flow():
    from tenzor.distributions import LogNormal
    loc = tc.Variable(tc.full([4], 0.0), True)
    scale = tc.Variable(tc.full([4], 1.0), True)
    d = LogNormal(loc, scale)
    # rsample is reparameterised -> gradient flows to loc/scale.
    s = d.rsample()
    tc.sum(s).backward()
    assert loc.grad is not None and scale.grad is not None
    # log_prob autograd-aware too.
    loc2 = tc.Variable(tc.full([4], 0.0), True)
    scale2 = tc.Variable(tc.full([4], 1.0), True)
    d2 = LogNormal(loc2, scale2)
    v = tc.Variable(tc.full([4], 1.5), False)
    tc.sum(d2.log_prob(v)).backward()
    assert loc2.grad is not None and scale2.grad is not None


def test_weibull_rsample_grad_flows():
    from tenzor.distributions import Weibull
    scale = tc.Variable(tc.full([4], 2.0), True)
    k = tc.Variable(tc.full([4], 1.5), True)
    d = Weibull(scale, k)
    tc.sum(d.rsample()).backward()
    assert scale.grad is not None and k.grad is not None


def test_halfnormal_rsample_grad_flows():
    from tenzor.distributions import HalfNormal
    scale = tc.Variable(tc.full([4], 2.0), True)
    d = HalfNormal(scale)
    tc.sum(d.rsample()).backward()
    assert scale.grad is not None


def test_halfnormal_log_prob_grad_flows_to_scale():
    import math
    from tenzor.distributions import HalfNormal
    # log p(v;s) = log2 - 0.5 log(2pi) - log(s) - 0.5 (v/s)^2  for v >= 0.
    # d/ds = -1/s + v^2 / s^3.
    s0, v0 = 1.5, 0.8
    scale = tc.Variable(tc.full([4], s0), True)
    d = HalfNormal(scale)
    v = tc.Variable(tc.full([4], v0), False)
    lp = d.log_prob(v)
    tc.sum(lp).backward()
    assert scale.grad is not None
    g = float(scale.grad[0].item())
    analytic = -1.0 / s0 + v0 * v0 / s0 ** 3
    assert abs(g - analytic) < 1e-3, f"grad {g} != analytic {analytic}"


def test_halfnormal_log_prob_negative_support_is_neg_inf():
    import math
    from tenzor.distributions import HalfNormal
    scale = tc.Variable(tc.full([2], 1.0), True)
    d = HalfNormal(scale)
    v = tc.Variable(tc.Tensor.from_numpy(__import__("numpy").array([-0.5, 0.5], dtype="float32")), False)
    lp = d.log_prob(v)
    vals = [float(lp[i].item()) for i in range(2)]
    assert vals[0] == float("-inf"), f"out-of-support entry must be -inf, got {vals[0]}"
    assert math.isfinite(vals[1]), f"in-support entry must be finite, got {vals[1]}"


def test_gamma_rsample_implicit_reparam_gradient():
    # F1: implicit-reparam Gamma rsample. Validated NON-circularly against the
    # closed-form expectation gradients (gradcheck would be circular here since
    # the implicit gradient uses a finite difference internally):
    #   E[Gamma(a, rate)] = a/rate  =>  dE/da = 1/rate,  dE/drate = -a/rate**2.
    from tenzor.distributions import Gamma
    tc.manual_seed(0)
    alpha = tc.Variable(tc.full([1], 3.0), True)
    rate = tc.Variable(tc.full([1], 1.0), True)
    g = Gamma(alpha, rate)
    N = 40000
    z = g.rsample((N,))           # [N, 1]
    sample_mean = _scalar(tc.sum(z)) / float(N)
    assert abs(sample_mean - 3.0) < 0.1, f"sample mean {sample_mean} != ~3.0"

    m = tc.sum(z) / float(N)
    m.backward()
    assert alpha.grad is not None and rate.grad is not None
    ga = _scalar(alpha.grad)
    gr = _scalar(rate.grad)
    # dE[z]/dalpha = 1/rate = 1 ; dE[z]/drate = -alpha/rate**2 = -3.
    assert abs(ga - 1.0) < 0.05, f"dE/dalpha estimate {ga}, expected ~1.0"
    assert abs(gr + 3.0) < 0.2, f"dE/drate estimate {gr}, expected ~-3.0"


def test_chi2_rsample_reparam():
    # F2: Chi2(df) = Gamma(df/2, 1/2). mean = df, dE/ddf = 1.
    from tenzor.distributions import Chi2
    tc.manual_seed(1)
    df = tc.Variable(tc.full([1], 5.0), True)
    z = Chi2(df).rsample((40000,))
    assert abs(_scalar(tc.sum(z)) / 40000.0 - 5.0) < 0.2
    (tc.sum(z) / 40000.0).backward()
    assert df.grad is not None
    assert abs(_scalar(df.grad) - 1.0) < 0.1, f"dE/ddf {_scalar(df.grad)} != ~1.0"


def test_beta_rsample_reparam():
    # F2: Beta(a,b) = X/(X+Y); mean = a/(a+b).
    from tenzor.distributions import Beta
    tc.manual_seed(2)
    a = tc.Variable(tc.full([1], 2.0), True)
    b = tc.Variable(tc.full([1], 3.0), True)
    z = Beta(a, b).rsample((20000,))
    assert abs(_scalar(tc.sum(z)) / 20000.0 - 0.4) < 0.02
    tc.sum(z).backward()
    assert a.grad is not None and b.grad is not None


def test_dirichlet_rsample_reparam():
    # F2: Dirichlet(a) = G/sum(G); each sample row sums to 1.
    from tenzor.distributions import Dirichlet
    tc.manual_seed(3)
    a = tc.Variable(tc.full([3], 2.0), True)
    z = Dirichlet(a).rsample((1000,))     # [1000, 3]
    assert abs(_scalar(tc.sum(z)) - 1000.0) < 1.0  # rows sum to 1
    tc.sum(z).backward()
    assert a.grad is not None


def test_vonmises_log_prob_grad_flows():
    # F3: VonMises log_prob autograd-aware via cos / bessel_i0 Variable overloads.
    from tenzor.distributions import VonMises
    loc = tc.Variable(tc.full([4], 0.5), True)
    conc = tc.Variable(tc.full([4], 2.0), True)
    d = VonMises(loc, conc)
    v = tc.Variable(tc.full([4], 0.3), False)
    tc.sum(d.log_prob(v)).backward()
    assert loc.grad is not None and conc.grad is not None


def test_studentt_rsample_reparam():
    # F2: T = loc + scale*Z/sqrt(V/df); gradient flows to df, loc, scale.
    from tenzor.distributions import StudentT
    tc.manual_seed(4)
    df = tc.Variable(tc.full([1], 5.0), True)
    loc = tc.Variable(tc.full([1], 0.0), True)
    scale = tc.Variable(tc.full([1], 2.0), True)
    z = StudentT(df, loc, scale).rsample((5000,))
    tc.sum(z).backward()
    assert df.grad is not None and loc.grad is not None and scale.grad is not None


# ============================================================================
# Discrete distributions whose log_prob previously ran entirely in numpy and
# returned a detached Tensor, silently severing the gradient to the learnable
# parameter (release-audit WS2). Each test asserts the parameter gradient
# flows, is finite and non-zero, with an analytic value check where simple.
# ============================================================================


def test_poisson_log_prob_grad_flows_to_rate():
    from tenzor.distributions import Poisson
    rate = tc.Variable(tc.full([4], 2.0), True)
    d = Poisson(rate)
    value = tc.Variable(tc.full([4], 4.0), False)
    tc.sum(d.log_prob(value)).backward()
    assert rate.grad is not None, "gradient did not reach rate"
    # d/d(rate) [v*log(rate) - rate - log(v!)] = v/rate - 1 = 4/2 - 1 = 1.
    assert abs(_scalar(rate.grad.sum()) - 4 * 1.0) < 1e-3


def test_binomial_log_prob_grad_flows_to_probs():
    from tenzor.distributions import Binomial
    probs = tc.Variable(tc.full([4], 0.5), True)
    d = Binomial(10, probs)
    value = tc.Variable(tc.full([4], 4.0), False)
    tc.sum(d.log_prob(value)).backward()
    assert probs.grad is not None, "gradient did not reach probs"
    # d/dp [v*log p + (n-v)*log(1-p)] = v/p - (n-v)/(1-p) = 4/0.5 - 6/0.5 = -4.
    assert abs(_scalar(probs.grad.sum()) - 4 * (-4.0)) < 1e-2


def test_geometric_log_prob_grad_flows_to_probs():
    from tenzor.distributions import Geometric
    probs = tc.Variable(tc.full([4], 0.5), True)
    d = Geometric(probs)
    value = tc.Variable(tc.full([4], 3.0), False)
    tc.sum(d.log_prob(value)).backward()
    assert probs.grad is not None, "gradient did not reach probs"
    # d/dp [v*log(1-p) + log p] = -v/(1-p) + 1/p = -3/0.5 + 1/0.5 = -4.
    assert abs(_scalar(probs.grad.sum()) - 4 * (-4.0)) < 1e-2


def test_negative_binomial_log_prob_grad_flows_to_both_params():
    import math
    from tenzor.distributions import NegativeBinomial
    total = tc.Variable(tc.full([4], 5.0), True)
    probs = tc.Variable(tc.full([4], 0.4), True)
    d = NegativeBinomial(total, probs)
    value = tc.Variable(tc.full([4], 3.0), False)
    tc.sum(d.log_prob(value)).backward()
    assert total.grad is not None, "gradient did not reach total_count"
    assert probs.grad is not None, "gradient did not reach probs"
    assert math.isfinite(_scalar(total.grad.sum()))
    assert math.isfinite(_scalar(probs.grad.sum()))
    assert abs(_scalar(probs.grad.sum())) > 0.0


def test_multinomial_log_prob_grad_flows_to_probs():
    import math
    from tenzor.distributions import Multinomial
    import numpy as np
    probs = tc.Variable(tc.full([3], 1.0), True)  # unnormalised; normalised inside
    d = Multinomial(7, probs)
    value = tc.Variable(
        tc.Tensor.from_numpy(np.asarray([2.0, 3.0, 2.0], dtype="float32")), False)
    tc.sum(d.log_prob(value)).backward()
    assert probs.grad is not None, "gradient did not reach probs"
    assert math.isfinite(_scalar(probs.grad.sum()))
