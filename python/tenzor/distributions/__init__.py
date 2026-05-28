"""Probability distributions — Tensor / Variable native.

Distributions in this namespace return Tenzor ``Tensor`` or ``Variable``
objects from every sampling and density operation; parameters can be
plain Python scalars, numpy arrays, or autograd Variables.

Reparameterised sampling
------------------------

The location-scale families implement a genuine ``rsample`` whose
gradient flows into the distribution parameters:

* ``Normal``, ``Uniform``, ``Laplace``, ``Exponential``, ``Cauchy``,
  ``Gumbel``.

For these classes ``has_rsample = True``.  Every other distribution
sets ``has_rsample = False`` and ``rsample`` raises
``NotImplementedError`` (never silently falls back to ``sample``).

Discrete distributions (``Bernoulli``, ``Categorical``, ``Binomial``,
``Multinomial``, ``Poisson``, ``Geometric``, ``NegativeBinomial``) draw
samples via numpy and wrap the result as a Tenzor Tensor.

Compatibility
-------------

The API is loosely modelled on ``torch.distributions`` but is not a
drop-in replacement:

* Distribution parameters are coerced to Tenzor objects on
  construction; do **not** assume ``dist.loc`` is still a numpy array.
* ``sample(()), sample(())`` and ``rsample`` return Tenzor objects, not
  numpy arrays.
* Reparameterisation for ``Gamma`` / ``Beta`` / ``Dirichlet`` /
  ``StudentT`` / ``Chi2`` is not implemented yet — those distributions
  have ``has_rsample = False``.

Example::

    import tenzor as tz
    from tenzor.distributions import Normal

    loc   = tz.tenzor_core.Variable(tz.zeros([4]), True)
    scale = tz.tenzor_core.Variable(tz.ones([4]),  True)
    dist  = Normal(loc, scale)

    z = dist.rsample()       # autograd-aware Tenzor Variable
    loss = tz.sum(z)
    loss.backward()
    assert loc.grad   is not None
    assert scale.grad is not None
"""
from .distribution import Distribution
from .normal import Normal
from .bernoulli import Bernoulli
from .categorical import Categorical
from .multinomial import Multinomial
from .dirichlet import Dirichlet
from .binomial import Binomial
from .poisson import Poisson
from .gamma import Gamma
from .beta import Beta
from .studentT import StudentT
from .cauchy import Cauchy
from .laplace import Laplace
from .gumbel import Gumbel
from .weibull import Weibull
from .exponential import Exponential
from .geometric import Geometric
from .chi2 import Chi2
from .log_normal import LogNormal
from .von_mises import VonMises
from .half_normal import HalfNormal
from .negative_binomial import NegativeBinomial
from .uniform import Uniform

__all__ = [
    "Distribution",
    "Normal",
    "Bernoulli",
    "Categorical",
    "Multinomial",
    "Dirichlet",
    "Binomial",
    "Poisson",
    "Gamma",
    "Beta",
    "StudentT",
    "Cauchy",
    "Laplace",
    "Gumbel",
    "Weibull",
    "Exponential",
    "Geometric",
    "Chi2",
    "LogNormal",
    "VonMises",
    "HalfNormal",
    "NegativeBinomial",
    "Uniform",
]
