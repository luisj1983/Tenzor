"""Probability distributions — mirrors torch.distributions.

All distributions use numpy for computation and accept Python scalars
or numpy arrays as parameters.

Example::

    import numpy as np
    from tenzor.distributions import Normal, Gamma, Dirichlet

    # Normal distribution
    n = Normal(loc=0.0, scale=1.0)
    samples = n.sample((1000,))    # shape (1000,)
    lp = n.log_prob(0.0)           # ~ -0.9189
    print(n.entropy())             # ~ 1.4189

    # Gamma distribution
    g = Gamma(concentration=2.0, rate=1.0)
    print(g.mean)    # 2.0
    print(g.variance) # 2.0

    # Dirichlet distribution
    d = Dirichlet(np.array([1.0, 2.0, 3.0]))
    s = d.sample()   # sum to 1
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
