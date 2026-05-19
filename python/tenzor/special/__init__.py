"""
tenzor.special — Special mathematical functions.

Mirrors the OpId surface for Bessel, error, gamma, and related functions.
All functions operate element-wise on Tensor inputs.

Functions available
-------------------
Bessel functions:
    j0, j1         — Bessel functions of the first kind (order 0, 1)
    y0, y1         — Bessel functions of the second kind (order 0, 1)
    i0, i1         — Modified Bessel functions of the first kind (order 0, 1)

Error / Gaussian:
    erf, erfc      — Error function and its complement
    erfinv         — Inverse error function
    ndtri          — Inverse of the standard normal CDF (= erfinv(2*p-1)*sqrt(2))

Gamma family:
    lgamma         — Natural log of the absolute value of the gamma function
    digamma        — Digamma function (logarithmic derivative of gamma)
    polygamma      — Polygamma function (n-th derivative of digamma)

Other:
    sinc           — Normalized sinc: sin(pi*x)/(pi*x)
    zeta           — Hurwitz zeta function
    xlogy          — x * log(y), returning 0 when x == 0
    xlog1py        — x * log(1+y), returning 0 when x == 0

Usage
-----
    import tenzor as tz
    x = tz.randn([8])
    print(tz.special.lgamma(x.abs()))
    print(tz.special.j0(x))
"""

from tenzor import (
    bessel_j0 as j0,
    bessel_j1 as j1,
    bessel_y0 as y0,
    bessel_y1 as y1,
    bessel_i0 as i0,
    bessel_i1 as i1,
    erf,
    erfc,
    erfinv,
    ndtri,
    lgamma,
    digamma,
    polygamma,
    sinc,
    zeta,
    xlogy,
    xlog1py,
)

__all__ = [
    # Bessel
    "j0", "j1", "y0", "y1", "i0", "i1",
    # Error / Gaussian
    "erf", "erfc", "erfinv", "ndtri",
    # Gamma
    "lgamma", "digamma", "polygamma",
    # Other
    "sinc", "zeta", "xlogy", "xlog1py",
]
