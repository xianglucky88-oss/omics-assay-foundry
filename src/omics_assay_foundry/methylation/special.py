"""Beta distribution helpers used by the control-calibrated methylation model.

Only the standard library is available, so the regularized incomplete beta
function uses the classical continued-fraction evaluation.  For very large
shape parameters (deep control or site coverage) the continued fraction can be
slow and numerically delicate, so quantiles fall back to a normal
approximation; the resulting error is far below the resolution of the reported
interval.
"""

import math
from statistics import NormalDist

_NORMAL = NormalDist()
_FPMIN = 1e-300
_EPS = 3e-16


def _beta_continued_fraction(a, b, x):
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, 301):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def beta_cdf(a, b, x):
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if a + b > 1e6:
        # Continued fractions converge slowly for enormous shapes; a normal
        # approximation of the Beta is accurate to far below any reported
        # resolution at this width.
        mean = a / (a + b)
        variance = a * b / ((a + b) ** 2 * (a + b + 1.0))
        return min(max(_NORMAL.cdf((x - mean) / math.sqrt(variance)), 0.0), 1.0)
    log_front = a * math.log(x) + b * math.log1p(-x) - (
        math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        value = math.exp(log_front) * _beta_continued_fraction(a, b, x) / a
    else:
        value = 1.0 - math.exp(log_front) * _beta_continued_fraction(b, a, 1.0 - x) / b
    return min(max(value, 0.0), 1.0)


def beta_quantile(q, a, b):
    """Quantile of Beta(a, b) for 0 < q < 1."""
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    if a + b > 1e6:
        mean = a / (a + b)
        variance = a * b / ((a + b) ** 2 * (a + b + 1.0))
        value = mean + _NORMAL.inv_cdf(q) * math.sqrt(variance)
        return min(max(value, 1e-15), 1.0 - 1e-15)
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if beta_cdf(a, b, middle) < q:
            low = middle
        else:
            high = middle
        if high - low < 1e-14:
            break
    return (low + high) / 2.0


def confidence_level(z):
    """Probability content of a two-sided normal interval with multiplier z."""
    return math.erf(z / math.sqrt(2.0))


def gauss_legendre_unit(count):
    """Gauss-Legendre nodes and weights on [0, 1]."""
    nodes = []
    for index in range(1, count + 1):
        x = math.cos(math.pi * (index - 0.25) / (count + 0.5))
        derivative = 0.0
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, count + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            derivative = count * (x * p1 - p0) / (x * x - 1.0)
            step = -p1 / derivative
            x += step
            if abs(step) < 1e-15:
                break
        weight = 2.0 / ((1.0 - x * x) * derivative * derivative)
        nodes.append((0.5 * (1.0 + x), 0.5 * weight))
    nodes.sort()
    return nodes
