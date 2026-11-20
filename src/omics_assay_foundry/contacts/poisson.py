"""Exact Poisson tail probabilities, Garwood rate intervals and BH q-values.

Poisson tails are evaluated through the regularized incomplete gamma
functions (series expansion below the mean, continued fraction above it),
so no normal approximation or pseudocount is involved at any count level.
"""
import math

_ITERATION_LIMIT = 1000000


def gamma_pq(a, x):
    """Regularized incomplete gammas (P, Q) at shape a > 0 and point x >= 0."""
    if x <= 0:
        return 0.0, 1.0
    if x < a + 1.0:
        p = _series_p(a, x)
        return p, 1.0 - p
    q = _fraction_q(a, x)
    return 1.0 - q, q


def _series_p(a, x):
    log_factor = -x + a * math.log(x) - math.lgamma(a)
    ap = a
    term = 1.0 / a
    total = term
    for _ in range(_ITERATION_LIMIT):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) <= abs(total) * 1e-15:
            break
    else:
        raise ArithmeticError("Incomplete gamma series did not converge")
    return min(1.0, max(0.0, total * math.exp(log_factor)))


def _fraction_q(a, x):
    log_factor = -x + a * math.log(x) - math.lgamma(a)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for n in range(1, _ITERATION_LIMIT):
        an = -n * (n - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= 1e-15:
            break
    else:
        raise ArithmeticError("Incomplete gamma continued fraction did not converge")
    return min(1.0, max(0.0, h * math.exp(log_factor)))


def poisson_tails(k, mu):
    """One-sided tail probabilities (P(X >= k), P(X <= k)) for X ~ Poisson(mu)."""
    if mu <= 0:
        raise ArithmeticError("Poisson expectation must be positive")
    if k == 0:
        return 1.0, math.exp(-mu)
    p_enrichment = gamma_pq(k, mu)[0]
    p_depletion = gamma_pq(k + 1, mu)[1]
    return p_enrichment, p_depletion


def chi_square_quantile(p, df):
    """Quantile of the chi-square distribution by safeguarded Newton iteration."""
    a = 0.5 * df
    lo, hi = 0.0, max(float(df), 1.0)
    for _ in range(200):
        if gamma_pq(a, 0.5 * hi)[0] >= p:
            break
        lo, hi = hi, 2.0 * hi
    else:
        raise ArithmeticError("Chi-square quantile bracket failed")
    x = 0.5 * (lo + hi)
    for _ in range(80):
        cdf = gamma_pq(a, 0.5 * x)[0]
        if cdf <= p:
            lo = x
        else:
            hi = x
        log_pdf = ((a - 1.0) * math.log(0.5 * x) - 0.5 * x
                   - math.lgamma(a) - math.log(2.0))
        pdf = math.exp(log_pdf)
        candidate = x - (cdf - p)/pdf if pdf > 0 else 0.5 * (lo + hi)
        if not math.isfinite(candidate) or not lo < candidate < hi:
            candidate = 0.5 * (lo + hi)
        if abs(candidate - x) <= 1e-12 * max(1.0, abs(x)):
            return candidate
        x = candidate
    return 0.5 * (lo + hi)


def rate_interval(k, alpha):
    """Garwood exact central interval for a Poisson rate at level 1 - alpha."""
    lower = 0.0 if k == 0 else 0.5 * chi_square_quantile(0.5 * alpha, 2 * k)
    upper = 0.5 * chi_square_quantile(1.0 - 0.5 * alpha, 2 * (k + 1))
    return lower, upper


def benjamini_hochberg(indexed):
    """q-values for (position, p-value) pairs; positions map to their q-value."""
    m = len(indexed)
    result = {}
    running = 1.0
    for rank, (position, p) in enumerate(sorted(indexed, key=lambda item: -item[1])):
        running = min(running, p * m / (m - rank))
        result[position] = min(1.0, running)
    return result
