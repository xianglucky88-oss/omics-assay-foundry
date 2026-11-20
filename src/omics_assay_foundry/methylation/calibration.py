"""Control-calibrated methylation model.

The observed methylated fraction of a site is not the biological fraction when
the assay converts some methylated cytosines and reads some unmethylated ones
as methylated.  With a fully methylated and a fully unmethylated control on the
same batch and strand we write

    q = u + p * (1 - c - u)

where ``p`` is the biological fraction, ``c`` the conversion failure rate
(true methylated reads reported as unmethylated) and ``u`` the background rate
(true unmethylated reads reported as methylated).  ``c`` and ``u`` receive Beta
posteriors from the control counts and ``p`` a uniform prior.

A region pools its sites under one regional ``p`` (the existing homogeneous-site
assumption).  Sites that share a control are combined into one binomial pool,
so the control posterior is integrated once per reported region and several
sites cannot multiply the calibration evidence: pooling the counts and pooling
the evidence give the same posterior.  Sites from different controls form
independent binomial pools that share ``p``.

Reported intervals are highest-posterior-density (HPD) credible intervals.
HPD is used rather than an equal-tailed interval because, when a site's reads
are all unmethylated (or all methylated), the posterior is largest at the
parameter boundary; an equal-tailed interval then excludes the boundary and
loses coverage exactly in the small-fraction regime the controls are meant to
resolve.  A boundary status marks those cases explicitly.
"""

import math
from statistics import NormalDist

from ..errors import DomainError
from .special import beta_cdf, beta_quantile, confidence_level, gauss_legendre_unit

_NORMAL = NormalDist()
NODE_GRID = 5
POSITIVE_SAMPLES = 96
_LEGENDRE = {size: gauss_legendre_unit(size) for size in (4, 5, 8)}
GRID_SIZE = 240
TINY = 1e-12
MAX_WORK = 20000000


def _expit(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


class Control:
    """Posterior summary of one batch/strand pair of conversion controls."""

    def __init__(self, record):
        self.id = record["id"]
        self.batch = record["batch"]
        self.strand = record["strand"]
        self.fully_methylated = record["fully_methylated"]
        self.fully_unmethylated = record["fully_unmethylated"]
        methylated, coverage = self.fully_methylated
        # Conversion rate: unmethylated reads from a fully methylated control.
        self._c_shapes = (coverage - methylated + 1.0, methylated + 1.0)
        methylated, coverage = self.fully_unmethylated
        # Background rate: methylated reads from a fully unmethylated control.
        self._u_shapes = (methylated + 1.0, coverage - methylated + 1.0)
        self._nodes = {}
        self._positive_mass = None
        self._intervals = {}

    def source(self):
        return {"id": self.id, "batch": self.batch, "strand": self.strand,
                "fully_methylated": {"methylated": self.fully_methylated[0],
                                     "coverage": self.fully_methylated[1]},
                "fully_unmethylated": {"methylated": self.fully_unmethylated[0],
                                       "coverage": self.fully_unmethylated[1]}}

    def has_coverage(self):
        return self.fully_methylated[1] > 0 and self.fully_unmethylated[1] > 0

    @property
    def positive_mass(self):
        """Posterior probability that the two biases leave usable signal."""
        if self._positive_mass is None:
            a_c, b_c = self._c_shapes
            a_u, b_u = self._u_shapes
            if not self.has_coverage():
                self._positive_mass = 0.0
            elif a_c + b_c > 1e5 or a_u + b_u > 1e5:
                # Narrow posteriors: a normal approximation of c + u is
                # accurate to far below the reported resolution.
                mean = a_c / (a_c + b_c) + a_u / (a_u + b_u)
                variance = (a_c * b_c / ((a_c + b_c) ** 2 * (a_c + b_c + 1.0))
                            + a_u * b_u / ((a_u + b_u) ** 2 * (a_u + b_u + 1.0)))
                self._positive_mass = min(max(
                    _NORMAL.cdf((1.0 - mean) / math.sqrt(variance)), 0.0), 1.0)
            else:
                # mass = E_c[F_u(1 - c)].  Sampling c at Beta quantiles makes
                # the expectation a unit-weight average with no density
                # evaluation, which stays accurate from shallow controls (very
                # broad posteriors) to deep ones (posteriors concentrated in a
                # tiny interval where a fixed grid would miss the mass).
                total = 0.0
                for index in range(POSITIVE_SAMPLES):
                    c = beta_quantile((index + 0.5) / POSITIVE_SAMPLES, a_c, b_c)
                    total += beta_cdf(a_u, b_u, 1.0 - c)
                self._positive_mass = min(max(total / POSITIVE_SAMPLES, 0.0), 1.0)
        return self._positive_mass

    def nodes(self, grid=NODE_GRID):
        """Quadrature nodes for the control posterior, as (u, 1-c-u) pairs.

        Gauss-Legendre nodes integrate the smooth Beta posteriors far more
        accurately than equal-probability nodes of the same count, which
        matters when a control is shallow and its posterior is broad.  Each
        node carries the log weight it represents in the two-dimensional
        posterior integral, normalised over the nodes that leave usable signal.
        """
        cached = self._nodes.get(grid)
        if cached is None:
            a_c, b_c = self._c_shapes
            a_u, b_u = self._u_shapes
            legendre = _LEGENDRE.get(grid) or gauss_legendre_unit(grid)
            c_values = [(beta_quantile(q, a_c, b_c), weight) for q, weight in legendre]
            u_values = [(beta_quantile(q, a_u, b_u), weight) for q, weight in legendre]
            kept = []
            for c, c_weight in c_values:
                for u, u_weight in u_values:
                    d = 1.0 - c - u
                    if d > 0.0:
                        kept.append((u, d, math.log(c_weight * u_weight)))
            total = math.fsum(math.exp(log_weight) for _, _, log_weight in kept)
            log_total = math.log(total) if total > 0.0 else 0.0
            cached = [(u, d, log_weight - log_total) for u, d, log_weight in kept]
            self._nodes[grid] = cached
        return cached

    def interval(self, methylated, coverage, z):
        """Posterior median and HPD interval for counts pooled on this control.

        Results are memoised because many sites share both a control and a
        coverage, and the posterior depends only on those three inputs.
        """
        key = (methylated, coverage, z)
        cached = self._intervals.get(key)
        if cached is None:
            cached = posterior_interval([(self.nodes(), methylated, coverage)], z)
            self._intervals[key] = cached
        return cached

    def unresolved_reason(self, level):
        if not self.has_coverage():
            return "no_control_coverage"
        if self.positive_mass < level:
            return "control_unresolved"
        return None


def _pool_log_likelihood(p, pools):
    """Log likelihood of independent binomial pools at one shared fraction."""
    total = 0.0
    for nodes, methylated, coverage in pools:
        unmethylated = coverage - methylated
        largest = -math.inf
        parts = []
        for u, d, log_weight in nodes:
            q = u + p * d
            if q <= 0.0:
                value = 0.0 if methylated == 0 else -math.inf
            elif q >= 1.0:
                value = 0.0 if unmethylated == 0 else -math.inf
            else:
                value = methylated * math.log(q) + unmethylated * math.log1p(-q)
            if value > -math.inf:
                parts.append(log_weight + value)
                if value > largest:
                    largest = value
        if not parts:
            return -math.inf
        total += largest + math.log(math.fsum(math.exp(value - largest) for value in parts))
    return total


def _clean(interval):
    low, high = interval
    low = min(max(low, 0.0), 1.0)
    high = min(max(high, 0.0), 1.0)
    if low > high:
        low, high = high, low
    return [low, high]


def _density_p(t, pools):
    """Log posterior density of p, evaluated at logit coordinate t.

    Uniform prior on p, so this is the likelihood up to a constant that the
    threshold comparison cancels.  It is deliberately *not* a logit-space
    density: the grid and cutoff both work in p.
    """
    return _pool_log_likelihood(_expit(t), pools)


def _crossing(log_density, threshold, start, end):
    """Bisect along the logit axis for the point whose density meets a level.

    ``start`` is a known-above-threshold coordinate (the mode) and ``end`` a
    known-below one (or a boundary); the search only assumes monotonicity
    between them, which holds for the unimodal posterior here.
    """
    low, high = sorted((start, end))
    for _ in range(40):
        middle = (low + high) / 2.0
        if log_density(middle) >= threshold:
            if end > start:
                low = middle
            else:
                high = middle
        else:
            if end > start:
                high = middle
            else:
                low = middle
        if high - low < 1e-10:
            break
    return (low + high) / 2.0


def posterior_interval(pools, z):
    """Median and HPD credible interval for the shared fraction ``p``.

    ``pools`` holds one ``(nodes, pooled methylated, pooled coverage)`` entry
    per distinct control, so each control posterior enters exactly once.
    """
    level = confidence_level(z)
    budget = [0]
    work_per_evaluation = sum(len(item[0]) for item in pools)

    def charge(work):
        budget[0] += work
        if budget[0] > MAX_WORK:
            raise DomainError("work_limit",
                              "Control-calibrated posterior search exceeded its budget")

    def density(t):
        charge(work_per_evaluation)
        return _density_p(t, pools)

    # Cells uniform in p, the coordinate used by the prior and the report, so a
    # cell's posterior share is its density divided by the total density.  The
    # scan itself is charged so a region spanning many controls cannot run
    # unbounded work before the budget is noticed.
    width = 1.0 / GRID_SIZE
    centres = [(index + 0.5) * width for index in range(GRID_SIZE)]
    charge(work_per_evaluation * GRID_SIZE)
    densities = [_pool_log_likelihood(p, pools) for p in centres]
    peak_value = max(densities)
    if peak_value == -math.inf:
        return None, [0.0, 1.0], True
    weights = [math.exp(value - peak_value) if value > -math.inf else 0.0
               for value in densities]
    normaliser = math.fsum(weights)
    if normaliser <= 0.0:
        return None, [0.0, 1.0], True

    running = 0.0
    median = centres[-1]
    for centre, weight in zip(centres, weights):
        running += weight
        if running >= 0.5 * normaliser:
            median = centre
            break

    # Highest-density cutoff: take cells densest first until the requested
    # mass is reached, interpolating the cutoff between the last included and
    # first excluded cell so it is not quantised to one grid step.
    order = sorted(range(GRID_SIZE), key=lambda index: -densities[index])
    cumulative = 0.0
    threshold = -math.inf
    previous = math.inf
    for index in order:
        share = weights[index] / normaliser
        if cumulative + share >= level:
            fraction = (level - cumulative) / share if share > 0.0 else 0.0
            threshold = (densities[index] + (1.0 - fraction) * (previous - densities[index])
                         if math.isfinite(previous) else densities[index])
            break
        cumulative += share
        previous = densities[index]

    peak_index = max(range(GRID_SIZE), key=lambda index: densities[index])
    peak_centre = centres[peak_index]
    mode_t = math.log(peak_centre / (1.0 - peak_centre)) if 0.0 < peak_centre < 1.0 else 0.0
    zero_t = math.log(TINY / (1.0 - TINY))
    one_t = -zero_t

    # The endpoints p = 0 and p = 1 are not grid cells but have a well-defined
    # density.  Include one whenever its own density clears the cutoff, so an
    # all-unmethylated or all-methylated site reports the boundary rather than
    # stopping just short of it.
    def edge(endpoint_density, nearest_cell_density, far_t, fallback):
        if endpoint_density >= threshold or nearest_cell_density >= threshold:
            return fallback
        return _expit(_crossing(density, threshold, mode_t, far_t))

    lower = edge(_pool_log_likelihood(0.0, pools), densities[0], zero_t, 0.0)
    upper = edge(_pool_log_likelihood(1.0, pools), densities[-1], one_t, 1.0)
    boundary = lower == 0.0 or upper == 1.0
    return median, _clean([lower, upper]), boundary
