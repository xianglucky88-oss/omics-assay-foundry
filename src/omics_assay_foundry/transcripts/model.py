"""Fragment-allocation model, identifiability and deterministic resampling.

A fragment observation is allocated over its compatible transcripts in
proportion to the current molecule abundance times the fragment's relative
bias ``b_ji`` for that transcript (default 1).  Conditional on a fragment's
declared member set, its length density and uniform start position are shared
by all members and cancel, so the observed fragment length does not reweight
the members; it enters abundance through each transcript's effective length
``L_i = E_l[max(T_i - l + 1, 0)]``.

Only ratios of ``b_ji`` within one fragment enter the allocation, so a global
rescaling of the bias cancels and no bias leaves the legacy unbiased
allocation.

The M-step converts the allocated count to molecules through the transcript's
effective yield ``D_i = L_i * B_i``, where ``B_i`` is the mean declared bias
over the observations that support ``i`` (1 when no bias is declared):

    theta_i ∝ (sum_j n_j r_ji) / D_i

With no declared bias ``D_i = L_i``, the same effective length used by the
input normalization and reported in the result, so an extended request that
carries no new measurement field reproduces the legacy estimator.  A declared
condition with higher detection raises ``D_i`` and therefore lowers the
molecules implied by the same observed fragments.

Identifiability is a property of the declared support, not of convergence.
Transcripts observed in identical fragments with proportional bias columns
enter the likelihood only through one combined term ``sum_i c_i * theta_i``;
no amount of iteration identifies their split.  They are reported as a group
with those coefficients and get no individual interval, because a narrow
interval for a member would describe the combination.

Resampling is a Poisson bootstrap over observation units.  Each of the ``n_j``
observations of fragment ``j`` receives an independent Poisson(1) draw, so the
fragment weight is Poisson(n_j) and repeated observations genuinely widen the
uncertainty.  Draws are keyed by ``(seed, replicate, global unit index)`` with
a counter-based generator, and allocation always accumulates in the supplied
fragment order.  A fragment's draw therefore depends only on its position in
the request, never on how the request is split into chunks: equal explicit
random configuration reproduces identical numbers for any ``chunk_size``.
"""
import math

from ..errors import DomainError

INTERVAL_LEVEL = 0.95
MASK = (1 << 64) - 1
_PROPORTIONAL_TOLERANCE = 1e-9

def _mix(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK
    value = ((value ^ (value >> 30))*0xBF58476D1CE4E5B9) & MASK
    value = ((value ^ (value >> 27))*0x94D049BB133111EB) & MASK
    return value ^ (value >> 31)

def _uniform(seed, replicate, index):
    """Counter-based uniform in [0, 1) fixed by seed, replicate and unit index.

    The three coordinates are folded in sequence rather than combined with
    XOR-additive terms, so two distinct ``(replicate, index)`` pairs cannot
    cancel each other out.
    """
    key = _mix((seed ^ _mix(replicate)) + index*0x9E3779B97F4A7C15)
    return (key >> 11)/2**53

def _poisson_unit(seed, replicate, index):
    """One Poisson(1) draw by inverse CDF (mean 1, finite loop)."""
    target = _uniform(seed, replicate, index)
    probability = math.exp(-1.0)
    cumulative = probability
    weight = 0
    while target >= cumulative:
        weight += 1
        probability /= weight
        cumulative += probability
        if weight > 64:
            break
    return weight

def allocate(size, rows, abundance):
    """Molecule-weighted observation counts allocated per transcript."""
    counts = [0.0]*size
    for members, bias, observations in rows:
        if observations <= 0:
            # A fragment with no observations contributes nothing and must
            # not demand a representable allocation.
            continue
        terms = [abundance[i]*weight for i, weight in zip(members, bias)]
        denominator = math.fsum(terms)
        if denominator <= 0:
            raise DomainError("numeric_failure", "A fragment has no representable supporting abundance")
        for i, term in zip(members, terms):
            counts[i] += observations*term/denominator
    return counts

def estimate(yields, rows, max_iterations, tolerance):
    """EM for molecule fractions under the declared fragment bias.

    The E-step allocates a fragment over its compatible transcripts in
    proportion to molecule abundance times the fragment's relative bias
    ``b_ji``.  The M-step divides the allocated count by the transcript's
    effective yield ``D_i = L_i * B_i`` -- effective length times mean declared
    detection bias -- and normalizes.  With no declared bias ``D_i = L_i``, the
    same length used by the input normalization.

    Returns ``(abundance, counts, iterations, converged, maximum_change)``.
    """
    size = len(yields)
    abundance = [1/size]*size
    counts = [0.0]*size
    delta = 0.0
    for iteration in range(1, max_iterations+1):
        counts = allocate(size, rows, abundance)
        rates = []
        for count, yield_value in zip(counts, yields):
            if yield_value <= 0:
                if count > 0:
                    raise DomainError("zero_yield",
                                      "A supported transcript has no positive effective yield")
                rates.append(0.0)
            else:
                rates.append(count/yield_value)
        total = math.fsum(rates)
        if total <= 0:
            raise DomainError("numeric_failure", "Fragments carry no positive allocation under the model")
        updated = [rate/total for rate in rates]
        delta = max(abs(a-b) for a, b in zip(abundance, updated))
        abundance = updated
        if delta <= tolerance:
            return abundance, counts, iteration, True, delta
    return abundance, counts, max_iterations, False, delta

# ---------------------------------------------------------------------------
# Identifiability.
# ---------------------------------------------------------------------------

def _proportional(left, right):
    factor = left[0]/right[0]
    return all(abs(a-factor*b) <= _PROPORTIONAL_TOLERANCE*max(abs(a), abs(b), 1e-300)
               for a, b in zip(left, right))

def indistinguishable_groups(rows, supported):
    """Group transcripts with identical support and proportional propensities.

    Returns ``(groups, assignment, coefficients)``.  For each group only the
    combination ``sum_i coefficient_i * theta_i`` is identified; the reported
    individual fractions are one arbitrary split of that combination.
    """
    profiles = {}
    for i in supported:
        entries = [(position, bias[members.index(i)])
                   for position, (members, bias, count) in enumerate(rows)
                   if i in members and count > 0 and bias[members.index(i)] > 0]
        if entries:
            profiles[i] = (tuple(position for position, _ in entries),
                           [value for _, value in entries])
    clusters = []
    for i in sorted(profiles):
        positions, values = profiles[i]
        for cluster in clusters:
            if cluster[0] == positions and _proportional(values, cluster[1]):
                cluster[2].append(i)
                break
        else:
            clusters.append([positions, values, [i]])
    groups = sorted((sorted(cluster[2]) for cluster in clusters if len(cluster[2]) > 1),
                    key=lambda members: members[0])
    assignment = {}
    coefficients = {}
    for group, members in enumerate(groups):
        base = profiles[members[0]][1][0]
        for i in members:
            assignment[i] = group
            coefficients[i] = profiles[i][1][0]/base
    return groups, assignment, coefficients

# ---------------------------------------------------------------------------
# Deterministic resampling.
# ---------------------------------------------------------------------------

def bootstrap_rows(rows, seed, replicate, chunk_size):
    """Poisson-weighted observations; identical for any chunk size.

    The draw for observation unit ``u`` is keyed by its global index, which
    follows the fragment order of the request.  Chunking only groups the work.
    """
    drawn = []
    offset = 0
    step = chunk_size if chunk_size > 0 else len(rows)
    for start in range(0, len(rows), step):
        for members, bias, observations in rows[start:start+step]:
            total = 0
            for unit in range(offset, offset+observations):
                total += _poisson_unit(seed, replicate, unit)
            if total:
                drawn.append((members, bias, total))
            offset += observations
    return drawn

def _percentile(values, quantile):
    ordered = sorted(values)
    position = quantile*(len(ordered)-1)
    lower = math.floor(position)
    upper = min(lower+1, len(ordered)-1)
    fraction = position-lower
    return ordered[lower]*(1-fraction)+ordered[upper]*fraction

def interval(samples):
    """Two-sided percentile interval, or ``None`` when it cannot be formed."""
    if samples is None or len(samples) < 2:
        return None
    return [_percentile(samples, (1-INTERVAL_LEVEL)/2),
            _percentile(samples, 1-(1-INTERVAL_LEVEL)/2)]

def bootstrap(yields, rows, groups, coefficients, spec):
    """Deterministic Poisson bootstrap.

    Returns ``(info, transcript_samples, group_samples)``.  Samples are
    ``None`` whenever the interval is not usable; ``info`` always states why.
    """
    resamples = spec["resamples"]
    info = {"method": "poisson-observation-bootstrap", "seed": spec["seed"],
            "resamples": resamples, "level": INTERVAL_LEVEL,
            "work_limit": spec["bootstrap_work_limit"], "effective_replicates": 0}
    if resamples == 0:
        info.update(status="not_requested", usable=False,
                    reason="No resampling was requested; no resampling interval is reported.")
        return info, None, None
    work = resamples*(len(rows)+max(spec["total_observations"], 1))
    info["work"] = work
    if work > spec["bootstrap_work_limit"]:
        info.update(status="budget_exceeded", usable=False,
                    reason="Resampling work exceeds the declared budget; uncertainty is not reported.")
        return info, None, None
    samples = [[] for _ in yields]
    group_samples = [[] for _ in groups]
    effective = 0
    skipped = 0
    for replicate in range(resamples):
        drawn = bootstrap_rows(rows, spec["seed"], replicate, spec["chunk_size"])
        if not drawn:
            skipped += 1
            continue
        abundance, _, _, converged, _ = estimate(yields, drawn, spec["max_iterations"],
                                                 spec["tolerance"])
        if not converged:
            skipped += 1
            continue
        effective += 1
        for i, value in enumerate(abundance):
            samples[i].append(value)
        for group, members in enumerate(groups):
            group_samples[group].append(
                math.fsum(coefficients[i]*abundance[i] for i in members))
    info["effective_replicates"] = effective
    info["skipped_replicates"] = skipped
    if effective < 2 or effective < resamples/2:
        info.update(status="insufficient_replicates", usable=False,
                    reason="Too few resamples converged to form an interval; uncertainty is not reported.")
        return info, None, None
    info.update(status="ok", usable=True,
                reason="Percentile intervals over converged Poisson observation resamples.")
    return info, samples, group_samples
