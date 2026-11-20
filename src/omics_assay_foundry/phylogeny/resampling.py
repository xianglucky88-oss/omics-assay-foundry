"""Site-resampling stability and evidence diagnostics.

The resampling here is a **nonparametric bootstrap over observed columns**.  For
each replicate the same sampled column indices are applied to every candidate
and every partition, so candidates are always compared on the same observation
population.  The resulting ``bootstrap_support_fraction`` is the fraction of
replicates in which a candidate attained the strictly highest total log
likelihood.  It is a stability description of the comparison, **not** a
posterior probability that a candidate tree is true.

The random stream is fully specified so that an independent implementation can
reproduce it: ``random.Random(seed)`` is constructed once, and for each
replicate ``n_sites`` indices are drawn with ``rng.randrange(n_sites)`` in
replicate order, with replacement.

A column with probability exactly zero for a candidate contributes ``-inf`` to
that candidate's replicate total.  Such a candidate loses any replicate that
resamples the column; it is *not* excused from the comparison.  Dropping the
column (or the replicate) would let a candidate that cannot explain difficult
observations appear better supported than one that can.

Only a column whose value is numerically indeterminate (not a scientific zero)
is excluded, and a replicate containing one is counted as incomparable and
reported separately.
"""
import math
import random

# A log likelihood within this distance of zero carries no information about
# relative candidate support; it is not the same thing as a zero probability.
NO_INFORMATION_TOLERANCE = 1e-9

# Replicate totals are sums of hundreds of log likelihoods.  Two totals that
# agree this closely are treated as a tie: otherwise floating-point round-off
# alone would decide the "winner" of pairs of candidates with identical
# support, and resampling fractions would depend on summation order.  The
# tolerance is far below any scientifically meaningful log-likelihood gap.
TIE_RELATIVE_TOLERANCE = 1e-9
TIE_ABSOLUTE_TOLERANCE = 1e-9


def _compare_totals(a, b):
    """Return -1, 0 or 1 with a round-off tolerance; ``-inf`` compares normally."""
    if a == b:
        return 0
    if math.isclose(a, b, rel_tol=TIE_RELATIVE_TOLERANCE, abs_tol=TIE_ABSOLUTE_TOLERANCE):
        return 0
    return 1 if a > b else -1


def _replicate_totals(site_values, counts):
    """Sum one candidate over a resample; ``None`` means numerically unavailable.

    ``-inf`` entries are genuine zero-probability columns and propagate as
    ``-inf`` rather than being dropped.
    """
    total = 0.0
    for index, count in counts.items():
        if not count:
            continue
        value = site_values[index]
        if value is None:
            return None
        if value == -math.inf:
            return -math.inf
        total += count * value
    return total


def bootstrap(site_totals, n_sites, replicates, seed):
    """Summarize nonparametric site-resampling stability.

    ``site_totals`` maps candidate id to a list of per-site log likelihoods.
    A finite entry is an ordinary log likelihood, ``-math.inf`` marks a column
    with probability exactly zero (the candidate genuinely loses that
    replicate), and ``None`` marks a numerically indeterminate column, which
    makes the replicate incomparable for the whole set rather than silently
    favourable or unfavourable to one candidate.
    """
    identifiers = list(site_totals)
    winners = {identifier: 0 for identifier in identifiers}
    ties = 0
    incomparable = 0
    zero_forced = {identifier: 0 for identifier in identifiers}
    pair_wins = {}
    pair_ties = {}
    for position, a in enumerate(identifiers):
        for b in identifiers[position + 1:]:
            pair_wins[(a, b)] = [0, 0]
            pair_ties[(a, b)] = 0
    rng = random.Random(seed)
    for _ in range(replicates):
        counts = {}
        for _ in range(n_sites):
            index = rng.randrange(n_sites)
            counts[index] = counts.get(index, 0) + 1
        totals = {}
        comparable = True
        for identifier in identifiers:
            total = _replicate_totals(site_totals[identifier], counts)
            if total is None:
                comparable = False
            totals[identifier] = total
        if not comparable:
            incomparable += 1
            continue
        for identifier in identifiers:
            if totals[identifier] == -math.inf:
                zero_forced[identifier] += 1
        if all(total == -math.inf for total in totals.values()):
            # Every candidate says some resampled column is impossible, so no
            # candidate can be preferred; record it as a tie for all pairs.
            ties += 1
            for (a, b) in pair_ties:
                pair_ties[(a, b)] += 1
            continue
        leaders = []
        for identifier in identifiers:
            if totals[identifier] == -math.inf:
                continue
            if all(_compare_totals(totals[identifier], totals[other]) >= 0
                   for other in identifiers if totals[other] != -math.inf):
                leaders.append(identifier)
        if len(leaders) == 1:
            winners[leaders[0]] += 1
        else:
            ties += 1
        for (a, b) in pair_wins:
            order = _compare_totals(totals[a], totals[b])
            if order > 0:
                pair_wins[(a, b)][0] += 1
            elif order < 0:
                pair_wins[(a, b)][1] += 1
            else:
                pair_ties[(a, b)] += 1
    comparable_total = replicates - incomparable
    support = {
        identifier: (winners[identifier] / comparable_total if comparable_total else None)
        for identifier in identifiers
    }
    pairs = [
        {
            "a": a, "b": b,
            "a_win_fraction": pair_wins[(a, b)][0] / comparable_total if comparable_total else None,
            "b_win_fraction": pair_wins[(a, b)][1] / comparable_total if comparable_total else None,
            "tie_fraction": pair_ties[(a, b)] / comparable_total if comparable_total else None,
            "comparable_replicates": comparable_total,
        }
        for (a, b) in sorted(pair_wins)
    ]
    return {
        "method": "nonparametric-bootstrap-over-aligned-columns",
        "replicates": replicates,
        "comparable_replicates": comparable_total,
        "incomparable_replicates": incomparable,
        "seed": seed,
        "rng": "random.Random(seed); per replicate, n_sites draws of rng.randrange(n_sites)",
        "same_columns_for_every_candidate": True,
        "winner_counts": winners,
        "tie_replicates": ties,
        "zero_probability_forced_replicates": zero_forced,
        "bootstrap_support_fraction": support,
        "pairwise_win_fractions": pairs,
        "support_interpretation": (
            "Fraction of site-resampling replicates in which a candidate ranked highest, "
            "not a posterior probability and not the probability that a candidate tree is true"
        ),
        "status": (
            "no_replicates_requested" if not replicates else
            "no_comparable_replicates" if not comparable_total else "ok"
        ),
    }


def _compositions(total, parts):
    """Yield every tuple of ``parts`` nonnegative integers summing to ``total``."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def _multinomial_log_probability(counts, n_sites):
    log_probability = math.lgamma(n_sites + 1) - n_sites * math.log(n_sites)
    for count in counts:
        log_probability -= math.lgamma(count + 1)
    return log_probability


def exact_bootstrap_reference(site_totals, n_sites, max_compositions=200_000):
    """Exact resampling distribution when the number of outcomes is small.

    Enumerates every multiset of ``n_sites`` draws from ``n_sites`` columns and
    weights it by its multinomial probability.  This is the infinite-replicate
    limit of :func:`bootstrap`, computed without the Monte Carlo random stream,
    so it is an independent reference for small candidate sets.
    """
    parts = n_sites
    # Number of multisets is C(2n-1, n-1); check before enumerating.
    from math import comb
    if comb(2 * parts - 1, parts - 1) > max_compositions:
        return None
    identifiers = list(site_totals)
    winners = {identifier: 0.0 for identifier in identifiers}
    comparable_probability = 0.0
    ties = 0.0
    pair_wins = {}
    pair_ties = {}
    for position, a in enumerate(identifiers):
        for b in identifiers[position + 1:]:
            pair_wins[(a, b)] = 0.0
            pair_ties[(a, b)] = 0.0
    for counts in _compositions(n_sites, parts):
        probability = math.exp(_multinomial_log_probability(counts, n_sites))
        totals = {}
        comparable = True
        for identifier in identifiers:
            total = _replicate_totals(site_totals[identifier], dict(enumerate(counts)))
            if total is None:
                comparable = False
                break
            totals[identifier] = total
        if not comparable:
            continue
        comparable_probability += probability
        if all(total == -math.inf for total in totals.values()):
            ties += probability
            for (a, b) in pair_ties:
                pair_ties[(a, b)] += probability
            continue
        leaders = []
        for identifier in identifiers:
            if totals[identifier] == -math.inf:
                continue
            if all(_compare_totals(totals[identifier], totals[other]) >= 0
                   for other in identifiers if totals[other] != -math.inf):
                leaders.append(identifier)
        if len(leaders) == 1:
            winners[leaders[0]] += probability
        else:
            ties += probability
        for (a, b) in pair_wins:
            order = _compare_totals(totals[a], totals[b])
            if order > 0 or order < 0:
                pair_wins[(a, b)] += probability
            else:
                pair_ties[(a, b)] += probability
    if comparable_probability <= 0.0:
        return None
    comparable_probability = min(1.0, comparable_probability)
    return {
        "method": "exact-multinomial-enumeration-of-resamples",
        "compositions_enumerated": comb(2 * parts - 1, parts - 1),
        "comparable_probability": comparable_probability,
        "tie_probability": ties / comparable_probability,
        "support_fraction": {
            identifier: winners[identifier] / comparable_probability for identifier in identifiers
        },
        "pairwise_comparable_probability": {
            f"{a}|{b}": (pair_wins[(a, b)] + pair_ties[(a, b)]) / comparable_probability
            for (a, b) in pair_wins
        },
    }


def compare_bootstrap(monte_carlo, exact, replicates):
    """Compare Monte Carlo support fractions with the exact reference.

    The Monte Carlo meaning is a sampling estimate; the reference is the exact
    finite-sample probability.  A deviation larger than roughly three standard
    errors is reported as a disagreement, never as evidence about trees.
    """
    if exact is None or not replicates:
        return {"status": "not_available"}
    worst = 0.0
    disagreements = []
    for identifier, value in monte_carlo.items():
        expected = exact["support_fraction"][identifier]
        deviation = abs(value - expected)
        worst = max(worst, deviation)
        standard_error = math.sqrt(max(expected * (1.0 - expected), 1e-12) / replicates)
        if deviation > 3.0 * standard_error + 1e-12:
            disagreements.append(identifier)
    return {
        "status": "ok" if not disagreements else "deviation_exceeds_sampling_error",
        "max_abs_deviation": worst,
        "disagreements": disagreements,
        "note": "Monte Carlo support fractions are estimates of the exact resampling distribution.",
    }


def summarize_sites(values, sites, zero_sites, nonfinite_sites,
                    zero_tolerance=NO_INFORMATION_TOLERANCE):
    """Describe one candidate's column contributions without dropping any.

    ``zero_sites``, ``nonfinite_sites`` and the returned ``uninformative_sites``
    are all expressed in global column indices, matching ``values`` position by
    position through ``sites``.
    """
    uninformative = [site for site, value in zip(sites, values)
                     if value is not None and abs(value) <= zero_tolerance]
    if zero_sites:
        status = "zero_probability"
    elif nonfinite_sites:
        status = "numeric_instability"
    else:
        status = "ok"
    return {
        "zero_probability_sites": list(zero_sites),
        "nonfinite_sites": list(nonfinite_sites),
        "uninformative_sites": uninformative,
        "log_likelihood": None if zero_sites or nonfinite_sites else math.fsum(values),
        "status": status,
    }


def dominating_sites(site_deltas, threshold):
    """Describe whether a few columns carry the leading evidence difference.

    ``site_deltas`` are per-column differences ``logL_leader - logL_runner_up``.
    A site is "dominating" when its share of the total positive difference
    reaches ``threshold``.  This describes where the evidence sits; it is not a
    significance test.
    """
    total = math.fsum(site_deltas)
    positive = [(index, value) for index, value in enumerate(site_deltas) if value > 0.0]
    positive.sort(key=lambda item: (-item[1], item[0]))
    positive_total = math.fsum(value for _, value in positive)
    if total <= 0.0 or not positive:
        return {"total_delta": total, "positive_delta_total": positive_total,
                "largest_single_site_share": None,
                "dominating_site_indices": [], "sites_for_half_delta": None,
                "status": "no_positive_difference"}
    largest = positive[0][1] / positive_total
    dominating = [index for index, value in positive if value / positive_total >= threshold]
    running = 0.0
    half = len(positive)
    for count, (_, value) in enumerate(positive, start=1):
        running += value
        if running >= 0.5 * positive_total:
            half = count
            break
    return {
        "total_delta": total,
        "positive_delta_total": positive_total,
        "largest_single_site_share": largest,
        "dominating_site_indices": dominating,
        "sites_for_half_delta": half,
        "status": "site_dominated" if largest >= threshold else "diffuse",
    }
