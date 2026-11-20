"""Observed marginals, positive-support components and stratum populations.

The background population of a distance stratum contains only listed pairs
of valid bins whose bins are connected through positive observed support and
whose correction converged. Unobserved pairs are absent (they are not zeros),
explicitly listed zero counts are present as true zeros, and masked bins are
excluded everywhere by contract.
"""
from .balance import solve_sparse


def marginals(bins, entries):
    """Row-sum marginals of the symmetric sparse count map."""
    result = [0]*bins
    for (i, j), count in entries.items():
        result[i] += count
        if j != i:
            result[j] += count
    return result


def support_components(bins, entries):
    """Connected components of the positive-count support graph.

    Returns (adjacency, component, count): adjacency lists (other, count) for
    bins with positive support, component gives each bin's component id or -1
    when the bin has no positive observed support, and count is the number of
    components.
    """
    adjacency = {}
    for (i, j), count in entries.items():
        if count > 0:
            adjacency.setdefault(i, []).append((j, count))
            if j != i:
                adjacency.setdefault(j, []).append((i, count))
    component = [-1]*bins
    number = 0
    for start in adjacency:
        if component[start] != -1:
            continue
        component[start] = number
        stack = [start]
        while stack:
            node = stack.pop()
            for other, _ in adjacency[node]:
                if component[other] == -1:
                    component[other] = number
                    stack.append(other)
        number += 1
    return adjacency, component, number


def correct(adjacency, component, tolerance, max_iterations):
    """Run the sparse unit-marginal correction over the positive support.

    Returns (iterations, converged, no_solution): converged holds the bins
    whose balanced marginal residual met the tolerance within the iteration
    budget, and no_solution holds every bin in a component where the
    correction diverged, i.e. where no unit-marginal solution exists.
    """
    if not adjacency:
        return 0, set(), set()
    iterations, residuals, failed = solve_sparse(adjacency, tolerance, max_iterations)
    failed_components = {component[i] for i in failed}
    no_solution = {i for i in adjacency if component[i] in failed_components}
    converged = {i for i, residual in residuals.items()
                 if residual <= tolerance and i not in no_solution}
    return iterations, converged, no_solution


def stratum_populations(entries, stratum_of, component, converged):
    """Group background-eligible pairs by stratum.

    A listed pair joins its stratum's population only when both bins belong
    to the same positive-support component and the correction converged for
    both. Listed zeros join as zeros; unlisted pairs never join.
    """
    populations = {}
    for (i, j), count in entries.items():
        stratum = stratum_of[j-i]
        if stratum is None:
            continue
        if component[i] < 0 or component[i] != component[j]:
            continue
        if i not in converged or j not in converged:
            continue
        populations.setdefault(stratum, []).append((i, j, count))
    return populations
