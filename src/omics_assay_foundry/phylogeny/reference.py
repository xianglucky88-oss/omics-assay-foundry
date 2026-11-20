"""Independent reference calculations for small evidence problems.

The pruning engine is fast but subtle.  This module recomputes a candidate's
column likelihood by direct enumeration of every assignment of states to the
internal nodes.  It shares no pruning code; it uses the same transition-matrix
definition so that an error in the model construction would show up as a
mismatch against closed-form checks, while an error in the pruning recursion
would show up here.

The reference is only produced when the enumeration is small enough to be
cheap; otherwise the caller receives ``None``.  This is a verification aid for
small candidate sets, not a second inference method or a tree search.
"""
import math
from itertools import product

from .substitution import transition_matrix

MAX_INTERNAL_NODES = 4
MAX_ENUMERATIONS = 2_000_000


def enumerations(nodes, n_sites, n_categories):
    internal = sum(1 for node in nodes if node["taxon"] is None)
    return (4 ** internal) * max(1, n_sites) * max(1, n_categories)


def brute_force_site(nodes, root, masks, site, observed, model, rate):
    """Directly enumerate internal states; return one category's probability."""
    internal_indices = [index for index, node in enumerate(nodes) if node["taxon"] is None]
    position = {node: index for index, node in enumerate(internal_indices)}
    parent = [None] * len(nodes)
    for index, node in enumerate(nodes):
        for child in node["children"]:
            parent[child] = index

    # A branch length is stored on the node below the branch, so the transition
    # from the parent state to a node uses that node's length.
    matrices = [transition_matrix(model, node["length"], rate) for node in nodes]
    frequencies = model["frequencies"]
    leaf_states = {}
    for index, node in enumerate(nodes):
        if node["taxon"] is not None:
            if node["taxon"] in observed:
                mask = masks[node["taxon"]][site]
                leaf_states[index] = [state for state in range(4) if mask & (1 << state)]
            else:
                leaf_states[index] = [0, 1, 2, 3]

    total = 0.0
    for combination in product(range(4), repeat=len(internal_indices)):
        states = {node: combination[position[node]] for node in internal_indices}
        term = frequencies[states[root]]
        if term == 0.0:
            continue
        for index, node in enumerate(nodes):
            if index == root:
                continue
            row = matrices[index][states[parent[index]]]
            if node["taxon"] is not None:
                term *= math.fsum(row[state] for state in leaf_states[index])
            else:
                term *= row[states[index]]
            if term == 0.0:
                break
        total += term
    return total


def reference_site_values(nodes, root, masks, sites, observed, model):
    """Reference log likelihoods for every site, or ``None`` if infeasible."""
    if enumerations(nodes, len(sites), len(model["rates"])) > MAX_ENUMERATIONS:
        return None
    values = []
    for site in sites:
        terms = []
        for rate, weight in zip(model["rates"], model["weights"]):
            probability = brute_force_site(nodes, root, masks, site, observed, model, rate)
            if probability > 0.0:
                terms.append(weight * probability)
        values.append(math.log(math.fsum(terms)) if terms else None)
    return values


def compare_site_values(reference, computed):
    """Compare two per-site log-likelihood vectors.

    Returns agreement details; a ``None``/finite mismatch is a disagreement
    about whether the column has positive probability at all.
    """
    worst = 0.0
    zero_agreement = True
    checked = 0
    for expected, actual in zip(reference, computed):
        if (expected is None) != (actual is None):
            zero_agreement = False
            continue
        if expected is not None:
            worst = max(worst, abs(expected - actual))
        checked += 1
    return {
        "sites_checked": checked,
        "max_abs_log_deviation": worst,
        "zero_probability_agreement": zero_agreement,
        "agrees": zero_agreement and worst <= 1e-8,
    }
