"""Partitioned tree likelihood with ambiguity masks and rate mixtures.

The likelihood is the standard pruning algorithm on a reversible continuous
time Markov chain.  Each observed column is evaluated for every taxon declared
observed by its partition; taxa excluded from a partition are marginalized
exactly (their conditional likelihood vector is ``[1, 1, 1, 1]``), which is the
same as pruning them from the tree but cannot change the topology by accident.

A column is never dropped because a candidate gives it zero probability, and a
taxon is never dropped because it carries an ambiguity code.  Ambiguity codes
are expanded into the set of admitted states, and missing characters admit all
four states.
"""
import math

from .substitution import transition_matrix

BITS = (1, 2, 4, 8)


def build_nodes(tree, taxon_index):
    """Return postorder nodes, root index, max depth and zero-length branches.

    Children always precede their parent, so the root is the last node.  The
    branch length on the root is retained but never used.
    """
    nodes = []
    max_depth = [0]
    zero_lengths = [0]

    def visit(node, depth):
        max_depth[0] = max(max_depth[0], depth)
        if "name" in node:
            nodes.append({"taxon": taxon_index[node["name"]], "length": node["length"],
                          "depth": depth, "children": ()})
            return len(nodes) - 1
        children = tuple(visit(child, depth + 1) for child in node["children"])
        if node["length"] == 0.0:
            zero_lengths[0] += 1
        nodes.append({"taxon": None, "length": node["length"], "depth": depth, "children": children})
        return len(nodes) - 1

    visit(tree, 0)
    return nodes, len(nodes) - 1, max_depth[0], zero_lengths[0]


def category_transitions(nodes, model, rate):
    """Transition matrices indexed by node; the root entry is ``None``.

    A node's length describes the branch above it, so the matrix keyed by that
    node is the transition from parent to node.  Returns the matrices and the
    clamping report produced by the transition-matrix builder.
    """
    report = {}
    matrices = [None] * len(nodes)
    for index, node in enumerate(nodes[:-1]):
        matrices[index] = transition_matrix(model, node["length"], rate, report)
    return matrices, report


def transition_diagnostics(matrices, clamping=None):
    """Summary of pathological transition probabilities."""
    minimum = math.inf
    negatives = 0
    worst_row_error = 0.0
    for matrix in matrices:
        if matrix is None:
            continue
        for row in matrix:
            entry_min = min(row)
            if entry_min < minimum:
                minimum = entry_min
            negatives += sum(1 for value in row if value < -1e-12)
            worst_row_error = max(worst_row_error, abs(math.fsum(row) - 1.0))
    clamping = clamping or {}
    return {"minimum_entry": minimum, "negative_entries": negatives,
            "max_row_sum_error": worst_row_error,
            "clamped_transition_entries": clamping.get("clamped_entries", 0),
            "most_negative_raw_transition": clamping.get("most_negative_raw")}


def _leaf_vector(mask):
    return [1.0 if mask & bit else 0.0 for bit in BITS]


def _category_log_likelihood(nodes, root, matrices, masks, site, observed, frequencies):
    """One rate category's column log likelihood.

    Returns ``(value, status)`` where status is ``"ok"``, ``"zero"`` (the
    observation has probability exactly zero under this model) or
    ``"nonfinite"`` (the computation left the representable range, which is a
    numerical-stability condition rather than a scientific zero).
    """
    partials = [None] * len(nodes)
    log_scale = 0.0
    for index, node in enumerate(nodes):
        taxon = node["taxon"]
        if taxon is not None:
            partials[index] = _leaf_vector(masks[taxon][site]) if taxon in observed else [1.0, 1.0, 1.0, 1.0]
            continue
        values = [1.0, 1.0, 1.0, 1.0]
        for child in node["children"]:
            vector = partials[child]
            matrix = matrices[child]
            v0, v1, v2, v3 = vector
            for state in range(4):
                row = matrix[state]
                values[state] *= row[0] * v0 + row[1] * v1 + row[2] * v2 + row[3] * v3
        maximum = max(values)
        if maximum <= 0.0:
            return None, "zero"
        if not math.isfinite(maximum):
            return None, "nonfinite"
        inverse = 1.0 / maximum
        partials[index] = [value * inverse for value in values]
        log_scale += math.log(maximum)
    root_vector = partials[root]
    probability = math.fsum(frequencies[k] * root_vector[k] for k in range(4))
    if probability <= 0.0:
        return None, "zero"
    if not math.isfinite(probability):
        return None, "nonfinite"
    result = math.log(probability) + log_scale
    return (result, "ok") if math.isfinite(result) else (None, "nonfinite")


def _logsumexp(values):
    maximum = max(values)
    if maximum == -math.inf:
        return None, "zero"
    total = math.fsum(math.exp(value - maximum) for value in values)
    result = maximum + math.log(total)
    return (result, "ok") if math.isfinite(result) else (None, "nonfinite")


def site_log_likelihood(nodes, root, matrices_by_category, masks, site, observed,
                        frequencies, log_weights):
    """Log likelihood of one column under a discrete-rate mixture.

    Each category uses its own scaled pruning pass; categories are combined on
    the log scale with the mixture weights.  Returns ``(value, status)``.
    A column has probability exactly zero only when *every* category gives it
    zero probability; a numerical failure in any contributing category is
    reported as ``nonfinite`` rather than being folded into zero.
    """
    terms = []
    numeric_failure = False
    for matrices, weight in zip(matrices_by_category, log_weights):
        value, status = _category_log_likelihood(nodes, root, matrices, masks, site, observed, frequencies)
        if status == "nonfinite":
            numeric_failure = True
        elif status == "ok":
            terms.append(weight + value)
    if not terms:
        return (None, "nonfinite") if numeric_failure else (None, "zero")
    return _logsumexp(terms)


class ModelRun:
    """Prepared transition matrices for one candidate under one partition."""

    def __init__(self, nodes, root, model):
        self.nodes = nodes
        self.root = root
        self.model = model
        self.frequencies = model["frequencies"]
        self.log_weights = [math.log(weight) for weight in model["weights"]]
        self.matrices = []
        clamping = {}
        for rate in model["rates"]:
            matrices, report = category_transitions(nodes, model, rate)
            self.matrices.append(matrices)
            clamping["clamped_entries"] = clamping.get("clamped_entries", 0) + report.get("clamped_entries", 0)
            if "most_negative_raw" in report:
                clamping["most_negative_raw"] = min(clamping.get("most_negative_raw", 0.0),
                                                    report["most_negative_raw"])
        self.diagnostics = transition_diagnostics(
            (matrix for group in self.matrices for matrix in group), clamping
        )

    def site(self, masks, site, observed):
        """Return ``(value, status)`` for one column."""
        return site_log_likelihood(self.nodes, self.root, self.matrices, masks, site,
                                   observed, self.frequencies, self.log_weights)
