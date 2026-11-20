"""Contact-enrichment analysis around the sparse marginal correction.

Model: each listed pair (i, j) with count k in distance stratum s is compared
against mu = e_s * m_i * m_j, where m are the observed marginals and e_s is
the stratum scale fitted to the background population. Every relation is
compared against a leave-one-out stratum scale that removes its own count
from the stratum totals, so a pair never votes for the scale it is tested
against. Zero marginals, disconnected support, unconverged corrections and
thin strata yield inestimable relations; no pseudocount is ever added to
make an estimate exist.
"""
import math

from . import background, poisson, sparse_support
from ..validation import integer, number
from ..errors import DomainError

MODEL = "sparse-multiplicative-marginal-distance-stratified-poisson-v1"
_BALANCE_WORK_LIMIT = 20000000
_GAMMA_WORK_LIMIT = 30000000


def _status(prep, counts, component, converged, no_solution, estimable, pair):
    """Estimability of one listed-or-queried relation, in precedence order."""
    i, j = pair
    entries = prep["entries"]
    if i not in prep["valid"] or j not in prep["valid"]:
        return "masked_bin"
    if pair not in entries:
        return "unobserved"
    if counts[i] == 0 or counts[j] == 0:
        return "unsupported_bin"
    if component[i] != component[j]:
        return "disconnected_support"
    if i in no_solution or j in no_solution:
        return "correction_no_solution"
    if i not in converged or j not in converged:
        return "correction_not_converged"
    stratum = prep["stratum_of"][j-i]
    if stratum is None:
        return "no_background_stratum"
    if not estimable[stratum]:
        return "insufficient_stratum_evidence"
    if counts[i]-entries[pair] == 0 or counts[j]-entries[pair] == 0:
        # The pair is a bin's only observed count: its marginal carries no
        # evidence independent of the relation itself.
        return "unsupported_bin"
    return "ok"


def process(payload):
    prep = sparse_support.prepare(payload)
    tolerance = number(payload.get("tolerance", 1e-9), "tolerance", minimum=1e-12, maximum=1e-2)
    limit = integer(payload.get("max_iterations", 2000), "max_iterations", minimum=1, maximum=10000)
    min_pairs = integer(payload.get("min_stratum_pairs", 3), "min_stratum_pairs", minimum=1, maximum=1000)
    min_count = integer(payload.get("min_stratum_count", 5), "min_stratum_count", minimum=1, maximum=10**9)
    z = number(payload.get("confidence_z", 1.959963984540054), "confidence_z", minimum=0.1, maximum=6)
    alpha = number(payload.get("fdr_alpha", 0.05), "fdr_alpha", minimum=1e-6, maximum=0.5)
    budget = integer(payload.get("max_relations", 1000), "max_relations", minimum=1, maximum=5000)

    entries = prep["entries"]
    positive = sum(1 for count in entries.values() if count > 0)
    if positive*limit > _BALANCE_WORK_LIMIT:
        raise DomainError("work_limit", "Support-iteration budget exceeds twenty million")

    counts = background.marginals(prep["bins"], entries)
    adjacency, component, components = background.support_components(prep["bins"], entries)
    iterations, converged, no_solution = background.correct(adjacency, component, tolerance, limit)
    populations = background.stratum_populations(entries, prep["stratum_of"], component, converged)

    totals, weight_sums, scales, estimable = [], [], [], []
    for index in range(len(prep["strata"])):
        pairs = populations.get(index, [])
        total = sum(count for _, _, count in pairs)
        weight = sum(counts[i]*counts[j] for i, j, _ in pairs)
        totals.append(total)
        weight_sums.append(weight)
        scales.append(total/weight if weight > 0 else None)
        estimable.append(len(pairs) >= min_pairs and total >= min_count)

    relations = []
    for position, (i, j) in enumerate(prep["relations"]):
        row = {"i": i, "j": j, "distance": j-i, "stratum": prep["stratum_of"][j-i],
               "status": None, "observed": entries.get((i, j)), "expected": None,
               "effect_log2": None, "effect_interval": None, "p_enrichment": None,
               "p_depletion": None, "q_enrichment": None, "q_depletion": None,
               "significant": None}
        if position >= budget:
            row["status"] = "not_computed"
        else:
            row["status"] = _status(prep, counts, component, converged, no_solution,
                                    estimable, (i, j))
        relations.append(row)

    # Leave-one-out stratum scales for the estimable relations, with a work
    # budget for the exact tail and interval evaluations that follow.
    work = 0.0
    for row in relations:
        if row["status"] != "ok":
            continue
        pair = (row["i"], row["j"])
        k = entries[pair]
        stratum = row["stratum"]
        leave_total = totals[stratum] - k
        leave_weight = weight_sums[stratum] - counts[pair[0]]*counts[pair[1]]
        if leave_total <= 0 or leave_weight <= 0:
            row["status"] = "insufficient_stratum_evidence"
            continue
        row["_expectation"] = (leave_total/leave_weight
                               * counts[pair[0]]*counts[pair[1]])
        work += 400.0*math.sqrt(max(k, row["_expectation"], 1.0)) + 2000.0
    if work > _GAMMA_WORK_LIMIT:
        raise DomainError("work_limit", "Tail-probability budget exceeds the supported work")

    computable = [row for row in relations if row["status"] == "ok"]
    confidence_alpha = math.erfc(z/math.sqrt(2))
    for row in computable:
        k = entries[(row["i"], row["j"])]
        mu = row["_expectation"]
        row["expected"] = mu
        lower, upper = poisson.rate_interval(k, confidence_alpha)
        row["effect_interval"] = [None if lower == 0 else math.log2(lower/mu),
                                  math.log2(upper/mu)]
        if k > 0:
            row["effect_log2"] = math.log2(k/mu)
        row["p_enrichment"], row["p_depletion"] = poisson.poisson_tails(k, mu)

    families = {"q_enrichment": "p_enrichment", "q_depletion": "p_depletion"}
    for q_field, p_field in families.items():
        q_values = poisson.benjamini_hochberg(
            [(position, row[p_field]) for position, row in enumerate(relations)
             if row["status"] == "ok"])
        for position, q_value in q_values.items():
            relations[position][q_field] = q_value
    for row in relations:
        if row["status"] != "ok":
            continue
        if row["q_enrichment"] <= alpha:
            row["significant"] = "enriched"
        elif row["q_depletion"] <= alpha:
            row["significant"] = "depleted"
        del row["_expectation"]

    supported = sum(1 for value in component if value >= 0)
    return {
        "model": MODEL,
        "complete": all(row["status"] != "not_computed" for row in relations),
        "correction": {"converged": supported > 0 and len(converged) == supported,
                       "iterations": iterations, "tolerance": tolerance,
                       "components": components, "supported_bins": supported,
                       "converged_bins": len(converged),
                       "no_solution_bins": len(no_solution)},
        "background": {
            "population": "listed valid-bin pairs inside converged positive-support components",
            "masked_bins": "excluded", "unobserved_pairs": "excluded",
            "observed_zeros": "included", "leave_one_out": "each relation removes its own count from the stratum scale",
            "strata": [{"min_distance": low, "max_distance": high,
                        "pairs": len(populations.get(index, [])),
                        "total_count": totals[index], "scale": scales[index],
                        "estimable": estimable[index]}
                       for index, (low, high) in enumerate(prep["strata"])]},
        "error_control": {"method": "benjamini-hochberg",
                          "families": ["enrichment", "depletion"],
                          "family_size": len(computable),
                          "alpha": alpha,
                          "scope": "computed relations with status ok"},
        "relations": relations,
    }
