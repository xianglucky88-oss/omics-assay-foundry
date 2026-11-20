"""Exact inference by bucket variable elimination.

This engine sums the same factor graph as joint enumeration but multiplies
sparse log-space factors and eliminates variables in reverse topological order
(children before parents). Inbreeding loops stay inside one model: a shared
ancestor appears in several transmission factors and is summed out only once.
Each member's marginal is obtained by retaining that variable while the others
are eliminated, which also yields a retention-invariant log evidence.
"""
import math
from .model import NEG_INF, variable_factors, transmission_factors
from ..errors import DomainError


def _log_sum_exp(values):
    if not values:
        return NEG_INF
    maximum = max(values)
    if maximum == NEG_INF:
        return NEG_INF
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def _merge(left, right, order=None):
    if order is None:
        scope = list(left["scope"])
        for name in right["scope"]:
            if name not in scope:
                scope.append(name)
    else:
        scope = [name for name in order if name in left["scope"] or name in right["scope"]]
    left_position = {name: scope.index(name) for name in left["scope"]}
    right_position = {name: scope.index(name) for name in right["scope"]}
    table = {}
    for left_key, left_weight in left["table"].items():
        left_values = {name: left_key[index] for name, index in
                       zip(left["scope"], range(len(left_key)))}
        for right_key, right_weight in right["table"].items():
            if any(left_values.get(name) != value for name, value in
                   zip(right["scope"], right_key) if name in left_values):
                continue
            key = [0] * len(scope)
            for index, value in enumerate(left_key):
                key[left_position[left["scope"][index]]] = value
            for index, value in enumerate(right_key):
                key[right_position[right["scope"][index]]] = value
            weight = left_weight + right_weight
            if math.isnan(weight) or weight == float("inf"):
                raise DomainError("numeric_range",
                                  "A factor product is outside the supported numerical range")
            table[tuple(key)] = weight
    return {"scope": scope, "table": table}


def _sum_out(factor, name, canonical=None):
    if name not in factor["scope"]:
        return factor
    position = factor["scope"].index(name)
    reduced_scope = factor["scope"][:position] + factor["scope"][position + 1:]
    grouped = {}
    for key, weight in factor["table"].items():
        reduced_key = key[:position] + key[position + 1:]
        grouped.setdefault(reduced_key, []).append(weight)
    if not reduced_scope:
        constant = _log_sum_exp([weight for weights in grouped.values() for weight in weights])
        return {"scope": [], "table": {(): constant}}
    if canonical is not None:
        scope = [n for n in canonical if n in reduced_scope]
        table = {}
        for reduced_key, weights in grouped.items():
            key = tuple(reduced_key[reduced_scope.index(n)] for n in scope)
            table[key] = _log_sum_exp(weights)
    else:
        scope = reduced_scope
        table = {key: _log_sum_exp(weights) for key, weights in grouped.items()}
    return {"scope": scope, "table": {key: value for key, value in table.items() if value != NEG_INF}}


def _eliminate_to(spec, factors, domains, retained, budget):
    """Eliminate every variable except ``retained``; returns its log marginal.

    Variables are summed out in reverse topological order (children before
    parents), which always yields a valid bucket elimination for this DAG and
    keeps a shared ancestor in scope until every child message is formed.
    Factors stay sparse: an assignment absent from a factor table has zero
    probability. Empty-scope results become global log constants.
    """
    factors = [dict(factor, table=dict(factor["table"])) for factor in factors]
    order = list(domains)
    peak = 1
    constant = 0.0
    for index in reversed(spec["order"]):
        name = spec["members"][index]["id"]
        if name == retained:
            continue
        bucket = [factor for factor in factors if name in factor["scope"]]
        factors = [factor for factor in factors if name not in factor["scope"]]
        product = {"scope": [], "table": {(): 0.0}}
        for factor in bucket:
            product = _merge(product, factor, order)
            peak = max(peak, len(product["table"]))
            if len(product["table"]) > budget:
                raise DomainError("budget_exhausted",
                                  "Variable-elimination table exceeds the configured inference budget")
        product = _sum_out(product, name, order)
        if product["scope"]:
            factors.append(product)
        else:
            constant += product["table"][()]
    retained_total = {"scope": [], "table": {(): constant}}
    for factor in factors:
        if retained in factor["scope"]:
            retained_total = _merge(retained_total, factor, order)
    if retained not in retained_total["scope"]:
        value = _log_sum_exp(list(retained_total["table"].values()))
        return {genotype: value for genotype in range(domains[retained])}, peak
    position = retained_total["scope"].index(retained)
    marginal = {key[position]: weight for key, weight in retained_total["table"].items()}
    return marginal, peak


def eliminate_posteriors(spec):
    unary, by_id = variable_factors(spec)
    factors = unary + transmission_factors(spec, by_id)
    domains = {member["id"]: member["domain"] for member in spec["members"]}
    budget = spec["max_states"]

    first = next(iter(domains))
    evidence_marginal, _ = _eliminate_to(spec, factors, domains, first, budget)
    log_evidence = _log_sum_exp(list(evidence_marginal.values()))
    if log_evidence == NEG_INF:
        raise DomainError("incompatible_evidence",
                          "Evidence has zero probability under the joint Mendelian model")
    if not math.isfinite(log_evidence):
        raise DomainError("numeric_range", "Log evidence is outside the supported numerical range")

    posteriors = {}
    largest_factor = 1
    for member in spec["members"]:
        marginal_logs, peak = _eliminate_to(spec, factors, domains, member["id"], budget)
        largest_factor = max(largest_factor, peak)
        row = [marginal_logs.get(genotype, NEG_INF) for genotype in range(member["domain"])]
        if all(value == NEG_INF for value in row):
            raise DomainError("incompatible_evidence",
                              f"Member {member['id']} has no genotype compatible with the evidence")
        posteriors[member["id"]] = [math.exp(value - log_evidence) if value != NEG_INF else 0.0
                                    for value in row]

    return posteriors, log_evidence, {"method": "variable_elimination",
                                      "joint_states": math.prod(domains.values()),
                                      "largest_factor": largest_factor}
