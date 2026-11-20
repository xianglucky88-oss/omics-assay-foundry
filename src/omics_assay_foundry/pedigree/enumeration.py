"""Exact inference by enumerating the full joint genotype space.

This is the reference engine. Every member state is summed jointly in one
model, so inbreeding loops and unobserved parents are integrated over rather
than approximated by averaging separate trio calls. Two passes keep the
arithmetic in scaled log space without storing the joint table.
"""
import math
from .model import (NEG_INF, founder_log_prior, likelihood_log_weight,
                    transmission_log_weight, _grid)
from ..errors import DomainError


def joint_state_count(spec):
    count = 1
    for member in spec["members"]:
        count *= member["domain"]
    return count


def _assignment_log_weight(assignment, spec, by_id):
    values = {member["id"]: value for member, value in zip(spec["members"], assignment)}
    total = 0.0
    for member, genotype in zip(spec["members"], assignment):
        weight = 0.0 if member["parents"] else \
            founder_log_prior(member["ploidy"], spec["frequency"], genotype)
        if member["likelihoods"] is not None:
            likelihood = likelihood_log_weight(member, genotype)
            if likelihood == NEG_INF or weight == NEG_INF:
                weight = NEG_INF
            else:
                weight += likelihood
        if weight == NEG_INF:
            return NEG_INF
        total += weight
    for child in spec["members"]:
        if not child["parents"]:
            continue
        weight = transmission_log_weight(child, values, by_id)
        if weight == NEG_INF:
            return NEG_INF
        total += weight
    return total


def _iter_weights(spec, by_id):
    domains = [member["domain"] for member in spec["members"]]
    for assignment in _grid(domains):
        yield assignment, _assignment_log_weight(assignment, spec, by_id)


def enumerate_posteriors(spec):
    if joint_state_count(spec) > spec["max_states"]:
        raise DomainError("budget_exhausted",
                          "Joint genotype space exceeds the configured inference budget")
    by_id = {member["id"]: member for member in spec["members"]}
    maximum = NEG_INF
    valid = 0
    for _, log_weight in _iter_weights(spec, by_id):
        if log_weight != NEG_INF:
            valid += 1
            if log_weight > maximum:
                maximum = log_weight
    if valid == 0:
        raise DomainError("incompatible_evidence",
                          "Evidence has zero probability under the joint Mendelian model")

    totals = {member["id"]: [0.0] * member["domain"] for member in spec["members"]}
    scaled_total = 0.0
    for assignment, log_weight in _iter_weights(spec, by_id):
        if log_weight == NEG_INF:
            continue
        scaled = math.exp(log_weight - maximum)
        scaled_total += scaled
        for member, genotype in zip(spec["members"], assignment):
            totals[member["id"]][genotype] += scaled

    if not math.isfinite(scaled_total) or scaled_total <= 0:
        raise DomainError("numeric_range", "Joint evidence is outside the supported numerical range")
    posteriors = {name: [value / scaled_total for value in row] for name, row in totals.items()}
    log_evidence = maximum + math.log(scaled_total)
    if not math.isfinite(log_evidence):
        raise DomainError("numeric_range", "Log evidence is outside the supported numerical range")
    return posteriors, log_evidence, {"method": "joint_enumeration",
                                      "joint_states": joint_state_count(spec),
                                      "valid_states": valid}
