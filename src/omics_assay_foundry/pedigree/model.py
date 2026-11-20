"""Biallelic, no-mutation Mendelian factors shared by every inference engine.

Genotype is the count of alternative alleles: diploid members take states
0/1/2, haploid members 0/1. All factors are produced as log weights; a value of
negative infinity marks an impossible assignment.
"""
import math

NEG_INF = float("-inf")


def founder_log_prior(ploidy, frequency, genotype):
    if ploidy == 1:
        probability = frequency if genotype == 1 else 1.0 - frequency
    else:
        probability = ((1.0 - frequency) ** 2,
                       2.0 * frequency * (1.0 - frequency),
                       frequency ** 2)[genotype]
    return math.log(probability) if probability > 0 else NEG_INF


def gamete_probability(ploidy, genotype, transmitted):
    """Probability that one gamete carries ``transmitted`` (0 or 1) alleles."""
    if ploidy == 1:
        return 1.0 if transmitted == genotype else 0.0
    if genotype == 0:
        return 1.0 if transmitted == 0 else 0.0
    if genotype == 2:
        return 1.0 if transmitted == 1 else 0.0
    return 0.5


def transmission_log_weight(child, assignments, members_by_id):
    """Mendelian transmission factor evaluated at one joint assignment."""
    child_genotype = assignments[child["id"]]
    parent_contributions = []
    for parent_id in child["parents"]:
        if child["ploidy"] == 1 and parent_id != child["source_parent"]:
            continue
        parent = members_by_id[parent_id]
        parent_genotype = assignments[parent_id]
        parent_contributions.append((parent, parent_genotype))
    if child["ploidy"] == 1:
        probability = gamete_probability(parent_contributions[0][0]["ploidy"],
                                         parent_contributions[0][1], child_genotype)
    else:
        (parent_a, genotype_a), (parent_b, genotype_b) = parent_contributions
        probability = sum(
            gamete_probability(parent_a["ploidy"], genotype_a, allele_a)
            * gamete_probability(parent_b["ploidy"], genotype_b, child_genotype - allele_a)
            for allele_a in (0, 1))
    return math.log(probability) if probability > 0 else NEG_INF


def likelihood_log_weight(member, genotype):
    likelihoods = member["likelihoods"]
    value = likelihoods[genotype]
    if value <= 0:
        return NEG_INF
    return math.log(value)


def variable_factors(spec):
    """Per-variable unary log factors: founder prior plus any assay likelihood."""
    members = spec["members"]
    by_id = {member["id"]: member for member in members}
    factors = []
    for member in members:
        if member["parents"]:
            table = [0.0 for _ in range(member["domain"])]
        else:
            table = [founder_log_prior(member["ploidy"], spec["frequency"], genotype)
                     for genotype in range(member["domain"])]
        if member["likelihoods"] is not None:
            table = [_add_log_weights(base, likelihood_log_weight(member, genotype))
                     for genotype, base in enumerate(table)]
        factors.append({"scope": [member["id"]],
                        "table": {(genotype,): weight for genotype, weight in enumerate(table)
                                  if weight != NEG_INF}})
    return factors, by_id


def transmission_factors(spec, by_id):
    factors = []
    for child in spec["members"]:
        if not child["parents"]:
            continue
        scope = [child["id"]] + [parent for parent in child["parents"]
                                 if not (child["ploidy"] == 1 and parent != child["source_parent"])]
        domains = [by_id[name]["domain"] for name in scope]
        table = {}
        for assignment in _grid(domains):
            assignments = {name: value for name, value in zip(scope, assignment)}
            weight = transmission_log_weight(child, assignments, by_id)
            if weight != NEG_INF:
                table[tuple(assignment)] = weight
        factors.append({"scope": scope, "table": table})
    return factors


def _add_log_weights(left, right):
    if left == NEG_INF or right == NEG_INF:
        return NEG_INF
    return left + right


def _grid(domains):
    if not domains:
        yield ()
        return
    current = [0] * len(domains)
    while True:
        yield tuple(current)
        position = len(domains) - 1
        while position >= 0 and current[position] == domains[position] - 1:
            current[position] = 0
            position -= 1
        if position < 0:
            return
        current[position] += 1
