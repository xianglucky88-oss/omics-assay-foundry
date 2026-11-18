import math
from itertools import product
from .factors import transmission
from ..errors import DomainError

def infer(prior, likelihoods):
    states = []
    for father, mother, child in product(range(3), repeat=3):
        factors = (prior[father], prior[mother], transmission(father, mother, child),
                   likelihoods["father"][father], likelihoods["mother"][mother], likelihoods["child"][child])
        if all(x > 0 for x in factors):
            states.append(((father, mother, child), sum(math.log(x) for x in factors)))
    if not states:
        raise DomainError("incompatible_evidence", "Evidence has zero probability under the diploid Mendelian model")
    maximum = max(x[1] for x in states)
    scaled = [(state, math.exp(value-maximum)) for state, value in states]
    total = math.fsum(value for _, value in scaled)
    posterior = {role: [0.0]*3 for role in ("father", "mother", "child")}
    for state, value in scaled:
        for role, genotype in zip(posterior, state):
            posterior[role][genotype] += value/total
    return posterior, maximum + math.log(total)
