from ..validation import fields, number, sequence
from ..errors import DomainError

def prepare(payload):
    fields(payload, ["allele_frequency", "likelihoods"])
    frequency = number(payload["allele_frequency"], "allele_frequency", minimum=0, maximum=1)
    data = payload["likelihoods"]
    fields(data, ["father", "mother", "child"])
    likelihoods = {}
    for role in ("father", "mother", "child"):
        values = sequence(data[role], role, minimum=3, maximum=3)
        likelihoods[role] = tuple(number(v, role, minimum=0) for v in values)
    prior = ((1-frequency)**2, 2*frequency*(1-frequency), frequency**2)
    return prior, likelihoods

def transmission(father, mother, child):
    p, q = father / 2, mother / 2
    return ((1-p)*(1-q), p*(1-q)+(1-p)*q, p*q)[child]
