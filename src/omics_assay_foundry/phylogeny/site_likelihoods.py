import math
from .model_blocks import prepare
from .tree_likelihood import site_log_likelihood

def process(payload):
    tree, sequences = prepare(payload)
    sites = [site_log_likelihood(tree,sequences,i) for i in range(len(next(iter(sequences.values()))))]
    possible = all(value is not None for value in sites)
    return {"model": "JC69-unit-substitution-rate", "root_prior": [0.25]*4,
            "site_log_likelihoods": sites, "log_likelihood": math.fsum(sites) if possible else None,
            "status": "ok" if possible else "zero_probability", "root_length_used": False}
