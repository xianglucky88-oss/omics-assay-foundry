from .factors import prepare
from .inference import infer

def process(payload):
    prior, likelihoods = prepare(payload)
    posterior, evidence = infer(prior, likelihoods)
    return {"model": "diploid-biallelic-trio-no-mutation", "genotypes": [0, 1, 2],
            "posterior": posterior, "log_evidence": evidence}
