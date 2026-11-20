"""Result assembly for the pedigree endpoint.

Two contracts share one route. The original complete diploid trio keeps its
exact request and response. The joint-pedigree contract accepts arbitrary
acyclic families with declared ploidy and unobserved members; it is evaluated
as one statistical model, never as an average of split trio calls.
"""
import math
from .factors import prepare as prepare_trio
from .inference import infer as infer_trio
from .graph import prepare as prepare_pedigree
from .enumeration import enumerate_posteriors, joint_state_count
from .elimination import eliminate_posteriors
from ..errors import DomainError


def _entropy(probabilities):
    return -sum(p * math.log(p) for p in probabilities if p > 0)


def _information_gain(posterior, prior):
    gain = 0.0
    for p, q in zip(posterior, prior):
        if p > 0:
            if q <= 0:
                raise DomainError("numeric_range", "Prior support is too narrow for a finite information gain")
            gain += p * math.log(p / q)
    return max(0.0, gain)


def _prior_marginals(spec):
    stripped = {"frequency": spec["frequency"], "max_states": spec["max_states"],
                "order": spec["order"], "members": []}
    for member in spec["members"]:
        clone = dict(member)
        clone["likelihoods"] = None
        stripped["members"].append(clone)
    if joint_state_count(stripped) <= stripped["max_states"]:
        marginals, _, _ = enumerate_posteriors(stripped)
    else:
        marginals, _, _ = eliminate_posteriors(stripped)
    return marginals


def _run_joint(spec):
    if joint_state_count(spec) <= spec["max_states"]:
        return enumerate_posteriors(spec)
    return eliminate_posteriors(spec)


def _process_pedigree(payload):
    spec = prepare_pedigree(payload)
    posteriors, log_evidence, inference = _run_joint(spec)
    priors = _prior_marginals(spec)

    notices = []
    prior_only = [member["id"] for member in spec["members"] if member["support"] == "prior_only"]
    if prior_only:
        notices.append({"code": "insufficient_information",
                        "message": "No assayed member is connected to these members; posteriors equal model priors",
                        "members": prior_only})

    members = []
    for member in spec["members"]:
        name = member["id"]
        posterior = posteriors[name]
        prior = priors[name]
        map_index = max(range(member["domain"]), key=lambda genotype: posterior[genotype])
        members.append({
            "id": name,
            "ploidy": member["ploidy"],
            "support": member["support"],
            "observed": member["likelihoods"] is not None,
            "genotypes": list(range(member["domain"])),
            "posterior": posterior,
            "prior": prior,
            "map_genotype": map_index,
            "map_posterior": posterior[map_index],
            "posterior_entropy_nats": _entropy(posterior),
            "information_gain_nats": _information_gain(posterior, prior),
        })

    return {
        "model": "joint-pedigree-biallelic-no-mutation-v1",
        "allele_frequency": spec["frequency"],
        "genotypes_diploid": [0, 1, 2],
        "genotypes_haploid": [0, 1],
        "log_evidence": log_evidence,
        "inference": inference,
        "common_ancestors": spec["common_ancestors"],
        "notices": notices,
        "members": members,
    }


def process(payload):
    if "members" in payload:
        return _process_pedigree(payload)
    prior, likelihoods = prepare_trio(payload)
    posterior, evidence = infer_trio(prior, likelihoods)
    return {"model": "diploid-biallelic-trio-no-mutation", "genotypes": [0, 1, 2],
            "posterior": posterior, "log_evidence": evidence}
