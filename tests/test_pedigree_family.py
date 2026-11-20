import json
import math
import itertools
import pytest
from omics_assay_foundry.api import dispatch, ROUTES
from omics_assay_foundry.errors import DomainError


def family_payload(**overrides):
    payload = {
        "allele_frequency": 0.2,
        "members": [
            {"id": "ancestor", "likelihoods": [0.05, 0.7, 0.25]},
            {"id": "paternal_other"},
            {"id": "maternal_other"},
            {"id": "father", "parents": ["ancestor", "paternal_other"]},
            {"id": "m", "parents": ["ancestor", "maternal_other"], "likelihoods": [0.1, 0.8, 0.1]},
            {"id": "sibling_one", "parents": ["father", "m"], "likelihoods": [0.02, 0.9, 0.08]},
            {"id": "sibling_two", "parents": ["father", "m"], "likelihoods": [0.01, 0.85, 0.14]},
            {"id": "gamete", "ploidy": 1, "parents": ["father", "m"], "source_parent": "m",
             "likelihoods": [0.2, 0.8]},
        ],
    }
    payload.update(overrides)
    return payload


def test_joint_pedigree_example_is_serializable(sample):
    result = dispatch("/pedigree", sample("pedigree_family"))
    json.dumps(result, allow_nan=False)
    assert result["model"] == "joint-pedigree-biallelic-no-mutation-v1"


def test_legacy_trio_contract_is_unchanged(sample):
    result = dispatch("/pedigree", sample("pedigree"))
    assert set(result) == {"model", "genotypes", "posterior", "log_evidence"}
    assert result["model"] == "diploid-biallelic-trio-no-mutation"
    assert set(result["posterior"]) == {"father", "mother", "child"}


def test_posteriors_normalize_and_report_support():
    result = dispatch("/pedigree", family_payload())
    for member in result["members"]:
        assert sum(member["posterior"]) == pytest.approx(1.0, abs=1e-12)
        assert len(member["posterior"]) == member["ploidy"] + 1
        assert member["support"] in {"observed", "inferred", "prior_only"}
        assert member["map_genotype"] == max(range(len(member["posterior"])),
                                              key=lambda g: member["posterior"][g])


def test_unobserved_parent_participates_in_the_joint_model():
    result = dispatch("/pedigree", family_payload())
    father = next(m for m in result["members"] if m["id"] == "father")
    assert father["observed"] is False and father["support"] == "inferred"
    # The two observed siblings update the ungenotyped father away from its prior.
    assert father["information_gain_nats"] > 0.0
    assert father["posterior"] != pytest.approx(father["prior"], abs=1e-12)


def test_inbreeding_common_ancestor_is_accepted_not_flagged_as_cycle():
    result = dispatch("/pedigree", family_payload())
    assert result["common_ancestors"] == ["ancestor"]


def test_directed_parent_cycle_is_rejected():
    payload = {"allele_frequency": 0.3, "members": [
        {"id": "a", "ploidy": 1, "parents": ["b"]},
        {"id": "b", "ploidy": 1, "parents": ["a"]}]}
    with pytest.raises(DomainError) as error:
        dispatch("/pedigree", payload)
    assert error.value.code == "illegal_cycle"


def test_enumeration_and_variable_elimination_agree():
    payload = family_payload()
    enumerated = dispatch("/pedigree", payload)
    payload["max_states"] = 500
    eliminated = dispatch("/pedigree", payload)
    assert enumerated["inference"]["method"] == "joint_enumeration"
    assert eliminated["inference"]["method"] == "variable_elimination"
    for a, b in zip(enumerated["members"], eliminated["members"]):
        assert a["posterior"] == pytest.approx(b["posterior"], abs=1e-10)
        assert a["prior"] == pytest.approx(b["prior"], abs=1e-10)
    assert enumerated["log_evidence"] == pytest.approx(eliminated["log_evidence"], abs=1e-10)


def test_inference_budget_exhaustion_is_distinct_from_incompatibility():
    payload = {"allele_frequency": 0.3, "max_states": 1, "members": [
        {"id": "f"}, {"id": "m"}, {"id": "c", "parents": ["f", "m"]}]}
    with pytest.raises(DomainError) as error:
        dispatch("/pedigree", payload)
    assert error.value.code == "budget_exhausted"


def test_incompatible_evidence_is_an_explicit_error():
    payload = {"allele_frequency": 0.3, "members": [
        {"id": "f", "likelihoods": [1, 0, 0]},
        {"id": "m", "likelihoods": [1, 0, 0]},
        {"id": "c", "parents": ["f", "m"], "likelihoods": [0, 0, 1]}]}
    with pytest.raises(DomainError) as error:
        dispatch("/pedigree", payload)
    assert error.value.code == "incompatible_evidence"


def test_disconnected_unobserved_member_keeps_its_prior_and_notices():
    payload = {"allele_frequency": 0.2, "members": [
        {"id": "lone"},
        {"id": "f", "likelihoods": [1, 0, 0]},
        {"id": "m", "likelihoods": [1, 0, 0]},
        {"id": "c", "parents": ["f", "m"]}]}
    result = dispatch("/pedigree", payload)
    lone = next(m for m in result["members"] if m["id"] == "lone")
    assert lone["support"] == "prior_only"
    assert lone["posterior"] == pytest.approx(lone["prior"], abs=1e-12)
    assert lone["information_gain_nats"] == 0.0
    assert result["notices"][0]["code"] == "insufficient_information"
    assert "lone" in result["notices"][0]["members"]


def test_haploid_member_inherits_from_declared_source_parent():
    payload = {"allele_frequency": 0.4, "members": [
        {"id": "f", "likelihoods": [1, 0, 0]},
        {"id": "m", "likelihoods": [0, 0, 1]},
        {"id": "h", "ploidy": 1, "parents": ["f", "m"], "source_parent": "m",
         "likelihoods": [1, 1]}]}
    result = dispatch("/pedigree", payload)
    haploid = next(m for m in result["members"] if m["id"] == "h")
    assert haploid["posterior"] == pytest.approx([0.0, 1.0])


def test_haploid_two_parents_requires_source_parent():
    payload = {"allele_frequency": 0.4, "members": [
        {"id": "f"}, {"id": "m"},
        {"id": "h", "ploidy": 1, "parents": ["f", "m"]}]}
    with pytest.raises(DomainError) as error:
        dispatch("/pedigree", payload)
    assert error.value.code == "invalid_source_parent"


def test_likelihood_scale_does_not_change_posteriors():
    base = family_payload()
    scaled = family_payload()
    for member in scaled["members"]:
        if member.get("likelihoods"):
            member["likelihoods"] = [value * 1e-200 for value in member["likelihoods"]]
    a = dispatch("/pedigree", base)
    b = dispatch("/pedigree", scaled)
    for left, right in zip(a["members"], b["members"]):
        assert left["posterior"] == pytest.approx(right["posterior"], abs=1e-12)
    assert math.isfinite(b["log_evidence"])


def test_extreme_likelihood_ratio_stays_finite_and_honest():
    payload = {"allele_frequency": 0.5, "members": [
        {"id": "f", "likelihoods": [1e300, 1e-300, 1e-300]},
        {"id": "m"}, {"id": "c", "parents": ["f", "m"]}]}
    result = dispatch("/pedigree", payload)
    assert math.isfinite(result["log_evidence"])
    father = next(m for m in result["members"] if m["id"] == "f")
    assert father["posterior"][0] == pytest.approx(1.0)
    assert all(math.isfinite(p) for p in father["posterior"])


def test_small_family_evidence_is_independently_recomputable():
    payload = {"allele_frequency": 0.5, "members": [
        {"id": "f", "likelihoods": [1, 0, 0]},
        {"id": "m", "likelihoods": [0, 0, 1]},
        {"id": "c", "parents": ["f", "m"], "likelihoods": [1, 1, 1]}]}
    result = dispatch("/pedigree", payload)
    frequency = 0.5

    def prior(genotype):
        return ((1 - frequency) ** 2,
                2 * frequency * (1 - frequency),
                frequency ** 2)[genotype]

    def transmission(fp, mp, cp):
        pg, mg = fp / 2, mp / 2
        return ((1 - pg) * (1 - mg), pg * (1 - mg) + (1 - pg) * mg, pg * mg)[cp]

    likelihoods = {"f": [1, 0, 0], "m": [0, 0, 1], "c": [1, 1, 1]}
    weights = {}
    for fg, mg, cg in itertools.product(range(3), repeat=3):
        weights[(fg, mg, cg)] = (prior(fg) * prior(mg) * transmission(fg, mg, cg)
                                 * likelihoods["f"][fg] * likelihoods["m"][mg]
                                 * likelihoods["c"][cg])
    evidence = sum(weights.values())
    hand_child = [sum(weights[(fg, mg, cg)] for fg in range(3) for mg in range(3)) / evidence
                  for cg in range(3)]
    service_child = next(m for m in result["members"] if m["id"] == "c")["posterior"]
    assert service_child == pytest.approx(hand_child, abs=1e-12)
    assert result["log_evidence"] == pytest.approx(math.log(evidence), abs=1e-12)


def test_unknown_parent_and_duplicate_member_are_rejected():
    payload = {"allele_frequency": 0.3, "members": [
        {"id": "f"}, {"id": "c", "parents": ["f", "ghost"]}]}
    with pytest.raises(DomainError) as error:
        dispatch("/pedigree", payload)
    assert error.value.code == "unknown_parent"

    duplicate = {"allele_frequency": 0.3, "members": [{"id": "a"}, {"id": "a"}]}
    with pytest.raises(DomainError) as error:
        dispatch("/pedigree", duplicate)
    assert error.value.code == "duplicate_member"


def test_diploid_member_requires_zero_or_two_parents():
    payload = {"allele_frequency": 0.3, "members": [
        {"id": "f"}, {"id": "c", "parents": ["f"]}]}
    with pytest.raises(DomainError) as error:
        dispatch("/pedigree", payload)
    assert error.value.code == "invalid_parent"


def test_invalid_ploidy_is_rejected():
    payload = {"allele_frequency": 0.3, "members": [{"id": "a", "ploidy": 3}]}
    with pytest.raises(DomainError) as error:
        dispatch("/pedigree", payload)
    assert error.value.code == "invalid_ploidy"


def test_unknown_option_is_rejected_for_joint_contract():
    payload = family_payload()
    payload["unexpected"] = True
    with pytest.raises(DomainError, match="Unsupported fields"):
        dispatch("/pedigree", payload)


def test_boundary_allele_frequency_zero():
    result = dispatch("/pedigree", {"allele_frequency": 0.0,
                                    "members": [{"id": "a"}, {"id": "b"},
                                                {"id": "c", "parents": ["a", "b"]}]})
    member = result["members"][0]
    assert member["posterior"] == pytest.approx([1.0, 0.0, 0.0])
    assert math.isfinite(result["log_evidence"])
