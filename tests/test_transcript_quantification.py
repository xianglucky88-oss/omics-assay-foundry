import json

import pytest

from omics_assay_foundry.api import dispatch
from omics_assay_foundry.errors import DomainError

def run(payload):
    return dispatch('/transcripts', payload)

def by_id(result):
    return {row["id"]: row for row in result["transcripts"]}

# --------------------------------------------------------------------------
# Backwards compatibility: the legacy shape keeps the original model.
# --------------------------------------------------------------------------

def test_legacy_shape_keeps_the_unbiased_model(sample):
    result = run(sample("transcripts"))
    assert result["model"] == "fixed-effective-length-unbiased-fragments"
    assert result["effective_length_model"] == "fixed-effective-length-v1"
    assert "uncertainty" not in result
    assert [row["tpm"] for row in result["transcripts"]] == pytest.approx(
        [828427.1247, 171572.8753], rel=1e-6)

def test_extended_request_without_new_measurements_is_unbiased():
    # No lengths, bias or replicates beyond the compatibility lists: the
    # read-fraction model reduces to the unbiased compatibility model, so
    # equal effective lengths give equal abundance.
    payload = {"effective_lengths": {"a": 100, "b": 100},
               "fragments": [["a"], ["b"], ["a", "b"]],
               "observation_counts": [1, 1, 1]}
    result = run(payload)
    assert result["bias_model"] == "no-bias"
    assert [row["molecule_fraction"] for row in result["transcripts"]] == pytest.approx(
        [0.5, 0.5])
    assert [row["expected_fragments"] for row in result["transcripts"]] == pytest.approx(
        [1.5, 1.5])

def test_small_sample_unique_and_ambiguous_allocation():
    payload = {"effective_lengths": {"a": 100, "b": 200},
               "fragments": [["a"], ["a"], ["b"], ["a", "b"]],
               "observation_counts": [2, 1, 3, 1]}
    result = run(payload)
    rows = by_id(result)
    assert sum(row["expected_fragments"] for row in result["transcripts"]) == pytest.approx(7)
    # Six observations are unique (three each); the ambiguous one is split by
    # current molecule abundance, and the M-step divides by the effective
    # length, so the longer b holds fewer molecules per allocated fragment.
    assert rows["a"]["expected_fragments"] + rows["b"]["expected_fragments"] == pytest.approx(7)
    share = rows["a"]["molecule_fraction"]/(rows["a"]["molecule_fraction"] + rows["b"]["molecule_fraction"])
    assert rows["a"]["expected_fragments"] == pytest.approx(3 + share)
    assert rows["b"]["expected_fragments"] == pytest.approx(3 + (1 - share))
    assert rows["a"]["tpm"] > rows["b"]["tpm"]
    # The 100:200 effective lengths imply 2:1 molecules at equal allocated
    # fragments; ambiguity tilts slightly towards a, so the ratio is above 2.
    assert rows["a"]["molecule_fraction"] / rows["b"]["molecule_fraction"] > 2
    assert result["converged"]
    assert [row["support_status"] for row in result["transcripts"]] == ["identified", "identified"]

# --------------------------------------------------------------------------
# Fragment-length distribution drives the effective length.
# --------------------------------------------------------------------------

def test_effective_length_uses_observed_fragment_lengths():
    payload = {"transcript_lengths": {"a": 500, "b": 1000},
               "fragments": [["a"], ["b"]],
               "fragment_lengths": [100, 100]}
    rows = by_id(run(payload))
    # Mean start positions available for a 100 nt fragment.
    assert rows["a"]["effective_length"] == pytest.approx(401)
    assert rows["b"]["effective_length"] == pytest.approx(901)
    # Molecule fraction is read fraction over effective length.
    assert rows["a"]["molecule_fraction"] == pytest.approx(901/1302)
    assert rows["b"]["molecule_fraction"] == pytest.approx(401/1302)

def test_distribution_and_per_fragment_share_one_effective_length():
    # Knowing each fragment's own length and knowing only its marginal
    # distribution carry different allocation information, so abundances need
    # not match; the effective length they share is the one common convention.
    base = {"transcript_lengths": {"a": 500, "b": 1000},
            "fragments": [["a"], ["b"], ["a", "b"]]}
    per_fragment = run(dict(base, fragment_lengths=[200, 300, 250]))
    histogram = run(dict(base, fragment_length_distribution={"200": 1, "250": 1, "300": 1}))
    assert [row["effective_length"] for row in per_fragment["transcripts"]] == pytest.approx(
        [row["effective_length"] for row in histogram["transcripts"]])
    assert per_fragment["length_observation"] == "per-fragment"
    assert histogram["length_observation"] == "distribution-only"

def test_supplied_effective_lengths_must_match_derivation():
    payload = {"transcript_lengths": {"a": 500}, "fragments": [["a"]],
               "fragment_lengths": [200], "effective_lengths": {"a": 999}}
    with pytest.raises(DomainError) as error:
        run(payload)
    assert error.value.code == "inconsistent_effective_length"

def test_transcript_shorter_than_every_fragment_is_rejected():
    payload = {"transcript_lengths": {"a": 100}, "fragments": [["a"]],
               "fragment_lengths": [500]}
    with pytest.raises(DomainError) as error:
        run(payload)
    assert error.value.code == "zero_effective_length"

def test_mixing_per_fragment_and_histogram_lengths_is_rejected():
    payload = {"transcript_lengths": {"a": 1000}, "fragments": [["a"]],
               "fragment_lengths": [50], "fragment_length_distribution": [50]}
    with pytest.raises(DomainError) as error:
        run(payload)
    assert error.value.code == "duplicate_option"

# --------------------------------------------------------------------------
# Relative bias weights change allocation within a fragment.
# --------------------------------------------------------------------------

def test_bias_weights_change_ambiguous_allocation():
    # Unique fragments anchor both transcripts; the bias then decides how the
    # ambiguous fragment is split.
    payload = {"effective_lengths": {"a": 100, "b": 100},
               "fragments": [["a"], ["b"], ["a", "b"]],
               "fragment_bias": [{"a": 1}, {"b": 1}, {"a": 3, "b": 1}]}
    result = run(payload)
    rows = by_id(result)
    unbiased = by_id(run({"effective_lengths": {"a": 100, "b": 100},
                          "fragments": [["a"], ["b"], ["a", "b"]]}))
    assert result["indistinguishable_groups"] == []
    # The declared 3:1 bias tilts ambiguous allocation towards a, so a receives
    # more of the shared fragments than under the unbiased split.
    assert rows["a"]["expected_fragments"] > unbiased["a"]["expected_fragments"]
    assert rows["a"]["expected_fragments"] > 1.5
    assert rows["a"]["expected_fragments"] + rows["b"]["expected_fragments"] == pytest.approx(3)
    # Higher detection efficiency means the same observations imply fewer
    # molecules, so a's effective yield is larger and its molecule fraction is
    # correspondingly lower.
    assert rows["a"]["effective_yield"] > rows["b"]["effective_yield"]
    assert rows["a"]["molecule_fraction"] < rows["b"]["molecule_fraction"]

def test_global_bias_rescaling_cancels():
    low = run({"effective_lengths": {"a": 100, "b": 100},
               "fragments": [["a"], ["a"], ["a", "b"]],
               "bias_weights": {"a": 1, "b": 3}})
    high = run({"effective_lengths": {"a": 100, "b": 100},
                "fragments": [["a"], ["a"], ["a", "b"]],
                "bias_weights": {"a": 10, "b": 30}})
    assert [row["tpm"] for row in low["transcripts"]] == pytest.approx(
        [row["tpm"] for row in high["transcripts"]])

def test_bias_does_not_change_a_fully_unique_transcript():
    plain = run({"effective_lengths": {"a": 100}, "fragments": [["a"]],
                 "bias_weights": {"a": 1}})
    favoured = run({"effective_lengths": {"a": 100}, "fragments": [["a"]],
                    "bias_weights": {"a": 4}})
    assert favoured["transcripts"][0]["tpm"] == pytest.approx(
        plain["transcripts"][0]["tpm"])

def test_bias_must_name_compatible_transcripts():
    with pytest.raises(DomainError) as error:
        run({"effective_lengths": {"a": 100, "b": 100}, "fragments": [["a"]],
             "fragment_bias": [{"b": 2}]})
    assert error.value.code == "invalid_bias"

# --------------------------------------------------------------------------
# Observation counts weight the estimate and widen resampling uncertainty.
# --------------------------------------------------------------------------

def test_observation_counts_weight_the_estimate():
    result = run({"effective_lengths": {"a": 100, "b": 100},
                  "fragments": [["a"], ["b"]], "observation_counts": [3, 1]})
    rows = by_id(result)
    assert rows["a"]["tpm"] == pytest.approx(750000)
    assert rows["b"]["tpm"] == pytest.approx(250000)
    assert result["total_observations"] == 4

def test_more_observations_narrow_the_resampling_interval():
    base = {"effective_lengths": {"a": 100, "b": 100, "c": 100},
            "fragments": [["a"], ["a", "b"], ["b", "c"], ["c"]],
            "resamples": 300, "seed": 7}
    single = run(base)
    repeated = run(dict(base, observation_counts=25))

    def width(row):
        return row["tpm_interval"][1] - row["tpm_interval"][0]
    for one, many in zip(single["transcripts"], repeated["transcripts"]):
        assert width(many) < width(one)

# --------------------------------------------------------------------------
# Deterministic resampling: explicit seed, chunk-size independence.
# --------------------------------------------------------------------------

def test_resampling_requires_explicit_seed():
    with pytest.raises(DomainError) as error:
        run({"effective_lengths": {"a": 100}, "fragments": [["a"]], "resamples": 10})
    assert error.value.code == "missing_seed"

def test_same_seed_reproduces_identical_result():
    payload = {"effective_lengths": {"a": 100, "b": 150, "c": 80},
               "fragments": [["a"], ["a", "b"], ["c"], ["b", "c"]],
               "observation_counts": [2, 1, 3, 1], "resamples": 40, "seed": 99}
    assert run(payload) == run(payload)

def test_chunk_size_does_not_change_results():
    payload = {"effective_lengths": {"a": 100, "b": 150, "c": 80},
               "fragments": [["a"], ["a", "b"], ["c"], ["b", "c"]],
               "observation_counts": [2, 1, 3, 1], "resamples": 40, "seed": 123}
    reference = json.dumps(run(payload), sort_keys=True)
    for chunk in (1, 2, 3, 5, 17):
        assert json.dumps(run(dict(payload, chunk_size=chunk)), sort_keys=True) == reference

def test_different_seeds_change_the_interval():
    base = {"effective_lengths": {"a": 100, "b": 100, "c": 100},
            "fragments": [["a"], ["a", "b"], ["b", "c"], ["c"]], "resamples": 60}
    first = run(dict(base, seed=1))
    second = run(dict(base, seed=2))
    assert first["transcripts"][0]["tpm_interval"] != second["transcripts"][0]["tpm_interval"]

# --------------------------------------------------------------------------
# Uncertainty statuses and budget exhaustion.
# --------------------------------------------------------------------------

def test_no_resampling_reports_no_interval():
    result = run({"effective_lengths": {"a": 100}, "fragments": [["a"]],
                  "observation_counts": [1]})
    assert result["uncertainty"]["status"] == "not_requested"
    assert result["uncertainty"]["usable"] is False
    assert result["transcripts"][0]["tpm_interval"] is None
    # No resampling was asked for, so the point estimate is still usable.
    assert result["usable_for_quantitative_conclusions"] is True

def test_budget_exhaustion_marks_conclusions_unusable():
    result = run({"effective_lengths": {"a": 100}, "fragments": [["a"]],
                  "resamples": 100, "seed": 1, "bootstrap_work_limit": 5})
    assert result["uncertainty"]["status"] == "budget_exceeded"
    assert result["uncertainty"]["usable"] is False
    assert result["usable_for_quantitative_conclusions"] is False
    assert all(row["tpm_interval"] is None for row in result["transcripts"])
    assert any("budget_exceeded" in note for note in result["comments"])

def test_insufficient_converged_replicates_hide_the_interval():
    payload = {"effective_lengths": {"a": 100, "b": 100},
               "fragments": [["a"], ["a", "b"]], "max_iterations": 1,
               "tolerance": 1e-14, "resamples": 10, "seed": 1}
    result = run(payload)
    assert result["converged"] is False
    assert result["uncertainty"]["status"] == "insufficient_replicates"
    assert result["usable_for_quantitative_conclusions"] is False

# --------------------------------------------------------------------------
# Identifiability, low abundance and zero support are distinct.
# --------------------------------------------------------------------------

def test_indistinguishable_group_has_no_individual_interval():
    result = run({"effective_lengths": {"a": 100, "b": 100, "c": 50},
                  "fragments": [["a", "b"], ["a", "b"], ["c"]], "seed": 5})
    assert result["indistinguishable_groups"] == [["a", "b"]]
    rows = by_id(result)
    for name in ("a", "b"):
        assert rows[name]["support_status"] == "indistinguishable"
        assert rows[name]["individually_identified"] is False
        assert rows[name]["tpm_interval"] is None
    assert result["status_counts"]["indistinguishable_groups"] == 1
    detail = result["group_details"][0]
    assert detail["members"] == ["a", "b"]
    assert detail["identified_combination_fraction"] == pytest.approx(0.5)
    assert detail["combination_interval"] is not None

def test_group_combination_respects_group_coefficients():
    result = run({"effective_lengths": {"a": 100, "b": 100},
                  "fragments": [["a", "b"], ["a", "b"]],
                  "fragment_bias": [{"a": 1, "b": 2}, {"a": 1, "b": 2}]})
    detail = result["group_details"][0]
    assert detail["coefficients"] == pytest.approx([1.0, 2.0])
    combination = detail["identified_combination_fraction"]
    assert combination == pytest.approx(
        sum(c*f for c, f in zip(detail["coefficients"],
                                [row["molecule_fraction"] for row in result["transcripts"]])))

def test_zero_support_is_reported_separately():
    result = run({"effective_lengths": {"a": 100, "b": 100},
                  "fragments": [["a"]], "seed": 1})
    rows = by_id(result)
    assert rows["a"]["support_status"] == "identified"
    assert rows["b"]["support_status"] == "zero_support"
    assert rows["b"]["tpm"] == 0
    assert rows["b"]["tpm_interval"] is None
    assert result["status_counts"]["zero_support"] == 1

def test_low_abundance_is_reported_separately():
    result = run({"effective_lengths": {"a": 100, "b": 100},
                  "fragments": [["a"], ["a"], ["a"], ["a"], ["a", "b"]],
                  "low_abundance_tpm": 100000, "resamples": 50, "seed": 1})
    rows = by_id(result)
    assert rows["a"]["support_status"] == "identified"
    assert rows["b"]["support_status"] == "low_abundance"
    assert rows["b"]["tpm"] < 100000

def test_convergence_does_not_imply_identifiability():
    result = run({"effective_lengths": {"a": 100, "b": 100},
                  "fragments": [["a", "b"], ["a", "b"]], "seed": 1})
    assert result["converged"] is True
    assert result["status_counts"]["indistinguishable_transcripts"] == 2
    assert "does not imply individual" in result["convergence_meaning"]

# --------------------------------------------------------------------------
# Input robustness.
# --------------------------------------------------------------------------

def test_fragment_objects_carry_lengths_bias_and_counts():
    payload = {"transcript_lengths": {"a": 500, "b": 1000},
               "fragments": [{"compatible": ["a"], "length": 200, "count": 2},
                             {"compatible": ["a", "b"], "length": 300, "bias": {"b": 2}}]}
    result = run(payload)
    assert result["length_observation"] == "per-fragment"
    rows = by_id(result)
    assert rows["a"]["observed_observations"] == 3
    assert rows["b"]["observed_observations"] == 1

def test_conflicting_aliases_are_rejected():
    with pytest.raises(DomainError) as error:
        run({"effective_lengths": {"a": 100}, "fragments": [["a"]],
             "resamples": 5, "bootstrap_resamples": 6, "seed": 1})
    assert error.value.code == "duplicate_option"

def test_nested_resampling_configuration_is_accepted():
    result = run({"effective_lengths": {"a": 100}, "fragments": [["a"]],
                  "random": {"seed": 3, "resamples": 10}})
    assert result["uncertainty"]["seed"] == 3
    assert result["uncertainty"]["resamples"] == 10

def test_unknown_extended_option_is_rejected():
    with pytest.raises(DomainError) as error:
        run({"effective_lengths": {"a": 100}, "fragments": [["a"]], "mystery": 1})
    assert error.value.code == "unsupported_field"

def test_zero_total_observations_is_rejected():
    with pytest.raises(DomainError) as error:
        run({"effective_lengths": {"a": 100}, "fragments": [["a"]],
             "observation_counts": [0]})
    assert error.value.code == "no_observations"

def test_result_is_json_serializable_without_nan(sample):
    payload = dict(sample("transcripts"), observation_counts=[1, 2, 1, 1],
                   resamples=20, seed=5)
    json.dumps(run(payload), allow_nan=False)

def test_group_combination_is_the_only_identified_quantity():
    from omics_assay_foundry.transcripts.profile import prepare
    from omics_assay_foundry.transcripts import model

    payload = {"effective_lengths": {"a": 100, "b": 100, "c": 100},
               "fragments": [["a", "b"], ["a", "b"], ["c"]]}
    spec = prepare(payload)
    abundance, counts, *_ = model.estimate(spec["effective_lengths"], spec["rows"], 2000, 1e-12)
    groups, _, coefficients = model.indistinguishable_groups(spec["rows"], spec["supported"])
    group = groups[0]
    combination = sum(coefficients[i]*abundance[i] for i in group)
    # Redistribute the group's mass while preserving the identified
    # combination; the group's total allocation must not move.
    moved = list(abundance)
    transfer = 0.4*abundance[group[0]]
    moved[group[0]] -= transfer
    moved[group[1]] += transfer
    assert sum(coefficients[i]*moved[i] for i in group) == pytest.approx(combination)
    before = model.allocate(len(spec["effective_lengths"]), spec["rows"], abundance)
    after = model.allocate(len(spec["effective_lengths"]), spec["rows"], moved)
    assert sum(before[i] for i in group) == pytest.approx(sum(after[i] for i in group))

def test_no_bias_effective_yield_equals_effective_length(sample):
    result = run(dict(sample("transcripts"), observation_counts=[1, 1, 1, 1]))
    for row in result["transcripts"]:
        assert row["effective_yield"] == pytest.approx(row["effective_length"])

def test_detection_bias_enters_abundance_through_the_yield():
    payload = {"effective_lengths": {"a": 100}, "fragments": [["a"]]}
    plain = run(dict(payload, bias_weights={"a": 1}))["transcripts"][0]
    favoured = run(dict(payload, bias_weights={"a": 4}))["transcripts"][0]
    assert favoured["effective_length"] == pytest.approx(plain["effective_length"])
    assert favoured["effective_yield"] == pytest.approx(4*plain["effective_yield"])
    # A single transcript still holds the whole fraction; with a second
    # transcript its higher yield would lower its share, which the ambiguous
    # bias test above checks.
    assert favoured["tpm"] == pytest.approx(1e6)

def test_fragment_length_alone_does_not_reweight_shared_members():
    # Two representations of the same length distribution must agree exactly:
    # fragment length acts through the effective length L_i, not as a
    # per-fragment reweighting of shared members.
    base = {"transcript_lengths": {"a": 500, "b": 1000}, "fragments": [["a", "b"], ["a", "b"]]}
    per_fragment = run(dict(base, fragment_lengths=[250, 250]))
    histogram = run(dict(base, fragment_length_distribution=[250, 250]))
    assert [row["effective_length"] for row in per_fragment["transcripts"]] == pytest.approx(
        [row["effective_length"] for row in histogram["transcripts"]])
    assert [row["tpm"] for row in per_fragment["transcripts"]] == pytest.approx(
        [row["tpm"] for row in histogram["transcripts"]])


def test_length_forms_agree_on_numbers():
    base = {"transcript_lengths": {"a": 600, "b": 1200},
            "fragments": [["a", "b"], ["a", "b"]]*3}
    per = run(dict(base, fragment_lengths=[300]*6))
    hist = run(dict(base, fragment_length_distribution=[300]*6))
    assert [row["effective_length"] for row in per["transcripts"]] == pytest.approx(
        [row["effective_length"] for row in hist["transcripts"]])
    assert [row["tpm"] for row in per["transcripts"]] == pytest.approx(
        [row["tpm"] for row in hist["transcripts"]])
    assert per["length_observation"] == "per-fragment"
    assert hist["length_observation"] == "distribution-only"
