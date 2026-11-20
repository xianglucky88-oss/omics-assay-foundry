import math
import itertools
import pytest

from omics_assay_foundry.api import dispatch
from omics_assay_foundry.errors import DomainError
from omics_assay_foundry.phylogeny.substitution import (
    compile_model, discrete_gamma, gamma_quantile, transition_matrix,
)


TREE_LADDER = {"children": [
    {"name": "a", "length": 0.1},
    {"length": 0.2, "children": [
        {"name": "b", "length": 0.1},
        {"length": 0.1, "children": [{"name": "c", "length": 0.15}, {"name": "d", "length": 0.15}]},
    ]},
]}

TREE_BALANCED = {"children": [
    {"length": 0.1, "children": [{"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]},
    {"length": 0.1, "children": [{"name": "c", "length": 0.1}, {"name": "d", "length": 0.1}]},
]}


def compare(payload, route="/phylogeny/evidence"):
    return dispatch(route, payload)


# ---------------------------------------------------------------------------
# Legacy behaviour and scale
# ---------------------------------------------------------------------------

def test_legacy_jc69_single_tree_unchanged(sample):
    assert sample("phylogeny") == {
        "tree": {"children": [{"name": "a", "length": 0.1}, {"name": "b", "length": 0.2}]},
        "sequences": {"a": "ACGT", "b": "AGGT"},
    }


def test_evidence_default_model_reproduces_legacy_jc69(sample):
    legacy = dispatch("/phylogeny", sample("phylogeny"))
    payload = {"sequences": sample("phylogeny")["sequences"],
               "candidates": [{"id": "only", "tree": sample("phylogeny")["tree"]}],
               "resampling": {"replicates": 0}}
    result = compare(payload)
    assert result["candidates"][0]["log_likelihood"] == pytest.approx(legacy["log_likelihood"], abs=1e-12)
    assert result["candidates"][0]["site_log_likelihoods"] == pytest.approx(
        legacy["site_log_likelihoods"], abs=1e-12)
    assert result["candidates"][0]["partitions"][0]["model"]["substitution"] == "JC69"


def test_jc69_transition_matrix_matches_closed_form():
    model = compile_model({})
    for t in (0.0, 0.05, 0.3, 5.0, 100.0):
        matrix = transition_matrix(model, t, 1.0)
        same = 0.25 + 0.75 * math.exp(-4 * t / 3)
        different = 0.25 - 0.25 * math.exp(-4 * t / 3)
        for i in range(4):
            assert matrix[i][i] == pytest.approx(same, abs=1e-14)
            for j in range(4):
                if i != j:
                    assert matrix[i][j] == pytest.approx(different, abs=1e-14)


def test_gtr_rate_matrix_rows_and_stationarity():
    model = compile_model({
        "substitution": "GTR",
        "exchangeabilities": [[0, 1.2, 3, 1], [1.2, 0, 2, 0.5], [3, 2, 0, 1.5], [1, 0.5, 1.5, 0]],
        "frequencies": [0.1, 0.2, 0.3, 0.4],
    })
    for t in (0.01, 0.5, 10.0):
        matrix = transition_matrix(model, t, 1.0)
        for row in matrix:
            assert math.fsum(row) == pytest.approx(1.0, abs=1e-12)
        # Stationary distribution is preserved: pi P = pi.
        for j in range(4):
            assert math.fsum(model["frequencies"][i] * matrix[i][j] for i in range(4)) \
                == pytest.approx(model["frequencies"][j], abs=1e-12)


def test_gtr_expected_rate_normalized_to_one():
    model = compile_model({
        "substitution": "GTR",
        "exchangeabilities": [[0, 50, 1, 1], [50, 0, 1, 1], [1, 1, 0, 50], [1, 1, 50, 0]],
        "frequencies": [0.25, 0.25, 0.25, 0.25],
    })
    # The derivative of the transition matrix at zero, weighted by the
    # stationary frequencies, is the expected substitution rate.
    epsilon = 1e-6
    near = transition_matrix(model, epsilon, 1.0)
    expected_rate = 0.0
    for i in range(4):
        for j in range(4):
            if i != j:
                expected_rate += model["frequencies"][i] * near[i][j] / epsilon
    assert expected_rate == pytest.approx(1.0, abs=1e-4)
    assert model["expected_rate"] == 1.0


# ---------------------------------------------------------------------------
# Discrete rate mixture
# ---------------------------------------------------------------------------

def test_discrete_gamma_matches_published_table():
    rates, weights = discrete_gamma(0.5, 4)
    assert rates == pytest.approx([0.0334, 0.2519, 0.8203, 2.8944], abs=5e-4)
    assert weights == pytest.approx([0.25] * 4)
    assert math.fsum(w * r for w, r in zip(weights, rates)) == pytest.approx(1.0, abs=1e-12)


def test_discrete_gamma_approaches_equal_rates_for_large_shape():
    rates, _ = discrete_gamma(1000.0, 4)
    assert rates == pytest.approx([1.0] * 4, abs=0.05)


def test_gamma_quantile_median_of_exponential():
    # shape 1 is the unit-mean exponential; median is ln 2.
    assert gamma_quantile(1.0, 0.5) == pytest.approx(math.log(2.0), abs=1e-10)


def test_transition_clamping_is_reported_not_hidden():
    from omics_assay_foundry.phylogeny.substitution import transition_matrix
    model = compile_model({})
    # A very long branch drives every off-diagonal toward 1/4; entries must
    # stay within [0, 1] and the report must exist (possibly empty).
    report = {}
    matrix = transition_matrix(model, 1000.0, 1.0, report)
    for row in matrix:
        for value in row:
            assert 0.0 <= value <= 1.0
    assert set(report).issubset({"clamped_entries", "most_negative_raw", "most_positive_raw"})


def test_rate_mixture_is_average_of_categories():
    sequences = {"a": "ACGT", "b": "AGGT"}
    tree = {"children": [{"name": "a", "length": 0.3}, {"name": "b", "length": 0.3}]}

    def total(model):
        return compare({"sequences": sequences, "resampling": {"replicates": 0},
                        "candidates": [{"id": "m", "tree": tree,
                                        "model": {"rate_categories": model}}]}
                       )["candidates"][0]["log_likelihood"]

    single = total({"rates": [1.0], "weights": [1.0]})
    duplicated = total({"rates": [1.0, 1.0], "weights": [0.5, 0.5]})
    assert duplicated == pytest.approx(single, abs=1e-12)
    # Equal weights over rate 0 and rate 2 have mean one, and their per-site
    # log likelihood is log((L(0) + L(2)) / 2).
    mixed = total({"rates": [0.0, 2.0], "weights": [0.5, 0.5]})
    assert mixed != pytest.approx(single, abs=1e-9)


def test_rate_mixture_per_site_is_log_mean_of_category_probabilities():
    sequences = {"a": "A", "b": "A"}
    tree = {"children": [{"name": "a", "length": 0.3}, {"name": "b", "length": 0.3}]}

    def site(model):
        return compare({"sequences": sequences, "resampling": {"replicates": 0},
                        "candidates": [{"id": "m", "tree": tree,
                                        "model": {"rate_categories": model}}]}
                       )["candidates"][0]["site_log_likelihoods"][0]

    # Direct two-taxon probability: sum_i pi_i P(i->A)^2.
    def exact(rate):
        model = compile_model({})
        matrix = transition_matrix(model, 0.3 * rate, 1.0)
        return 0.25 * math.fsum(matrix[i][0] ** 2 for i in range(4))

    expected = math.log(0.5 * exact(0.0) + 0.5 * exact(2.0))
    assert site({"rates": [0.0, 2.0], "weights": [0.5, 0.5]}) == pytest.approx(expected, abs=1e-12)

# ---------------------------------------------------------------------------
# Independent verification on small problems
# ---------------------------------------------------------------------------

def test_gtr_two_taxon_probability_matches_analytic():
    spec = {"substitution": "GTR",
            "exchangeabilities": [[0, 2, 1, 0.5], [2, 0, 1, 1], [1, 1, 0, 2], [0.5, 1, 2, 0]],
            "frequencies": [0.2, 0.3, 0.25, 0.25]}
    model = compile_model(spec)
    length = 0.15
    matrix = transition_matrix(model, length, 1.0)
    frequencies = model["frequencies"]
    expected = math.fsum(frequencies[i] * matrix[i][0] * matrix[i][1] for i in range(4))
    result = compare({"sequences": {"a": "A", "b": "C"},
                      "candidates": [{"id": "x", "tree": {"children": [
                          {"name": "a", "length": length}, {"name": "b", "length": length}]},
                          "model": spec}],
                      "resampling": {"replicates": 0}})
    assert math.exp(result["candidates"][0]["log_likelihood"]) == pytest.approx(expected, rel=1e-12)


def test_gamma_mixture_two_taxon_probability_matches_analytic():
    spec = {"substitution": "GTR",
            "exchangeabilities": [[0, 2, 1, 0.5], [2, 0, 1, 1], [1, 1, 0, 2], [0.5, 1, 2, 0]],
            "frequencies": [0.2, 0.3, 0.25, 0.25],
            "rate_categories": {"gamma_shape": 0.7, "categories": 4}}
    model = compile_model(spec)
    length = 0.15
    frequencies = model["frequencies"]
    expected = math.fsum(
        weight * math.fsum(frequencies[i] * transition_matrix(model, length, rate)[i][0]
                           * transition_matrix(model, length, rate)[i][1] for i in range(4))
        for rate, weight in zip(model["rates"], model["weights"]))
    result = compare({"sequences": {"a": "A", "b": "C"},
                      "candidates": [{"id": "x", "tree": {"children": [
                          {"name": "a", "length": length}, {"name": "b", "length": length}]},
                          "model": spec}],
                      "resampling": {"replicates": 0}})
    assert math.exp(result["candidates"][0]["log_likelihood"]) == pytest.approx(expected, rel=1e-12)


def test_small_candidate_set_verifies_against_enumeration():
    sequences = {"a": "ACGT", "b": "AGGT", "c": "ACGA"}
    result = compare({
        "sequences": sequences,
        "candidates": [{"id": "t", "tree": {"children": [
            {"name": "a", "length": 0.1},
            {"length": 0.2, "children": [{"name": "b", "length": 0.1}, {"name": "c", "length": 0.2}]},
        ]}}],
        "resampling": {"replicates": 0},
    })
    verification = result["verification"][0]["partitions"][0]
    assert verification["status"] == "ok"
    assert verification["agrees"] is True
    assert verification["max_abs_log_deviation"] < 1e-10


def test_verification_matches_manual_ancestor_sum():
    tree = {"children": [{"name": "a", "length": 0.2},
                         {"length": 0.3, "children": [{"name": "b", "length": 0.4},
                                                      {"name": "c", "length": 0.5}]}]}

    def transition(i, j, t):
        return 0.25 + ((0.75 if i == j else -0.25) * math.exp(-4 * t / 3))

    expected = sum(0.25 * transition(root, 0, 0.2) * transition(root, ancestor, 0.3)
                   * transition(ancestor, 1, 0.4) * transition(ancestor, 2, 0.5)
                   for root, ancestor in itertools.product(range(4), repeat=2))
    result = compare({"sequences": {"a": "A", "b": "C", "c": "G"},
                      "candidates": [{"id": "t", "tree": tree}],
                      "resampling": {"replicates": 0}})
    assert math.exp(result["candidates"][0]["log_likelihood"]) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Observation population: partitions, ambiguity, missing data
# ---------------------------------------------------------------------------

def test_ambiguity_and_missing_data_keep_every_column():
    clean = {"a": "ACGT", "b": "AGGT"}
    ambiguous = {"a": "ACGN", "b": "AGGT"}
    tree = {"children": [{"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]}
    clean_result = compare({"sequences": clean, "candidates": [{"id": "t", "tree": tree}],
                            "resampling": {"replicates": 0}})
    ambiguous_result = compare({"sequences": ambiguous, "candidates": [{"id": "t", "tree": tree}],
                                "resampling": {"replicates": 0}})
    assert ambiguous_result["observation"]["sites"] == 4
    assert ambiguous_result["observation"]["ambiguity"]["fully_ambiguous_characters"] == 1
    # An N makes the column uninformative about the taxon that carries it; the
    # remaining taxon's prior probability (1/4) is what the column contributes.
    assert len(ambiguous_result["candidates"][0]["site_log_likelihoods"]) == 4
    assert ambiguous_result["candidates"][0]["site_log_likelihoods"][3] == pytest.approx(
        math.log(0.25), abs=1e-12)
    # And the ambiguous column is never dropped: totals use all four columns.
    assert ambiguous_result["candidates"][0]["log_likelihood"] != pytest.approx(
        clean_result["candidates"][0]["log_likelihood"], abs=1e-12)


def test_iupac_codes_admit_states():
    tree = {"children": [{"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]}
    # R admits A or G, so P(R, A) = P(A, A) + P(G, A) > P(A, A).
    r_site = compare({"sequences": {"a": "R", "b": "A"}, "candidates": [{"id": "t", "tree": tree}],
                      "resampling": {"replicates": 0}})
    c_site = compare({"sequences": {"a": "C", "b": "A"}, "candidates": [{"id": "t", "tree": tree}],
                      "resampling": {"replicates": 0}})
    a_site = compare({"sequences": {"a": "A", "b": "A"}, "candidates": [{"id": "t", "tree": tree}],
                      "resampling": {"replicates": 0}})
    assert r_site["candidates"][0]["log_likelihood"] > c_site["candidates"][0]["log_likelihood"]
    assert r_site["candidates"][0]["log_likelihood"] > a_site["candidates"][0]["log_likelihood"]


def test_partition_taxon_exclusion_marginalizes_but_keeps_columns():
    tree = {"children": [{"name": "a", "length": 0.1},
                         {"length": 0.1, "children": [{"name": "b", "length": 0.1},
                                                      {"name": "c", "length": 0.1}]}]}
    sequences = {"a": "ACGT", "b": "AGGT", "c": "AGGA"}
    result = compare({
        "sequences": sequences,
        "candidates": [{"id": "t", "tree": tree}],
        "partitions": [{"id": "ab", "sites": {"start": 0, "end": 2}, "taxa": ["a", "b"]},
                       {"id": "all", "sites": {"start": 2, "end": 4}, "taxa": ["a", "b", "c"]}],
        "resampling": {"replicates": 0},
    })
    assert result["observation"]["sites"] == 4
    assert sum(len(p["site_log_likelihoods"]) for p in result["candidates"][0]["partitions"]) == 4
    # Excluding c marginalizes it exactly, which is the same as observing c as
    # completely missing (N) while keeping the same tree topology.
    marginal = compare({"sequences": {"a": "AC", "b": "AG", "c": "NN"},
                        "candidates": [{"id": "t", "tree": tree}],
                        "resampling": {"replicates": 0}})
    assert result["candidates"][0]["partitions"][0]["log_likelihood"] == pytest.approx(
        marginal["candidates"][0]["log_likelihood"], abs=1e-12)


def test_every_site_must_be_assigned_exactly_once():
    payload = {"sequences": {"a": "ACGT", "b": "AGGT"},
               "candidates": [{"id": "t", "tree": {"children": [
                   {"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]}}],
               "partitions": [{"id": "p", "sites": [0, 1]}]}
    with pytest.raises(DomainError) as error:
        compare(payload)
    assert error.value.code == "partition_coverage"


def test_overlapping_partitions_rejected():
    payload = {"sequences": {"a": "ACGT", "b": "AGGT"},
               "candidates": [{"id": "t", "tree": {"children": [
                   {"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]}}],
               "partitions": [{"id": "p", "sites": [0, 1, 2, 3]},
                              {"id": "q", "sites": [3]}]}
    with pytest.raises(DomainError) as error:
        compare(payload)
    assert error.value.code == "partition_coverage"


def test_candidate_with_different_taxon_coverage_rejected():
    payload = {"sequences": {"a": "ACGT", "b": "AGGT", "c": "AGGA"},
               "candidates": [{"id": "t", "tree": {"children": [
                   {"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]}}]}
    with pytest.raises(DomainError) as error:
        compare(payload)
    assert error.value.code == "candidate_coverage"


# ---------------------------------------------------------------------------
# Zero probability versus no information
# ---------------------------------------------------------------------------

def test_true_zero_probability_is_distinct_from_no_information():
    tree = {"children": [{"name": "a"}, {"name": "b"}]}  # zero-length branches
    result = compare({"sequences": {"a": "A", "b": "C"},
                      "candidates": [{"id": "t", "tree": tree}],
                      "resampling": {"replicates": 0}})
    candidate = result["candidates"][0]
    assert candidate["status"] == "zero_probability"
    assert candidate["log_likelihood"] is None
    assert candidate["zero_probability_sites"] == [0]
    # The column was reported, not dropped.
    assert candidate["site_log_likelihoods"] == [None]


def test_zero_probability_candidate_loses_resamples_containing_the_column():
    forced = {"children": [{"name": "a", "length": 0.1},
                           {"length": 0.1, "children": [{"name": "b", "length": 0.0},
                                                        {"name": "c", "length": 0.0}]}]}
    free = {"children": [{"name": "a", "length": 0.1},
                         {"length": 0.1, "children": [{"name": "b", "length": 0.1},
                                                      {"name": "c", "length": 0.1}]}]}
    result = compare({"sequences": {"a": "AA", "b": "AC", "c": "AG"},
                      "candidates": [{"id": "forced", "tree": forced},
                                     {"id": "free", "tree": free}],
                      "resampling": {"replicates": 100, "seed": 3}})
    resampling = result["resampling"]
    # The zero is a genuine loss, never an excuse to drop the replicate.
    assert resampling["zero_probability_forced_replicates"]["forced"] > 0
    assert resampling["bootstrap_support_fraction"]["forced"] < 0.5
    assert resampling["comparable_replicates"] == resampling["replicates"]
    # The free candidate explains every column and is preferred.
    assert result["comparison"]["best"] == "free"


def test_uninformative_column_is_not_zero_probability():
    # An entirely missing column has log likelihood exactly zero.
    tree = {"children": [{"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]}
    result = compare({"sequences": {"a": "N", "b": "?"},
                      "candidates": [{"id": "t", "tree": tree}],
                      "resampling": {"replicates": 0}})
    candidate = result["candidates"][0]
    assert candidate["status"] == "ok"
    assert 0 in candidate["uninformative_sites"]
    assert candidate["zero_probability_sites"] == []


def test_uninformative_sites_use_global_column_indices():
    payload = {"sequences": {"a": "ACNN", "b": "AGNN"},
               "candidates": [{"id": "t", "tree": {"children": [
                   {"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]}}],
               "partitions": [{"id": "p1", "sites": [0, 1]}, {"id": "p2", "sites": [2, 3]}],
               "resampling": {"replicates": 0}}
    result = compare(payload)
    candidate = result["candidates"][0]
    assert candidate["uninformative_sites"] == [2, 3]
    second = candidate["partitions"][1]
    assert second["uninformative_sites"] == [2, 3]
    assert candidate["partitions"][0]["uninformative_sites"] == []


# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

def test_shared_parameter_must_agree():
    base = {"sequences": {"a": "ACGT", "b": "AGGT"},
            "resampling": {"replicates": 0},
            "shared_parameters": ["frequencies"]}
    gtr_a = {"substitution": "GTR",
             "exchangeabilities": [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
             "frequencies": [0.1, 0.2, 0.3, 0.4]}
    gtr_b = dict(gtr_a, frequencies=[0.4, 0.3, 0.2, 0.1])
    tree = {"children": [{"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]}
    payload = dict(base, candidates=[{"id": "a", "tree": tree, "model": gtr_a},
                                     {"id": "b", "tree": tree, "model": gtr_b}])
    with pytest.raises(DomainError) as error:
        compare(payload)
    assert error.value.code == "shared_parameter_violation"


def test_shared_parameter_accepted_when_identical():
    model = {"substitution": "GTR",
             "exchangeabilities": [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
             "frequencies": [0.25, 0.25, 0.25, 0.25]}
    tree = {"children": [{"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]}
    result = compare({"sequences": {"a": "ACGT", "b": "AGGT"},
                      "shared_parameters": ["substitution", "frequencies", "exchangeabilities"],
                      "resampling": {"replicates": 0},
                      "candidates": [{"id": "a", "tree": tree, "model": model},
                                     {"id": "b", "tree": tree, "model": model}]})
    assert result["comparison"]["status"] in ("indistinguishable", "distinguished")


# ---------------------------------------------------------------------------
# Comparison results and resampling interpretation
# ---------------------------------------------------------------------------

def test_deltas_are_relative_to_the_best_candidate():
    result = compare(json_payload())
    deltas = {item["id"]: item["delta_from_best"] for item in result["comparison"]["delta_from_best"]}
    best = result["comparison"]["best"]
    assert deltas[best] == 0.0
    assert all(value >= 0.0 for value in deltas.values())
    totals = {candidate["id"]: candidate["log_likelihood"] for candidate in result["candidates"]}
    for identifier, delta in deltas.items():
        assert delta == pytest.approx(totals[best] - totals[identifier], abs=1e-12)


def test_partition_deltas_sum_to_total_delta():
    result = compare(json_payload())
    best = result["comparison"]["best"]
    for item in result["comparison"]["delta_from_best"]:
        partition_sum = math.fsum(row["delta_from_best_by_candidate"][item["id"]]
                                  for row in result["comparison"]["partition_deltas"]
                                  if row["delta_from_best_by_candidate"][item["id"]] is not None)
        assert partition_sum == pytest.approx(item["delta_from_best"], abs=1e-9)


def test_per_site_deltas_sum_to_total_delta():
    result = compare(json_payload())
    best = result["comparison"]["best"]
    for item in result["comparison"]["delta_from_best"]:
        values = result["site_evidence"]["per_site_delta_from_best"][item["id"]]
        assert math.fsum(value for value in values if value is not None) == pytest.approx(
            item["delta_from_best"], abs=1e-9)
    assert result["site_evidence"]["per_site_delta_from_best"][best] == [0.0] * len(
        result["site_evidence"]["per_site_delta_from_best"][best])


def test_site_partition_map_covers_every_column():
    result = compare(json_payload())
    site_map = result["site_evidence"]["site_partition"]
    assert sorted(int(key) for key in site_map) == list(range(result["observation"]["sites"]))


def test_resampling_support_is_labelled_as_not_a_probability():
    result = compare(json_payload())
    resampling = result["resampling"]
    assert "not a posterior probability" in resampling["support_interpretation"]
    assert "not" in resampling["support_interpretation"]
    fraction = resampling["bootstrap_support_fraction"]
    assert math.fsum(fraction.values()) + resampling["tie_replicates"] / max(
        resampling["comparable_replicates"], 1) == pytest.approx(1.0, abs=1e-9)


def test_resampling_reproducible_with_seed():
    payload = json_payload()
    first = compare(payload)["resampling"]["bootstrap_support_fraction"]
    second = compare(payload)["resampling"]["bootstrap_support_fraction"]
    assert first == second
    payload["resampling"]["seed"] = 999
    third = compare(payload)["resampling"]["bootstrap_support_fraction"]
    assert third != first or payload["resampling"]["replicates"] == 0


def test_exact_resampling_reference_matches_monte_carlo():
    result = compare(json_payload())
    resampling = result["resampling"]
    assert "exact_reference" in resampling
    assert resampling["monte_carlo_check"]["status"] == "ok"


def test_round_off_does_not_decide_resampling_winners():
    # Two candidates whose per-site values differ only by floating-point noise
    # must be tied, not decided by summation order.  Resampling support has to
    # agree between the Monte Carlo and exact formulations.
    table = {"a": [-1.8665336922519162, -5.125073705622339, -1.8665336922519151, -5.125073705622338],
             "b": [-1.8665336922519162, -5.125073705622339, -1.8665336922519153, -4.065703956264499]}
    from omics_assay_foundry.phylogeny.resampling import (
        bootstrap, compare_bootstrap, exact_bootstrap_reference)
    exact = exact_bootstrap_reference(table, 4)
    assert exact["support_fraction"]["a"] == 0.0
    assert exact["tie_probability"] > 0.0
    monte_carlo = bootstrap(table, 4, 5000, seed=2)
    assert monte_carlo["bootstrap_support_fraction"]["a"] == 0.0
    assert compare_bootstrap(monte_carlo["bootstrap_support_fraction"], exact, 5000)["status"] == "ok"


def test_zero_replicates_is_an_explicit_state():
    payload = json_payload()
    payload["resampling"]["replicates"] = 0
    resampling = compare(payload)["resampling"]
    assert resampling["status"] == "no_replicates_requested"
    assert all(value is None for value in resampling["bootstrap_support_fraction"].values())


def test_budget_exhaustion_is_visible_and_not_a_zero():
    payload = json_payload()
    payload["budget"] = {"max_site_likelihoods": 1}
    result = compare(payload)
    assert all(candidate["status"] == "budget_exhausted" for candidate in result["candidates"])
    assert all(candidate["log_likelihood"] is None for candidate in result["candidates"])
    assert result["comparison"]["status"] == "no_comparable_candidate"
    assert any("budget" in caveat.lower() for caveat in result["caveats"])
    # Nothing is reported as zero probability just because work was skipped.
    assert all(candidate["zero_probability_sites"] == [] for candidate in result["candidates"])


def test_indistinguishable_candidates_are_reported():
    payload = {"sequences": {"a": "ACGT", "b": "AGGT"},
               "candidates": [{"id": "a", "tree": {"children": [
                   {"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]}},
                   {"id": "b", "tree": {"children": [
                       {"name": "b", "length": 0.1}, {"name": "a", "length": 0.1}]}}],
               "resampling": {"replicates": 0}}
    result = compare(payload)
    assert result["comparison"]["status"] == "indistinguishable"
    assert set(result["comparison"]["indistinguishable_candidates"]) == {"a", "b"}


def test_reordering_candidates_does_not_change_likelihoods():
    payload = json_payload()
    first = compare(payload)
    payload["candidates"].reverse()
    payload["resampling"]["seed"] = 5
    second = compare(payload)
    login = {c["id"]: c["log_likelihood"] for c in first["candidates"]}
    login2 = {c["id"]: c["log_likelihood"] for c in second["candidates"]}
    assert login == pytest.approx(login2, abs=1e-12)


def test_reordering_taxa_in_sequences_does_not_change_likelihoods():
    sequences = {"a": "ACGT", "b": "AGGT"}
    payload = {"candidates": [{"id": "t", "tree": {"children": [
        {"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]}}],
        "resampling": {"replicates": 0}}
    first = compare(dict(payload, sequences=sequences))
    second = compare(dict(payload, sequences={"b": "AGGT", "a": "ACGT"}))
    assert first["candidates"][0]["log_likelihood"] == pytest.approx(
        second["candidates"][0]["log_likelihood"], abs=1e-15)


# ---------------------------------------------------------------------------
# Validation and malformed model declarations
# ---------------------------------------------------------------------------

def test_unknown_options_rejected():
    payload = json_payload()
    payload["unknown_option"] = True
    with pytest.raises(DomainError):
        compare(payload)


@pytest.mark.parametrize("model", [
    {"substitution": "GTR"},
    {"substitution": "GTR", "exchangeabilities": [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]},
    {"substitution": "GTR", "exchangeabilities": [[0, 1, 1, 1], [2, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
     "frequencies": [0.25] * 4},
    {"frequencies": [0.25] * 4, "exchangeabilities": [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]},
    {"rate_categories": {"rates": [1.0, 2.0], "weights": [0.5, 0.5]}},
    {"rate_categories": {"gamma_shape": 0.0, "categories": 4}},
    {"rate_categories": {"gamma_shape": 0.5, "categories": 1, "rates": [1.0]}},
])
def test_invalid_models_rejected(model):
    payload = {"sequences": {"a": "ACGT", "b": "AGGT"},
               "candidates": [{"id": "t", "tree": {"children": [
                   {"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]},
                   "model": model}]}
    with pytest.raises(DomainError):
        compare(payload)


@pytest.mark.parametrize("spec", [
    {"substitution": "GTR", "exchangeabilities": [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
     "frequencies": [0.25] * 4},
    {"substitution": "GTR", "exchangeabilities": [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
     "frequencies": [0.25] * 4},
    {"substitution": "GTR", "exchangeabilities": [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]],
     "frequencies": [0.5, 0.5, 0.5, 0.5]},
    {"substitution": "JC69", "frequencies": [0.25] * 4},
    {"rate_categories": {"rates": [1, 1], "weights": [0.3, 0.3]}},
    {"rate_categories": {"rates": [1, 1], "weights": [1.0]}},
    {"rate_categories": {"rates": [1.0, 3.0], "weights": [0.5, 0.5]}},
])
def test_model_compilation_rejects_inconsistent_declarations(spec):
    with pytest.raises(DomainError):
        compile_model(spec)


def test_model_compilation_accepts_valid_gamma_and_explicit_rates():
    gamma = compile_model({"rate_categories": {"gamma_shape": 0.5, "categories": 4}})
    assert math.fsum(w * r for w, r in zip(gamma["weights"], gamma["rates"])) == pytest.approx(1.0)
    explicit = compile_model({"rate_categories": {"rates": [0.5, 1.5], "weights": [0.5, 0.5]}})
    assert explicit["rate_source"] == "explicit"
    assert math.fsum(w * r for w, r in zip(explicit["weights"], explicit["rates"])) == pytest.approx(1.0)


def test_duplicate_candidate_ids_rejected():
    tree = {"children": [{"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]}
    payload = {"sequences": {"a": "ACGT", "b": "AGGT"},
               "candidates": [{"id": "same", "tree": tree}, {"id": "same", "tree": tree}]}
    with pytest.raises(DomainError):
        compare(payload)


def test_sequence_character_validation():
    with pytest.raises(DomainError):
        compare({"sequences": {"a": "ACGT", "b": "AGGU"},
                 "candidates": [{"id": "t", "tree": {"children": [
                     {"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]}}]})
    with pytest.raises(DomainError):
        compare({"sequences": {"a": "acgt", "b": "ACGT"},
                 "candidates": [{"id": "t", "tree": {"children": [
                     {"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]}}]})


def test_candidate_name_is_an_alternative_identifier():
    tree = {"children": [{"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]}
    result = compare({"sequences": {"a": "ACGT", "b": "AGGT"},
                      "candidates": [{"name": "named", "tree": tree},
                                     {"tree": tree}],
                      "resampling": {"replicates": 0}})
    assert [candidate["id"] for candidate in result["candidates"]] == ["named", "candidate-2"]


def test_candidate_id_and_name_are_mutually_exclusive():
    with pytest.raises(DomainError):
        compare({"sequences": {"a": "ACGT", "b": "AGGT"},
                 "candidates": [{"id": "x", "name": "y", "tree": {"children": [
                     {"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]}}]})


def test_predictable_partition_spec_errors():
    tree = {"children": [{"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]}
    base = {"sequences": {"a": "ACGT", "b": "AGGT"},
            "candidates": [{"id": "t", "tree": tree}], "resampling": {"replicates": 0}}
    with pytest.raises(DomainError):
        compare({**base, "partitions": [{"id": "p", "sites": {"sites": [0, 1, 2, 3], "start": 0}}]})
    with pytest.raises(DomainError):
        compare({**base, "partitions": [{"id": "p", "sites": {"start": 0}}]})
    with pytest.raises(DomainError):
        compare({**base, "partitions": [{"id": "p", "sites": {"start": 2, "end": 2}}]})
    with pytest.raises(DomainError):
        compare({**base, "partitions": [{"id": "p", "sites": [0, 0, 1, 2, 3]}]})
    with pytest.raises(DomainError):
        compare({**base, "partitions": [{"id": "p", "sites": [0, 1, 2, 3],
                                         "taxa": ["a", "zzz"]}]})
    with pytest.raises(DomainError):
        compare({**base, "partitions": [{"id": "p", "sites": [0, 1, 2, 3], "taxa": ["a", "a"]}]})


def test_shared_parameter_unknown_name_rejected():
    tree = {"children": [{"name": "a", "length": 0.1}, {"name": "b", "length": 0.1}]}
    with pytest.raises(DomainError) as error:
        compare({"sequences": {"a": "ACGT", "b": "AGGT"},
                 "candidates": [{"id": "t", "tree": tree}],
                 "shared_parameters": ["branch_order"],
                 "resampling": {"replicates": 0}})
    assert error.value.code == "shared_parameter"


def test_candidate_tree_must_be_rooted_full_tree():
    with pytest.raises(DomainError):
        compare({"sequences": {"a": "ACGT"},
                 "candidates": [{"id": "t", "tree": {"name": "a"}}]})


def json_payload():
    return {
        "sequences": {"a": "ACGTACGT", "b": "AGGTACGT", "c": "ATGTACGA", "d": "AGGTACGA"},
        "candidates": [
            {"id": "ladder", "tree": TREE_LADDER},
            {"id": "balanced", "tree": TREE_BALANCED},
        ],
        "partitions": [
            {"id": "first", "sites": {"start": 0, "end": 4}},
            {"id": "second", "sites": {"start": 4, "end": 8}, "taxa": ["a", "b"]},
        ],
        "resampling": {"replicates": 200, "seed": 17},
    }
