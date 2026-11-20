"""Contact-enrichment tests against small declared-generation data.

Generative conditions are declared here. Null counts follow
Poisson(mu_ij) with mu_ij = e_d * x_i * x_j over pairs at distances 0..3,
where the visibility vector x solves A x = 1 for the distance kernel
A[i][j] = e_|i-j| (including the diagonal self-pair scale). Then the
expected observed marginals equal x exactly, so the generated null lies
inside the fitted model family and only Poisson estimation error remains.
"""
import math
import random

import pytest

from omics_assay_foundry.api import dispatch
from omics_assay_foundry.errors import DomainError

BINS = 12
STRATUM_SCALE = {0: 2.0, 1: 1.0, 2: 0.6, 3: 0.35}
PAIRS = [(i, j) for i in range(BINS) for j in range(i, BINS) if j-i <= 3]
STRATA = [{"min_distance": 0, "max_distance": 0},
          {"min_distance": 1, "max_distance": 1},
          {"min_distance": 2, "max_distance": 2},
          {"min_distance": 3, "max_distance": 3}]


def solve_kernel(scales):
    """Solve A x = 1 for the distance kernel by Gaussian elimination."""
    n = BINS
    system = []
    for i in range(n):
        row = [scales[abs(i-j)] if abs(i-j) <= 3 else 0.0 for j in range(n)]
        system.append(row + [1.0])
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(system[r][col]))
        system[col], system[pivot] = system[pivot], system[col]
        for r in range(n):
            if r != col:
                factor = system[r][col]/system[col][col]
                for c in range(col, n+1):
                    system[r][c] -= factor*system[col][c]
    return [system[i][n]/system[i][i] for i in range(n)]


VISIBILITY = solve_kernel(STRATUM_SCALE)
assert all(x > 0 for x in VISIBILITY)
_STRETCH = 10.0/min(STRATUM_SCALE[j-i]*VISIBILITY[i]*VISIBILITY[j] for i, j in PAIRS)
VISIBILITY = [x*math.sqrt(_STRETCH) for x in VISIBILITY]
NULL_MU = {(i, j): STRATUM_SCALE[j-i]*VISIBILITY[i]*VISIBILITY[j] for i, j in PAIRS}
NULL_COUNTS = {pair: round(mu) for pair, mu in NULL_MU.items()}


def declared_request(counts, **overrides):
    request = {"bins": BINS,
               "entries": [{"i": i, "j": j, "count": counts[i, j]} for i, j in PAIRS],
               "strata": [dict(s) for s in STRATA],
               "relations": [{"i": i, "j": j} for i, j in PAIRS]}
    request.update(overrides)
    return request


def model_expectations(counts):
    """Independent recomputation of the declared leave-one-out background."""
    marginal = [0]*BINS
    for (i, j), k in counts.items():
        marginal[i] += k
        if j != i:
            marginal[j] += k
    fitted = {}
    for distance in STRATUM_SCALE:
        pairs = [(i, j) for (i, j) in counts if j-i == distance]
        total = sum(counts[pair] for pair in pairs)
        weight = sum(marginal[i]*marginal[j] for i, j in pairs)
        for i, j in pairs:
            leave_total = total - counts[i, j]
            leave_weight = weight - marginal[i]*marginal[j]
            fitted[i, j] = leave_total/leave_weight*marginal[i]*marginal[j]
    return fitted


def test_expectations_match_declared_model_formula():
    result = dispatch("/contact-enrichment", declared_request(NULL_COUNTS))
    assert result["correction"]["converged"]
    assert result["complete"]
    assert [s["estimable"] for s in result["background"]["strata"]] == [True]*4
    assert [s["pairs"] for s in result["background"]["strata"]] == [12, 11, 10, 9]
    fitted = model_expectations(NULL_COUNTS)
    for row in result["relations"]:
        assert row["status"] == "ok"
        mu = fitted[row["i"], row["j"]]
        assert row["expected"] == pytest.approx(mu, rel=1e-9)
        k = NULL_COUNTS[row["i"], row["j"]]
        assert row["effect_log2"] == pytest.approx(math.log2(k/mu), rel=1e-9)


def test_background_recovers_declared_generation():
    result = dispatch("/contact-enrichment", declared_request(NULL_COUNTS))
    for row in result["relations"]:
        # Rounded null counts: fitted expectations stay near the declared mu
        # and no relation is called.
        assert row["expected"] == pytest.approx(NULL_MU[row["i"], row["j"]], rel=0.15)
        assert row["effect_log2"] == pytest.approx(0, abs=0.2)
        assert row["significant"] is None


def test_injected_fold_change_is_enriched():
    target = min(PAIRS, key=lambda pair: NULL_MU[pair])
    counts = dict(NULL_COUNTS)
    counts[target] = round(NULL_MU[target]*100)
    result = dispatch("/contact-enrichment", declared_request(counts))
    row = next(r for r in result["relations"] if (r["i"], r["j"]) == target)
    assert row["significant"] == "enriched"
    assert row["q_enrichment"] <= 0.05
    # Marginal backgrounds attenuate injected folds; the direction and a
    # strongly positive effect must survive.
    assert row["effect_log2"] > 1.5
    assert row["effect_interval"][0] > 0


def poisson(rng, mu):
    """Exact Knuth sampler; the declared means all stay below 40."""
    assert 0 < mu < 1000
    threshold = math.exp(-mu)
    k, product = 0, 1.0
    while product > threshold:
        k += 1
        product *= rng.random()
    return k-1


def test_false_positive_rate_is_controlled_under_null():
    significant_calls = 0
    for seed in range(8):
        rng = random.Random(20261120 + seed)
        counts = {pair: poisson(rng, NULL_MU[pair]) for pair in PAIRS}
        result = dispatch("/contact-enrichment", declared_request(counts))
        assert result["error_control"]["family_size"] == len(PAIRS)
        for row in result["relations"]:
            assert row["status"] == "ok"
            if row["significant"] is not None:
                significant_calls += 1
    # Benjamini-Hochberg at alpha 0.05 admits a false discovery in a seed
    # with probability at most 0.05; across 8 seeds and 240 null relations
    # the declared allowance is far above the expected number of calls.
    assert significant_calls <= 2


def test_masked_unobserved_and_zero_pairs_differ_in_population():
    request = {
        "bins": 6,
        "valid_bins": [0, 1, 2, 3, 4],
        "entries": [{"i": 0, "j": 1, "count": 10}, {"i": 1, "j": 2, "count": 10},
                    {"i": 2, "j": 3, "count": 10}, {"i": 3, "j": 4, "count": 10},
                    {"i": 0, "j": 2, "count": 8}, {"i": 1, "j": 3, "count": 8},
                    {"i": 0, "j": 3, "count": 0}, {"i": 1, "j": 4, "count": 6}],
        "strata": [{"min_distance": 1, "max_distance": 1},
                   {"min_distance": 2, "max_distance": 2},
                   {"min_distance": 3, "max_distance": 3}],
        "relations": [{"i": 0, "j": 3}, {"i": 1, "j": 5}, {"i": 2, "j": 4}, {"i": 0, "j": 2}],
        "min_stratum_pairs": 2,
        "min_stratum_count": 1,
    }
    result = dispatch("/contact-enrichment", request)
    strata = result["background"]["strata"]
    # The listed zero (0, 3) joins its stratum population as a true zero.
    assert strata[2]["pairs"] == 2 and strata[2]["total_count"] == 6
    # The unlisted pair (2, 4) never joins: stratum 2 keeps only listed pairs.
    assert strata[1]["pairs"] == 2 and strata[1]["total_count"] == 16
    rows = {(r["i"], r["j"]): r for r in result["relations"]}
    zero = rows[0, 3]
    assert zero["status"] == "ok" and zero["observed"] == 0
    assert zero["effect_log2"] is None and zero["effect_interval"][0] is None
    assert zero["p_depletion"] < 0.05
    assert rows[1, 5]["status"] == "masked_bin"
    assert rows[2, 4]["status"] == "unobserved" and rows[2, 4]["observed"] is None
    assert rows[0, 2]["status"] == "ok"


def test_unsupported_and_disconnected_support_stay_inestimable():
    request = {
        "bins": 8,
        "entries": [{"i": 0, "j": 1, "count": 10}, {"i": 1, "j": 2, "count": 10},
                    {"i": 0, "j": 2, "count": 6}, {"i": 4, "j": 5, "count": 12},
                    {"i": 2, "j": 3, "count": 0}, {"i": 0, "j": 4, "count": 0}],
        "strata": [{"min_distance": 1, "max_distance": 2}],
        "relations": [{"i": 2, "j": 3}, {"i": 0, "j": 4}, {"i": 4, "j": 5}, {"i": 0, "j": 1}],
    }
    result = dispatch("/contact-enrichment", request)
    rows = {(r["i"], r["j"]): r for r in result["relations"]}
    # Bin 3 has only a listed zero: the correction has no solution for it.
    assert rows[2, 3]["status"] == "unsupported_bin"
    # (0, 4) is a listed zero across disconnected positive support.
    assert rows[0, 4]["status"] == "disconnected_support"
    # (4, 5) is the only count of its component: no independent support.
    assert rows[4, 5]["status"] == "unsupported_bin"
    for pair in [(2, 3), (0, 4), (4, 5)]:
        row = rows[pair]
        # No pseudocount fills in an expectation or a significance call.
        assert row["expected"] is None and row["significant"] is None
        assert row["p_enrichment"] is None and row["q_enrichment"] is None
    assert rows[0, 1]["status"] == "ok"


def test_unconverged_correction_blocks_significance():
    request = declared_request(NULL_COUNTS, max_iterations=1, tolerance=1e-12)
    result = dispatch("/contact-enrichment", request)
    assert not result["correction"]["converged"]
    assert all(r["status"] == "correction_not_converged" for r in result["relations"])
    assert all(r["significant"] is None for r in result["relations"])
    assert result["error_control"]["family_size"] == 0


def test_unbalanceable_support_reports_no_solution():
    # A 3-bin path has no unit-marginal balancing: the correction diverges.
    request = {
        "bins": 6,
        "entries": [{"i": 0, "j": 1, "count": 5}, {"i": 1, "j": 2, "count": 7},
                    {"i": 3, "j": 4, "count": 9}, {"i": 4, "j": 5, "count": 11},
                    {"i": 3, "j": 5, "count": 6}],
        "strata": [{"min_distance": 1, "max_distance": 2}],
        "relations": [{"i": 0, "j": 1}, {"i": 1, "j": 2}, {"i": 3, "j": 4}],
        "min_stratum_pairs": 1,
        "min_stratum_count": 1,
        "max_iterations": 10000,
    }
    result = dispatch("/contact-enrichment", request)
    correction = result["correction"]
    assert not correction["converged"] and correction["no_solution_bins"] == 3
    rows = {(r["i"], r["j"]): r for r in result["relations"]}
    assert rows[0, 1]["status"] == "correction_no_solution"
    assert rows[1, 2]["status"] == "correction_no_solution"
    assert rows[0, 1]["significant"] is None and rows[0, 1]["expected"] is None
    # The triangle component in the same request is unaffected.
    assert rows[3, 4]["status"] == "ok"


def test_thin_stratum_evidence_blocks_significance():
    result = dispatch("/contact-enrichment", declared_request(NULL_COUNTS, min_stratum_pairs=13))
    assert [s["estimable"] for s in result["background"]["strata"]] == [False]*4
    assert all(r["status"] == "insufficient_stratum_evidence" for r in result["relations"])
    low = dispatch("/contact-enrichment",
                   declared_request({pair: 1 for pair in PAIRS}, min_stratum_count=100))
    assert all(r["status"] == "insufficient_stratum_evidence" for r in low["relations"])


def test_relation_budget_marks_analysis_incomplete():
    result = dispatch("/contact-enrichment", declared_request(NULL_COUNTS, max_relations=5))
    assert not result["complete"]
    computed, deferred = result["relations"][:5], result["relations"][5:]
    assert all(r["status"] == "ok" for r in computed)
    assert all(r["status"] == "not_computed" for r in deferred)
    assert all(r["significant"] is None and r["expected"] is None for r in deferred)
    assert result["error_control"]["family_size"] == 5


def test_documented_example_statuses(sample):
    result = dispatch("/contact-enrichment", sample("contact-enrichment"))
    rows = {(r["i"], r["j"]): r for r in result["relations"]}
    assert rows[0, 3]["significant"] == "enriched"
    assert rows[2, 7]["status"] == "masked_bin"
    assert rows[1, 5]["status"] == "unobserved"
    assert rows[0, 4]["status"] == "no_background_stratum"
    assert rows[0, 4]["observed"] == 0


def test_balance_endpoint_is_unchanged():
    result = dispatch("/contacts", {"matrix": [[4, 1], [1, 9]]})
    assert result["converged"]
    assert result["weights"] == pytest.approx([math.sqrt(3/14), math.sqrt(2/21)], rel=1e-7)


def test_poisson_tails_match_direct_summation():
    from omics_assay_foundry.contacts.poisson import poisson_tails
    for mu in [0.5, 4.0, 17.3, 100.0, 1000.0]:
        for k in [0, 1, 2, 5, 9, 40, 120]:
            p_enrichment, p_depletion = poisson_tails(k, mu)
            reference_le = math.fsum(
                math.exp(-mu + x*math.log(mu) - math.lgamma(x+1)) for x in range(k+1))
            reference_ge = 1.0 - (reference_le - math.exp(-mu + k*math.log(mu) - math.lgamma(k+1)))
            assert p_depletion == pytest.approx(reference_le, abs=1e-12)
            assert p_enrichment == pytest.approx(reference_ge, abs=1e-12)


def test_rate_interval_matches_known_garwood_values():
    from omics_assay_foundry.contacts.poisson import rate_interval
    lower, upper = rate_interval(5, 0.05)
    assert (lower, upper) == pytest.approx([1.623486, 11.668332], abs=1e-5)
    # Central interval: the zero-count upper bound is 0.5*chi2_{0.975}(2).
    lower, upper = rate_interval(0, 0.05)
    assert lower == 0.0 and upper == pytest.approx(3.688879, abs=1e-5)


def test_benjamini_hochberg_step_up():
    from omics_assay_foundry.contacts.poisson import benjamini_hochberg
    q_values = benjamini_hochberg([(0, 0.001), (1, 0.01), (2, 0.5)])
    assert q_values == pytest.approx({0: 0.003, 1: 0.015, 2: 0.5})


@pytest.mark.parametrize("mutation,code", [
    (lambda p: p["entries"].append(dict(p["entries"][0])), "duplicate_entry"),
    (lambda p: p["entries"].append({"i": 2, "j": 1, "count": 1}), "pair_order"),
    (lambda p: p["entries"].append({"i": 0, "j": 11, "count": 1, "x": 1}), "unsupported_field"),
    (lambda p: p.update(valid_bins=[0, 1]), "masked_entry"),
    (lambda p: p.update(strata=[{"min_distance": 1, "max_distance": 2},
                                {"min_distance": 2, "max_distance": 3}]), "overlapping_strata"),
    (lambda p: p.update(strata=[{"min_distance": 3, "max_distance": 1}]), "stratum_order"),
    (lambda p: p["relations"].append(dict(p["relations"][0])), "duplicate_relation"),
    (lambda p: p["entries"][0].update(count=-1), "invalid_integer"),
    (lambda p: p["entries"][0].update(count=1.5), "invalid_integer"),
    (lambda p: p.update(max_iterations=0), "invalid_integer"),
])
def test_invalid_requests_are_rejected(mutation, code):
    request = declared_request(NULL_COUNTS)
    mutation(request)
    with pytest.raises(DomainError) as error:
        dispatch("/contact-enrichment", request)
    assert error.value.code == code
