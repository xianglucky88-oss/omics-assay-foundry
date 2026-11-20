"""Statistical checks for the control-calibrated methylation model.

The coverage tests simulate repeated measurement under a known conversion
setting: for each replicate they draw a site count from the true model and
control counts from the same setting, run the public dispatch path, and check
that the reported interval contains the true biological fraction.  This is the
property that a point-estimate-only calibration would silently lose.
"""

import math
import random

import pytest

from omics_assay_foundry.api import dispatch
from omics_assay_foundry.errors import DomainError


Z = 1.959963984540054


def control(control_id, batch, strand, methylated, methylated_coverage,
            unmethylated, unmethylated_coverage):
    return {"id": control_id, "batch": batch, "strand": strand,
            "fully_methylated": {"methylated": methylated, "coverage": methylated_coverage},
            "fully_unmethylated": {"methylated": unmethylated,
                                   "coverage": unmethylated_coverage}}


def calibrated(site, registry, **extra):
    payload = {"controls": registry, "sites": [site]}
    payload.update(extra)
    return dispatch("/methylation", payload)


def site(name, methylated, coverage, **extra):
    entry = {"id": name, "methylated": methylated, "coverage": coverage}
    entry.update(extra)
    return entry


CLEAN = control("c1", "lane", "+", 98, 100, 1, 100)


def test_calibration_moves_the_point_estimate_towards_zero():
    # A quarter of the reads are methylated, but background alone explains
    # more than that, so the biological fraction must drop towards zero.
    background = control("c", "b", "+", 100, 100, 30, 100)
    legacy = dispatch("/methylation", {"sites": [site("s", 5, 20)]})
    result = calibrated(site("s", 5, 20), [background])
    assert legacy["sites"][0]["fraction"] == pytest.approx(0.25)
    assert result["sites"][0]["fraction"] < 0.25
    assert result["model"] != legacy["model"]


def test_calibrated_interval_is_not_narrower_than_a_naive_rescaled_interval():
    # Rescaling the point estimate by (1 - c - u) would shrink the limits by
    # the same factor while keeping the zero-error width.  The calibrated
    # interval must also carry the control uncertainty, so it may not be
    # narrower than that rescaling.
    registry = [control("c", "b", "+", 20, 20, 1, 20)]
    legacy = dispatch("/methylation", {"sites": [site("s", 10, 20)]})["sites"][0]
    result = calibrated(site("s", 10, 20), registry)["sites"][0]
    c_hat = 0.0 / 20
    u_hat = 1.0 / 20
    factor = 1.0 - c_hat - u_hat
    naive_low = legacy["interval"][0] / factor
    naive_high = legacy["interval"][1] / factor
    naive_width = min(naive_high, 1.0) - max(naive_low, 0.0)
    width = result["interval"][1] - result["interval"][0]
    assert result["status"] in {"ok", "near_boundary"}
    assert width >= naive_width * 0.95


def test_deeper_controls_narrow_the_interval():
    shallow = calibrated(site("s", 10, 40), [control("c", "b", "+", 5, 10, 1, 10)])
    deep = calibrated(site("s", 10, 40), [control("c", "b", "+", 495, 1000, 1, 1000)])
    shallow_width = shallow["sites"][0]["interval"][1] - shallow["sites"][0]["interval"][0]
    deep_width = deep["sites"][0]["interval"][1] - deep["sites"][0]["interval"][0]
    assert deep_width < shallow_width


def test_control_posterior_reaches_the_reported_interval():
    # Shallow controls must widen the interval even though the site is unchanged.
    shallow = calibrated(site("s", 10, 40), [control("c", "b", "+", 1, 4, 1, 4)])
    wide = calibrated(site("s", 10, 40), [control("c", "b", "+", 0, 4, 4, 4)])
    assert shallow["sites"][0]["status"] in {"ok", "control_unresolved"}
    assert wide["sites"][0]["status"] != "ok"
    assert wide["sites"][0]["interval"] == pytest.approx([0.0, 1.0])


def test_shared_control_is_not_counted_as_independent_evidence():
    registry = [control("c", "b", "+", 95, 100, 1, 100)]
    spread = {"controls": registry,
              "sites": [site("a", 8, 10, region="r"), site("b", 4, 10, region="r")]}
    pooled = {"controls": registry, "sites": [site("ab", 12, 20, region="r")]}
    a = dispatch("/methylation", spread)["regions"][0]
    b = dispatch("/methylation", pooled)["regions"][0]
    assert a["coverage"] == 20 and b["coverage"] == 20
    assert a["fraction"] == pytest.approx(b["fraction"])
    assert a["interval"] == pytest.approx(b["interval"])


def test_region_interval_wider_than_each_constituent_site_is_expected():
    # Pooling coverage may only tighten; the calibration evidence is not
    # counted twice, so the region must not be narrower than a single pooled
    # site of the same total coverage.
    registry = [control("c", "b", "+", 95, 100, 1, 100)]
    spread = {"controls": registry,
              "sites": [site("a", 8, 10, region="r"), site("b", 4, 10, region="r")]}
    result = dispatch("/methylation", spread)
    region = result["regions"][0]
    assert region["coverage"] == 20
    assert region["interval"][1] - region["interval"][0] < 1.0


def test_pooled_region_does_not_gain_coverage_from_calibration():
    registry = [control("c", "b", "+", 95, 100, 1, 100)]
    sites = [site(f"s{i}", 2, 5, region="r") for i in range(4)]
    result = dispatch("/methylation", {"controls": registry, "sites": sites})
    region = result["regions"][0]
    assert region["coverage"] == 20
    assert region["methylated"] == 8


def test_control_provenance_is_reported_on_every_calibrated_cell():
    registry = [control("c1", "lane", "+", 95, 100, 1, 100),
                control("c2", "lane", "-", 96, 100, 2, 100)]
    result = dispatch("/methylation", {
        "controls": registry,
        "sites": [site("a", 10, 20, control="c1", region="r"),
                  site("b", 14, 20, control="c2", region="r")]})
    assert [item["id"] for item in result["controls"]] == ["c1", "c2"]
    assert result["sites"][0]["control"]["id"] == "c1"
    assert result["sites"][1]["control"]["id"] == "c2"
    assert result["sites"][0]["control"]["fully_methylated"] == {"methylated": 95, "coverage": 100}
    assert [item["id"] for item in result["regions"][0]["control"]] == ["c1", "c2"]


def test_site_can_select_control_by_batch_and_strand():
    registry = [control("f", "lane1", "+", 95, 100, 1, 100),
                control("r", "lane1", "-", 97, 100, 2, 100)]
    result = dispatch("/methylation", {
        "controls": registry,
        "sites": [site("a", 10, 20, batch="lane1", strand="+"),
                  site("b", 12, 20, batch="lane1", strand="-")]})
    assert [entry["control"]["id"] for entry in result["sites"]] == ["f", "r"]


def test_control_can_be_selected_by_its_own_id_only():
    registry = [control("c1", "lane1", "+", 95, 100, 1, 100)]
    result = dispatch("/methylation", {"controls": registry,
                                       "sites": [site("a", 10, 20, control="c1")]})
    assert result["sites"][0]["status"] == "ok"


def test_missing_strand_control_is_reported_not_rejected():
    registry = [control("c1", "lane1", "+", 95, 100, 1, 100)]
    result = dispatch("/methylation", {
        "controls": registry,
        "sites": [site("a", 10, 20, batch="lane1", strand="-", region="r")]})
    cell = result["sites"][0]
    assert cell["status"] in {"missing_control", "ambiguous_control"}
    assert cell["interval"] == pytest.approx([0.0, 1.0])
    region = result["regions"][0]
    assert region["status"] == cell["status"]
    assert region["uncalibrated_sites"] == 1


def test_ambiguous_control_selection_is_unidentifiable():
    registry = [control("c1", "lane1", "+", 95, 100, 1, 100),
                control("c2", "lane1", "+", 96, 100, 2, 100)]
    result = dispatch("/methylation", {"controls": registry,
                                       "sites": [site("a", 10, 20, batch="lane1")]})
    assert result["sites"][0]["status"] == "ambiguous_control"
    assert result["sites"][0]["control"] is None


def test_unknown_control_reference_is_rejected():
    with pytest.raises(DomainError) as error:
        calibrated(site("a", 10, 20, control="nope"), [CLEAN])
    assert error.value.code == "unknown_control"


def test_indistinguishable_controls_are_unidentifiable():
    # Both controls report the same fully methylated and fully unmethylated
    # rates, so conversion cannot be separated from background.
    contradictory = control("c", "b", "+", 0, 8, 8, 8)
    result = calibrated(site("s", 5, 10), [contradictory])
    assert result["sites"][0]["status"] == "control_unresolved"
    assert result["sites"][0]["interval"] == pytest.approx([0.0, 1.0])
    assert result["sites"][0]["control"]["id"] == "c"


def test_no_control_coverage_is_its_own_status():
    empty = control("c", "b", "+", 0, 0, 0, 0)
    result = calibrated(site("s", 5, 10), [empty])
    assert result["sites"][0]["status"] == "no_control_coverage"
    assert result["sites"][0]["interval"] == pytest.approx([0.0, 1.0])


def test_low_control_coverage_widens_the_interval_but_keeps_a_finite_bound():
    thin = control("c", "b", "+", 7, 10, 1, 10)
    deep = control("c", "b", "+", 99, 100, 1, 100)
    narrow = calibrated(site("s", 5, 20), [deep])["sites"][0]
    cell = calibrated(site("s", 5, 20), [thin])["sites"][0]
    assert cell["fraction"] is not None and cell["interval"] is not None
    assert cell["status"] in {"ok", "near_boundary"}
    thin_width = cell["interval"][1] - cell["interval"][0]
    deep_width = narrow["interval"][1] - narrow["interval"][0]
    assert thin_width > deep_width


def test_site_without_coverage_stays_no_coverage_under_controls():
    result = calibrated(site("s", 0, 0), [CLEAN])
    assert result["sites"][0]["status"] == "no_coverage"
    assert result["sites"][0]["fraction"] is None


def test_all_unmethylated_site_reaches_the_lower_boundary():
    result = calibrated(site("s", 0, 20), [CLEAN])
    cell = result["sites"][0]
    assert cell["status"] == "near_boundary"
    assert cell["interval"][0] == 0.0


def test_all_methylated_site_reaches_the_upper_boundary():
    result = calibrated(site("s", 20, 20), [CLEAN])
    cell = result["sites"][0]
    assert cell["status"] == "near_boundary"
    assert cell["interval"][1] == 1.0


def test_region_with_sites_from_two_controls_uses_both_but_once_each():
    registry = [control("c1", "lane1", "+", 95, 100, 1, 100),
                control("c2", "lane1", "-", 95, 100, 1, 100)]
    once = {"controls": [registry[0]],
            "sites": [site("a", 10, 20, control="c1", region="r")]}
    twice = {"controls": registry,
             "sites": [site("a", 10, 20, control="c1", region="r"),
                       site("b", 10, 20, control="c2", region="r")]}
    single = dispatch("/methylation", once)["regions"][0]
    paired = dispatch("/methylation", twice)["regions"][0]
    assert paired["coverage"] == 40
    assert len(paired["control"]) == 2
    single_width = single["interval"][1] - single["interval"][0]
    paired_width = paired["interval"][1] - paired["interval"][0]
    assert paired_width < single_width


def test_region_across_controls_is_not_narrower_than_pooled_counts():
    registry = [control("c1", "lane1", "+", 95, 100, 1, 100),
                control("c2", "lane1", "-", 95, 100, 1, 100)]
    result = dispatch("/methylation", {
        "controls": registry,
        "sites": [site("a", 10, 20, control="c1", region="r"),
                  site("b", 10, 20, control="c2", region="r")]})
    region = result["regions"][0]
    assert region["coverage"] == 40
    assert 0.0 < region["interval"][0] < region["fraction"] < region["interval"][1] < 1.0


def test_legacy_requests_keep_the_original_model_and_numbers():
    payload = {"sites": [{"id": "s", "methylated": 5, "coverage": 10, "region": "r"}]}
    result = dispatch("/methylation", payload)
    assert result["model"] == "independent-homogeneous-binomial-zero-conversion-error"
    assert result["interval"] == "Wilson score"
    assert result["sites"][0]["interval"] == pytest.approx([0.236593090512564,
                                                            0.763406909487436])
    assert "controls" not in result
    assert result["regions"][0]["interval"] == pytest.approx([0.236593090512564,
                                                              0.763406909487436])


def test_legacy_request_cannot_smuggle_control_fields():
    with pytest.raises(DomainError):
        dispatch("/methylation", {"sites": [site("a", 1, 10, control="c1")]})


def test_nonzero_conversion_error_is_still_rejected():
    with pytest.raises(DomainError) as error:
        dispatch("/methylation", {"sites": [site("a", 1, 10)], "conversion_error": 0.1})
    assert error.value.code == "unsupported_model"


def test_zero_conversion_error_is_accepted_alongside_controls():
    result = calibrated(site("s", 5, 10), [CLEAN], conversion_error=0)
    assert result["sites"][0]["status"] == "ok"


def test_region_result_is_order_independent():
    registry = [control("c1", "b1", "+", 97, 100, 1, 100),
                control("c2", "b2", "-", 94, 100, 2, 100)]
    sites = [site("a", 6, 10, control="c1", region="r"),
             site("b", 9, 10, control="c2", region="r"),
             site("c", 5, 10, control="c1", region="r")]
    forward = dispatch("/methylation", {"controls": registry, "sites": list(sites)})
    backward = dispatch("/methylation", {"controls": registry + [], "sites": list(reversed(sites))})
    region_forward = next(item for item in forward["regions"] if item["id"] == "r")
    region_backward = next(item for item in backward["regions"] if item["id"] == "r")
    assert region_forward["fraction"] == pytest.approx(region_backward["fraction"])
    assert region_forward["interval"] == pytest.approx(region_backward["interval"])
    assert [item["id"] for item in region_forward["control"]] == ["c1", "c2"]


def test_control_registry_accepts_batch_then_strand_mapping():
    result = dispatch("/methylation", {
        "controls": {"lane": {"+": {"fully_methylated": {"methylated": 98, "coverage": 100},
                                    "fully_unmethylated": {"methylated": 1, "coverage": 100}}}},
        "sites": [site("a", 10, 20, batch="lane", strand="+")]})
    assert result["controls"][0]["batch"] == "lane"
    assert result["sites"][0]["control"]["id"] == "lane:+"
    assert result["sites"][0]["status"] == "ok"


def test_control_counts_may_be_given_as_unmethylated_reads():
    registry = [{"id": "c", "batch": "b", "strand": "+",
                 "fully_methylated": {"methylated": 2, "unmethylated": 98},
                 "fully_unmethylated": {"unmethylated": 99, "methylated": 1}}]
    result = calibrated(site("s", 10, 20), registry)
    assert result["controls"][0]["fully_methylated"] == {"methylated": 2, "coverage": 100}


def test_control_is_not_written_back_to_any_file(tmp_path):
    before = sorted(tmp_path.iterdir())
    calibrated(site("s", 10, 20), [CLEAN])
    assert sorted(tmp_path.iterdir()) == before


def _simulate(c, u, p, coverage, control_coverage, replicates, seed):
    """Draw repeated measurements and count interval coverage of the truth."""
    generator = random.Random(seed)
    observed = u + p * (1.0 - c - u)
    covered = 0
    for _ in range(replicates):
        methylated = sum(generator.random() < observed for _ in range(coverage))
        methylated_control = sum(generator.random() < (1.0 - c)
                                 for _ in range(control_coverage))
        unmethylated_control = sum(generator.random() < (1.0 - u)
                                   for _ in range(control_coverage))
        registry = [control("c", "b", "+", methylated_control, control_coverage,
                            control_coverage - unmethylated_control, control_coverage)]
        result = calibrated(site("s", methylated, coverage), registry)
        interval = result["sites"][0]["interval"]
        if interval[0] <= p <= interval[1]:
            covered += 1
    return covered / replicates


def test_repeated_measurement_coverage_is_at_least_nominal():
    # Known conversion setting, repeated draws: the calibrated interval must
    # cover the true fraction near its nominal rate rather than only matching
    # a de-biased point estimate.
    rate = _simulate(0.05, 0.01, 0.5, 20, 100, 300, seed=20261119)
    assert 0.90 <= rate <= 0.995


def test_repeated_measurement_coverage_holds_near_the_upper_boundary():
    rate = _simulate(0.05, 0.01, 0.95, 20, 100, 300, seed=20261120)
    assert rate >= 0.90


def test_small_true_fraction_coverage_does_not_collapse_at_the_boundary():
    rate = _simulate(0.05, 0.01, 0.01, 20, 100, 300, seed=20261121)
    assert rate >= 0.90


def test_legacy_interval_misses_when_conversion_is_present():
    # The same simulated setting shows why the zero-error Wilson interval is
    # not a safe substitute: it under-covers once the truth is near 1.
    generator = random.Random(20261122)
    c, u, p, coverage = 0.05, 0.01, 0.95, 20
    observed = u + p * (1.0 - c - u)
    covered = 0
    replicates = 400
    for _ in range(replicates):
        methylated = sum(generator.random() < observed for _ in range(coverage))
        result = dispatch("/methylation", {"sites": [site("s", methylated, coverage)]})
        interval = result["sites"][0]["interval"]
        if interval[0] <= p <= interval[1]:
            covered += 1
    assert covered / replicates < 0.85


def test_deep_control_coverage_stays_finite_and_ordered():
    result = calibrated(site("s", 500, 1000),
                        [control("c", "b", "+", 999999, 1000000, 10, 1000000)])
    cell = result["sites"][0]
    assert cell["status"] == "ok"
    assert 0.0 < cell["interval"][0] <= cell["fraction"] <= cell["interval"][1] < 1.0


def test_control_field_validation_rejects_bad_shapes():
    with pytest.raises(DomainError):
        calibrated(site("s", 10, 20), [{"id": "c", "batch": "b", "strand": "+",
                                        "fully_methylated": {"methylated": 5, "coverage": 2},
                                        "fully_unmethylated": {"methylated": 1, "coverage": 2}}])


def test_sites_with_shared_control_are_independent_binomial_pools_not_shared_calibration():
    # Two sites on the same control: the regional interval must be at least as
    # wide as one site alone but the control must not be counted twice.  The
    # exact check is that the region matches a single pooled binomial count.
    registry = [control("c", "b", "+", 90, 100, 2, 100)]
    split = dispatch("/methylation", {
        "controls": registry,
        "sites": [site("a", 7, 10, region="r"), site("b", 3, 10, region="r")]})
    pooled = dispatch("/methylation", {
        "controls": registry,
        "sites": [site("ab", 10, 20, region="r")]})
    assert split["regions"][0]["fraction"] == pytest.approx(pooled["regions"][0]["fraction"])
    assert split["regions"][0]["interval"] == pytest.approx(pooled["regions"][0]["interval"])

def test_documented_control_example_produces_a_calibrated_result(sample):
    payload = sample("methylation_controls")
    result = dispatch("/methylation", payload)
    assert result["model"] == "control-calibrated-homogeneous-binomial"
    assert len(result["controls"]) == 2
    for cell in result["sites"]:
        assert cell["control"]["id"] in {"lane1-fwd", "lane1-rev"}
        assert cell["interval"][0] <= cell["interval"][1]
    assert {region["id"] for region in result["regions"]} == {"r1", "r2"}
