"""Assemble site and region evidence for the methylation endpoint."""

import math

from .calibration import Control, posterior_interval
from .controls import prepare
from .evidence import wilson
from ..validation import number

CALIBRATED_MODEL = "control-calibrated-homogeneous-binomial"
LEGACY_MODEL = "independent-homogeneous-binomial-zero-conversion-error"
NO_COVERAGE_INTERVAL = [0.0, 1.0]


def _no_coverage():
    return {"fraction": None, "interval": list(NO_COVERAGE_INTERVAL), "status": "no_coverage"}


def _site_cell(raw, control, z, level):
    cell = {"id": raw["id"], "region": raw["region"],
            "methylated": raw["methylated"], "coverage": raw["coverage"],
            "control": control.source() if control is not None else None}
    if raw["batch"] is not None:
        cell["batch"] = raw["batch"]
    if raw["strand"] is not None:
        cell["strand"] = raw["strand"]
    if control is None:
        cell["reason"] = raw["reason"] or "missing_control"
        cell.update({"fraction": None, "interval": list(NO_COVERAGE_INTERVAL),
                     "status": cell["reason"]})
        return cell
    if raw["coverage"] == 0:
        cell.update(_no_coverage())
        return cell
    reason = control.unresolved_reason(level)
    if reason is not None:
        cell.update({"fraction": None, "interval": list(NO_COVERAGE_INTERVAL), "status": reason})
        return cell
    median, interval, boundary = control.interval(raw["methylated"], raw["coverage"], z)
    cell.update({"fraction": median, "interval": interval,
                 "status": "near_boundary" if boundary else "ok"})
    return cell


def _region_cell(region, z, level):
    sites = region["sites"]
    methylated = sum(site["methylated"] for site in sites)
    coverage = sum(site["coverage"] for site in sites)
    entry = {"id": region["id"], "methylated": methylated, "coverage": coverage,
             "sites": len(sites)}
    missing = [site for site in sites if site["reason"] is not None]
    if missing:
        reasons = {site["reason"] for site in missing}
        entry.update({"fraction": None, "interval": list(NO_COVERAGE_INTERVAL),
                      "status": "missing_control" if "missing_control" in reasons else "ambiguous_control",
                      "control": None, "uncalibrated_sites": len(missing)})
        return entry
    if coverage == 0:
        entry.update(_no_coverage())
        entry["control"] = None
        return entry
    pools = {}
    for site in sites:
        control = site["control"]
        if control.id not in pools:
            pools[control.id] = [control, 0, 0]
        pools[control.id][1] += site["methylated"]
        pools[control.id][2] += site["coverage"]
    # Reported in registry order so the output does not depend on site order.
    ordered = [pools[key] for key in region["control_order"] if key in pools]
    entry["control"] = (ordered[0][0].source() if len(ordered) == 1
                        else [item[0].source() for item in ordered])
    unresolved = [item[0].unresolved_reason(level) for item in ordered]
    if any(reason is not None for reason in unresolved):
        status = next(reason for reason in unresolved if reason is not None)
        entry.update({"fraction": None, "interval": list(NO_COVERAGE_INTERVAL),
                      "status": status})
        return entry
    # One posterior per control: a control shared by several sites still
    # contributes its calibration uncertainty only once, and pooling the
    # counts of those sites keeps the coverage at the observed total.  A
    # region spanning many controls uses a coarser node grid so the cost stays
    # bounded; the per-control mixture is still integrated exactly once.
    if len(ordered) == 1:
        control, pooled_methylated, pooled_coverage = ordered[0]
        median, interval, boundary = control.interval(
            pooled_methylated, pooled_coverage, z)
    else:
        # The node grid trades quadrature accuracy for cost.  A handful of
        # controls gets the fine grid; a region spanning hundreds of controls
        # falls back to a single node each, whose mean preserves the posterior
        # location and whose spread was checked against brute-force integration.
        if len(ordered) <= 16:
            grid = 4
        elif len(ordered) <= 128:
            grid = 3
        else:
            grid = 1
        pooled = [(control.nodes(grid), pooled_methylated, pooled_coverage)
                  for control, pooled_methylated, pooled_coverage in ordered]
        median, interval, boundary = posterior_interval(pooled, z)
    entry.update({"fraction": median, "interval": interval,
                  "status": "near_boundary" if boundary else "ok"})
    return entry


def _legacy_result(payload):
    z = number(payload.get("confidence_z", 1.959963984540054), "confidence_z",
                minimum=0.1, maximum=10)
    parsed = prepare(payload)
    groups = {}
    sites = []
    for site in parsed["sites"]:
        sites.append({"id": site["id"], "region": site["region"],
                      "methylated": site["methylated"], "coverage": site["coverage"],
                      **wilson(site["methylated"], site["coverage"], z)})
        group = groups.setdefault(site["region"], [0, 0, 0])
        group[0] += site["methylated"]
        group[1] += site["coverage"]
        group[2] += 1
    regions = [{"id": name, "methylated": values[0], "coverage": values[1], "sites": values[2],
                **wilson(values[0], values[1], z)}
               for name, values in sorted(groups.items())]
    return {"model": LEGACY_MODEL, "interval": "Wilson score", "confidence_z": z,
            "sites": sites, "regions": regions}


def process(payload):
    if "controls" not in payload:
        return _legacy_result(payload)
    z = number(payload.get("confidence_z", 1.959963984540054), "confidence_z",
               minimum=0.1, maximum=10)
    level = math.erf(z / math.sqrt(2.0))
    parsed = prepare(payload)
    controls = [Control(record) for record in parsed["controls"]]
    registry = {control.id: control for control in controls}
    order = [control.id for control in controls]
    sites = []
    groups = {}
    for raw in parsed["sites"]:
        control = registry[raw["control"]["id"]] if raw["control"] is not None else None
        sites.append(_site_cell(raw, control, z, level))
        group = groups.setdefault(raw["region"], {"id": raw["region"], "sites": [],
                                                  "control_order": order})
        group["sites"].append({"methylated": raw["methylated"], "coverage": raw["coverage"],
                               "control": control, "reason": raw["reason"]})
    regions = [_region_cell(groups[name], z, level) for name in sorted(groups)]
    return {"model": CALIBRATED_MODEL,
            "interval": "control-calibrated posterior credible interval",
            "confidence_z": z, "controls": [control.source() for control in controls],
            "sites": sites, "regions": regions}
