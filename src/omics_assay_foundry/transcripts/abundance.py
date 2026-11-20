"""Result assembly for ``/transcripts``.

Two request shapes share this entry point:

* the legacy shape (``effective_lengths`` + ``fragments``) keeps the original
  fixed-length, unbiased-fragment model and its output fields;
* an extended shape adds fragment lengths, relative fragment bias weights,
  observation counts and optional deterministic resampling, and reports
  identifiability, interval usability and low/zero support separately.

Convergence, identifiability and interval usability are three distinct
statements.  A converged estimate says only that the iteration stopped moving;
it does not say every transcript is identified.  Members of an
indistinguishable group receive no individual interval, because a narrow
interval would describe the group's combined share, not the individual split.
Zero-support transcripts have no resampling distribution at all, so their
interval is left empty rather than reported as a degenerate ``[0, 0]``.
"""
import math

from . import model
from .compatibility import prepare as prepare_legacy
from .likelihood import estimate as estimate_legacy
from .profile import is_extended, prepare as prepare_extended
from ..errors import DomainError
from ..validation import integer, number

LEGACY_MODEL = "fixed-effective-length-unbiased-fragments"
WEIGHTED_MODEL = "fragment-length-and-bias-weighted"

def _tpm_interval(interval):
    return None if interval is None else [value*1e6 for value in interval]

def _legacy(payload):
    names, lengths, compatibility = prepare_legacy(payload)
    iterations = integer(payload.get("max_iterations", 1000), "max_iterations", minimum=1, maximum=5000)
    tolerance = number(payload.get("tolerance", 1e-10), "tolerance", minimum=1e-14, maximum=1e-2)
    if len(compatibility)*iterations > 5_000_000:
        raise DomainError("work_limit", "Fragment-iteration product exceeds five million")
    abundance, counts, used, converged, delta = estimate_legacy(
        lengths, compatibility, iterations, tolerance)
    signatures = {}
    for i, name in enumerate(names):
        signature = tuple(k for k, members in enumerate(compatibility) if i in members)
        signatures.setdefault(signature, []).append(name)
    unidentifiable = [members for present, members in signatures.items() if present and len(members) > 1]
    return {"model": LEGACY_MODEL, "effective_length_model": "fixed-effective-length-v1",
            "converged": converged, "iterations": used, "maximum_change": delta,
            "indistinguishable_groups": unidentifiable,
            "transcripts": [{"id": name, "molecule_fraction": abundance[i], "tpm": abundance[i]*1e6,
                             "expected_fragments": counts[i]} for i, name in enumerate(names)]}

def _status_for(index, supported, assignment, tpm, low_tpm):
    if index not in supported:
        return "zero_support"
    if index in assignment:
        return "indistinguishable"
    if tpm < low_tpm:
        return "low_abundance"
    return "identified"

def _comments(statuses, groups, converged, uncertainty):
    notes = []
    if any(flag == "low_abundance" for flag in statuses):
        notes.append("low_abundance: the estimate is below the declared TPM threshold; treat it "
                     "as a detection-scale value rather than a precise quantity.")
    if any(flag == "zero_support" for flag in statuses):
        notes.append("zero_support: no observed fragment supports this transcript; its abundance "
                     "is a model zero, not a measured absence.")
    if groups:
        notes.append("indistinguishable: these transcripts can only be quantified jointly; the "
                     "interval belongs to the group, not to its members.")
    if not converged:
        notes.append("not_converged: the iteration budget ran out before the estimate stabilized; "
                     "the reported abundances are provisional.")
    if uncertainty["status"] not in ("ok", "not_requested"):
        notes.append(f"uncertainty_{uncertainty['status']}: {uncertainty['reason']}")
    if not converged or uncertainty["status"] not in ("ok", "not_requested"):
        notes.append("usable_conclusions: point estimates may be inspected, but quantitative "
                     "downstream conclusions should wait for convergence and a usable interval.")
    return notes

def _extended(payload):
    spec = prepare_extended(payload)
    names = spec["names"]
    lengths = spec["effective_lengths"]
    yields = spec["effective_yields"]
    rows = spec["rows"]
    supported = spec["supported"]
    if len(rows)*spec["max_iterations"] > 5_000_000:
        raise DomainError("work_limit", "Fragment-iteration product exceeds five million")

    abundance, counts, used, converged, delta = model.estimate(
        yields, rows, spec["max_iterations"], spec["tolerance"])
    tpm = [value*1e6 for value in abundance]

    groups, assignment, coefficients = model.indistinguishable_groups(rows, supported)
    uncertainty, samples, group_samples = model.bootstrap(yields, rows, groups, coefficients, spec)
    intervals = [None]*len(names)
    group_intervals = [None]*len(groups)
    if uncertainty["usable"]:
        intervals = [_tpm_interval(model.interval(values)) for values in samples]
        group_intervals = [model.interval(values) for values in group_samples]

    low_tpm = spec["low_abundance_tpm"]
    statuses = [_status_for(i, supported, assignment, tpm[i], low_tpm) for i in range(len(names))]
    observed = [0]*len(names)
    for members, _, observations in rows:
        for i in members:
            observed[i] += observations

    transcripts = []
    for i, name in enumerate(names):
        status = statuses[i]
        report_interval = uncertainty["usable"] and status in ("identified", "low_abundance")
        transcripts.append({
            "id": name,
            "effective_length": lengths[i],
            "effective_yield": yields[i],
            "molecule_fraction": abundance[i],
            "tpm": tpm[i],
            "expected_fragments": counts[i],
            "observed_observations": observed[i],
            "support_status": status,
            "individually_identified": status in ("identified", "low_abundance"),
            "indistinguishable_group": assignment.get(i),
            "group_coefficient": coefficients.get(i),
            "tpm_interval": intervals[i] if report_interval else None,
            "interval_usable": report_interval,
        })

    group_details = []
    for group, members in enumerate(groups):
        split = math.fsum(abundance[i] for i in members)
        combination = math.fsum(coefficients[i]*abundance[i] for i in members)
        interval = group_intervals[group]
        group_details.append({
            "group": group,
            "members": [names[i] for i in members],
            "coefficients": [coefficients[i] for i in members],
            "individually_arbitrary_fraction": split,
            "individually_arbitrary_tpm": split*1e6,
            "identified_combination_fraction": combination,
            "identified_combination_tpm": combination*1e6,
            "combination_interval": interval,
            "combination_tpm_interval": _tpm_interval(interval),
            "interval_usable": uncertainty["usable"] and interval is not None,
            "note": "Observed support identifies only "
                    "sum(coefficient * fraction) for this group, never the split between "
                    "members. The individual fractions are one arbitrary decomposition. "
                    "Individual intervals are withheld because they would describe the "
                    "combination, not the member.",
        })

    summary = {
        "identified": sum(1 for value in statuses if value == "identified"),
        "low_abundance": sum(1 for value in statuses if value == "low_abundance"),
        "zero_support": sum(1 for value in statuses if value == "zero_support"),
        "indistinguishable_transcripts": sum(1 for value in statuses if value == "indistinguishable"),
        "indistinguishable_groups": len(groups),
    }
    return {
        "model": WEIGHTED_MODEL,
        "effective_length_model": spec["effective_length_model"],
        "normalization": "molecule fractions share one effective-length convention; "
                         "TPM = molecule_fraction * 1e6",
        "bias_model": "relative-fragment-weights" if spec["has_bias"] else "no-bias",
        "length_observation": ("per-fragment" if spec["has_fragment_lengths"]
                               else "distribution-only" if spec["has_distribution"]
                               else "not-observed"),
        "observation_model": "replicate-observation-counts",
        "converged": converged,
        "iterations": used,
        "maximum_change": delta,
        "convergence_meaning": "The iteration stopped moving; this does not imply individual "
                               "transcript identification.",
        "low_abundance_threshold_tpm": low_tpm,
        "total_observations": spec["total_observations"],
        "uncertainty": uncertainty,
        "indistinguishable_groups": [[names[i] for i in members] for members in groups],
        "group_details": group_details,
        "status_counts": summary,
        "comments": _comments(statuses, groups, converged, uncertainty),
        "usable_for_quantitative_conclusions": converged and uncertainty["status"] in ("ok", "not_requested"),
        "transcripts": transcripts,
    }

def process(payload):
    return _extended(payload) if is_extended(payload) else _legacy(payload)
