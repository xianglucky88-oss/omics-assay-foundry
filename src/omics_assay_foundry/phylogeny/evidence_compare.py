"""Candidate-tree evidence comparison for a fixed multi-sequence observation.

This endpoint compares a *finite* set of supplied candidates under one explicit
observation population and one explicit scale.  It never searches for trees and
never re-estimates branch lengths.

Key guarantees:

* Every candidate must cover exactly the observed taxa.  A candidate that omitted
  a taxon would evaluate fewer columns and could look unfairly good, so
  differing coverage is rejected rather than silently accommodated.
* Every observed column belongs to exactly one partition.  A partition may
  observe only a subset of taxa; an excluded taxon is marginalized (its
  conditional likelihood vector is all ones), so the column is still evaluated
  and cannot be dropped.
* Ambiguous IUPAC codes and missing data widen the leaf state set instead of
  removing the column.
* All likelihoods lie on one scale: branch lengths are expected substitutions
  per site and every substitution model is normalized to an expected rate of
  one.  A declared shared parameter that differs between effective models is an
  error rather than a hidden adjustment.
* A column with probability exactly zero and a column contributing no
  information (log likelihood exactly zero) are reported separately.  The zero
  is a genuine scientific statement and makes the candidate lose any resampling
  replicate that contains the column; it is never converted into a dropped
  column or an excused replicate.
"""
import math

from .partitioned_likelihood import ModelRun, build_nodes
from .partitions import effective_models, parse_comparison
from .reference import (MAX_ENUMERATIONS, compare_site_values, enumerations,
                        reference_site_values)
from .resampling import (bootstrap, compare_bootstrap, dominating_sites,
                         exact_bootstrap_reference, summarize_sites)
from .substitution import describe_model


def _tree_copy(tree):
    if "name" in tree:
        return {"name": tree["name"], "length": tree["length"]}
    return {"children": [_tree_copy(child) for child in tree["children"]],
            "length": tree["length"]}


def _evaluate_candidate(candidate, partitions, models, masks_by_taxon, taxa):
    """Evaluate one candidate across all partitions.

    Returns ``(partition_details, site_values, site_statuses, diagnostics)``.
    ``site_values`` and ``site_statuses`` are indexed by the global column
    index.  The work budget is checked for the whole request before any
    candidate runs, so every candidate is either evaluated completely or not at
    all; a partially evaluated candidate would make the comparison unfair.
    """
    taxon_index = {name: index for index, name in enumerate(taxa)}
    n_sites = sum(len(partition["sites"]) for partition in partitions)
    site_values = [None] * n_sites
    site_statuses = ["unassigned"] * n_sites
    partition_details = []
    depth = 0
    minimum_entry = math.inf
    negative_entries = 0
    max_row_error = 0.0
    zero_branches = 0
    clamped_entries = 0
    most_negative = 0.0
    for partition in partitions:
        model = models[(candidate["id"], partition["id"])]
        nodes, root, partition_depth, partition_zero = build_nodes(candidate["tree"], taxon_index)
        run = ModelRun(nodes, root, model)
        depth = max(depth, partition_depth)
        zero_branches += partition_zero
        minimum_entry = min(minimum_entry, run.diagnostics["minimum_entry"])
        negative_entries += run.diagnostics["negative_entries"]
        max_row_error = max(max_row_error, run.diagnostics["max_row_sum_error"])
        clamped_entries += run.diagnostics["clamped_transition_entries"]
        if run.diagnostics["most_negative_raw_transition"] is not None:
            most_negative = min(most_negative, run.diagnostics["most_negative_raw_transition"])
        observed = frozenset(taxon_index[name] for name in partition["taxa"])
        values = []
        statuses = []
        for site in partition["sites"]:
            value, status = run.site(masks_by_taxon, site, observed)
            values.append(value)
            statuses.append(status)
        zero_sites = [site for site, status in zip(partition["sites"], statuses) if status == "zero"]
        nonfinite_sites = [site for site, status in zip(partition["sites"], statuses) if status == "nonfinite"]
        summary = summarize_sites(values, partition["sites"], zero_sites, nonfinite_sites)
        partition_details.append({
            "id": partition["id"],
            "taxa": partition["taxa"],
            "sites": len(partition["sites"]),
            "model": describe_model(model),
            "site_log_likelihoods": values,
            "log_likelihood": summary["log_likelihood"],
            "uninformative_sites": summary["uninformative_sites"],
            "zero_probability_sites": zero_sites,
            "nonfinite_sites": nonfinite_sites,
            "status": summary["status"],
        })
        for site, value, status in zip(partition["sites"], values, statuses):
            site_values[site] = value
            site_statuses[site] = {
                "ok": "ok", "zero": "zero_probability", "nonfinite": "numeric_instability",
            }[status]
    diagnostics = {
        "minimum_transition_probability": None if minimum_entry == math.inf else minimum_entry,
        "negative_transition_entries": negative_entries,
        "max_transition_row_sum_error": max_row_error,
        "clamped_transition_entries": clamped_entries,
        "most_negative_raw_transition": most_negative if clamped_entries else None,
        "zero_length_branches": zero_branches,
        "max_tree_depth": depth,
        "budget_exhausted": False,
    }
    return partition_details, site_values, site_statuses, diagnostics


def _candidate_status(partition_details, site_statuses):
    if any(detail["status"] == "budget_exhausted" for detail in partition_details):
        return "budget_exhausted"
    if any(status == "zero_probability" for status in site_statuses):
        return "zero_probability"
    if any(status == "numeric_instability" for status in site_statuses):
        return "numeric_instability"
    return "ok"


def _candidate_total(site_values, status):
    if status != "ok":
        return None
    return math.fsum(site_values)


def _delta_from_best(candidate_results, threshold):
    comparable = [result for result in candidate_results if result["log_likelihood"] is not None]
    order = sorted(candidate_results,
                   key=lambda item: (-(item["log_likelihood"] if item["log_likelihood"] is not None else -math.inf),
                                     item["id"]))
    if not comparable:
        return {
            "best": None,
            "ranking": [item["id"] for item in order],
            "delta_from_best": [{"id": item["id"], "delta_from_best": None,
                                 "status": "not_comparable"} for item in order],
            "distinguishability_threshold": threshold,
            "indistinguishable_candidates": [],
            "status": "no_comparable_candidate",
        }
    best = order[0]
    best_total = best["log_likelihood"]
    deltas = []
    for result in order:
        if result["log_likelihood"] is None:
            # Distinguish a genuine scientific zero from a skipped candidate.
            deltas.append({
                "id": result["id"],
                "delta_from_best": None,
                "status": ("zero_probability" if result["status"] == "zero_probability"
                           else "not_comparable"),
            })
        else:
            deltas.append({
                "id": result["id"],
                "delta_from_best": best_total - result["log_likelihood"],
                "status": "best" if result["id"] == best["id"] else "ok",
            })
    leaders = [result for result in comparable
               if best_total - result["log_likelihood"] <= threshold]
    indistinguishable = len(leaders) > 1
    return {
        "best": best["id"],
        "ranking": [result["id"] for result in order],
        "delta_from_best": deltas,
        "distinguishability_threshold": threshold,
        "indistinguishable_candidates": [result["id"] for result in leaders] if indistinguishable else [],
        "status": "indistinguishable" if indistinguishable else "distinguished",
    }


def _best_candidate(candidate_results):
    """Highest total log likelihood; ties broken by lexical id for stability."""
    comparable = [result for result in candidate_results if result["log_likelihood"] is not None]
    if not comparable:
        return None
    return sorted(comparable, key=lambda item: (-item["log_likelihood"], item["id"]))[0]


def _partition_deltas(candidate_results, partitions):
    """Per-partition log-likelihood contributions and best-vs-other deltas."""
    best = _best_candidate(candidate_results)
    if best is None:
        return []
    best_by_partition = {detail["id"]: detail["log_likelihood"] for detail in best["partitions"]}
    rows = []
    for partition in partitions:
        contributions = {}
        for result in candidate_results:
            detail = next(detail for detail in result["partitions"] if detail["id"] == partition["id"])
            contributions[result["id"]] = detail["log_likelihood"]
        rows.append({
            "id": partition["id"],
            "sites": len(partition["sites"]),
            "log_likelihood_by_candidate": contributions,
            "delta_from_best_by_candidate": {
                identifier: (None if value is None or best_by_partition[partition["id"]] is None
                             else best_by_partition[partition["id"]] - value)
                for identifier, value in contributions.items()
            },
        })
    return rows


def _site_table(candidate_results, partitions):
    """Per-column values for resampling.

    A column with probability exactly zero becomes ``-inf`` so the candidate
    genuinely loses a replicate that resamples it.  ``None`` is reserved for a
    numerically indeterminate column, which makes a replicate incomparable.
    """
    n_sites = sum(len(partition["sites"]) for partition in partitions)
    table = {}
    for result in candidate_results:
        row = [None] * n_sites
        zero = set(result.get("zero_probability_sites", ()))
        for detail in result["partitions"]:
            if detail["site_log_likelihoods"] is None:
                continue
            for position, site in enumerate(next(
                    partition["sites"] for partition in partitions if partition["id"] == detail["id"])):
                row[site] = -math.inf if site in zero else detail["site_log_likelihoods"][position]
        table[result["id"]] = row
    return table


def _serializable_site_table(table):
    """Replace ``-inf`` markers with ``None`` for JSON output."""
    return {
        identifier: [None if value == -math.inf else value for value in row]
        for identifier, row in table.items()
    }


def _site_deltas(table, best):
    """Per-column contribution to each candidate's gap from ``best``.

    For candidate ``c`` and column ``s`` the value is
    ``logL_best(s) - logL_c(s)``.  Applied to a partition, these deltas sum to
    the candidate's total delta from the best.  When either side has zero
    probability the delta is unbounded; ``None`` is emitted instead of a fake
    finite number.
    """
    if best is None or best not in table:
        return None
    best_row = table[best]
    result = {}
    for identifier, row in table.items():
        if identifier == best:
            result[identifier] = [0.0] * len(best_row)
            continue
        deltas = []
        for best_value, value in zip(best_row, row):
            if best_value is None or value is None:
                deltas.append(None)
            elif best_value == -math.inf or value == -math.inf:
                deltas.append(None)
            else:
                deltas.append(best_value - value)
        result[identifier] = deltas
    return result


def _site_partition_map(partitions):
    """Map each column index to its partition id."""
    mapping = {}
    for partition in partitions:
        for site in partition["sites"]:
            mapping[site] = partition["id"]
    return mapping


def _verification(candidates, partitions, models, mask_rows, candidate_results, taxa):
    reports = []
    for candidate in candidates:
        result = next(item for item in candidate_results if item["id"] == candidate["id"])
        entry = {"id": candidate["id"], "partitions": []}
        taxon_index = {name: index for index, name in enumerate(taxa)}
        for partition in partitions:
            model = models[(candidate["id"], partition["id"])]
            nodes, root, _, _ = build_nodes(candidate["tree"], taxon_index)
            detail = next(item for item in result["partitions"] if item["id"] == partition["id"])
            if detail["site_log_likelihoods"] is None:
                entry["partitions"].append({"id": partition["id"], "status": "not_compared"})
                continue
            if enumerations(nodes, len(partition["sites"]), len(model["rates"])) > MAX_ENUMERATIONS:
                entry["partitions"].append({"id": partition["id"], "status": "skipped_too_large"})
                continue
            observed = frozenset(taxon_index[name] for name in partition["taxa"])
            expected = reference_site_values(nodes, root, mask_rows, partition["sites"], observed, model)
            comparison = compare_site_values(expected, detail["site_log_likelihoods"])
            entry["partitions"].append({
                "id": partition["id"],
                **comparison,
                "status": "ok" if comparison["agrees"] else "disagreement",
            })
        reports.append(entry)
    return reports


def _empty_candidate(candidate, partitions, models, status, reason_detail):
    """A candidate result that was not evaluated, for explicable budget states."""
    details = []
    for partition in partitions:
        details.append({
            "id": partition["id"],
            "taxa": partition["taxa"],
            "sites": len(partition["sites"]),
            "model": describe_model(models[(candidate["id"], partition["id"])]),
            "site_log_likelihoods": None,
            "log_likelihood": None,
            "status": status,
        })
    n_sites = sum(len(partition["sites"]) for partition in partitions)
    return {
        "id": candidate["id"],
        "tree": _tree_copy(candidate["tree"]),
        "partitions": details,
        "site_log_likelihoods": [None] * n_sites,
        "log_likelihood": None,
        "status": status,
        "zero_probability_sites": [],
        "nonfinite_sites": [],
        "uninformative_sites": [],
        "status_detail": reason_detail,
    }


def process(payload):
    setup = parse_comparison(payload)
    taxa = setup["taxa"]
    masks = setup["masks"]
    mask_rows = [masks[name] for name in taxa]
    partitions = setup["partitions"]
    candidates = setup["candidates"]
    models = effective_models(setup["candidates"], partitions, setup["shared"])
    required = len(candidates) * setup["n_sites"]
    within_budget = required <= setup["budget"]["max_site_likelihoods"]
    candidate_results = []
    diagnostics = {}
    if not within_budget:
        detail = (f"Comparison needs {required} site likelihoods but the budget allows "
                  f"{setup['budget']['max_site_likelihoods']}")
        for candidate in candidates:
            candidate_results.append(_empty_candidate(candidate, partitions, models,
                                                      "budget_exhausted", detail))
            diagnostics[candidate["id"]] = {"budget_exhausted": True}
    else:
        for candidate in candidates:
            details, site_values, site_statuses, extra = _evaluate_candidate(
                candidate, partitions, models, mask_rows, taxa)
            status = _candidate_status(details, site_statuses)
            zero_sites = [index for index, value in enumerate(site_statuses) if value == "zero_probability"]
            nonfinite_sites = [index for index, value in enumerate(site_statuses) if value == "numeric_instability"]
            uninformative = [index for index, value in enumerate(site_values)
                             if value is not None and abs(value) <= 1e-9]
            candidate_results.append({
                "id": candidate["id"],
                "tree": _tree_copy(candidate["tree"]),
                "partitions": details,
                "site_log_likelihoods": site_values,
                "log_likelihood": _candidate_total(site_values, status),
                "status": status,
                "zero_probability_sites": zero_sites,
                "nonfinite_sites": nonfinite_sites,
                "uninformative_sites": uninformative,
            })
            diagnostics[candidate["id"]] = extra

    comparison = _delta_from_best(candidate_results, setup["distinguishability_threshold"])
    comparison["partition_deltas"] = _partition_deltas(candidate_results, partitions)
    table = _site_table(candidate_results, partitions)

    resampling_setup = setup["resampling"]
    resampling_work = resampling_setup["replicates"] * setup["n_sites"]
    if resampling_setup["replicates"] and resampling_work > setup["budget"]["max_resampling_work"]:
        stability = {
            "method": "nonparametric-bootstrap-over-aligned-columns",
            "replicates": resampling_setup["replicates"],
            "seed": resampling_setup["seed"],
            "budget_exhausted": True,
            "status": "budget_exhausted",
            "status_detail": (f"Resampling needs {resampling_work} column draws but the budget "
                              f"allows {setup['budget']['max_resampling_work']}"),
            "bootstrap_support_fraction": {result["id"]: None for result in candidate_results},
            "support_interpretation": (
                "Fraction of site-resampling replicates, not a posterior probability "
                "and not the probability that a candidate tree is true"
            ),
        }
    else:
        stability = bootstrap(table, setup["n_sites"], resampling_setup["replicates"],
                              resampling_setup["seed"])
        exact = exact_bootstrap_reference(table, setup["n_sites"])
        if exact is not None:
            stability["exact_reference"] = exact
            if stability.get("bootstrap_support_fraction"):
                stability["monte_carlo_check"] = compare_bootstrap(
                    stability["bootstrap_support_fraction"], exact, resampling_setup["replicates"])
        else:
            stability["exact_reference"] = {
                "status": "skipped_too_large",
                "reason": "The exact resampling distribution has too many outcome classes for this site count",
            }

    dominance = None
    comparable = [result for result in candidate_results if result["log_likelihood"] is not None]
    if len(comparable) >= 2:
        ordered = sorted(comparable, key=lambda item: (-item["log_likelihood"], item["id"]))
        leader, runner_up = ordered[0], ordered[1]
        deltas = []
        for first, second in zip(table[leader["id"]], table[runner_up["id"]]):
            if first is None or second is None:
                deltas.append(0.0)
            else:
                deltas.append(first - second)
        dominance = dominating_sites(deltas, setup["site_dominance_threshold"])
        dominance["leader"] = leader["id"]
        dominance["runner_up"] = runner_up["id"]

    caveats = []
    if comparison["status"] == "indistinguishable":
        caveats.append(
            "Candidates within the distinguishability threshold cannot be separated by this "
            "observation and model."
        )
    if any(result["status"] == "zero_probability" for result in candidate_results):
        caveats.append(
            "At least one candidate assigns exactly zero probability to at least one observed "
            "column; that column is reported, not dropped."
        )
    if any(result["status"] == "budget_exhausted" for result in candidate_results):
        caveats.append(
            "The site-likelihood budget was exhausted; affected candidate totals are unavailable, "
            "not zero."
        )
    if any(result["status"] == "numeric_instability" for result in candidate_results):
        caveats.append(
            "At least one column left the representable range; the candidate is reported as "
            "numerically unstable rather than as having zero probability."
        )
    if dominance is not None and dominance["status"] == "site_dominated":
        caveats.append(
            "A small number of columns carries much of the leading evidence difference; treat the "
            "ranking as sensitive to those columns."
        )
    numerical_deep = [identifier for identifier, detail in diagnostics.items()
                      if detail.get("negative_transition_entries")
                      or detail.get("max_transition_row_sum_error", 0.0) > 1e-8
                      or (detail.get("most_negative_raw_transition") or 0.0) < -1e-9]
    if numerical_deep:
        caveats.append(
            "Transition probabilities on deep branches show numerical error for: "
            + ", ".join(sorted(numerical_deep))
        )
    if comparison["status"] == "no_comparable_candidate":
        caveats.append("No candidate produced a finite total log likelihood.")

    return {
        "model": "candidate-evidence-comparison",
        "observation": {
            "taxa": taxa,
            "sites": setup["n_sites"],
            "partitions": [{"id": partition["id"], "sites": len(partition["sites"]),
                            "taxa": partition["taxa"]} for partition in partitions],
            "ambiguity": setup["ambiguity"],
            "root_prior": "stationary frequencies of each candidate's substitution model",
            "scale": "expected substitutions per site, expected rate normalized to one",
        },
        "shared_parameters": setup["shared"],
        "candidates": candidate_results,
        "comparison": comparison,
        "comparison_interpretation": {
            "delta_units": "natural-log likelihood units on the shared expected-substitutions-per-site scale",
            "delta_meaning": (
                "Differences in log likelihood under the supplied candidates; they are not "
                "posterior probabilities and do not account for unequal candidate complexity"
            ),
            "indistinguishable_rule": (
                "A candidate is indistinguishable from the best when its total log likelihood is "
                "within distinguishability_threshold of the best"
            ),
            "model_precedence": (
                "Effective model per candidate and partition: partition model, else candidate "
                "model, else JC69"
            ),
        },
        "site_evidence": {
            "per_site_log_likelihoods": _serializable_site_table(table),
            "per_site_delta_from_best": _site_deltas(table, comparison.get("best")),
            "null_marker_note": (
                "A null per-site value is explained per candidate: it is in "
                "zero_probability_sites when probability is exactly zero, or in "
                "nonfinite_sites when the calculation left the representable range"
            ),
            "site_partition": _site_partition_map(partitions),
        },
        "site_dominance": dominance,
        "resampling": stability,
        "diagnostics": diagnostics,
        "verification": _verification(candidates, partitions, models, mask_rows, candidate_results, taxa),
        "caveats": caveats,
    }
