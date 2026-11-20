"""Validation of candidate trees, site partitions and comparison controls."""
from ..errors import DomainError
from ..validation import fields, mapping, sequence, label, number, integer
from .ambiguity import prepare_aligned
from .model_blocks import parse_tree
from .substitution import compile_model

MAX_CANDIDATES = 16
MAX_PARTITIONS = 64
MAX_REPLICATES = 100000
DEFAULT_MAX_SITE_LIKELIHOODS = 2_000_000
DEFAULT_MAX_RESAMPLING_WORK = 200_000_000
SHARED_PARAMETERS = ("substitution", "frequencies", "exchangeabilities", "rate_categories")


def _candidate_identifier(entry, index):
    has_id = "id" in entry
    has_name = "name" in entry
    if has_id and has_name:
        raise DomainError("candidate_id", "Use one of 'id' or 'name' per candidate")
    if has_id:
        return label(entry["id"], "candidate id")
    if has_name:
        return label(entry["name"], "candidate name")
    return f"candidate-{index + 1}"


def parse_candidates(value, taxa):
    entries = sequence(value, "candidates", minimum=1, maximum=MAX_CANDIDATES)
    result = []
    identifiers = set()
    for index, entry in enumerate(entries):
        mapping(entry, "candidate")
        fields(entry, ["tree"], ["id", "name", "model"])
        identifier = _candidate_identifier(entry, index)
        if identifier in identifiers:
            raise DomainError("candidate_id", "Candidate identifiers must be unique")
        identifiers.add(identifier)
        tree, leaves = parse_tree(entry["tree"])
        if len(leaves) != len(set(leaves)) or set(leaves) != set(taxa):
            raise DomainError(
                "candidate_coverage",
                f"Candidate {identifier!r} leaves must match the observed taxa exactly; "
                "different taxon coverage would compare different observation populations",
            )
        model = compile_model(entry["model"]) if "model" in entry else None
        result.append({"id": identifier, "tree": tree, "leaves": leaves, "model": model})
    return result


def _parse_sites(spec, name, n_sites):
    if isinstance(spec, dict):
        fields(spec, [], ["sites", "start", "end"])
        has_list = "sites" in spec
        has_range = "start" in spec or "end" in spec
        if has_list and has_range:
            raise DomainError("partition_sites", f"{name} must not mix 'sites' with 'start'/'end'")
        if has_list:
            raw = sequence(spec["sites"], f"{name} sites", minimum=1, maximum=n_sites)
            return [integer(x, "site index", minimum=0, maximum=n_sites - 1) for x in raw]
        if "start" not in spec or "end" not in spec:
            raise DomainError("partition_sites", f"{name} requires both 'start' and 'end'")
        start = integer(spec["start"], "start", minimum=0, maximum=n_sites)
        end = integer(spec["end"], "end", minimum=0, maximum=n_sites)
        if end <= start:
            raise DomainError("partition_sites", f"{name} range must be nonempty and half-open")
        return list(range(start, end))
    raw = sequence(spec, f"{name} sites", minimum=1, maximum=n_sites)
    return [integer(x, "site index", minimum=0, maximum=n_sites - 1) for x in raw]


def parse_partitions(value, n_sites, taxa):
    if value is None:
        return [{"id": "all", "sites": list(range(n_sites)), "taxa": list(taxa), "model": None}]
    entries = sequence(value, "partitions", minimum=1, maximum=MAX_PARTITIONS)
    partitions = []
    identifiers = set()
    owner = [None] * n_sites
    for index, entry in enumerate(entries):
        mapping(entry, "partition")
        fields(entry, ["id", "sites"], ["taxa", "model"])
        identifier = label(entry["id"], "partition id")
        if identifier in identifiers:
            raise DomainError("partition_id", "Partition identifiers must be unique")
        identifiers.add(identifier)
        sites = _parse_sites(entry["sites"], identifier, n_sites)
        if len(sites) != len(set(sites)):
            raise DomainError("partition_sites", f"Partition {identifier!r} lists a site more than once")
        for site in sites:
            if owner[site] is not None:
                raise DomainError(
                    "partition_coverage",
                    f"Site {site} belongs to partitions {owner[site]!r} and {identifier!r}",
                )
            owner[site] = identifier
        if "taxa" in entry:
            raw = sequence(entry["taxa"], "partition taxa", minimum=1, maximum=len(taxa))
            observed = [label(x, "partition taxon") for x in raw]
            if len(observed) != len(set(observed)):
                raise DomainError("partition_taxa", "Partition taxa must be unique")
            unknown = set(observed) - set(taxa)
            if unknown:
                raise DomainError(
                    "partition_taxa",
                    "Partition taxa are not present in sequences: " + ", ".join(sorted(unknown)),
                )
        else:
            observed = list(taxa)
        model = compile_model(entry["model"]) if "model" in entry else None
        partitions.append({"id": identifier, "sites": sorted(sites),
                           "taxa": sorted(observed), "model": model})
    unassigned = [site for site, name in enumerate(owner) if name is None]
    if unassigned:
        raise DomainError(
            "partition_coverage",
            f"{len(unassigned)} site(s) are not assigned to any partition; "
            "every observed column must be evaluated",
        )
    return partitions


def parse_shared_parameters(value):
    if value is None:
        return []
    raw = sequence(value, "shared_parameters", minimum=0, maximum=len(SHARED_PARAMETERS))
    names = []
    for item in raw:
        name = label(item, "shared parameter")
        if name not in SHARED_PARAMETERS:
            raise DomainError(
                "shared_parameter",
                "shared_parameters accepts: " + ", ".join(SHARED_PARAMETERS),
            )
        if name not in names:
            names.append(name)
    return names


def parse_resampling(value):
    if value is None:
        return {"replicates": 200, "seed": 0}
    mapping(value, "resampling")
    fields(value, [], ["replicates", "seed"])
    replicates = integer(value.get("replicates", 200), "replicates", minimum=0, maximum=MAX_REPLICATES)
    seed = integer(value.get("seed", 0), "seed", minimum=0, maximum=2**63 - 1)
    return {"replicates": replicates, "seed": seed}


def parse_budget(value):
    if value is None:
        return {"max_site_likelihoods": DEFAULT_MAX_SITE_LIKELIHOODS,
                "max_resampling_work": DEFAULT_MAX_RESAMPLING_WORK}
    mapping(value, "budget")
    fields(value, [], ["max_site_likelihoods", "max_resampling_work"])
    return {
        "max_site_likelihoods": integer(
            value.get("max_site_likelihoods", DEFAULT_MAX_SITE_LIKELIHOODS),
            "max_site_likelihoods", minimum=1, maximum=10**12),
        "max_resampling_work": integer(
            value.get("max_resampling_work", DEFAULT_MAX_RESAMPLING_WORK),
            "max_resampling_work", minimum=1, maximum=10**15),
    }


def parse_comparison(payload):
    fields(payload, ["sequences", "candidates"],
           ["partitions", "shared_parameters", "resampling", "budget",
            "distinguishability_threshold", "site_dominance_threshold"])
    taxa, masks, n_sites, ambiguity = prepare_aligned(payload["sequences"])
    candidates = parse_candidates(payload["candidates"], taxa)
    partitions = parse_partitions(payload.get("partitions"), n_sites, taxa)
    shared = parse_shared_parameters(payload.get("shared_parameters"))
    resampling = parse_resampling(payload.get("resampling"))
    budget = parse_budget(payload.get("budget"))
    threshold = number(payload.get("distinguishability_threshold", 2.0),
                       "distinguishability_threshold", minimum=0)
    dominance = number(payload.get("site_dominance_threshold", 0.5),
                       "site_dominance_threshold", minimum=1e-9, maximum=1.0)
    return {
        "taxa": taxa, "masks": masks, "n_sites": n_sites, "ambiguity": ambiguity,
        "candidates": candidates, "partitions": partitions, "shared": shared,
        "resampling": resampling, "budget": budget,
        "distinguishability_threshold": threshold,
        "site_dominance_threshold": dominance,
    }


def effective_models(candidates, partitions, shared):
    """Resolve one model per (candidate, partition) and enforce sharing.

    Precedence is partition model, then candidate model, then the default
    JC69 model.  A declared shared parameter must agree across every effective
    model, otherwise the comparison would silently mix scales.
    """
    from .substitution import model_parameter, parameters_equal
    resolved = {}
    for candidate in candidates:
        for partition in partitions:
            model = partition["model"] or candidate["model"] or compile_model({})
            resolved[(candidate["id"], partition["id"])] = model
    values = list(resolved.values())
    for name in shared:
        reference = model_parameter(values[0], name)
        for model in values[1:]:
            if not parameters_equal(values[0], model, name):
                raise DomainError(
                    "shared_parameter_violation",
                    f"Declared shared parameter {name!r} differs between effective models",
                )
    return resolved
