"""Request contract for sparse stratified contact-enrichment analysis."""
from ..validation import fields, integer, sequence
from ..errors import DomainError

MAX_BINS = 4096
MAX_ENTRIES = 20000
MAX_STRATA = 64
MAX_RELATIONS = 5000
MAX_COUNT = 10**9


def _pair(item, name, bins):
    i = integer(item["i"], name + " i", minimum=0, maximum=bins-1)
    j = integer(item["j"], name + " j", minimum=0, maximum=bins-1)
    if i > j:
        raise DomainError("pair_order", f"{name} must satisfy i <= j")
    return i, j


def prepare(payload):
    fields(payload, ["bins", "entries", "strata", "relations"],
           ["valid_bins", "tolerance", "max_iterations", "min_stratum_pairs",
            "min_stratum_count", "confidence_z", "fdr_alpha", "max_relations"])
    bins = integer(payload["bins"], "bins", minimum=1, maximum=MAX_BINS)
    if "valid_bins" in payload:
        listed = sequence(payload["valid_bins"], "valid_bins", minimum=1, maximum=bins)
        valid = set()
        for value in listed:
            bin_id = integer(value, "valid bin", minimum=0, maximum=bins-1)
            if bin_id in valid:
                raise DomainError("duplicate_bin", "Valid bins must be unique")
            valid.add(bin_id)
    else:
        valid = set(range(bins))
    entries = {}
    for item in sequence(payload["entries"], "entries", minimum=1, maximum=MAX_ENTRIES):
        fields(item, ["i", "j", "count"])
        pair = _pair(item, "entry", bins)
        if pair[0] not in valid or pair[1] not in valid:
            raise DomainError("masked_entry", "Entries must reference valid bins")
        if pair in entries:
            raise DomainError("duplicate_entry", "Each pair may be listed at most once")
        entries[pair] = integer(item["count"], "count", minimum=0, maximum=MAX_COUNT)
    strata = []
    for item in sequence(payload["strata"], "strata", minimum=1, maximum=MAX_STRATA):
        fields(item, ["min_distance", "max_distance"])
        low = integer(item["min_distance"], "min_distance", minimum=0, maximum=bins-1)
        high = integer(item["max_distance"], "max_distance", minimum=0, maximum=bins-1)
        if low > high:
            raise DomainError("stratum_order", "Strata must satisfy min_distance <= max_distance")
        strata.append((low, high))
    ordered = sorted(strata)
    for position in range(1, len(ordered)):
        if ordered[position][0] <= ordered[position-1][1]:
            raise DomainError("overlapping_strata", "Distance strata must be disjoint")
    stratum_of = [None]*bins
    for index, (low, high) in enumerate(strata):
        for distance in range(low, high+1):
            stratum_of[distance] = index
    relations = []
    seen_relations = set()
    for item in sequence(payload["relations"], "relations", minimum=1, maximum=MAX_RELATIONS):
        fields(item, ["i", "j"])
        pair = _pair(item, "relation", bins)
        if pair in seen_relations:
            raise DomainError("duplicate_relation", "Each relation may be queried at most once")
        seen_relations.add(pair)
        relations.append(pair)
    return {"bins": bins, "valid": valid, "entries": entries, "strata": strata,
            "stratum_of": stratum_of, "relations": relations}
