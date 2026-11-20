"""Input parsing for the extended transcript-quantification request.

A request that carries only ``effective_lengths`` and ``fragments`` keeps the
original unbiased, fixed-length model (``compatibility`` + ``likelihood``).
Any extended measurement field switches to the weighted model.

Measurement fields
------------------
``transcript_lengths``
    Annotated transcript lengths.  Together with the observed fragment-length
    distribution they determine the effective length ``L_i = E_l[A_i(l)]``,
    where ``A_i(l) = max(T_i - l + 1, 0)`` is the number of start positions a
    fragment of length ``l`` has on transcript ``i``.  The distribution is
    given by ``fragment_lengths`` (one length per fragment) or
    ``fragment_length_distribution`` (histogram, length list, or
    ``[length, weight]`` pairs).  ``effective_lengths`` may be supplied too;
    it is then checked against the derived value instead of being preferred.

``fragment_bias`` / ``bias_weights`` (alias ``bias``)
    Relative fragment-to-transcript detection weights.  ``fragment_bias`` is
    one weight object or vector per fragment; ``bias_weights`` is one weight
    per transcript applied to all of its fragments.  Weights are nonnegative
    and relative.  Only ratios within one fragment matter, so a global
    rescaling cancels: the bias reweights how an ambiguous fragment is split
    between its compatible transcripts, it does not rescale a transcript that
    every one of its fragments supports alone.

``observation_counts`` (alias ``fragment_counts``/``replicate_counts``)
    Replicate observations per fragment, or one integer applied to every
    fragment.  Replicates weight the estimate and are resampled as independent
    observation units, so more observations narrow the resampling interval.

Resampling configuration
------------------------
``resamples``, ``seed``, ``chunk_size`` and ``bootstrap_work_limit`` (also
accepted nested under ``random``, ``resampling`` or ``bootstrap``).  A
positive ``resamples`` requires an explicit integer ``seed``; the same seed
and fragment list reproduce identical output for any ``chunk_size``.

Fragments may be objects ``{"compatible": [...], "length": ..., "bias": {...},
"count": ...}`` instead of name lists with parallel arrays.  The two forms
cannot be mixed.  Recognized aliases are folded onto canonical fields;
conflicting values for one canonical field are rejected, never preferred.
"""
import math

from ..errors import DomainError
from ..validation import fields, integer, label, mapping, number, sequence

ALIASES = {
    "fragment_lengths": "fragment_lengths",
    "fragment_sizes": "fragment_lengths",
    "read_lengths": "fragment_lengths",
    "fragment_length_distribution": "fragment_length_distribution",
    "length_distribution": "fragment_length_distribution",
    "fragment_size_distribution": "fragment_length_distribution",
    "transcript_lengths": "transcript_lengths",
    "annotated_lengths": "transcript_lengths",
    "fragment_bias": "fragment_bias",
    "fragment_biases": "fragment_bias",
    "fragment_to_transcript_bias": "fragment_bias",
    "bias_weights": "bias_weights",
    "transcript_bias": "bias_weights",
    "relative_bias": "bias_weights",
    "observation_counts": "observation_counts",
    "fragment_counts": "observation_counts",
    "replicate_counts": "observation_counts",
    "replicates": "observation_counts",
    "resamples": "resamples",
    "bootstrap_resamples": "resamples",
    "n_resamples": "resamples",
    "seed": "seed",
    "random_seed": "seed",
    "rng_seed": "seed",
    "chunk_size": "chunk_size",
    "fragment_chunk_size": "chunk_size",
    "batch_size": "chunk_size",
    "low_abundance_tpm": "low_abundance_tpm",
    "low_abundance_threshold": "low_abundance_tpm",
    "abundance_threshold_tpm": "low_abundance_tpm",
    "bootstrap_work_limit": "bootstrap_work_limit",
    "work_limit": "bootstrap_work_limit",
    "budget": "bootstrap_work_limit",
}

# Extended-model fields.  ``max_iterations`` and ``tolerance`` are shared with
# the legacy model and do not switch models on their own.
EXTENDED_OPTIONS = tuple(ALIASES)
NESTED = ("random", "resampling", "bootstrap")
BIAS = "bias"

MEMBER_KEYS = ("compatible", "transcripts", "candidates",
               "compatible_transcripts", "supported_transcripts")
LENGTH_KEYS = ("length", "fragment_length", "size")
BIAS_KEYS = ("bias", "weights", "bias_weights")
COUNT_KEYS = ("count", "observations", "observation_count", "replicates")

MAX_MEMBERSHIPS = 5_000_000
MAX_TOTAL_OBSERVATIONS = 10_000_000

def is_extended(payload):
    return isinstance(payload, dict) and any(
        name in payload for name in (*EXTENDED_OPTIONS, BIAS, *NESTED))

def _first(data, keys):
    present = [key for key in keys if key in data]
    if len(present) > 1:
        raise DomainError("duplicate_option", "Use only one of: " + ", ".join(present))
    return data[present[0]] if present else None

def _canonical(payload):
    allowed = ["fragments", "effective_lengths", "max_iterations", "tolerance",
               BIAS, *NESTED, *ALIASES]
    fields(payload, ["fragments"], allowed)
    resolved = {key: payload[key] for key in
                ("fragments", "effective_lengths", "max_iterations", "tolerance")
                if key in payload}

    def merge(canonical, value, source):
        if canonical in resolved and resolved[canonical] != value:
            raise DomainError("duplicate_option",
                              f"Conflicting values for {canonical} (from {source})")
        resolved.setdefault(canonical, value)

    for key, value in payload.items():
        if key == BIAS:
            if isinstance(value, dict):
                merge("bias_weights", value, "bias")
            elif isinstance(value, list):
                merge("fragment_bias", value, "bias")
            else:
                raise DomainError("invalid_bias", "bias must be an object or a list")
        elif key in NESTED:
            config = mapping(value, key)
            unknown = set(config) - set(ALIASES) - {"tolerance", "max_iterations"}
            if unknown:
                raise DomainError("unsupported_field",
                                  "Unsupported fields: " + ", ".join(sorted(unknown)))
            for inner, item in config.items():
                merge(ALIASES.get(inner, inner), item, f"{key}.{inner}")
        elif key in ALIASES:
            merge(ALIASES[key], value, key)
    return resolved

def _lengths(value, name, field):
    data = mapping(value, name)
    if not 1 <= len(data) <= 1000:
        raise DomainError("size_limit", "Use between one and 1000 transcripts")
    result = {}
    for key, item in data.items():
        label(key, "transcript")
        result[key] = number(item, field, minimum=1, maximum=1e9)
    return result

def _fragments(value, index):
    """Parse fragments into ``{members, length, bias, count}`` records."""
    entries = sequence(value, "fragments", minimum=1, maximum=50000)
    object_form = isinstance(entries[0], dict)
    if any(isinstance(entry, dict) != object_form for entry in entries):
        raise DomainError("invalid_compatibility",
                          "Fragments must be all member lists or all objects")
    records = []
    memberships = 0
    for entry in entries:
        if object_form:
            unknown = set(entry) - set(MEMBER_KEYS) - set(LENGTH_KEYS) - set(BIAS_KEYS) - set(COUNT_KEYS)
            if unknown:
                raise DomainError("unsupported_field",
                                  "Unsupported fragment fields: " + ", ".join(sorted(unknown)))
            members = _first(entry, MEMBER_KEYS)
            length = _first(entry, LENGTH_KEYS)
            bias = _first(entry, BIAS_KEYS)
            count = _first(entry, COUNT_KEYS)
        else:
            members, length, bias, count = entry, None, None, None
        names = sequence(members, "compatible transcripts", minimum=1, maximum=len(index))
        if any(not isinstance(name, str) or name not in index for name in names) or len(set(names)) != len(names):
            raise DomainError("invalid_compatibility", "Each fragment must name distinct known transcripts")
        records.append({"members": tuple(sorted(index[name] for name in names)),
                        "length": length, "bias": bias, "count": count})
        memberships += len(names)
    if memberships > MAX_MEMBERSHIPS:
        raise DomainError("work_limit", "Fragment-transcript memberships exceed five million")
    return records

def _observations(value, records):
    if value is None:
        result = [1 if record["count"] is None
                  else integer(record["count"], "observation count", minimum=0, maximum=1_000_000)
                  for record in records]
    else:
        if any(record["count"] is not None for record in records):
            raise DomainError("duplicate_option",
                              "Use either per-fragment counts or observation_counts, not both")
        if isinstance(value, int) and not isinstance(value, bool):
            count = integer(value, "observation_counts", minimum=0, maximum=1_000_000)
            result = [count]*len(records)
        else:
            rows = sequence(value, "observation_counts", minimum=len(records), maximum=len(records))
            result = [integer(item, "observation count", minimum=0, maximum=1_000_000)
                      for item in rows]
    for record, count in zip(records, result):
        record["count"] = count
    if sum(result) > MAX_TOTAL_OBSERVATIONS:
        raise DomainError("work_limit", "Total observation count exceeds ten million")
    if sum(result) == 0:
        raise DomainError("no_observations", "Observation counts must describe at least one observation")
    return result

def _fragment_lengths(value, records):
    rows = sequence(value, "fragment_lengths", minimum=len(records), maximum=len(records))
    result = [number(item, "fragment length", minimum=1, maximum=1e9) for item in rows]
    for record, length in zip(records, result):
        if record["length"] is not None:
            raise DomainError("duplicate_option",
                              "Use either per-fragment lengths or fragment_lengths, not both")
        record["length"] = length
    return result

def _weight(value, name):
    return number(value, name, minimum=0, maximum=1e12)

def _distribution(value):
    """Normalized fragment-length histogram as sorted ``[(length, mass)]``."""
    entries = []
    if isinstance(value, dict) and {"mean"} & value.keys():
        # Summary form: a normal fragment-length model discretized over whole
        # bases, truncated at length 1.  The tail outside four standard
        # deviations is negligible and is not represented.
        mean = number(value["mean"], "fragment length mean", minimum=1, maximum=1e9)
        spread = value.get("sd", value.get("std", value.get("sigma")))
        if spread is None:
            raise DomainError("invalid_distribution",
                              "A summary distribution needs mean and sd")
        sd = number(spread, "fragment length sd", positive=True, maximum=1e9)
        low, high = max(1, math.floor(mean-4*sd)), math.ceil(mean+4*sd)
        if high-low > 100000:
            raise DomainError("invalid_distribution",
                              "Fragment length summary spans too many whole lengths")
        for length in range(low, high+1):
            z = (length-mean)/sd
            entries.append((float(length), math.exp(-0.5*z*z)))
    elif isinstance(value, dict):
        if "lengths" in value:
            if not {"weights", "counts"} & value.keys():
                raise DomainError("invalid_distribution",
                                  "A length distribution object needs lengths and weights/counts")
            lengths = sequence(value["lengths"], "lengths", minimum=1, maximum=100000)
            weights = sequence(value.get("weights", value.get("counts")), "weights",
                               minimum=len(lengths), maximum=len(lengths))
            for length, weight in zip(lengths, weights):
                entries.append((_weight(length, "fragment length"),
                                _weight(weight, "fragment length weight")))
        else:
            for key, weight in value.items():
                try:
                    length = float(key)
                except (TypeError, ValueError):
                    raise DomainError("invalid_distribution",
                                      "Fragment length distribution keys must be numeric lengths") from None
                entries.append((_weight(length, "fragment length"),
                                _weight(weight, "fragment length weight")))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, list):
                pair = sequence(item, "length-weight pair", minimum=2, maximum=2)
                entries.append((_weight(pair[0], "fragment length"),
                                _weight(pair[1], "fragment length weight")))
            else:
                entries.append((_weight(item, "fragment length"), 1.0))
        if not 1 <= len(entries) <= 100000:
            raise DomainError("invalid_distribution", "Fragment length distribution has an invalid size")
    else:
        raise DomainError("invalid_distribution", "Fragment length distribution must be an object or array")
    total = math.fsum(weight for _, weight in entries)
    if total <= 0:
        raise DomainError("invalid_distribution", "Fragment length distribution carries no mass")
    return sorted((length, weight/total) for length, weight in entries if weight > 0)

def _row_bias(value, members, names):
    """One fragment's relative weights aligned to its member order."""
    if isinstance(value, dict):
        compatible = {names[i] for i in members}
        if set(value) - compatible:
            raise DomainError("invalid_bias",
                              "Bias weights must name transcripts compatible with the same fragment")
        values = [value.get(names[i], 1.0) for i in members]
    else:
        row = sequence(value, "fragment_bias", minimum=1, maximum=len(names))
        if len(row) == len(members):
            values = row
        elif len(row) == len(names):
            values = [row[i] for i in members]
        else:
            raise DomainError("invalid_bias",
                              "A bias vector must align with the fragment's members or all transcripts")
    return tuple(_weight(item, "fragment bias") for item in values)

def _fragment_bias(value, records, names):
    if value is None:
        return
    rows = sequence(value, "fragment_bias", minimum=len(records), maximum=len(records))
    for record, row in zip(records, rows):
        if record["bias"] is not None:
            raise DomainError("duplicate_option",
                              "Use either per-fragment bias or fragment_bias, not both")
        record["bias"] = row

def _global_bias(value, names):
    data = mapping(value, "bias_weights")
    if set(data) - set(names):
        raise DomainError("invalid_bias", "Bias weights must name known transcripts")
    return {name: _weight(data.get(name, 1.0), "transcript bias") for name in names}

def _availability(transcript_length, fragment_length):
    return max(transcript_length-fragment_length+1.0, 0.0)

def prepare(payload):
    """Return a validated description of an extended request."""
    resolved = _canonical(payload)
    annotated = (_lengths(resolved["transcript_lengths"], "transcript_lengths", "transcript length")
                 if "transcript_lengths" in resolved else None)
    effective = (_lengths(resolved["effective_lengths"], "effective_lengths", "effective_length")
                 if "effective_lengths" in resolved else None)
    if annotated is None and effective is None:
        raise DomainError("missing_field",
                          "Missing fields: transcript_lengths or effective_lengths")
    if annotated is not None and effective is not None and set(annotated) != set(effective):
        raise DomainError("transcript_mismatch",
                          "transcript_lengths and effective_lengths must describe the same transcripts")
    if annotated is None and ({"fragment_lengths", "fragment_length_distribution"} & resolved.keys()):
        raise DomainError("missing_field",
                          "Fragment lengths require transcript_lengths for effective-length derivation")

    names = sorted(annotated if annotated is not None else effective)
    index = {name: i for i, name in enumerate(names)}
    records = _fragments(resolved["fragments"], index)
    observations = _observations(resolved.get("observation_counts"), records)
    total = sum(observations)

    # Fragment lengths may arrive as a parallel array or inside the fragment
    # objects, or as a histogram.  Exactly one form is allowed, and all three
    # require annotated transcript lengths to derive the effective length.
    object_lengths = any(record["length"] is not None for record in records)
    if object_lengths and not all(record["length"] is not None for record in records):
        raise DomainError("missing_field", "Either every fragment or none must carry a length")
    if "fragment_lengths" in resolved:
        if object_lengths:
            raise DomainError("duplicate_option",
                              "Use either per-fragment lengths or fragment_lengths, not both")
        _fragment_lengths(resolved["fragment_lengths"], records)
    distribution = (_distribution(resolved["fragment_length_distribution"])
                    if "fragment_length_distribution" in resolved else None)
    if distribution is not None and (object_lengths or "fragment_lengths" in resolved):
        raise DomainError("duplicate_option",
                          "Use either per-fragment lengths or fragment_length_distribution, not both")
    per_record_lengths = object_lengths or "fragment_lengths" in resolved
    has_lengths = per_record_lengths or distribution is not None
    if annotated is None and has_lengths:
        raise DomainError("missing_field",
                          "Fragment lengths require transcript_lengths for effective-length derivation")

    _fragment_bias(resolved.get("fragment_bias"), records, names)
    global_bias = _global_bias(resolved["bias_weights"], names) if "bias_weights" in resolved else None
    if global_bias is not None and any(record["bias"] is not None for record in records):
        raise DomainError("duplicate_option",
                          "Use either fragment_bias or bias_weights, not both")

    # Effective length: mean start-position availability over the observed
    # fragment-length distribution.  With fixed effective lengths (no length
    # distribution) the supplied value is used directly.
    if annotated is not None:
        if not has_lengths:
            raise DomainError("missing_field", "Missing fields: fragment_lengths")
        if per_record_lengths:
            lengths = [math.fsum(record["count"]*_availability(annotated[name], record["length"])
                                 for record in records)/total for name in names]
        else:
            lengths = [math.fsum(mass*_availability(annotated[name], value)
                                 for value, mass in distribution) for name in names]
        length_model = "empirical-fragment-length-availability-v1"
        for name, value in zip(names, lengths):
            if value <= 0:
                raise DomainError("zero_effective_length",
                                  "A transcript is shorter than every observed fragment length")
        if effective is not None:
            for name, value in zip(names, lengths):
                declared = effective[name]
                if abs(declared-value) > 1e-6*max(1.0, abs(declared), abs(value)):
                    raise DomainError("inconsistent_effective_length",
                                      "Supplied effective lengths disagree with the fragment-length distribution")
    else:
        lengths = [effective[name] for name in names]
        length_model = "fixed-effective-length-v1"

    # Relative per-fragment bias b_ji, the declared relative
    # fragment-to-transcript detection weight (default 1).  Conditional on a
    # fragment having been observed as compatible with its member set, its
    # length density p(l) and uniform start position are identical across
    # those members and cancel, so the observed fragment length does not
    # reweight the members: it enters abundance only through each transcript's
    # effective length and effective yield in the M-step.  Only bias ratios
    # within a fragment matter, so a global rescaling cancels, and no bias
    # leaves b_ji = 1, which reproduces the unbiased compatibility model.
    rows = []
    for record in records:
        members, count = record["members"], record["count"]
        if annotated is not None and record["length"] is not None:
            # The declared length must leave at least one start position on
            # every compatible transcript; otherwise the compatibility claim
            # itself is contradictory.
            available = [_availability(annotated[names[i]], record["length"]) for i in members]
            if count > 0 and all(value == 0.0 for value in available):
                raise DomainError("incompatible_fragment_length",
                                  "A fragment is at least as long as every compatible transcript")
        if record["bias"] is not None:
            bias = _row_bias(record["bias"], members, names)
        elif global_bias is not None:
            bias = tuple(global_bias[names[i]] for i in members)
        else:
            bias = tuple(1.0 for _ in members)
        if count > 0 and not any(value > 0 for value in bias):
            raise DomainError("zero_yield",
                              "Every compatible transcript has zero relative bias for an observed fragment")
        rows.append((members, bias, count))

    supported = set()
    for members, bias, count in rows:
        if count > 0:
            supported.update(i for i, value in zip(members, bias) if value > 0)

    support_count = [0]*len(names)
    support_bias = [0.0]*len(names)
    for members, bias, count in rows:
        if count > 0:
            for i, value in zip(members, bias):
                if value > 0:
                    support_count[i] += count
                    support_bias[i] += count*value

    def effective_yield(i):
        """Per-molecule expected fragment count used by the M-step.

        ``D_i = L_i * B_i`` where ``B_i`` is the mean declared bias over the
        observations that can support transcript ``i`` (1 when no bias is
        declared).  With no bias ``B_i = 1`` and ``D_i`` is exactly the reported
        effective length; a detection-favouring condition raises ``D_i`` and so
        lowers the molecules implied by the same observations.  Zero-weight
        fragments cannot arise from the transcript, not merely rarely, so they
        are excluded rather than averaged in.
        """
        if support_count[i] == 0:
            return lengths[i]
        return lengths[i]*(support_bias[i]/support_count[i])

    yields = [effective_yield(i) for i in range(len(names))]
    for name, length, yield_value in zip(names, lengths, yields):
        if index[name] in supported and (length <= 0 or yield_value <= 0):
            raise DomainError("zero_yield", "A supported transcript has no positive effective yield")

    seed = resolved.get("seed")
    if seed is not None:
        seed = integer(seed, "seed", minimum=0, maximum=2**63-1)
    # An explicit seed is a request for a reproducible uncertainty estimate, so
    # it selects the default resample count when that count is not stated.
    default_resamples = 200 if seed is not None else 0
    resamples = integer(resolved.get("resamples", default_resamples), "resamples",
                        minimum=0, maximum=10000)
    if resamples > 0 and seed is None:
        raise DomainError("missing_seed", "Resampling requires an explicit integer seed")

    return {
        "names": names,
        "effective_lengths": lengths,
        "effective_yields": yields,
        "effective_length_model": length_model,
        "rows": rows,
        "supported": supported,
        "has_fragment_lengths": per_record_lengths,
        "has_distribution": distribution is not None,
        "has_bias": global_bias is not None or any(record["bias"] is not None for record in records),
        "resamples": resamples,
        "seed": seed,
        "chunk_size": integer(resolved.get("chunk_size", 0), "chunk_size", minimum=0, maximum=50000),
        "low_abundance_tpm": number(resolved.get("low_abundance_tpm", 1.0), "low_abundance_tpm",
                                    minimum=0, maximum=1e6),
        "bootstrap_work_limit": integer(resolved.get("bootstrap_work_limit", 50_000_000),
                                        "bootstrap_work_limit", minimum=1, maximum=10**15),
        "max_iterations": integer(resolved.get("max_iterations", 1000), "max_iterations",
                                  minimum=1, maximum=5000),
        "tolerance": number(resolved.get("tolerance", 1e-10), "tolerance",
                            minimum=1e-14, maximum=1e-2),
        "total_observations": total,
    }
