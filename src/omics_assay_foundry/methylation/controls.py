"""Request parsing and control registry for the methylation endpoint.

A calibrated request carries one or more spike-in conversion controls.  Each
control entry describes two measurements made on one batch and one strand:

* a fully methylated control, whose unmethylated reads are conversion
  failures, and
* a fully unmethylated control, whose methylated reads are background.

A site selects a control by ``control`` id, by its ``batch``/``strand`` pair,
or implicitly when exactly one control fits.  A site with no matching control
is still a valid request: it is reported as unidentifiable rather than
rejected, because the missing measurement belongs to the assay rather than to
the request syntax.
"""

from ..errors import DomainError
from ..validation import fields, integer, label, mapping, number, sequence

_STRAND_ALIASES = {
    "+": "forward", "fwd": "forward", "for": "forward", "forward": "forward",
    "plus": "forward", "1": "forward", "watson": "forward",
    "-": "reverse", "rev": "reverse", "reverse": "reverse", "minus": "reverse",
    "2": "reverse", "crick": "reverse",
}

_METHYLATED_KEYS = ("fully_methylated", "fully_methylated_control", "methylated_control",
                    "all_methylated", "full_methylated")
_UNMETHYLATED_KEYS = ("fully_unmethylated", "fully_unmethylated_control", "unmethylated_control",
                      "all_unmethylated", "full_unmethylated")
_ENTRY_KEYS = ({"id", "batch", "strand"}
               | set(_METHYLATED_KEYS) | set(_UNMETHYLATED_KEYS))

_COUNT_KEYS = {"methylated", "unmethylated", "coverage", "total", "reads",
               "methylated_reads", "unmethylated_reads", "total_reads"}


def canonical_strand(value):
    text = label(value, "strand").strip().lower()
    return _STRAND_ALIASES.get(text, text)


def _counts(value, name):
    """Read a {methylated, coverage}-style measurement."""
    mapping(value)
    unknown = value.keys() - _COUNT_KEYS
    if unknown:
        raise DomainError("unsupported_field", "Unsupported fields: " + ", ".join(sorted(unknown)))
    marked = None
    clear = None
    total = None
    duplicates = []
    if "methylated" in value and "methylated_reads" in value:
        duplicates.append("methylated")
    if "unmethylated" in value and "unmethylated_reads" in value:
        duplicates.append("unmethylated")
    for key in ("coverage", "total", "reads", "total_reads"):
        if key in value:
            if total is not None:
                duplicates.append(key)
            total = integer(value[key], f"{name} coverage", minimum=0, maximum=10**9)
    if "methylated" in value:
        marked = integer(value["methylated"], f"{name} methylated", minimum=0, maximum=10**9)
    elif "methylated_reads" in value:
        marked = integer(value["methylated_reads"], f"{name} methylated", minimum=0, maximum=10**9)
    if "unmethylated" in value:
        clear = integer(value["unmethylated"], f"{name} unmethylated", minimum=0, maximum=10**9)
    elif "unmethylated_reads" in value:
        clear = integer(value["unmethylated_reads"], f"{name} unmethylated", minimum=0, maximum=10**9)
    if duplicates:
        raise DomainError("duplicate_field", f"{name} gives {duplicates[0]} more than once")
    if total is None:
        if marked is not None and clear is not None:
            total = marked + clear
        elif marked is not None:
            total = marked
        elif clear is not None:
            total = clear
        else:
            raise DomainError("missing_field",
                              f"{name} needs coverage, or methylated/unmethylated counts")
    if marked is None:
        if clear is None:
            raise DomainError("missing_field", f"{name} needs a methylated count")
        marked = total - clear
        if marked < 0:
            raise DomainError("invalid_integer", f"{name} unmethylated count exceeds coverage")
    if clear is not None and marked + clear > total:
        raise DomainError("invalid_integer", f"{name} counts exceed coverage")
    if marked > total:
        raise DomainError("invalid_integer", f"{name} methylated count exceeds coverage")
    return marked, total


def _pick(record, keys, name):
    present = [key for key in keys if key in record]
    if not present:
        raise DomainError("missing_field", f"Missing fields: {name}")
    if len(present) > 1:
        raise DomainError("duplicate_field", f"{name} is given more than once")
    return record[present[0]]


def _entry(record):
    mapping(record)
    unknown = record.keys() - _ENTRY_KEYS
    if unknown:
        raise DomainError("unsupported_field", "Unsupported fields: " + ", ".join(sorted(unknown)))
    if "strand" not in record:
        raise DomainError("missing_field", "Missing fields: strand")
    raw_strand = label(record["strand"], "control strand")
    strand = canonical_strand(raw_strand)
    batch = label(record["batch"], "control batch") if "batch" in record else ""
    control_id = label(record["id"], "control id") if "id" in record else (
        f"{batch}:{raw_strand}" if batch else raw_strand
    )
    methylated = _counts(_pick(record, _METHYLATED_KEYS, "fully_methylated"), "fully_methylated")
    unmethylated = _counts(_pick(record, _UNMETHYLATED_KEYS, "fully_unmethylated"),
                           "fully_unmethylated")
    return {"id": control_id, "batch": batch, "strand": raw_strand, "key": strand,
            "fully_methylated": methylated, "fully_unmethylated": unmethylated}


def _normalise_controls(raw):
    """Accept a list of control records or a batch -> strand -> record mapping."""
    if isinstance(raw, list):
        return [_entry(record) for record in sequence(raw, "controls", minimum=1, maximum=1000)]
    mapping(raw, "controls")
    if not raw:
        raise DomainError("invalid_array", "controls must not be empty")
    entries = []
    for name, value in raw.items():
        key = label(name, "control key")
        if not isinstance(value, dict):
            raise DomainError("invalid_object", "Each control must be an object")
        if any(item in value for item in _METHYLATED_KEYS + _UNMETHYLATED_KEYS):
            record = dict(value)
            if ":" in key and "batch" not in record and "strand" not in record:
                batch, _, strand = key.partition(":")
                record["batch"] = batch
                record["strand"] = strand
            elif "strand" not in record:
                record["strand"] = key
            entries.append(_entry(record))
        else:
            for strand, inner in value.items():
                if not isinstance(inner, dict):
                    raise DomainError("invalid_object", "Each control must be an object")
                record = dict(inner)
                record.setdefault("batch", key)
                record.setdefault("strand", strand)
                entries.append(_entry(record))
    if not entries:
        raise DomainError("invalid_array", "controls must not be empty")
    return entries


def _select(batch, strand, controls, by_id, control_id, explicit):
    if explicit:
        if control_id not in by_id:
            raise DomainError("unknown_control", "A site names a control that is not in the registry")
        return by_id[control_id], None
    candidates = [
        control for control in controls
        if (batch is None or control["batch"] == batch)
        and (strand is None or control["key"] == strand)
    ]
    if len(candidates) == 1:
        return candidates[0], None
    if not candidates:
        return None, "missing_control"
    return None, "ambiguous_control"


def prepare(payload):
    fields(payload, ["sites"], ["confidence_z", "conversion_error", "controls"])
    if number(payload.get("conversion_error", 0), "conversion_error", minimum=0, maximum=1) != 0:
        raise DomainError("unsupported_model",
                          "conversion_error must be zero; conversion bias is estimated from controls")
    legacy = "controls" not in payload
    controls = []
    by_id = {}
    if not legacy:
        controls = _normalise_controls(payload["controls"])
        for control in controls:
            if control["id"] in by_id:
                raise DomainError("duplicate_control", "Control identifiers must be unique")
            by_id[control["id"]] = control
    sites = sequence(payload["sites"], "sites", minimum=1, maximum=10000)
    result = []
    seen = set()
    site_fields = ["region"] if legacy else ["region", "batch", "strand", "control", "control_id"]
    for site in sites:
        fields(site, ["id", "methylated", "coverage"], site_fields)
        name = label(site["id"], "site id")
        if name in seen:
            raise DomainError("duplicate_site", "Site identifiers must be unique")
        seen.add(name)
        total = integer(site["coverage"], "coverage", minimum=0, maximum=10**9)
        marked = integer(site["methylated"], "methylated", minimum=0, maximum=total)
        region = label(site.get("region", name), "region")
        batch = label(site["batch"], "site batch") if "batch" in site else None
        raw_strand = label(site["strand"], "site strand") if "strand" in site else None
        strand = canonical_strand(raw_strand) if raw_strand is not None else None
        explicit = "control" in site or "control_id" in site
        if legacy:
            control = None
            reason = None
        else:
            if "control" in site and "control_id" in site:
                raise DomainError("duplicate_field", "A site names its control more than once")
            control_id = label(site.get("control", site.get("control_id")), "site control") \
                if explicit else None
            control, reason = _select(batch, strand, controls, by_id, control_id, explicit)
        result.append({"id": name, "region": region, "methylated": marked, "coverage": total,
                       "batch": batch, "strand": raw_strand, "control": control,
                       "reason": reason})
    return {"legacy": legacy, "controls": controls, "sites": result}
