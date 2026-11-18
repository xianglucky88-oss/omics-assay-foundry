from ..validation import fields, integer, label, sequence, number
from ..errors import DomainError

def prepare(payload):
    fields(payload, ["sites"], ["confidence_z", "conversion_error"])
    if number(payload.get("conversion_error", 0), "conversion_error", minimum=0, maximum=1) != 0:
        raise DomainError("unsupported_model", "Only the zero-conversion-error model is supported")
    sites = sequence(payload["sites"], "sites", minimum=1, maximum=10000)
    result = []
    seen = set()
    for site in sites:
        fields(site, ["id", "methylated", "coverage"], ["region"])
        name = label(site["id"], "site id")
        if name in seen:
            raise DomainError("duplicate_site", "Site identifiers must be unique")
        seen.add(name)
        total = integer(site["coverage"], "coverage", minimum=0, maximum=10**9)
        marked = integer(site["methylated"], "methylated", minimum=0, maximum=total)
        region = label(site.get("region", name), "region")
        result.append((name, marked, total, region))
    return result
