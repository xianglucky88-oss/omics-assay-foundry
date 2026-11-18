from .compatibility import prepare
from .likelihood import estimate
from ..validation import integer, number
from ..errors import DomainError

def process(payload):
    names, lengths, compatibility = prepare(payload)
    iterations = integer(payload.get("max_iterations", 1000), "max_iterations", minimum=1, maximum=5000)
    tolerance = number(payload.get("tolerance", 1e-10), "tolerance", minimum=1e-14, maximum=1e-2)
    if len(compatibility)*iterations > 5000000:
        raise DomainError("work_limit", "Fragment-iteration product exceeds five million")
    abundance, counts, used, converged, delta = estimate(lengths, compatibility, iterations, tolerance)
    signatures = {}
    for i, name in enumerate(names):
        signature = tuple(k for k, members in enumerate(compatibility) if i in members)
        signatures.setdefault(signature, []).append(name)
    unidentifiable = [members for present, members in signatures.items() if present and len(members) > 1]
    return {"model": "fixed-effective-length-unbiased-fragments", "converged": converged,
            "iterations": used, "maximum_change": delta, "indistinguishable_groups": unidentifiable,
            "transcripts": [{"id": name, "molecule_fraction": abundance[i], "tpm": abundance[i]*1e6,
                             "expected_fragments": counts[i]} for i, name in enumerate(names)]}
