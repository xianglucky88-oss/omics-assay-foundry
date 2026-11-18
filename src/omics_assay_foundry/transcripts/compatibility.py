from ..validation import fields, mapping, number, sequence, label
from ..errors import DomainError

def prepare(payload):
    fields(payload, ["effective_lengths", "fragments"], ["max_iterations", "tolerance"])
    lengths = mapping(payload["effective_lengths"], "effective_lengths")
    if not 1 <= len(lengths) <= 1000:
        raise DomainError("size_limit", "Use between one and 1000 transcripts")
    names = sorted(label(name, "transcript") for name in lengths)
    values = [number(lengths[name], "effective_length", minimum=1, maximum=1e9) for name in names]
    index = {name: i for i, name in enumerate(names)}
    fragments = sequence(payload["fragments"], "fragments", minimum=1, maximum=50000)
    compatibility = []
    for fragment in fragments:
        members = sequence(fragment, "compatible transcripts", minimum=1, maximum=len(names))
        if any(not isinstance(name, str) or name not in index for name in members) or len(set(members)) != len(members):
            raise DomainError("invalid_compatibility", "Each fragment must name distinct known transcripts")
        compatibility.append(tuple(sorted(index[name] for name in members)))
    return names, values, compatibility
