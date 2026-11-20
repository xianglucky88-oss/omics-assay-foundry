"""IUPAC nucleotide ambiguity decoding for evidence comparison.

The legacy single-tree endpoint deliberately accepts only complete ``ACGT``
sequences.  Evidence comparison must instead keep every observed column in the
observation population, so ambiguous characters are expanded to the set of
states they admit and missing data are expanded to all four states.  A leaf
with state set ``S`` contributes likelihood 1 for each state in ``S`` and 0
otherwise; this is a sum over possible states, not an average or a silent
deletion of the column.
"""
from ..errors import DomainError
from ..validation import label, mapping

BASES = "ACGT"
BASE_BITS = {"A": 1, "C": 2, "G": 4, "T": 8}
FULL = 15

# IUPAC nucleotide codes.  ``-`` and ``?`` denote missing data (any state).
CODES = {
    "A": BASE_BITS["A"], "C": BASE_BITS["C"], "G": BASE_BITS["G"], "T": BASE_BITS["T"],
    "R": BASE_BITS["A"] | BASE_BITS["G"],
    "Y": BASE_BITS["C"] | BASE_BITS["T"],
    "S": BASE_BITS["G"] | BASE_BITS["C"],
    "W": BASE_BITS["A"] | BASE_BITS["T"],
    "K": BASE_BITS["G"] | BASE_BITS["T"],
    "M": BASE_BITS["A"] | BASE_BITS["C"],
    "B": BASE_BITS["C"] | BASE_BITS["G"] | BASE_BITS["T"],
    "D": BASE_BITS["A"] | BASE_BITS["G"] | BASE_BITS["T"],
    "H": BASE_BITS["A"] | BASE_BITS["C"] | BASE_BITS["T"],
    "V": BASE_BITS["A"] | BASE_BITS["C"] | BASE_BITS["G"],
    "N": FULL,
    "-": FULL,
    "?": FULL,
}
AMBIGUOUS_CODES = set("RYSWKMBDHVN")
MISSING_CODES = set("-?")

MAX_LEAVES = 32
MAX_SITES = 2000


def decode_character(character, name):
    code = CODES.get(character)
    if code is None:
        raise DomainError(
            "unsupported_sequence",
            f"Sequence {name!r} contains unsupported character {character!r}; "
            "use uppercase IUPAC nucleotide codes, '-' or '?'",
        )
    return code


def prepare_aligned(payload_sequences):
    """Validate aligned sequences and return masks plus ambiguity statistics.

    Returns ``(taxa, masks, n_sites, stats)`` where ``masks[taxon]`` is a list
    of 4-bit state masks, one per column, and ``stats`` counts characters by
    kind.  Every observed column is retained for every taxon.
    """
    sequences = mapping(payload_sequences, "sequences")
    if not 2 <= len(sequences) <= MAX_LEAVES:
        raise DomainError("size_limit", "Use two to 32 leaves")
    masks = {}
    lengths = set()
    ambiguous_characters = 0
    missing_characters = 0
    fully_ambiguous_characters = 0
    for name, sequence in sequences.items():
        label(name, "leaf")
        if not isinstance(sequence, str) or not 1 <= len(sequence) <= MAX_SITES:
            raise DomainError(
                "unsupported_sequence",
                "Use sequences of length 1 to 2000",
            )
        lengths.add(len(sequence))
        row = []
        for character in sequence:
            code = decode_character(character, name)
            if character in MISSING_CODES:
                missing_characters += 1
            elif character == "N":
                fully_ambiguous_characters += 1
            elif character in AMBIGUOUS_CODES:
                ambiguous_characters += 1
            row.append(code)
        masks[name] = row
    if len(lengths) != 1:
        raise DomainError("unaligned_sequences", "All sequences must have the same length")
    n_sites = lengths.pop()
    # The same taxon order everywhere makes results independent of JSON order.
    taxa = sorted(masks)
    stats = {
        "ambiguous_characters": ambiguous_characters + fully_ambiguous_characters,
        "missing_characters": missing_characters,
        "fully_ambiguous_characters": fully_ambiguous_characters,
    }
    return taxa, masks, n_sites, stats
