import math
from ..errors import DomainError

def estimate(lengths, compatibility, max_iterations, tolerance):
    size = len(lengths)
    abundance = [1/size]*size
    for iteration in range(1, max_iterations+1):
        counts = [0.0]*size
        for members in compatibility:
            denominator = math.fsum(abundance[i] for i in members)
            if denominator <= 0:
                raise DomainError("numeric_failure", "A fragment has no representable supporting abundance")
            for i in members:
                counts[i] += abundance[i]/denominator
        rates = [count/length for count, length in zip(counts, lengths)]
        total = math.fsum(rates)
        updated = [rate/total for rate in rates]
        delta = max(abs(a-b) for a, b in zip(abundance, updated))
        abundance = updated
        if delta <= tolerance:
            return abundance, counts, iteration, True, delta
    return abundance, counts, max_iterations, False, delta
