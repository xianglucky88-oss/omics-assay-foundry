import math

PAIR_ENERGIES = {"AU": -2.0, "UA": -2.0, "GC": -3.0, "CG": -3.0, "GU": -1.0, "UG": -1.0}
MODEL = "pair-additive-dimensionless-v1"

def energy(sequence, structure):
    return math.fsum(PAIR_ENERGIES[sequence[i]+sequence[j]] for i,j in structure)
