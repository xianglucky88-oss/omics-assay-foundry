import math
from .feasible_ensemble import prepare, enumerate_structures
from .energy_contract import energy, MODEL
from ..validation import number
from ..errors import DomainError

def process(payload):
    seq, loop, required, forbidden = prepare(payload)
    beta = number(payload.get("beta",1.0), "beta", minimum=0, maximum=100)
    structures = enumerate_structures(seq,loop,required,forbidden)
    if not structures:
        raise DomainError("infeasible_constraints", "No structure satisfies the declared constraints")
    logs = [-beta*energy(seq,s) for s in structures]
    maximum = max(logs)
    weights = [math.exp(value-maximum) for value in logs]
    total = math.fsum(weights)
    marginal = [[0.0]*len(seq) for _ in seq]
    for structure,weight in zip(structures,weights):
        for i,j in structure:
            marginal[i][j] += weight/total
            marginal[j][i] += weight/total
    return {"model": MODEL, "beta": beta, "minimum_loop": loop,
            "structure_count": len(structures), "log_partition": maximum+math.log(total),
            "pair_probabilities": marginal,
            "unpaired_probabilities": [max(0,1-math.fsum(row)) for row in marginal]}
