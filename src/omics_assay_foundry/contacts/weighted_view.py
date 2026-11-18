import math
from .support import prepare
from .balance import solve
from ..validation import integer, number
from ..errors import DomainError

def process(payload):
    matrix = prepare(payload)
    tolerance = number(payload.get("tolerance", 1e-9), "tolerance", minimum=1e-12, maximum=1e-2)
    limit = integer(payload.get("max_iterations", 2000), "max_iterations", minimum=1, maximum=10000)
    if len(matrix)**2 * limit > 20000000:
        raise DomainError("work_limit", "Matrix-iteration budget exceeds twenty million")
    weights, used, converged = solve(matrix, tolerance, limit)
    balanced = [[weights[i]*value*weights[j] for j,value in enumerate(row)] for i,row in enumerate(matrix)]
    residual = max(abs(math.fsum(row)-1) for row in balanced)
    return {"model": "positive-symmetric-unit-marginals", "weights": weights, "matrix": balanced,
            "residual": residual, "converged": converged or residual<=tolerance, "iterations": used}
