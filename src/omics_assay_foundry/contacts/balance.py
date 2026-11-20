import math

def _row_sums(normalized, weights):
    return {i: weights[i]*math.fsum(count*weights[j] for j,count in pairs)
            for i,pairs in normalized.items()}

def solve(matrix, tolerance, max_iterations):
    size = len(matrix)
    scale = max(max(row) for row in matrix)
    normalized = [[value/scale for value in row] for row in matrix]
    weights = [1.0]*size
    for iteration in range(1, max_iterations+1):
        row_sums = [weights[i]*math.fsum(value*weights[j] for j,value in enumerate(row))
                    for i,row in enumerate(normalized)]
        residual = max(abs(value-1) for value in row_sums)
        if residual <= tolerance:
            return [weight/math.sqrt(scale) for weight in weights], iteration, True
        weights = [weight/math.sqrt(value) for weight,value in zip(weights,row_sums)]
    return [weight/math.sqrt(scale) for weight in weights], max_iterations, False

def solve_sparse(adjacency, tolerance, max_iterations):
    """Balance a sparse positive support to unit marginals.

    adjacency maps each supported bin to its (other bin, positive count) pairs,
    with self loops listed once. Returns (iterations, residuals, failed):
    the iteration count, the per-bin balanced-marginal residuals of the final
    weights, and the bins whose row sums collapsed to zero or non-finite
    values. A nonempty failed set means the correction diverged: the support
    has no unit-marginal solution (for example tree-structured support), and
    the failed bins' components must be treated as unsolved. Bins without
    positive support are absent from the adjacency and have no solution by
    construction.
    """
    scale = max(count for pairs in adjacency.values() for _,count in pairs)
    normalized = {i: [(j, count/scale) for j,count in pairs]
                  for i,pairs in adjacency.items()}
    weights = {i: 1.0 for i in adjacency}
    for iteration in range(1, max_iterations+1):
        rows = _row_sums(normalized, weights)
        residuals = {i: abs(value-1) for i,value in rows.items()}
        if max(residuals.values()) <= tolerance:
            return iteration, residuals, set()
        failed = {i for i,value in rows.items()
                  if value == 0.0 or not math.isfinite(value)}
        if failed:
            return iteration, residuals, failed
        weights = {i: weights[i]/math.sqrt(rows[i]) for i in weights}
    rows = _row_sums(normalized, weights)
    return max_iterations, {i: abs(value-1) for i,value in rows.items()}, set()
