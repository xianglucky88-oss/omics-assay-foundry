import math

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
