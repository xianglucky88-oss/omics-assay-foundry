from ..validation import fields, sequence, number
from ..errors import DomainError

def prepare(payload):
    fields(payload, ["matrix"], ["tolerance", "max_iterations"])
    rows = sequence(payload["matrix"], "matrix", minimum=1, maximum=128)
    size = len(rows)
    matrix = []
    for row in rows:
        matrix.append([number(x, "contact", minimum=1e-12, maximum=1e12)
                       for x in sequence(row, "row", minimum=size, maximum=size)])
    if any(abs(matrix[i][j]-matrix[j][i]) > 1e-12*max(matrix[i][j], matrix[j][i])
           for i in range(size) for j in range(i)):
        raise DomainError("asymmetric_matrix", "Contact matrix must be symmetric")
    return matrix
