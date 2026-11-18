import math

def site_log_likelihood(tree, sequences, site):
    bases = "ACGT"
    def visit(node):
        if "name" in node:
            return [float(base == sequences[node["name"]][site]) for base in bases], 0.0
        values = [1.0]*4
        log_scale = 0.0
        for child in node["children"]:
            partial, scale = visit(child)
            log_scale += scale
            exponent = -4*child["length"]/3
            different = -math.expm1(exponent)/4
            same = 1-3*different
            for i in range(4):
                values[i] *= same*partial[i]+different*math.fsum(partial[j] for j in range(4) if j != i)
            maximum = max(values)
            if maximum == 0:
                return [0.0]*4, 0.0
            values = [value/maximum for value in values]
            log_scale += math.log(maximum)
        return values, log_scale
    values, scale = visit(tree)
    root = math.fsum(values)/4
    return None if root == 0 else math.log(root)+scale
