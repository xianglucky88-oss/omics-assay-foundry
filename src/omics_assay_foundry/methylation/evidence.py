import math

def wilson(successes, total, z):
    if total == 0:
        return {"fraction": None, "interval": [0.0, 1.0], "status": "no_coverage"}
    p = successes/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return {"fraction": p, "interval": [max(0.0, center-radius), min(1.0, center+radius)], "status": "ok"}
