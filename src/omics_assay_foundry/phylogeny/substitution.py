"""Substitution models and discrete rate mixtures for evidence comparison.

Two substitution families are supported:

* ``JC69``: equal frequencies and equal exchangeabilities.  This is exactly the
  model used by the original single-tree endpoint.
* ``GTR``: a user-supplied symmetric exchangeability matrix and stationary
  frequencies.  The rate matrix is normalized to an expected rate of one, so a
  branch length is always "expected substitutions per site" and candidates
  computed with different parameters remain on the same explicit scale.

A discrete rate mixture multiplies the normalized rate matrix by category rates
whose weighted mean is one.  Categories can be supplied directly or generated
with Yang's equal-probability discrete-gamma construction.

Only the Python standard library is used: a cyclic Jacobi eigensolver handles
the 4x4 reversible rate matrix and an incomplete-gamma implementation supplies
gamma quantiles.
"""
import math

from ..errors import DomainError
from ..validation import fields, mapping, number, integer, sequence

BASES = "ACGT"
MAX_RATE_CATEGORIES = 16
MODEL_KEYS = ("substitution", "exchangeabilities", "frequencies", "rate_categories")


# ---------------------------------------------------------------------------
# Symmetric eigensolver (4x4)
# ---------------------------------------------------------------------------

def symmetric_eigen(matrix):
    """Return ``(eigenvalues, eigenvectors)`` for a small symmetric matrix.

    ``eigenvectors[i][k]`` is component ``i`` of eigenvector ``k``.  The cyclic
    Jacobi iteration is deterministic and accurate to machine precision for the
    4x4 matrices used here.
    """
    size = len(matrix)
    a = [list(row) for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    for _ in range(100):
        off = math.fsum(a[i][j] ** 2 for i in range(size) for j in range(i + 1, size))
        if off <= 1e-32:
            break
        for p in range(size - 1):
            for q in range(p + 1, size):
                apq = a[p][q]
                if abs(apq) <= 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * apq)
                if theta == 0.0:
                    t = 1.0
                else:
                    t = math.copysign(1.0, theta) / (abs(theta) + math.hypot(theta, 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                app, aqq = a[p][p], a[q][q]
                a[p][p] = app - t * apq
                a[q][q] = aqq + t * apq
                a[p][q] = a[q][p] = 0.0
                for i in range(size):
                    if i != p and i != q:
                        aip, aiq = a[i][p], a[i][q]
                        a[i][p] = a[p][i] = c * aip - s * aiq
                        a[i][q] = a[q][i] = s * aip + c * aiq
                for i in range(size):
                    vip, viq = v[i][p], v[i][q]
                    v[i][p] = c * vip - s * viq
                    v[i][q] = s * vip + c * viq
    return [a[i][i] for i in range(size)], v


# ---------------------------------------------------------------------------
# Incomplete gamma and discrete gamma rates
# ---------------------------------------------------------------------------

_EPS = 1e-15
_FP_MIN = 1e-300


def regularized_gamma_lower(shape, x):
    """Regularized lower incomplete gamma ``P(shape, x)``."""
    if x <= 0.0:
        return 0.0
    if math.isinf(x):
        return 1.0
    log_prefix = shape * math.log(x) - x - math.lgamma(shape)
    if x < shape + 1.0:
        term = 1.0 / shape
        total = term
        n = shape
        for _ in range(100000):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) <= abs(total) * _EPS:
                break
        return min(1.0, total * math.exp(log_prefix))
    # Lentz continued fraction for Q(shape, x).
    b = x + 1.0 - shape
    c = 1.0 / _FP_MIN
    d = 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - shape)
        b += 2.0
        d = an * d + b
        if abs(d) < _FP_MIN:
            d = _FP_MIN
        c = b + an / c
        if abs(c) < _FP_MIN:
            c = _FP_MIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= _EPS:
            break
    return max(0.0, min(1.0, 1.0 - h * math.exp(log_prefix)))


def gamma_quantile(shape, probability):
    """Quantile of the unit-mean Gamma distribution with given ``shape``.

    The distribution has rate equal to ``shape``, so its mean is one and its
    variance is ``1/shape``.  Working in the scale parameterization internally
    keeps the incomplete-gamma routine simple.
    """
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return math.inf
    # ``regularized_gamma_lower(shape, x)`` is the CDF of Gamma(shape, rate=1),
    # so solve for that first, then rescale the unit-rate quantile to rate
    # ``shape`` (divide by shape).
    low, high = -745.0, 710.0
    for _ in range(300):
        mid = 0.5 * (low + high)
        value = regularized_gamma_lower(shape, math.exp(mid))
        if value < probability:
            low = mid
        else:
            high = mid
    return math.exp(0.5 * (low + high)) / shape


def discrete_gamma(shape, categories):
    """Equal-probability discrete gamma with mean rate one (Yang 1994)."""
    boundaries = [gamma_quantile(shape, k / categories) for k in range(1, categories)]
    # ``shape`` here is the rate; the argument to P(shape+1, .) is rate * boundary.
    cumulative = [regularized_gamma_lower(shape + 1.0, shape * q) for q in boundaries]
    cumulative = [0.0] + cumulative + [1.0]
    rates = []
    for k in range(categories):
        value = categories * (cumulative[k + 1] - cumulative[k])
        rates.append(max(0.0, value))
    weights = [1.0 / categories] * categories
    mean = math.fsum(w * r for w, r in zip(weights, rates))
    if not math.isfinite(mean) or mean <= 0.0:
        raise DomainError("invalid_model", "Discrete gamma categories have a nonpositive mean rate")
    rates = [r / mean for r in rates]
    return rates, weights


# ---------------------------------------------------------------------------
# Model validation and compilation
# ---------------------------------------------------------------------------

def _parse_exchangeabilities(value):
    rows = sequence(value, "exchangeabilities", minimum=4, maximum=4)
    matrix = []
    for i, row in enumerate(rows):
        row = sequence(row, "exchangeability row", minimum=4, maximum=4)
        matrix.append([number(x, "exchangeability", minimum=0, maximum=1e12) for x in row])
    for i in range(4):
        if abs(matrix[i][i]) > 1e-12:
            raise DomainError("invalid_model", "Exchangeability matrix diagonal must be zero")
        for j in range(i + 1, 4):
            if not math.isclose(matrix[i][j], matrix[j][i], rel_tol=1e-9, abs_tol=1e-12):
                raise DomainError("invalid_model", "Exchangeability matrix must be symmetric")
            if matrix[i][j] <= 0.0:
                raise DomainError("invalid_model", "Off-diagonal exchangeabilities must be positive")
    return matrix


def _parse_frequencies(value):
    values = sequence(value, "frequencies", minimum=4, maximum=4)
    frequencies = [number(x, "frequency", positive=True, maximum=1e12) for x in values]
    total = math.fsum(frequencies)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise DomainError("invalid_model", "Frequencies must sum to one")
    return tuple(x / total for x in frequencies)


def _parse_rate_categories(value):
    fields(value, [], ["rates", "weights", "gamma_shape", "categories"])
    has_gamma = "gamma_shape" in value or "categories" in value
    has_rates = "rates" in value or "weights" in value
    if has_gamma and has_rates:
        raise DomainError("invalid_model", "Use either explicit rates/weights or gamma_shape/categories")
    if has_gamma:
        if "gamma_shape" not in value or "categories" not in value:
            raise DomainError("missing_field", "gamma_shape and categories must be given together")
        shape = number(value["gamma_shape"], "gamma_shape", minimum=1e-3, maximum=1e4)
        count = integer(value["categories"], "categories", minimum=1, maximum=MAX_RATE_CATEGORIES)
        if count == 1:
            return [1.0], [1.0], "gamma"
        return (*discrete_gamma(shape, count), "gamma")
    if "rates" not in value or "weights" not in value:
        raise DomainError("missing_field", "rates and weights must be given together")
    rates = sequence(value["rates"], "rates", minimum=1, maximum=MAX_RATE_CATEGORIES)
    weights = sequence(value["weights"], "weights", minimum=1, maximum=MAX_RATE_CATEGORIES)
    if len(rates) != len(weights):
        raise DomainError("invalid_model", "rates and weights must have the same length")
    parsed_rates = [number(x, "rate", minimum=0, maximum=1e12) for x in rates]
    parsed_weights = [number(x, "weight", positive=True, maximum=1e12) for x in weights]
    weight_total = math.fsum(parsed_weights)
    if not math.isclose(weight_total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise DomainError("invalid_model", "Rate weights must sum to one")
    parsed_weights = [w / weight_total for w in parsed_weights]
    mean = math.fsum(w * r for w, r in zip(parsed_weights, parsed_rates))
    if not math.isfinite(mean) or mean <= 0.0:
        raise DomainError("invalid_model", "Rate categories must have a positive mean rate")
    if not math.isclose(mean, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise DomainError(
            "invalid_model",
            "Explicit rate categories must have weighted mean rate one, matching the "
            "expected-substitutions-per-site branch length scale; rescale them explicitly",
        )
    return parsed_rates, parsed_weights, "explicit"


def compile_model(spec):
    """Validate a model block and return its normalized representation."""
    mapping(spec, "model")
    fields(spec, [], MODEL_KEYS)
    substitution = spec.get("substitution", "JC69")
    if substitution not in ("JC69", "GTR"):
        raise DomainError("unsupported_model", "substitution must be 'JC69' or 'GTR'")
    if substitution == "JC69":
        if "exchangeabilities" in spec:
            raise DomainError("unsupported_parameter", "JC69 does not accept exchangeabilities; use GTR")
        if "frequencies" in spec:
            raise DomainError("unsupported_parameter", "JC69 uses uniform frequencies; use GTR")
        frequencies = (0.25, 0.25, 0.25, 0.25)
        exchangeabilities = [[0.0, 1.0, 1.0, 1.0],
                             [1.0, 0.0, 1.0, 1.0],
                             [1.0, 1.0, 0.0, 1.0],
                             [1.0, 1.0, 1.0, 0.0]]
    else:
        if "exchangeabilities" not in spec:
            raise DomainError("missing_field", "GTR requires exchangeabilities")
        if "frequencies" not in spec:
            raise DomainError("missing_field", "GTR requires frequencies; a uniform default would silently change the model")
        exchangeabilities = _parse_exchangeabilities(spec["exchangeabilities"])
        frequencies = _parse_frequencies(spec["frequencies"])

    if "rate_categories" in spec:
        rates, weights, source = _parse_rate_categories(spec["rate_categories"])
    else:
        rates, weights, source = [1.0], [1.0], "default"

    # Expected rate of the unscaled process: sum_{i != j} pi_i pi_j s_ij.
    mean_rate = math.fsum(frequencies[i] * frequencies[j] * exchangeabilities[i][j]
                          for i in range(4) for j in range(4) if i != j)
    if not math.isfinite(mean_rate) or mean_rate <= 0.0:
        raise DomainError("invalid_model", "Substitution model has a nonpositive expected rate")

    # B = D^(1/2) Q D^(-1/2) is symmetric for a reversible model.
    b = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            if i == j:
                b[i][i] = -math.fsum(exchangeabilities[i][k] * frequencies[k] for k in range(4) if k != i) / mean_rate
            else:
                b[i][j] = exchangeabilities[i][j] * math.sqrt(frequencies[i] * frequencies[j]) / mean_rate
    eigenvalues, eigenvectors = symmetric_eigen(b)
    # The stationary eigenvalue is zero; never allow a tiny positive round-off
    # to grow with branch length.
    eigenvalues = [0.0 if abs(value) <= 1e-12 else min(value, 0.0) for value in eigenvalues]
    return {
        "substitution": substitution,
        "frequencies": frequencies,
        "exchangeabilities": exchangeabilities,
        "rates": tuple(rates),
        "weights": tuple(weights),
        "rate_source": source,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "expected_rate": 1.0,
    }


def _close(a, b, rel_tol=1e-9, abs_tol=1e-12):
    return all(math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol) for x, y in zip(a, b))


def model_parameter(model, name):
    """Extract a comparable value for a declared shared parameter."""
    if name == "substitution":
        return model["substitution"]
    if name == "frequencies":
        return model["frequencies"]
    if name == "exchangeabilities":
        flat = tuple(model["exchangeabilities"][i][j] for i in range(4) for j in range(4) if i < j)
        return flat
    if name == "rate_categories":
        return (model["rates"], model["weights"])
    raise DomainError("shared_parameter", f"Unknown shared parameter {name!r}")


def parameters_equal(model_a, model_b, name):
    a = model_parameter(model_a, name)
    b = model_parameter(model_b, name)
    if isinstance(a, str):
        return a == b
    if name == "rate_categories":
        return _close(a[0], b[0]) and _close(a[1], b[1])
    return _close(a, b)


def transition_matrix(model, branch_length, category_rate, report=None):
    """Transition probabilities ``exp(Q * rate * t)`` as a list of rows.

    ``Q`` is already normalized to an expected rate of one, so the product
    ``rate * t`` is expressed in expected substitutions per site.  Computed
    values below zero or above one by round-off are clamped to the probability
    range.  When ``report`` is a dict it receives the count of clamped entries
    and the most negative raw entry, so the correction stays visible.
    """
    if branch_length == 0.0 or category_rate == 0.0:
        return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    t = branch_length * category_rate
    frequencies = model["frequencies"]
    eigenvalues = model["eigenvalues"]
    vectors = model["eigenvectors"]
    decay = [math.exp(value * t) for value in eigenvalues]
    matrix = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            total = math.fsum(vectors[i][k] * vectors[j][k] * decay[k] for k in range(4))
            value = total * math.sqrt(frequencies[j] / frequencies[i])
            if value < 0.0:
                if report is not None:
                    report["clamped_entries"] = report.get("clamped_entries", 0) + 1
                    report["most_negative_raw"] = min(report.get("most_negative_raw", 0.0), value)
                value = 0.0
            elif value > 1.0:
                if report is not None:
                    report["clamped_entries"] = report.get("clamped_entries", 0) + 1
                    report["most_positive_raw"] = max(report.get("most_positive_raw", 1.0), value)
                value = 1.0
            matrix[i][j] = value
    return matrix


def describe_model(model):
    """Serializable description used in results."""
    return {
        "substitution": model["substitution"],
        "frequencies": list(model["frequencies"]),
        "exchangeabilities": ([list(row) for row in model["exchangeabilities"]]
                              if model["substitution"] == "GTR" else None),
        "rate_categories": {
            "source": model["rate_source"],
            "rates": list(model["rates"]),
            "weights": list(model["weights"]),
        },
        "branch_length_unit": "expected substitutions per site",
        "expected_rate": 1.0,
    }
