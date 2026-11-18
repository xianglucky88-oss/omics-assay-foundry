import math
from .errors import DomainError

def number(value, name, *, minimum=None, maximum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DomainError("invalid_number", f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise DomainError("invalid_number", f"{name} is outside the numeric range") from None
    if not math.isfinite(result):
        raise DomainError("invalid_number", f"{name} must be finite")
    if positive and result <= 0:
        raise DomainError("invalid_number", f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise DomainError("invalid_number", f"{name} is below its supported minimum")
    if maximum is not None and result > maximum:
        raise DomainError("invalid_number", f"{name} exceeds its supported maximum")
    return result

def integer(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        raise DomainError("invalid_integer", f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise DomainError("invalid_integer", f"{name} is below its supported minimum")
    if maximum is not None and value > maximum:
        raise DomainError("invalid_integer", f"{name} exceeds its supported maximum")
    return value

def mapping(value, name="request"):
    if not isinstance(value, dict):
        raise DomainError("invalid_object", f"{name} must be an object")
    if not all(isinstance(k, str) for k in value):
        raise DomainError("invalid_key", f"{name} keys must be strings")
    return value

def sequence(value, name, *, minimum=0, maximum=100000):
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise DomainError("invalid_array", f"{name} has an invalid type or length")
    return value

def fields(value, required, optional=()):
    mapping(value)
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing:
        raise DomainError("missing_field", "Missing fields: " + ", ".join(sorted(missing)))
    if unknown:
        raise DomainError("unsupported_field", "Unsupported fields: " + ", ".join(sorted(unknown)))

def label(value, name):
    if not isinstance(value, str) or not value or len(value) > 128:
        raise DomainError("invalid_label", f"{name} must be a nonempty string of at most 128 characters")
    return value
