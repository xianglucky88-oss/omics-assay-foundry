"""Legacy fixed-length, unbiased-compatibility estimator.

This keeps the original ``/transcripts`` model intact for requests that use
only ``effective_lengths`` and ``fragments``.  Fragments carry unit relative
propensity, so allocation is proportional to molecule abundance alone and the
M-step divides by the fixed effective length.  The extended estimator in
``model`` reduces to this behaviour when no measurement field is supplied.
"""
from .model import estimate as _estimate

def estimate(lengths, compatibility, max_iterations, tolerance):
    """Return ``(abundance, counts, iterations, converged, maximum_change)``."""
    rows = [(members, (1.0,)*len(members), 1) for members in compatibility]
    return _estimate(lengths, rows, max_iterations, tolerance)
