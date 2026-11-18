import pytest
from omics_assay_foundry.api import dispatch, ROUTES
from omics_assay_foundry.errors import DomainError

@pytest.mark.parametrize("route",sorted(ROUTES))
def test_unknown_options_are_not_silently_ignored(route,sample):
    payload=sample(route[1:]);payload["unknown_scientific_option"]=True
    with pytest.raises(DomainError,match="Unsupported fields"):
        dispatch(route,payload)

@pytest.mark.parametrize("route",sorted(ROUTES))
def test_nonobject_requests_are_rejected(route):
    with pytest.raises(DomainError):
        dispatch(route,[])

@pytest.mark.parametrize("route",sorted(ROUTES))
def test_documented_sample_produces_serializable_result(route,sample):
    import json
    json.dumps(dispatch(route,sample(route[1:])),allow_nan=False)
