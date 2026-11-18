import json
from pathlib import Path
import pytest

@pytest.fixture
def sample():
    root=Path(__file__).resolve().parents[1]/"examples"
    def load(name):
        return json.loads((root/(name+".json")).read_text())
    return load
