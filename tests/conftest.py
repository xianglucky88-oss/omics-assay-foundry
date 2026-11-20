import json
from pathlib import Path
import pytest

@pytest.fixture
def sample():
    root=Path(__file__).resolve().parents[1]/"examples"
    aliases={"phylogeny/evidence":"phylogeny-evidence"}
    def load(name):
        return json.loads((root/(aliases.get(name,name)+".json")).read_text())
    return load
