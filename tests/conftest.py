import pytest

from escalation_discovery import config as C


@pytest.fixture
def tmp_outputs(tmp_path, monkeypatch):
    """Send results/ and data/derived/ of a test to a temporary folder."""
    real = C._settings()

    def fake():
        return dict(real, results=str(tmp_path / "results"), derived=str(tmp_path / "derived"))

    monkeypatch.setattr(C, "_settings", fake)
    return tmp_path
