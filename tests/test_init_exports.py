import pymisha as pm
from pymisha import _shared


def test_runtime_state_exports_are_live(monkeypatch):
    sentinel_vtracks = {"vt": {"expr": "dense_track"}}
    monkeypatch.setattr(_shared, "_GROOT", "/tmp/live-groot")
    monkeypatch.setattr(_shared, "_UROOT", "/tmp/live-uroot")
    monkeypatch.setattr(_shared, "_VTRACKS", sentinel_vtracks)

    assert pm._GROOT == "/tmp/live-groot"
    assert pm._UROOT == "/tmp/live-uroot"
    assert pm._VTRACKS is sentinel_vtracks
