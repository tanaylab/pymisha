"""Tests for gvtrack_iterator_2d."""

import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


@pytest.fixture(autouse=True)
def _clean_vtracks():
    yield
    pm.gvtrack_clear()


class TestGvtrackIterator2d:
    """Tests for gvtrack_iterator_2d."""

    def test_sets_2d_iterator_defaults(self):
        """Setting 2D iterator with defaults stores zero shifts."""
        pm.gvtrack_create("vt1", "dense_track")
        pm.gvtrack_iterator_2d("vt1")
        info = pm.gvtrack_info("vt1")
        assert info.get("itr_type") == "2d"
        assert info.get("sshift1") == 0
        assert info.get("eshift1") == 0
        assert info.get("sshift2") == 0
        assert info.get("eshift2") == 0

    def test_sets_custom_shifts(self):
        """Setting 2D iterator with custom shifts stores them."""
        pm.gvtrack_create("vt2", "dense_track")
        pm.gvtrack_iterator_2d("vt2", sshift1=-100, eshift1=100, sshift2=-200, eshift2=200)
        info = pm.gvtrack_info("vt2")
        assert info["itr_type"] == "2d"
        assert info["sshift1"] == -100
        assert info["eshift1"] == 100
        assert info["sshift2"] == -200
        assert info["eshift2"] == 200

    def test_nonexistent_vtrack_raises(self):
        """Setting iterator on nonexistent vtrack raises KeyError."""
        with pytest.raises(KeyError):
            pm.gvtrack_iterator_2d("nonexistent")

    def test_overwrites_previous_iterator(self):
        """Setting 2D iterator replaces any previous 1D iterator."""
        pm.gvtrack_create("vt3", "dense_track")
        pm.gvtrack_iterator("vt3", sshift=-50, eshift=50)
        info1 = pm.gvtrack_info("vt3")
        assert info1.get("sshift") == -50

        pm.gvtrack_iterator_2d("vt3", sshift1=10, eshift2=20)
        info2 = pm.gvtrack_info("vt3")
        assert info2["itr_type"] == "2d"
        assert info2["sshift1"] == 10
        assert info2["eshift2"] == 20
