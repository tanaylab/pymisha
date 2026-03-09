"""Tests for gintervals_attr_get/set/export/import."""

import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _init_db():
    pm.gdb_init_examples()


def _create_iset(name):
    """Create a temporary interval set for testing."""
    pm.gintervals_rm(name, force=True)
    pm.gintervals_save(pm.gintervals(["1", "2"]), name)
    return name


@pytest.fixture()
def iset(request):
    name = f"temp.iattr_{request.node.name}"
    _create_iset(name)
    yield name
    pm.gintervals_rm(name, force=True)


@pytest.fixture()
def iset_pair(request):
    n1 = f"temp.iattr_{request.node.name}_1"
    n2 = f"temp.iattr_{request.node.name}_2"
    _create_iset(n1)
    _create_iset(n2)
    yield n1, n2
    pm.gintervals_rm(n1, force=True)
    pm.gintervals_rm(n2, force=True)


# ---- 1. Basic get/set ----


class TestBasicGetSet:
    def test_set_and_get(self, iset):
        pm.gintervals_attr_set(iset, "myattr", "hello")
        assert pm.gintervals_attr_get(iset, "myattr") == "hello"

    def test_set_empty_removes(self, iset):
        pm.gintervals_attr_set(iset, "myattr", "hello")
        assert pm.gintervals_attr_get(iset, "myattr") == "hello"
        pm.gintervals_attr_set(iset, "myattr", "")
        assert pm.gintervals_attr_get(iset, "myattr") == ""

    def test_get_nonexistent_returns_empty(self, iset):
        assert pm.gintervals_attr_get(iset, "no_such_attr") == ""


# ---- 2. Multiple attributes ----


class TestMultipleAttrs:
    def test_multiple_attrs(self, iset):
        pm.gintervals_attr_set(iset, "attr_a", "value_a")
        pm.gintervals_attr_set(iset, "attr_b", "value_b")
        pm.gintervals_attr_set(iset, "attr_c", "value_c")
        assert pm.gintervals_attr_get(iset, "attr_a") == "value_a"
        assert pm.gintervals_attr_get(iset, "attr_b") == "value_b"
        assert pm.gintervals_attr_get(iset, "attr_c") == "value_c"

    def test_overwrite(self, iset):
        pm.gintervals_attr_set(iset, "myattr", "original")
        assert pm.gintervals_attr_get(iset, "myattr") == "original"
        pm.gintervals_attr_set(iset, "myattr", "updated")
        assert pm.gintervals_attr_get(iset, "myattr") == "updated"


# ---- 3. Export ----


class TestExport:
    def test_export_specific_sets(self, iset_pair):
        iset1, iset2 = iset_pair
        pm.gintervals_attr_set(iset1, "color", "red")
        pm.gintervals_attr_set(iset1, "size", "large")
        pm.gintervals_attr_set(iset2, "color", "blue")

        r = pm.gintervals_attr_export([iset1, iset2])
        assert isinstance(r, pd.DataFrame)
        assert iset1 in r.index
        assert iset2 in r.index
        assert "color" in r.columns
        assert "size" in r.columns
        assert r.loc[iset1, "color"] == "red"
        assert r.loc[iset1, "size"] == "large"
        assert r.loc[iset2, "color"] == "blue"
        assert r.loc[iset2, "size"] == ""

    def test_export_specific_attrs(self, iset):
        pm.gintervals_attr_set(iset, "color", "red")
        pm.gintervals_attr_set(iset, "size", "large")
        pm.gintervals_attr_set(iset, "weight", "heavy")

        r = pm.gintervals_attr_export(iset, attrs=["color", "weight"])
        assert r.shape[1] == 2
        assert "color" in r.columns
        assert "weight" in r.columns
        assert "size" not in r.columns
        assert r.loc[iset, "color"] == "red"
        assert r.loc[iset, "weight"] == "heavy"

    def test_export_nonexistent_errors(self):
        with pytest.raises(ValueError, match="does not exist"):
            pm.gintervals_attr_export("no_such_interval_set_xyz")

    def test_export_no_attrs_returns_empty_df(self, iset):
        r = pm.gintervals_attr_export(iset)
        assert isinstance(r, pd.DataFrame)
        assert len(r) == 1
        assert r.shape[1] == 0
        assert r.index[0] == iset

    def test_export_all_sets(self, iset):
        pm.gintervals_attr_set(iset, "tag", "test")
        r = pm.gintervals_attr_export()
        assert isinstance(r, pd.DataFrame)
        assert iset in r.index

    def test_export_missing_attrs_returns_empty_strings(self, iset):
        pm.gintervals_attr_set(iset, "real", "exists")
        r = pm.gintervals_attr_export(iset, attrs=["real", "fake"])
        assert r.loc[iset, "real"] == "exists"
        assert r.loc[iset, "fake"] == ""


# ---- 4. Import ----


class TestImport:
    def test_import_from_dataframe(self, iset_pair):
        iset1, iset2 = iset_pair
        tbl = pd.DataFrame(
            {"attr_x": ["val_x1", "val_x2"], "attr_y": ["val_y1", "val_y2"]},
            index=[iset1, iset2],
        )
        pm.gintervals_attr_import(tbl)
        assert pm.gintervals_attr_get(iset1, "attr_x") == "val_x1"
        assert pm.gintervals_attr_get(iset2, "attr_x") == "val_x2"
        assert pm.gintervals_attr_get(iset1, "attr_y") == "val_y1"
        assert pm.gintervals_attr_get(iset2, "attr_y") == "val_y2"

    def test_import_preserve_existing(self, iset):
        pm.gintervals_attr_set(iset, "existing", "keep_me")
        tbl = pd.DataFrame({"new_attr": ["new_val"]}, index=[iset])
        pm.gintervals_attr_import(tbl, remove_others=False)
        assert pm.gintervals_attr_get(iset, "existing") == "keep_me"
        assert pm.gintervals_attr_get(iset, "new_attr") == "new_val"

    def test_import_remove_others(self, iset):
        pm.gintervals_attr_set(iset, "existing", "will_go_away")
        pm.gintervals_attr_set(iset, "keeper", "stays")
        tbl = pd.DataFrame({"keeper": ["stays_updated"]}, index=[iset])
        pm.gintervals_attr_import(tbl, remove_others=True)
        assert pm.gintervals_attr_get(iset, "keeper") == "stays_updated"
        assert pm.gintervals_attr_get(iset, "existing") == ""

    def test_import_empty_string_removes(self, iset):
        pm.gintervals_attr_set(iset, "to_remove", "present")
        assert pm.gintervals_attr_get(iset, "to_remove") == "present"
        tbl = pd.DataFrame({"to_remove": [""]}, index=[iset])
        pm.gintervals_attr_import(tbl)
        assert pm.gintervals_attr_get(iset, "to_remove") == ""

    def test_import_nonexistent_set_errors(self):
        tbl = pd.DataFrame({"myattr": ["val"]}, index=["no_such_iset_xyz"])
        with pytest.raises(ValueError, match="does not exist"):
            pm.gintervals_attr_import(tbl)

    def test_import_duplicate_attrs_errors(self, iset):
        tbl = pd.DataFrame({"a": ["v1"], "b": ["v2"]}, index=[iset])
        tbl.columns = ["myattr", "myattr"]
        with pytest.raises(ValueError, match="appears more than once"):
            pm.gintervals_attr_import(tbl)

    def test_import_none_errors(self):
        with pytest.raises(ValueError):
            pm.gintervals_attr_import(None)

    def test_import_bulk_then_export(self, iset_pair):
        iset1, iset2 = iset_pair
        tbl = pd.DataFrame(
            {
                "color": ["red", "blue"],
                "size": ["10", "20"],
                "label": ["first", "second"],
            },
            index=[iset1, iset2],
        )
        pm.gintervals_attr_import(tbl)
        r = pm.gintervals_attr_export([iset1, iset2], attrs=["color", "size", "label"])
        assert r.loc[iset1, "color"] == "red"
        assert r.loc[iset2, "color"] == "blue"
        assert r.loc[iset1, "size"] == "10"
        assert r.loc[iset2, "size"] == "20"
        assert r.loc[iset1, "label"] == "first"
        assert r.loc[iset2, "label"] == "second"


# ---- 5. Integration with gintervals_rm ----


class TestRmCleanup:
    def test_rm_cleans_up_iattr(self):
        name = "temp.iattr_rm_cleanup"
        _create_iset(name)
        pm.gintervals_attr_set(name, "tag", "value")
        assert pm.gintervals_attr_get(name, "tag") == "value"

        from pymisha.intervals_attr import _iattr_path

        path = _iattr_path(name)
        assert path.exists()

        pm.gintervals_rm(name, force=True)
        assert not path.exists()


# ---- 6. Edge cases ----


class TestEdgeCases:
    def test_many_attrs(self, iset):
        n = 50
        attr_names = [f"attr_{i:03d}" for i in range(n)]
        attr_values = [f"value_{i}" for i in range(n)]

        for name, val in zip(attr_names, attr_values, strict=True):
            pm.gintervals_attr_set(iset, name, val)

        for name, val in zip(attr_names, attr_values, strict=True):
            assert pm.gintervals_attr_get(iset, name) == val

        r = pm.gintervals_attr_export(iset)
        assert r.shape[1] == n
        for name, val in zip(attr_names, attr_values, strict=True):
            assert r.loc[iset, name] == val

    def test_special_chars_in_value(self, iset):
        pm.gintervals_attr_set(iset, "path", "/some/path/to/file.txt")
        assert pm.gintervals_attr_get(iset, "path") == "/some/path/to/file.txt"
        pm.gintervals_attr_set(iset, "desc", "value with spaces and tabs")
        assert pm.gintervals_attr_get(iset, "desc") == "value with spaces and tabs"

    def test_special_chars_in_name(self, iset):
        pm.gintervals_attr_set(iset, "my.attr", "dotted")
        assert pm.gintervals_attr_get(iset, "my.attr") == "dotted"
        pm.gintervals_attr_set(iset, "my_attr", "underscored")
        assert pm.gintervals_attr_get(iset, "my_attr") == "underscored"

    def test_null_args_error(self):
        with pytest.raises(ValueError):
            pm.gintervals_attr_get(None, "attr")
        with pytest.raises(ValueError):
            pm.gintervals_attr_get("annotations", None)
        with pytest.raises(ValueError):
            pm.gintervals_attr_set(None, "attr", "val")
        with pytest.raises(ValueError):
            pm.gintervals_attr_set("annotations", None, "val")
        with pytest.raises(ValueError):
            pm.gintervals_attr_set("annotations", "attr", None)
