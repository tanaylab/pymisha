"""The failures pymisha recovers from must be reportable, not invisible.

pymisha catches a lot of exceptions on purpose - an optional dependency that
is not installed, a probe asking "is this a COMPUTED track", a best-effort
cleanup. Every one of those used to be silent, so a ``MemoryError`` or a
corrupt file reduced to the same fallback as the harmless case and the user
saw a plausible wrong answer instead of a stack trace.

These tests pin the reporting contract:

* the package logger is inert on import (stdlib library convention),
* each tier of handler reports on the channel it is supposed to,
* the fallback that used to happen still happens.
"""

import builtins
import logging
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import _pymisha
import pytest
from _dbpath import TESTDB_ROOT

import pymisha as pm
from pymisha import _shared
from pymisha._log import PymishaWarning

TEST_DB = TESTDB_ROOT


# ---------------------------------------------------------------------------
# The stdlib contract: a library configures no logging of its own.
# ---------------------------------------------------------------------------

def test_import_configures_no_handler_and_emits_nothing():
    """Importing pymisha must not touch the root logger nor print anything."""
    code = (
        "import logging, sys\n"
        "root_before = list(logging.getLogger().handlers)\n"
        "import pymisha\n"
        "root_after = list(logging.getLogger().handlers)\n"
        "assert root_before == root_after, root_after\n"
        "pkg = logging.getLogger('pymisha')\n"
        "assert all(isinstance(h, logging.NullHandler) for h in pkg.handlers), pkg.handlers\n"
        "assert pkg.level == logging.NOTSET, pkg.level\n"
        "assert pkg.propagate is True\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == ""
    assert proc.stderr == ""


def test_package_logger_is_the_parent_of_the_module_loggers():
    """``logging.getLogger("pymisha").setLevel(...)`` must reach every module."""
    from pymisha import tracks

    assert tracks._logger.name.startswith("pymisha.")
    assert tracks._logger.parent is logging.getLogger("pymisha")


# ---------------------------------------------------------------------------
# Tier 1: expected fallback, narrowed type -> logger.debug
# ---------------------------------------------------------------------------

def test_computed_probe_logs_at_debug_and_still_falls_back(caplog, monkeypatch):
    from pymisha import tracks

    def _boom(_track):
        raise _pymisha.error("Track 'no.such.track.at.all' does not exist")

    monkeypatch.setattr(_pymisha, "pm_track_path", _boom)
    tracks._clear_computed_track_cache()
    with caplog.at_level(logging.DEBUG, logger="pymisha"):
        assert tracks._is_computed_track_supported("no.such.track.at.all") is False
    assert any(
        rec.levelno == logging.DEBUG and "no.such.track.at.all" in rec.getMessage()
        for rec in caplog.records
    ), caplog.records
    # The negative-caching fallback still happened.
    assert tracks._COMPUTED_TYPE_OK_CACHE["no.such.track.at.all"] is False


def test_format_percentile_logs_at_debug_and_still_falls_back(caplog):
    from pymisha.summary import _format_percentile

    with caplog.at_level(logging.DEBUG, logger="pymisha"):
        assert _format_percentile("not-a-number") == "not-a-number"
    assert any(rec.levelno == logging.DEBUG for rec in caplog.records), caplog.records


# ---------------------------------------------------------------------------
# Tier 2: still caught broadly -> logger.warning(..., exc_info=True)
# ---------------------------------------------------------------------------

def test_interval_set_registration_failure_warns_and_still_saves(tmp_path, caplog, monkeypatch):
    """A failed C++ name-registration must be logged, and the set still saved."""
    pm.gdb_init(str(TEST_DB))
    name = "test_excvis_register"

    def _boom(_name):
        raise _pymisha.error("registration exploded")

    monkeypatch.setattr(_pymisha, "pm_interv_register", _boom)
    intervals = pm.gintervals("chr1", 100, 200)
    try:
        with caplog.at_level(logging.DEBUG, logger="pymisha"):
            pm.gintervals_save(intervals, name)
        assert any(
            rec.levelno == logging.WARNING and rec.exc_info is not None
            for rec in caplog.records
        ), caplog.records
        monkeypatch.undo()
        pm.gdb_reload()
        assert pm.gintervals_exists(name)
    finally:
        monkeypatch.undo()
        if pm.gintervals_exists(name):
            pm.gintervals_rm(name, force=True)


def test_yaml_metadata_failure_warns_and_falls_back_to_plain_parsing(tmp_path, caplog, monkeypatch):
    from pymisha import dataset

    meta = tmp_path / "meta.yaml"
    meta.write_text("name: demo\ndescription: something\n", encoding="utf-8")

    class _BoomYaml:
        YAMLError = ValueError

        @staticmethod
        def safe_load(_text):
            raise RuntimeError("yaml exploded")

    monkeypatch.setitem(sys.modules, "yaml", _BoomYaml)
    with caplog.at_level(logging.DEBUG, logger="pymisha"):
        parsed = dataset._parse_dataset_metadata(meta)
    # The plain "key: value" fallback parser still ran.
    assert parsed["name"] == "demo"
    assert any(
        rec.levelno == logging.WARNING and rec.exc_info is not None
        for rec in caplog.records
    ), caplog.records


# ---------------------------------------------------------------------------
# Tier 3: R-parity - where R misha warns, pymisha warns
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_touch_db_cache_dirty_warns_on_readonly_groot(tmp_path):
    """The reported bug's canonical instance: R warns here, pymisha was silent."""
    groot = tmp_path / "ro_groot"
    groot.mkdir()
    os.chmod(groot, 0o500)
    try:
        with pytest.warns(PymishaWarning, match="db.cache.dirty"):
            _shared._touch_db_cache_dirty(str(groot))
    finally:
        os.chmod(groot, 0o700)
    # Best-effort semantics preserved: no exception, and no sentinel written.
    assert not (groot / ".db.cache.dirty").exists()


def test_touch_db_cache_dirty_is_silent_when_it_works(tmp_path):
    groot = tmp_path / "rw_groot"
    groot.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _shared._touch_db_cache_dirty(str(groot))
    assert (groot / ".db.cache.dirty").exists()


def test_pymisha_warning_is_a_runtime_warning_and_shown_by_default():
    assert issubclass(PymishaWarning, RuntimeWarning)
    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        warnings.warn("visible", PymishaWarning, stacklevel=1)
    assert len(caught) == 1


def test_touch_db_cache_dirty_warning_points_at_the_caller(tmp_path):
    """A warning that blames pymisha's own internals is nearly useless."""
    groot = tmp_path / "ro_groot2"
    groot.mkdir()
    os.chmod(groot, 0o500)
    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _shared._touch_db_cache_dirty(str(groot))  # <- this line must be blamed
    finally:
        os.chmod(groot, 0o700)
    assert len(caught) == 1
    assert Path(caught[0].filename).name == "test_exception_visibility.py"


# ---------------------------------------------------------------------------
# Tier 4: the two `except BaseException` sites are deliberate cleanup-on-
# interrupt handlers that re-raise. Deleting them would leak a temp directory
# on Ctrl-C, so they stay - and these tests pin why.
# ---------------------------------------------------------------------------

def test_atomic_track_create_cleans_up_on_keyboard_interrupt():
    from pymisha.tracks import _atomic_track_create

    pm.gdb_init(str(TEST_DB))
    leftovers = []
    with pytest.raises(KeyboardInterrupt), _atomic_track_create("test_excvis_interrupt") as tmp_dir:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        leftovers.append(tmp_dir)
        raise KeyboardInterrupt
    assert leftovers and not leftovers[0].exists()
    assert not pm.gtrack_exists("test_excvis_interrupt")


def test_replace_intervals_set_cleans_up_on_keyboard_interrupt(monkeypatch):
    from pymisha import intervals as iv

    pm.gdb_init(str(TEST_DB))
    name = "test_excvis_replace"
    pm.gintervals_save(pm.gintervals("chr1", 100, 200), name)
    try:
        def _boom(*_a, **_kw):
            raise KeyboardInterrupt

        monkeypatch.setattr(iv.os, "replace", _boom)
        with pytest.raises(KeyboardInterrupt):
            iv._replace_intervals_set(pm.gintervals("chr1", 300, 400), name)
        monkeypatch.undo()
        # The original survived, so the temporary copy was dropped.
        assert pm.gintervals_exists(name)
        assert not [n for n in (pm.gintervals_ls() or []) if n.startswith(f"{name}_pmtmp")]
    finally:
        monkeypatch.undo()
        if pm.gintervals_exists(name):
            pm.gintervals_rm(name, force=True)


# ---------------------------------------------------------------------------
# Behaviour preservation for a sample of converted sites.
# ---------------------------------------------------------------------------

@pytest.fixture
def indexed_db(tmp_path):
    """A throwaway copy of the test DB with one track in *indexed* format.

    ``gtrack_info``'s 2D-header sniff only runs for ``format == "indexed"``, and
    every track in the canonical test DB is per-chromosome - so a test that does
    not convert one never enters the branch it means to exercise.
    """
    dst = tmp_path / "test"
    shutil.copytree(TEST_DB, dst)
    prev = _shared._GROOT
    pm.gdb_init(str(dst))
    pm.gtrack_convert_to_indexed("dense_track")
    assert pm.gtrack_info("dense_track")["format"] == "indexed"
    yield dst
    if prev:
        pm.gdb_init(prev)


def test_gtrack_info_still_returns_info_when_the_2d_header_read_fails(indexed_db, caplog):
    """An unreadable track.idx must be reported, and must not lose the info dict.

    Regression: the handler used to name ``struct.error`` while ``struct`` was a
    *function-local* import made further down the same ``try``. When ``open()``
    failed first - the ordinary case - evaluating the except clause raised
    ``UnboundLocalError`` and that escaped ``gtrack_info`` entirely.
    """
    real_open = builtins.open

    def _boom(path, *args, **kwargs):
        if str(path).endswith("track.idx"):
            raise PermissionError(13, "Permission denied", str(path))
        return real_open(path, *args, **kwargs)

    builtins.open = _boom
    try:
        with caplog.at_level(logging.DEBUG, logger="pymisha"):
            info = pm.gtrack_info("dense_track")
    finally:
        builtins.open = real_open

    # The fallback still happens: the track is reported as the engine sees it.
    assert info["type"] == "dense"
    # And the handler was actually reached, which is what the branch assertion
    # in the fixture and this log record together prove.
    assert any(
        rec.levelno == logging.WARNING and "track.idx" in rec.getMessage()
        for rec in caplog.records
    ), caplog.records


def test_gtrack_info_2d_header_sniff_survives_a_truncated_idx(indexed_db, caplog):
    """The other half of the tuple: struct.error from a short read."""
    real_open = builtins.open

    class _ShortFile:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def seek(self, *_a):
            return 0

        def read(self, _n=-1):
            return b"MISHT2D\x00"  # magic matches, header then runs out

    def _short(path, *args, **kwargs):
        if str(path).endswith("track.idx"):
            return _ShortFile()
        return real_open(path, *args, **kwargs)

    builtins.open = _short
    try:
        with caplog.at_level(logging.DEBUG, logger="pymisha"):
            info = pm.gtrack_info("dense_track")
    finally:
        builtins.open = real_open

    assert info["type"] in ("dense", "points", "rectangles")
    assert any(rec.levelno == logging.WARNING for rec in caplog.records), caplog.records


def test_chrom_sizes_probe_returns_none_with_no_database(caplog):
    """The documented "no active root" fallback must not raise.

    ``gintervals_all()`` reaches ``_checkroot()``, which raises ``RuntimeError``,
    not a pymisha or value error - so a tuple without it turned a documented
    skip-the-boundary-check into a public-API failure.
    """
    from pymisha import intervals as iv

    prev = _shared._GROOT
    pm.gdb_unload()
    try:
        iv._chrom_sizes_cache = None
        with caplog.at_level(logging.DEBUG, logger="pymisha"):
            assert iv._chrom_sizes_for_2d_verify() is None
        assert any(rec.levelno == logging.DEBUG for rec in caplog.records), caplog.records
    finally:
        iv._chrom_sizes_cache = None
        if prev:
            pm.gdb_init(prev)


def test_2d_set_ops_still_work_with_no_database():
    """The public consequence of the probe above: these must not need a groot."""
    import pandas as pd

    from pymisha import intervals as iv

    rect = pd.DataFrame({
        "chrom1": ["chr1"], "start1": [0], "end1": [100],
        "chrom2": ["chr1"], "start2": [0], "end2": [100],
    })
    prev = _shared._GROOT
    pm.gdb_unload()
    try:
        iv._chrom_sizes_cache = None
        # Row counts verified against the base commit in a worktree: with no
        # groot the boundary check is skipped, intersect yields the one
        # rectangle and union does not merge (2 rows).
        assert len(pm.gintervals_2d_intersect(rect, rect)) == 1
        assert len(pm.gintervals_2d_union(rect, rect)) == 2
        assert len(pm.gintervals_2d_band_intersect(rect, (-1000, 1000))) == 1
    finally:
        iv._chrom_sizes_cache = None
        if prev:
            pm.gdb_init(prev)


def test_normalize_chroms_probe_still_falls_back_to_the_raw_name(caplog, monkeypatch):
    from pymisha import tracks

    pm.gdb_init(str(TEST_DB))

    def _boom(_chroms):
        raise _pymisha.error("normalization exploded")

    monkeypatch.setattr(_pymisha, "pm_normalize_chroms", _boom)
    import pandas as pd

    df = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10], "value": [1.0]})
    with caplog.at_level(logging.DEBUG, logger="pymisha"):
        out = tracks._canonicalize_known_chroms(df)
    # Every row is dropped exactly as before, and the reason is now logged.
    assert len(out) == 0
    assert any(rec.levelno == logging.DEBUG for rec in caplog.records), caplog.records

def test_import_set_warns_on_each_failed_file(tmp_path):
    """R's gtrack.import_set reports each failure with message(), visible with no
    configuration. A silent log would hide exactly the case a bulk import needs
    surfaced, so pymisha warns to match."""
    import pymisha as pm
    from pymisha._log import PymishaWarning

    bad = tmp_path / "broken.wig"
    bad.write_text("this is not a wig file at all\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pm.gtrack_import_set(
            "probe", str(tmp_path / "*.wig"), 100, track_prefix="excvis_importset_"
        )

    assert result.get("files_failed") == ["broken.wig"]
    parity = [w for w in caught if issubclass(w.category, PymishaWarning)]
    assert len(parity) == 1, f"expected one PymishaWarning, got {[w.category for w in caught]}"
    assert "broken.wig" in str(parity[0].message)
    assert "excvis_importset_broken" in str(parity[0].message)
