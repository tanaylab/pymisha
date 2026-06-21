"""Tests for gdb_convert_to_indexed and gdb_create_linked."""

import contextlib

import pytest

import pymisha as pm


def _write_per_chrom_db(root, chrom_rows, seq_name_map=None):
    seq_dir = root / "seq"
    tracks_dir = root / "tracks"
    pssms_dir = root / "pssms"
    seq_dir.mkdir(parents=True)
    tracks_dir.mkdir()
    pssms_dir.mkdir()

    with open(root / "chrom_sizes.txt", "w", encoding="utf-8") as fh:
        for chrom, seq in chrom_rows:
            fh.write(f"{chrom}\t{len(seq)}\n")
            seq_name = seq_name_map(chrom) if seq_name_map else chrom
            (seq_dir / f"{seq_name}.seq").write_bytes(seq.encode("ascii"))


@pytest.fixture
def restore_db():
    old_root = pm._shared._GROOT
    old_user = pm._shared._UROOT
    yield
    if old_root is None:
        pm.gdb_unload()
    else:
        pm.gdb_init(old_root, old_user)


def test_gdb_convert_to_indexed_preserves_order(tmp_path, restore_db):
    root = tmp_path / "db"
    chrom_rows = [
        ("chr15", "A" * 15),
        ("chr10", "C" * 10),
        ("chr1", "T" * 20),
    ]
    _write_per_chrom_db(root, chrom_rows)

    pm.gdb_convert_to_indexed(groot=str(root), force=True, validate=True)

    assert (root / "seq" / "genome.idx").is_file()
    assert (root / "seq" / "genome.seq").is_file()
    assert (root / "seq" / "genome.seq").read_bytes() == (
        b"A" * 15 + b"C" * 10 + b"T" * 20
    )

    rows = (root / "chrom_sizes.txt").read_text(encoding="utf-8").strip().splitlines()
    names = [row.split("\t")[0] for row in rows]
    assert names == [c for c, _ in chrom_rows]

    info = pm.gdb_info(str(root))
    assert info["is_db"] is True
    assert info["format"] == "indexed"


def test_gdb_convert_to_indexed_resolves_chr_prefix_mismatch(tmp_path, restore_db):
    root = tmp_path / "db"
    chrom_rows = [
        ("15", "A" * 15),
        ("10", "C" * 10),
        ("1", "T" * 20),
    ]
    _write_per_chrom_db(root, chrom_rows, seq_name_map=lambda chrom: f"chr{chrom}")

    pm.gdb_convert_to_indexed(groot=str(root), validate=True)

    rows = (root / "chrom_sizes.txt").read_text(encoding="utf-8").strip().splitlines()
    names = [row.split("\t")[0] for row in rows]
    sizes = [int(row.split("\t")[1]) for row in rows]
    assert names == ["chr15", "chr10", "chr1"]
    assert sizes == [15, 10, 20]


def test_gdb_convert_to_indexed_remove_old_files(tmp_path, restore_db):
    root = tmp_path / "db"
    chrom_rows = [("chr1", "ACTGACTG")]
    _write_per_chrom_db(root, chrom_rows)

    old_seq = root / "seq" / "chr1.seq"
    assert old_seq.exists()

    pm.gdb_convert_to_indexed(groot=str(root), remove_old_files=True, validate=False)

    assert (root / "seq" / "genome.idx").is_file()
    assert (root / "seq" / "genome.seq").is_file()
    assert not old_seq.exists()


def test_gdb_convert_to_indexed_converts_tracks_and_intervals(tmp_path, restore_db):
    root = tmp_path / "db"
    chrom_rows = [("chr1", "A" * 100)]
    _write_per_chrom_db(root, chrom_rows)

    pm.gdb_init(str(root))
    intervals = pm.gintervals(["chr1"], [0], [10])
    pm.gtrack_create_sparse("tmp_sparse", "tmp", intervals, [1.0])

    intervals_dir = root / "tracks" / "tmp_set.interv"
    intervals_dir.mkdir()
    (intervals_dir / "chr1").write_bytes(b"\x01\x02\x03\x04")

    pm.gdb_convert_to_indexed(
        groot=str(root),
        validate=False,
        convert_tracks=True,
        convert_intervals=True,
        remove_old_files=True,
    )

    track_dir = root / "tracks" / "tmp_sparse.track"
    assert (track_dir / "track.idx").is_file()
    assert (track_dir / "track.dat").is_file()

    assert (intervals_dir / "intervals.idx").is_file()
    assert (intervals_dir / "intervals.dat").is_file()
    assert not (intervals_dir / "chr1").exists()


def test_gdb_convert_to_indexed_sparse_track_data_roundtrip(tmp_path, restore_db):
    """An indexed sparse track must read back the same intervals and values.

    Regression: ``PMSparseIterator::load_chrom`` gated chromosome loading on the
    existence of a per-chromosome file. Indexed tracks keep their data in
    ``track.dat`` (addressed via ``track.idx``) with no per-chrom file, so every
    chromosome was skipped and ``gextract`` returned ``None`` (``gsummary`` saw
    0 intervals). The earlier convert test only asserted the index files exist,
    never that the data is readable.
    """
    import _pymisha

    root = tmp_path / "db"
    _write_per_chrom_db(root, [("chr1", "A" * 1000), ("chr2", "C" * 1000)])
    pm.gdb_init(str(root))

    intervals = pm.gintervals(
        ["chr1", "chr1", "chr2"], [0, 100, 200], [10, 150, 250]
    )
    pm.gtrack_create_sparse("s", "sparse roundtrip", intervals, [1.5, 2.5, 3.5])

    before = pm.gextract("s", pm.gintervals_all())
    assert before is not None and len(before) == 3
    before = before.sort_values(["chrom", "start"]).reset_index(drop=True)

    pm.gtrack_convert_to_indexed("s")

    # A real indexed track keeps its data in track.dat/track.idx with no
    # per-chromosome files. Remove them so the read path must go through the
    # index (this is the condition under which the bug surfaced: the lab
    # indexed test DB has no per-chrom files).
    track_dir = root / "tracks" / "s.track"
    keep = {"track.idx", "track.dat", ".attributes", ".attributes.yaml", ".meta", "vars"}
    for entry in track_dir.iterdir():
        if entry.name not in keep and entry.is_file():
            entry.unlink()
    assert (track_dir / "track.idx").is_file()
    assert (track_dir / "track.dat").is_file()

    _pymisha.pm_dbreload()
    pm.gdb_init(str(root))
    assert pm.gtrack_info("s")["format"] == "indexed"

    after = pm.gextract("s", pm.gintervals_all())
    assert after is not None, "indexed sparse track read back as empty (None)"
    after = after.sort_values(["chrom", "start"]).reset_index(drop=True)

    assert len(after) == 3
    assert list(after["start"]) == list(before["start"])
    assert list(after["end"]) == list(before["end"])
    assert list(after["s"]) == pytest.approx(list(before["s"]))


def test_gdb_convert_to_indexed_dense_global_percentile_roundtrip(tmp_path, restore_db):
    """A ``global.percentile`` vtrack over an indexed dense track must read back.

    Regression: ``pm_vtrack_compute`` (the Python-fallback C++ entry used by the
    ``global.percentile*`` vtrack functions, which are not scanner-eligible)
    gated chromosome loading on the existence of a per-chromosome file. Indexed
    tracks keep their data in ``track.dat`` (addressed via ``track.idx``) with no
    per-chrom file, so every chromosome was skipped and the per-interval
    statistic came back ``NaN`` -- making the whole ``global.percentile`` result
    ``NaN``. The scanner path (``avg``/``min``/``max`` etc.) was already fixed;
    this standalone entry was not.
    """
    import _pymisha

    from pymisha.vtracks import _GLOBAL_PERCENTILE_CACHE

    root = tmp_path / "db"
    _write_per_chrom_db(root, [("chr1", "A" * 1000), ("chr2", "C" * 1000)])
    pm.gdb_init(str(root))

    # Dense track with distinct per-bin values so percentiles are well-defined.
    intervals = pm.gintervals(["chr1", "chr1", "chr2"], [0, 100, 200], [50, 150, 260])
    pm.gtrack_create_dense("d", "dense pct", intervals, [0.2, 0.7, 0.9], binsize=10)

    pm.gvtrack_create("vp", "d", func="global.percentile")
    before = pm.gextract("vp", pm.gintervals_all(), iterator=10)
    assert before is not None
    before = before.sort_values(["chrom", "start"]).reset_index(drop=True)
    vcol = next(c for c in before.columns if c not in ("chrom", "start", "end", "intervalID"))
    assert before[vcol].notna().any(), "expected some non-NaN percentiles pre-conversion"

    pm.gtrack_convert_to_indexed("d")

    # Force the read through the index: a real indexed track has no per-chrom files.
    track_dir = root / "tracks" / "d.track"
    keep = {"track.idx", "track.dat", ".attributes", ".attributes.yaml", ".meta", "vars"}
    for entry in track_dir.iterdir():
        if entry.name not in keep and entry.is_file():
            entry.unlink()
    assert (track_dir / "track.idx").is_file()
    assert (track_dir / "track.dat").is_file()

    _pymisha.pm_dbreload()
    pm.gdb_init(str(root))
    _GLOBAL_PERCENTILE_CACHE.clear()  # recompute reference + stats against the indexed track
    assert pm.gtrack_info("d")["format"] == "indexed"

    pm.gvtrack_create("vp", "d", func="global.percentile")
    after = pm.gextract("vp", pm.gintervals_all(), iterator=10)
    assert after is not None
    after = after.sort_values(["chrom", "start"]).reset_index(drop=True)

    assert after[vcol].notna().any(), "indexed dense global.percentile read back as all-NaN"
    assert list(after["start"]) == list(before["start"])
    # Same non-NaN mask and values as the per-chrom result.
    assert list(after[vcol].isna()) == list(before[vcol].isna())
    m = before[vcol].notna().to_numpy()
    assert list(after[vcol][m]) == pytest.approx(list(before[vcol][m]))


def test_gdb_convert_to_indexed_array_track_data_roundtrip(tmp_path, restore_db):
    """An indexed array track must read back the same per-column values.

    Regression: ``_array_track.extract_array`` only knew the legacy
    per-chromosome files. Indexed tracks keep their data in ``track.dat``
    (addressed via ``track.idx``) with no per-chrom file, so every chromosome was
    skipped and ``gtrack_array_extract`` returned an empty DataFrame (column
    names still read from ``.colnames``). The reader now also reads each
    chromosome's block from the index.
    """
    import _pymisha

    root = tmp_path / "db"
    _write_per_chrom_db(root, [("chr1", "A" * 1000), ("chr2", "C" * 1000)])
    pm.gdb_init(str(root))

    intervals = pm.gintervals(["chr1", "chr1", "chr2"], [0, 100, 200], [10, 150, 250])
    values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    colnames = ["a", "b", "c"]
    pm.gtrack_array_create("arr", "array roundtrip", intervals, values, colnames)

    before = pm.gtrack_array_extract("arr", None, pm.gintervals_all())
    assert before is not None and len(before) == 3
    before = before.sort_values(["chrom", "start"]).reset_index(drop=True)

    pm.gtrack_convert_to_indexed("arr")

    # Force the read through the index: a real indexed track has no per-chrom files.
    track_dir = root / "tracks" / "arr.track"
    keep = {"track.idx", "track.dat", ".attributes", ".attributes.yaml", ".meta", ".colnames", "vars"}
    for entry in track_dir.iterdir():
        if entry.name not in keep and entry.is_file():
            entry.unlink()
    assert (track_dir / "track.idx").is_file()
    assert (track_dir / "track.dat").is_file()

    _pymisha.pm_dbreload()
    pm.gdb_init(str(root))
    assert pm.gtrack_info("arr")["format"] == "indexed"

    after = pm.gtrack_array_extract("arr", None, pm.gintervals_all())
    assert after is not None and len(after) > 0, "indexed array track read back as empty"
    after = after.sort_values(["chrom", "start"]).reset_index(drop=True)

    assert len(after) == len(before)
    for col in colnames:
        assert list(after[col]) == pytest.approx(list(before[col]))


def test_gdb_convert_to_indexed_reverse_chr_prefix(tmp_path, restore_db):
    """Port of: gdb.convert_to_indexed preserves order when removing chr prefix.

    chrom_sizes.txt has 'chr' prefix, but .seq files do NOT.
    """
    root = tmp_path / "db"
    chrom_rows = [
        ("chr15", "A" * 15),
        ("chr10", "C" * 10),
        ("chr1", "T" * 20),
    ]
    seq_dir = root / "seq"
    tracks_dir = root / "tracks"
    pssms_dir = root / "pssms"
    seq_dir.mkdir(parents=True)
    tracks_dir.mkdir()
    pssms_dir.mkdir()

    with open(root / "chrom_sizes.txt", "w", encoding="utf-8") as fh:
        for chrom, seq in chrom_rows:
            fh.write(f"{chrom}\t{len(seq)}\n")
            # Write files WITHOUT chr prefix (strip it)
            name_no_prefix = chrom.replace("chr", "")
            (seq_dir / f"{name_no_prefix}.seq").write_bytes(seq.encode("ascii"))

    pm.gdb_convert_to_indexed(groot=str(root), force=True, validate=False)

    assert (root / "seq" / "genome.idx").is_file()
    assert (root / "seq" / "genome.seq").is_file()

    rows = (root / "chrom_sizes.txt").read_text(encoding="utf-8").strip().splitlines()
    names = [row.split("\t")[0] for row in rows]
    sizes = [int(row.split("\t")[1]) for row in rows]
    # Order should be preserved
    assert sizes == [15, 10, 20]
    # Names should still be recognizable (either with or without chr prefix)
    assert len(names) == 3


def test_gdb_create_linked_creates_symlinked_db(tmp_path):
    parent = tmp_path / "parent"
    _write_per_chrom_db(parent, [("chr1", "ACGT")])
    child = tmp_path / "linked"

    assert pm.gdb_create_linked(str(child), str(parent)) is True

    assert (child / "tracks").is_dir()
    assert (child / "chrom_sizes.txt").is_symlink()
    assert (child / "seq").is_symlink()
    assert (child / "chrom_sizes.txt").resolve() == (parent / "chrom_sizes.txt").resolve()
    assert (child / "seq").resolve() == (parent / "seq").resolve()


def test_gdb_create_linked_validation(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "chrom_sizes.txt").write_text("chr1\t10\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="seq"):
        pm.gdb_create_linked(str(tmp_path / "linked"), str(parent))

    (parent / "seq").mkdir()
    linked = tmp_path / "linked"
    linked.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        pm.gdb_create_linked(str(linked), str(parent))


# ===========================================================================
# Track format conversion tests
# (ported from R test-gtrack-format-conversion.R)
# ===========================================================================

from _dbpath import TESTDB_ROOT
TEST_DB = TESTDB_ROOT


@pytest.fixture
def restore_test_db():
    """Ensure the test DB is re-initialized after track conversion tests."""
    yield
    pm.gdb_init(str(TEST_DB))


def _cleanup_track(name):
    """Remove a track if it exists, then reload the DB."""
    with contextlib.suppress(Exception):
        pm.gtrack_rm(name, force=True)
    pm._pymisha.pm_dbreload()


def _vals_equal(a, b, tol=1e-6):
    """Compare two float arrays, treating NaN == NaN as True."""
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    if not (nan_a == nan_b).all():
        return False
    mask = ~nan_a
    if not mask.any():
        return True
    return np.allclose(a[mask], b[mask], atol=tol, rtol=tol)


class TestTrackFormatConversion:
    """Tests for gtrack_convert_to_indexed on the live test DB.

    Ported from R test-gtrack-format-conversion.R.
    """

    def test_gtrack_info_reports_format_per_chromosome(self):
        """gtrack_info reports 'per-chromosome' for standard test tracks.

        Ported from R line 19-32.
        """
        info_dense = pm.gtrack_info("dense_track")
        assert "format" in info_dense
        assert info_dense["format"] == "per-chromosome"

        info_sparse = pm.gtrack_info("sparse_track")
        assert info_sparse["format"] == "per-chromosome"

    def test_convert_nonexistent_track_raises(self):
        """gtrack_convert_to_indexed fails for non-existent track.

        Ported from R line 34-39.
        """
        with pytest.raises(Exception, match="does not exist|not found|Track.*nonexistent"):
            pm.gtrack_convert_to_indexed("nonexistent_track_xyz")

    def test_convert_dense_track(self, restore_test_db):
        """gtrack_convert_to_indexed works for dense tracks.

        Ported from R line 48-60.
        Creates a copy of dense_track, converts it to indexed format,
        then verifies data integrity.
        """
        track = "temp_converted_dense"
        try:
            pm.gtrack_create(track, "", "dense_track")
            assert pm.gtrack_info(track)["format"] == "per-chromosome"

            pm.gtrack_convert_to_indexed(track)
            assert pm.gtrack_info(track)["format"] == "indexed"

            # Verify data is intact
            intervals = pm.gintervals(["1"], 0, 1000)
            r = pm.gextract(track, intervals)
            r_orig = pm.gextract("dense_track", intervals)
            assert _vals_equal(r[track].values, r_orig["dense_track"].values)
        finally:
            _cleanup_track(track)

    def test_convert_sparse_track(self, restore_test_db):
        """gtrack_convert_to_indexed works for sparse tracks.

        Ported from R line 62-74.
        """
        track = "temp_converted_sparse"
        try:
            # Copy sparse track by extracting and re-creating
            intervals = pm.gintervals(["1", "2"])
            orig = pm.gextract("sparse_track", intervals)
            track_intervals = orig[["chrom", "start", "end"]].copy()
            values = orig["sparse_track"].tolist()
            pm.gtrack_create_sparse(track, "", track_intervals, values)
            assert pm.gtrack_info(track)["format"] == "per-chromosome"

            pm.gtrack_convert_to_indexed(track)
            assert pm.gtrack_info(track)["format"] == "indexed"

            # Verify data is intact
            r = pm.gextract(track, intervals)
            assert len(r) == len(orig)
        finally:
            _cleanup_track(track)

    def test_track_expressions_with_converted_dense(self, restore_test_db):
        """Track expressions work with converted dense tracks.

        Ported from R line 94-111.
        Tests arithmetic expressions on converted tracks.
        """
        track = "temp_converted_expr"
        try:
            pm.gtrack_create(track, "", "dense_track")
            pm.gtrack_convert_to_indexed(track)

            intervals = pm.gintervals(["1"], 0, 5000)

            # Simple addition
            r1 = pm.gextract(f"{track} + 1", intervals)
            r_orig = pm.gextract("dense_track + 1", intervals)
            assert _vals_equal(r1.iloc[:, 3].values, r_orig.iloc[:, 3].values)

            # Multiplication
            r2 = pm.gextract(f"{track} * 2", intervals)
            r_orig2 = pm.gextract("dense_track * 2", intervals)
            assert _vals_equal(r2.iloc[:, 3].values, r_orig2.iloc[:, 3].values)

            # Complex expression across multiple chroms
            intervals_mc = pm.gintervals(["1", "2"], 0, 10000)
            r3 = pm.gextract(f"2 * {track} + 0.5", intervals_mc)
            r_orig3 = pm.gextract("2 * dense_track + 0.5", intervals_mc)
            assert _vals_equal(r3.iloc[:, 3].values, r_orig3.iloc[:, 3].values)
        finally:
            _cleanup_track(track)

    def test_track_expressions_with_converted_sparse(self, restore_test_db):
        """Track expressions work with converted sparse tracks.

        Ported from R line 113-122.
        """
        track = "temp_converted_sparse_expr"
        try:
            # Copy sparse track by extracting and re-creating
            intervals = pm.gintervals(["1", "2"])
            orig = pm.gextract("sparse_track", intervals)
            track_intervals = orig[["chrom", "start", "end"]].copy()
            values = orig["sparse_track"].tolist()
            pm.gtrack_create_sparse(track, "", track_intervals, values)
            pm.gtrack_convert_to_indexed(track)

            # Extract with the same intervals (no explicit iterator needed)
            r1 = pm.gextract(f"{track} * 3", intervals)
            r_orig = pm.gextract("sparse_track * 3", intervals)
            assert len(r1) == len(r_orig)
            assert _vals_equal(r1.iloc[:, 3].values, r_orig.iloc[:, 3].values)
        finally:
            _cleanup_track(track)

    def test_mixed_expressions_converted_and_normal(self, restore_test_db):
        """Mixed expressions with converted and non-converted tracks.

        Ported from R line 124-137.
        """
        track = "temp_converted_mixed"
        try:
            pm.gtrack_create(track, "", "dense_track")
            pm.gtrack_convert_to_indexed(track)

            intervals = pm.gintervals(["1"], 0, 5000)
            r1 = pm.gextract(f"{track} + dense_track", intervals)
            r_orig = pm.gextract("dense_track + dense_track", intervals)
            assert _vals_equal(r1.iloc[:, 3].values, r_orig.iloc[:, 3].values)

            intervals_mc = pm.gintervals(["1", "2"], 0, 10000)
            r2 = pm.gextract(f"{track} * dense_track + 0.1", intervals_mc)
            r_orig2 = pm.gextract("dense_track * dense_track + 0.1", intervals_mc)
            assert _vals_equal(r2.iloc[:, 3].values, r_orig2.iloc[:, 3].values)
        finally:
            _cleanup_track(track)

    def test_gtrack_create_from_converted_track(self, restore_test_db):
        """gtrack_create with expression using converted track.

        Ported from R line 139-151.
        """
        src_track = "temp_converted_source"
        new_track = "temp_created_from_converted"
        try:
            pm.gtrack_create(src_track, "", "dense_track")
            pm.gtrack_convert_to_indexed(src_track)

            pm.gtrack_create(new_track, "", f"{src_track} * 2 + 5")
            intervals = pm.gintervals(["1"], 0, 5000)
            r1 = pm.gextract(new_track, intervals)
            r_orig = pm.gextract("dense_track * 2 + 5", intervals)
            assert _vals_equal(r1.iloc[:, 3].values, r_orig.iloc[:, 3].values)
        finally:
            _cleanup_track(new_track)
            _cleanup_track(src_track)

    def test_vtrack_on_converted_dense(self, restore_test_db):
        """Vtrack based on converted dense track works.

        Ported from R line 157-169.
        Compare a vtrack on the converted track against a vtrack on the
        original track (both using func='avg'), so the aggregation is
        identical.
        """
        track = "temp_converted_vtrack"
        vt_conv = "v_converted"
        vt_orig = "v_orig"
        try:
            pm.gtrack_create(track, "", "dense_track")
            pm.gtrack_convert_to_indexed(track)

            pm.gvtrack_create(vt_conv, track, func="avg")
            pm.gvtrack_create(vt_orig, "dense_track", func="avg")
            intervals = pm.gintervals(["1", "2"])
            r = pm.gextract(vt_conv, intervals, iterator=100)
            r_orig = pm.gextract(vt_orig, intervals, iterator=100)
            assert len(r) == len(r_orig)
            assert _vals_equal(r[vt_conv].values, r_orig[vt_orig].values)
        finally:
            for vt in (vt_conv, vt_orig):
                with contextlib.suppress(Exception):
                    pm.gvtrack_rm(vt)
            _cleanup_track(track)

    def test_vtrack_avg_on_converted_track(self, restore_test_db):
        """Vtrack with avg function on converted track.

        Ported from R line 171-185.
        """
        track = "temp_converted_vtrack_avg"
        vt = "v_converted_avg"
        try:
            pm.gtrack_create(track, "", "dense_track")
            pm.gtrack_convert_to_indexed(track)

            pm.gvtrack_create(vt, track, func="avg")
            intervals = pm.gintervals(["1", "2"])
            r = pm.gextract(vt, intervals, iterator=500)
            r_orig = pm.gextract("dense_track", intervals, iterator=500)
            assert len(r) == len(r_orig)
        finally:
            with contextlib.suppress(Exception):
                pm.gvtrack_rm(vt)
            _cleanup_track(track)

    def test_gtrack_modify_on_converted_track(self, restore_test_db):
        """gtrack_modify works on converted tracks.

        Ported from R line 248-261.
        """
        track = "temp_converted_modify"
        try:
            pm.gtrack_create(track, "", "dense_track")
            pm.gtrack_convert_to_indexed(track)

            intervals = pm.gintervals(["1"], 1000, 5000)
            before = pm.gextract(track, intervals)

            pm.gtrack_modify(track, f"{track} * 2", intervals)

            after = pm.gextract(track, intervals)
            import numpy as np
            np.testing.assert_allclose(
                after[track].values, before[track].values * 2
            )
        finally:
            _cleanup_track(track)

    def test_numeric_iterator_with_converted_track(self, restore_test_db):
        """Numeric iterator works with converted track.

        Ported from R line 298-308.
        """
        track = "temp_converted_iter"
        try:
            pm.gtrack_create(track, "", "dense_track")
            pm.gtrack_convert_to_indexed(track)

            intervals = pm.gintervals(["1"], 0, 10000)
            r = pm.gextract(track, intervals, iterator=500)
            r_orig = pm.gextract("dense_track", intervals, iterator=500)
            assert len(r) == len(r_orig)
            assert _vals_equal(r.iloc[:, 3].values, r_orig.iloc[:, 3].values)
        finally:
            _cleanup_track(track)

    def test_gscreen_with_converted_track(self, restore_test_db):
        """gscreen works with converted tracks.

        Ported from R line 350-359.
        """
        track = "temp_converted_screen"
        try:
            pm.gtrack_create(track, "", "dense_track")
            pm.gtrack_convert_to_indexed(track)

            intervals = pm.gintervals(["1", "2"])
            r = pm.gscreen(f"{track} > 0.2", intervals)
            r_orig = pm.gscreen("dense_track > 0.2", intervals)
            if r is None or r_orig is None:
                assert r is None and r_orig is None
            else:
                assert len(r) == len(r_orig)
        finally:
            _cleanup_track(track)

    def test_converted_track_with_allgenome(self, restore_test_db):
        """Converted track with ALLGENOME extraction.

        Ported from R line 418-428.
        """
        track = "temp_converted_allgenome"
        try:
            pm.gtrack_create(track, "", "dense_track")
            pm.gtrack_convert_to_indexed(track)

            r = pm.gextract(track, pm.gintervals_all())
            r_orig = pm.gextract("dense_track", pm.gintervals_all())
            assert len(r) == len(r_orig)
        finally:
            _cleanup_track(track)

    def test_multiple_operations_on_converted_track(self, restore_test_db):
        """Multiple operations on same converted track.

        Ported from R line 389-416.
        Tests extract, vtrack, and modify in sequence on one converted track.
        """
        track = "temp_converted_multi_ops"
        vt = "v_multi"
        try:
            pm.gtrack_create(track, "", "dense_track")
            pm.gtrack_convert_to_indexed(track)

            # Extract with expression
            intervals_small = pm.gintervals(["1"], 0, 5000)
            r1 = pm.gextract(f"{track} * 2", intervals_small)
            assert len(r1) > 0

            # Create vtrack and use expression
            pm.gvtrack_create(vt, track)
            r2 = pm.gextract(f"{vt} + 0.5", intervals_small, iterator=100)
            assert len(r2) > 0

            # Modify in-place
            intervs = pm.gintervals(["1"], 1000, 3000)
            pm.gtrack_modify(track, f"{track} * 1.5", intervs)
            r3 = pm.gextract(track, intervs)
            assert len(r3) > 0

            # Extract again after modification
            r4 = pm.gextract(track, intervals_small)
            assert len(r4) > 0
        finally:
            with contextlib.suppress(Exception):
                pm.gvtrack_rm(vt)
            _cleanup_track(track)


class TestGdbConvertThreads:
    def test_threads_in_signature_with_correct_default(self):
        import inspect
        sig = inspect.signature(pm.gdb_convert_to_indexed)
        assert "threads" in sig.parameters
        assert sig.parameters["threads"].default is None

    def test_threads_zero_raises(self, tmp_path, restore_db):
        # Build a minimal indexed DB so the early validation passes.
        root = tmp_path / "db"
        chrom_rows = [("chr1", "AC" * 10)]
        _write_per_chrom_db(root, chrom_rows)
        pm.gdb_convert_to_indexed(groot=str(root))
        with pytest.raises(ValueError, match="threads must be"):
            pm.gdb_convert_to_indexed(
                groot=str(root),
                threads=0,
                convert_tracks=False,
                convert_intervals=False,
            )

    def test_threads_1_uses_serial_path(self, tmp_path, restore_db, monkeypatch):
        # With threads=1, Pool must not be instantiated.
        import multiprocessing
        called = []
        original_pool = multiprocessing.context.ForkContext.Pool

        def spy_pool(self, *a, **k):
            called.append(True)
            return original_pool(self, *a, **k)

        monkeypatch.setattr(
            multiprocessing.context.ForkContext, "Pool", spy_pool
        )
        root = tmp_path / "db"
        chrom_rows = [("chr1", "AC" * 10)]
        _write_per_chrom_db(root, chrom_rows)
        pm.gdb_convert_to_indexed(
            groot=str(root),
            threads=1,
            convert_tracks=False,
            convert_intervals=False,
        )
        assert called == []
