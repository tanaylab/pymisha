"""Tests for gtrack_export_bedgraph and gtrack_export_bigwig."""

import gzip
import os

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


class TestBedGraphExport:
    """Tests for gtrack_export_bedgraph."""

    def test_basic_export(self, tmp_path):
        """Export a dense track to bedGraph and verify format."""
        outfile = str(tmp_path / "dense.bedgraph")
        pm.gtrack_export_bedgraph("dense_track", outfile)

        assert os.path.exists(outfile)
        with open(outfile) as f:
            lines = f.readlines()

        # First line should be the header
        assert lines[0].startswith("track type=bedGraph")
        assert 'name="dense_track"' in lines[0]

        # Data lines should have 4 tab-separated columns
        for line in lines[1:]:
            parts = line.strip().split("\t")
            assert len(parts) == 4, f"Expected 4 columns, got {len(parts)}: {line}"
            # chrom should be a non-empty string
            assert len(parts[0]) > 0
            # start and end should be integers
            int(parts[1])
            int(parts[2])
            # value should be numeric
            float(parts[3])

    def test_export_with_intervals(self, tmp_path):
        """Export with specific intervals produces data only for those intervals."""
        outfile = str(tmp_path / "chr1.bedgraph")
        intervals = pm.gintervals("1", 0, 1000)
        pm.gtrack_export_bedgraph("dense_track", outfile, intervals=intervals)

        assert os.path.exists(outfile)
        with open(outfile) as f:
            lines = f.readlines()

        # Header + at least some data
        assert len(lines) >= 2

        # All data lines should be for chromosome 1 (name may or may not have "chr" prefix)
        genome_chroms = pm.gintervals_all()
        chr1_name = genome_chroms[genome_chroms["chrom"].str.contains("1")]["chrom"].iloc[0]
        for line in lines[1:]:
            parts = line.strip().split("\t")
            assert parts[0] == chr1_name

    def test_export_with_iterator(self, tmp_path):
        """Export with explicit iterator bin size."""
        outfile = str(tmp_path / "iter.bedgraph")
        intervals = pm.gintervals("1", 0, 1000)
        pm.gtrack_export_bedgraph(
            "dense_track", outfile, intervals=intervals, iterator=200
        )

        assert os.path.exists(outfile)
        with open(outfile) as f:
            lines = f.readlines()

        # Should have header + 5 bins (1000/200)
        assert len(lines) == 6  # 1 header + 5 data lines

    def test_gzip_compression(self, tmp_path):
        """Export to .gz file produces valid gzip output."""
        outfile = str(tmp_path / "dense.bedgraph.gz")
        intervals = pm.gintervals("1", 0, 1000)
        pm.gtrack_export_bedgraph(
            "dense_track", outfile, intervals=intervals, iterator=200
        )

        assert os.path.exists(outfile)

        # Verify it's a valid gzip file
        with gzip.open(outfile, "rt") as f:
            lines = f.readlines()

        assert lines[0].startswith("track type=bedGraph")
        assert len(lines) == 6  # header + 5 data lines

    def test_nan_exclusion(self, tmp_path):
        """NaN values are excluded from bedGraph output."""
        outfile = str(tmp_path / "sparse.bedgraph")
        # Use a region where sparse_track has no data (expect NaNs)
        intervals = pm.gintervals("X", 190000, 200000)
        pm.gtrack_export_bedgraph(
            "sparse_track", outfile, intervals=intervals, iterator=1000
        )

        assert os.path.exists(outfile)
        with open(outfile) as f:
            lines = f.readlines()

        # Data lines should not contain NaN
        for line in lines[1:]:
            parts = line.strip().split("\t")
            assert parts[3].lower() != "nan", f"NaN found in output: {line}"

    def test_custom_name(self, tmp_path):
        """Custom name appears in bedGraph header."""
        outfile = str(tmp_path / "custom.bedgraph")
        intervals = pm.gintervals("1", 0, 1000)
        pm.gtrack_export_bedgraph(
            "dense_track", outfile, intervals=intervals, name="my_track"
        )

        with open(outfile) as f:
            header = f.readline()

        assert 'name="my_track"' in header

    def test_sorted_output(self, tmp_path):
        """Output should be sorted by chrom (genome order) then start."""
        outfile = str(tmp_path / "sorted.bedgraph")
        # Use intervals across multiple chromosomes
        intervals = pd.concat(
            [
                pm.gintervals("2", 0, 500),
                pm.gintervals("1", 0, 500),
            ],
            ignore_index=True,
        )
        pm.gtrack_export_bedgraph(
            "dense_track", outfile, intervals=intervals, iterator=100
        )

        with open(outfile) as f:
            lines = f.readlines()

        data_lines = lines[1:]
        chroms = [line.split("\t")[0] for line in data_lines]
        starts = [int(line.split("\t")[1]) for line in data_lines]

        # chr1 should come before chr2
        chr1_indices = [i for i, c in enumerate(chroms) if c == "chr1"]
        chr2_indices = [i for i, c in enumerate(chroms) if c == "chr2"]
        if chr1_indices and chr2_indices:
            assert max(chr1_indices) < min(chr2_indices)

        # Within each chromosome, starts should be sorted
        for chrom in set(chroms):
            chrom_starts = [
                starts[i] for i, c in enumerate(chroms) if c == chrom
            ]
            assert chrom_starts == sorted(chrom_starts)

    def test_track_expression(self, tmp_path):
        """Export a track expression (not just a track name)."""
        outfile = str(tmp_path / "expr.bedgraph")
        intervals = pm.gintervals("1", 0, 1000)
        pm.gtrack_export_bedgraph(
            "dense_track * 2",
            outfile,
            intervals=intervals,
            iterator=200,
        )

        assert os.path.exists(outfile)
        with open(outfile) as f:
            lines = f.readlines()

        assert len(lines) >= 2
        # Header should contain the expression
        assert 'name="dense_track * 2"' in lines[0]

    def test_nonexistent_directory_raises(self, tmp_path):
        """Writing to a nonexistent directory raises FileNotFoundError."""
        outfile = str(tmp_path / "nonexistent" / "dense.bedgraph")
        with pytest.raises(FileNotFoundError):
            pm.gtrack_export_bedgraph("dense_track", outfile)

    def test_sparse_track_export(self, tmp_path):
        """Export a sparse track to bedGraph."""
        outfile = str(tmp_path / "sparse.bedgraph")
        pm.gtrack_export_bedgraph("sparse_track", outfile)

        assert os.path.exists(outfile)
        with open(outfile) as f:
            lines = f.readlines()

        assert lines[0].startswith("track type=bedGraph")
        # Sparse track should still have some data
        assert len(lines) >= 2


# ---- misha 5.17.0: the lines R misha writes (test-track-export.R) -----------------------------

_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "misha_5_17_export")


def test_bedgraph_export_writes_the_lines_r_misha_writes(tmp_path):
    # written by R misha 5.17.0 (R 4.4.1) from this database: paste(chrom, sprintf("%.0f", start),
    # sprintf("%.0f", end), value, sep = "\t"), values as.character() of doubles
    outfile = str(tmp_path / "x.bedgraph")
    pm.gtrack_export_bedgraph("dense_track * 1e5 / 3", outfile, intervals=pm.gintervals(["1", "2"], 0, 20000),
                              iterator=50)
    with open(os.path.join(_FIXTURES, "dense_scaled.bedgraph"), "rb") as f, open(outfile, "rb") as g:
        assert g.read() == f.read()


@pytest.mark.skipif(np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant,
                    reason="the fixture is from R on x86_64, which formats with an 80-bit long double; "
                           "where long double is double (arm64 macOS) the last digit can differ")
def test_doubles_are_formatted_as_r_as_character_formats_them():
    import _pymisha

    # sprintf("%a", x) and as.character(x) of R 4.4.1, from the value classes of the R test
    with open(os.path.join(_FIXTURES, "as_character.tsv")) as f:
        rows = [line.rstrip("\n").split("\t") for line in f if not line.startswith("#")]
    vals = np.array([float.fromhex(h) if h not in ("Inf", "-Inf") else float(h) for h, _ in rows])
    n = len(vals)
    got = _pymisha.pm_bedgraph_lines(np.zeros(n, dtype=np.int32), ["c"], np.zeros(n, dtype=np.int64),
                                     np.ones(n, dtype=np.int64), vals).decode()
    assert got.split("\n")[:-1] == [f"c\t0\t1\t{r}" for _, r in rows]


def test_bedgraph_lines_skip_nan_rows_and_keep_the_order_given():
    import _pymisha

    vals = np.array([1.5, np.nan, -2.0, 1e5, np.nan, 0.1 + 0.2])
    got = _pymisha.pm_bedgraph_lines(np.array([1, 0, 0, 1, 1, 0], dtype=np.int32), ["chrA", "chrB"],
                                     np.arange(6) * 10, np.arange(6) * 10 + 5, vals)
    assert got == b"chrB\t0\t5\t1.5\nchrA\t20\t25\t-2\nchrB\t30\t35\t1e+05\nchrA\t50\t55\t0.3\n"


def test_bedgraph_export_over_intervals_with_no_values_writes_the_header_only(tmp_path):
    outfile = str(tmp_path / "x.bedgraph")
    # sparse_track has no values there (its first two intervals are chr1:0-50 and chr1:100-150)
    empty = pm.gintervals("1", 60, 90)
    assert pm.gextract("sparse_track", intervals=empty) is None
    with pytest.warns(UserWarning, match="no values in the intervals") as caught:
        pm.gtrack_export_bedgraph("sparse_track", outfile, intervals=empty)
    assert len(caught) == 1
    with open(outfile) as f:
        assert f.read().splitlines() == ['track type=bedGraph name="sparse_track"']


# ---- bigWig export ----

def expected_records(expr, intervals, iterator=None):
    """The records gextract returns, without NaN, values as float32: what a bigWig export of the
    same expression holds."""
    kwargs = {} if iterator is None else {"iterator": iterator}
    d = pm.gextract(expr, intervals=intervals, colnames="value", **kwargs)
    d = d[~d["value"].isna()].reset_index(drop=True)
    return d["chrom"].astype(str).to_numpy(), d["start"].to_numpy(np.int64), d["end"].to_numpy(np.int64), \
        d["value"].to_numpy(np.float64).astype(np.float32)


def assert_records(f, expr, intervals, iterator=None):
    chrom, start, end, value = expected_records(expr, intervals, iterator)
    got = pm.gintervals_import_bigwig(f, pm.gintervals_all())
    np.testing.assert_array_equal(got["chrom"].astype(str).to_numpy(), chrom)
    np.testing.assert_array_equal(got["start"].to_numpy(np.int64), start)
    np.testing.assert_array_equal(got["end"].to_numpy(np.int64), end)
    np.testing.assert_array_equal(got["value"].to_numpy().astype(np.float32), value)
    return chrom, start, end, value


def read_bigwig_meta(f):
    """The parts of a bigWig file that misha's reader does not read: the header, the total
    summary, the chromosome tree and the records of each zoom level."""
    import struct
    import zlib

    with open(f, "rb") as fh:
        raw = fh.read()

    def u(fmt, off):
        return struct.unpack_from("<" + fmt, raw, off)[0]

    chroms = []
    tree = u("Q", 8)
    key_size = u("I", tree + 8)

    def read_node(off):
        is_leaf, count = raw[off], u("H", off + 2)
        for i in range(count):
            item = off + 4 + i * (key_size + 8)
            key = raw[item:item + key_size].rstrip(b"\0").decode()
            if is_leaf:
                chroms.append((key, u("I", item + key_size), u("I", item + key_size + 4)))
            else:
                read_node(u("Q", item + key_size))

    read_node(tree + 32)

    def rtree_blocks(off):
        res = []

        def walk(node):
            is_leaf, count = raw[node], u("H", node + 2)
            for i in range(count):
                if is_leaf:
                    item = node + 4 + i * 32
                    res.append((u("Q", item + 16), u("Q", item + 24)))
                else:
                    walk(u("Q", node + 4 + i * 24 + 16))

        walk(off + 48)
        return res

    zooms = []
    for z in range(u("H", 6)):
        hdr = 64 + 24 * z
        recs = []
        for off, size in rtree_blocks(u("Q", hdr + 16)):
            data = zlib.decompress(raw[off:off + size])
            recs.extend(struct.iter_unpack("<IIIIffff", data))
        zooms.append((u("I", hdr), np.array(recs, dtype=np.float64).reshape(-1, 8)))

    so = u("Q", 44)
    summary = (u("Q", so),) + struct.unpack_from("<dddd", raw, so + 8)
    return {"magic": u("I", 0), "version": u("H", 4), "chroms": chroms, "zooms": zooms, "summary": summary}


def make_sparse_db(root, sizes):
    """A per-chromosome database whose sequence files are sparse: only their sizes matter."""
    (root / "tracks").mkdir(parents=True)
    (root / "seq").mkdir()
    (root / "chrom_sizes.txt").write_text("".join(f"{c}\t{n}\n" for c, n in sizes.items()))
    for c, n in sizes.items():
        with open(root / "seq" / f"chr{c}.seq", "wb") as fh:
            fh.truncate(n)
    pm.gdb_init(str(root))


class TestBigWigExport:
    """Tests for gtrack_export_bigwig."""

    def test_dense_track_round_trip(self, tmp_path):
        f = str(tmp_path / "dense.bw")
        pm.gtrack_export_bigwig("dense_track", f)
        with open(f, "rb") as fh:
            assert int.from_bytes(fh.read(4), "little") == 0x888FFC26
        assert_records(f, "dense_track", pm.gintervals_all())

        try:
            pm.gtrack_import("bw_back_dense", "", f, binsize=50)
            back = pm.gextract("bw_back_dense", pm.gintervals_all())["bw_back_dense"].to_numpy()
            orig = pm.gextract("dense_track", pm.gintervals_all())["dense_track"].to_numpy()
            np.testing.assert_array_equal(np.isnan(back), np.isnan(orig))
            np.testing.assert_array_equal(back[~np.isnan(back)].astype(np.float32),
                                          orig[~np.isnan(orig)].astype(np.float32))
        finally:
            pm.gtrack_rm("bw_back_dense", force=True)

    def test_sparse_track_round_trip(self, tmp_path):
        f = str(tmp_path / "sparse.bw")
        pm.gtrack_export_bigwig("sparse_track", f)
        assert_records(f, "sparse_track", pm.gintervals_all())

    def test_expression_over_overlapping_intervals_writes_each_base_once(self, tmp_path):
        f = str(tmp_path / "expr.bw")
        ivs = pd.concat([pm.gintervals("1", 1000, 50000), pm.gintervals("1", 30000, 80000),
                         pm.gintervals("2", 0, 120000), pm.gintervals("X", 5, 99999)], ignore_index=True)
        pm.gtrack_export_bigwig("dense_track * 2 + 1", f, intervals=ivs, iterator=100)
        chrom, start, _, _ = assert_records(f, "dense_track * 2 + 1",
                                            pm.gintervals_canonic(ivs, unify_touching_intervals=False), 100)
        # the partial bins at the ends of the intervals are kept
        assert list(start[chrom == "chrX"][:2]) == [5, 100]

    def test_array_track_and_virtual_track(self, tmp_path):
        f = str(tmp_path / "a.bw")
        pm.gtrack_export_bigwig("array_track", f)
        assert_records(f, "array_track", pm.gintervals_all())

        pm.gvtrack_create("bw_vt", "dense_track", func="max")
        try:
            scope = pm.gintervals(["1", "2"], 0, 33333)
            pm.gtrack_export_bigwig("bw_vt", f, intervals=scope, iterator=37)
            assert_records(f, "bw_vt", scope, 37)
        finally:
            pm.gvtrack_rm("bw_vt")

    def test_summary_and_zoom_levels(self, tmp_path):
        f = str(tmp_path / "dense.bw")
        pm.gtrack_export_bigwig("dense_track", f)
        meta = read_bigwig_meta(f)
        _, start, end, value = expected_records("dense_track", pm.gintervals_all())
        w = (end - start).astype(np.float64)
        v = value.astype(np.float64)

        assert meta["version"] == 4
        all_ = pm.gintervals_all()
        sizes = zip(all_["chrom"], all_["end"], strict=True)
        assert meta["chroms"] == [(c, i, int(n)) for i, (c, n) in enumerate(sizes)]
        np.testing.assert_allclose(meta["summary"], [w.sum(), v.min(), v.max(), (v * w).sum(), (v * v * w).sum()])

        # 16 times the mean record width (50 bp bins), then 4 times larger at each level
        reductions = [r for r, _ in meta["zooms"]]
        assert reductions == [800 * 4**i for i in range(len(reductions))]
        for red, r in meta["zooms"]:
            assert r[:, 3].sum() == w.sum()
            np.testing.assert_allclose(r[:, 6].sum(), (v * w).sum(), rtol=1e-6)
            assert r[:, 4].min() == v.min() and r[:, 5].max() == v.max()
            # each record lies inside one bin of the level's size
            assert (r[:, 1] // red == (r[:, 2] - 1) // red).all()

    def test_chromosomes_in_name_order_when_the_database_order_differs(self, tmp_path):
        make_sparse_db(tmp_path / "unsorted", {"2": 2000, "10": 1500, "1": 1000, "X": 1200, "1_KI270706v1_random": 500})
        all_ = pm.gintervals_all()
        bins = pm.giterator_intervals(intervals=all_, iterator=10)
        rng = np.random.default_rng(1)
        v = np.round(rng.normal(size=len(bins)), 3)
        v[rng.choice(len(v), len(v) // 4, replace=False)] = np.nan   # NaN gaps
        v[(bins["chrom"].astype(str) == "chr10").to_numpy()] = np.nan   # a chromosome without values
        pm.gtrack_create_dense("t", "", bins, v, 10, defval=np.nan)

        f = str(tmp_path / "t.bw")
        pm.gtrack_export_bigwig("t", f)
        names = [str(c) for c in all_["chrom"]]
        meta = read_bigwig_meta(f)
        assert [c for c, _, _ in meta["chroms"]] == sorted(names, key=str.encode)
        assert [i for _, i, _ in meta["chroms"]] == list(range(len(names)))
        chrom, _, _, _ = assert_records(f, "t", all_)
        assert "chr10" not in set(chrom)

        pm.gtrack_import("back", "", f, binsize=10)
        back = pm.gextract("back", all_, iterator=10)["back"].to_numpy()
        orig = pm.gextract("t", all_, iterator=10)["t"].to_numpy()
        np.testing.assert_array_equal(np.isnan(back), np.isnan(orig))
        np.testing.assert_array_equal(back[~np.isnan(back)].astype(np.float32),
                                      orig[~np.isnan(orig)].astype(np.float32))

    def test_coordinates_past_2_31(self, tmp_path):
        make_sparse_db(tmp_path / "large", {"1": 4_200_000_000, "2": 1_000_000})
        starts = np.array([10, 2**31 - 5, 2**31 + 100, 3_500_000_000, 4_200_000_000 - 100])
        ends = starts + np.array([50, 10, 1000, 1_000_000, 100])
        vals = [1.5, -2, 3e10, 0.25, 7]
        pm.gtrack_create_sparse("s", "", pd.DataFrame({"chrom": "chr1", "start": starts, "end": ends}), vals)

        f = str(tmp_path / "s.bw")
        pm.gtrack_export_bigwig("s", f)
        got = pm.gintervals_import_bigwig(f, pm.gintervals_all())
        np.testing.assert_array_equal(got["start"].to_numpy(np.int64), starts)
        np.testing.assert_array_equal(got["end"].to_numpy(np.int64), ends)
        np.testing.assert_array_equal(got["value"].to_numpy().astype(np.float32), np.float32(vals))

        # bins across a batch boundary of the numeric iterator (2^22 bins of 1000 bp)
        from pymisha.track_export import _bigwig_batches

        scope = pm.gintervals("1", 4_190_000_000, 4_200_000_000)
        batches = _bigwig_batches(scope, 1000, pm.gintervals_all(), ["chr1", "chr2"])[0]()
        assert batches[0]["end"].iloc[0] == 1000 * 2**22
        pm.gtrack_export_bigwig("s", f, intervals=scope, iterator=1000)
        assert_records(f, "s", scope, 1000)

    def test_intervals_with_no_values_write_a_file_without_data(self, tmp_path):
        f = str(tmp_path / "empty.bw")
        with pytest.warns(UserWarning, match="no values in the intervals"):
            pm.gtrack_export_bigwig("sparse_track", f, intervals=pm.gintervals("1", 60, 90))
        assert len(pm.gintervals_import_bigwig(f, pm.gintervals_all())) == 0
        meta = read_bigwig_meta(f)
        assert [c for c, _, _ in meta["chroms"]] == ["chr1", "chr2", "chrX"]
        assert meta["zooms"] == [] and meta["summary"][0] == 0

    def test_a_failed_export_leaves_no_file(self, tmp_path):
        f = str(tmp_path / "fail.bw")
        with pytest.raises(Exception):  # noqa: B017
            pm.gtrack_export_bigwig("dense_track + bw_no_such_variable", f)
        assert not os.path.exists(f)

    def test_an_export_that_cannot_open_its_file_keeps_the_file_already_there(self, tmp_path):
        f = tmp_path / "readonly.bw"
        f.write_text("keep me")
        f.chmod(0o444)
        try:
            if os.access(f, os.W_OK):
                pytest.skip("file is writable (running as root)")
            with pytest.raises(Exception):  # noqa: B017
                pm.gtrack_export_bigwig("dense_track", str(f))
            assert f.read_text() == "keep me"
        finally:
            f.chmod(0o644)

    def test_the_writer_refuses_unsorted_and_overlapping_records(self, tmp_path):
        from pymisha._shared import _pymisha

        f = str(tmp_path / "w.bw")
        state = _pymisha.pm_bigwig_create(f, ["chr1", "chr2"], np.array([1000, 1000]))

        def append(chromids, starts, ends, values):
            _pymisha.pm_bigwig_append(f, state, np.array(chromids), np.array(starts), np.array(ends), np.array(values))

        with pytest.raises(Exception, match="must not overlap"):
            append([0, 0], [0, 50], [100, 200], [1.0, 2.0])
        with pytest.raises(Exception, match="sorted by chromosome"):
            append([1, 0], [0, 0], [10, 10], [1.0, 2.0])
        with pytest.raises(Exception, match="1000 bases long"):
            append([0], [0], [1001], [1.0])
        with pytest.raises(Exception, match="sorted by name"):
            _pymisha.pm_bigwig_create(f, ["chr2", "chr1"], np.array([1000, 1000]))

    def test_bigwig_export_when_the_genome_order_is_not_the_c_order_of_the_chromosome_names(self, tmp_path):
        from test_gtrack_import_bigwig import make_db

        sizes = {"1": 1000, "2": 1000, "10": 1000}
        make_db(tmp_path / "num_db", sizes)
        if pm.gintervals_all()["chrom"].astype(str).tolist() != list(sizes):
            pytest.skip("the database sorts its chromosomes")
        pm.gtrack_create_dense("t", "", pm.gintervals_all(), [1.0, 2.0, 3.0], 100)
        f = str(tmp_path / "t.bw")
        pm.gtrack_export_bigwig("t", f)
        assert [c for c, _, _ in read_bigwig_meta(f)["chroms"]] == ["1", "10", "2"]
        pm.gtrack_import("back", "", f, binsize=100)
        np.testing.assert_array_equal(pm.gextract("back", pm.gintervals_all(), iterator=100)["back"],
                                      pm.gextract("t", pm.gintervals_all(), iterator=100)["t"])

    def test_bigwig_errors_on_2d_track(self, tmp_path):
        with pytest.raises(ValueError, match="2D tracks are not supported"):
            pm.gtrack_export_bigwig("rects_track", str(tmp_path / "r.bw"))

    def test_bigwig_nonexistent_directory_raises(self, tmp_path):
        """Writing BigWig to a nonexistent directory raises FileNotFoundError."""
        outfile = str(tmp_path / "nonexistent" / "dense.bw")
        with pytest.raises(FileNotFoundError):
            pm.gtrack_export_bigwig("dense_track", outfile)
