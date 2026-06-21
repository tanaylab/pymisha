"""Direct unit tests for _pymisha.pm_read_source_track_1d (G1.P3.A)."""

import os
import struct
from pathlib import Path

import _pymisha
import numpy as np
import pandas as pd
import pytest

from _dbpath import TESTDB_ROOT
TEST_DB = TESTDB_ROOT
DENSE_TRACK = TEST_DB / "tracks" / "dense_track.track"
SPARSE_TRACK = TEST_DB / "tracks" / "sparse_track.track"


def _call(path):
    """Invoke the C++ reader and return (track_type, df_dict)."""
    return _pymisha.pm_read_source_track_1d(str(path))


def _make_dense_payload(bin_size: int, vals: list[float]) -> bytes:
    """Build a dense per-chrom payload: int32 bin_size + float32 vals."""
    out = struct.pack("<i", bin_size)
    out += b"".join(struct.pack("<f", float(v)) for v in vals)
    return out


def _make_sparse_payload_64(records: list[tuple[int, int, float]]) -> bytes:
    """Build a sparse payload with 64-bit start/end: int32 sig=-1 + (i8,i8,f4)*N."""
    out = struct.pack("<i", -1)
    for s, e, v in records:
        out += struct.pack("<qqf", int(s), int(e), float(v))
    return out


def _make_sparse_payload_32(records: list[tuple[int, int, float]]) -> bytes:
    """Build a sparse payload with 32-bit start/end: int32 sig=-1 + (i4,i4,f4)*N."""
    out = struct.pack("<i", -1)
    for s, e, v in records:
        out += struct.pack("<iif", int(s), int(e), float(v))
    return out


_TRACK_IDX_MAGIC = b"MISHATDX"
_TRACK_IDX_VERSION = 1
_TRACK_IDX_FLAG_LITTLE_ENDIAN = 0x01
_TRACK_TYPE_DENSE = 0    # MishaTrackType::DENSE
_TRACK_TYPE_SPARSE = 1   # MishaTrackType::SPARSE


def _make_indexed_dir(
    tmp_path: Path,
    *,
    chroms: list[tuple[str, int]],  # [(chrom_name, chrom_size), ...]
    track_subdir: str,
    payloads: dict[int, bytes],
    track_type_raw: int,
) -> Path:
    """Construct a fake DB layout: tmp_path/db/tracks/<track_subdir>/{track.idx,track.dat}.

    Returns the track subdirectory path.
    """
    from pymisha.liftover import _compute_track_idx_checksum

    db = tmp_path / "db"
    tracks = db / "tracks"
    track_dir = tracks / track_subdir
    track_dir.mkdir(parents=True)

    # chrom_sizes.txt
    with open(db / "chrom_sizes.txt", "w") as fh:
        for name, size in chroms:
            fh.write(f"{name}\t{size}\n")

    # Build entries (chrom_id ordered, offset packed sequentially)
    entries = []
    blob = b""
    for chrom_id, _name_size in enumerate(chroms):
        payload = payloads.get(chrom_id, b"")
        offset = len(blob)
        length = len(payload)
        entries.append((chrom_id, offset, length, 0))
        blob += payload

    checksum = _compute_track_idx_checksum(entries)

    with open(track_dir / "track.idx", "wb") as fh:
        fh.write(_TRACK_IDX_MAGIC)
        fh.write(struct.pack("<I", _TRACK_IDX_VERSION))
        fh.write(struct.pack("<I", track_type_raw))
        fh.write(struct.pack("<I", len(chroms)))
        fh.write(struct.pack("<Q", _TRACK_IDX_FLAG_LITTLE_ENDIAN))
        fh.write(struct.pack("<Q", checksum))
        for chrom_id, offset, length, reserved in entries:
            fh.write(struct.pack("<IQQI", chrom_id, offset, length, reserved))

    with open(track_dir / "track.dat", "wb") as fh:
        fh.write(blob)

    return track_dir


class TestDensePerChrom:
    """Read a dense per-chrom directory built from scratch."""

    def test_single_chrom_three_bins(self, tmp_path):
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        bin_size = 100
        (track_dir / "1").write_bytes(_make_dense_payload(bin_size, [0.5, 1.0, -2.5]))

        track_type, df = _call(track_dir)

        assert track_type == "dense"
        assert isinstance(df, dict)
        assert list(df.keys()) == ["chrom", "start", "end", "value"]
        assert len(df["chrom"]) == 3
        assert df["chrom"][0] == "1"
        assert df["chrom"][1] == "1"
        assert df["chrom"][2] == "1"
        np.testing.assert_array_equal(df["start"], [0, 100, 200])
        np.testing.assert_array_equal(df["end"], [100, 200, 300])
        np.testing.assert_array_almost_equal(df["value"], [0.5, 1.0, -2.5])

    def test_drops_nan_and_inf(self, tmp_path):
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        (track_dir / "1").write_bytes(
            _make_dense_payload(100, [0.5, float("nan"), 1.0, float("inf"), 2.0])
        )

        track_type, df = _call(track_dir)

        assert track_type == "dense"
        assert len(df["chrom"]) == 3
        np.testing.assert_array_equal(df["start"], [0, 200, 400])
        np.testing.assert_array_equal(df["end"], [100, 300, 500])
        np.testing.assert_array_almost_equal(df["value"], [0.5, 1.0, 2.0])

    def test_dtypes(self, tmp_path):
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        (track_dir / "1").write_bytes(_make_dense_payload(100, [0.5]))

        _, df = _call(track_dir)
        assert df["chrom"].dtype == np.object_
        assert df["start"].dtype == np.int64
        assert df["end"].dtype == np.int64
        assert df["value"].dtype == np.float64

    def test_multiple_chroms(self, tmp_path):
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        (track_dir / "1").write_bytes(_make_dense_payload(100, [0.5, 1.5]))
        (track_dir / "X").write_bytes(_make_dense_payload(100, [2.5]))

        track_type, df = _call(track_dir)
        assert track_type == "dense"
        assert len(df["chrom"]) == 3
        # Order depends on sorted(os.listdir), so "1" then "X".
        np.testing.assert_array_equal(df["chrom"], ["1", "1", "X"])
        np.testing.assert_array_almost_equal(df["value"], [0.5, 1.5, 2.5])


class TestSparsePerChrom:
    """Read a sparse per-chrom directory built from scratch."""

    def test_64bit_records(self, tmp_path):
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        records = [(100, 200, 0.5), (300, 500, 1.5), (1000, 1500, -2.5)]
        (track_dir / "1").write_bytes(_make_sparse_payload_64(records))

        track_type, df = _call(track_dir)

        assert track_type == "sparse"
        assert len(df["chrom"]) == 3
        np.testing.assert_array_equal(df["chrom"], ["1", "1", "1"])
        np.testing.assert_array_equal(df["start"], [100, 300, 1000])
        np.testing.assert_array_equal(df["end"], [200, 500, 1500])
        np.testing.assert_array_almost_equal(df["value"], [0.5, 1.5, -2.5])

    def test_32bit_records_when_only_32_validates(self, tmp_path):
        """Use 32-bit decode when payload length only validates as 32-bit."""
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        # 4 (sig) + 12 (one 32-bit rec) = 16 bytes. 16-4=12, 12 % 20 != 0
        # (20-byte rec), 12 % 12 == 0. Only 32-bit layout matches.
        rec = struct.pack("<iif", 100, 200, 0.5)
        (track_dir / "1").write_bytes(struct.pack("<i", -1) + rec)

        track_type, df = _call(track_dir)
        assert track_type == "sparse"
        assert len(df["chrom"]) == 1
        np.testing.assert_array_equal(df["start"], [100])
        np.testing.assert_array_equal(df["end"], [200])

    def test_prefer_64bit_when_both_validate(self, tmp_path):
        """When length divides both 12 and 20, prefer the 64-bit decode."""
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        # 5 × 12 = 60 bytes of records → divisible by both 12 and 20.
        # Pack such that 64-bit decode gives sensible values.
        records_64 = [(0, 100, 0.5), (200, 300, 1.0), (400, 500, 1.5)]
        body64 = b"".join(struct.pack("<qqf", s, e, v) for s, e, v in records_64)
        # 3 × 20 = 60. Confirm divisibility.
        assert len(body64) % 20 == 0 and len(body64) % 12 == 0
        (track_dir / "1").write_bytes(struct.pack("<i", -1) + body64)

        track_type, df = _call(track_dir)
        assert track_type == "sparse"
        # Expect 3 64-bit records.
        assert len(df["chrom"]) == 3
        np.testing.assert_array_equal(df["start"], [0, 200, 400])
        np.testing.assert_array_equal(df["end"], [100, 300, 500])
        np.testing.assert_array_almost_equal(df["value"], [0.5, 1.0, 1.5])

    def test_drops_nan_records(self, tmp_path):
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        records = [(100, 200, 0.5), (300, 400, float("nan")), (500, 600, 1.5)]
        (track_dir / "1").write_bytes(_make_sparse_payload_64(records))

        _, df = _call(track_dir)
        assert len(df["chrom"]) == 2
        np.testing.assert_array_equal(df["start"], [100, 500])

    def test_corrupt_length_raises(self, tmp_path):
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        # 4 (sig) + 7 bytes body, not divisible by either 12 or 20.
        (track_dir / "1").write_bytes(struct.pack("<i", -1) + b"abcdefg")

        with pytest.raises(ValueError, match="Corrupt sparse track payload length"):
            _call(track_dir)

    def test_corrupt_sparse_records_raises(self, tmp_path):
        """Body divides both 12 and 20 but records are corrupt (negative starts)."""
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        # 60 bytes (= 3*20 = 5*12) of negative-start records.
        # Use 64-bit packing with negative starts; 32-bit re-interpretation
        # also yields negatives because all bytes are 0xFF / large negative.
        body = b""
        for _ in range(3):
            body += struct.pack("<qqf", -1, -1, 0.5)
        assert len(body) == 60
        (track_dir / "1").write_bytes(struct.pack("<i", -1) + body)
        with pytest.raises(ValueError, match="Corrupt sparse track payload records"):
            _call(track_dir)

    def test_empty_sparse_body(self, tmp_path):
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        (track_dir / "1").write_bytes(struct.pack("<i", -1))

        track_type, df = _call(track_dir)
        assert track_type == "sparse"
        assert len(df["chrom"]) == 0


class TestMixed:
    def test_mixed_dense_sparse_raises(self, tmp_path):
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        (track_dir / "1").write_bytes(_make_dense_payload(100, [0.5]))
        (track_dir / "X").write_bytes(_make_sparse_payload_64([(0, 100, 0.5)]))

        with pytest.raises(ValueError, match="Mixed dense/sparse"):
            _call(track_dir)

    def test_skips_hidden_files(self, tmp_path):
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        (track_dir / ".attributes").write_text("hidden=1\n")
        (track_dir / "1").write_bytes(_make_dense_payload(100, [0.5]))

        track_type, df = _call(track_dir)
        assert track_type == "dense"
        assert len(df["chrom"]) == 1

    def test_indexed_with_vars_subdir(self, tmp_path):
        """Indexed track with a 'vars' subdirectory must use the indexed path."""
        payloads = {
            0: _make_dense_payload(100, [0.5, 1.0]),
            1: _make_dense_payload(100, [2.0]),
        }
        track_dir = _make_indexed_dir(
            tmp_path,
            chroms=[("chr1", 1000), ("chr2", 500)],
            track_subdir="t",
            payloads=payloads,
            track_type_raw=_TRACK_TYPE_DENSE,
        )
        # Simulate a 'vars' subdirectory (as created by gtrack_convert_to_indexed).
        (track_dir / "vars").mkdir()

        track_type, df = _call(track_dir)
        assert track_type == "dense"
        assert len(df["chrom"]) == 3  # 2 bins from chr1 + 1 from chr2


class TestIndexedDense:
    def test_two_chrom_dense(self, tmp_path):
        payloads = {
            0: _make_dense_payload(100, [0.5, 1.0, 2.0]),
            1: _make_dense_payload(100, [3.0]),
        }
        track_dir = _make_indexed_dir(
            tmp_path,
            chroms=[("chr1", 1000), ("chr2", 500)],
            track_subdir="my_track",
            payloads=payloads,
            track_type_raw=_TRACK_TYPE_DENSE,
        )

        track_type, df = _call(track_dir)

        assert track_type == "dense"
        assert len(df["chrom"]) == 4
        np.testing.assert_array_equal(df["chrom"], ["chr1", "chr1", "chr1", "chr2"])
        np.testing.assert_array_equal(df["start"], [0, 100, 200, 0])
        np.testing.assert_array_equal(df["end"], [100, 200, 300, 100])
        np.testing.assert_array_almost_equal(df["value"], [0.5, 1.0, 2.0, 3.0])

    def test_skips_zero_length_entries(self, tmp_path):
        payloads = {
            0: _make_dense_payload(100, [0.5]),
            # chrom_id=1 has length=0 (skipped)
            2: _make_dense_payload(100, [2.0]),
        }
        track_dir = _make_indexed_dir(
            tmp_path,
            chroms=[("chr1", 500), ("chr2", 500), ("chrX", 500)],
            track_subdir="t",
            payloads=payloads,
            track_type_raw=_TRACK_TYPE_DENSE,
        )

        _, df = _call(track_dir)
        np.testing.assert_array_equal(df["chrom"], ["chr1", "chrX"])


class TestIndexedSparse:
    def test_sparse_two_chroms(self, tmp_path):
        payloads = {
            0: _make_sparse_payload_64([(0, 100, 0.5), (200, 300, 1.0)]),
            1: _make_sparse_payload_64([(0, 500, 2.0)]),
        }
        track_dir = _make_indexed_dir(
            tmp_path,
            chroms=[("chr1", 1000), ("chr2", 500)],
            track_subdir="t",
            payloads=payloads,
            track_type_raw=_TRACK_TYPE_SPARSE,
        )

        track_type, df = _call(track_dir)
        assert track_type == "sparse"
        assert len(df["chrom"]) == 3
        np.testing.assert_array_equal(df["chrom"], ["chr1", "chr1", "chr2"])
        np.testing.assert_array_equal(df["start"], [0, 200, 0])
        np.testing.assert_array_equal(df["end"], [100, 300, 500])


class TestIndexedErrors:
    def test_bad_magic_raises(self, tmp_path):
        track_dir = _make_indexed_dir(
            tmp_path, chroms=[("chr1", 500)], track_subdir="t",
            payloads={0: _make_sparse_payload_64([(0, 100, 0.5)])},
            track_type_raw=_TRACK_TYPE_SPARSE,
        )
        # Stomp the magic.
        with open(track_dir / "track.idx", "r+b") as fh:
            fh.seek(0)
            fh.write(b"NOTHEMAG")

        with pytest.raises(ValueError, match="Invalid track index header"):
            _call(track_dir)

    def test_bad_version_raises(self, tmp_path):
        track_dir = _make_indexed_dir(
            tmp_path, chroms=[("chr1", 500)], track_subdir="t",
            payloads={0: _make_sparse_payload_64([(0, 100, 0.5)])},
            track_type_raw=_TRACK_TYPE_SPARSE,
        )
        with open(track_dir / "track.idx", "r+b") as fh:
            fh.seek(8)
            fh.write(struct.pack("<I", 99))

        with pytest.raises(ValueError, match="Unsupported track index version"):
            _call(track_dir)

    def test_checksum_mismatch_raises(self, tmp_path):
        track_dir = _make_indexed_dir(
            tmp_path, chroms=[("chr1", 500)], track_subdir="t",
            payloads={0: _make_sparse_payload_64([(0, 100, 0.5)])},
            track_type_raw=_TRACK_TYPE_SPARSE,
        )
        with open(track_dir / "track.idx", "r+b") as fh:
            fh.seek(28)  # checksum is at offset 8+4+4+4+8 = 28
            fh.write(struct.pack("<Q", 0xDEADBEEFDEADBEEF))

        with pytest.raises(ValueError, match="checksum mismatch"):
            _call(track_dir)

    def test_missing_chrom_sizes_raises(self, tmp_path):
        track_dir = _make_indexed_dir(
            tmp_path, chroms=[("chr1", 500)], track_subdir="t",
            payloads={0: _make_sparse_payload_64([(0, 100, 0.5)])},
            track_type_raw=_TRACK_TYPE_SPARSE,
        )
        (tmp_path / "db" / "chrom_sizes.txt").unlink()

        with pytest.raises(ValueError, match="chrom_sizes.txt"):
            _call(track_dir)


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------

def _python_read(src_dir) -> tuple[str, pd.DataFrame]:
    """Run the pure-Python implementation via the env-var fallback path."""
    import pymisha.liftover as lf
    os.environ["PYMISHA_FORCE_PY_READ_SOURCE_TRACK"] = "1"
    try:
        return lf._read_source_track(str(src_dir))
    finally:
        del os.environ["PYMISHA_FORCE_PY_READ_SOURCE_TRACK"]


def _cpp_read(src_dir) -> tuple[str, pd.DataFrame]:
    """Run the C++ implementation through the dispatcher (env var unset)."""
    os.environ.pop("PYMISHA_FORCE_PY_READ_SOURCE_TRACK", None)
    import pymisha.liftover as lf
    return lf._read_source_track(str(src_dir))


class TestCrossValidatePython:
    """Cell-by-cell parity vs the renamed _read_source_track_python."""

    @pytest.mark.parametrize("track_name", ["dense_track.track", "sparse_track.track"])
    def test_testdb_track(self, track_name):
        track_dir = TEST_DB / "tracks" / track_name
        if not track_dir.exists():
            pytest.skip(f"{track_dir} not found")

        py_type, py_df = _python_read(track_dir)
        cpp_type, cpp_df = _cpp_read(track_dir)

        assert py_type == cpp_type
        # Reset index because Python may have built rows in a different but
        # deterministic order; sort both by (chrom, start, end) before diffing.
        py_df = py_df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
        cpp_df = cpp_df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(py_df, cpp_df, check_dtype=True)

    def test_synthetic_dense(self, tmp_path):
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        (track_dir / "1").write_bytes(
            _make_dense_payload(50, [0.5, float("nan"), 1.0, float("inf"), 2.0, -3.0])
        )
        (track_dir / "X").write_bytes(_make_dense_payload(50, [4.0, float("nan")]))

        py_type, py_df = _python_read(track_dir)
        cpp_type, cpp_df = _cpp_read(track_dir)

        assert py_type == cpp_type
        py_df = py_df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
        cpp_df = cpp_df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(py_df, cpp_df, check_dtype=True)

    def test_synthetic_sparse_64bit(self, tmp_path):
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        recs = [(100, 200, 0.5), (300, 500, 1.5), (1000, 1500, -2.5), (2000, 2001, float("nan"))]
        (track_dir / "1").write_bytes(_make_sparse_payload_64(recs))
        (track_dir / "X").write_bytes(_make_sparse_payload_64([(0, 50, 5.0)]))

        py_type, py_df = _python_read(track_dir)
        cpp_type, cpp_df = _cpp_read(track_dir)

        assert py_type == cpp_type
        py_df = py_df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
        cpp_df = cpp_df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(py_df, cpp_df, check_dtype=True)

    def test_synthetic_indexed_dense(self, tmp_path):
        payloads = {
            0: _make_dense_payload(100, [0.5, float("nan"), 1.0]),
            1: _make_dense_payload(100, [2.0]),
        }
        track_dir = _make_indexed_dir(
            tmp_path,
            chroms=[("chr1", 1000), ("chr2", 500), ("chrX", 800)],
            track_subdir="t",
            payloads=payloads,
            track_type_raw=_TRACK_TYPE_DENSE,
        )

        py_type, py_df = _python_read(track_dir)
        cpp_type, cpp_df = _cpp_read(track_dir)

        assert py_type == cpp_type
        py_df = py_df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
        cpp_df = cpp_df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(py_df, cpp_df, check_dtype=True)


class TestEdgeCases:
    def test_missing_directory_raises(self, tmp_path):
        # Through the Python dispatcher we get ValueError("Source track directory does not exist")
        import pymisha.liftover as lf
        with pytest.raises(ValueError, match="does not exist"):
            lf._read_source_track(str(tmp_path / "nope"))

    def test_completely_empty_dir(self, tmp_path):
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        track_type, df = _call(track_dir)
        assert track_type == "sparse"  # fallback
        assert len(df["chrom"]) == 0

    def test_only_hidden_files(self, tmp_path):
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        (track_dir / ".attributes").write_text("name=foo\n")
        (track_dir / ".gitignore").write_text("ignored\n")
        track_type, df = _call(track_dir)
        assert len(df["chrom"]) == 0

    def test_small_file_skipped(self, tmp_path):
        """Files with <4 bytes are silently skipped (no signature parsable)."""
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        (track_dir / "1").write_bytes(b"abc")  # 3 bytes
        (track_dir / "2").write_bytes(_make_dense_payload(100, [0.5]))
        track_type, df = _call(track_dir)
        assert track_type == "dense"
        assert len(df["chrom"]) == 1
        assert df["chrom"][0] == "2"

    def test_unknown_signature_silently_skipped(self, tmp_path):
        """Sig that is neither >0 nor -1 is ignored (matches Python's if/elif)."""
        track_dir = tmp_path / "t"
        track_dir.mkdir()
        (track_dir / "1").write_bytes(struct.pack("<i", -5) + b"\x00" * 20)
        (track_dir / "2").write_bytes(_make_sparse_payload_64([(0, 100, 0.5)]))
        track_type, df = _call(track_dir)
        assert track_type == "sparse"
        assert len(df["chrom"]) == 1

    def test_indexed_only_does_not_iterate_perchrom(self, tmp_path):
        """If track.idx+track.dat are present AND there are per-chrom files,
        Python iterates the per-chrom files (current behavior). Cross-validate."""
        payloads = {0: _make_dense_payload(100, [0.5])}
        track_dir = _make_indexed_dir(
            tmp_path,
            chroms=[("chr1", 500)],
            track_subdir="t",
            payloads=payloads,
            track_type_raw=_TRACK_TYPE_DENSE,
        )
        # Also drop a per-chrom file that should win.
        (track_dir / "extra1").write_bytes(_make_dense_payload(100, [9.5]))
        py_type, py_df = _python_read(track_dir)
        cpp_type, cpp_df = _cpp_read(track_dir)
        py_df = py_df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
        cpp_df = cpp_df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
        assert py_type == cpp_type
        pd.testing.assert_frame_equal(py_df, cpp_df, check_dtype=True)
