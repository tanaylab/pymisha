"""Tests for the native R-serialize reader (`pymisha._r_serialize`).

Drops the Rscript hard-dependency for legacy bigset metadata and for
R-written track variables. Fixtures are generated on the fly via
``saveRDS`` when an Rscript is available; the test is skipped otherwise
so CI machines without R still pass.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymisha import _r_serialize


_RSCRIPT = shutil.which("Rscript")


def _save_with_r(r_expr: str, out_path: Path) -> None:
    if _RSCRIPT is None:
        pytest.skip("Rscript not available; cannot build R-serialize fixtures")
    script = f"saveRDS({r_expr}, '{out_path}')"
    subprocess.run(
        [_RSCRIPT, "-e", script], check=True, capture_output=True, text=True
    )


def _save_xdr_with_r(r_expr: str, out_path: Path) -> None:
    """Write *r_expr* to *out_path* in uncompressed XDR (no gzip)."""
    if _RSCRIPT is None:
        pytest.skip("Rscript not available; cannot build R-serialize fixtures")
    script = (
        f"con<-file('{out_path}','wb'); "
        f"serialize({r_expr}, con, ascii=FALSE, xdr=TRUE); close(con)"
    )
    subprocess.run(
        [_RSCRIPT, "-e", script], check=True, capture_output=True, text=True
    )


class TestRSerializeReader:
    def test_character_scalar(self, tmp_path):
        _save_with_r('"hello"', tmp_path / "x.rds")
        assert _r_serialize.read(tmp_path / "x.rds") == ["hello"]

    def test_character_vector(self, tmp_path):
        _save_with_r('c("a","b","c")', tmp_path / "x.rds")
        assert _r_serialize.read(tmp_path / "x.rds") == ["a", "b", "c"]

    def test_named_integer_vector(self, tmp_path):
        _save_with_r('c(x=1L, y=2L, z=3L)', tmp_path / "x.rds")
        val = _r_serialize.read(tmp_path / "x.rds")
        assert list(val) == [1, 2, 3]
        assert val.names == ["x", "y", "z"]

    def test_real_vector(self, tmp_path):
        _save_with_r('c(1.5, 2.5, 3.5)', tmp_path / "x.rds")
        np.testing.assert_array_equal(
            _r_serialize.read(tmp_path / "x.rds"), [1.5, 2.5, 3.5]
        )

    def test_logical_vector(self, tmp_path):
        _save_with_r('c(TRUE, FALSE, TRUE)', tmp_path / "x.rds")
        val = _r_serialize.read(tmp_path / "x.rds")
        assert val.dtype == bool
        np.testing.assert_array_equal(val, [True, False, True])

    def test_null(self, tmp_path):
        _save_with_r('NULL', tmp_path / "x.rds")
        assert _r_serialize.read(tmp_path / "x.rds") is None

    def test_named_list(self, tmp_path):
        _save_with_r('list(a=1, b="hi")', tmp_path / "x.rds")
        val = _r_serialize.read(tmp_path / "x.rds")
        assert set(val.keys()) == {"a", "b"}
        assert val["b"] == ["hi"]

    def test_data_frame(self, tmp_path):
        _save_with_r(
            'data.frame(x=1:3, y=c("a","b","c"), stringsAsFactors=FALSE)',
            tmp_path / "x.rds",
        )
        df = _r_serialize.read(tmp_path / "x.rds")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["x", "y"]
        assert list(df["y"]) == ["a", "b", "c"]
        assert list(df["x"]) == [1, 2, 3]

    def test_raw_bytes(self, tmp_path):
        _save_with_r('as.raw(c(1,2,3,255))', tmp_path / "x.rds")
        assert _r_serialize.read(tmp_path / "x.rds") == b"\x01\x02\x03\xff"

    def test_xdr_uncompressed(self, tmp_path):
        """The misha `.meta` and `.colnames` files are written via
        `serialize(con, ascii=FALSE)`, NOT saveRDS - i.e. uncompressed.
        The reader must handle that too.
        """
        _save_xdr_with_r('c("col0","col1","col2")', tmp_path / "raw.bin")
        val = _r_serialize.read(tmp_path / "raw.bin")
        assert val == ["col0", "col1", "col2"]

    def test_meta_like_list_of_dataframes(self, tmp_path):
        """The legacy bigset .meta shape that drove the Rscript dep."""
        _save_xdr_with_r(
            'list(stats=data.frame(chrom="1", n=10L, stringsAsFactors=FALSE),'
            '     zeroline=data.frame(chrom="1", zero=0L, stringsAsFactors=FALSE))',
            tmp_path / "meta.bin",
        )
        obj = _r_serialize.read(tmp_path / "meta.bin")
        assert set(obj.keys()) == {"stats", "zeroline"}
        assert isinstance(obj["stats"], pd.DataFrame)
        assert list(obj["stats"].columns) == ["chrom", "n"]
        assert obj["stats"]["n"].iloc[0] == 10


class TestArrayTrackColnames:
    """The bundled array_track ships a .colnames file in R-serialize
    format. Reading it must not require Rscript."""

    def test_bundled_colnames_decodes(self):
        path = Path(
            "pymisha/examples/trackdb/test/tracks/array_track.track/.colnames"
        )
        assert path.exists()
        names = _r_serialize.read_named_vector(path)
        assert names == [f"col{i}" for i in range(10)]
