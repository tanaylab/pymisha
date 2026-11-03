"""2D track files carry no uninitialised struct padding.

Ported from R misha's tests/testthat/test-2d-track-padding.R. The quad-tree records
and nodes are written to the file verbatim, padding included, so the padding must be
zero: otherwise a track is not byte-reproducible and memory leaks into its files.
The last test is pymisha's own: it checks the internal nodes, whose padding clang
left undefined until both packages zeroed it.

The record sizes double as a pin on the on-disk layout.
"""

from __future__ import annotations

import struct

import numpy as np
import pandas as pd

import pymisha as pm

# Layout of a 2D quad-tree track file that holds a single leaf
QT_HEADER = 4 + 8 + 8  # format signature, number of objects, root chunk position
QT_CHUNK_HEADER = 8 + 8  # chunk size, offset of the chunk's top node
QT_LEAF = 80  # is_leaf + padding + stat + arena + num_objs + padding
QT_RECT_OBJ = 48  # id + x1,y1,x2,y2 + float value + 4 padding bytes
QT_POINT_OBJ = 32  # id + x,y + float value + 4 padding bytes
QT_OBJS_START = QT_HEADER + QT_CHUNK_HEADER + QT_LEAF
QT_PAD = 4  # tail padding of every object record


def _i64(r, off):
    return struct.unpack_from("<q", r, off)[0]


def _f32(r, off):
    return struct.unpack_from("<f", r, off)[0]


def _pad_test_db(tmp_path):
    """Two 2000 bp chromosomes in the per-chromosome layout (an indexed database moves
    the same bytes into one .dat, which would only make them harder to address)."""
    fastas = []
    for chrom in ("chr1", "chr2"):
        fasta = tmp_path / f"{chrom}.fa"
        fasta.write_text(f">{chrom}\n{'A' * 2000}\n")
        fastas.append(str(fasta))
    pm.gdb_create(str(tmp_path / "db"), fastas, db_format="per-chromosome")
    pm.gdb_init(str(tmp_path / "db"))
    return tmp_path / "db"


def _pair_file(db, track, pair="chr1-chr2"):
    return (db / "tracks" / f"{track.replace('.', '/')}.track" / pair).read_bytes()


def test_rectangle_records_of_a_2d_track_carry_no_uninitialised_padding(tmp_path):
    db = _pad_test_db(tmp_path)
    x1 = np.array([100, 300, 500, 700, 900])
    y1 = np.array([150, 350, 550, 750, 950])
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    pm.gtrack_2d_create("pad_rects", "padding regression",
                        pm.gintervals_2d("chr1", x1, x1 + 100, "chr2", y1, y1 + 100), vals)
    r = _pair_file(db, "pad_rects")

    # a handful of rectangles fit in the root leaf, so the layout is fully determined
    assert len(r) == QT_OBJS_START + len(x1) * QT_RECT_OBJ
    assert _i64(r, 4) == len(x1)
    # the leaf header's own padding: after the 1-byte is_leaf and after the 4-byte object count
    assert list(r[37:44]) == [0] * 7
    assert list(r[112:116]) == [0] * 4

    offs = [QT_OBJS_START + i * QT_RECT_OBJ for i in range(len(x1))]
    recs = sorted((_i64(r, o + 8), _i64(r, o + 16), _i64(r, o + 24), _i64(r, o + 32), _f32(r, o + 40)) for o in offs)
    assert [rec[0] for rec in recs] == list(x1)
    assert [rec[2] for rec in recs] == list(x1 + 100)
    assert [rec[1] for rec in recs] == list(y1)
    assert [rec[3] for rec in recs] == list(y1 + 100)
    assert [rec[4] for rec in recs] == vals
    # the four bytes after the float value are structure padding, written to the file
    for i, o in enumerate(offs):
        assert list(r[o + 44:o + 44 + QT_PAD]) == [0] * QT_PAD, f"padding of rectangle record {i + 1}"


def test_point_records_of_a_2d_track_carry_no_uninitialised_padding(tmp_path):
    db = _pad_test_db(tmp_path)
    x = np.array([100, 300, 500])
    y = np.array([150, 350, 550])
    vals = [1.0, 2.0, 3.0]
    # all-1bp intervals make gtrack_2d_import write a POINTS track
    src = tmp_path / "points.txt"
    pd.DataFrame({"chrom1": "chr1", "start1": x, "end1": x + 1,
                  "chrom2": "chr2", "start2": y, "end2": y + 1, "value": vals}).to_csv(src, sep="\t", index=False)
    pm.gtrack_2d_import("pad_points", "padding regression", [str(src)])
    assert pm.gtrack_info("pad_points")["type"] == "points"

    r = _pair_file(db, "pad_points")
    assert len(r) == QT_OBJS_START + len(x) * QT_POINT_OBJ
    assert _i64(r, 4) == len(x)
    offs = [QT_OBJS_START + i * QT_POINT_OBJ for i in range(len(x))]
    recs = sorted((_i64(r, o + 8), _i64(r, o + 16), _f32(r, o + 24)) for o in offs)
    assert [rec[0] for rec in recs] == list(x)
    assert [rec[1] for rec in recs] == list(y)
    assert [rec[2] for rec in recs] == vals
    for i, o in enumerate(offs):
        assert list(r[o + 28:o + 28 + QT_PAD]) == [0] * QT_PAD, f"padding of point record {i + 1}"


def _many_rects():
    rng = np.random.default_rng(60427)
    x1 = np.arange(500) * 4
    y1 = rng.choice(np.arange(0, 1901, 4), 500)
    return x1, y1, np.arange(1, 501) / 3


def test_writing_the_same_2d_track_twice_produces_byte_identical_files(tmp_path):
    db = _pad_test_db(tmp_path)
    x1, y1, vals = _many_rects()
    intervs = pm.gintervals_2d("chr1", x1, x1 + 2, "chr2", y1, y1 + 2)
    pm.gtrack_2d_create("pad_twice1", "padding regression", intervs, vals)
    pm.gtrack_2d_create("pad_twice2", "padding regression", intervs, vals)
    assert _pair_file(db, "pad_twice1") == _pair_file(db, "pad_twice2")

    # the tree spans several leaves: its values survive the round trip unchanged
    got = pm.gextract("pad_twice1", pm.gintervals_2d_all(), colnames=["v"]).sort_values("start1")
    assert len(got) == len(x1)
    np.testing.assert_allclose(got["v"].to_numpy(), vals, rtol=1e-6)


def test_the_nodes_of_a_2d_track_carry_no_uninitialised_padding(tmp_path, monkeypatch):
    """The 7 bytes after a node's 1-byte is_leaf, in the root of a tree that spans several
    leaves and in the root of a tree built in subtrees (gmax.data.size), whose top tree
    is written apart."""
    db = _pad_test_db(tmp_path)
    x1, y1, vals = _many_rects()
    intervs = pm.gintervals_2d("chr1", x1, x1 + 2, "chr2", y1, y1 + 2)
    pm.gtrack_2d_create("pad_tree", "padding regression", intervs, vals)
    monkeypatch.setitem(pm.CONFIG, "max_data_size", 50)
    src = tmp_path / "points.txt"
    pd.DataFrame({"chrom1": "chr1", "start1": x1, "end1": x1 + 1,
                  "chrom2": "chr2", "start2": y1, "end2": y1 + 1, "value": vals}).to_csv(src, sep="\t", index=False)
    pm.gtrack_2d_import("pad_subtrees", "padding regression", [str(src)])

    for track in ("pad_tree", "pad_subtrees"):
        r = _pair_file(db, track)
        root_chunk = _i64(r, 12)
        node = root_chunk + _i64(r, root_chunk + 8)
        assert r[node] == 0, track  # a node, not a leaf
        # after is_leaf (8 bytes with its padding) and the stat (32), the arena: the whole pair
        assert struct.unpack_from("<4q", r, node + 40) == (0, 0, 2000, 2000), track
        assert list(r[node + 1:node + 8]) == [0] * 7, track
