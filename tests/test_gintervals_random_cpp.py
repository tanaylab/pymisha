"""Tests for the C++ port of gintervals_random.

The C++ path is exercised both directly (via ``_pymisha.pm_intervals_random``)
and through the public ``gintervals_random`` router. The router heuristic
dispatches to C++ when the genome has >1000 contigs OR >10M total bp, so
tests use a large synthetic groot to force the C++ branch.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import _pymisha
import numpy as np
import pandas as pd

import pymisha as pm
from pymisha._shared import _df2pymisha

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


def _restore_test_db() -> None:
    if TEST_DB.exists():
        pm.gdb_init(str(TEST_DB))


def _build_groot(path: Path, num_chroms: int, chrom_size: int = 100) -> None:
    fasta = path.with_suffix(".fa")
    base_seq = "ACGT" * (chrom_size // 4)
    lines: list[str] = []
    for i in range(num_chroms):
        lines.append(f">c{i}")
        lines.append(base_seq)
    fasta.write_text("\n".join(lines) + "\n")
    pm.gdb_create(str(path), str(fasta), db_format="indexed", verbose=False)


# ---------- 1. Correctness: small genome, no filter --------------------------
def test_cpp_correctness_no_filter():
    """C++ direct call: shape, count, bounds within the test DB."""
    genome = pm.gintervals_all()
    genome_pm = _df2pymisha(genome[["chrom", "start", "end"]])

    n = 200
    size = 50
    res = _pymisha.pm_intervals_random(size, n, 0, genome_pm, None, 60427)

    df = pd.DataFrame(res)
    assert list(df.columns) == ["chrom", "start", "end"]
    assert len(df) == n
    assert (df["end"] - df["start"] == size).all()

    # All intervals must fit inside the genome.
    for chrom, sub in df.groupby("chrom"):
        crow = genome[genome["chrom"] == chrom].iloc[0]
        assert (sub["start"] >= int(crow["start"])).all()
        assert (sub["end"] <= int(crow["end"])).all()


# ---------- 2. Filter respected ---------------------------------------------
def test_cpp_filter_no_overlap():
    """No sampled interval overlaps any filter interval."""
    genome = pm.gintervals_all()
    genome_pm = _df2pymisha(genome[["chrom", "start", "end"]])

    # Mask the middle 60% of chrom "1" (which is 0..500000).
    filter_df = pd.DataFrame({
        "chrom": ["1"],
        "start": [100_000],
        "end": [400_000],
    })
    filter_pm = _df2pymisha(filter_df)

    n = 500
    size = 100
    res = _pymisha.pm_intervals_random(size, n, 0, genome_pm, filter_pm, 60427)
    df = pd.DataFrame(res)
    assert len(df) == n

    chr1 = df[df["chrom"] == "1"]
    # No sampled interval on chrom 1 may overlap [100_000, 400_000).
    overlaps = ((chr1["start"] < 400_000) & (chr1["end"] > 100_000)).sum()
    assert overlaps == 0, f"{overlaps} sampled intervals overlap the filter"


# ---------- 3. Deterministic within C++ -------------------------------------
def test_cpp_deterministic_same_seed():
    """Two calls with the same seed produce identical output."""
    genome = pm.gintervals_all()
    genome_pm = _df2pymisha(genome[["chrom", "start", "end"]])

    a = pd.DataFrame(
        _pymisha.pm_intervals_random(100, 300, 0, genome_pm, None, 60427)
    )
    b = pd.DataFrame(
        _pymisha.pm_intervals_random(100, 300, 0, genome_pm, None, 60427)
    )
    pd.testing.assert_frame_equal(a, b)

    # Different seed -> different output (overwhelmingly likely).
    c = pd.DataFrame(
        _pymisha.pm_intervals_random(100, 300, 0, genome_pm, None, 60428)
    )
    assert not a.equals(c)


# ---------- 4. Statistical equivalence vs Python ----------------------------
def test_cpp_python_statistical_equivalence():
    """Per-chrom counts from C++ and Python paths should be within 15%.

    With n=5000 the variance is small enough that both paths should
    converge to similar chrom-level proportions (weighted by length).
    """
    genome = pm.gintervals_all()
    genome_pm = _df2pymisha(genome[["chrom", "start", "end"]])
    n = 5000
    size = 50

    res_cpp = _pymisha.pm_intervals_random(size, n, 0, genome_pm, None, 60427)
    df_cpp = pd.DataFrame(res_cpp)

    np.random.seed(60427)
    from pymisha.intervals import _gintervals_random_python
    df_py = _gintervals_random_python(size, n, 0, genome, None)

    c_cpp = df_cpp["chrom"].value_counts().reindex(genome["chrom"], fill_value=0)
    c_py = df_py["chrom"].value_counts().reindex(genome["chrom"], fill_value=0)

    # Each chrom's count ratio should be within 15% of the expected
    # weight (length-proportional) given n samples.
    weights = (genome["end"] - genome["start"]).astype(float)
    expected = weights / weights.sum() * n
    for chrom in genome["chrom"]:
        e = float(expected[genome["chrom"] == chrom].iloc[0])
        diff_cpp = abs(float(c_cpp[chrom]) - e) / e
        diff_py = abs(float(c_py[chrom]) - e) / e
        assert diff_cpp < 0.15, f"C++ chrom {chrom} count diverges {diff_cpp:.3f}"
        assert diff_py < 0.15, f"Python chrom {chrom} count diverges {diff_py:.3f}"


# ---------- 5. Auto-routing: large synthetic genome forces C++ --------------
def test_router_dispatches_cpp_on_large_genome(tmp_path):
    """gintervals_random on a >1000-contig groot must call the C++ helper."""
    try:
        groot = tmp_path / "big"
        _build_groot(groot, num_chroms=1500, chrom_size=400)
        pm.gdb_init(str(groot))

        # Patch the python path so any accidental fallback raises loudly.
        from pymisha import intervals as ivmod
        called = {"cpp": 0, "py": 0}
        orig_cpp = ivmod._gintervals_random_cpp
        orig_py = ivmod._gintervals_random_python

        def spy_cpp(*a, **kw):
            called["cpp"] += 1
            return orig_cpp(*a, **kw)

        def spy_py(*a, **kw):  # pragma: no cover - should not be called
            called["py"] += 1
            return orig_py(*a, **kw)

        with patch.object(ivmod, "_gintervals_random_cpp", spy_cpp), \
             patch.object(ivmod, "_gintervals_random_python", spy_py):
            df = pm.gintervals_random(50, 100, dist_from_edge=0, seed=60427)

        assert called["cpp"] == 1
        assert called["py"] == 0
        assert len(df) == 100
        assert (df["end"] - df["start"] == 50).all()
    finally:
        _restore_test_db()


def test_router_dispatches_python_on_small_genome():
    """Small DB (3 contigs, ~1Mbp) routes to Python by default."""
    from pymisha import intervals as ivmod
    called = {"cpp": 0, "py": 0}
    orig_cpp = ivmod._gintervals_random_cpp
    orig_py = ivmod._gintervals_random_python

    def spy_cpp(*a, **kw):  # pragma: no cover - should not be called
        called["cpp"] += 1
        return orig_cpp(*a, **kw)

    def spy_py(*a, **kw):
        called["py"] += 1
        return orig_py(*a, **kw)

    np.random.seed(60427)
    with patch.object(ivmod, "_gintervals_random_cpp", spy_cpp), \
         patch.object(ivmod, "_gintervals_random_python", spy_py):
        df = pm.gintervals_random(50, 30, dist_from_edge=0)

    assert called["py"] == 1
    assert called["cpp"] == 0
    assert len(df) == 30


# ---------- 6. Edge case: contig exactly size+2*dist_from_edge --------------
def test_cpp_edge_case_minimum_contig(tmp_path):
    """Chromosome whose length == size + 2*dist_from_edge yields one valid start.

    Covers R 5.6.30 commit 1b41bceb's fix: a contig of length L=size+2*dfe
    must produce exactly one valid start (L - 2*dfe - size + 1 = 1).
    """
    try:
        groot = tmp_path / "g"
        # size=20, dfe=10 -> contig of length 40 has exactly 1 valid start (pos 10).
        size = 20
        dfe = 10
        chrom_len = size + 2 * dfe  # 40
        fasta = groot.with_suffix(".fa")
        fasta.write_text(">a\n" + "ACGT" * (chrom_len // 4) + "\n")
        pm.gdb_create(str(groot), str(fasta), db_format="indexed", verbose=False)
        pm.gdb_init(str(groot))

        genome = pm.gintervals_all()
        genome_pm = _df2pymisha(genome[["chrom", "start", "end"]])

        # Many draws should all yield start == 10, end == 30.
        res = _pymisha.pm_intervals_random(size, 200, dfe, genome_pm, None, 60427)
        df = pd.DataFrame(res)
        assert len(df) == 200
        assert (df["start"] == 10).all(), \
            f"Expected start==10 for all rows, got unique starts {df['start'].unique()}"
        assert (df["end"] == 30).all()
    finally:
        _restore_test_db()


# ---------- Additional smoke tests ------------------------------------------
def test_cpp_filter_unknown_chrom_skipped():
    """Filter rows on unknown chromosomes are ignored, not errors."""
    genome = pm.gintervals_all()
    genome_pm = _df2pymisha(genome[["chrom", "start", "end"]])

    # NB: this filter mentions a chrom that doesn't exist in the test DB.
    filter_df = pd.DataFrame({
        "chrom": ["nonexistent_chrom"],
        "start": [0],
        "end": [50],
    })
    filter_pm = _df2pymisha(filter_df)
    res = _pymisha.pm_intervals_random(50, 30, 0, genome_pm, filter_pm, 60427)
    assert len(pd.DataFrame(res)) == 30


def test_cpp_filter_left_expansion():
    """A filter at [start, end) excludes start positions in [start-size+1, end).

    Verifies the size-1 left-expansion: an interval of length L can start as
    late as ``end - 1`` and still overlap a filter region (overlap iff
    ``start < f.end`` and ``start + L > f.start``).
    """
    genome = pm.gintervals_all()
    genome_pm = _df2pymisha(genome[["chrom", "start", "end"]])

    # Mask [200_000, 200_001) on chrom "1". With size=100 a sampled start
    # in [199_901, 200_001) would still overlap the masked base. None of the
    # sampled intervals should overlap the masked base.
    filter_df = pd.DataFrame({"chrom": ["1"], "start": [200_000], "end": [200_001]})
    filter_pm = _df2pymisha(filter_df)
    res = _pymisha.pm_intervals_random(100, 2000, 0, genome_pm, filter_pm, 60427)
    df = pd.DataFrame(res)
    chr1 = df[df["chrom"] == "1"]
    # Overlap: start < 200_001 AND end > 200_000
    overlaps = ((chr1["start"] < 200_001) & (chr1["end"] > 200_000)).sum()
    assert overlaps == 0


def test_cpp_router_seed_reproducibility(tmp_path):
    """Through the router, same seed -> same output on a large genome."""
    try:
        groot = tmp_path / "big2"
        _build_groot(groot, num_chroms=1100, chrom_size=200)
        pm.gdb_init(str(groot))

        df1 = pm.gintervals_random(20, 200, dist_from_edge=0, seed=12345)
        df2 = pm.gintervals_random(20, 200, dist_from_edge=0, seed=12345)
        pd.testing.assert_frame_equal(df1, df2)

        df3 = pm.gintervals_random(20, 200, dist_from_edge=0, seed=12346)
        assert not df1.equals(df3)
    finally:
        _restore_test_db()
