"""No-NFS tests for the 2D ``gintervals_neighbors`` implementation.

Validates the per-axis-gap distances, the distance-window filter, the
Manhattan-distance ordering, ``maxneighbors`` and ``na_if_notfound`` against an
independent brute-force reference, and the R-style column layout (intervals2's
colliding columns get a ``1`` suffix; trailing ``dist1`` / ``dist2``).
"""
from __future__ import annotations

import random

import pandas as pd

import pymisha as pm
from pymisha.intervals import _neighbors_2d

_COLS6 = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]


def _axis_gap(qa1, qa2, ta1, ta2):
    if qa1 >= ta2:
        return qa1 - ta2
    if qa2 <= ta1:
        return ta1 - qa2
    return 0


def _brute(i1, i2, maxn, md1, xd1, md2, xd2, na):
    rows = []
    for a in i1.itertuples():
        cands = []
        for b in i2.itertuples():
            if a.chrom1 != b.chrom1 or a.chrom2 != b.chrom2:
                continue
            d1 = _axis_gap(a.start1, a.end1, b.start1, b.end1)
            d2 = _axis_gap(a.start2, a.end2, b.start2, b.end2)
            if md1 <= d1 <= xd1 and md2 <= d2 <= xd2:
                cands.append((d1 + d2, int(b.Index), d1, d2))
        cands.sort(key=lambda t: (t[0], t[1]))
        if cands:
            for (_m, k, d1, d2) in cands[:maxn]:
                rows.append((int(a.Index), k, d1, d2))
        elif na:
            rows.append((int(a.Index), -1, None, None))
    rows.sort(key=lambda r: (r[0], abs((r[2] + r[3]) if r[1] >= 0 else 0), r[1]))
    return rows


def _rand_2d(rng, n, pairs):
    recs = []
    for _ in range(n):
        c1, c2 = rng.choice(pairs)
        x = rng.randint(0, 990)
        y = rng.randint(0, 990)
        recs.append((c1, x, x + rng.randint(1, 20), c2, y, y + rng.randint(1, 20)))
    return pd.DataFrame(recs, columns=_COLS6)


def _result_tuples(df):
    if df is None:
        return []
    out = []
    for r in df.itertuples():
        id2 = -1 if pd.isna(r.start11) else None
        out.append(r)
    return out


class TestNeighbors2DBruteForce:
    def test_matches_brute_force_bounded(self):
        rng = random.Random(60427)
        pairs = [("1", "1"), ("1", "2"), ("2", "2")]
        for _ in range(40):
            i1 = _rand_2d(rng, rng.randint(1, 30), pairs).reset_index(drop=True)
            i2 = _rand_2d(rng, rng.randint(0, 30), pairs).reset_index(drop=True)
            maxn = rng.choice([1, 2, 100])
            md1, xd1 = 0, rng.choice([20, 100, 500])
            md2, xd2 = 0, rng.choice([20, 100, 500])
            df = _neighbors_2d(i1, i2, maxn, md1, xd1, md2, xd2, na_if_notfound=False)
            got = []
            if df is not None:
                for r in df.itertuples():
                    got.append((int(r.start1), int(r.start11), int(r.dist1), int(r.dist2)))
            # Reference keyed by start coords (unique enough for the assertion).
            ref = _brute(i1, i2, maxn, md1, xd1, md2, xd2, na=False)
            ref_keyed = [
                (int(i1.iloc[a].start1), int(i2.iloc[k].start1), d1, d2)
                for (a, k, d1, d2) in ref
            ]
            assert sorted(got) == sorted(ref_keyed)

    def test_na_if_notfound(self):
        i1 = pd.DataFrame([("1", 0, 10, "1", 0, 10)], columns=_COLS6)
        i2 = pd.DataFrame([("2", 0, 10, "2", 0, 10)], columns=_COLS6)  # different pair
        df = _neighbors_2d(i1, i2, 1, 0, 100, 0, 100, na_if_notfound=True)
        assert df is not None and len(df) == 1
        assert pd.isna(df.iloc[0]["start11"])
        assert pd.isna(df.iloc[0]["dist1"])

    def test_no_neighbor_returns_none(self):
        i1 = pd.DataFrame([("1", 0, 10, "1", 0, 10)], columns=_COLS6)
        i2 = pd.DataFrame([("2", 0, 10, "2", 0, 10)], columns=_COLS6)
        assert _neighbors_2d(i1, i2, 1, 0, 100, 0, 100, na_if_notfound=False) is None

    def test_column_layout(self):
        i1 = pd.DataFrame([("1", 0, 10, "1", 0, 10)], columns=_COLS6)
        i2 = pd.DataFrame([("1", 50, 60, "1", 50, 60)], columns=_COLS6)
        df = _neighbors_2d(i1, i2, 1, 0, 1000, 0, 1000, na_if_notfound=False)
        assert list(df.columns) == _COLS6 + [
            "chrom11", "start11", "end11", "chrom21", "start21", "end21", "dist1", "dist2"
        ]
        # gap: query.end=10, target.start=50 -> 40 on each axis
        assert int(df.iloc[0]["dist1"]) == 40
        assert int(df.iloc[0]["dist2"]) == 40
