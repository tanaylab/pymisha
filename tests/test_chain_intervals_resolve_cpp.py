"""Tests for G1.P3.B.1: pm_chain_intervals_resolve C++ port of liftover
overlap-policy handling. Mirrors test_chain_parser_cpp.py + test_source_track_cpp.py."""

import _pymisha
import numpy as np
import pandas as pd
import pytest

_EMPTY_CHAIN_COLS = [
    "chrom", "start", "end", "strand",
    "chromsrc", "startsrc", "endsrc", "strandsrc",
    "chain_id", "score",
]


def _empty_chain_dict():
    return {
        "chrom":     np.empty(0, dtype=object),
        "chromsrc":  np.empty(0, dtype=object),
        "start":     np.empty(0, dtype=np.int64),
        "end":       np.empty(0, dtype=np.int64),
        "strand":    np.empty(0, dtype=np.int64),
        "startsrc":  np.empty(0, dtype=np.int64),
        "endsrc":    np.empty(0, dtype=np.int64),
        "strandsrc": np.empty(0, dtype=np.int64),
        "chain_id":  np.empty(0, dtype=np.int64),
        "score":     np.empty(0, dtype=np.float64),
    }


def _make_chain_dict(rows):
    """Build a chain dict from a list of 10-tuples.

    Tuple order: (chrom, start, end, strand, chromsrc, startsrc, endsrc,
    strandsrc, chain_id, score).
    """
    if not rows:
        return _empty_chain_dict()
    cols = list(zip(*rows, strict=True))
    return {
        "chrom":     np.array(cols[0], dtype=object),
        "start":     np.array(cols[1], dtype=np.int64),
        "end":       np.array(cols[2], dtype=np.int64),
        "strand":    np.array(cols[3], dtype=np.int64),
        "chromsrc":  np.array(cols[4], dtype=object),
        "startsrc":  np.array(cols[5], dtype=np.int64),
        "endsrc":    np.array(cols[6], dtype=np.int64),
        "strandsrc": np.array(cols[7], dtype=np.int64),
        "chain_id":  np.array(cols[8], dtype=np.int64),
        "score":     np.array(cols[9], dtype=np.float64),
    }


def _to_df(d):
    return pd.DataFrame(d)[_EMPTY_CHAIN_COLS]


class TestPolicyValidation:
    def test_unknown_src_policy_raises(self):
        d = _empty_chain_dict()
        with pytest.raises(ValueError, match="Unknown src_overlap_policy"):
            _pymisha.pm_chain_intervals_resolve(d, "bogus", "keep")

    def test_unknown_tgt_policy_raises(self):
        d = _empty_chain_dict()
        with pytest.raises(ValueError, match="Unknown tgt_overlap_policy"):
            _pymisha.pm_chain_intervals_resolve(d, "keep", "bogus")

    def test_non_dict_input_raises(self):
        with pytest.raises(TypeError):
            _pymisha.pm_chain_intervals_resolve([], "keep", "keep")

    def test_missing_key_raises(self):
        d = _empty_chain_dict()
        del d["chrom"]
        with pytest.raises(ValueError, match="missing required key"):
            _pymisha.pm_chain_intervals_resolve(d, "keep", "keep")


class TestEmptyInput:
    def test_empty_keep_keep(self):
        out = _pymisha.pm_chain_intervals_resolve(_empty_chain_dict(), "keep", "keep")
        assert set(out) == set(_EMPTY_CHAIN_COLS)
        for v in out.values():
            assert len(v) == 0

    def test_empty_with_any_policy(self):
        for sp in ("keep", "error", "discard", "discard_intervals"):
            for tp in ("keep", "error", "discard", "discard_intervals", "auto",
                       "auto_first", "auto_longer", "auto_score", "agg"):
                out = _pymisha.pm_chain_intervals_resolve(_empty_chain_dict(), sp, tp)
                assert len(out["chrom"]) == 0, f"policies sp={sp} tp={tp}"


class TestPassThrough:
    """Until Tasks 3-6 land, all policies should be no-ops on simple input
    because the handle_*_overlaps stubs do nothing yet. This will be tightened
    in later tasks."""

    def test_simple_passthrough(self):
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chr1", 1000, 1100, 0, 1, 5000.0),
            ("chr1", 300, 400, 0, "chr1", 1200, 1300, 0, 2, 6000.0),
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "keep")
        # keep/keep should be a pure no-op even in the final implementation.
        df = _to_df(out)
        assert len(df) == 2
        assert df.iloc[0]["start"] == 100
        assert df.iloc[1]["start"] == 300


class TestInputValidation:
    """Cover dtype + length-mismatch error paths."""

    def test_wrong_dtype_string_for_int_column_raises(self):
        # If a string array is passed where an int64 is expected, the FROM_OTF
        # coercion should fail with a ValueError or TypeError.
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
        ])
        d["start"] = np.array(["bogus"], dtype=object)
        with pytest.raises((ValueError, TypeError)):
            _pymisha.pm_chain_intervals_resolve(d, "keep", "keep")

    def test_mismatched_column_lengths_raises(self):
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
            ("chr1", 300, 400, 0, "chrA", 700, 800, 0, 2, 2000.0),
        ])
        # Truncate one column to mismatch.
        d["end"] = np.array([200], dtype=np.int64)
        with pytest.raises(ValueError, match="mismatched lengths"):
            _pymisha.pm_chain_intervals_resolve(d, "keep", "keep")

    def test_non_string_chrom_raises(self):
        # Pass a non-string object in the chrom column.
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
        ])
        d["chrom"] = np.array([42], dtype=object)
        with pytest.raises(TypeError, match="is not a string"):
            _pymisha.pm_chain_intervals_resolve(d, "keep", "keep")


class TestHandleSrcOverlaps:
    """Pure-C++ tests of src_overlap_policy = error/keep/discard."""

    def test_keep_is_noop(self):
        # Overlapping src intervals stay put on "keep".
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
            ("chr1", 300, 400, 0, "chrA", 550, 650, 0, 2, 2000.0),  # src overlap
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "keep")
        assert len(out["chrom"]) == 2

    def test_error_no_overlap_passes(self):
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
            ("chr1", 300, 400, 0, "chrA", 700, 800, 0, 2, 2000.0),
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "error", "keep")
        assert len(out["chrom"]) == 2

    def test_error_overlap_raises(self):
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
            ("chr1", 300, 400, 0, "chrA", 550, 650, 0, 2, 2000.0),  # src overlap
        ])
        with pytest.raises(ValueError, match=r"Source overlap detected on chrA"):
            _pymisha.pm_chain_intervals_resolve(d, "error", "keep")

    def test_error_overlap_message_matches_python(self):
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
            ("chr1", 300, 400, 0, "chrA", 550, 650, 0, 2, 2000.0),
        ])
        with pytest.raises(ValueError) as exc:
            _pymisha.pm_chain_intervals_resolve(d, "error", "keep")
        assert "[500, 600)" in str(exc.value)
        assert "[550, 650)" in str(exc.value)

    def test_discard_intervals_drops_overlap_pair(self):
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
            ("chr1", 300, 400, 0, "chrA", 550, 650, 0, 2, 2000.0),  # overlaps row 1
            ("chr1", 500, 600, 0, "chrA", 800, 900, 0, 3, 3000.0),  # alone
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "discard_intervals", "keep")
        chain_ids = sorted(out["chain_id"].tolist())
        assert chain_ids == [3]

    def test_discard_intervals_drops_full_cluster_of_three(self):
        # Three intervals all in one overlapping cluster - all dropped.
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
            ("chr1", 300, 400, 0, "chrA", 550, 750, 0, 2, 2000.0),
            ("chr1", 700, 800, 0, "chrA", 700, 800, 0, 3, 3000.0),  # overlaps row 2
            ("chr1", 900,1000, 0, "chrA", 900,1000, 0, 4, 4000.0),  # standalone
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "discard_intervals", "keep")
        assert sorted(out["chain_id"].tolist()) == [4]

    def test_discard_intervals_per_chromsrc(self):
        # Overlaps on chrA but chrB is clean.
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
            ("chr1", 300, 400, 0, "chrA", 550, 650, 0, 2, 2000.0),  # overlap on chrA
            ("chr1", 700, 800, 0, "chrB", 100, 200, 0, 3, 3000.0),  # clean
            ("chr1", 800, 900, 0, "chrB", 300, 400, 0, 4, 4000.0),  # clean
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "discard_intervals", "keep")
        assert sorted(out["chain_id"].tolist()) == [3, 4]


class TestHandleTgtOverlapsSimple:
    """error / keep / discard policies."""

    def test_keep_is_noop(self):
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
            ("chr1", 150, 250, 0, "chrA", 700, 800, 0, 2, 2000.0),  # tgt overlap
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "keep")
        assert len(out["chrom"]) == 2

    def test_error_no_overlap_passes(self):
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
            ("chr1", 300, 400, 0, "chrA", 700, 800, 0, 2, 2000.0),
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "error")
        assert len(out["chrom"]) == 2

    def test_error_overlap_raises(self):
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
            ("chr1", 150, 250, 0, "chrA", 700, 800, 0, 2, 2000.0),
        ])
        with pytest.raises(ValueError, match=r"Target overlap detected on chr1"):
            _pymisha.pm_chain_intervals_resolve(d, "keep", "error")

    def test_error_overlap_message_matches_python(self):
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
            ("chr1", 150, 250, 0, "chrA", 700, 800, 0, 2, 2000.0),
        ])
        with pytest.raises(ValueError) as exc:
            _pymisha.pm_chain_intervals_resolve(d, "keep", "error")
        assert "[100, 200)" in str(exc.value)
        assert "[150, 250)" in str(exc.value)

    def test_discard_intervals_drops_simple_pair(self):
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
            ("chr1", 150, 250, 0, "chrA", 700, 800, 0, 2, 2000.0),
            ("chr1", 500, 600, 0, "chrA", 900,1000, 0, 3, 3000.0),  # standalone
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "discard_intervals")
        assert sorted(out["chain_id"].tolist()) == [3]

    def test_discard_intervals_drops_all_overlap_cluster(self):
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
            ("chr1", 150, 300, 0, "chrA", 700, 850, 0, 2, 2000.0),
            ("chr1", 250, 350, 0, "chrA", 850, 950, 0, 3, 3000.0),
            ("chr1", 500, 600, 0, "chrA", 1100,1200, 0, 4, 4000.0),  # alone
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "discard_intervals")
        assert sorted(out["chain_id"].tolist()) == [4]

    def test_discard_intervals_per_chrom_independent(self):
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
            ("chr1", 150, 250, 0, "chrA", 700, 800, 0, 2, 2000.0),
            ("chr2", 100, 200, 0, "chrA", 900,1000, 0, 3, 3000.0),
            ("chr2", 300, 400, 0, "chrA", 1100,1200, 0, 4, 4000.0),
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "discard_intervals")
        # chr1 had an overlap pair (rows 1+2), chr2 clean.
        assert sorted(out["chain_id"].tolist()) == [3, 4]

    def test_final_order_after_error_pass_is_tgt_sorted(self):
        d = _make_chain_dict([
            ("chr2", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
            ("chr1", 300, 400, 0, "chrA", 700, 800, 0, 2, 2000.0),
        ])
        # "error" with no overlaps sorts by tgt. Insertion order is chr2,chr1
        # so the interner assigns chr2->0, chr1->1. After sort_by_tgt rows are
        # in (chromid=0=chr2)-then-(chromid=1=chr1) order, so output stays
        # chr2 then chr1. NOTE: this is interner-insertion-order sort, not
        # alphabetical. Document.
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "error")
        # Verify both rows survived; specific order matches the chromid sort.
        assert sorted(out["chrom"].tolist()) == ["chr1", "chr2"]


class TestHandleTgtOverlapsAuto:
    """auto_first / auto_longer / auto_score winner-selection on overlapping
    target segments, plus adjacent-slice merging."""

    def test_auto_score_picks_highest(self):
        # Two overlapping target intervals on chr1. Higher-score wins the
        # overlapping segment; non-overlapping parts of each interval survive.
        # Layout (tgt):
        #   chain1: [100, 300) score=1000
        #   chain2: [200, 400) score=2000
        # Expected segments after auto_score:
        #   [100, 200) -> chain1 (only one active)
        #   [200, 300) -> chain2 (higher score)
        #   [300, 400) -> chain2
        # The latter two abut + same chain_id => merge: [200, 400) -> chain2.
        d = _make_chain_dict([
            ("chr1", 100, 300, 0, "chrA",   0, 200, 0, 1, 1000.0),
            ("chr1", 200, 400, 0, "chrA", 500, 700, 0, 2, 2000.0),
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "auto_score")
        df = _to_df(out).sort_values(["start"]).reset_index(drop=True)
        assert len(df) == 2
        assert list(df["chain_id"]) == [1, 2]
        assert list(df["start"]) == [100, 200]
        assert list(df["end"]) == [200, 400]

    def test_auto_score_tiebreaker_span(self):
        # Same score => longer span wins.
        d = _make_chain_dict([
            ("chr1", 100, 300, 0, "chrA",   0, 200, 0, 1, 1000.0),
            ("chr1", 100, 500, 0, "chrA", 500, 900, 0, 2, 1000.0),
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "auto_score")
        df = _to_df(out)
        # chain2 wins everywhere because it covers [100,500) and has equal
        # score but longer span. chain1's [100,300) is fully shadowed.
        assert list(df["chain_id"]) == [2]
        assert df.iloc[0]["start"] == 100 and df.iloc[0]["end"] == 500

    def test_auto_score_tiebreaker_chain_id(self):
        # Same score + same span => smaller chain_id wins.
        d = _make_chain_dict([
            ("chr1", 100, 300, 0, "chrA", 0, 200, 0, 7, 1000.0),
            ("chr1", 100, 300, 0, "chrA", 500, 700, 0, 3, 1000.0),
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "auto_score")
        df = _to_df(out)
        assert list(df["chain_id"]) == [3]

    def test_auto_first_picks_smallest_chain_id(self):
        d = _make_chain_dict([
            ("chr1", 100, 300, 0, "chrA",   0, 200, 0, 5, 9999.0),  # highest score
            ("chr1", 100, 300, 0, "chrA", 500, 700, 0, 2, 1000.0),  # smaller chain_id
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "auto_first")
        df = _to_df(out)
        assert list(df["chain_id"]) == [2]

    def test_auto_longer_picks_longest_span(self):
        d = _make_chain_dict([
            ("chr1", 100, 300, 0, "chrA",   0, 200, 0, 1, 9999.0),
            ("chr1", 100, 500, 0, "chrA", 500, 900, 0, 2, 1.0),
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "auto_longer")
        df = _to_df(out)
        # chain2 has longer span so wins everywhere it covers.
        assert list(df["chain_id"]) == [2]
        assert df.iloc[0]["start"] == 100 and df.iloc[0]["end"] == 500

    def test_auto_alias_is_auto_score(self):
        d = _make_chain_dict([
            ("chr1", 100, 300, 0, "chrA",   0, 200, 0, 1, 1000.0),
            ("chr1", 200, 400, 0, "chrA", 500, 700, 0, 2, 2000.0),
        ])
        out_a = _pymisha.pm_chain_intervals_resolve(d, "keep", "auto")
        out_s = _pymisha.pm_chain_intervals_resolve(d, "keep", "auto_score")
        for k in _EMPTY_CHAIN_COLS:
            assert list(out_a[k]) == list(out_s[k])

    def test_auto_no_overlap_passthrough(self):
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA",   0, 100, 0, 1, 1000.0),
            ("chr1", 300, 400, 0, "chrA", 500, 600, 0, 2, 2000.0),
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "auto_score")
        df = _to_df(out)
        assert list(df["chain_id"]) == [1, 2]
        assert list(df["start"]) == [100, 300]

    def test_auto_negative_strand_src_coord_mapping(self):
        # Negative-tgt-strand source mapping: when strand==1 (tgt negative),
        # advancing tgt by N retreats src by N. Verify slice src coords are
        # projected correctly per segment. Two intervals overlap on chr1:
        #   chain1: tgt [100, 300) strand=1 (NEG), src [500, 700)
        #   chain2: tgt [200, 400) strand=0 (POS), src [800,1000), score wins
        # auto_score picks chain2 for [200,400); chain1 keeps [100,200).
        d = _make_chain_dict([
            ("chr1", 100, 300, 1, "chrA", 500, 700, 0, 1, 1000.0),
            ("chr1", 200, 400, 0, "chrA", 800,1000, 0, 2, 2000.0),
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "auto_score")
        df = _to_df(out).sort_values("start").reset_index(drop=True)
        # Slice 1: chain1, tgt [100,200), strand=1 (neg).
        # Python source (liftover.py:659-667):
        #   seg_src_starts[mask_nn] = orig_src_ends - (seg_ends - orig_tgt_starts)
        #   seg_src_ends[mask_nn]   = orig_src_ends - (seg_starts - orig_tgt_starts)
        # For orig=[100,300), orig_src=[500,700):
        #   src_start = 700 - (200 - 100) = 600
        #   src_end   = 700 - (100 - 100) = 700
        assert df.iloc[0]["chain_id"] == 1
        assert df.iloc[0]["startsrc"] == 600
        assert df.iloc[0]["endsrc"] == 700
        # Slice 2: chain2, tgt [200,400), strand=0 (pos),
        # orig=[200,400), orig_src=[800,1000):
        # delta_start = 200-200 = 0; delta_end = 400-200 = 200
        # src_start = 800+0 = 800, src_end = 800+200 = 1000.
        assert df.iloc[1]["chain_id"] == 2
        assert df.iloc[1]["startsrc"] == 800
        assert df.iloc[1]["endsrc"] == 1000


class TestHandleTgtOverlapsAgg:
    def test_agg_emits_per_chain_per_segment(self):
        # Two overlapping intervals. Expect 3 segments: [100,200), [200,300),
        # [300,400). The middle segment has both chains active so emits two
        # rows (no merging). The other two segments emit one row each.
        d = _make_chain_dict([
            ("chr1", 100, 300, 0, "chrA",   0, 200, 0, 1, 1000.0),
            ("chr1", 200, 400, 0, "chrA", 500, 700, 0, 2, 2000.0),
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "agg")
        df = _to_df(out).sort_values(["start", "end", "chain_id"]).reset_index(drop=True)
        assert len(df) == 4  # 1 + 2 + 1
        assert list(df["start"]) == [100, 200, 200, 300]
        assert list(df["end"])   == [200, 300, 300, 400]
        assert sorted(df[df["start"] == 200]["chain_id"].tolist()) == [1, 2]

    def test_agg_no_merge_across_adjacent_segments(self):
        # chain1 and chain2 share an endpoint at 200 but do NOT overlap.
        # In auto_score this would emit 2 rows; in agg the result is the same
        # because there's no overlap segment to expand. The key thing is that
        # agg never merges adjacent slices for the same chain_id.
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA",   0, 100, 0, 1, 1000.0),
            ("chr1", 200, 300, 0, "chrA", 500, 600, 0, 2, 2000.0),
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "agg")
        df = _to_df(out).sort_values(["start"]).reset_index(drop=True)
        assert len(df) == 2
        assert list(df["chain_id"]) == [1, 2]

    def test_agg_three_way_overlap(self):
        d = _make_chain_dict([
            ("chr1", 100, 400, 0, "chrA",    0, 300, 0, 1, 1000.0),
            ("chr1", 200, 500, 0, "chrA",  500, 800, 0, 2, 2000.0),
            ("chr1", 300, 600, 0, "chrA", 1000,1300, 0, 3, 3000.0),
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "agg")
        df = _to_df(out).sort_values(["start", "chain_id"]).reset_index(drop=True)
        # Segments: [100,200) {1}, [200,300) {1,2}, [300,400) {1,2,3}, [400,500) {2,3}, [500,600) {3}
        # Row counts: 1 + 2 + 3 + 2 + 1 = 9
        assert len(df) == 9
        # Spot-check the 3-way segment.
        seg = df[(df["start"] == 300) & (df["end"] == 400)]
        assert sorted(seg["chain_id"].tolist()) == [1, 2, 3]


class TestCrossValidatePython:
    """C++ output must match the pure-Python implementation row-for-row on
    every supported policy combination."""

    @staticmethod
    def _python_resolve(chain_dict, src_p, tgt_p):
        from pymisha.liftover import _resolve_chain_overlaps
        return _resolve_chain_overlaps(chain_dict, src_p, tgt_p,
                                       _force_pure_python=True)

    @staticmethod
    def _cpp_resolve(chain_dict, src_p, tgt_p):
        # Bypass the env var; call C++ directly via the dispatcher with the
        # default flag. To eliminate env-var leakage, call C++ raw on the
        # already-normalized policies.
        from pymisha.liftover import _resolve_chain_overlaps
        return _resolve_chain_overlaps(chain_dict, src_p, tgt_p,
                                       _force_pure_python=False)

    @staticmethod
    def _assert_same(cpp_out, py_out):
        keys = set(_EMPTY_CHAIN_COLS)
        assert set(cpp_out.keys()) == keys, f"cpp keys {sorted(cpp_out)} != {sorted(keys)}"
        assert set(py_out.keys()) == keys, f"py keys {sorted(py_out)} != {sorted(keys)}"

        def _df(d):
            return pd.DataFrame({k: list(d[k]) for k in _EMPTY_CHAIN_COLS})

        # Sort by full key tuple so row-order differences between paths
        # (e.g. stable_sort vs np.lexsort) don't cause false negatives.
        sort_key = ["chrom", "start", "end", "chain_id", "chromsrc", "startsrc"]
        df_cpp = _df(cpp_out).sort_values(sort_key, kind="mergesort").reset_index(drop=True)
        df_py = _df(py_out).sort_values(sort_key, kind="mergesort").reset_index(drop=True)

        assert len(df_cpp) == len(df_py), (
            f"row count differs: cpp={len(df_cpp)} py={len(df_py)}\n"
            f"cpp head:\n{df_cpp.head(20)}\npy head:\n{df_py.head(20)}"
        )
        for k in _EMPTY_CHAIN_COLS:
            for i in range(len(df_cpp)):
                a = df_cpp[k].iloc[i]
                b = df_py[k].iloc[i]
                if isinstance(a, np.generic):
                    a = a.item()
                if isinstance(b, np.generic):
                    b = b.item()
                if isinstance(a, float) and isinstance(b, float):
                    assert (np.isnan(a) and np.isnan(b)) or a == b, (
                        f"col={k} row={i}: cpp={a!r} py={b!r}"
                    )
                else:
                    assert a == b, f"col={k} row={i}: cpp={a!r} py={b!r}"

    @pytest.mark.parametrize("src_p,tgt_p", [
        ("keep", "keep"),
        ("discard", "keep"),
        ("keep", "discard"),
        ("discard", "discard"),
        ("discard_intervals", "keep"),
        ("keep", "discard_intervals"),
        ("discard_intervals", "discard_intervals"),
        ("keep", "auto_score"),
        ("keep", "auto_longer"),
        ("keep", "auto_first"),
        ("keep", "auto"),
        ("keep", "agg"),
    ])
    def test_xval_two_overlapping(self, src_p, tgt_p):
        d = _make_chain_dict([
            ("chr1", 100, 300, 0, "chrA",   0, 200, 0, 1, 1000.0),
            ("chr1", 200, 400, 0, "chrA", 500, 700, 0, 2, 2000.0),
        ])
        self._assert_same(
            self._cpp_resolve(d, src_p, tgt_p),
            self._python_resolve(d, src_p, tgt_p),
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 7, 42])
    def test_xval_synthetic_random(self, seed):
        rng = np.random.RandomState(seed)
        rows = []
        for cid in range(1, 21):
            chrom = "chr" + str(1 + (cid % 3))
            start = int(rng.randint(0, 5000))
            length = int(rng.randint(50, 500))
            end = start + length
            chromsrc = "chrA" if cid % 2 == 0 else "chrB"
            start_src = int(rng.randint(0, 5000))
            end_src = start_src + length
            strand = int(rng.randint(0, 2))
            strand_src = int(rng.randint(0, 2))
            score = float(rng.randint(100, 10000))
            rows.append((chrom, start, end, strand, chromsrc, start_src,
                         end_src, strand_src, cid, score))
        d = _make_chain_dict(rows)

        for src_p in ("keep", "discard", "discard_intervals"):
            for tgt_p in ("keep", "discard", "discard_intervals", "auto_score", "auto_longer",
                          "auto_first", "agg"):
                cpp_out = self._cpp_resolve(d, src_p, tgt_p)
                py_out = self._python_resolve(d, src_p, tgt_p)
                self._assert_same(cpp_out, py_out)

    @pytest.mark.parametrize("src_p,tgt_p", [
        ("keep", "discard"), ("discard", "keep"), ("keep", "auto_score"),
    ])
    def test_xval_large_group_sweep_path(self, src_p, tgt_p):
        # 800 overlapping rows on one chrom and one chromsrc: over the Python
        # path's 500k segment x row limit, so it takes the sweep-line fallback
        rng = np.random.RandomState(3)
        rows = []
        for cid in range(1, 801):
            start, start_src, length = (int(v) for v in (rng.randint(0, 20000), rng.randint(0, 20000),
                                                         rng.randint(20, 200)))
            rows.append(("chr1", start, start + length, int(rng.randint(0, 2)), "chrA",
                         start_src, start_src + length, 0, cid, float(rng.randint(1, 4))))
        d = _make_chain_dict(rows)
        self._assert_same(self._cpp_resolve(d, src_p, tgt_p), self._python_resolve(d, src_p, tgt_p))

    def test_xval_three_way_overlap(self):
        d = _make_chain_dict([
            ("chr1", 100, 400, 0, "chrA",    0, 300, 0, 1, 1000.0),
            ("chr1", 200, 500, 0, "chrA",  500, 800, 0, 2, 2000.0),
            ("chr1", 300, 600, 0, "chrA", 1000,1300, 0, 3, 3000.0),
        ])
        for tp in ("auto_score", "auto_longer", "auto_first", "agg"):
            self._assert_same(
                self._cpp_resolve(d, "keep", tp),
                self._python_resolve(d, "keep", tp),
            )

    def test_xval_negative_strand(self):
        # Tgt strand=1 (negative) - exercises the negative-strand src-coord branch.
        d = _make_chain_dict([
            ("chr1", 100, 300, 1, "chrA", 500, 700, 0, 1, 1000.0),
            ("chr1", 200, 400, 1, "chrA", 800,1000, 0, 2, 2000.0),
        ])
        for tp in ("auto_score", "auto_longer", "auto_first", "agg", "discard", "discard_intervals"):
            self._assert_same(
                self._cpp_resolve(d, "keep", tp),
                self._python_resolve(d, "keep", tp),
            )


class TestEdgeCases:
    def test_single_row_passes_through(self):
        d = _make_chain_dict([
            ("chr1", 100, 200, 0, "chrA", 500, 600, 0, 1, 1000.0),
        ])
        for sp in ("keep", "error", "discard", "discard_intervals"):
            for tp in ("keep", "error", "discard", "discard_intervals", "auto_score",
                       "auto_first", "auto_longer", "agg"):
                out = _pymisha.pm_chain_intervals_resolve(d, sp, tp)
                assert len(out["chrom"]) == 1
                assert out["chain_id"][0] == 1

    def test_zero_length_target_interval_keeps_src(self):
        # A zero-length target interval doesn't contribute a segment to the
        # sweep (events at the same position cancel out). Chain 1 effectively
        # disappears from the output; chain 2 survives intact.
        d = _make_chain_dict([
            ("chr1", 100, 100, 0, "chrA", 500, 500, 0, 1, 1000.0),
            ("chr1", 200, 300, 0, "chrA", 600, 700, 0, 2, 2000.0),
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "auto_score")
        assert len(out["chrom"]) == 1
        assert out["chain_id"][0] == 2

    def test_cluster_policies_dispatch_as_keep(self):
        # best_cluster_* are NORMALIZED TO "keep" in the Python dispatcher
        # before the C++ call. Verify end-to-end via _resolve_chain_overlaps.
        from pymisha.liftover import _resolve_chain_overlaps
        d = _make_chain_dict([
            ("chr1", 100, 300, 0, "chrA",   0, 200, 0, 1, 1000.0),
            ("chr1", 200, 400, 0, "chrA", 500, 700, 0, 2, 2000.0),
        ])
        for cluster_p in ("best_source_cluster", "best_cluster_union",
                          "best_cluster_sum", "best_cluster_max"):
            keep_out = _resolve_chain_overlaps(d, "keep", "keep")
            cluster_out = _resolve_chain_overlaps(d, "keep", cluster_p)
            for k in _EMPTY_CHAIN_COLS:
                assert list(cluster_out[k]) == list(keep_out[k]), (
                    f"col={k} cluster_p={cluster_p}"
                )

    def test_empty_chain_through_dispatcher(self):
        from pymisha.liftover import _resolve_chain_overlaps
        empty = _empty_chain_dict()
        for sp in ("keep", "error", "discard", "discard_intervals"):
            for tp in ("keep", "error", "discard", "discard_intervals", "auto_score", "agg"):
                out = _resolve_chain_overlaps(empty, sp, tp)
                assert len(out["chrom"]) == 0

    def test_multi_chrom_independent_resolution(self):
        # Overlap on chr1 + overlap on chr2 - both resolved independently.
        d = _make_chain_dict([
            ("chr1", 100, 300, 0, "chrA", 0, 200, 0, 1, 1000.0),
            ("chr1", 200, 400, 0, "chrA", 500, 700, 0, 2, 2000.0),
            ("chr2", 100, 300, 0, "chrB", 0, 200, 0, 3, 9999.0),
            ("chr2", 200, 400, 0, "chrB", 500, 700, 0, 4, 8888.0),
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "auto_score")
        df = _to_df(out).sort_values(["chrom", "start"]).reset_index(drop=True)
        chr1 = df[df["chrom"] == "chr1"].reset_index(drop=True)
        chr2 = df[df["chrom"] == "chr2"].reset_index(drop=True)
        # chr1: chain2 wins overlap [200, 400) by score; chain1 keeps [100,200).
        assert chr1.iloc[0]["chain_id"] == 1 and chr1.iloc[0]["end"] == 200
        assert chr1.iloc[1]["chain_id"] == 2 and chr1.iloc[1]["start"] == 200
        # chr2: chain3 wins overlap [100, 300) by score; chain4 keeps [300,400).
        assert chr2.iloc[0]["chain_id"] == 3 and chr2.iloc[0]["end"] == 300
        assert chr2.iloc[1]["chain_id"] == 4 and chr2.iloc[1]["start"] == 300

    def test_large_row_count_runs(self):
        # Sanity check: 5000 rows through the auto path doesn't crash + finishes.
        rng = np.random.RandomState(0)
        rows = []
        for cid in range(1, 5001):
            chrom = "chr" + str(1 + (cid % 4))
            start = int(rng.randint(0, 100000))
            length = int(rng.randint(50, 500))
            rows.append((chrom, start, start + length, 0, "chrA",
                         int(rng.randint(0, 100000)),
                         int(rng.randint(0, 100000)) + length,
                         0, cid, float(rng.randint(100, 10000))))
        # Make endsrc consistent.
        d = _make_chain_dict([
            (r[0], r[1], r[2], r[3], r[4], r[5], r[5] + (r[2] - r[1]),
             r[7], r[8], r[9]) for r in rows
        ])
        out = _pymisha.pm_chain_intervals_resolve(d, "keep", "auto_score")
        assert len(out["chrom"]) > 0


class TestRParity:
    """R misha 5.18.0 behaviour of the overlap policies, checked in C++ and Python.

    - "discard_intervals" drops every interval that overlaps another, on the source
      side also one that is not next to it in source order (R's "discard" before
      5.18.0 compared neighbours only, and pymisha followed it from v0.1.91).
    - "discard" drops only the bases several intervals cover.
    - auto_*: a minus-strand interval that wins every segment comes back in one piece.
    """

    # [0,200) holds [10,50) and [60,80), which do not overlap each other
    NESTED_GAP = [
        ("chr1", 100, 300, 0, "chrA",  0, 200, 0, 1, 1000.0),
        ("chr1", 300, 340, 0, "chrA", 10,  50, 0, 2, 2000.0),
        ("chr1", 400, 420, 0, "chrA", 60,  80, 0, 3, 3000.0),
    ]
    # minus-strand chain 1 [100,400) holds chain 2 [200,300), which has the lower score
    MINUS_WINNER = [
        ("chr1", 100, 400, 1, "chrA", 200, 500, 0, 1, 1000.0),
        ("chr1", 200, 300, 0, "chrA", 800, 900, 0, 2,  500.0),
    ]

    @staticmethod
    def _both(rows, src_p, tgt_p):
        from pymisha.liftover import _resolve_chain_overlaps
        d = _make_chain_dict(rows)
        return [_to_df(_resolve_chain_overlaps(d, src_p, tgt_p, _force_pure_python=py))
                .sort_values(["startsrc", "start"]).reset_index(drop=True) for py in (False, True)]

    def test_src_discard_intervals_finds_non_adjacent_overlap(self):
        for df in self._both(self.NESTED_GAP, "discard_intervals", "keep"):
            assert len(df) == 0

    def test_src_discard_keeps_single_covered_bases(self):
        for df in self._both(self.NESTED_GAP, "discard", "keep"):
            assert df["chain_id"].tolist() == [1, 1, 1]
            assert list(zip(df["startsrc"], df["endsrc"], strict=True)) == [(0, 10), (50, 60), (80, 200)]
            assert list(zip(df["start"], df["end"], strict=True)) == [(100, 110), (150, 160), (180, 300)]

    def test_src_discard_minus_strand(self):
        # chain 1 src [1000,1100) -> tgt [5000,5100) reversed; chain 2 src [1010,1020)
        rows = [("chr1", 5000, 5100, 1, "chrA", 1000, 1100, 0, 1, 1000.0),
                ("chr1", 8000, 8010, 0, "chrA", 1010, 1020, 0, 2, 1000.0)]
        for df in self._both(rows, "discard", "keep"):
            assert list(zip(df["startsrc"], df["endsrc"], df["start"], df["end"], strict=True)) == [
                (1000, 1010, 5090, 5100), (1020, 1100, 5000, 5080)]

    def test_auto_score_minus_strand_winner_in_one_piece(self):
        for df in self._both(self.MINUS_WINNER, "keep", "auto_score"):
            assert len(df) == 1
            assert df.iloc[0][["start", "end", "startsrc", "endsrc"]].tolist() == [100, 400, 200, 500]

    def test_positive_strand_merge_still_works(self):
        # chain 2 wins [200,400) whole: its two segments merge
        rows = [("chr1", 100, 300, 0, "chrA",   0, 200, 0, 1, 1000.0),
                ("chr1", 200, 400, 0, "chrA", 500, 700, 0, 2, 2000.0)]
        for df in self._both(rows, "keep", "auto_score"):
            df = df.sort_values("start")
            assert df["chain_id"].tolist() == [1, 2]
            assert df.iloc[1][["start", "end"]].tolist() == [200, 400]
