"""Port of R misha test-liftover.R:2348-2699 (gintervals.as_chain)."""

import pandas as pd
import pytest

import pymisha as pm


def _row(**over):
    d = dict(chrom="chr1", start=0, end=100, strand=0, chromsrc="chr1", startsrc=0, endsrc=100,
             strandsrc=0, chain_id=1, score=1000.0)
    d.update(over)
    return pd.DataFrame({k: [v] for k, v in d.items()})


@pytest.mark.parametrize("over,msg", [
    ({"start": "0"}, "start and end columns must be numeric"),
    ({"chain_id": "1"}, "chain_id column must be integer or numeric"),
    ({"score": "1000.0"}, "score column must be numeric"),
    ({"strand": 2}, "strand values must be -1, 0, or 1"),
    ({"strandsrc": -2}, "strandsrc values must be -1, 0, or 1"),
])
def test_validates_columns(over, msg):
    with pytest.raises(ValueError, match=msg):
        pm.gintervals_as_chain(_row(**over))


@pytest.mark.parametrize("kw", [{"src_overlap_policy": "bad"}, {"tgt_overlap_policy": "bad"}, {"min_score": "x"}])
def test_validates_policies(kw):
    with pytest.raises(ValueError):
        pm.gintervals_as_chain(_row(), **kw)


def test_creates_valid_chain_with_attributes():
    chain = pm.gintervals_as_chain(
        _row(strand=1, chromsrc="chr2", startsrc=50, endsrc=150, strandsrc=-1, chain_id=42.0, score=5000.0),
        src_overlap_policy="keep", tgt_overlap_policy="auto", min_score=1000)
    r = chain.iloc[0]
    assert (r["chrom"], r["start"], r["end"], r["strand"]) == ("chr1", 0, 100, 1)
    assert (r["chromsrc"], r["startsrc"], r["endsrc"], r["strandsrc"]) == ("chr2", 50, 150, -1)
    assert r["chain_id"] == 42 and chain["chain_id"].dtype.kind == "i"
    assert r["score"] == 5000.0
    assert chain.attrs == {"src_overlap_policy": "keep", "tgt_overlap_policy": "auto_score", "min_score": 1000}


def test_empty_and_multiple():
    empty = pm.gintervals_as_chain(_row().iloc[:0])
    assert len(empty) == 0
    assert empty.attrs["src_overlap_policy"] == "error" and empty.attrs["tgt_overlap_policy"] == "auto_score"
    two = pm.gintervals_as_chain(pd.concat([_row(), _row(chrom="chr2", chain_id=2)], ignore_index=True))
    assert two["chrom"].tolist() == ["chr1", "chr2"] and two["chain_id"].tolist() == [1, 2]


def test_as_chain_of_a_loaded_chain_lifts_the_same(tmp_path):
    p = tmp_path / "c.chain"
    p.write_text("chain 1000 chrS 100000 + 0 100 chr1 500000 - 499300 499400 1\n100\n\n")
    loaded = pm.gintervals_load_chain(str(p))
    converted = pm.gintervals_as_chain(pd.DataFrame(loaded))
    q = pd.DataFrame({"chrom": ["chrS"], "start": [10], "end": [20]})
    pd.testing.assert_frame_equal(pm.gintervals_liftover(q, loaded), pm.gintervals_liftover(q, converted))
