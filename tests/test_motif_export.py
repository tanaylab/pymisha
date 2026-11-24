"""Tests for the motif writers: gseq_write_meme, gseq_write_jaspar, gseq_write_homer.

The golden files in test_data/motifs/export were written by R misha's
gseq.write_meme/write_jaspar/write_homer (its tests/testthat/test-motif-export.R
has the same inputs and checks the same files), so comparing bytes with them is
the R parity check. ``test_random_motifs_match_r_live`` also runs R's writers on
fresh random motifs when ``MISHA_R_SRC`` points to an R misha source tree.
"""

import contextlib
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

FIXTURES = Path(__file__).resolve().parent / "test_data" / "motifs"
EXPORT = FIXTURES / "export"
ACGT = ["A", "C", "G", "T"]
META_KEYS = ("nsites", "E", "log_odds_threshold", "log_p_value")


# ---------------------------------------------------------------------------
# Inputs shared with R misha's tests
# ---------------------------------------------------------------------------

def _hex(x: str) -> float | None:
    return None if x == "" else float.fromhex(x)


def random_motifs() -> dict[str, pd.DataFrame]:
    """Random motifs with metadata, stored as hex floats so R and Python read the same doubles."""
    mt = pd.read_csv(EXPORT / "random_matrices.tsv", sep="\t", dtype=str, keep_default_na=False)
    meta = pd.read_csv(EXPORT / "random_meta.tsv", sep="\t", dtype=str, keep_default_na=False)
    out = {}
    for _, row in meta.iterrows():
        rows = mt[mt["id"] == row["id"]]
        df = pd.DataFrame({b: [float.fromhex(v) for v in rows[b]] for b in ACGT})
        if row["name"] != "":
            df.attrs["name"] = row["name"]
        for key in ("nsites", "E", "log_odds_threshold", "log_p_value"):
            v = _hex(row[key])
            if v is not None:
                df.attrs[key] = v
        out[row["id"]] = df
    return out


def edge_motifs() -> dict[str, pd.DataFrame]:
    """Width 1, zeros, whitespace in names, values that round (as in R's tests)."""
    w1 = pd.DataFrame([[0.1, 0.2, 0.3, 0.4]], columns=ACGT)
    w1.attrs.update(name="  one\tposition  name ", nsites=7.5, E=0.0)
    zeros = pd.DataFrame(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0.5, 0.5, 0, 0], [0, 0.5, 0, 0.5]], columns=ACGT, dtype=float
    )
    thirds = pd.DataFrame(
        [[1 / 3, 1 / 3, 1 / 6, 1 / 6], [1 / 7, 2 / 7, 3 / 7, 1 / 7], [0.45, 0.4, 0.1, 0.05], [0.3, 0.3, 0.2, 0.2]],
        columns=ACGT,
    )
    thirds.attrs.update(name="thirds", log_odds_threshold=1.23456789, url="http://example.org/thirds")
    return {"w1": w1, "zeros": zeros, "thirds": thirds}


def golden_cases():
    sets = {
        "from_meme": pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme")),
        "from_jaspar": pm.gseq_read_jaspar(str(FIXTURES / "test_motifs.jaspar")),
        "from_homer": pm.gseq_read_homer(str(FIXTURES / "test_motifs.homer.motif")),
        "random": random_motifs(),
        "edge": edge_motifs(),
    }
    cases = {}
    for nm, x in sets.items():
        cases[f"{nm}.meme"] = lambda f, x=x: pm.gseq_write_meme(x, f)
        cases[f"{nm}.jaspar"] = lambda f, x=x: pm.gseq_write_jaspar(x, f)
        cases[f"{nm}.motif"] = lambda f, x=x: pm.gseq_write_homer(x, f)
    edge = sets["edge"]
    cases["edge_simple.pfm"] = lambda f: pm.gseq_write_jaspar(edge, f, format="simple")
    cases["edge_args.meme"] = lambda f: pm.gseq_write_meme(
        edge, f, background={"T": 0.3, "G": 0.2, "C": 0.2, "A": 0.3}, strands="+"
    )
    cases["edge_args.jaspar"] = lambda f: pm.gseq_write_jaspar(edge, f, nsites=[10, 1000, 3])
    cases["edge_exact.jaspar"] = lambda f: pm.gseq_write_jaspar(edge, f, nsites=[10, 1000, 3], round_counts=False)
    cases["random_exact.jaspar"] = lambda f: pm.gseq_write_jaspar(sets["random"], f, round_counts=False)
    cases["from_meme_exact.jaspar"] = lambda f: pm.gseq_write_jaspar(sets["from_meme"], f, round_counts=False)
    cases["edge_args.motif"] = lambda f: pm.gseq_write_homer(edge, f, threshold_frac=0.8)
    return cases


def max_diff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    return float(np.max(np.abs(a[ACGT].to_numpy() - b[ACGT].to_numpy())))


# ---------------------------------------------------------------------------
# R parity
# ---------------------------------------------------------------------------

def test_writers_match_r_golden_files(tmp_path):
    cases = golden_cases()
    assert len(cases) == 22
    for nm, write in cases.items():
        f = tmp_path / nm
        with _ignore_renorm():
            write(str(f))
        assert f.read_bytes() == (EXPORT / nm).read_bytes(), nm


@contextlib.contextmanager
def _ignore_renorm():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*re-normalizing.*")
        yield


_R_SCRIPT = r"""
args <- commandArgs(TRUE)
src <- args[1]; d <- args[2]
source(file.path(src, "R", "motif-import.R")); source(file.path(src, "R", "motif-export.R"))
acgt <- c("A", "C", "G", "T")
mt <- read.delim(file.path(d, "m.tsv"), colClasses = "character")
meta <- read.delim(file.path(d, "meta.tsv"), colClasses = "character", na.strings = character(0))
num <- function(x) if (x == "") NA_real_ else as.numeric(x)
x <- lapply(meta$id, function(id) {
    rows <- mt[mt$id == id, ]
    m <- matrix(as.numeric(unlist(rows[, acgt])), ncol = 4, dimnames = list(NULL, acgt))
    row <- meta[meta$id == id, ]
    if (row$name != "") attr(m, "name") <- row$name
    for (key in c("nsites", "E", "log_odds_threshold", "log_p_value")) {
        v <- num(row[[key]])
        if (!is.na(v)) attr(m, key) <- v
    }
    m
})
names(x) <- meta$id
suppressWarnings({
    gseq.write_meme(x, file.path(d, "r.meme"))
    gseq.write_meme(x, file.path(d, "r_bg.meme"), background = c(0.31, 0.19, 0.2, 0.3), strands = "+")
    gseq.write_jaspar(x, file.path(d, "r.jaspar"))
    gseq.write_jaspar(x, file.path(d, "r_ns.jaspar"), nsites = 1)
    gseq.write_jaspar(x, file.path(d, "r_exact.jaspar"), round_counts = FALSE)
    gseq.write_jaspar(x, file.path(d, "r.pfm"), format = "simple")
    gseq.write_homer(x, file.path(d, "r.motif"))
    gseq.write_homer(x, file.path(d, "r_frac.motif"), threshold_frac = 0.37)
})
"""


def _r_src():
    src = os.environ.get("MISHA_R_SRC")
    if not src or not shutil.which("Rscript"):
        return None
    if not (Path(src) / "R" / "motif-export.R").is_file():
        return None
    return src


def _hexs(x) -> str:
    return "" if x is None else float(x).hex()


@pytest.mark.skipif(
    _r_src() is None, reason="set MISHA_R_SRC to an R misha source tree (with Rscript) to compare with R"
)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_motifs_match_r_live(tmp_path, seed):
    rng = np.random.default_rng(seed)
    mrows, meta = [], []
    motifs = {}
    for i in range(40):
        mid = f"S{seed}_{i}"
        w = int(rng.integers(1, 26))
        m = rng.gamma(rng.choice([0.05, 0.3, 1.0, 5.0]), size=(w, 4))
        m[rng.random((w, 4)) < 0.2] = 0
        if i % 7 == 3:
            m = m * 37.0  # counts, re-normalized by the writers
        elif i % 7 == 5:
            m[0, :] = 0  # an all-zero row
        else:
            s = m.sum(axis=1)
            m[s == 0, 2] = 1
            m = m / m.sum(axis=1, keepdims=True)
        df = pd.DataFrame(m, columns=ACGT)
        attrs = {
            "name": rng.choice(["", f"TF{i}", f" tf {i}  with   spaces ", "a/b(c)::d"]),
            "nsites": rng.choice([None, float(rng.integers(1, 1000)), float(rng.uniform(0.5, 80))]),
            "E": rng.choice([None, float(10 ** rng.uniform(-300, 3)), 0.0]),
            "log_odds_threshold": rng.choice([None, float(rng.uniform(-2, 20))]),
            "log_p_value": rng.choice([None, float(-rng.uniform(0, 1e4))]),
        }
        for k, v in attrs.items():
            if v is not None and v != "":
                df.attrs[k] = v
        motifs[mid] = df
        for p in range(w):
            mrows.append([mid, str(p + 1)] + [float(v).hex() for v in m[p]])
        meta.append([mid, attrs["name"]] + [_hexs(attrs[k]) for k in META_KEYS])
    pd.DataFrame(mrows, columns=["id", "pos"] + ACGT).to_csv(tmp_path / "m.tsv", sep="\t", index=False)
    pd.DataFrame(meta, columns=["id", "name", *META_KEYS]).to_csv(
        tmp_path / "meta.tsv", sep="\t", index=False
    )
    script = tmp_path / "w.R"
    script.write_text(_R_SCRIPT)
    subprocess.run(["Rscript", str(script), _r_src(), str(tmp_path)], check=True, capture_output=True)

    with _ignore_renorm():
        pm.gseq_write_meme(motifs, str(tmp_path / "p.meme"))
        pm.gseq_write_meme(motifs, str(tmp_path / "p_bg.meme"), background=[0.31, 0.19, 0.2, 0.3], strands="+")
        pm.gseq_write_jaspar(motifs, str(tmp_path / "p.jaspar"))
        pm.gseq_write_jaspar(motifs, str(tmp_path / "p_ns.jaspar"), nsites=1)
        pm.gseq_write_jaspar(motifs, str(tmp_path / "p_exact.jaspar"), round_counts=False)
        pm.gseq_write_jaspar(motifs, str(tmp_path / "p.pfm"), format="simple")
        pm.gseq_write_homer(motifs, str(tmp_path / "p.motif"))
        pm.gseq_write_homer(motifs, str(tmp_path / "p_frac.motif"), threshold_frac=0.37)
    for stem in ("", "_bg", "_ns", "_exact", "_frac"):
        for ext in (".meme", ".jaspar", ".pfm", ".motif"):
            r = tmp_path / f"r{stem}{ext}"
            if r.exists():
                assert (tmp_path / f"p{stem}{ext}").read_bytes() == r.read_bytes(), r.name


# ---------------------------------------------------------------------------
# Round trips on the reader fixtures
# ---------------------------------------------------------------------------

def _assert_same(x: dict, y: dict, tol: float = 1e-12, attrs: bool = True):
    assert list(x) == list(y)
    for k in x:
        assert max_diff(x[k], y[k]) <= tol, k
        if attrs:
            for key, v in x[k].attrs.items():
                w = y[k].attrs[key]
                if isinstance(v, float) and isinstance(w, float):
                    assert w == pytest.approx(v, rel=1e-12), (k, key)
                else:
                    assert w == v, (k, key)


def test_meme_round_trip_keeps_metadata(tmp_path):
    x = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
    f = str(tmp_path / "x.meme")
    assert pm.gseq_write_meme(x, f) is None
    y = pm.gseq_read_meme(f)
    _assert_same(x, y)
    assert y["MA0001.1"].attrs["background"] == {"A": 0.29, "C": 0.21, "G": 0.21, "T": 0.29}
    assert y["MA0001.1"].attrs["url"] == "http://jaspar.genereg.net/matrix/MA0001.1"
    assert y["MA0003.2"].attrs["E"] is None


def test_jaspar_round_trip_keeps_counts(tmp_path):
    x = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs.jaspar"))
    f = tmp_path / "x.jaspar"
    pm.gseq_write_jaspar(x, str(f))
    _assert_same(x, pm.gseq_read_jaspar(str(f)))
    assert f.read_text().splitlines()[1] == "A  [ 10 12  4  1  0  5 ]"


def test_jaspar_simple_round_trip(tmp_path):
    x = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs_simple.pfm"))
    f = tmp_path / "test_motifs_simple.pfm"
    # without nsites (simple files have none), counts are scaled to 100
    pm.gseq_write_jaspar(x, str(f), format="simple", round_counts=False)
    _assert_same(x, pm.gseq_read_jaspar(str(f)), tol=1e-8)
    assert f.read_text().splitlines()[0] == "66.666667        80 26.666667  6.666667"
    pm.gseq_write_jaspar(x, str(f), format="simple")
    assert f.read_text().splitlines()[0] == "67 80 27  7"
    pm.gseq_write_jaspar(x, str(f), nsites=15, format="simple")
    _assert_same(x, pm.gseq_read_jaspar(str(f)))
    assert f.read_text().splitlines() == ["10 12  4  1", " 0  0  7  0", " 3  2  1  0", " 2  1  3 14"]


def test_homer_round_trip_keeps_metadata(tmp_path):
    x = pm.gseq_read_homer(str(FIXTURES / "test_motifs.homer.motif"))
    f = tmp_path / "x.motif"
    pm.gseq_write_homer(x, str(f))
    _assert_same(x, pm.gseq_read_homer(str(f)))
    assert f.read_text().splitlines()[0] == ">ATGACTCA\tAP1(bZIP)/ThP1-cJun\t8.036341\t-4130.668834"


# ---------------------------------------------------------------------------
# Round trips on random matrices
# ---------------------------------------------------------------------------

def test_random_motifs_round_trip(tmp_path):
    x = random_motifs()
    assert 1 in {len(df) for df in x.values()}
    f = str(tmp_path / "r.meme")
    pm.gseq_write_meme(x, f)
    y = pm.gseq_read_meme(f)
    assert list(y) == list(x)
    for k in x:
        assert max_diff(x[k], y[k]) <= 5e-7 + 1e-12
        name = x[k].attrs.get("name")
        assert y[k].attrs["name"] == (None if name is None else " ".join(name.split()))
        for key in ("nsites", "E"):
            v = x[k].attrs.get(key)
            assert (y[k].attrs[key] is None) if v is None else y[k].attrs[key] == pytest.approx(v, rel=1e-14)

    f = str(tmp_path / "r.jaspar")
    pm.gseq_write_jaspar(x, f, round_counts=False)
    y = pm.gseq_read_jaspar(f)
    for k in x:
        assert max_diff(x[k], y[k]) <= 1.5e-6 / x[k].attrs.get("nsites", 100.0)
        assert y[k].attrs["nsites"] == pytest.approx(x[k].attrs.get("nsites", 100.0), rel=1e-6)

    f = str(tmp_path / "r.motif")
    pm.gseq_write_homer(x, f)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Duplicate motif IDs")  # random consensuses repeat
        y = list(pm.gseq_read_homer(f).values())
    assert len(y) == len(x)
    for a, b in zip(x.values(), y, strict=True):
        assert max_diff(a, b) <= 5e-7 + 1e-12
        if "log_odds_threshold" in a.attrs:
            assert b.attrs["log_odds_threshold"] == pytest.approx(a.attrs["log_odds_threshold"], abs=5e-7)
        lp = a.attrs.get("log_p_value")
        assert (b.attrs["log_p_value"] is None) if lp is None else b.attrs["log_p_value"] == pytest.approx(lp, abs=5e-7)


@pytest.mark.parametrize("fmt", ["meme", "homer"])
def test_second_write_gives_same_bytes(tmp_path, fmt):
    # Not JASPAR: reading divides each position by its own count total, which
    # can differ from nsites in the last written digit.
    write = getattr(pm, f"gseq_write_{fmt}")
    read = getattr(pm, f"gseq_read_{fmt}")
    x = random_motifs()
    write(x, str(tmp_path / "a"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = read(str(tmp_path / "a"))
    write(y, str(tmp_path / "b"))
    assert (tmp_path / "a").read_bytes() == (tmp_path / "b").read_bytes()


# ---------------------------------------------------------------------------
# Cross-format conversion
# ---------------------------------------------------------------------------

def test_meme_to_jaspar_and_homer(tmp_path):
    x = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
    f = str(tmp_path / "x")
    pm.gseq_write_jaspar(x, f, round_counts=False)
    y = pm.gseq_read_jaspar(f)
    _assert_same(x, y, tol=1e-9, attrs=False)
    assert [d.attrs["name"] for d in y.values()] == ["AGL3", "RUNX1", "TFAP2A"]
    assert [d.attrs["nsites"] for d in y.values()] == [20, 15, 18]
    pm.gseq_write_homer(x, f)
    y = pm.gseq_read_homer(f)
    assert list(y) == ["CAGT", "TAGGTN", "CGATN"]
    assert [d.attrs["name"] for d in y.values()] == ["AGL3", "RUNX1", "TFAP2A"]


def test_homer_to_meme(tmp_path):
    x = pm.gseq_read_homer(str(FIXTURES / "test_motifs.homer.motif"))
    f = str(tmp_path / "x")
    pm.gseq_write_meme(x, f)
    y = pm.gseq_read_meme(f)
    _assert_same(x, y, tol=1e-6, attrs=False)
    a = y["ATGACTCA"].attrs
    assert a["name"] == "AP1(bZIP)/ThP1-cJun"
    assert a["nsites"] is None
    assert a["background"] == {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    assert a["strand"] == "+ -"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def test_plain_array_written_with_defaults(tmp_path):
    m = np.array([[0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]])
    f = tmp_path / "x"
    pm.gseq_write_meme(m, str(f))
    assert f.read_text().splitlines()[9:11] == ["MOTIF motif_1", "letter-probability matrix: alength= 4 w= 2"]
    y = pm.gseq_read_meme(str(f))["motif_1"]
    assert y.attrs["name"] is None and y.attrs["nsites"] is None
    np.testing.assert_array_equal(y.to_numpy(), m)
    pm.gseq_write_jaspar(m, str(f))
    assert f.read_text().splitlines() == [
        ">motif_1 motif_1", "A  [ 70 25 ]", "C  [ 10 25 ]", "G  [ 10 25 ]", "T  [ 10 25 ]"
    ]
    pm.gseq_write_homer(m, str(f))
    assert f.read_text().splitlines()[0].split("\t")[:2] == [">AN", "motif_1"]


def test_list_input_gets_motif_ids(tmp_path):
    m = np.full((2, 4), 0.25)
    f = str(tmp_path / "x")
    pm.gseq_write_meme([m, m], f)
    assert list(pm.gseq_read_meme(f)) == ["motif_1", "motif_2"]
    pm.gseq_write_meme({"": m, "X": m}, f)
    assert list(pm.gseq_read_meme(f)) == ["motif_1", "X"]


def test_columns_matched_by_name_and_transposed(tmp_path):
    m = pd.DataFrame([[0.7, 0.1, 0.1, 0.1], [0.1, 0.2, 0.3, 0.4]], columns=ACGT)
    outs = []
    for i, v in enumerate([m, m[["T", "G", "C", "A"]], m.T, m.to_numpy(), m.rename(columns=str.lower)]):
        f = tmp_path / f"x{i}"
        pm.gseq_write_meme({"M": v}, str(f))
        outs.append(f.read_bytes())
    assert all(o == outs[0] for o in outs)


def test_rows_renormalized_with_warning(tmp_path):
    m = np.array([[1, 2, 3, 4], [0, 0, 0, 0], [0.5, 0.5, 0, 0]], dtype=float)
    f = str(tmp_path / "x")
    with pytest.warns(UserWarning, match="re-normalizing 2 row"):
        pm.gseq_write_meme(m, f)
    y = pm.gseq_read_meme(f)["motif_1"].to_numpy()
    np.testing.assert_allclose(y, [[0.1, 0.2, 0.3, 0.4], [0.25] * 4, [0.5, 0.5, 0, 0]])


def test_negative_zeros_written_as_zeros(tmp_path):
    m = np.array([[-0.0, 0.5, 0.5, -0.0]])
    f = tmp_path / "x"
    pm.gseq_write_meme(m, str(f), background=[0.5, 0.5, -0.0, 0.0])
    assert "-0" not in f.read_text()
    pm.gseq_write_jaspar(m, str(f))
    assert "-0" not in f.read_text()


def test_invalid_matrices_raise(tmp_path):
    f = str(tmp_path / "x")
    with pytest.raises(ValueError, match="4 columns"):
        pm.gseq_write_meme(np.full((2, 5), 0.2), f)
    with pytest.raises(ValueError, match="negative"):
        pm.gseq_write_meme(np.array([[-1, 1, 0.5, 0.5]]), f)
    with pytest.raises(ValueError, match="missing"):
        pm.gseq_write_meme(np.array([[np.nan, 1, 0, 0]]), f)
    with pytest.raises(ValueError, match="no positions"):
        pm.gseq_write_meme(np.zeros((0, 4)), f)
    with pytest.raises(ValueError, match="non-empty"):
        pm.gseq_write_meme({}, f)
    with pytest.raises(ValueError, match="4 columns"):
        pm.gseq_write_meme(pd.DataFrame([[0.25] * 4], columns=list("ACGU")), f)
    with pytest.raises(TypeError):
        pm.gseq_write_meme(np.full((1, 4), 0.25), 3)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["meme", "jaspar", "homer"])
def test_edge_motifs_round_trip(tmp_path, fmt):
    x = edge_motifs()
    f = str(tmp_path / "x")
    kwargs = {"round_counts": False} if fmt == "jaspar" else {}
    getattr(pm, f"gseq_write_{fmt}")(x, f, **kwargs)
    y = list(getattr(pm, f"gseq_read_{fmt}")(f).values())
    assert len(y) == 3
    for a, b in zip(x.values(), y, strict=True):
        assert max_diff(a, b) <= 5e-7 + 1e-12
    assert len(y[0]) == 1
    assert list(y[1].iloc[0]) == [1, 0, 0, 0]


def test_whitespace_in_names_and_ids(tmp_path):
    m = pd.DataFrame([[0.25] * 4], columns=ACGT)
    m.attrs["name"] = "  my \t motif  name "
    f = tmp_path / "x"
    with pytest.warns(UserWarning, match="whitespace"):
        pm.gseq_write_meme({"my id": m}, str(f))
    assert f.read_text().splitlines()[9] == "MOTIF my_id my motif name"
    y = pm.gseq_read_meme(str(f))
    assert list(y) == ["my_id"] and y["my_id"].attrs["name"] == "my motif name"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pm.gseq_write_jaspar({"my id": m}, str(f))
    assert pm.gseq_read_jaspar(str(f))["my_id"].attrs["name"] == "my motif name"
    pm.gseq_write_homer({"id": m}, str(f))
    assert next(iter(pm.gseq_read_homer(str(f)).values())).attrs["name"] == "my motif name"


def test_missing_name_not_in_meme_and_id_in_jaspar(tmp_path):
    m = pd.DataFrame([[0.25] * 4], columns=ACGT)
    m.attrs["name"] = ""
    f = tmp_path / "x"
    pm.gseq_write_meme({"A1": m}, str(f))
    assert f.read_text().splitlines()[9] == "MOTIF A1"
    assert pm.gseq_read_meme(str(f))["A1"].attrs["name"] is None
    m.attrs["name"] = None
    pm.gseq_write_jaspar({"A1": m}, str(f))
    assert f.read_text().splitlines()[0] == ">A1 A1"
    assert pm.gseq_read_jaspar(str(f))["A1"].attrs["name"] == "A1"


def test_duplicate_ids_written_and_told_apart(tmp_path):
    m = np.full((2, 4), 0.25)
    ms = [pd.DataFrame(m, columns=ACGT) for _ in range(3)]
    # A dict cannot repeat keys, but HOMER IDs are consensus sequences, which can
    f = str(tmp_path / "x")
    for d in ms:
        d.attrs["consensus"] = "NN"
    pm.gseq_write_homer(ms, f)
    with pytest.warns(UserWarning, match="Duplicate"):
        y = pm.gseq_read_homer(f)
    assert list(y) == ["NN.1", "NN.2", "NN.3"]


# ---------------------------------------------------------------------------
# MEME options
# ---------------------------------------------------------------------------

def test_meme_background_and_strands(tmp_path):
    m = {"M": np.full((1, 4), 0.25)}
    f = tmp_path / "x"
    pm.gseq_write_meme(m, str(f), background={"G": 2, "C": 2, "T": 3, "A": 3}, strands="+")
    lines = f.read_text().splitlines()
    assert lines[4] == "strands: +"
    assert lines[7] == "A 0.300000 C 0.200000 G 0.200000 T 0.300000"
    a = pm.gseq_read_meme(str(f))["M"].attrs
    assert a["background"] == {"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}
    assert a["strand"] == "+"
    pm.gseq_write_meme(m, str(f), background=pd.Series([0.1, 0.4, 0.4, 0.1], index=ACGT))
    assert f.read_text().splitlines()[7] == "A 0.100000 C 0.400000 G 0.400000 T 0.100000"
    pm.gseq_write_meme(m, str(f), background=np.array([0.1, 0.4, 0.4, 0.1]))
    assert f.read_text().splitlines()[7] == "A 0.100000 C 0.400000 G 0.400000 T 0.100000"


def test_meme_shared_background_and_strands(tmp_path):
    m = pd.DataFrame([[0.25] * 4], columns=ACGT)
    m.attrs.update(background={"A": 0.3, "C": 0.2, "G": 0.2, "T": 0.3}, strand="+")
    f = tmp_path / "x"
    pm.gseq_write_meme({"a": m, "b": m}, str(f))
    lines = f.read_text().splitlines()
    assert (lines[4], lines[7]) == ("strands: +", "A 0.300000 C 0.200000 G 0.200000 T 0.300000")
    m2 = m.copy()
    m2.attrs = dict(m.attrs, background={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25})
    with pytest.warns(UserWarning, match="different backgrounds"):
        pm.gseq_write_meme({"a": m, "b": m2}, str(f))
    assert f.read_text().splitlines()[7] == "A 0.250000 C 0.250000 G 0.250000 T 0.250000"


def test_meme_invalid_background_and_strands(tmp_path):
    m = np.full((1, 4), 0.25)
    f = str(tmp_path / "x")
    for bg in ([0.5, 0.5], {"A": 0.5, "C": 0.5, "G": 0, "U": 0}, [-1, 1, 0.5, 0.5], [0, 0, 0, 0], "ACGT"):
        with pytest.raises(ValueError, match="background"):
            pm.gseq_write_meme(m, f, background=bg)
    with pytest.raises(ValueError, match="strands"):
        pm.gseq_write_meme(m, f, strands="-")


# ---------------------------------------------------------------------------
# JASPAR options
# ---------------------------------------------------------------------------

def test_jaspar_nsites(tmp_path):
    m = pd.DataFrame([[0.5, 0.25, 0.25, 0.0]], columns=ACGT)
    m2 = m.copy()
    m2.attrs["nsites"] = 8
    f = tmp_path / "x"
    pm.gseq_write_jaspar({"a": m, "b": m2}, str(f))
    lines = f.read_text().splitlines()
    assert lines[1] == "A  [ 50 ]" and lines[7] == "A  [ 4 ]"
    pm.gseq_write_jaspar({"a": m, "b": m2}, str(f), nsites=1, round_counts=False)
    assert [f.read_text().splitlines()[i] for i in (1, 2, 4)] == ["A  [  0.5 ]", "C  [ 0.25 ]", "T  [    0 ]"]
    assert [d.attrs["nsites"] for d in pm.gseq_read_jaspar(str(f)).values()] == [1, 1]
    pm.gseq_write_jaspar({"a": m, "b": m2}, str(f), nsites=[2, 3], round_counts=False)
    assert [d.attrs["nsites"] for d in pm.gseq_read_jaspar(str(f)).values()] == [2, 3]
    with pytest.raises(ValueError, match="positive"):
        pm.gseq_write_jaspar(m, str(f), nsites=0)
    with pytest.raises(ValueError, match="one per motif"):
        pm.gseq_write_jaspar(m, str(f), nsites=[1, 2])
    with pytest.raises(ValueError, match="format"):
        pm.gseq_write_jaspar(m, str(f), format="transfac")


def test_jaspar_round_counts(tmp_path):
    m = np.array([[1 / 6, 1 / 6, 1 / 3, 1 / 3], [0.45, 0.4, 0.1, 0.05]])
    f = tmp_path / "x"
    pm.gseq_write_jaspar({"a": m}, str(f), nsites=3, round_counts=True)
    # 0.5 rounds to even, as in R
    assert f.read_text().splitlines()[1:5] == ["A  [ 0 1 ]", "C  [ 0 1 ]", "G  [ 1 0 ]", "T  [ 1 0 ]"]
    pm.gseq_write_jaspar({"a": m}, str(f), nsites=1000, round_counts=True)
    assert np.max(np.abs(pm.gseq_read_jaspar(str(f))["a"].to_numpy() - m)) <= 1e-3
    with pytest.raises(ValueError, match="round_counts"):
        pm.gseq_write_jaspar(m, str(f), round_counts=None)
    # the default
    pm.gseq_write_jaspar({"a": m}, str(f), nsites=3)
    assert f.read_text().splitlines()[1] == "A  [ 0 1 ]"


# ---------------------------------------------------------------------------
# HOMER options
# ---------------------------------------------------------------------------

def test_homer_consensus_rule(tmp_path):
    m = np.array([
        [0.97, 0.01, 0.01, 0.01],
        [0.45, 0.40, 0.10, 0.05],
        [0.40, 0.05, 0.30, 0.25],
        [0.10, 0.30, 0.30, 0.30],
        [0.25, 0.25, 0.25, 0.25],
        [0.05, 0.40, 0.15, 0.40],
    ])
    f = str(tmp_path / "x")
    pm.gseq_write_homer({"id": m}, f)
    assert list(pm.gseq_read_homer(f)) == ["AMRBNY"]


def test_homer_stored_consensus(tmp_path):
    m = next(iter(pm.gseq_read_homer(str(FIXTURES / "test_motifs.homer.motif")).values()))
    f = str(tmp_path / "x")
    m.attrs["consensus"] = "NNNNNNNN"
    pm.gseq_write_homer({"a": m}, f)
    assert list(pm.gseq_read_homer(f)) == ["NNNNNNNN"]
    m.attrs["consensus"] = "NNN"
    pm.gseq_write_homer({"a": m}, f)
    assert list(pm.gseq_read_homer(f)) == ["DTTACGTN"]


def test_homer_thresholds(tmp_path):
    m = pd.DataFrame([[1, 0, 0, 0], [0.5, 0.5, 0, 0]], columns=ACGT, dtype=float)
    best = np.log(1 / 0.25) + np.log(0.5 / 0.25)
    f = str(tmp_path / "x")

    def thr():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Duplicate motif IDs")
            return [d.attrs["log_odds_threshold"] for d in pm.gseq_read_homer(f).values()]

    pm.gseq_write_homer({"a": m}, f)
    assert thr()[0] == pytest.approx(0.6 * best, abs=5e-7)
    pm.gseq_write_homer({"a": m}, f, threshold_frac=1)
    assert thr()[0] == pytest.approx(best, abs=5e-7)
    m.attrs["log_odds_threshold"] = 1.5
    pm.gseq_write_homer({"a": m}, f)
    assert thr() == [1.5]
    pm.gseq_write_homer([m, m], f, threshold=[2, 3])
    assert thr() == [2, 3]
    with pytest.raises(ValueError, match="threshold"):
        pm.gseq_write_homer(m, f, threshold=float("nan"))
    with pytest.raises(ValueError, match="threshold_frac"):
        pm.gseq_write_homer(m, f, threshold_frac="a")


def test_homer_name_defaults_to_id(tmp_path):
    f = tmp_path / "x"
    pm.gseq_write_homer({"MY_ID": np.full((1, 4), 0.25)}, str(f))
    assert next(iter(pm.gseq_read_homer(str(f)).values())).attrs["name"] == "MY_ID"
    assert len(f.read_text().splitlines()[0].split("\t")) == 3
