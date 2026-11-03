"""Chromosome names and chrom ids as R misha's gsetroot gives them (``.gdb.chrom_order``).

A per-chromosome database whose chrom_sizes.txt names lack the "chr" prefix of its seq
files gets prefixed names, and its chrom ids follow R's ``order()`` of those names. Any
other database keeps the chrom_sizes.txt names and order. The cross-reading tests against
R are in ``r_parity/test_chrom_names.py``.
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import _pymisha
import pytest

import pymisha as pm

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "r_order_chrom_names.txt"


def _make_db(root: Path, chroms: list[str], seqs: list[str], genome_idx: bool = False) -> Path:
    (root / "seq").mkdir(parents=True)
    (root / "tracks").mkdir()
    for s in seqs:
        (root / "seq" / f"{s}.seq").write_text("ACGT" * 10)
    if genome_idx:
        # indexed: genome.idx and genome.seq both
        (root / "seq" / "genome.idx").write_bytes(b"")
        (root / "seq" / "genome.seq").write_bytes(b"")
    (root / "chrom_sizes.txt").write_text("".join(f"{c}\t40\n" for c in chroms))
    return root


_UNSORTED = ["2", "10", "1", "X", "1_KI270706v1_random"]
_SORTED_PREFIXED = ["chr1", "chr1_KI270706v1_random", "chr10", "chr2", "chrX"]


@pytest.mark.parametrize(
    "chroms, seqs, expected",
    [
        # per-chromosome: prefixed and sorted as R's order() sorts (not as Python's sorted)
        (_UNSORTED, [f"chr{c}" for c in _UNSORTED], _SORTED_PREFIXED),
        # 80% unprefixed is enough; a prefixed name is kept
        (["2", "10", "1", "chrX", "1_KI270706v1_random"], [f"chr{c}" for c in _UNSORTED], _SORTED_PREFIXED),
        # under 80% unprefixed: as in chrom_sizes.txt
        (["2", "10", "chr1", "chrX", "1_KI270706v1_random"], [f"chr{c}" for c in _UNSORTED],
         ["2", "10", "chr1", "chrX", "1_KI270706v1_random"]),
        # an unprefixed seq file among the first five: as in chrom_sizes.txt
        (_UNSORTED, ["2"] + [f"chr{c}" for c in _UNSORTED], _UNSORTED),
        # a missing chr-prefixed seq file among the first five: as in chrom_sizes.txt
        (_UNSORTED, [f"chr{c}" for c in _UNSORTED[1:]], _UNSORTED),
        # only the first five unprefixed names are looked at
        (["2", "10", "1", "X", "Y", "1_KI270706v1_random"], [f"chr{c}" for c in ["2", "10", "1", "X", "Y"]],
         ["chr1", "chr1_KI270706v1_random", "chr10", "chr2", "chrX", "chrY"]),
        # names that already have the prefix: as in chrom_sizes.txt
        (["chr2", "chr10", "chr1"], ["chr2", "chr10", "chr1"], ["chr2", "chr10", "chr1"]),
    ],
)
def test_names_and_order_follow_r_rule(tmp_path, chroms, seqs, expected):
    db = _make_db(tmp_path / "db", chroms, seqs)
    names, sizes, file_names, per_chromosome = _pymisha.pm_gdb_chrom_order(str(db))
    assert names == expected
    assert per_chromosome == (expected != chroms)
    # the chrom_sizes.txt name of each chromosome, by id
    assert [n if n in chroms else n[3:] for n in names] == file_names
    pm.gdb_init(str(db))
    assert pm.gintervals_all()["chrom"].tolist() == expected


def test_indexed_genome_keeps_chrom_sizes_order(tmp_path):
    db = _make_db(tmp_path / "db", _UNSORTED, [f"chr{c}" for c in _UNSORTED], genome_idx=True)
    names, _, _, per_chromosome = _pymisha.pm_gdb_chrom_order(str(db))
    assert names == _UNSORTED
    assert not per_chromosome


def test_order_is_r_order(tmp_path):
    """The chrom ids follow R's order() in the lab's locale (ICU collation), frozen in
    the fixture: 1448 names, the hg38 contigs and random printable ASCII ones."""
    r_order = _FIXTURE.read_text().splitlines()
    assert r_order != sorted(r_order)  # Python's order differs
    shuffled = r_order[:]
    random.Random(5).shuffle(shuffled)
    # the per-chromosome rule looks at the seq files of the first five names
    first = [n for n in shuffled if "/" not in n][:5]
    shuffled = first + [n for n in shuffled if n not in first]
    db = tmp_path / "db"
    (db / "seq").mkdir(parents=True)
    (db / "tracks").mkdir()
    for n in first:
        (db / "seq" / f"{n}.seq").write_text("A")
    (db / "chrom_sizes.txt").write_text("".join(f"{n[3:]}\t1\n" for n in shuffled))
    names, _, file_names, per_chromosome = _pymisha.pm_gdb_chrom_order(str(db))
    assert per_chromosome
    assert names == r_order
    assert file_names == [n[3:] for n in r_order]


def test_aliases_and_canonical_names(tmp_path):
    db = _make_db(tmp_path / "db", _UNSORTED + ["MT"], [f"chr{c}" for c in _UNSORTED + ["MT"]])
    pm.gdb_init(str(db))
    assert pm.gintervals("1", 0, 10)["chrom"].tolist() == ["chr1"]
    assert pm.gintervals("chr1", 0, 10)["chrom"].tolist() == ["chr1"]
    assert pm.gintervals("M", 0, 10)["chrom"].tolist() == ["chrMT"]
    assert pm.gintervals(["X", "10", "1_KI270706v1_random"], 0, 10)["chrom"].tolist() == [
        "chr1_KI270706v1_random", "chr10", "chrX"
    ]


def test_duplicate_after_prefixing_is_an_error(tmp_path):
    db = _make_db(tmp_path / "db", ["1", "2", "3", "4", "chr1"], [f"chr{c}" for c in ["1", "2", "3", "4"]])
    with pytest.raises(Exception, match="Chromosome \"chr1\" appears more than once"):
        pm.gdb_init(str(db))
    # the failure leaves nothing behind for the next database
    good = _make_db(tmp_path / "good", _UNSORTED, [f"chr{c}" for c in _UNSORTED])
    pm.gdb_init(str(good))
    assert pm.gintervals_all()["chrom"].tolist() == _SORTED_PREFIXED


# --- R misha 5.13.1 (PR #193): the order in every locale, and an indexed seq/ checked ---

from _chromdb_helpers import (  # noqa: E402
    SORTED_PREFIXED,
    UNSORTED_BASES,
    chrom_sizes_rows,
    create_test_db,
    first_bases,
    unsorted_db,
    write_index,
)

# R's test-chromid-ordering.R names, and the order R's order() gives them under ICU's root
# collation (R misha origin/master, icuSetCollate(locale = "root"))
_R_KEY_NAMES = [
    "chr1", "chr10", "chr1_KI270706v1_random", "chr2", "chrX", "chrY", "chrM", "chrEBV",
    "chrUn_GL000220v1", "chrUn_KI270302v1", "chr6_GL000250v2_alt", "chr2_KI270894v1_alt",
    "HLA-A*01:01:01:01", "HLA-A*01:01:01:02N", "HLA-B*07:02:01", "HLA-DRB1*15:01:01:01",
    "NC_000001.11", "NC_000010.11", "NT_187361.1", "chr1.1", "chr1-1", "chr1_1",
    "Chr1", "CHR1", "chr1a", "chr1A", "scaffold_10", "scaffold-10", "scaffold.10", "Scaffold_9",
    "super-scaffold_1", "Super_Scaffold_1", "MT", "mt", "contig#2", "contig@2", "contig(2)",
]
_R_KEY_SORTED = [
    "chr1", "Chr1", "CHR1", "chr1_1", "chr1_KI270706v1_random", "chr1-1", "chr1.1",
    "chr10", "chr1a", "chr1A", "chr2", "chr2_KI270894v1_alt", "chr6_GL000250v2_alt", "chrEBV",
    "chrM", "chrUn_GL000220v1", "chrUn_KI270302v1", "chrX", "chrY", "contig(2)", "contig@2",
    "contig#2", "HLA-A*01:01:01:01", "HLA-A*01:01:01:02N", "HLA-B*07:02:01",
    "HLA-DRB1*15:01:01:01", "mt", "MT", "NC_000001.11", "NC_000010.11", "NT_187361.1",
    "scaffold_10", "Scaffold_9", "scaffold-10", "scaffold.10", "Super_Scaffold_1",
    "super-scaffold_1",
]


def test_sort_key_orders_names_as_r_icu_root(tmp_path):
    """R: the chrom sort key orders names as order() does under ICU root collation. The
    names get a common prefix ("q", then "chr" as a per-chromosome database adds it), which
    leaves their order as it is. R's "chr1 1" is left out: pymisha reads chrom_sizes.txt
    split at any whitespace, not only at the tab."""
    db = tmp_path / "db"
    (db / "seq").mkdir(parents=True)
    (db / "tracks").mkdir()
    for name in _R_KEY_NAMES[:5]:
        (db / "seq" / f"chrq{name}.seq").write_text("A")
    (db / "chrom_sizes.txt").write_text("".join(f"q{n}\t1\n" for n in _R_KEY_NAMES))
    names, _, _, per_chromosome = _pymisha.pm_gdb_chrom_order(str(db))
    assert per_chromosome
    assert [n[4:] for n in names] == _R_KEY_SORTED


@pytest.mark.parametrize("collate", ["C", "en_US.UTF-8"])
def test_same_chrom_ids_in_every_locale(tmp_path, collate):
    """R: a per-chromosome database gets the same chrom ids in a C and an en_US locale
    (a C locale's byte order would put chr10 before chr1_KI270706v1_random)."""
    import locale

    old = locale.setlocale(locale.LC_COLLATE)
    try:
        locale.setlocale(locale.LC_COLLATE, collate)
    except locale.Error:
        pytest.skip(f"no {collate} locale")
    try:
        db = unsorted_db(tmp_path / "db")
        assert _pymisha.pm_gdb_chrom_order(db)[0] == SORTED_PREFIXED
        pm.gdb_init(db)
        assert pm.gintervals_all()["chrom"].tolist() == SORTED_PREFIXED
    finally:
        locale.setlocale(locale.LC_COLLATE, old)


def test_genome_idx_without_genome_seq_is_not_indexed(tmp_path):
    """R: a genome.idx left without genome.seq does not make a per-chromosome db count as
    indexed; it converts to the order a clean copy of the database converts to."""
    db = unsorted_db(tmp_path / "db")
    ref = unsorted_db(tmp_path / "ref")
    (Path(db) / "seq" / "genome.idx").write_bytes(b"")
    assert _pymisha.pm_gdb_chrom_order(db)[3]
    pm.gdb_convert_to_indexed(groot=db, validate=False)
    pm.gdb_convert_to_indexed(groot=ref, validate=False)
    assert (Path(db) / "chrom_sizes.txt").read_text() == (Path(ref) / "chrom_sizes.txt").read_text()


def test_gsetroot_stops_when_index_has_another_order(tmp_path):
    """R: gsetroot stops when seq/genome.idx has the chromosomes of chrom_sizes.txt in
    another order (equal sizes: only the names tell)."""
    parent = tmp_path / "parent"
    (parent / "tracks").mkdir(parents=True)
    (parent / "seq").mkdir()
    for chrom, base in zip(["2", "10", "1"], "ACG", strict=True):
        (parent / "seq" / f"chr{chrom}.seq").write_bytes(base.encode() * 1000)
    (parent / "chrom_sizes.txt").write_text("2\t1000\n10\t1000\n1\t1000\n")
    pm.gdb_init(str(parent))
    pm.gtrack_create_sparse("s", "x", pm.gintervals("chr1", 0, 10), [1])
    ds = tmp_path / "ds"
    pm.gdataset_save(str(ds), "d", tracks="s")
    pm.gdb_unload()
    pm.gdb_convert_to_indexed(groot=str(parent))
    with pytest.raises(Exception, match="2 is chrom id 0 in chrom_sizes.txt and 2 in the index"):
        pm.gdb_init(str(ds))


def test_gsetroot_loads_index_whose_first_contigs_are_listed(tmp_path):
    """R: gsetroot loads an indexed database whose chrom_sizes.txt lists only the index's
    first contigs, and stops for any other subset."""
    db = unsorted_db(tmp_path / "db")
    pm.gdb_convert_to_indexed(groot=db, validate=False)
    cs = Path(db) / "chrom_sizes.txt"
    full = cs.read_text().splitlines()
    # chr1 1000, chr1_KI270706v1_random 500, chr10 1500, chr2 2000, chrX 1200
    cs.write_text("\n".join(full[:3]) + "\n")
    pm.gdb_init(db)
    first3 = list(UNSORTED_BASES)[:3]
    assert [pm.gseq_extract(pm.gintervals(c, 0, 1))[0].upper() for c in first3] == [
        UNSORTED_BASES[c] for c in first3
    ]
    pm.gdb_unload()
    stops = {
        "a contig left out of the middle": [full[0], full[2], full[3]],
        "a size changed": [full[0], "chr1_KI270706v1_random\t501"],
        "more contigs than the index": [*full, "chrY\t100"],
    }
    for lines in stops.values():
        cs.write_text("\n".join(lines) + "\n")
        with pytest.raises(Exception, match="does not match chrom_sizes.txt"):
            pm.gdb_init(db)
    # a name that differs among the first ones
    cs.write_text(f"{full[0]}\nchrY\t500\n")
    with pytest.raises(Exception, match="lists 2 of the index's 5 contigs, which have to be its first ones"):
        pm.gdb_init(db)


def test_gsetroot_stops_for_a_chromosome_listed_twice(tmp_path):
    """R: gsetroot stops for an indexed database whose chrom_sizes.txt lists a chromosome
    twice (by its name, or with and without "chr"); names equal to the index's are not
    compared without "chr"."""
    db = unsorted_db(tmp_path / "db")
    pm.gdb_convert_to_indexed(groot=db, validate=False)
    cs = Path(db) / "chrom_sizes.txt"
    full = cs.read_text().splitlines()
    cs.write_text("\n".join([*full[:4], "1\t1200"]) + "\n")
    with pytest.raises(Exception, match=r"lists the same chromosome twice: chr1 \(chrom id 0\) and 1 \(chrom id 4\)"):
        pm.gdb_init(db)
    cs.write_text("\n".join([*full[:3], "chr10\t2000", full[4]]) + "\n")
    twice = r"lists the same chromosome twice: chr10 \(chrom id 2\) and chr10 \(chrom id 3\)"
    with pytest.raises(Exception, match=twice):
        pm.gdb_init(db)

    # "chr1" and "1" can be two contigs
    two = tmp_path / "two"
    (two / "tracks").mkdir(parents=True)
    write_index(two, ["chr1", "1"], [100, 100])
    with open(two / "seq" / "genome.seq", "r+b") as fh:
        fh.seek(100)
        fh.write(b"C" * 100)
    (two / "chrom_sizes.txt").write_text("chr1\t100\n1\t100\n")
    pm.gdb_init(str(two))
    assert [s.upper() for s in pm.gseq_extract(pm.gintervals(["chr1", "1"], 0, 3))] == ["AAA", "CCC"]
    # those two, and a third contig named otherwise in chrom_sizes.txt: only that one is compared
    write_index(two, ["chr1", "1", "contig3"], [100, 100, 100])
    (two / "chrom_sizes.txt").write_text("chr1\t100\n1\t100\nchrY\t100\n")
    with pytest.warns(UserWarning, match="names 1 of its 3 contigs differently"):
        pm.gdb_init(str(two))
    # names of other contigs with and without "chr", which no index contig has, load as before;
    # a name listed twice does not load
    for idx, names, loads in [(["c1", "c2"], ["chr5", "5"], True), (["chr1", "c2"], ["chr1", "chrchr1"], True),
                              (["A", "B"], ["X", "X"], False)]:
        write_index(two, idx, [100, 100])
        (two / "chrom_sizes.txt").write_text("".join(f"{n}\t100\n" for n in names))
        if loads:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pm.gdb_init(str(two))
            assert pm.gintervals_all()["chrom"].tolist() == names
        else:
            with pytest.raises(Exception, match="lists the same chromosome twice"):
                pm.gdb_init(str(two))


def test_gsetroot_warns_when_only_index_names_differ(tmp_path):
    """R: gsetroot warns when only the names in seq/genome.idx differ from chrom_sizes.txt."""
    db = unsorted_db(tmp_path / "db")
    pm.gdb_convert_to_indexed(groot=db, validate=False)
    cs = Path(db) / "chrom_sizes.txt"
    cs.write_text(cs.read_text().replace("chrX\t", "chrW\t"))
    with pytest.warns(UserWarning, match="names 1 of its 5 contigs differently"):
        pm.gdb_init(db)
    assert pm.gseq_extract(pm.gintervals("chrW", 0, 1))[0].upper() == "T"


def test_failed_gdb_init_keeps_the_loaded_database(tmp_path):
    """R: gsetroot stops before the session is touched (an index that does not match, a
    chromosome listed twice): the database loaded before stays loaded."""
    good = unsorted_db(tmp_path / "good")
    bad = unsorted_db(tmp_path / "bad")
    pm.gdb_convert_to_indexed(groot=bad, validate=False)
    cs = Path(bad) / "chrom_sizes.txt"
    cs.write_text(cs.read_text().replace("chr2\t2000", "chr2\t2001"))
    pm.gdb_init(good)
    pm.gtrack_create_sparse("t", "x", pm.gintervals("chr1", 0, 10), [1])
    with pytest.raises(Exception, match="does not match chrom_sizes.txt"):
        pm.gdb_init(bad)
    assert good == pm._shared._GROOT
    assert pm.gextract("t", pm.gintervals("chr1", 0, 10))["t"].tolist() == [1]
    assert first_bases() == UNSORTED_BASES


def test_chrom_names_at_and_is_indexed_at_with_no_database(tmp_path):
    """R: .gdb.chrom_names_at and .gdb.is_indexed_at work with no database loaded."""
    from pymisha._shared import _gdb_chrom_names_at
    from pymisha.tracks import _db_is_indexed_at

    db = unsorted_db(tmp_path / "db")
    empty = tmp_path / "empty"
    (empty / "seq").mkdir(parents=True)
    pm.gdb_unload()
    assert _gdb_chrom_names_at(db) == SORTED_PREFIXED
    assert not _db_is_indexed_at(db)
    assert not _db_is_indexed_at(str(empty))


def test_chrom_names_at_probe_without_loading(tmp_path):
    """R: .gdb.is_indexed_at and .gdb.chrom_names_at probe a db without loading it; and
    .gdb.chrom_names_at gives the chrom id order gdb_init gives a per-chromosome db."""
    from pymisha._shared import _gdb_chrom_names_at
    from pymisha.tracks import _db_is_indexed_at

    perchrom = create_test_db(tmp_path / "perchrom_db")
    assert not _db_is_indexed_at(perchrom)
    assert _gdb_chrom_names_at(perchrom) == ["chr1", "chr2"]
    db = unsorted_db(tmp_path / "unsorted_db")
    names_at = _gdb_chrom_names_at(db)
    pm.gdb_init(db)
    assert names_at == pm.gintervals_all()["chrom"].tolist()
    assert chrom_sizes_rows(db)[0] == ("2", 2000)
