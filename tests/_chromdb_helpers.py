"""Databases for the chromosome order tests, as R misha's test helpers build them
(helper-test_db.R: create_test_db, create_db_with_unsorted_chrom_sizes)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pymisha as pm
from pymisha.db_create import _write_index_file

# The bases of unsorted_db()'s chromosomes, by name
UNSORTED_BASES = {"chr1": "G", "chr1_KI270706v1_random": "N", "chr10": "C", "chr2": "A", "chrX": "T"}
SORTED_PREFIXED = list(UNSORTED_BASES)


def unsorted_db(path: str | Path) -> str:
    """A per-chromosome database laid out like the lab's hg38 and mm10: chrom_sizes.txt
    names have no "chr" prefix and are not sorted, the .seq files have the prefix, and
    there is no seq/genome.idx. gdb_init names the chromosomes chr2, chr10, chr1, chrX and
    chr1_KI270706v1_random and numbers them in the order of the sorted names. Each
    chromosome's sequence is a different letter, so a sequence read from the wrong
    chromosome shows."""
    path = Path(path)
    chroms = ["2", "10", "1", "X", "1_KI270706v1_random"]
    sizes = [2000, 1500, 1000, 1200, 500]
    (path / "tracks").mkdir(parents=True, exist_ok=True)
    (path / "seq").mkdir(exist_ok=True)
    for chrom, size, base in zip(chroms, sizes, "ACGTN", strict=True):
        (path / "seq" / f"chr{chrom}.seq").write_bytes(base.encode() * size)
    (path / "chrom_sizes.txt").write_text("".join(f"{c}\t{s}\n" for c, s in zip(chroms, sizes, strict=True)))
    return str(path.resolve())


def create_test_db(path: str | Path, chrom_sizes: list[tuple[str, int]] | None = None) -> str:
    """A per-chromosome database with .seq files named by the chrom_sizes.txt names."""
    path = Path(path)
    chrom_sizes = chrom_sizes or [("chr1", 10000), ("chr2", 10000)]
    (path / "tracks").mkdir(parents=True, exist_ok=True)
    (path / "seq").mkdir(exist_ok=True)
    for chrom, size in chrom_sizes:
        (path / "seq" / f"{chrom}.seq").write_bytes(b"A" * size)
    (path / "chrom_sizes.txt").write_text("".join(f"{c}\t{s}\n" for c, s in chrom_sizes))
    return str(path.resolve())


def write_index(path: str | Path, names: list[str], sizes: list[int]) -> None:
    """seq/genome.seq and seq/genome.idx with the given contigs, in that order."""
    seq = Path(path) / "seq"
    seq.mkdir(parents=True, exist_ok=True)
    entries, offset = [], 0
    with open(seq / "genome.seq", "wb") as fh:
        for k, (name, size) in enumerate(zip(names, sizes, strict=True)):
            fh.write(b"A" * size)
            entries.append({"chromid": k, "name": name, "offset": offset, "length": size})
            offset += size
    _write_index_file(str(seq / "genome.idx"), entries)


def chrom_sizes_rows(db: str | Path) -> list[tuple[str, int]]:
    lines = (Path(db) / "chrom_sizes.txt").read_text().splitlines()
    return [(line.split("\t")[0], int(line.split("\t")[1])) for line in lines if line.strip()]


def indexed_like(src: str | Path, path: str | Path, contig: str | None = None) -> str:
    """An indexed database with the chrom_sizes.txt of src, its chrom ids in
    chrom_sizes.txt order; with contig, its index names the contigs <contig>1, ..."""
    path = Path(path)
    (path / "tracks").mkdir(parents=True)
    rows = chrom_sizes_rows(src)
    names = [f"{contig}{k + 1}" for k in range(len(rows))] if contig else [c for c, _ in rows]
    write_index(path, names, [s for _, s in rows])
    (path / "chrom_sizes.txt").write_bytes((Path(src) / "chrom_sizes.txt").read_bytes())
    return str(path.resolve())


def first_bases() -> dict[str, str]:
    """The first base of each of unsorted_db()'s chromosomes, as the loaded database reads it."""
    return {c: pm.gseq_extract(pm.gintervals(c, 0, 1))[0].upper() for c in UNSORTED_BASES}


def with_indexed_track(db: str, name: str = "it") -> str:
    """db, loaded, with an indexed sparse track holding 1, 2, ... by chrom id."""
    pm.gdb_init(db)
    iv = pm.gintervals_all()
    iv["end"] = 10
    pm.gtrack_create_sparse(name, "x", iv, list(range(1, len(iv) + 1)))
    pm.gtrack_convert_to_indexed(name)
    assert (Path(db) / "tracks" / f"{name}.track" / "track.idx").exists()
    return db


_CHROMDB = Path(__file__).resolve().parent / "r_parity" / "chromdb" / "db"


def copy_r_bigset(name: str, dest: str | Path) -> Path:
    """A big intervals set R misha wrote (r_parity/chromdb: big1d, big2d and their indexed
    versions), on the chromosomes of unsorted_db(), copied to dest/tracks/<name>.interv."""
    target = Path(dest) / "tracks" / f"{name}.interv"
    shutil.copytree(_CHROMDB / "tracks" / f"{name}.interv", target)
    return target
