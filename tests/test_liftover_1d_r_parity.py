"""gtrack_liftover of a sparse or a dense track gives the track R misha gives.

R reads every source value, NaN included and an infinite one as NaN. For a
sparse source it cuts the lifted pieces of each target chromosome at their
boundaries and gives each segment aggregate_values over the pieces covering it
(the pieces of one source interval through one chain once), keeping a NaN
result. For a dense source it aggregates what lands in each bin. The sources,
the chain, and the intervals, values and file hashes of R's lifted tracks are
frozen in tests/test_data/liftover_1d (written by
tools/generate_liftover_1d_baseline.R with R misha 5.18.0), so no R is needed.
"""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import pymisha as pm

FIXTURE = Path(__file__).parent / "test_data" / "liftover_1d"
BASELINE = json.loads((FIXTURE / "baseline.json").read_text())
SEQS = {"chrT1": 8000, "chrT2": 6000, "chrT3": 3000}


@pytest.fixture(scope="module")
def dbs(tmp_path_factory):
    """The target database of the baseline, per format."""
    base = tmp_path_factory.mktemp("liftover_1d_r_parity")
    fastas = []
    for c, n in SEQS.items():
        (base / f"{c}.fasta").write_text(f">{c}\n{'A' * n}\n")
        fastas.append(str(base / f"{c}.fasta"))
    pm.gdb_create(str(base / "per-chromosome"), fastas, db_format="per-chromosome")
    (base / "genome.fasta").write_text("".join(f">{c}\n{'A' * n}\n" for c, n in SEQS.items()))
    pm.gdb_create(str(base / "indexed"), [str(base / "genome.fasta")], db_format="indexed")
    return base


def test_baseline_is_r_5_18_0():
    assert BASELINE["misha"] == "5.18.0"


@pytest.mark.parametrize("force_python", [False, True], ids=["cpp", "python"])
@pytest.mark.parametrize("case", BASELINE["cases"], ids=lambda c: c["name"])
def test_lifted_track_is_rs(dbs, case, force_python, monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "progress", False)
    db = dbs / case["tgt"]
    pm.gdb_init(str(db))
    name = case["name"]
    track_dir = db / "tracks" / f"{name}.track"
    if track_dir.exists():
        shutil.rmtree(track_dir)
        pm.gdb_reload()
    kwargs = {}
    if case["nth"] is not None:
        kwargs["params"] = int(case["nth"])
    if case["min_n"] is not None:
        kwargs["min_n"] = int(case["min_n"])
    pm.gtrack_liftover(
        name, "x", str(FIXTURE / "src" / "tracks" / f"{case['track']}.track"), str(FIXTURE / "test.chain"),
        src_overlap_policy="keep", tgt_overlap_policy=case["policy"], multi_target_agg=case["agg"],
        na_rm=case["na_rm"], _force_pure_python=force_python, **kwargs)

    # the intervals and values R reads back
    if case["track"] == "sp":
        got = pm.gextract(name, pm.gintervals_all(), iterator=name)
    else:
        covered = pm.gintervals(["chrT1", "chrT1", "chrT1", "chrT2", "chrT3"], [0, 2000, 7500, 100, 0],
                                [700, 2500, 8000, 590, 100])
        got = pm.gextract(name, covered, iterator=20)
    exp = case["intervals"]
    assert list(zip(got["chrom"].astype(str), got["start"], got["end"], strict=True)) == list(
        zip(np.atleast_1d(exp["chrom"]), np.atleast_1d(exp["start"]), np.atleast_1d(exp["end"]), strict=True))
    np.testing.assert_array_equal(got[name].to_numpy(dtype=np.float32),
                                  np.array([float(v) for v in np.atleast_1d(exp["value"])], dtype=np.float32))

    # and the files R writes, but for the empty chromosome of a sparse indexed
    # track, which R's lift writes as a 4-byte header and pymisha as nothing
    if case["track"] == "sp" and case["tgt"] == "indexed":
        return
    files = {p.name: (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
             for p in sorted(track_dir.iterdir()) if not p.name.startswith(".")}
    assert files == {f: (v["size"], v["sha256"]) for f, v in case["files"].items()}
