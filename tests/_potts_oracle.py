"""The pure-Python Potts oracle, ported from R misha's tests/testthat/helper-potts.R.

Deliberately slow, deliberately non-reusing: it is the ground truth for
src/PottsModel.h, so it must not share a line of logic with it. Do not
"optimise" it and do not fix it to match the kernel.

Base order is A, C, G, T everywhere. A model here is a plain dict, which is also
the shape ``gvtrack_create()`` takes as ``params`` and the shape a fitted Potts
model already has.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

POTTS_BASES = ("A", "C", "G", "T")
_COMP = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


def log_sum_exp(values) -> float:
    """log(sum(exp(v))) over a finite, non-empty sequence."""
    vals = [float(v) for v in values]
    if not vals:
        return float("-inf")
    m = max(vals)
    if math.isinf(m) and m < 0:
        return m
    return m + math.log(sum(math.exp(v - m) for v in vals))


def revcomp(seq: str) -> str:
    return "".join(_COMP[c] for c in reversed(seq.upper()))


def potts_ref_model(w: int, npair_mode: str = "full", seed: int = 60427) -> dict:
    """A reproducible test model.

    ``npair_mode`` is "full" (every pair), "sparse" (every third pair) or
    "none" (order 1).
    """
    if npair_mode not in ("full", "sparse", "none"):
        raise ValueError("npair_mode must be 'full', 'sparse' or 'none'")
    rng = np.random.default_rng(seed)
    e = np.round(rng.normal(size=(w, 4)), 4)
    full = np.array(
        [[i + 1, j + 1] for i, j in itertools.combinations(range(w), 2)],
        dtype=np.int64,
    ).reshape(-1, 2)
    if npair_mode == "full":
        pairs = full
    elif npair_mode == "sparse":
        pairs = full[::3]
    else:
        pairs = np.zeros((0, 2), dtype=np.int64)
    j = [np.round(rng.normal(size=(4, 4)), 4) for _ in range(len(pairs))]
    return {
        "e": e,
        "J": j,
        "pairs": pairs,
        "intercept": float(np.round(rng.normal(), 4)),
    }


def potts_ref_window(win, model: dict) -> float:
    """Score ONE window of exactly ``len(model['e'])`` bases.

    NaN for any non-ACGT base, for None, and for the wrong width - a Potts has
    no prior, so an N has no energy.
    """
    w = len(model["e"])
    if win is None or not isinstance(win, str) or len(win) != w:
        return float("nan")
    try:
        codes = [POTTS_BASES.index(c) for c in win.upper()]
    except ValueError:
        return float("nan")
    s = float(model["intercept"])
    for i in range(w):
        s += float(model["e"][i][codes[i]])
    for k in range(len(model["pairs"])):
        i = int(model["pairs"][k][0])
        j = int(model["pairs"][k][1])
        s += float(model["J"][k][codes[i - 1]][codes[j - 1]])
    return s


def potts_ref_rc(model: dict) -> dict:
    """The reverse-complement twin.

        e'[i][b]         = e[W+1-i][comp(b)]
        pairs'           = (W+1-p2, W+1-p1), re-sorted
        J'_{(u,v)}[a][b] = J_k[comp(b)][comp(a)]

    so that ``potts_ref_window(w, potts_ref_rc(m))`` equals
    ``potts_ref_window(revcomp(w), m)``.
    """
    w = len(model["e"])
    comp = (3, 2, 1, 0)  # A<->T, C<->G in A,C,G,T order
    e2 = np.zeros((w, 4), dtype=float)
    for i in range(w):
        for b in range(4):
            e2[i][b] = float(model["e"][w - 1 - i][comp[b]])

    pairs = np.asarray(model["pairs"], dtype=np.int64).reshape(-1, 2)
    if len(pairs):
        u = w + 1 - pairs[:, 1]
        v = w + 1 - pairs[:, 0]
        order = sorted(range(len(pairs)), key=lambda k: (u[k], v[k]))
        pairs2 = np.array([[u[k], v[k]] for k in order], dtype=np.int64)
        j2 = []
        for k in order:
            jk = np.asarray(model["J"][k], dtype=float)
            m = np.zeros((4, 4), dtype=float)
            for a in range(4):
                for b in range(4):
                    m[a][b] = jk[comp[b]][comp[a]]
            j2.append(m)
    else:
        pairs2 = np.zeros((0, 2), dtype=np.int64)
        j2 = []
    return {"e": e2, "J": j2, "pairs": pairs2, "intercept": model["intercept"]}


def potts_ref_anchors(
    seq: str,
    model: dict,
    bidirect: bool = True,
    strand: int = 1,
    union: str = "lse",
) -> np.ndarray:
    """Per-anchor values over ``seq``, in the original genome's forward orientation.

    ``union`` says how the two strands combine at one anchor: "lse" for potts,
    potts.max and potts.count; "max" for potts.max.pos. This asymmetry is
    inherited from the pwm family on purpose.
    """
    if union not in ("lse", "max"):
        raise ValueError("union must be 'lse' or 'max'")
    w = len(model["e"])
    n = len(seq) - w + 1
    if n < 1:
        return np.zeros(0, dtype=float)
    rc = potts_ref_rc(model)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        win = seq[i:i + w]
        fwd = potts_ref_window(win, model)
        rev = potts_ref_window(win, rc)
        if not bidirect:
            out[i] = rev if strand == -1 else fwd
            continue
        if math.isnan(fwd) or math.isnan(rev):
            out[i] = float("nan")
            continue
        out[i] = max(fwd, rev) if union == "max" else log_sum_exp([fwd, rev])
    return out


def random_seqs(n: int, length: int, rng: np.random.Generator) -> list[str]:
    """``n`` random ACGT sequences of the given length."""
    idx = rng.integers(0, 4, size=(n, length))
    return ["".join(POTTS_BASES[c] for c in row) for row in idx]
