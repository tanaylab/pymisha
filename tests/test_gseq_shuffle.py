"""Tests for gseq_shuffle: exact k-mer preserving shuffles (C++ core shared with R misha)."""

import collections
import math
import re
import time

import numpy as np
import pandas as pd
import pytest

import pymisha as pm


@pytest.fixture(autouse=True)
def _ensure_db(_init_db):
    pass


def kmer_counts(s, k):
    s = s.upper()
    return collections.Counter(s[i : i + k] for i in range(len(s) - k + 1))


def enumerate_shuffles(s, k):
    """Every distinct sequence with the k-mer counts and the first k-1 bases of s
    (A/C/G/T only), by depth-first search over the k-mers left to place."""
    n = len(s)
    if k == 1:
        out = set()

        def perm(prefix, left):
            if not left:
                out.add(prefix)
                return
            for c in sorted(left):
                if left[c]:
                    left[c] -= 1
                    if left[c] == 0:
                        del left[c]
                    perm(prefix + c, left)
                    left[c] = left.get(c, 0) + 1

        perm("", collections.Counter(s))
        return sorted(out)

    left = kmer_counts(s, k)
    out = set()

    def rec(cur):
        if len(cur) == n:
            out.add(cur)
            return
        w = cur[len(cur) - k + 1 :]
        for km in list(left):
            if left[km] and km[:-1] == w:
                left[km] -= 1
                rec(cur + km[-1])
                left[km] += 1

    rec(s[: k - 1])
    return sorted(out)


def shortcut_shuffle(s, k, n, rng):
    """The DeepLIFT / tangermeme shortcut: every (k-1)-mer keeps the edge it was
    last left through in s and only its other out-edges are permuted. Valid but
    not uniform; written from that description as the negative control."""
    m = len(s) - k + 1
    node = [s[i : i + k - 1] for i in range(m + 1)]
    nexts = collections.defaultdict(list)
    for e in range(m):
        nexts[node[e]].append(e + 1)
    draws = []
    for _ in range(n):
        order = {}
        for v, lst in nexts.items():
            free = list(lst[:-1])
            rng.shuffle(free)
            order[v] = free + [lst[-1]]
        ptr = collections.Counter()
        cur = node[0]
        out = [node[0]]
        for _p in range(m):
            nx = order[cur][ptr[cur]]
            ptr[cur] += 1
            cur = node[nx]
            out.append(cur[-1])
        draws.append("".join(out))
    return draws


def _chi2_sf(x, df):
    """P(X >= x) for X ~ chi-square(df): the regularized upper incomplete gamma Q(df/2, x/2), by
    its series below a + 1 and its continued fraction above (Numerical Recipes gammq). Keeps the
    test free of scipy, which CI does not install."""
    a, x = df / 2.0, x / 2.0
    if x <= 0:
        return 1.0
    lg = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1:
        term = total = 1.0 / a
        n = a
        while abs(term) > abs(total) * 1e-15:
            n += 1
            term *= x / n
            total += term
        return max(0.0, 1.0 - total * math.exp(lg))
    tiny = 1e-300
    b = x + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return math.exp(lg) * h


def uniformity_p(draws, valid):
    cnt = collections.Counter(draws)
    obs = np.array([cnt.get(v, 0) for v in valid], dtype=float)
    expected = obs.sum() / len(obs)
    stat = float(((obs - expected) ** 2 / expected).sum())
    return _chi2_sf(stat, len(obs) - 1)


def random_dna(rng, n, alphabet="ACGT"):
    return "".join(rng.choice(list(alphabet), size=n))


# Golden values shared with R misha's tests/testthat/test-gseq-shuffle.R: the core
# and its generator are the same C++, so both packages must give these bytes.
GOLDEN_SEQS = ["ACGTTGCAACGTAGCTAGCTTTACG", "ccgNNacgtacgtTTGACGatcg", "AAAAAAAAAA", "ACGT"]
GOLDEN_K2_SEED42 = ["AACACTACGTTTAGTTAGCTGCGCG", "ccgNNacgattcgTGACGTtacg", "AAAAAAAAAA", "ACGT"]
GOLDEN_K3_SEEDM7 = ["ACGTAGCTTGCTAGCAACGTTTACG", "ACGTAGCAACGTTGCTTTAGCTACG", "ACGTAGCTTGCTACGTTTAGCAACG"]
GOLDEN_K1_SEED2P40 = "ACTTTTCCAAGGACTTAGGAGCCTG"


class TestCounts:
    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
    def test_exact_kmer_counts_random(self, k):
        rng = np.random.default_rng(17 + k)
        seqs = [random_dna(rng, n) for n in (2, 5, 10, 37, 100, 500, 2000)]
        seqs += [random_dna(rng, 300, "AAACGT"), random_dna(rng, 400, "AT"), "ACGTACGTACGTACGT", "CG" * 50]
        sh = pm.gseq_shuffle(seqs, k=k, n=3, seed=k)
        assert len(sh) == len(seqs)
        for s, row in zip(seqs, sh, strict=True):
            assert len(row) == 3
            for x in row:
                assert len(x) == len(s)
                assert kmer_counts(x, k) == kmer_counts(s, k)
                if k > 1 and len(s) >= k:
                    assert x[: k - 1] == s[: k - 1]
                    assert x[len(x) - k + 1 :] == s[len(s) - k + 1 :]
        assert pm.gseq_shuffle(seqs[5], seed=1) != seqs[5]

    def test_exact_kmer_counts_real_sequence(self):
        intervs = pm.gintervals(["1", "1", "2"], [10000, 50000, 20000], [12000, 55000, 21000])
        seqs = pm.gseq_extract(intervs)
        for k in range(1, 7):
            sh = pm.gseq_shuffle(seqs, k=k, seed=42)
            for s, x in zip(seqs, sh, strict=True):
                assert kmer_counts(x, k) == kmer_counts(s, k)
                assert x != s
        # an intervals DataFrame is read with gseq_extract
        assert pm.gseq_shuffle(intervs, k=3, seed=5) == pm.gseq_shuffle(seqs, k=3, seed=5)


class TestUniformity:
    def test_uniform_k2(self):
        s = "ACAGTCATGACTG"
        valid = enumerate_shuffles(s, 2)
        assert len(valid) > 20
        draws = pm.gseq_shuffle(s, k=2, n=30000, seed=2024)
        assert set(draws) == set(valid)
        assert uniformity_p(draws, valid) > 1e-3

    def test_uniform_k1_k3(self):
        s1 = "AACCGT"
        valid1 = enumerate_shuffles(s1, 1)
        assert len(valid1) == 180
        draws1 = pm.gseq_shuffle(s1, k=1, n=36000, seed=3)
        assert set(draws1) == set(valid1)
        assert uniformity_p(draws1, valid1) > 1e-3

        s3 = "AACAACATAACATCATAA"
        valid3 = enumerate_shuffles(s3, 3)
        assert len(valid3) > 5
        draws3 = pm.gseq_shuffle(s3, k=3, n=20000, seed=4)
        assert set(draws3) == set(valid3)
        assert uniformity_p(draws3, valid3) > 1e-3

    def test_rejects_fixed_last_edge_shortcut(self):
        s = "ACAGTCATGACTG"
        valid = enumerate_shuffles(s, 2)
        draws = shortcut_shuffle(s, 2, 5000, np.random.default_rng(99))
        assert set(draws) <= set(valid)
        assert len(set(draws)) < len(valid)
        assert uniformity_p(draws, valid) < 1e-10


class TestSeeds:
    SEQS = ["ACGTTGCAACGTAGCTAGCTTTACG", "GGGCCCATATCGCGATTA"]

    def test_reproducible(self):
        a = pm.gseq_shuffle(self.SEQS, seed=11)
        assert pm.gseq_shuffle(self.SEQS, seed=11) == a
        assert pm.gseq_shuffle(self.SEQS, seed=12) != a
        # numpy's global generator, when no seed is given
        np.random.seed(5)
        b1 = pm.gseq_shuffle(self.SEQS, n=4)
        np.random.seed(5)
        b2 = pm.gseq_shuffle(self.SEQS, n=4)
        assert b1 == b2
        # a Generator
        assert pm.gseq_shuffle(self.SEQS, seed=np.random.default_rng(3)) == pm.gseq_shuffle(
            self.SEQS, seed=np.random.default_rng(3)
        )
        # a numpy integer is an integer
        assert pm.gseq_shuffle(self.SEQS, seed=np.int64(11)) == a

    def test_streams(self):
        a = pm.gseq_shuffle(self.SEQS, seed=11)
        assert pm.gseq_shuffle(self.SEQS[0], seed=11) == a[0]
        m = pm.gseq_shuffle(self.SEQS, n=5, seed=11)
        assert [row[0] for row in m] == a
        assert isinstance(pm.gseq_shuffle(self.SEQS, seed=-3), list)
        assert isinstance(pm.gseq_shuffle(self.SEQS, seed=2**53), list)

    def test_same_as_r_misha(self):
        assert pm.gseq_shuffle(GOLDEN_SEQS, k=2, seed=42) == GOLDEN_K2_SEED42
        assert pm.gseq_shuffle(GOLDEN_SEQS[0], k=3, n=3, seed=-7) == GOLDEN_K3_SEEDM7
        assert pm.gseq_shuffle(GOLDEN_SEQS[0], k=1, seed=2**40) == GOLDEN_K1_SEED2P40


class TestCharacters:
    def test_n_and_other_characters_stay(self):
        rng = np.random.default_rng(3)
        s = (
            random_dna(rng, 40)
            + "NNNN"
            + random_dna(rng, 30)
            + "N"
            + random_dna(rng, 3)
            + "R"
            + random_dna(rng, 50)
            + "-"
            + random_dna(rng, 1)
            + "N"
        )
        fixed = [i for i, c in enumerate(s) if c not in "ACGT"]
        sh = pm.gseq_shuffle(s, k=2, n=50, seed=8)
        for x in sh:
            assert [x[i] for i in fixed] == [s[i] for i in fixed]
            assert kmer_counts(x, 2) == kmer_counts(s, 2)
            for a, b in zip(re.split("[^ACGT]+", x), re.split("[^ACGT]+", s), strict=True):
                assert kmer_counts(a, 2) == kmer_counts(b, 2)
        assert len(set(sh)) > 40
        assert all(x[4:6] == "NN" for x in pm.gseq_shuffle("ACGTNNACGT", k=1, n=20, seed=1))

    def test_case(self):
        s = "ACGTacgtAAccGGttACGTAcgTacGT"
        for x in pm.gseq_shuffle(s, k=2, n=30, seed=1):
            assert [c.islower() for c in x] == [c.islower() for c in s]
            assert kmer_counts(x, 2) == kmer_counts(s, 2)
        assert pm.gseq_shuffle(s, k=3, seed=9).upper() == pm.gseq_shuffle(s.upper(), k=3, seed=9)

    def test_non_ascii_left_in_place(self):
        s = "ACGTé" + "ACGTTGCA" * 3 + "ü"
        x = pm.gseq_shuffle(s, seed=1)
        assert x[4] == "é" and x[-1] == "ü"
        assert kmer_counts(x, 2) == kmer_counts(s, 2)
        s2 = "ACGT☃" + "ACGTTGCA" * 3  # a UTF-8 str
        x2 = pm.gseq_shuffle(s2, seed=1)
        assert x2[4] == "☃" and kmer_counts(x2, 2) == kmer_counts(s2, 2)


class TestEdgeCases:
    def test_empty_and_na(self):
        assert pm.gseq_shuffle([], seed=1) == []
        assert pm.gseq_shuffle([], n=3, seed=1) == []
        assert pm.gseq_shuffle("", seed=1) == ""
        assert pm.gseq_shuffle([None], seed=1) == [None]
        assert pm.gseq_shuffle(["ACGT", None], n=2, seed=1)[1] == [None, None]
        assert pm.gseq_shuffle([float("nan")], seed=1) == [None]
        assert pm.gseq_shuffle(pd.Series(["ACGTAC", pd.NA]), seed=1)[1] is None

    def test_no_other_arrangement(self):
        assert pm.gseq_shuffle(["A", "AC", "ACG"], k=3, seed=1) == ["A", "AC", "ACG"]
        assert pm.gseq_shuffle("ACGTACGTAC", k=10, seed=1) == "ACGTACGTAC"
        assert pm.gseq_shuffle("ACGTACGTAC", k=32, seed=1) == "ACGTACGTAC"
        assert pm.gseq_shuffle("AAAAAAAAAAAA", n=3, seed=1) == ["AAAAAAAAAAAA"] * 3
        assert pm.gseq_shuffle("ACGTTGCA", k=4, seed=1) == "ACGTTGCA"
        assert pm.gseq_shuffle("NNNNN", seed=1) == "NNNNN"

    def test_errors(self):
        with pytest.raises(ValueError, match="k must be"):
            pm.gseq_shuffle("ACGT", k=0)
        with pytest.raises(ValueError, match="k must be"):
            pm.gseq_shuffle("ACGT", k=33)
        with pytest.raises(TypeError, match="k must be"):
            pm.gseq_shuffle("ACGT", k=2.5)
        with pytest.raises(ValueError, match="n must be"):
            pm.gseq_shuffle("ACGT", n=0)
        with pytest.raises(TypeError, match="seed must be"):
            pm.gseq_shuffle("ACGT", seed=1.5)
        with pytest.raises(ValueError, match="seed must be"):
            pm.gseq_shuffle("ACGT", seed=2**54)
        with pytest.raises(TypeError, match="seqs must be"):
            pm.gseq_shuffle([1, 2, 3])

    def test_long_sequence(self):
        s = random_dna(np.random.default_rng(2), 200_000)
        t0 = time.monotonic()
        x = pm.gseq_shuffle(s, k=6, seed=1)
        assert time.monotonic() - t0 < 30
        assert kmer_counts(x, 6) == kmer_counts(s, 6)
