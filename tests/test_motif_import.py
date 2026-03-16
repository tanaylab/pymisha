"""Tests for motif import functions: gseq_read_meme, gseq_read_jaspar, gseq_read_homer."""

import os
import tempfile
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import pymisha as pm

FIXTURES = Path(__file__).resolve().parent / "test_data" / "motifs"


# ---------------------------------------------------------------------------
# MEME format tests
# ---------------------------------------------------------------------------

class TestReadMeme:
    def test_returns_dict(self):
        motifs = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
        assert isinstance(motifs, dict)

    def test_correct_number_of_motifs(self):
        motifs = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
        assert len(motifs) == 3

    def test_correct_ids(self):
        motifs = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
        assert list(motifs.keys()) == ["MA0001.1", "MA0002.1", "MA0003.2"]

    def test_dataframe_columns(self):
        motifs = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
        for mid, df in motifs.items():
            assert isinstance(df, pd.DataFrame), f"motif {mid} is not a DataFrame"
            assert list(df.columns) == ["A", "C", "G", "T"]

    def test_matrix_values_motif1(self):
        motifs = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
        m = motifs["MA0001.1"]
        assert m.shape == (4, 4)
        npt.assert_allclose(m.iloc[0].values, [0.250, 0.500, 0.125, 0.125])
        npt.assert_allclose(m.iloc[1].values, [0.875, 0.000, 0.125, 0.000])
        npt.assert_allclose(m.iloc[2].values, [0.000, 0.000, 1.000, 0.000])
        npt.assert_allclose(m.iloc[3].values, [0.100, 0.100, 0.100, 0.700])

    def test_matrix_values_motif2(self):
        motifs = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
        m = motifs["MA0002.1"]
        assert m.shape == (6, 4)
        npt.assert_allclose(m.iloc[0].values, [0.200, 0.200, 0.200, 0.400])
        npt.assert_allclose(m.iloc[4].values, [0.000, 0.000, 0.000, 1.000])

    def test_rows_sum_to_one(self):
        motifs = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
        for mid, df in motifs.items():
            row_sums = df.sum(axis=1)
            npt.assert_allclose(row_sums.values, 1.0, atol=1e-6,
                                err_msg=f"Row sums not 1.0 for motif {mid}")

    def test_attrs_name(self):
        motifs = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
        assert motifs["MA0001.1"].attrs["name"] == "AGL3"
        assert motifs["MA0002.1"].attrs["name"] == "RUNX1"
        assert motifs["MA0003.2"].attrs["name"] == "TFAP2A"

    def test_attrs_w(self):
        motifs = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
        assert motifs["MA0001.1"].attrs["w"] == 4
        assert motifs["MA0002.1"].attrs["w"] == 6
        assert motifs["MA0003.2"].attrs["w"] == 5

    def test_attrs_nsites(self):
        motifs = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
        assert motifs["MA0001.1"].attrs["nsites"] == 20
        assert motifs["MA0002.1"].attrs["nsites"] == 15
        assert motifs["MA0003.2"].attrs["nsites"] == 18

    def test_attrs_E(self):
        motifs = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
        npt.assert_allclose(motifs["MA0001.1"].attrs["E"], 1.2e-005)
        npt.assert_allclose(motifs["MA0002.1"].attrs["E"], 5.3e-004)
        assert motifs["MA0003.2"].attrs["E"] is None

    def test_attrs_url(self):
        motifs = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
        assert motifs["MA0001.1"].attrs["url"] == "http://jaspar.genereg.net/matrix/MA0001.1"
        assert motifs["MA0002.1"].attrs["url"] is None

    def test_attrs_strand(self):
        motifs = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
        assert motifs["MA0001.1"].attrs["strand"] == "+ -"

    def test_attrs_background(self):
        motifs = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
        bg = motifs["MA0001.1"].attrs["background"]
        assert isinstance(bg, dict)
        npt.assert_allclose(bg["A"], 0.29)
        npt.assert_allclose(bg["C"], 0.21)
        npt.assert_allclose(bg["G"], 0.21)
        npt.assert_allclose(bg["T"], 0.29)

    def test_attrs_alength(self):
        motifs = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
        assert motifs["MA0001.1"].attrs["alength"] == 4

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            pm.gseq_read_meme("/nonexistent/path/motifs.meme")

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".meme", delete=False) as f:
            f.write("")
            tmpfile = f.name
        try:
            with pytest.raises(ValueError, match="No motifs found"):
                pm.gseq_read_meme(tmpfile)
        finally:
            os.unlink(tmpfile)

    def test_log_odds_rejected(self):
        content = """\
MEME version 5

MOTIF test1
letter-probability matrix: alength= 4 w= 2
 -0.5  0.3  0.1  0.1
  0.2  0.3  0.3  0.2
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".meme", delete=False) as f:
            f.write(content)
            tmpfile = f.name
        try:
            with pytest.raises(ValueError, match="Log-odds"):
                pm.gseq_read_meme(tmpfile)
        finally:
            os.unlink(tmpfile)

    def test_renormalization_warning(self):
        content = """\
MEME version 5

MOTIF test_unnorm
letter-probability matrix: alength= 4 w= 2
 0.3  0.3  0.3  0.3
 0.25  0.25  0.25  0.25
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".meme", delete=False) as f:
            f.write(content)
            tmpfile = f.name
        try:
            with pytest.warns(UserWarning, match="re-normalizing"):
                motifs = pm.gseq_read_meme(tmpfile)
            npt.assert_allclose(motifs["test_unnorm"].iloc[0].sum(), 1.0)
        finally:
            os.unlink(tmpfile)

    def test_duplicate_ids(self):
        content = """\
MEME version 5

MOTIF DUPE
letter-probability matrix: alength= 4 w= 2
 0.25  0.25  0.25  0.25
 0.25  0.25  0.25  0.25

MOTIF DUPE
letter-probability matrix: alength= 4 w= 2
 0.5  0.5  0.0  0.0
 0.0  0.0  0.5  0.5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".meme", delete=False) as f:
            f.write(content)
            tmpfile = f.name
        try:
            with pytest.warns(UserWarning, match="Duplicate motif IDs"):
                motifs = pm.gseq_read_meme(tmpfile)
            assert "DUPE.1" in motifs
            assert "DUPE.2" in motifs
        finally:
            os.unlink(tmpfile)

    def test_wrong_column_count(self):
        content = """\
MEME version 5

MOTIF bad
letter-probability matrix: alength= 4 w= 1
 0.25  0.25  0.50
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".meme", delete=False) as f:
            f.write(content)
            tmpfile = f.name
        try:
            with pytest.raises(ValueError, match="Expected 4 columns"):
                pm.gseq_read_meme(tmpfile)
        finally:
            os.unlink(tmpfile)

    def test_w_mismatch(self):
        content = """\
MEME version 5

MOTIF mismatch
letter-probability matrix: alength= 4 w= 3
 0.25  0.25  0.25  0.25
 0.25  0.25  0.25  0.25
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".meme", delete=False) as f:
            f.write(content)
            tmpfile = f.name
        try:
            with pytest.raises(ValueError, match="Expected 3 rows but found 2"):
                pm.gseq_read_meme(tmpfile)
        finally:
            os.unlink(tmpfile)


# ---------------------------------------------------------------------------
# JASPAR format tests (header format)
# ---------------------------------------------------------------------------

class TestReadJasparHeader:
    def test_returns_dict(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs.jaspar"))
        assert isinstance(motifs, dict)

    def test_correct_number_of_motifs(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs.jaspar"))
        assert len(motifs) == 2

    def test_correct_ids(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs.jaspar"))
        assert list(motifs.keys()) == ["MA0002.1", "MA0004.1"]

    def test_dataframe_columns(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs.jaspar"))
        for mid, df in motifs.items():
            assert list(df.columns) == ["A", "C", "G", "T"]

    def test_count_to_probability_conversion(self):
        """Test that counts are correctly converted to probabilities."""
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs.jaspar"))
        m = motifs["MA0002.1"]
        # First column: A=10, C=0, G=3, T=2, total=15
        npt.assert_allclose(m.iloc[0]["A"], 10 / 15)
        npt.assert_allclose(m.iloc[0]["C"], 0 / 15)
        npt.assert_allclose(m.iloc[0]["G"], 3 / 15)
        npt.assert_allclose(m.iloc[0]["T"], 2 / 15)

    def test_rows_sum_to_one(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs.jaspar"))
        for mid, df in motifs.items():
            row_sums = df.sum(axis=1)
            npt.assert_allclose(row_sums.values, 1.0, atol=1e-6,
                                err_msg=f"Row sums not 1.0 for motif {mid}")

    def test_matrix_shape(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs.jaspar"))
        assert motifs["MA0002.1"].shape == (6, 4)
        assert motifs["MA0004.1"].shape == (6, 4)

    def test_attrs_name(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs.jaspar"))
        assert motifs["MA0002.1"].attrs["name"] == "RUNX1"
        assert motifs["MA0004.1"].attrs["name"] == "Arnt"

    def test_attrs_w(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs.jaspar"))
        assert motifs["MA0002.1"].attrs["w"] == 6

    def test_attrs_nsites(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs.jaspar"))
        # nsites = column sum (first column: 10+0+3+2 = 15)
        assert motifs["MA0002.1"].attrs["nsites"] == 15.0

    def test_attrs_format(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs.jaspar"))
        assert motifs["MA0002.1"].attrs["format"] == "jaspar"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            pm.gseq_read_jaspar("/nonexistent/path/motifs.jaspar")

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jaspar", delete=False) as f:
            f.write("")
            tmpfile = f.name
        try:
            with pytest.raises(ValueError, match="No motifs found"):
                pm.gseq_read_jaspar(tmpfile)
        finally:
            os.unlink(tmpfile)

    def test_wrong_row_count(self):
        content = """\
>BAD1 test
A [ 1 2 3 ]
C [ 1 2 3 ]
G [ 1 2 3 ]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jaspar", delete=False) as f:
            f.write(content)
            tmpfile = f.name
        try:
            with pytest.raises(ValueError, match="Expected 4 rows"):
                pm.gseq_read_jaspar(tmpfile)
        finally:
            os.unlink(tmpfile)

    def test_negative_counts(self):
        content = """\
>NEG1 test
A [ -1 2 3 ]
C [ 1 2 3 ]
G [ 1 2 3 ]
T [ 1 2 3 ]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jaspar", delete=False) as f:
            f.write(content)
            tmpfile = f.name
        try:
            with pytest.raises(ValueError, match="Negative count"):
                pm.gseq_read_jaspar(tmpfile)
        finally:
            os.unlink(tmpfile)

    def test_bad_row_label(self):
        content = """\
>BAD1 test
A [ 1 2 3 ]
C [ 1 2 3 ]
X [ 1 2 3 ]
T [ 1 2 3 ]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jaspar", delete=False) as f:
            f.write(content)
            tmpfile = f.name
        try:
            with pytest.raises(ValueError, match="Unexpected row label"):
                pm.gseq_read_jaspar(tmpfile)
        finally:
            os.unlink(tmpfile)


# ---------------------------------------------------------------------------
# JASPAR simple PFM format tests
# ---------------------------------------------------------------------------

class TestReadJasparSimple:
    def test_returns_dict(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs_simple.pfm"))
        assert isinstance(motifs, dict)

    def test_correct_number_of_motifs(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs_simple.pfm"))
        assert len(motifs) == 1

    def test_correct_id_from_filename(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs_simple.pfm"))
        assert "test_motifs_simple" in motifs

    def test_matrix_shape(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs_simple.pfm"))
        m = list(motifs.values())[0]
        assert m.shape == (4, 4)  # 4 positions x 4 bases

    def test_count_to_probability_conversion(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs_simple.pfm"))
        m = list(motifs.values())[0]
        # First column: A=10, C=0, G=3, T=2, total=15
        npt.assert_allclose(m.iloc[0]["A"], 10 / 15)
        npt.assert_allclose(m.iloc[0]["C"], 0 / 15)

    def test_rows_sum_to_one(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs_simple.pfm"))
        for mid, df in motifs.items():
            row_sums = df.sum(axis=1)
            npt.assert_allclose(row_sums.values, 1.0, atol=1e-6)

    def test_attrs_format(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs_simple.pfm"))
        m = list(motifs.values())[0]
        assert m.attrs["format"] == "simple"

    def test_attrs_name_is_none(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs_simple.pfm"))
        m = list(motifs.values())[0]
        assert m.attrs["name"] is None

    def test_attrs_nsites_is_none(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs_simple.pfm"))
        m = list(motifs.values())[0]
        assert m.attrs["nsites"] is None


# ---------------------------------------------------------------------------
# HOMER format tests
# ---------------------------------------------------------------------------

class TestReadHomer:
    def test_returns_dict(self):
        motifs = pm.gseq_read_homer(str(FIXTURES / "test_motifs.homer.motif"))
        assert isinstance(motifs, dict)

    def test_correct_number_of_motifs(self):
        motifs = pm.gseq_read_homer(str(FIXTURES / "test_motifs.homer.motif"))
        assert len(motifs) == 2

    def test_correct_ids(self):
        motifs = pm.gseq_read_homer(str(FIXTURES / "test_motifs.homer.motif"))
        assert "ATGACTCA" in motifs
        assert "GATAAG" in motifs

    def test_dataframe_columns(self):
        motifs = pm.gseq_read_homer(str(FIXTURES / "test_motifs.homer.motif"))
        for mid, df in motifs.items():
            assert list(df.columns) == ["A", "C", "G", "T"]

    def test_matrix_values(self):
        motifs = pm.gseq_read_homer(str(FIXTURES / "test_motifs.homer.motif"))
        m = motifs["ATGACTCA"]
        npt.assert_allclose(m.iloc[0].values, [0.353, 0.143, 0.220, 0.284])
        npt.assert_allclose(m.iloc[1].values, [0.136, 0.178, 0.108, 0.578])

    def test_matrix_shape(self):
        motifs = pm.gseq_read_homer(str(FIXTURES / "test_motifs.homer.motif"))
        assert motifs["ATGACTCA"].shape == (8, 4)
        assert motifs["GATAAG"].shape == (6, 4)

    def test_rows_sum_to_one(self):
        motifs = pm.gseq_read_homer(str(FIXTURES / "test_motifs.homer.motif"))
        for mid, df in motifs.items():
            row_sums = df.sum(axis=1)
            npt.assert_allclose(row_sums.values, 1.0, atol=1e-6,
                                err_msg=f"Row sums not 1.0 for motif {mid}")

    def test_attrs_name(self):
        motifs = pm.gseq_read_homer(str(FIXTURES / "test_motifs.homer.motif"))
        assert motifs["ATGACTCA"].attrs["name"] == "AP1(bZIP)/ThP1-cJun"
        assert motifs["GATAAG"].attrs["name"] == "GATA(Zn)/ThP1-GATA3"

    def test_attrs_consensus(self):
        motifs = pm.gseq_read_homer(str(FIXTURES / "test_motifs.homer.motif"))
        assert motifs["ATGACTCA"].attrs["consensus"] == "ATGACTCA"

    def test_attrs_log_odds_threshold(self):
        motifs = pm.gseq_read_homer(str(FIXTURES / "test_motifs.homer.motif"))
        npt.assert_allclose(
            motifs["ATGACTCA"].attrs["log_odds_threshold"], 8.036341
        )

    def test_attrs_log_p_value(self):
        motifs = pm.gseq_read_homer(str(FIXTURES / "test_motifs.homer.motif"))
        npt.assert_allclose(
            motifs["ATGACTCA"].attrs["log_p_value"], -4130.668834
        )

    def test_attrs_w(self):
        motifs = pm.gseq_read_homer(str(FIXTURES / "test_motifs.homer.motif"))
        assert motifs["ATGACTCA"].attrs["w"] == 8
        assert motifs["GATAAG"].attrs["w"] == 6

    def test_attrs_source(self):
        motifs = pm.gseq_read_homer(str(FIXTURES / "test_motifs.homer.motif"))
        assert motifs["ATGACTCA"].attrs["source"] == "homer"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            pm.gseq_read_homer("/nonexistent/path/motifs.motif")

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".motif", delete=False) as f:
            f.write("")
            tmpfile = f.name
        try:
            with pytest.raises(ValueError, match="No motifs found"):
                pm.gseq_read_homer(tmpfile)
        finally:
            os.unlink(tmpfile)

    def test_negative_values(self):
        content = """\
>ACGT\ttest\t1.0\t-1.0
-0.1\t0.3\t0.4\t0.4
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".motif", delete=False) as f:
            f.write(content)
            tmpfile = f.name
        try:
            with pytest.raises(ValueError, match="Negative probability"):
                pm.gseq_read_homer(tmpfile)
        finally:
            os.unlink(tmpfile)


# ---------------------------------------------------------------------------
# Integration test: DataFrames work with gseq_pwm
# ---------------------------------------------------------------------------

class TestMotifPwmIntegration:
    """Test that parsed motif DataFrames work with gseq_pwm."""

    def test_meme_motif_with_gseq_pwm(self):
        motifs = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
        m = motifs["MA0001.1"]
        scores = pm.gseq_pwm("ACGTACGTACGT", pssm=m, mode="max")
        assert len(scores) == 1
        assert np.isfinite(scores[0])

    def test_jaspar_motif_with_gseq_pwm(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs.jaspar"))
        m = motifs["MA0002.1"]
        scores = pm.gseq_pwm("ACGTACGTACGT", pssm=m, mode="max")
        assert len(scores) == 1
        assert np.isfinite(scores[0])

    def test_homer_motif_with_gseq_pwm(self):
        motifs = pm.gseq_read_homer(str(FIXTURES / "test_motifs.homer.motif"))
        m = motifs["ATGACTCA"]
        scores = pm.gseq_pwm("ATGACTCAATGACTCA", pssm=m, mode="lse")
        assert len(scores) == 1
        assert np.isfinite(scores[0])

    def test_simple_pfm_with_gseq_pwm(self):
        motifs = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs_simple.pfm"))
        m = list(motifs.values())[0]
        scores = pm.gseq_pwm("ACGTACGTACGT", pssm=m, mode="max")
        assert len(scores) == 1
        assert np.isfinite(scores[0])

    def test_meme_motif_with_gseq_pwm_multiple_seqs(self):
        motifs = pm.gseq_read_meme(str(FIXTURES / "test_motifs.meme"))
        m = motifs["MA0001.1"]
        seqs = ["ACGTACGTACGT", "TTTTTTTTTTTTTTT", "GGGGGGGGGGGG"]
        scores = pm.gseq_pwm(seqs, pssm=m, mode="max")
        assert len(scores) == 3
        assert all(np.isfinite(s) for s in scores)


# ---------------------------------------------------------------------------
# Cross-format consistency tests
# ---------------------------------------------------------------------------

class TestCrossFormat:
    """Verify consistency between formats using the same underlying motif."""

    def test_jaspar_simple_matches_header_for_same_motif(self):
        """The simple PFM file has the same counts as the first motif
        in the header JASPAR file (MA0002.1), limited to 4 positions."""
        simple = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs_simple.pfm"))
        header = pm.gseq_read_jaspar(str(FIXTURES / "test_motifs.jaspar"))
        # Simple has the first 4 columns of MA0002.1
        m_simple = list(simple.values())[0]
        m_header = header["MA0002.1"]
        npt.assert_allclose(
            m_simple.values, m_header.iloc[:4].values, atol=1e-10
        )
