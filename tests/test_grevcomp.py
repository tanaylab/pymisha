"""Tests for grevcomp standalone reverse complement function."""

import pymisha as pm


class TestGrevcomp:
    """Tests for pm.grevcomp()."""

    def test_basic_reverse_complement(self):
        """Basic DNA reverse complement."""
        assert pm.grevcomp("ACTG") == "CAGT"
        assert pm.grevcomp("GCGC") == "GCGC"  # palindromic
        assert pm.grevcomp("AAAA") == "TTTT"
        assert pm.grevcomp("CCCC") == "GGGG"

    def test_preserves_case(self):
        """Lowercase bases should stay lowercase after complement."""
        assert pm.grevcomp("AcTg") == "cAgT"
        assert pm.grevcomp("actg") == "cagt"
        assert pm.grevcomp("ACTG") == "CAGT"

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert pm.grevcomp("") == ""

    def test_single_base(self):
        """Single bases reverse-complement to their complement."""
        assert pm.grevcomp("A") == "T"
        assert pm.grevcomp("C") == "G"
        assert pm.grevcomp("G") == "C"
        assert pm.grevcomp("T") == "A"
        assert pm.grevcomp("a") == "t"

    def test_n_handling(self):
        """N bases should complement to N."""
        assert pm.grevcomp("N") == "N"
        assert pm.grevcomp("n") == "n"
        assert pm.grevcomp("ANG") == "CNT"
        assert pm.grevcomp("AnG") == "CnT"

    def test_list_input(self):
        """List of sequences should return list of reverse complements."""
        result = pm.grevcomp(["ACTG", "GGCC"])
        assert result == ["CAGT", "GGCC"]

    def test_list_with_empty(self):
        """List containing empty strings."""
        result = pm.grevcomp(["ACTG", "", "GCTA"])
        assert result == ["CAGT", "", "TAGC"]

    def test_double_application_is_identity(self):
        """Applying grevcomp twice should return the original."""
        seqs = ["ACTG", "GCTA", "AAAAAA", "GCGCGC", "AcTg"]
        for s in seqs:
            assert pm.grevcomp(pm.grevcomp(s)) == s

    def test_matches_gseq_revcomp(self):
        """grevcomp should produce identical results to gseq_revcomp."""
        cases = ["ACTG", "actg", "AcTg", "", "AANG", "TTTT"]
        for s in cases:
            assert pm.grevcomp(s) == pm.gseq_revcomp(s)

        list_cases = [["ACTG", "GCTA"], ["", "AAA"]]
        for lst in list_cases:
            assert pm.grevcomp(lst) == pm.gseq_revcomp(lst)

    def test_long_sequence(self):
        """Long repeated sequence."""
        long_seq = "ACTG" * 1000
        expected = "CAGT" * 1000
        assert pm.grevcomp(long_seq) == expected
