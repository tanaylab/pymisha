"""Tests for gtrack_create_pwm_energy and _load_pssm_from_db."""

import contextlib
import shutil
from pathlib import Path

import numpy as np
import pytest

import pymisha as pm
from pymisha.tracks import _load_pssm_from_db

TEST_DB = Path(__file__).resolve().parent / "testdb" / "trackdb" / "test"


# ---------------------------------------------------------------------------
# _load_pssm_from_db tests
# ---------------------------------------------------------------------------


class TestLoadPssmFromDb:
    def test_load_pssm_0(self):
        """Load PSSM id 0 — ATTAAT motif with flanking uniform."""
        pssm = _load_pssm_from_db("pssm", 0)
        assert pssm.shape == (32, 4)
        # Position 13 should be A-enriched (0.785)
        np.testing.assert_allclose(pssm[13], [0.785, 0.071, 0.071, 0.071])
        # Flanking positions should be uniform
        np.testing.assert_allclose(pssm[0], [0.25, 0.25, 0.25, 0.25])

    def test_load_pssm_1(self):
        """Load PSSM id 1 — longer motif."""
        pssm = _load_pssm_from_db("pssm", 1)
        assert pssm.shape == (33, 4)
        # Position 14: strong C signal (0.921)
        assert pssm[14, 1] == pytest.approx(0.921, abs=0.001)

    def test_load_pssm_invalid_id(self):
        """Non-existent PSSM id should raise."""
        with pytest.raises(ValueError, match="PSSM id 999"):
            _load_pssm_from_db("pssm", 999)

    def test_load_pssm_invalid_set(self):
        """Non-existent PSSM set should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _load_pssm_from_db("nonexistent_set", 0)

    def test_all_pssm_ids(self):
        """All 8 PSSMs (ids 0-7) should load without error."""
        for pid in range(8):
            pssm = _load_pssm_from_db("pssm", pid)
            assert pssm.ndim == 2
            assert pssm.shape[1] == 4
            # Rows should sum to ~1.0
            row_sums = pssm.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# gtrack_create_pwm_energy tests
# ---------------------------------------------------------------------------


class TestGtrackCreatePwmEnergy:
    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        for name in ["test_pwm_e", "test_pwm_e2", "test_pwm_e3"]:
            with contextlib.suppress(Exception):
                pm.gtrack_rm(name, force=True)
        pm._pymisha.pm_dbreload()

    def test_basic_creation(self):
        """Create a PWM energy track from PSSM id 0."""
        pm.gtrack_create_pwm_energy(
            "test_pwm_e", "Test PWM track", "pssm", 0, 0.01, 50
        )
        assert pm.gtrack_exists("test_pwm_e")
        info = pm.gtrack_info("test_pwm_e")
        assert info["type"] == "dense"
        assert info["bin_size"] == 50

    def test_values_are_finite(self):
        """PWM energy values should be finite (not all NaN)."""
        pm.gtrack_create_pwm_energy(
            "test_pwm_e", "Test", "pssm", 0, 0.01, 100
        )
        intervals = pm.gintervals(["1"], [0], [10000])
        vals = pm.gextract("test_pwm_e", intervals=intervals, iterator=100)
        data = vals["test_pwm_e"].values
        assert np.isfinite(data).sum() > 0

    def test_values_are_negative(self):
        """LSE PWM energy scores should be negative (log-probabilities)."""
        pm.gtrack_create_pwm_energy(
            "test_pwm_e", "Test", "pssm", 0, 0.01, 100
        )
        intervals = pm.gintervals(["1"], [0], [10000])
        vals = pm.gextract("test_pwm_e", intervals=intervals, iterator=100)
        data = vals["test_pwm_e"].values
        finite = data[np.isfinite(data)]
        assert len(finite) > 0
        assert (finite < 0).all(), "LSE scores should be negative"

    def test_different_pssm_ids(self):
        """Different PSSM ids should produce different track values."""
        pm.gtrack_create_pwm_energy(
            "test_pwm_e", "Test0", "pssm", 0, 0.01, 200
        )
        pm.gtrack_create_pwm_energy(
            "test_pwm_e2", "Test3", "pssm", 3, 0.01, 200
        )

        intervals = pm.gintervals(["1"], [0], [10000])
        v0 = pm.gextract("test_pwm_e", intervals=intervals, iterator=200)
        v3 = pm.gextract("test_pwm_e2", intervals=intervals, iterator=200)

        data0 = v0["test_pwm_e"].values
        data3 = v3["test_pwm_e2"].values
        # Should not be identical
        assert not np.allclose(data0, data3, equal_nan=True)

    def test_prior_affects_values(self):
        """Different prior values should produce different scores."""
        pm.gtrack_create_pwm_energy(
            "test_pwm_e", "Test prior=0.01", "pssm", 0, 0.01, 200
        )
        pm.gtrack_create_pwm_energy(
            "test_pwm_e2", "Test prior=0.5", "pssm", 0, 0.5, 200
        )

        intervals = pm.gintervals(["1"], [0], [10000])
        v1 = pm.gextract("test_pwm_e", intervals=intervals, iterator=200)
        v2 = pm.gextract("test_pwm_e2", intervals=intervals, iterator=200)

        d1 = v1["test_pwm_e"].values
        d2 = v2["test_pwm_e2"].values
        assert not np.allclose(d1, d2, equal_nan=True)

    def test_attributes_set(self):
        """Track should have correct attributes after creation."""
        pm.gtrack_create_pwm_energy(
            "test_pwm_e", "My PWM track", "pssm", 0, 0.01, 50
        )
        attrs = pm.gtrack_info("test_pwm_e").get("attributes", {})
        assert "created.by" in attrs
        assert "pssm" in attrs["created.by"]
        assert attrs.get("description") == "My PWM track"

    def test_duplicate_track_raises(self):
        """Creating a track that already exists should raise."""
        pm.gtrack_create_pwm_energy(
            "test_pwm_e", "First", "pssm", 0, 0.01, 100
        )
        with pytest.raises(ValueError, match="already exists"):
            pm.gtrack_create_pwm_energy(
                "test_pwm_e", "Second", "pssm", 0, 0.01, 100
            )

    def test_invalid_pssmset_raises(self):
        """Non-existent PSSM set should raise."""
        with pytest.raises(FileNotFoundError):
            pm.gtrack_create_pwm_energy(
                "test_pwm_e", "Test", "nonexistent", 0, 0.01, 100
            )

    def test_invalid_pssmid_raises(self):
        """Non-existent PSSM id should raise."""
        with pytest.raises(ValueError, match="PSSM id"):
            pm.gtrack_create_pwm_energy(
                "test_pwm_e", "Test", "pssm", 999, 0.01, 100
            )

    def test_none_params_raise(self):
        """All parameters are required."""
        with pytest.raises(ValueError):
            pm.gtrack_create_pwm_energy(
                None, "Test", "pssm", 0, 0.01, 100
            )
        with pytest.raises(ValueError):
            pm.gtrack_create_pwm_energy(
                "test_pwm_e", "Test", "pssm", 0, None, 100
            )

    def test_invalid_iterator_raises(self):
        """Non-positive iterator should raise."""
        with pytest.raises(ValueError, match="positive"):
            pm.gtrack_create_pwm_energy(
                "test_pwm_e", "Test", "pssm", 0, 0.01, -1
            )
        with pytest.raises(ValueError, match="positive"):
            pm.gtrack_create_pwm_energy(
                "test_pwm_e", "Test", "pssm", 0, 0.01, 0
            )

    def test_consistency_with_vtrack_extract(self):
        """Track values should match direct PWM vtrack extraction."""
        pssm = _load_pssm_from_db("pssm", 1)
        prior = 0.01
        iterator = 200

        # Create track via gtrack_create_pwm_energy
        pm.gtrack_create_pwm_energy(
            "test_pwm_e", "Consistency test", "pssm", 1, prior, iterator
        )

        # Direct vtrack extraction
        pm.gvtrack_create("_test_pwm_vt", None, func="pwm", pssm=pssm, prior=prior)
        try:
            intervals = pm.gintervals(["1"], [0], [50000])
            track_vals = pm.gextract("test_pwm_e", intervals=intervals, iterator=iterator)
            vt_vals = pm.gextract("_test_pwm_vt", intervals=intervals, iterator=iterator)

            np.testing.assert_allclose(
                track_vals["test_pwm_e"].values,
                vt_vals["_test_pwm_vt"].values,
                rtol=1e-5,
            )
        finally:
            pm.gvtrack_rm("_test_pwm_vt")


# ---------------------------------------------------------------------------
# PWM energy track on indexed databases
# (ported from R test-pwm-indexed-gtrack-create.R)
# ---------------------------------------------------------------------------


def _create_test_pssm():
    """Create a simple 2-position PSSM matching 'AC' exactly."""
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # Only A
            [0.0, 1.0, 0.0, 0.0],  # Only C
        ]
    )


class TestGtrackCreatePwmEnergyIndexed:
    """PWM energy track creation on indexed databases.

    Ported from R test-pwm-indexed-gtrack-create.R. This is a regression test
    for a bug where gtrack.create would fail with "Invalid format of intervals
    argument" when creating tracks from PWM virtual tracks on indexed databases
    with many chromosomes.

    In pymisha, gtrack_create accepts only numeric iterators (not interval-
    based iterators), so we test the equivalent code path via
    gtrack_create_pwm_energy and gtrack_create with numeric iterator on both
    indexed and regular databases.
    """

    @pytest.fixture
    def indexed_db(self, tmp_path):
        """Create an indexed database with longer chromosomes for PWM testing."""
        # Use longer chromosomes (500bp) so PSSMs from the DB (32-33 positions) fit
        import random
        rng = random.Random(42)
        bases = "ACGT"
        chr1_seq = "".join(rng.choice(bases) for _ in range(500))
        chr2_seq = "".join(rng.choice(bases) for _ in range(300))
        chr3_seq = "".join(rng.choice(bases) for _ in range(200))

        fasta_path = tmp_path / "test.fasta"
        fasta_path.write_text(
            f">chr1\n{chr1_seq}\n"
            f">chr2\n{chr2_seq}\n"
            f">chr3\n{chr3_seq}\n"
        )
        db_path = tmp_path / "testdb_indexed"
        pm.gdb_create(str(db_path), str(fasta_path), db_format="indexed")

        # Copy PSSM files from test DB to new indexed DB
        pssm_src = TEST_DB / "pssms"
        pssm_dst = db_path / "pssms"
        if pssm_src.exists():
            if pssm_dst.exists():
                shutil.rmtree(str(pssm_dst))
            shutil.copytree(str(pssm_src), str(pssm_dst))

        yield str(db_path)
        # Restore original test DB after test
        pm.gdb_init(str(TEST_DB))

    def test_pwm_energy_track_on_indexed_db(self, indexed_db):
        """gtrack_create_pwm_energy works on indexed databases."""
        pm.gdb_init(indexed_db)

        # Create a PWM energy track -- should NOT fail on indexed DB
        pm.gtrack_create_pwm_energy(
            "test_pwm_idx", "PWM on indexed DB", "pssm", 0, 0.01, 50
        )

        assert pm.gtrack_exists("test_pwm_idx")
        info = pm.gtrack_info("test_pwm_idx")
        assert info["type"] == "dense"

        # Verify we can extract from it
        intervals = pm.gintervals(["chr1"], [0], [500])
        result = pm.gextract("test_pwm_idx", intervals=intervals, iterator=50)
        assert len(result) > 0
        assert "test_pwm_idx" in result.columns
        assert result["test_pwm_idx"].dtype in [np.float64, np.float32]

        # Track values should be finite log-probabilities
        data = result["test_pwm_idx"].values
        finite = data[np.isfinite(data)]
        assert len(finite) > 0
        assert (finite < 0).all()

    def test_pwm_energy_round_trip_on_indexed_db(self, indexed_db):
        """PWM energy track round-trip: create then extract on indexed DB."""
        pm.gdb_init(indexed_db)

        # Create via gtrack_create_pwm_energy and verify round-trip
        pm.gtrack_create_pwm_energy(
            "test_pwm_idx_rt", "Round-trip test", "pssm", 0, 0.01, 50
        )

        assert pm.gtrack_exists("test_pwm_idx_rt")

        # Extract and verify values match a fresh vtrack computation
        pssm = _load_pssm_from_db("pssm", 0)
        pm.gvtrack_create("_vt_check", None, func="pwm", pssm=pssm, prior=0.01)
        try:
            intervals = pm.gintervals(["chr1"], [0], [500])
            track_vals = pm.gextract("test_pwm_idx_rt", intervals=intervals, iterator=50)
            vt_vals = pm.gextract("_vt_check", intervals=intervals, iterator=50)

            np.testing.assert_allclose(
                track_vals["test_pwm_idx_rt"].values,
                vt_vals["_vt_check"].values,
                rtol=1e-5,
            )
        finally:
            pm.gvtrack_rm("_vt_check")
            with contextlib.suppress(Exception):
                pm.gtrack_rm("test_pwm_idx_rt", force=True)

    def test_pwm_energy_track_on_regular_db(self):
        """gtrack_create_pwm_energy works on regular (non-indexed) databases.

        Verifies that the indexed DB support does not break regular databases.
        """
        try:
            pm.gtrack_create_pwm_energy(
                "test_tracks_pwm_regular",
                "PWM track on regular DB",
                "pssm", 0, 0.01, 50,
            )

            assert pm.gtrack_exists("test_tracks_pwm_regular")

            result = pm.gextract(
                "test_tracks_pwm_regular",
                pm.gintervals(["1"], [200], [10000]),
                iterator=50,
            )
            assert len(result) > 0
            data = result["test_tracks_pwm_regular"].values
            finite = data[np.isfinite(data)]
            assert len(finite) > 0
            assert (finite < 0).all()
        finally:
            with contextlib.suppress(Exception):
                pm.gtrack_rm("test_tracks_pwm_regular", force=True)
            pm._pymisha.pm_dbreload()


# ---------------------------------------------------------------------------
# Motifs / gtrack_create_pwm_energy regression test
# (ported from R test-motifs.R -- adapted for pymisha test DB PSSM set)
# ---------------------------------------------------------------------------


class TestGtrackCreatePwmEnergyMotifs:
    """Port of R test-motifs.R gtrack.create_pwm_energy regression test.

    The R test uses the "misha_motifs" PSSM set which does not exist in the
    pymisha test database. We adapt the test to use the available "pssm" set
    (ids 0-7), testing the same code path: create energy track from DB PSSM,
    extract, verify non-trivial values.
    """

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        for name in ["test_motifs_e"]:
            with contextlib.suppress(Exception):
                pm.gtrack_rm(name, force=True)
        pm._pymisha.pm_dbreload()

    def test_create_pwm_energy_from_db_pssm(self):
        """gtrack_create_pwm_energy with DB PSSM set creates extractable track."""
        pm.gtrack_create_pwm_energy(
            "test_motifs_e", "Motifs regression test", "pssm", 0, 0.02, 50
        )

        assert pm.gtrack_exists("test_motifs_e")

        intervals = pm.gintervals(["1"], [0], [10000])
        result = pm.gextract("test_motifs_e", intervals=intervals, iterator=50)
        data = result["test_motifs_e"].values
        finite = data[np.isfinite(data)]

        # Should have non-trivial finite values (log-probabilities, negative)
        assert len(finite) > 0
        assert (finite < 0).all(), "LSE scores should be negative"

    def test_create_pwm_energy_multiple_pssm_ids(self):
        """gtrack_create_pwm_energy works for multiple PSSM ids from same set."""
        # This verifies the motif loading and track creation pipeline
        # works across different motifs in the DB
        track_names = []
        try:
            for pid in [0, 3, 7]:
                name = f"test_motifs_e_{pid}"
                track_names.append(name)
                pm.gtrack_create_pwm_energy(
                    name, f"Motif {pid}", "pssm", pid, 0.02, 100
                )
                assert pm.gtrack_exists(name)
                intervals = pm.gintervals(["1"], [0], [5000])
                result = pm.gextract(name, intervals=intervals, iterator=100)
                data = result[name].values
                assert np.isfinite(data).sum() > 0
        finally:
            for name in track_names:
                with contextlib.suppress(Exception):
                    pm.gtrack_rm(name, force=True)
            pm._pymisha.pm_dbreload()
