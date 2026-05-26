"""Parity port of R misha ``test-motifs.R`` (PWM-energy track creation).

``skip`` -- ``GAP_PWM_ENERGY_SCOPE``:

R limits the creation scope by mutating ``.misha$ALLGENOME`` to the first 1 Mb of
every chromosome, then ``gtrack.create_pwm_energy`` builds the track only there
(baseline: 480000 bins over 24 chroms at iterator=50). pymisha's
``gtrack_create_pwm_energy`` has no scope/intervals argument, so it scans the
*whole* genome (~60M bins at 50 bp) -- which ran >7 min without finishing in a
probe, so it can't be exercised here. Becomes portable once the function accepts
a creation scope (or by replicating its internal PWM vtrack over a limited
scope, given a way to load a PSSM from a named set).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="gtrack_create_pwm_energy has no scope arg; R limits creation to 1 Mb/chrom, "
    "pymisha would scan the whole genome (impractically slow)"
)


def test_gtrack_create_pwm_energy():
    raise AssertionError("unreachable (module skipped)")
