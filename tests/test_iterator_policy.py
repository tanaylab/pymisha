import pandas as pd
import pytest

from pymisha import CartesianGridSpec, FixedRectPolicy, IntervalsPolicy, TrackRectsPolicy
from pymisha._iterator_policy import parse_iterator_policy


def test_intervals_policy_kind():
    pol = IntervalsPolicy()
    assert pol.kind == "intervals"


def test_intervals_policy_is_frozen():
    pol = IntervalsPolicy()
    with pytest.raises(Exception):  # FrozenInstanceError
        pol.kind = "other"


def test_tuple_of_two_ints_is_fixed_rect():
    pol = parse_iterator_policy((100, 200), intervals_is_2d=True)
    assert isinstance(pol, FixedRectPolicy)
    assert pol.width == 100
    assert pol.height == 200


def test_list_of_two_ints_is_fixed_rect():
    pol = parse_iterator_policy([1000, 1000], intervals_is_2d=True)
    assert isinstance(pol, FixedRectPolicy)
    assert pol.width == 1000
    assert pol.height == 1000


def test_fixed_rect_rejects_non_positive():
    with pytest.raises(ValueError, match="positive"):
        parse_iterator_policy((0, 100), intervals_is_2d=True)
    with pytest.raises(ValueError, match="positive"):
        parse_iterator_policy((100, -1), intervals_is_2d=True)


def test_fixed_rect_requires_2d_intervals():
    with pytest.raises(ValueError, match="2D"):
        parse_iterator_policy((100, 100), intervals_is_2d=False)


def test_fixed_rect_rejects_non_int():
    with pytest.raises(ValueError, match="integer"):
        parse_iterator_policy((100.5, 200), intervals_is_2d=True)


def test_fixed_rect_rejects_wrong_arity():
    with pytest.raises(ValueError, match="exactly two"):
        parse_iterator_policy((100,), intervals_is_2d=True)
    with pytest.raises(ValueError, match="exactly two"):
        parse_iterator_policy((100, 200, 300), intervals_is_2d=True)


def test_tuple_of_numpy_ints_is_fixed_rect():
    import numpy as np

    pol = parse_iterator_policy((np.int64(100), np.int64(200)), intervals_is_2d=True)
    assert isinstance(pol, FixedRectPolicy)
    assert pol.width == 100
    assert pol.height == 200


def test_fixed_rect_2d_check_precedes_type_check():
    # If both intervals_is_2d=False AND values are non-int, the 2D-scope
    # error is the more useful one to surface.
    with pytest.raises(ValueError, match="2D"):
        parse_iterator_policy((100.5, 200), intervals_is_2d=False)


def test_string_is_track_rects():
    pol = parse_iterator_policy("my_track", intervals_is_2d=True)
    assert isinstance(pol, TrackRectsPolicy)
    assert pol.track_name == "my_track"


def test_track_rects_requires_2d_intervals():
    with pytest.raises(ValueError, match="2D"):
        parse_iterator_policy("my_track", intervals_is_2d=False)


def test_empty_string_track_name_accepted_by_parser():
    # The parser doesn't validate existence; that's the caller's job.
    pol = parse_iterator_policy("", intervals_is_2d=True)
    assert isinstance(pol, TrackRectsPolicy)
    assert pol.track_name == ""


def _sample_intervals_1d():
    return pd.DataFrame({
        "chrom": ["1", "1"],
        "start": [0, 100_000],
        "end":   [50_000, 150_000],
    })


def test_cartesian_grid_spec_basic():
    spec = CartesianGridSpec(
        intervals1=_sample_intervals_1d(),
        expansion1=[-50_000, -10_000, 10_000, 50_000],
    )
    assert spec.kind == "cartesian_grid"
    assert spec.expansion1 == (-50_000, -10_000, 10_000, 50_000)
    # When intervals2/expansion2 unset, default to intervals1/expansion1:
    assert spec.intervals2 is None
    assert spec.expansion2 == spec.expansion1


def test_cartesian_grid_spec_separate_axes():
    df1 = _sample_intervals_1d()
    df2 = _sample_intervals_1d()
    spec = CartesianGridSpec(
        intervals1=df1, expansion1=[-1000, 0, 1000],
        intervals2=df2, expansion2=[-500, 0, 500],
    )
    assert spec.expansion1 == (-1000, 0, 1000)
    assert spec.expansion2 == (-500, 0, 500)


def test_cartesian_grid_spec_expansion_sorted_dedup():
    spec = CartesianGridSpec(
        intervals1=_sample_intervals_1d(),
        expansion1=[50, -50, 0, 0, 50],  # duplicates + out of order
    )
    assert spec.expansion1 == (-50, 0, 50)


def test_cartesian_grid_spec_expansion_too_few_values():
    with pytest.raises(ValueError, match="at least 2 unique values"):
        CartesianGridSpec(
            intervals1=_sample_intervals_1d(),
            expansion1=[100, 100],  # only 1 unique
        )


def test_cartesian_grid_spec_band_idx_partial_rejected():
    with pytest.raises(ValueError, match="both"):
        CartesianGridSpec(
            intervals1=_sample_intervals_1d(),
            expansion1=[-50, 0, 50],
            min_band_idx=1,
        )


def test_cartesian_grid_spec_band_idx_with_intervals2_rejected():
    df = _sample_intervals_1d()
    with pytest.raises(ValueError, match="intervals2"):
        CartesianGridSpec(
            intervals1=df,
            expansion1=[-50, 0, 50],
            intervals2=df,
            expansion2=[-50, 0, 50],
            min_band_idx=1,
            max_band_idx=10,
        )


def test_cartesian_grid_spec_band_idx_inverted_rejected():
    with pytest.raises(ValueError, match="min_band_idx exceeds"):
        CartesianGridSpec(
            intervals1=_sample_intervals_1d(),
            expansion1=[-50, 0, 50],
            min_band_idx=5,
            max_band_idx=1,
        )


def test_parse_cartesian_grid_spec_passes_through():
    spec = CartesianGridSpec(
        intervals1=_sample_intervals_1d(),
        expansion1=[-50, 0, 50],
    )
    out = parse_iterator_policy(spec, intervals_is_2d=True)
    assert out is spec


def test_parse_cartesian_grid_spec_requires_2d():
    spec = CartesianGridSpec(
        intervals1=_sample_intervals_1d(),
        expansion1=[-50, 0, 50],
    )
    with pytest.raises(ValueError, match="2D"):
        parse_iterator_policy(spec, intervals_is_2d=False)
