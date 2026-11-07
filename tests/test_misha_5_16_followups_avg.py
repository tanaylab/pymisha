"""Averages and sums of a track over intervals not aligned to its bins, as R misha 5.16.0.

The expected values are R misha 5.16.0's (gmultitasking = FALSE) on the test database, printed
with sprintf("%.17g"), and are compared exactly: a dense track sums the bins in double and
rounds the mean to float once, and a filtered avg or sum pools the float mean or sum of each
unmasked part."""

import numpy as np
import pandas as pd
import pytest

import pymisha as pm

CHROMS = ["chr1"] * 14 + ["chr2"] * 3 + ["chrX"] * 3
STARTS = [434, 1479, 2894, 5191, 8279, 9386, 9659, 12801, 14062, 15175, 17141, 18988, 21665, 24319,
          413, 3752, 4402, 457, 2065, 4651]
ENDS = [535, 2742, 4534, 8083, 9078, 9472, 12279, 13284, 15045, 17134, 18810, 21210, 24126, 26000,
        3346, 3988, 5455, 1891, 4542, 7145]

R_EXTRACT = [0.093333333730697632, 0.030769230797886848, 0.042941175401210785, 0.04237288236618042,
             0.034117646515369415, 0.093333333730697632, 0.066792450845241547, 0.079999998211860657,
             0.048000000417232513, 0.08999999612569809, 0.091428570449352264, 0.061304345726966858,
             0.076799996197223663, 0.07235293835401535, 0.071186438202857971, 0.067999996244907379,
             0.13727273046970367, 0.064827583730220795, 0.063999995589256287, 0.078400000929832458]
# gextract("dense_track", intervals = gintervals("chr1", 0, 30000), iterator = 170), first 40 rows
R_ITER170 = [0.16944444179534912, 0.17000000178813934, 0.11999999731779099, 0.029999999329447746, 0,
             0.05000000074505806, 0.030000001192092896, 0.024000000208616257, 0.10499999672174454,
             0.054999999701976776, 0.0099999997764825821, 0.035000000149011612, 0.048000000417232513,
             0.019999999552965164, 0.030000001192092896, 0.0099999997764825821, 0.064999997615814209,
             0.067999996244907379, 0.044999998062849045, 0.019999999552965164, 0.064999997615814209,
             0.035000000149011612, 0.035999998450279236, 0.054999999701976776, 0.029999999329447746,
             0.044999998062849045, 0.035000000149011612, 0.064000003039836884, 0.075000002980232239,
             0.024999998509883881, 0.054999999701976776, 0.039999999105930328, 0.043999999761581421,
             0.014999999664723873, 0.049999997019767761, 0.004999999888241291, 0.059999998658895493,
             0.12399999797344208, 0.035000000149011612, 0.049999997019767761]
R_GSUMMARY = [20, 0, 0.030769230797886848, 0.13727273046970367, 1.4072326440364122, 0.070361632201820609,
              0.024765757204888109]
R_VSUM_WITH_MAX = [0.2800000011920929, 0.80000001192092896, 1.4599999189376831, 2.5, 0.57999998331069946,
                   0.2800000011920929, 3.5399999618530273, 0.80000001192092896, 0.95999997854232788,
                   3.5999999046325684, 3.2000000476837158, 2.8199999332427979, 3.8399999141693115,
                   2.4600000381469727, 4.1999998092651367, 0.34000000357627869, 3.0199999809265137,
                   1.8799999952316284, 3.2000000476837158, 3.9200000762939453]

nan = np.nan
R_FILTER = {
    ("dense_track", "avg"): [
        0.059999998658895493, 0.027999998927116396, 0.044666664053996406, 0.044347824690782509,
        0.032499999040737748, 0.14000000059604645, 0.069090908731926567, 0.091428570449352264,
        0.038888887191812195, 0.090857141784259257, 0.095555555213380744, 0.064999998267740014,
        0.084761903842999822, 0.073548385452839635, 0.072075471563159299, 0.059999999900658928,
        0.14666667083899179, 0.064615383744239807, 0.068000000218550363, 0.086666668720898177],
    ("dense_track", "sum"): [
        0.11999999731779099, 0.69999998435378075, 1.3400000035762787, 2.0399999655783176,
        0.51999998092651367, 0.14000000059604645, 3.0400000214576721, 0.64000000059604645,
        0.69999998807907104, 3.1799998879432678, 2.5800000578165054, 2.5999999791383743,
        3.5599999725818634, 2.2799999415874481, 3.8199999332427979, 0.17999999970197678,
        2.6400000154972076, 1.6799999475479126, 3.0600000023841858, 3.6400001049041748],
    ("sparse_track", "avg"): [
        nan, nan, nan, 0.43999999761581421, nan, nan, 0.39000000804662704, nan, nan, 0.4160000145435333,
        0.4699999988079071, 0.4699999988079071, 0.54000000655651093, 0.57999999821186066,
        0.45600000619888303, nan, 0.52499997615814209, nan, 0.45888890326023102, 0.46107142791152],
    ("sparse_track", "sum"): [
        nan, nan, nan, 0.43999999761581421, nan, nan, 1.5600000321865082, nan, nan, 2.0800000727176666,
        1.8799999952316284, 0.93999999761581421, 1.0800000131130219, 1.1599999964237213,
        2.2800000309944153, nan, 1.0499999523162842, nan, 2.7533334493637085, 3.6885714530944824],
    ("array_track", "avg"): [
        nan, nan, nan, 503.20001220703125, nan, nan, 754.66668701171875, nan, nan, 1605.1400390624999,
        2054.386962890625, 2355.28564453125, 2754.142822265625, 2954.375, 72004.5703125, nan,
        72854.171875, nan, 134854.53125, 135554.8515625],
    ("array_track", "sum"): [
        nan, nan, nan, 503.20001220703125, nan, nan, 3018.666748046875, nan, nan, 8025.7001953125,
        8217.5478515625, 4710.5712890625, 5508.28564453125, 5908.75, 360022.8515625, nan,
        145708.34375, nan, 809127.1875, 1084438.8125],
}


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setitem(pm.CONFIG, "multitasking", False)
    yield
    pm.gvtrack_clear()


def _intervals():
    return pd.DataFrame({"chrom": CHROMS, "start": STARTS, "end": ENDS})


def _mask():
    """Holes of 40-209 bp every 700 bp, which cut most of the intervals into several parts."""
    starts = np.arange(300, 40001, 700)
    ends = starts + 40 + starts % 170
    chroms = np.repeat(["chr1", "chr2", "chrX"], len(starts))
    return pd.DataFrame({"chrom": chroms, "start": np.tile(starts, 3), "end": np.tile(ends, 3)})


def _assert_same(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float))


def test_dense_avg_over_unaligned_intervals():
    iv = _intervals()
    _assert_same(pm.gextract("dense_track", iv, iterator=iv)["dense_track"], R_EXTRACT)


def test_dense_avg_with_unaligned_iterator():
    res = pm.gextract("dense_track", pm.gintervals("chr1", 0, 30000), iterator=170)
    _assert_same(res["dense_track"].iloc[:len(R_ITER170)], R_ITER170)


def test_dense_avg_in_gsummary():
    iv = _intervals()
    _assert_same(pm.gsummary("dense_track", iv, iterator=iv), R_GSUMMARY)


def test_dense_sum_vtrack_next_to_another_function():
    iv = _intervals()
    pm.gvtrack_create("v_sum", "dense_track", "sum")
    pm.gvtrack_create("v_max", "dense_track", "max")
    _assert_same(pm.gextract(["v_sum", "v_max"], iv, iterator=iv)["v_sum"], R_VSUM_WITH_MAX)


@pytest.mark.parametrize(("track", "func"), list(R_FILTER))
def test_filtered_avg_and_sum_pool_the_parts(track, func):
    iv = _intervals()
    pm.gvtrack_create("v_f", track, func)
    pm.gvtrack_filter("v_f", _mask())
    _assert_same(pm.gextract("v_f", iv, iterator=iv)["v_f"], R_FILTER[(track, func)])


def test_gintervals_summary_vtrack_interval_without_values():
    pm.gvtrack_create("v_a", "dense_track", "avg")
    res = pm.gintervals_summary("v_a", pm.gintervals(["chr1", "chr1"], [434, 170000], [535, 171000]))
    empty = res.iloc[1]
    assert empty["Total intervals"] == 20
    assert empty["NaN intervals"] == 20
    for col in ("Min", "Max", "Sum", "Mean", "Std dev"):
        assert np.isnan(empty[col]), col
    assert res.iloc[0]["Total intervals"] == 3
    assert res.iloc[0]["Sum"] == pytest.approx(0.28)
