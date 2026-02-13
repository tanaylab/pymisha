
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from pymisha import summary


def test_gintervals_summary_optimization():
    # Setup mock data - scaled up
    n_intervals = 10000
    n_points = 100000

    intervals = pd.DataFrame({
        'chrom': ['chr1'] * n_intervals,
        'start': np.arange(n_intervals) * 100,
        'end': np.arange(n_intervals) * 100 + 50
    })

    # Mock gextract return value
    # Randomly assign points to intervals
    interval_ids = np.random.randint(1, n_intervals + 1, n_points)
    values = np.random.randn(n_points)

    extract_df = pd.DataFrame({
        'chrom': ['chr1'] * n_points,
        'start': np.zeros(n_points),
        'end': np.zeros(n_points),
        'intervalID': interval_ids,
        'val': values
    })

    # Patch gextract and _find_vtracks_in_expr
    with patch('pymisha.summary.gextract', return_value=extract_df), \
         patch('pymisha.summary._find_vtracks_in_expr', return_value=True), \
         patch('pymisha.summary._checkroot'):

        print(f"Running summary with {n_intervals} intervals and {n_points} points...")
        start_time = time.time()
        result = summary.gintervals_summary("vtrack.test", intervals)
        end_time = time.time()

        print(f"Summary execution time: {end_time - start_time:.4f} seconds")

        # Basic validation
        assert len(result) == n_intervals
        assert 'Mean' in result.columns
        assert not result['Mean'].isnull().all()

def test_gintervals_quantiles_optimization():
    # Setup mock data - scaled up
    n_intervals = 10000
    n_points = 100000

    intervals = pd.DataFrame({
        'chrom': ['chr1'] * n_intervals,
        'start': np.arange(n_intervals) * 100,
        'end': np.arange(n_intervals) * 100 + 50
    })

    interval_ids = np.random.randint(1, n_intervals + 1, n_points)
    values = np.random.randn(n_points)

    extract_df = pd.DataFrame({
        'chrom': ['chr1'] * n_points,
        'start': np.zeros(n_points),
        'end': np.zeros(n_points),
        'intervalID': interval_ids,
        'val': values
    })

    with patch('pymisha.summary.gextract', return_value=extract_df), \
         patch('pymisha.summary._find_vtracks_in_expr', return_value=True), \
         patch('pymisha.summary._checkroot'):

        print(f"Running quantiles with {n_intervals} intervals and {n_points} points...")
        start_time = time.time()
        result = summary.gintervals_quantiles("vtrack.test", [0.5], intervals)
        end_time = time.time()

        print(f"Quantiles execution time: {end_time - start_time:.4f} seconds")

        assert len(result) == n_intervals
        assert '0.5' in result.columns

if __name__ == "__main__":
    test_gintervals_summary_optimization()
    test_gintervals_quantiles_optimization()
