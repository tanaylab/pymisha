"""Common type aliases for the pymisha package.

These aliases capture the recurring parameter and return types used
across the public API.  They are intentionally broad to support
gradual typing adoption without breaking existing call sites.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import pandas as pd

# -- Genomic interval DataFrame ------------------------------------------------
# A pandas DataFrame with at least ``chrom``, ``start``, ``end`` columns.
Intervals: TypeAlias = pd.DataFrame

# -- C++ wire format -----------------------------------------------------------
# The internal "PMDataFrame" returned by ``_pymisha`` C++ functions:
# a list whose first element is a numpy object-array of column names
# followed by one numpy array (or [categories, codes] pair) per column.
# ``_pymisha2df()`` converts this to a pandas DataFrame.
PMDataFrame: TypeAlias = list[np.ndarray | list[np.ndarray]]

# -- Iterator parameter -------------------------------------------------------
# Controls how intervals are iterated (bin size, or None for no binning).
Iterator: TypeAlias = int | float | None

# -- Track expression ----------------------------------------------------------
# A single track expression string, or a list of expressions.
TrackExpr: TypeAlias = str | list[str]

# -- Chromosome specification --------------------------------------------------
# Chromosomes can be given as a single name/number, a list of names/numbers,
# or None (meaning all chromosomes).
Chroms: TypeAlias = str | int | Sequence[str | int] | None

# -- Numpy array shorthand -----------------------------------------------------
NumpyArray: TypeAlias = np.ndarray
