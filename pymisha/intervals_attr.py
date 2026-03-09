"""Interval set attribute management (get/set/export/import).

Attributes are stored as `.iattr` binary files next to `.interv` files.
The binary format uses null-terminated key-value pairs:
    key1\0value1\0key2\0value2\0...
"""

from pathlib import Path

from ._shared import _checkroot


def _iattr_path(intervals_set):
    """Resolve the filesystem path of the ``.iattr`` file for an interval set."""
    from .intervals import gintervals_dataset

    db_root = gintervals_dataset(intervals_set)
    if db_root is None:
        raise ValueError(f"Intervals set {intervals_set} does not exist")

    path_part = intervals_set.replace(".", "/")
    interv_path = Path(db_root) / "tracks" / f"{path_part}.interv"

    if interv_path.is_dir():
        # Big set: store inside the directory
        return interv_path / ".iattr"
    # Small set: sibling .iattr file
    return Path(db_root) / "tracks" / f"{path_part}.iattr"


def _iattr_read(path):
    """Read a binary ``.iattr`` file into a dict of ``{key: value}``."""
    path = Path(path)
    if not path.exists():
        return {}
    data = path.read_bytes()
    if not data:
        return {}

    # Split on null bytes
    parts = data.split(b"\x00")
    # Drop trailing empty part (last null byte produces it)
    if parts and parts[-1] == b"":
        parts = parts[:-1]
    # Must have even number of strings (key-value pairs)
    if len(parts) % 2 != 0:
        parts = parts[: len(parts) - 1]

    result = {}
    for i in range(0, len(parts), 2):
        key = parts[i].decode("utf-8")
        value = parts[i + 1].decode("utf-8")
        result[key] = value
    return result


def _iattr_write(path, attrs):
    """Write a dict of ``{key: value}`` to a binary ``.iattr`` file."""
    path = Path(path)
    if not attrs:
        if path.exists():
            path.unlink()
        return

    chunks = []
    for key, value in attrs.items():
        chunks.append(key.encode("utf-8"))
        chunks.append(b"\x00")
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\x00")
    path.write_bytes(b"".join(chunks))


def _check_writable(intervals_set):
    """Check that an interval set belongs to the working database (not read-only)."""
    from . import _shared
    from .intervals import gintervals_dataset

    db_root = gintervals_dataset(intervals_set)
    if db_root is not None and db_root != _shared._GROOT:
        raise ValueError(
            f"Intervals set {intervals_set} belongs to a read-only dataset and cannot be modified"
        )


def gintervals_attr_get(intervals_set, attr):
    """Return the value of an interval set attribute.

    Parameters
    ----------
    intervals_set : str
        Interval set name (e.g. ``"annotations"``).
    attr : str
        Attribute name.

    Returns
    -------
    str
        Attribute value, or ``""`` if the attribute does not exist.

    See Also
    --------
    gintervals_attr_set : Assign an attribute value.
    gintervals_attr_export : Export attributes as a DataFrame.
    gintervals_attr_import : Import attributes from a DataFrame.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_attr_set("annotations", "test_key", "test_val")
    >>> pm.gintervals_attr_get("annotations", "test_key")
    'test_val'
    >>> pm.gintervals_attr_set("annotations", "test_key", "")
    """
    if intervals_set is None or attr is None:
        raise ValueError("Usage: gintervals_attr_get(intervals_set, attr)")
    _checkroot()

    result = gintervals_attr_export(intervals_set, attr)
    return result.iloc[0, 0]


def gintervals_attr_set(intervals_set, attr, value):
    """Assign a value to an interval set attribute.

    If *value* is an empty string the attribute is removed.

    Parameters
    ----------
    intervals_set : str
        Interval set name.
    attr : str
        Attribute name.
    value : str
        Attribute value. Empty string removes the attribute.

    Returns
    -------
    None

    See Also
    --------
    gintervals_attr_get : Retrieve an attribute value.
    gintervals_attr_export : Export attributes as a DataFrame.
    gintervals_attr_import : Import attributes from a DataFrame.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_attr_set("annotations", "color", "red")
    >>> pm.gintervals_attr_get("annotations", "color")
    'red'
    >>> pm.gintervals_attr_set("annotations", "color", "")
    """
    if intervals_set is None or attr is None or value is None:
        raise ValueError("Usage: gintervals_attr_set(intervals_set, attr, value)")
    _checkroot()

    import pandas as pd

    table = pd.DataFrame({attr: [value]}, index=[intervals_set])
    gintervals_attr_import(table, remove_others=False)


def gintervals_attr_export(intervals_set=None, attrs=None):
    """Return interval set attributes as a DataFrame.

    Parameters
    ----------
    intervals_set : str, list of str, or None
        Interval set name(s). If ``None``, all existing interval sets are used.
    attrs : str, list of str, or None
        Attribute name(s) to retrieve. If ``None``, all attributes are returned
        sorted by popularity (most common first).

    Returns
    -------
    pandas.DataFrame
        DataFrame with interval set names as the index and attribute names
        as columns. Missing attributes are represented as ``""``.

    Raises
    ------
    ValueError
        If a named interval set does not exist.

    See Also
    --------
    gintervals_attr_import : Import attributes from a DataFrame.
    gintervals_attr_get : Retrieve a single attribute value.
    gintervals_attr_set : Set a single attribute value.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gintervals_attr_set("annotations", "color", "red")
    >>> pm.gintervals_attr_export("annotations")  # doctest: +SKIP
    >>> pm.gintervals_attr_set("annotations", "color", "")
    """
    import pandas as pd

    _checkroot()
    from .intervals import gintervals_exists, gintervals_ls

    # Normalize intervals_set
    if intervals_set is None:
        isets = gintervals_ls()
    elif isinstance(intervals_set, str):
        isets = [intervals_set]
    else:
        isets = list(dict.fromkeys(intervals_set))  # unique, order-preserving

    # Validate
    for iset in isets:
        if not gintervals_exists(iset):
            raise ValueError(f"Intervals set {iset} does not exist")

    # Normalize attrs
    if attrs is not None:
        attrs = [attrs] if isinstance(attrs, str) else list(dict.fromkeys(attrs))

    # Read all attributes
    all_attrs = {}
    for iset in isets:
        path = _iattr_path(iset)
        all_attrs[iset] = _iattr_read(path)

    # Determine column names
    if attrs is None:
        from collections import Counter

        name_counts = Counter()
        for d in all_attrs.values():
            name_counts.update(d.keys())
        if not name_counts:
            return pd.DataFrame(index=isets)
        # Sort by popularity (most common first)
        attr_names = [name for name, _ in name_counts.most_common()]
    else:
        attr_names = attrs

    if not attr_names:
        return pd.DataFrame(index=isets)

    # Build result
    data = {a: [""] * len(isets) for a in attr_names}
    for i, iset in enumerate(isets):
        iattrs = all_attrs[iset]
        for a in attr_names:
            if a in iattrs:
                data[a][i] = iattrs[a]

    return pd.DataFrame(data, index=isets)


def gintervals_attr_import(table, remove_others=False):
    """Import interval set attributes from a DataFrame.

    Parameters
    ----------
    table : pandas.DataFrame
        DataFrame with interval set names as the index and attribute names
        as columns. Values must be strings. An empty string removes the
        attribute.
    remove_others : bool, default False
        If ``True``, attributes not present in the table are removed.
        If ``False``, existing attributes not in the table are preserved.

    Raises
    ------
    ValueError
        If *table* is ``None``, has invalid format, references non-existent
        interval sets, or contains duplicate interval set names or attribute
        names.

    See Also
    --------
    gintervals_attr_export : Export attributes as a DataFrame.
    gintervals_attr_get : Retrieve a single attribute value.
    gintervals_attr_set : Set a single attribute value.

    Examples
    --------
    >>> import pymisha as pm
    >>> import pandas as pd
    >>> _ = pm.gdb_init_examples()
    >>> t = pd.DataFrame({"myattr": ["val"]}, index=["annotations"])
    >>> pm.gintervals_attr_import(t)
    >>> pm.gintervals_attr_export("annotations", "myattr")  # doctest: +SKIP
    >>> t = pd.DataFrame({"myattr": [""]}, index=["annotations"])
    >>> pm.gintervals_attr_import(t)
    """
    import pandas as pd

    if table is None:
        raise ValueError("Usage: gintervals_attr_import(table, remove_others=False)")
    _checkroot()
    from .intervals import gintervals_exists

    if not isinstance(table, pd.DataFrame) or table.shape[0] < 1 or table.shape[1] < 1:
        raise ValueError("Invalid format of attributes table")

    isets = list(table.index)
    attr_names = list(table.columns)

    if any(pd.isna(isets)) or any(pd.isna(attr_names)) or any(a == "" for a in attr_names):
        raise ValueError("Invalid format of attributes table")

    # Validate interval sets exist
    for iset in isets:
        if not gintervals_exists(iset):
            raise ValueError(f"Intervals set {iset} does not exist")

    # Check duplicates
    if len(set(isets)) != len(isets):
        dup = next(n for n in isets if isets.count(n) > 1)
        raise ValueError(f"Intervals set {dup} appears more than once")

    if len(set(attr_names)) != len(attr_names):
        dup = next(a for a in attr_names if attr_names.count(a) > 1)
        raise ValueError(f"Attribute {dup} appears more than once")

    # Coerce all values to string
    table = table.astype(str)

    for iset in isets:
        _check_writable(iset)

        path = _iattr_path(iset)
        existing = _iattr_read(path)

        new_attrs = {} if remove_others else dict(existing)

        for a in attr_names:
            val = table.loc[iset, a]
            if pd.isna(val) or val == "":
                new_attrs.pop(a, None)
            else:
                new_attrs[a] = val

        _iattr_write(path, new_attrs)
