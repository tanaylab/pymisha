"""Track listing, metadata, and creation/import helpers."""

import bz2
import contextlib
import datetime as _datetime
import fnmatch
import ftplib
import getpass as _getpass
import glob
import gzip
import io
import lzma
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from . import _shared
from ._name_validation import validate_dotted_name
from ._safe_pickle import restricted_load, restricted_loads
from ._shared import _apply_gwd_to_names, _checkroot, _df2pymisha, _pymisha


def _load_track_attributes(track_name):
    """
    Load track attributes from .attributes file (binary) or .attributes.yaml.
    """
    track_path = _pymisha.pm_track_path(track_name)
    if not track_path:
        return {}

    attrs = {}

    bin_path = os.path.join(track_path, ".attributes")
    if os.path.exists(bin_path):
        try:
            with open(bin_path, 'rb') as f:
                data = f.read()
            parts = data.split(b'\x00')
            parts = [p.decode('utf-8', errors='replace') for p in parts if p]
            for i in range(0, len(parts) - 1, 2):
                attrs[parts[i]] = parts[i + 1]
            return attrs
        except (UnicodeDecodeError, ValueError):
            pass

    yaml_path = os.path.join(track_path, ".attributes.yaml")
    if os.path.exists(yaml_path):
        try:
            import yaml
        except ImportError:
            return attrs
        try:
            with open(yaml_path) as f:
                attrs = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            pass

    return attrs


def gtrack_ls(*patterns, ignore_case=False, **attr_filters):
    """
    Return a list of track names in the Genomic Database.

    Returns track names that match all supplied patterns. Name patterns are
    applied as regex searches against track names. Attribute patterns are
    matched against the corresponding track attribute values. Multiple
    patterns are applied conjunctively (all must match).

    Parameters
    ----------
    *patterns : str
        Regex patterns to filter track names. Each pattern is applied
        sequentially; only tracks matching all patterns are returned.
    ignore_case : bool, default False
        If True, pattern matching is case-insensitive.
    **attr_filters : str
        Keyword arguments of the form ``attribute_name=pattern`` where
        underscores in the keyword are converted to dots for the attribute
        lookup (e.g., ``created_by="sparse"`` matches attribute
        ``created.by``).

    Returns
    -------
    list of str or None
        Sorted list of matching track names, or None if no tracks match.

    Raises
    ------
    ValueError
        If a regex pattern is invalid.

    See Also
    --------
    gtrack_exists : Test whether a single track exists.
    gtrack_info : Get metadata for a track.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_ls()  # doctest: +SKIP
    ['array_track', 'dense_track', 'rects_track', 'sparse_track', 'subdir.dense_track2']
    >>> pm.gtrack_ls("dense")  # doctest: +SKIP
    ['dense_track', 'subdir.dense_track2']
    >>> pm.gtrack_ls(created_by="create_sparse")  # doctest: +SKIP
    ['sparse_track']
    """
    _checkroot()
    tracks = _pymisha.pm_track_names()

    if tracks is None or len(tracks) == 0:
        return None

    # Filter/rebase by current working directory
    tracks = _apply_gwd_to_names(tracks)
    if not tracks:
        return None

    flags = re.IGNORECASE if ignore_case else 0

    for pattern in patterns:
        try:
            regex = re.compile(pattern, flags)
            tracks = [t for t in tracks if regex.search(t)]
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

    if not tracks:
        return None

    if attr_filters:
        converted_filters = {key.replace('_', '.'): pattern for key, pattern in attr_filters.items()}
        filtered_tracks = []
        for track in tracks:
            attrs = _load_track_attributes(track)

            all_match = True
            for attr_name, pattern in converted_filters.items():
                attr_value = attrs.get(attr_name, "")
                try:
                    regex = re.compile(pattern, flags)
                    if not regex.search(str(attr_value)):
                        all_match = False
                        break
                except re.error as e:
                    raise ValueError(f"Invalid regex pattern '{pattern}' for attribute '{attr_name}': {e}") from e

            if all_match:
                filtered_tracks.append(track)

        tracks = filtered_tracks

    if not tracks:
        return None

    return tracks


def gtrack_info(track):
    """
    Return metadata about a track.

    Returns a dictionary containing track properties such as type,
    dimensions, bin size, total size in bytes, and any user-defined
    attributes. The fields vary depending on the track type (Dense,
    Sparse, Rectangles, Points).

    Parameters
    ----------
    track : str
        Track name.

    Returns
    -------
    dict
        Dictionary of track properties. Common keys include ``"type"``
        (``"dense"``, ``"sparse"``, ``"rectangles"``, ``"points"``),
        ``"bin_size"`` (for dense tracks), ``"total_size"``, and
        ``"attributes"`` (dict of user-set attributes, if any).

    Raises
    ------
    ValueError
        If the track does not exist.

    See Also
    --------
    gtrack_exists : Test whether a track exists.
    gtrack_ls : List available tracks.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_info("dense_track")  # doctest: +SKIP
    {'type': 'dense', 'dimensions': 1, ...}
    >>> pm.gtrack_info("sparse_track")  # doctest: +SKIP
    {'type': 'sparse', 'dimensions': 1, ...}
    """
    _checkroot()
    info = _pymisha.pm_track_info(track)

    if 'attributes_path' in info:
        del info['attributes_path']

    attrs = _load_track_attributes(track)
    if attrs:
        info['attributes'] = attrs

    return info


def gtrack_dataset(track):
    """
    Return the database root path that contains a track.

    When multiple databases are connected, this identifies which database
    a track belongs to by returning the filesystem path of that database
    root.

    Parameters
    ----------
    track : str
        Track name.

    Returns
    -------
    str
        Absolute filesystem path of the database root containing the track.

    Raises
    ------
    ValueError
        If track is None or the track does not exist.

    See Also
    --------
    gtrack_info : Get full metadata for a track.
    gtrack_ls : List available tracks.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_dataset("dense_track")  # doctest: +SKIP
    '.../trackdb/test'
    """
    if track is None:
        raise ValueError("track cannot be None")
    _checkroot()
    return _pymisha.pm_track_dataset(track)


def _save_track_attributes(track_name, attrs):
    """
    Save track attributes to the .attributes file (binary format).
    """
    track_path = _pymisha.pm_track_path(track_name)
    if not track_path:
        raise ValueError(f"Track '{track_name}' does not exist")

    bin_path = os.path.join(track_path, ".attributes")

    # Filter out empty values (empty means delete attribute).
    # Keep falsy-but-valid values like 0 and False.
    attrs = {k: v for k, v in attrs.items() if v is not None and v != ""}

    if not attrs:
        # If no attributes, remove the file if it exists
        if os.path.exists(bin_path):
            os.remove(bin_path)
        return

    # Build binary format: key\0value\0key\0value\0...
    parts = []
    for key, value in sorted(attrs.items()):
        parts.append(key.encode('utf-8'))
        parts.append(b'\x00')
        parts.append(str(value).encode('utf-8'))
        parts.append(b'\x00')

    with open(bin_path, 'wb') as f:
        f.write(b''.join(parts))


def _track_exists(track_name):
    """Check if a track exists."""
    track_path = _pymisha.pm_track_path(track_name)
    return track_path is not None and track_path != ""


def _validate_track_name(track):
    validate_dotted_name(track, "track name")


def _validate_track_var_name(var):
    if not isinstance(var, str) or not var:
        raise ValueError("var must be a non-empty string")
    if "\x00" in var:
        raise ValueError("var must not contain NUL bytes")
    if os.path.isabs(var):
        raise ValueError("var must be a relative name")
    if var in {".", ".."}:
        raise ValueError("var cannot be '.' or '..'")
    if ".." in var.split("/"):
        raise ValueError("var cannot contain path traversal components")
    if os.sep in var or (os.altsep and os.altsep in var):
        raise ValueError("var must not contain path separators")


def _target_root():
    return _shared._UROOT or _shared._GROOT


def _track_dir_for_create(track):
    root = _target_root()
    if not root:
        raise ValueError("Database not initialized. Call gdb_init() first.")
    return Path(root) / "tracks" / f"{track.replace('.', '/')}.track"


def _db_is_indexed(root):
    if not root:
        return False
    seq_dir = Path(root) / "seq"
    return (seq_dir / "genome.idx").exists() and (seq_dir / "genome.seq").exists()


def _normalize_intervals_df(intervals):
    if intervals is None:
        raise ValueError("intervals cannot be None")
    if not isinstance(intervals, pd.DataFrame):
        intervals = pd.DataFrame(intervals)
    required = {"chrom", "start", "end"}
    if not required.issubset(intervals.columns):
        raise ValueError("intervals must contain columns: chrom, start, end")

    out = intervals.copy()
    out = out.loc[:, ["chrom", "start", "end"]]
    out["chrom"] = out["chrom"].astype(str)
    out["start"] = pd.to_numeric(out["start"], errors="coerce")
    out["end"] = pd.to_numeric(out["end"], errors="coerce")
    out = out.dropna(subset=["chrom", "start", "end"])
    if len(out) == 0:
        raise ValueError("intervals is empty after dropping invalid rows")

    out["start"] = out["start"].astype(np.int64)
    out["end"] = out["end"].astype(np.int64)
    out = out[(out["start"] >= 0) & (out["end"] > out["start"])].copy()
    if len(out) == 0:
        raise ValueError("intervals must contain at least one valid row with 0 <= start < end")
    return out.reset_index(drop=True)


def _canonicalize_known_chroms(df):
    from .intervals import gintervals_all

    chrom_sizes = gintervals_all()
    known = set(chrom_sizes["chrom"].astype(str).tolist())
    keep = []
    canon = []
    for chrom in df["chrom"].astype(str).tolist():
        try:
            c = _pymisha.pm_normalize_chroms([chrom])[0]
        except Exception:
            keep.append(False)
            canon.append(chrom)
            continue
        keep.append(c in known)
        canon.append(c)
    out = df.loc[keep].copy()
    out["chrom"] = [c for k, c in zip(keep, canon, strict=False) if k]
    return out.reset_index(drop=True)


def _set_created_attrs(track, description, created_by, attrs=None):
    # Bypass readonly check for internal track creation attrs
    existing_attrs = _load_track_attributes(track)
    existing_attrs["created.by"] = created_by
    existing_attrs["created.date"] = _datetime.datetime.now().ctime()
    existing_attrs["created.user"] = _getpass.getuser()
    existing_attrs["description"] = str(description)
    if attrs is not None:
        if not isinstance(attrs, dict):
            raise ValueError("attrs must be a dict of attribute name -> value")
        for k, v in attrs.items():
            if not isinstance(k, str) or not k:
                raise ValueError("attrs keys must be non-empty strings")
            existing_attrs[k] = "" if v is None else str(v)
    _save_track_attributes(track, existing_attrs)


def _open_text_auto(path):
    lower = str(path).lower()
    if lower.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    if lower.endswith(".bz2"):
        return io.TextIOWrapper(bz2.open(path, "rb"), encoding="utf-8", errors="replace")
    if lower.endswith((".xz", ".lzma")):
        return io.TextIOWrapper(lzma.open(path, "rb"), encoding="utf-8", errors="replace")
    if lower.endswith(".zip"):
        zf = zipfile.ZipFile(path, "r")
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if len(names) == 0:
            zf.close()
            raise ValueError(f"Zip file '{path}' does not contain a regular file")
        stream = io.TextIOWrapper(zf.open(names[0], "r"), encoding="utf-8", errors="replace")
        base_close = stream.close

        def _close_with_zip(*args, **kwargs):
            try:
                return base_close(*args, **kwargs)
            finally:
                zf.close()

        stream.close = _close_with_zip
        return stream
    return open(path, encoding="utf-8", errors="replace")


def _close_text_auto(stream):
    stream.close()


def _parse_bed(path):
    chrom, start, end, value = [], [], [], []
    stream = _open_text_auto(path)
    try:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"Malformed BED line: {line}")
            chrom.append(fields[0])
            start.append(int(float(fields[1])))
            end.append(int(float(fields[2])))
            v = 1.0
            if len(fields) >= 5:
                try:
                    v = float(fields[4])
                except ValueError:
                    v = 1.0
            value.append(v)
    finally:
        _close_text_auto(stream)
    if len(chrom) == 0:
        raise ValueError(f"BED file '{path}' contains no intervals")
    return pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})


def _parse_wig_or_bedgraph(path):
    chrom, start, end, value = [], [], [], []
    mode = None
    cur_chrom = None
    cur_step = 1
    cur_span = 1
    cur_pos0 = 0

    stream = _open_text_auto(path)
    try:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith(("#", "track", "browser")):
                continue

            lower = line.lower()
            if lower.startswith("fixedstep"):
                mode = "fixed"
                tokens = line.split()
                kv = {}
                for tok in tokens[1:]:
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        kv[k.lower()] = v
                if "chrom" not in kv or "start" not in kv:
                    raise ValueError(f"Malformed fixedStep line: {line}")
                cur_chrom = kv["chrom"]
                cur_step = int(kv.get("step", "1"))
                cur_span = int(kv.get("span", "1"))
                cur_pos0 = int(kv["start"]) - 1
                continue

            if lower.startswith("variablestep"):
                mode = "var"
                tokens = line.split()
                kv = {}
                for tok in tokens[1:]:
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        kv[k.lower()] = v
                if "chrom" not in kv:
                    raise ValueError(f"Malformed variableStep line: {line}")
                cur_chrom = kv["chrom"]
                cur_span = int(kv.get("span", "1"))
                continue

            fields = line.split()
            if mode == "fixed" and len(fields) == 1:
                v = float(fields[0])
                chrom.append(cur_chrom)
                start.append(cur_pos0)
                end.append(cur_pos0 + cur_span)
                value.append(v)
                cur_pos0 += cur_step
                continue
            if mode == "var" and len(fields) >= 2:
                pos0 = int(float(fields[0])) - 1
                v = float(fields[1])
                chrom.append(cur_chrom)
                start.append(pos0)
                end.append(pos0 + cur_span)
                value.append(v)
                continue
            if len(fields) >= 4:
                chrom.append(fields[0])
                start.append(int(float(fields[1])))
                end.append(int(float(fields[2])))
                value.append(float(fields[3]))
                continue
            raise ValueError(f"Cannot parse WIG/BedGraph line: {line}")
    finally:
        _close_text_auto(stream)

    if len(chrom) == 0:
        raise ValueError(f"WIG/BedGraph file '{path}' contains no intervals")
    return pd.DataFrame({"chrom": chrom, "start": start, "end": end, "value": value})


def _parse_tabular_track(path):
    stream = _open_text_auto(path)
    header = None
    rows = []
    try:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if header is None:
                header = line.split("\t") if "\t" in line else line.split()
                continue
            fields = line.split("\t") if "\t" in line else line.split()
            if len(fields) == 0:
                continue
            rows.append(fields)
    finally:
        _close_text_auto(stream)

    if header is None:
        raise ValueError(f"File '{path}' is empty")
    cols = [c.strip() for c in header]
    req = ["chrom", "start", "end"]
    for c in req:
        if c not in cols:
            raise ValueError(f"Tabular track file must contain '{c}' column")
    val_cols = [c for c in cols if c not in req]
    if len(val_cols) != 1:
        raise ValueError("Tabular track file must contain exactly one value column besides chrom/start/end")

    idx = {c: i for i, c in enumerate(cols)}
    c_val = val_cols[0]
    out = {"chrom": [], "start": [], "end": [], "value": []}
    for fields in rows:
        if len(fields) < len(cols):
            continue
        out["chrom"].append(fields[idx["chrom"]])
        out["start"].append(int(float(fields[idx["start"]])))
        out["end"].append(int(float(fields[idx["end"]])))
        out["value"].append(float(fields[idx[c_val]]))
    if len(out["chrom"]) == 0:
        raise ValueError(f"File '{path}' contains no data rows")
    return pd.DataFrame(out)


def _parse_bigwig(path):
    try:
        import pyBigWig
    except ImportError as exc:
        raise ImportError(
            "BigWig import requires pyBigWig. Install with: pip install pyBigWig"
        ) from exc

    from .intervals import gintervals_all

    bw = pyBigWig.open(path)
    if bw is None:
        raise ValueError(f"Failed to open BigWig file '{path}'")
    try:
        known = set(gintervals_all()["chrom"].astype(str).tolist())
        out = {"chrom": [], "start": [], "end": [], "value": []}
        for chrom in (bw.chroms() or {}):
            try:
                norm = _pymisha.pm_normalize_chroms([chrom])[0]
            except Exception:
                continue
            if norm not in known:
                continue
            intervals = bw.intervals(chrom)
            if not intervals:
                continue
            for start, end, value in intervals:
                out["chrom"].append(norm)
                out["start"].append(int(start))
                out["end"].append(int(end))
                out["value"].append(float(value))
    finally:
        bw.close()

    if len(out["chrom"]) == 0:
        raise ValueError(f"BigWig file '{path}' contains no intervals for known chromosomes")
    return pd.DataFrame(out)


def _ensure_track_absent(track):
    if _track_exists(track):
        raise ValueError(f"Track '{track}' already exists")


def gtrack_create_sparse(track, description, intervals, values):
    """
    Create a Sparse track from intervals and values.

    Creates a new Sparse track where each interval carries an associated
    numeric value. Intervals must be non-overlapping within each
    chromosome. Chromosome names are normalized and filtered to those
    present in the current genome database. The description is stored
    as a track attribute.

    Parameters
    ----------
    track : str
        Name for the new track. Must start with a letter and contain
        only alphanumeric characters, underscores, and dots.
    description : str
        Human-readable description stored as a track attribute.
    intervals : pandas.DataFrame
        One-dimensional intervals with columns ``chrom``, ``start``,
        ``end``.
    values : array-like of float
        Numeric values, one per interval. Length must match the number
        of rows in *intervals*.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track already exists, intervals overlap, values length
        does not match intervals, or no intervals map to known
        chromosomes.

    See Also
    --------
    gtrack_create_dense : Create a Dense (fixed-bin) track.
    gtrack_create : Create a track from a track expression.
    gtrack_import : Import a track from a file.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> import pandas as pd
    >>> _ = pm.gdb_init_examples()
    >>> intervs = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [100]})
    >>> pm.gtrack_create_sparse("test_sp", "Test", intervs, [1.0])  # doctest: +SKIP
    >>> pm.gtrack_rm("test_sp", force=True)  # doctest: +SKIP
    """
    _checkroot()
    _validate_track_name(track)
    _ensure_track_absent(track)

    data = _normalize_intervals_df(intervals)
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) != len(data):
        raise ValueError("Length of values must match number of intervals")
    data = data.copy()
    data["value"] = vals
    data = _canonicalize_known_chroms(data)

    if len(data) == 0:
        raise ValueError("No intervals map to known chromosomes")

    data = data.sort_values(["chrom", "start", "end"], kind="mergesort").reset_index(drop=True)
    prev = data.groupby("chrom")["end"].shift(1)
    overlap = data["start"] < prev
    if bool(overlap.fillna(False).any()):
        raise ValueError("Sparse intervals must be sorted and non-overlapping per chromosome")

    track_dir = _track_dir_for_create(track)
    created_new = not track_dir.exists()
    try:
        _pymisha.pm_track_create_sparse(track, _df2pymisha(data))
        _pymisha.pm_dbreload()
        _set_created_attrs(track, description, f'gtrack.create_sparse("{track}", description, intervals, values)')
        if _db_is_indexed(_shared._GROOT):
            gtrack_convert_to_indexed(track, remove_old=False)
        _pymisha.pm_dbreload()
    except Exception:
        if created_new and track_dir.exists():
            shutil.rmtree(track_dir, ignore_errors=True)
            _pymisha.pm_dbreload()
        raise


def gtrack_create_dense(track, description, intervals, values, binsize, defval=np.nan):
    """
    Create a Dense (fixed-bin) track from intervals and values.

    Creates a new Dense track whose genome is tiled into fixed-size bins.
    Each bin stores a single numeric value. Bins not covered by any of
    the supplied intervals are filled with *defval*. The description is
    stored as a track attribute.

    Parameters
    ----------
    track : str
        Name for the new track. Must start with a letter and contain
        only alphanumeric characters, underscores, and dots.
    description : str
        Human-readable description stored as a track attribute.
    intervals : pandas.DataFrame
        One-dimensional intervals with columns ``chrom``, ``start``,
        ``end``.
    values : array-like of float
        Numeric values, one per interval. Length must match the number
        of rows in *intervals*.
    binsize : int
        Bin size in base pairs. Must be a positive integer.
    defval : float, default numpy.nan
        Default value for bins not covered by any interval.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track already exists, binsize is not positive, values
        length does not match intervals, or no intervals map to known
        chromosomes.

    See Also
    --------
    gtrack_create_sparse : Create a Sparse track.
    gtrack_create : Create a track from a track expression.
    gtrack_modify : Modify values of an existing Dense track.
    gtrack_import : Import a track from a file.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> import pandas as pd
    >>> _ = pm.gdb_init_examples()
    >>> intervs = pd.DataFrame({"chrom": ["1"], "start": [0], "end": [100]})
    >>> pm.gtrack_create_dense("test_dn", "Test", intervs, [5.0], 50)  # doctest: +SKIP
    >>> pm.gtrack_rm("test_dn", force=True)  # doctest: +SKIP
    """
    _checkroot()
    _validate_track_name(track)
    _ensure_track_absent(track)

    binsize = int(binsize)
    if binsize <= 0:
        raise ValueError("binsize must be a positive integer")
    defval = float(defval)

    data = _normalize_intervals_df(intervals)
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) != len(data):
        raise ValueError("Length of values must match number of intervals")
    data = data.copy()
    data["value"] = vals
    data = _canonicalize_known_chroms(data)
    if len(data) == 0:
        raise ValueError("No intervals map to known chromosomes")

    track_dir = _track_dir_for_create(track)
    created_new = not track_dir.exists()
    try:
        _pymisha.pm_track_create_dense(track, _df2pymisha(data), int(binsize), float(defval))
        _pymisha.pm_dbreload()
        _set_created_attrs(
            track,
            description,
            f'gtrack.create_dense("{track}", description, intervals, values, {binsize}, {defval:g})',
        )
        gtrack_attr_set(track, "type", "dense")
        gtrack_attr_set(track, "binsize", str(binsize))
        if _db_is_indexed(_shared._GROOT):
            gtrack_convert_to_indexed(track, remove_old=False)
        _pymisha.pm_dbreload()
    except Exception:
        if created_new and track_dir.exists():
            shutil.rmtree(track_dir, ignore_errors=True)
            _pymisha.pm_dbreload()
        raise


def gtrack_modify(track, expr, intervals=None):
    """
    Modify a Dense track's values in-place by evaluating an expression.

    Overwrites the values of an existing Dense track with the result of
    evaluating *expr*. The iterator policy is automatically set to the
    track's bin size. Only Dense (fixed-bin) tracks are supported.

    Parameters
    ----------
    track : str
        Name of the dense track to modify.
    expr : str
        Track expression to evaluate (may reference the track itself).
    intervals : pandas.DataFrame or None, optional
        Genomic scope for modification. If None, the entire genome
        (ALLGENOME) is used.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track does not exist, is not a Dense track, or *expr* is
        None.

    See Also
    --------
    gtrack_create : Create a new track from a track expression.
    gtrack_smooth : Create a smoothed copy of a track.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_modify("dense_track", "dense_track * 2")  # doctest: +SKIP
    >>> pm.gtrack_modify("dense_track", "dense_track / 2")  # doctest: +SKIP
    """
    from .intervals import gintervals_all

    _checkroot()
    if not isinstance(track, str) or not track:
        raise ValueError("track must be a non-empty string")
    if expr is None:
        raise ValueError("expr cannot be None")
    if not _pymisha.pm_track_info(track):
        raise ValueError(f"Track '{track}' does not exist")

    info = gtrack_info(track)
    if info.get("type") != "dense":
        raise ValueError(f"gtrack_modify only supports dense tracks, got '{info.get('type')}'")

    binsize = int(info["bin_size"])

    if intervals is None:
        intervals = gintervals_all()

    _pymisha.pm_modify(track, str(expr), _df2pymisha(intervals), binsize)

    # Update created.by attribute (bypass readonly check for internal update)
    modify_str = f'gtrack.modify({track}, {str(expr)}, intervs)'
    attrs = _load_track_attributes(track)
    existing = attrs.get("created.by", "")
    if existing:
        attrs["created.by"] = existing + "\n" + modify_str
    else:
        attrs["created.by"] = modify_str
    _save_track_attributes(track, attrs)


def gtrack_smooth(track, description, expr, winsize, weight_thr=0, smooth_nans=False,
                  alg="LINEAR_RAMP", iterator=None):
    """
    Create a new Dense track with smoothed values from a track expression.

    Each output bin at coordinate C is computed by smoothing the non-NaN
    values of *expr* within a window of size *winsize* (in coordinate
    units) around C. The smoothing algorithm and handling of NaN /
    edge-of-chromosome gaps are controlled by the remaining parameters.

    Parameters
    ----------
    track : str
        Name of the new track to create.
    description : str
        Human-readable description stored as a track attribute.
    expr : str
        Track expression whose values are smoothed.
    winsize : float
        Smoothing window size in coordinate units. Defines the total
        region considered on both sides of the central point.
    weight_thr : float, default 0
        Weight sum threshold below which the smoothed value is NaN
        instead of a partial-window estimate.
    smooth_nans : bool, default False
        If False, output NaN whenever the central window value is NaN,
        regardless of *weight_thr*. If True, NaN center values are
        filled from surrounding non-NaN values.
    alg : str, default ``"LINEAR_RAMP"``
        Smoothing algorithm. ``"LINEAR_RAMP"`` uses a weighted average
        with linearly decreasing weights. ``"MEAN"`` uses a simple
        arithmetic average.
    iterator : int or None, optional
        Fixed-bin iterator bin size for the new track. If None, the bin
        size is inferred from the track expression.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track already exists, *expr* is None, *winsize* is not
        positive, or *alg* is not one of the supported algorithms.

    See Also
    --------
    gtrack_create : Create a track from a track expression.
    gtrack_modify : Modify an existing Dense track in-place.
    gtrack_create_sparse : Create a Sparse track.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_smooth("smoothed", "Test", "dense_track", 500)  # doctest: +SKIP
    >>> pm.gtrack_rm("smoothed", force=True)  # doctest: +SKIP
    """
    from .intervals import gintervals_all

    _checkroot()
    if expr is None:
        raise ValueError("expr cannot be None")
    if winsize is None or winsize <= 0:
        raise ValueError("winsize must be a positive number")
    if alg not in ("LINEAR_RAMP", "MEAN"):
        raise ValueError(f"Invalid algorithm '{alg}'. Use 'LINEAR_RAMP' or 'MEAN'.")

    _validate_track_name(track)
    _ensure_track_absent(track)

    all_intervs = gintervals_all()
    track_dir = _track_dir_for_create(track)
    created_new = not track_dir.exists()

    # Determine iterator: if None, infer from the expression (use track name as iterator)
    iter_val = iterator
    if iter_val is None:
        # Try to infer from expression - use expr as iterator policy
        # The C++ scanner will resolve the track's bin size
        iter_val = 0  # Let C++ determine from expression

    try:
        _pymisha.pm_smooth(
            track, str(expr), _df2pymisha(all_intervs),
            iter_val, float(winsize), float(weight_thr),
            int(bool(smooth_nans)), alg,
        )
        _pymisha.pm_dbreload()
        created_by = (
            f'gtrack.smooth({track}, description, {str(expr)}, {winsize}, '
            f'{weight_thr}, {smooth_nans}, {alg})'
        )
        _set_created_attrs(track, description, created_by)
        new_info = gtrack_info(track)
        if new_info.get("type") == "dense":
            gtrack_attr_set(track, "type", "dense")
            if "bin_size" in new_info:
                gtrack_attr_set(track, "binsize", str(int(new_info["bin_size"])))
        if _db_is_indexed(_shared._GROOT):
            gtrack_convert_to_indexed(track, remove_old=False)
        _pymisha.pm_dbreload()
    except Exception:
        if created_new and track_dir.exists():
            shutil.rmtree(track_dir, ignore_errors=True)
            _pymisha.pm_dbreload()
        raise


def gtrack_create(track, description, expr, iterator=None, band=None):
    """
    Create a track from a track expression.

    Creates a new track whose values are determined by evaluating *expr*
    over the entire genome. The type of the new track (Dense, Sparse, or
    Rectangles) is determined by the iterator policy. The description is
    stored as a track attribute.

    Parameters
    ----------
    track : str
        Name for the new track. Must start with a letter and contain
        only alphanumeric characters, underscores, and dots.
    description : str
        Human-readable description stored as a track attribute.
    expr : str
        Numeric track expression to evaluate.
    iterator : int or None, optional
        Fixed-bin iterator bin size. If None, the iterator is determined
        implicitly from the track expression.
    band : tuple or None, optional
        Track expression band. Currently not supported in pymisha and
        raises ``ValueError`` if provided.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track already exists, *expr* is None, or *band* is not
        None.

    See Also
    --------
    gtrack_create_sparse : Create a Sparse track from intervals/values.
    gtrack_create_dense : Create a Dense track from intervals/values.
    gtrack_2d_create : Create a 2D track.
    gtrack_smooth : Create a smoothed track.
    gtrack_modify : Modify an existing Dense track.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_create("mixed", "Test", "dense_track * 2", iterator=70)  # doctest: +SKIP
    >>> pm.gtrack_info("mixed")  # doctest: +SKIP
    >>> pm.gtrack_rm("mixed", force=True)  # doctest: +SKIP
    """
    if band is not None:
        raise ValueError("band is not supported in pymisha gtrack_create yet")
    from .intervals import gintervals_all

    _checkroot()
    if expr is None:
        raise ValueError("expr cannot be None")
    _validate_track_name(track)
    _ensure_track_absent(track)

    all_intervs = gintervals_all()
    track_dir = _track_dir_for_create(track)
    created_new = not track_dir.exists()
    try:
        _pymisha.pm_track_create_expr(track, str(expr), _df2pymisha(all_intervs), iterator, _shared.CONFIG)
        _pymisha.pm_dbreload()
        _set_created_attrs(
            track,
            description,
            f'gtrack.create("{track}", description, {str(expr)!r}, iterator={iterator!r})',
        )
        info = gtrack_info(track)
        if info.get("type") == "dense":
            gtrack_attr_set(track, "type", "dense")
            if "bin_size" in info:
                gtrack_attr_set(track, "binsize", str(int(info["bin_size"])))
        if _db_is_indexed(_shared._GROOT):
            gtrack_convert_to_indexed(track, remove_old=False)
        _pymisha.pm_dbreload()
    except Exception:
        if created_new and track_dir.exists():
            shutil.rmtree(track_dir, ignore_errors=True)
            _pymisha.pm_dbreload()
        raise


def _load_pssm_from_db(pssmset, pssmid):
    """Load a PSSM matrix from the database's pssms/ directory.

    Reads ``GROOT/pssms/{pssmset}.key`` and ``GROOT/pssms/{pssmset}.data``.

    Parameters
    ----------
    pssmset : str
        Name of the PSSM set (file basename without extension).
    pssmid : int
        Numeric ID of the PSSM within the set.

    Returns
    -------
    numpy.ndarray
        PSSM matrix of shape (L, 4) with columns A, C, G, T.
    """
    import numpy as np

    groot = _shared._GROOT
    if groot is None:
        raise ValueError("No genome database is initialized")

    pssm_dir = Path(groot) / "pssms"
    key_file = pssm_dir / f"{pssmset}.key"
    data_file = pssm_dir / f"{pssmset}.data"

    if not key_file.exists():
        raise FileNotFoundError(f"PSSM key file not found: {key_file}")
    if not data_file.exists():
        raise FileNotFoundError(f"PSSM data file not found: {data_file}")

    # Verify the pssmid exists in the key file
    pssmid = int(pssmid)
    found = False
    with open(key_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 1 and int(parts[0]) == pssmid:
                found = True
                break
    if not found:
        raise ValueError(f"PSSM id {pssmid} not found in {key_file}")

    # Read the data file and extract rows for our pssmid
    positions = {}
    with open(data_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            row_id = int(parts[0])
            if row_id != pssmid:
                continue
            pos = int(parts[1])
            a, c, g, t = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            positions[pos] = (a, c, g, t)

    if not positions:
        raise ValueError(f"No data found for PSSM id {pssmid} in {data_file}")

    max_pos = max(positions.keys())
    pssm = np.full((max_pos + 1, 4), 0.25)
    for pos, (a, c, g, t) in positions.items():
        pssm[pos] = [a, c, g, t]

    return pssm


def gtrack_create_pwm_energy(track, description, pssmset, pssmid, prior, iterator):
    """
    Create a track from a PSSM energy function.

    Creates a new Dense track with values of a PSSM energy function
    (log-sum-exp scoring). PSSM parameters are read from
    ``{pssmset}.key`` and ``{pssmset}.data`` files in ``GROOT/pssms/``.
    Internally creates a temporary PWM virtual track, extracts values at
    the given iterator resolution, and writes them to a new Dense track.

    Parameters
    ----------
    track : str
        Name for the new track.
    description : str
        Human-readable description stored as a track attribute.
    pssmset : str
        Name of PSSM set. Files ``{pssmset}.key`` and ``{pssmset}.data``
        must exist in ``GROOT/pssms/``.
    pssmid : int
        PSSM id within the set.
    prior : float
        Dirichlet prior for the PSSM.
    iterator : int
        Fixed-bin iterator bin size for the new track. Must be a
        positive integer.

    Raises
    ------
    ValueError
        If the track already exists, any required argument is None,
        *iterator* is not positive, or the PSSM set/id is not found.
    FileNotFoundError
        If the PSSM key or data file does not exist.

    Returns
    -------
    None

    See Also
    --------
    gtrack_create : Create a track from a general track expression.
    gtrack_create_dense : Create a Dense track from intervals/values.
    gtrack_smooth : Create a smoothed track.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_create_pwm_energy(  # doctest: +SKIP
    ...     "pwm_track", "Test", "pssm", 3, 0.01, iterator=100
    ... )
    >>> pm.gtrack_rm("pwm_track", force=True)  # doctest: +SKIP
    """
    from .extract import gextract
    from .intervals import gintervals_all
    from .vtracks import gvtrack_create, gvtrack_rm

    _checkroot()

    if track is None or description is None or pssmset is None or pssmid is None or prior is None or iterator is None:
        raise ValueError(
            "Usage: gtrack_create_pwm_energy(track, description, pssmset, pssmid, prior, iterator)"
        )

    _validate_track_name(track)
    _ensure_track_absent(track)

    iterator = int(iterator)
    if iterator <= 0:
        raise ValueError("iterator must be a positive integer")

    pssm = _load_pssm_from_db(pssmset, int(pssmid))

    # Create a temporary PWM virtual track and extract values
    vtrack_name = f"_pm_pwm_tmp_{track}"
    try:
        gvtrack_create(vtrack_name, None, func="pwm", pssm=pssm, prior=float(prior))
        all_intervs = gintervals_all()
        df = gextract(vtrack_name, intervals=all_intervs, iterator=iterator)
        if df is None or len(df) == 0:
            raise ValueError("No values extracted for PWM energy track")

        intervals = df[["chrom", "start", "end"]]
        values = df[vtrack_name].values

        gtrack_create_dense(
            track, description, intervals, values, iterator, defval=np.nan,
        )

        # Overwrite the created.by attribute to match R's format
        created_by = (
            f'gtrack.create_pwm_energy("{track}", description, '
            f'"{pssmset}", {int(pssmid)}, {float(prior)}, {iterator})'
        )
        # Bypass readonly check for internal overwrite
        attrs = _load_track_attributes(track)
        attrs["created.by"] = created_by
        _save_track_attributes(track, attrs)
    except Exception:
        # Clean up if track was partially created
        with contextlib.suppress(Exception):
            gtrack_rm(track, force=True)
        raise
    finally:
        with contextlib.suppress(Exception):
            gvtrack_rm(vtrack_name)


def gtrack_import(track, description, file, binsize=None, defval=np.nan, attrs=None):
    """
    Create a track from a WIG, BigWig, BedGraph, BED, or tab-delimited file.

    Parses the input file and creates either a Sparse or Dense track
    depending on *binsize*. File format is detected from the extension.
    Compressed files (``.gz``, ``.zip``) are supported for all formats
    except BigWig. Tab-delimited files must have a header with columns
    ``chrom``, ``start``, ``end``, and exactly one value column.

    Parameters
    ----------
    track : str
        Name for the new track.
    description : str
        Human-readable description stored as a track attribute.
    file : str
        Path to the input file. Supported extensions: ``.wig``,
        ``.bedgraph``, ``.bed``, ``.bw`` / ``.bigwig``, or tab-delimited
        (any other extension). May include ``.gz`` or ``.zip`` suffix.
    binsize : int or None, optional
        Bin size for a Dense track. If None or 0, a Sparse track is
        created. If positive, a Dense track with the given bin size is
        created.
    defval : float, default numpy.nan
        Default value for Dense track bins not covered by any interval.
        Ignored when creating Sparse tracks.
    attrs : dict or None, optional
        Additional attributes to set on the track after import, as a
        dict mapping attribute names to string values.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track already exists, *file* is None, or the file
        contains no valid intervals.

    See Also
    --------
    gtrack_import_set : Batch-import multiple files into tracks.
    gtrack_create_sparse : Create a Sparse track programmatically.
    gtrack_create_dense : Create a Dense track programmatically.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_import("wig_track", "From WIG", "data.wig", binsize=10)
    """
    _checkroot()
    _validate_track_name(track)
    _ensure_track_absent(track)
    if file is None:
        raise ValueError("file cannot be None")
    file_path = str(file)
    lower = file_path.lower()

    if lower.endswith((".bw.gz", ".bigwig.gz", ".bw.zip", ".bigwig.zip")):
        raise ValueError("Compressed BigWig is not supported; provide a plain .bw/.bigwig file")

    if lower.endswith((".bw", ".bigwig")):
        parsed = _parse_bigwig(file_path)
    elif lower.endswith((".bed", ".bed.gz", ".bed.zip")):
        parsed = _parse_bed(file_path)
    elif lower.endswith((".wig", ".wig.gz", ".wig.zip", ".bedgraph", ".bedgraph.gz", ".bedgraph.zip")):
        parsed = _parse_wig_or_bedgraph(file_path)
    else:
        parsed = _parse_tabular_track(file_path)

    if binsize is None:
        binsize = 0
    binsize = int(binsize)

    if binsize > 0:
        gtrack_create_dense(track, description, parsed[["chrom", "start", "end"]], parsed["value"], binsize, defval)
    else:
        gtrack_create_sparse(track, description, parsed[["chrom", "start", "end"]], parsed["value"])

    created_by = f'gtrack.import("{track}", description, "{file_path}", {binsize}, {float(defval):g}, attrs)'
    _set_created_attrs(track, description, created_by, attrs=attrs)


def _download_ftp_matches(path_pattern, tmpdir):
    parsed = urlparse(path_pattern)
    if parsed.scheme.lower() != "ftp":
        raise ValueError("Only ftp:// URLs are supported for remote import sets")
    host = parsed.hostname
    if not host:
        raise ValueError("Invalid FTP URL")
    allowed_hosts_env = os.environ.get("PYMISHA_ALLOWED_FTP_HOSTS", "").strip()
    if allowed_hosts_env:
        allowed_hosts = {h.strip() for h in allowed_hosts_env.split(",") if h.strip()}
        if host not in allowed_hosts:
            raise ValueError(
                f"FTP host '{host}' is not in PYMISHA_ALLOWED_FTP_HOSTS allow-list"
            )
    max_file_bytes = int(os.environ.get("PYMISHA_MAX_FTP_FILE_BYTES", str(512 * 1024 * 1024)))
    remote_path = parsed.path or "/"
    if "/" not in remote_path.strip("/"):
        remote_dir = "/"
        pattern = remote_path.lstrip("/")
    else:
        remote_dir = remote_path.rsplit("/", 1)[0] or "/"
        pattern = remote_path.rsplit("/", 1)[1]

    ftp = ftplib.FTP()
    ftp.connect(host, parsed.port or 21, timeout=30)
    ftp.login(parsed.username or "anonymous", parsed.password or "")
    try:
        names = ftp.nlst(remote_dir)
        matched = []
        for n in names:
            bn = os.path.basename(n)
            if fnmatch.fnmatch(bn, pattern):
                matched.append(n)

        out = []
        for remote in matched:
            remote_size = ftp.size(remote)
            if remote_size is None:
                raise ValueError(f"Could not determine FTP file size for '{remote}'")
            if remote_size > max_file_bytes:
                raise ValueError(
                    f"FTP file '{remote}' is too large ({remote_size} bytes > {max_file_bytes})"
                )
            local = Path(tmpdir) / os.path.basename(remote)
            with open(local, "wb") as f:
                ftp.retrbinary(f"RETR {remote}", f.write)
            out.append(str(local))
        return out
    finally:
        try:
            ftp.quit()
        except Exception:
            ftp.close()


def gtrack_import_set(description, path, binsize, track_prefix=None, defval=np.nan):
    """
    Create one or more tracks from multiple WIG/BedGraph/BigWig/tab files.

    Similar to `gtrack_import` but operates on multiple files at once.
    Files can be specified by a local glob pattern or an FTP URL with
    wildcards. Each file produces one track named
    ``{track_prefix}{filestem}``. Existing tracks are skipped. The
    function continues importing even if individual files fail.

    Parameters
    ----------
    description : str
        Human-readable description stored as a track attribute on
        every imported track.
    path : str
        Local file glob pattern (e.g., ``"/data/*.wig"``) or FTP URL
        (e.g., ``"ftp://host/path/*.wig.gz"``).
    binsize : int
        Bin size for Dense tracks. If 0, Sparse tracks are created.
    track_prefix : str or None, optional
        Prefix prepended to each track name derived from the filename
        stem. If None, no prefix is used.
    defval : float, default numpy.nan
        Default value for Dense track bins not covered by any interval.

    Returns
    -------
    dict
        Dictionary with keys ``"files_imported"`` (list of successfully
        imported filenames) and/or ``"files_failed"`` (list of filenames
        that failed to import).

    Raises
    ------
    ValueError
        If *description*, *path*, or *binsize* is None, or no files
        match the pattern.

    See Also
    --------
    gtrack_import : Import a single file into a track.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_import_set("Batch", "/data/*.wig", binsize=100, track_prefix="wigs.")
    """
    _checkroot()
    if description is None:
        raise ValueError("description cannot be None")
    if path is None:
        raise ValueError("path cannot be None")
    if binsize is None:
        raise ValueError("binsize cannot be None")

    track_prefix = "" if track_prefix is None else str(track_prefix)
    binsize = int(binsize)

    tmpdir = None
    files = []
    path_str = str(path)
    if path_str.lower().startswith("ftp://"):
        downloads_root = Path(_shared._GROOT) / "downloads"
        downloads_root.mkdir(parents=True, exist_ok=True)
        tmpdir = tempfile.mkdtemp(prefix="pymisha-import-set-", dir=str(downloads_root))
        files = _download_ftp_matches(path_str, tmpdir)
    else:
        files = glob.glob(path_str)

    files = [f for f in files if os.path.isfile(f)]
    if not files:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        raise ValueError("No files to import")

    imported = []
    failed = []
    try:
        for file in files:
            file_base = os.path.basename(file)
            stem = file_base.split(".", 1)[0]
            track_name = f"{track_prefix}{stem}"
            try:
                gtrack_import(track_name, description, file, binsize=binsize, defval=defval)
                imported.append(file_base)
            except Exception:
                failed.append(file_base)
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    result = {}
    if failed:
        result["files_failed"] = failed
    if imported:
        result["files_imported"] = imported
    return result


def _cleanup_empty_track_parents(track_dir, db_root):
    tracks_root = Path(db_root) / "tracks"
    cur = Path(track_dir).parent
    while cur != tracks_root and cur.exists():
        try:
            cur.rmdir()
        except OSError:
            break
        cur = cur.parent


def gtrack_mv(src, dest):
    """
    Rename or move a track within the same database.

    Renames a track or moves it to a different namespace (directory)
    within its source database. The track cannot be moved across
    databases; use `gtrack_copy` followed by `gtrack_rm` for that.

    Parameters
    ----------
    src : str
        Current track name.
    dest : str
        New track name.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If source and destination are identical, the source track does
        not exist, or the destination track already exists.

    See Also
    --------
    gtrack_copy : Copy a track (possibly across databases).
    gtrack_rm : Delete a track.
    gtrack_exists : Test whether a track exists.
    gtrack_ls : List available tracks.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_mv("old_name", "new_name")
    """
    _checkroot()
    _validate_track_name(src)
    _validate_track_name(dest)
    if src == dest:
        raise ValueError("Source and destination track names are the same")
    if not _track_exists(src):
        raise ValueError(f"Track '{src}' does not exist")
    if _track_exists(dest):
        raise ValueError(f"Track '{dest}' already exists")

    src_dir = Path(_pymisha.pm_track_path(src))
    src_db = gtrack_dataset(src)
    dest_dir = Path(src_db) / "tracks" / f"{dest.replace('.', '/')}.track"
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if dest_dir.exists():
        raise ValueError(f"Destination track directory already exists: {dest_dir}")

    try:
        src_dir.rename(dest_dir)
    except OSError:
        shutil.move(str(src_dir), str(dest_dir))

    _cleanup_empty_track_parents(src_dir, src_db)
    _pymisha.pm_dbreload()


def gtrack_copy(src, dest):
    """
    Create a copy of an existing track.

    Copies a track's on-disk directory to the current writable database
    root. The source track may reside in a different database when
    multiple databases are connected.

    Parameters
    ----------
    src : str
        Name of the source track.
    dest : str
        Name for the new copy.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If source and destination are identical, the source track does
        not exist, or the destination track already exists.

    See Also
    --------
    gtrack_mv : Rename / move a track within the same database.
    gtrack_rm : Delete a track.
    gtrack_exists : Test whether a track exists.
    gtrack_ls : List available tracks.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_copy("dense_track", "dense_track_copy")  # doctest: +SKIP
    >>> pm.gtrack_exists("dense_track_copy")  # doctest: +SKIP
    True
    >>> pm.gtrack_rm("dense_track_copy", force=True)  # doctest: +SKIP
    """
    _checkroot()
    _validate_track_name(src)
    _validate_track_name(dest)
    if src == dest:
        raise ValueError("Source and destination track names are the same")
    if not _track_exists(src):
        raise ValueError(f"Track '{src}' does not exist")
    if _track_exists(dest):
        raise ValueError(f"Track '{dest}' already exists")

    src_dir = Path(_pymisha.pm_track_path(src))
    dest_dir = _track_dir_for_create(dest)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if dest_dir.exists():
        raise ValueError(f"Destination track directory already exists: {dest_dir}")

    shutil.copytree(src_dir, dest_dir)
    _pymisha.pm_dbreload()


def gtrack_rm(track, force=False, db=None):
    """
    Remove a track from disk.

    Permanently deletes the track directory and all associated files
    (per-chromosome data, attributes, variables). Empty parent
    directories are cleaned up automatically.

    Parameters
    ----------
    track : str
        Name of the track to remove.
    force : bool, default False
        If True, suppress errors when the track does not exist and
        allow deletion without confirmation. If False, raises
        ``ValueError`` when the track is missing.
    db : str or None, optional
        Explicit database root path. If None, the track is located in
        the currently initialized databases.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track does not exist (when *force* is False) or if
        *force* is False (safety guard).

    See Also
    --------
    gtrack_ls : List available tracks.
    gtrack_exists : Test whether a track exists.
    gtrack_mv : Rename or move a track.
    gtrack_copy : Copy a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_rm("my_track", force=True)
    """
    _checkroot()
    _validate_track_name(track)

    if db is None:
        path = _pymisha.pm_track_path(track)
        if not path:
            if force:
                return
            raise ValueError(f"Track '{track}' does not exist")
        db_root = gtrack_dataset(track)
        track_dir = Path(path)
    else:
        db_root = str(Path(db))
        track_dir = Path(db_root) / "tracks" / f"{track.replace('.', '/')}.track"
        if not track_dir.exists():
            if force:
                return
            raise ValueError(f"Track '{track}' does not exist in database '{db_root}'")

    if not force:
        raise ValueError("Set force=True to delete a track")

    shutil.rmtree(track_dir, ignore_errors=True)
    _cleanup_empty_track_parents(track_dir, db_root)
    _pymisha.pm_dbreload()


def gtrack_import_mappedseq(
    track,
    description,
    file,
    pileup=0,
    binsize=-1,
    cols_order=(9, 11, 13, 14),
    remove_dups=True,
):
    """
    Import mapped sequences from SAM/tab-delimited text into a track.

    Reads aligned sequence data from a SAM file or a tab-delimited text
    file and creates either a Sparse (per-read) or Dense (pileup) track.
    Duplicate reads at the same position and strand can optionally be
    removed.

    Parameters
    ----------
    track : str
        Name for the new track.
    description : str
        Human-readable description stored as a track attribute.
    file : str
        Path to a SAM or tab-delimited text file.
    pileup : int, default 0
        If 0, create a Sparse track with one interval per mapped read.
        If positive, create a Dense pileup track where each bin stores
        the number of reads covering it. Reads are extended to this
        length from their start position.
    binsize : int, default -1
        Bin size for Dense (pileup) tracks. Required when *pileup* > 0.
        Must be -1 when *pileup* is 0.
    cols_order : tuple of int or None, default (9, 11, 13, 14)
        Column indices (1-based) for sequence, chromosome, coordinate,
        and strand in a tab-delimited file. Set to None for SAM format.
    remove_dups : bool, default True
        If True, remove duplicate reads at the same position and strand.

    Returns
    -------
    dict
        Dictionary with keys ``"total"`` (dict with ``"total"``,
        ``"total.mapped"``, ``"total.unmapped"``, ``"total.dups"``) and
        ``"chrom"`` (pandas.DataFrame with per-chromosome mapping stats).

    Raises
    ------
    ValueError
        If the track already exists, *file* is None, column indices are
        invalid, or pileup/binsize combination is inconsistent.

    See Also
    --------
    gtrack_import : Import from WIG/BedGraph/BED/BigWig files.
    gtrack_create_sparse : Create a Sparse track from intervals.
    gtrack_create_dense : Create a Dense track from intervals.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_import_mappedseq("reads", "Test", "reads.sam")
    """
    _checkroot()
    _validate_track_name(track)
    _ensure_track_absent(track)
    if file is None:
        raise ValueError("file cannot be None")

    pileup = int(pileup)
    binsize = int(binsize)
    if pileup < 0:
        raise ValueError("pileup cannot be negative")
    if pileup == 0 and binsize >= 0:
        raise ValueError("For pileup=0 (sparse), binsize must be -1")
    if pileup > 0 and binsize <= 0:
        raise ValueError("For pileup>0 (dense), binsize must be > 0")

    is_sam = cols_order is None
    if not is_sam:
        if len(cols_order) != 4:
            raise ValueError("cols_order must have 4 entries: sequence, chromosome, coordinate, strand")
        cols_order = [int(x) for x in cols_order]
        if min(cols_order) <= 0:
            raise ValueError("cols_order indices are 1-based and must be positive")
        if len(set(cols_order)) != 4:
            raise ValueError("cols_order entries must be unique")

    from .intervals import gintervals_all

    chrom_sizes_df = gintervals_all()
    chrom_sizes = {c: int(e) for c, e in zip(chrom_sizes_df["chrom"], chrom_sizes_df["end"], strict=False)}
    chroms = list(chrom_sizes.keys())
    nchrom = len(chroms)
    idx_by_chrom = {c: i for i, c in enumerate(chroms)}
    mapped = np.zeros(nchrom, dtype=np.int64)
    dups = np.zeros(nchrom, dtype=np.int64)
    total_unmapped = 0
    plus = [[] for _ in range(nchrom)]
    minus = [[] for _ in range(nchrom)]

    path = str(file)
    if not os.path.exists(path):
        raise ValueError(f"File not found: {path}")

    stream = _open_text_auto(path)
    try:
        for raw in stream:
            line = raw.strip()
            if not line:
                continue
            if is_sam and line.startswith("@"):
                continue

            fields = line.split("\t")
            if len(fields) == 1:
                fields = line.split()

            try:
                if is_sam:
                    seq = fields[9]
                    chrom = fields[2]
                    coord = int(fields[3])
                    flag = int(fields[1], 0)
                    strand = "-" if (flag & 0x10) else "+"
                else:
                    seq = fields[cols_order[0] - 1]
                    chrom = fields[cols_order[1] - 1]
                    coord = int(fields[cols_order[2] - 1])
                    strand = fields[cols_order[3] - 1]
            except Exception:
                total_unmapped += 1
                continue

            try:
                chrom = _pymisha.pm_normalize_chroms([chrom])[0]
            except Exception:
                total_unmapped += 1
                continue

            if chrom not in chrom_sizes:
                total_unmapped += 1
                continue
            chrom_len = chrom_sizes[chrom]
            if coord < 0 or coord >= chrom_len:
                total_unmapped += 1
                continue

            ci = idx_by_chrom[chrom]
            mapped[ci] += 1
            if strand in ("+", "F"):
                plus[ci].append(coord)
            elif strand in ("-", "R"):
                minus[ci].append(coord + len(seq))
            else:
                mapped[ci] -= 1
                total_unmapped += 1
                continue
    finally:
        _close_text_auto(stream)

    if pileup > 0:
        dense_rows = {"chrom": [], "start": [], "end": [], "value": []}
        for ci, chrom in enumerate(chroms):
            chrom_len = chrom_sizes[chrom]
            nbins = int(np.ceil(chrom_len / binsize))
            vals = np.zeros(nbins, dtype=np.float64)

            for strand_idx, coords in enumerate((plus[ci], minus[ci])):
                coords.sort()
                prev = None
                for c in coords:
                    if remove_dups and prev is not None and c == prev:
                        dups[ci] += 1
                        continue
                    prev = c
                    if strand_idx == 0:
                        from_coord = max(c, 0)
                        to_coord = min(c + pileup, chrom_len)
                    else:
                        from_coord = max(c - pileup, 0)
                        to_coord = min(c, chrom_len)
                    if to_coord <= from_coord:
                        continue

                    fb = int(from_coord // binsize)
                    tb = int(np.ceil(to_coord / binsize) - 1)
                    if fb >= tb:
                        vals[fb] += (to_coord - from_coord) / binsize
                    else:
                        vals[fb] += (fb + 1) - (from_coord / binsize)
                        vals[tb] += (to_coord / binsize) - tb
                        if tb > fb + 1:
                            vals[fb + 1:tb] += 1.0

            for b in range(nbins):
                start = b * binsize
                end = min((b + 1) * binsize, chrom_len)
                dense_rows["chrom"].append(chrom)
                dense_rows["start"].append(start)
                dense_rows["end"].append(end)
                dense_rows["value"].append(float(vals[b]))

        ddf = pd.DataFrame(dense_rows)
        gtrack_create_dense(track, description, ddf[["chrom", "start", "end"]], ddf["value"], binsize, np.nan)
    else:
        sparse_rows = {"chrom": [], "start": [], "end": [], "value": []}
        for ci, chrom in enumerate(chroms):
            p = sorted(plus[ci])
            m = sorted(minus[ci])
            i = j = 0
            while i < len(p) or j < len(m):
                val = 0.0
                coord = None
                if i < len(p) and (j >= len(m) or m[j] >= p[i]):
                    coord = p[i]
                    val = max(val + (0.0 if remove_dups else 1.0), 1.0)
                    i += 1
                    while i < len(p) and p[i] == coord:
                        dups[ci] += 1
                        if not remove_dups:
                            val += 1.0
                        i += 1
                if j < len(m) and (coord is None or m[j] == coord):
                    coord = m[j]
                    val = max(val + (0.0 if remove_dups else 1.0), 1.0)
                    j += 1
                    while j < len(m) and m[j] == coord:
                        dups[ci] += 1
                        if not remove_dups:
                            val += 1.0
                        j += 1
                if coord is None:
                    continue
                sparse_rows["chrom"].append(chrom)
                sparse_rows["start"].append(coord)
                sparse_rows["end"].append(coord + 1)
                sparse_rows["value"].append(val)

        sdf = pd.DataFrame(sparse_rows)
        gtrack_create_sparse(track, description, sdf[["chrom", "start", "end"]], sdf["value"])

    created_by = (
        f'gtrack.import_mappedseq("{track}", description, "{path}", '
        f"pileup={pileup}, binsize={binsize}, remove.dups={bool(remove_dups)})"
    )
    _set_created_attrs(track, description, created_by)

    chrom_stat = pd.DataFrame({"chrom": chroms, "mapped": mapped.astype(float), "dups": dups.astype(float)})
    total_mapped = int(mapped.sum())
    total_dups = int(dups.sum())
    total = {
        "total": float(total_mapped + total_unmapped + total_dups),
        "total.mapped": float(total_mapped),
        "total.unmapped": float(total_unmapped),
        "total.dups": float(total_dups),
    }
    return {"total": total, "chrom": chrom_stat}


def gtrack_exists(track):
    """
    Test for track existence in the Genomic Database.

    Parameters
    ----------
    track : str
        Track name to check.

    Returns
    -------
    bool
        True if the track exists, False otherwise.

    Raises
    ------
    ValueError
        If track is None.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_exists("dense_track")
    True
    >>> pm.gtrack_exists("nonexistent_track")
    False

    See Also
    --------
    gtrack_ls : List available tracks.
    gtrack_info : Get metadata for a track.
    gtrack_rm : Delete a track.
    """
    if track is None:
        raise ValueError("track cannot be None")

    _checkroot()

    if track == "":
        return False

    return _track_exists(track)


def gtrack_attr_get(track, attr):
    """
    Get a single track attribute value.

    Parameters
    ----------
    track : str
        Track name.
    attr : str
        Attribute name.

    Returns
    -------
    str
        Attribute value, or empty string if attribute doesn't exist.

    Raises
    ------
    ValueError
        If track does not exist.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_attr_get("sparse_track", "created.by")  # doctest: +SKIP
    '...'

    See Also
    --------
    gtrack_attr_set : Set a track attribute.
    gtrack_attr_export : Export attributes for multiple tracks.
    gtrack_attr_import : Batch-import attributes from a table.
    """
    if track is None:
        raise ValueError("track cannot be None")
    if attr is None:
        raise ValueError("attr cannot be None")

    _checkroot()

    if not _track_exists(track):
        raise ValueError(f"Track '{track}' does not exist")

    attrs = _load_track_attributes(track)
    return attrs.get(attr, "")


def gtrack_convert_to_indexed(track, remove_old=False):
    """
    Convert a per-chromosome track to indexed format.

    Reads the per-chromosome binary files and writes a unified
    ``track.idx`` / ``track.dat`` pair. Optionally removes the original
    per-chromosome files after conversion.

    Parameters
    ----------
    track : str
        Name of the track to convert.
    remove_old : bool, default False
        If True, remove the original per-chromosome files after
        successful conversion.

    Returns
    -------
    None

    See Also
    --------
    gtrack_create_empty_indexed : Create empty indexed files.
    gdb_convert_to_indexed : Convert an entire database.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_convert_to_indexed("my_track", remove_old=True)
    """
    if track is None:
        raise ValueError("track cannot be None")
    _checkroot()
    return _pymisha.pm_track_convert_to_indexed(track, bool(remove_old))


def gtrack_create_empty_indexed(track):
    """
    Create empty indexed files for an existing track directory.

    Writes an empty ``track.idx`` and ``track.dat`` pair in the track
    directory. Useful when the track has no data yet but indexed format
    is required by the database.

    Parameters
    ----------
    track : str
        Name of an existing track whose directory should receive the
        indexed files.

    Returns
    -------
    None

    See Also
    --------
    gtrack_convert_to_indexed : Convert per-chromosome files to indexed.
    gdb_convert_to_indexed : Convert an entire database.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_create_empty_indexed("my_track")
    """
    if track is None:
        raise ValueError("track cannot be None")
    _checkroot()
    return _pymisha.pm_track_create_empty_indexed(track)


def gtrack_attr_set(track, attr, value):
    """
    Set a track attribute value.

    Parameters
    ----------
    track : str
        Track name.
    attr : str
        Attribute name.
    value : str
        Attribute value. Set to empty string "" to remove the attribute.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If track does not exist or the attribute is read-only.

    See Also
    --------
    gtrack_attr_get : Read a single track attribute.
    gtrack_attr_export : Export attributes for multiple tracks.
    gtrack_attr_import : Batch-import attributes from a table.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_attr_set("sparse_track", "test_attr", "test_value")  # doctest: +SKIP
    >>> pm.gtrack_attr_get("sparse_track", "test_attr")  # doctest: +SKIP
    'test_value'
    >>> pm.gtrack_attr_set("sparse_track", "test_attr", "")  # doctest: +SKIP
    """
    if track is None:
        raise ValueError("track cannot be None")
    if attr is None:
        raise ValueError("attr cannot be None")
    if value is None:
        raise ValueError("value cannot be None")

    _checkroot()

    if not _track_exists(track):
        raise ValueError(f"Track '{track}' does not exist")

    from .db_attrs import gdb_get_readonly_attrs
    readonly_attrs = set(gdb_get_readonly_attrs() or [])
    if attr in readonly_attrs:
        raise ValueError(f"Attribute '{attr}' is read-only")

    # Load existing attributes
    attrs = _load_track_attributes(track)

    # Set or remove attribute
    if value == "":
        if attr in attrs:
            del attrs[attr]
    else:
        attrs[attr] = str(value)

    # Save back
    _save_track_attributes(track, attrs)


def gtrack_attr_import(table, remove_others=False):
    """
    Bulk import track attributes from a DataFrame.

    Parameters
    ----------
    table : DataFrame
        DataFrame with track names as index and attribute names as columns.
        Values are converted to strings. Empty string values are skipped
        (attribute not set for that track).
    remove_others : bool, default False
        If True, remove all non-readonly attributes not present in the table
        for tracks listed in the table.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If table is empty, any track in the index does not exist, or
        any attribute is read-only.

    See Also
    --------
    gtrack_attr_export : Export attributes to a DataFrame.
    gtrack_attr_get : Read a single attribute.
    gtrack_attr_set : Set a single attribute.

    Examples
    --------
    >>> import pymisha as pm
    >>> import pandas as pd
    >>> _ = pm.gdb_init_examples()
    >>> tbl = pd.DataFrame({"description": ["test"]}, index=["dense_track"])
    >>> pm.gtrack_attr_import(tbl)  # doctest: +SKIP
    """
    import pandas as pd

    _checkroot()

    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError("Invalid format of attributes table")

    tracks = list(table.index)
    attrs = list(table.columns)

    if not tracks or not attrs:
        raise ValueError("Invalid format of attributes table")

    seen_tracks = set()
    for track in tracks:
        if track in seen_tracks:
            raise ValueError(f"Track '{track}' appears more than once")
        seen_tracks.add(track)

    seen_attrs = set()
    for attr in attrs:
        if attr in seen_attrs:
            raise ValueError(f"Attribute '{attr}' appears more than once")
        seen_attrs.add(attr)

    if any((not isinstance(attr, str)) or attr == "" for attr in attrs):
        raise ValueError("Invalid format of attributes table")

    # Validate all tracks exist
    for track in tracks:
        if not _track_exists(track):
            raise ValueError(f"Track '{track}' does not exist")

    from .db_attrs import gdb_get_readonly_attrs
    readonly_attrs = set(gdb_get_readonly_attrs() or [])
    for attr in attrs:
        if attr in readonly_attrs:
            raise ValueError(f"Attribute '{attr}' is read-only")

    # Convert all values to strings
    table = table.astype(str)

    for track in tracks:
        existing_attrs = _load_track_attributes(track)

        if remove_others:
            # Remove attrs not in table columns, but keep readonly attributes.
            new_attrs = {k: v for k, v in existing_attrs.items() if k in readonly_attrs}
        else:
            new_attrs = dict(existing_attrs)

        for attr in attrs:
            val = table.at[track, attr]
            if val != "" and val != "nan":
                new_attrs[attr] = val

        _save_track_attributes(track, new_attrs)


def gtrack_attr_export(tracks=None, attrs=None):
    """
    Export track attributes as a DataFrame.

    Parameters
    ----------
    tracks : list of str, optional
        List of track names. If None, all tracks.
    attrs : list of str, optional
        List of attribute names to include. If None, all attributes.

    Returns
    -------
    DataFrame
        DataFrame with tracks as rows and attributes as columns.

    Raises
    ------
    ValueError
        If any specified track does not exist.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_attr_export()  # doctest: +SKIP
    >>> pm.gtrack_attr_export(tracks=["sparse_track", "dense_track"])  # doctest: +SKIP
    >>> pm.gtrack_attr_export(attrs=["created.by"])  # doctest: +SKIP

    See Also
    --------
    gtrack_attr_import : Batch-import attributes from a DataFrame.
    gtrack_attr_get : Read a single attribute.
    gtrack_attr_set : Set a single attribute.
    """
    import pandas as pd

    _checkroot()

    # Get list of tracks
    if tracks is None:
        tracks = gtrack_ls()
        if tracks is None:
            tracks = []
    else:
        # Validate tracks exist
        for track in tracks:
            if not _track_exists(track):
                raise ValueError(f"Track '{track}' does not exist")

    # Collect all attributes
    all_attrs = {}  # track -> {attr: value}
    all_attr_names = set()

    for track in tracks:
        track_attrs = _load_track_attributes(track)
        all_attrs[track] = track_attrs
        all_attr_names.update(track_attrs.keys())

    # Filter attributes if specified
    if attrs is not None:
        all_attr_names = set(attrs)

    # Sort attribute names by popularity (number of tracks having this attr)
    attr_counts = {}
    for attr_name in all_attr_names:
        count = sum(1 for t in tracks if attr_name in all_attrs.get(t, {}))
        attr_counts[attr_name] = count

    sorted_attrs = sorted(all_attr_names, key=lambda a: (-attr_counts[a], a))

    # Build DataFrame
    data = {}
    for attr_name in sorted_attrs:
        data[attr_name] = [all_attrs.get(t, {}).get(attr_name, "") for t in tracks]

    return pd.DataFrame(data, index=tracks)


# ---------------------------------------------------------------------------
#  Track variables  (gtrack.var.* in R)
# ---------------------------------------------------------------------------

def _track_var_dir(track_name):
    """Return the path to the vars/ directory for a track, creating it if needed."""
    track_path = _pymisha.pm_track_path(track_name)
    if not track_path:
        raise ValueError(f"Track '{track_name}' does not exist")
    return os.path.join(track_path, "vars")


def gtrack_var_ls(track, pattern=""):
    """
    List track variables.

    Parameters
    ----------
    track : str
        Track name.
    pattern : str, optional
        Regex pattern to filter variable names. Default ``""`` matches all.

    Returns
    -------
    list of str
        Sorted list of variable names matching the pattern.

    Raises
    ------
    ValueError
        If track does not exist.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> pm.gtrack_var_ls("dense_track")
    []

    See Also
    --------
    gtrack_var_get : Read a variable's value.
    gtrack_var_set : Store a variable.
    gtrack_var_rm : Delete a variable.
    """
    _checkroot()

    if not _track_exists(track):
        raise ValueError(f"Track '{track}' does not exist")

    var_dir = _track_var_dir(track)
    if not os.path.isdir(var_dir):
        return []

    files = os.listdir(var_dir)
    if not files:
        return []

    if pattern:
        files = [f for f in files if re.search(pattern, f)]

    return sorted(files)


def gtrack_var_get(track, var):
    """
    Get the value of a track variable.

    Parameters
    ----------
    track : str
        Track name.
    var : str
        Variable name.

    Returns
    -------
    object
        The stored Python object.

    Raises
    ------
    ValueError
        If the track or variable does not exist.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_var_get("dense_track", "my_var")

    See Also
    --------
    gtrack_var_set : Store a variable.
    gtrack_var_ls : List variables for a track.
    gtrack_var_rm : Delete a variable.
    """
    import pickle

    _checkroot()

    if not _track_exists(track):
        raise ValueError(f"Track '{track}' does not exist")
    _validate_track_var_name(var)

    var_dir = _track_var_dir(track)
    filepath = os.path.join(var_dir, var)

    if not os.path.exists(filepath):
        raise ValueError(
            f"Variable '{var}' does not exist for track '{track}'"
        )

    # Try pickle (pymisha-native) first
    try:
        with open(filepath, "rb") as f:
            return restricted_load(f)
    except (pickle.UnpicklingError, EOFError, ModuleNotFoundError):
        pass

    # Fallback: try R serialized format via pyreadr
    try:
        import pyreadr
        result = pyreadr.read_rds(filepath)
        if isinstance(result, pd.DataFrame):
            # Single column → return as array
            if result.shape[1] == 1:
                return result.iloc[:, 0].values
            return result
        return result
    except Exception:
        pass

    raise ValueError(
        f"Cannot read variable '{var}' for track '{track}': unknown or unsafe format"
    )


def gtrack_var_set(track, var, value):
    """
    Set the value of a track variable.

    Parameters
    ----------
    track : str
        Track name.
    var : str
        Variable name.
    value : object
        Value to store. Can be any pickle-able Python object.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track does not exist.

    See Also
    --------
    gtrack_var_get : Read a variable's value.
    gtrack_var_ls : List variables for a track.
    gtrack_var_rm : Delete a variable.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_var_set("dense_track", "my_var", [1, 2, 3])
    """
    import pickle

    _checkroot()

    if not _track_exists(track):
        raise ValueError(f"Track '{track}' does not exist")
    _validate_track_var_name(var)

    var_dir = _track_var_dir(track)
    os.makedirs(var_dir, exist_ok=True)

    filepath = os.path.join(var_dir, var)
    try:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise TypeError("value is not serializable") from exc
    try:
        restricted_loads(payload)
    except pickle.UnpicklingError as exc:
        raise TypeError(
            "value contains unsupported objects for secure track-variable serialization"
        ) from exc
    with open(filepath, "wb") as f:
        f.write(payload)


def gtrack_var_rm(track, var):
    """
    Remove a track variable.

    Parameters
    ----------
    track : str
        Track name.
    var : str
        Variable name to remove.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track does not exist.

    See Also
    --------
    gtrack_var_set : Store a variable.
    gtrack_var_get : Read a variable's value.
    gtrack_var_ls : List variables for a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_var_rm("dense_track", "my_var")
    """
    _checkroot()

    if not _track_exists(track):
        raise ValueError(f"Track '{track}' does not exist")
    _validate_track_var_name(var)

    var_dir = _track_var_dir(track)
    filepath = os.path.join(var_dir, var)

    if os.path.exists(filepath):
        os.remove(filepath)


# ---------------------------------------------------------------------------
# 2D track creation
# ---------------------------------------------------------------------------


def _normalize_2d_intervals(intervals):
    """
    Validate and normalize a 2D intervals DataFrame.

    Returns a copy with columns: chrom1, start1, end1, chrom2, start2, end2.
    Chromosome names are normalized via pm_normalize_chroms.
    """
    if not isinstance(intervals, pd.DataFrame):
        raise ValueError("intervals must be a DataFrame")
    required = {"chrom1", "start1", "end1", "chrom2", "start2", "end2"}
    if not required.issubset(intervals.columns):
        raise ValueError(f"intervals must contain columns: {', '.join(sorted(required))}")

    out = intervals[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].copy()
    out["chrom1"] = out["chrom1"].astype(str)
    out["chrom2"] = out["chrom2"].astype(str)
    for col in ("start1", "end1", "start2", "end2"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.int64)

    # Normalize chromosome names
    all_chroms = list(set(out["chrom1"].tolist() + out["chrom2"].tolist()))
    try:
        norm = _pymisha.pm_normalize_chroms(all_chroms)
        cmap = dict(zip(all_chroms, norm, strict=False))
    except Exception:
        cmap = {c: c for c in all_chroms}
    out["chrom1"] = out["chrom1"].map(cmap)
    out["chrom2"] = out["chrom2"].map(cmap)

    # Filter to known chroms
    from .intervals import gintervals_all
    chrom_sizes_df = gintervals_all()
    known = set(chrom_sizes_df["chrom"].astype(str).tolist())
    mask = out["chrom1"].isin(known) & out["chrom2"].isin(known)
    out = out[mask].reset_index(drop=True)
    return out, chrom_sizes_df


def _detect_points_vs_rects(intervals):
    """
    Detect if all intervals are unit-sized (points) or general rectangles.

    Returns True for points, False for rectangles.
    """
    widths1 = intervals["end1"] - intervals["start1"]
    widths2 = intervals["end2"] - intervals["start2"]
    return bool((widths1 == 1).all() and (widths2 == 1).all())


def gtrack_2d_create(track, description, intervals, values):
    """
    Create a 2D track from intervals and values.

    Parameters
    ----------
    track : str
        Track name (dot-separated namespace).
    description : str
        Track description.
    intervals : DataFrame
        2D intervals with columns: chrom1, start1, end1, chrom2, start2, end2.
    values : array-like
        Numeric values, one per interval.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track already exists, values length does not match
        intervals, no valid intervals remain after normalization, or
        overlapping intervals are detected within the same chromosome pair.

    See Also
    --------
    gtrack_2d_import : Create a 2D track from a file.
    gtrack_2d_import_contacts : Import HiC contact data as a 2D track.
    gtrack_create_sparse : Create a 1D Sparse track.
    gtrack_rm : Delete a track.

    Notes
    -----
    Automatically detects POINTS vs RECTS format based on interval sizes.
    All unit-size intervals (end-start==1) produce a POINTS track.
    Overlapping intervals within the same chromosome pair raise an error.

    Examples
    --------
    >>> import pymisha as pm
    >>> import pandas as pd
    >>> _ = pm.gdb_init_examples()
    >>> ivs = pd.DataFrame({
    ...     "chrom1": ["1"], "start1": [0], "end1": [100],
    ...     "chrom2": ["1"], "start2": [200], "end2": [300],
    ... })
    >>> pm.gtrack_2d_create("test_2d", "Test", ivs, [1.0])  # doctest: +SKIP
    >>> pm.gtrack_rm("test_2d", force=True)  # doctest: +SKIP
    """
    from ._quadtree import verify_no_overlaps_2d, write_2d_track_file

    _checkroot()
    _validate_track_name(track)
    _ensure_track_absent(track)

    intervals_df, chrom_sizes_df = _normalize_2d_intervals(intervals)
    values_arr = np.asarray(values, dtype=np.float32)
    if len(values_arr) != len(intervals_df):
        raise ValueError(
            f"Number of values ({len(values_arr)}) must match number of "
            f"intervals ({len(intervals_df)})"
        )

    if len(intervals_df) == 0:
        raise ValueError("No valid intervals after normalization")

    is_points = _detect_points_vs_rects(intervals_df)

    # Build chrom size lookup
    chrom_size = {
        str(c): int(e)
        for c, e in zip(chrom_sizes_df["chrom"], chrom_sizes_df["end"], strict=False)
    }

    # Sort by (chrom1, chrom2, start1, start2) — same as R
    intervals_df["_orig_idx"] = np.arange(len(intervals_df))
    intervals_df = intervals_df.sort_values(
        ["chrom1", "chrom2", "start1", "start2"]
    ).reset_index(drop=True)
    orig_idx = intervals_df["_orig_idx"].values
    intervals_df = intervals_df.drop(columns=["_orig_idx"])

    # Create track directory
    track_dir = _track_dir_for_create(track)
    created_new = not track_dir.exists()
    track_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Group by chromosome pair and write per-pair files
        for (c1, c2), group in intervals_df.groupby(["chrom1", "chrom2"]):
            cs1 = chrom_size.get(str(c1))
            cs2 = chrom_size.get(str(c2))
            if cs1 is None or cs2 is None:
                continue

            arena = (0, 0, cs1, cs2)

            # Collect objects and check overlaps
            if is_points:
                objs = []
                for _, row in group.iterrows():
                    oi = row.name  # position in sorted df
                    objs.append((int(row["start1"]), int(row["start2"]),
                                 float(values_arr[orig_idx[oi]])))
                # Points can't overlap (they're 1x1)
            else:
                rects_for_check = []
                objs = []
                for _, row in group.iterrows():
                    oi = row.name
                    r = (int(row["start1"]), int(row["start2"]),
                         int(row["end1"]), int(row["end2"]))
                    rects_for_check.append(r)
                    objs.append((int(row["start1"]), int(row["start2"]),
                                 int(row["end1"]), int(row["end2"]),
                                 float(values_arr[orig_idx[oi]])))
                verify_no_overlaps_2d(rects_for_check)

            filename = os.path.join(str(track_dir), f"{c1}-{c2}")
            write_2d_track_file(filename, objs, arena, is_points=is_points)

        _pymisha.pm_dbreload()
        _set_created_attrs(
            track,
            description,
            f'gtrack.2d.create("{track}", description, intervals, values)',
        )
    except Exception:
        if created_new and track_dir.exists():
            shutil.rmtree(track_dir, ignore_errors=True)
        _pymisha.pm_dbreload()
        raise


def gtrack_2d_import(track, description, file):
    """
    Import a 2D track from a tab-delimited file.

    Parameters
    ----------
    track : str
        Track name.
    description : str
        Track description.
    file : str
        Path to tab-delimited file with header:
        chrom1, start1, end1, chrom2, start2, end2, <value_column>

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track already exists, file is not found, or the file has
        fewer than 7 columns.

    See Also
    --------
    gtrack_2d_create : Create a 2D track from a DataFrame.
    gtrack_2d_import_contacts : Import HiC contacts as a 2D track.
    gtrack_rm : Delete a track.

    Notes
    -----
    The value column is the 7th column (0-indexed: column 6).
    Automatically detects POINTS vs RECTS format.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_2d_import("test_2d", "Test", "contacts.tsv")
    """
    _checkroot()
    _validate_track_name(track)
    _ensure_track_absent(track)

    if not os.path.exists(file):
        raise ValueError(f"File not found: {file}")

    df = pd.read_csv(file, sep="\t")
    if len(df.columns) < 7:
        raise ValueError(
            "File must have at least 7 columns: "
            "chrom1, start1, end1, chrom2, start2, end2, value"
        )

    # Use first 6 columns as interval coords, 7th as value
    cols = list(df.columns)
    df = df.rename(columns={
        cols[0]: "chrom1", cols[1]: "start1", cols[2]: "end1",
        cols[3]: "chrom2", cols[4]: "start2", cols[5]: "end2",
    })
    value_col = cols[6]
    values = df[value_col].values

    gtrack_2d_create(track, description, df, values)


def gtrack_2d_import_contacts(
    track, description, contacts, fends=None, allow_duplicates=True
):
    """
    Create a 2D Points track from inter-genomic contacts.

    Parameters
    ----------
    track : str
        Track name (dot-separated namespace).
    description : str
        Track description.
    contacts : list of str
        Paths to contact files. If ``fends`` is None the files must be in
        "intervals-value" tab-separated format (columns: chrom1, start1, end1,
        chrom2, start2, end2, <value>).  Otherwise they must be in
        "fends-value" format (columns: fend1, fend2, count).
    fends : str or None
        Path to a fragment-ends file with columns: fend, chr, coord.
    allow_duplicates : bool, default True
        If True, duplicate contacts (same midpoint pair) are summed.
        If False, duplicates raise ``ValueError``.

    Notes
    -----
    * Intervals are converted to midpoints: X = (start1+end1)//2,
      Y = (start2+end2)//2.
    * Contacts are canonically ordered: if chrom2 < chrom1 (or same chrom
      and coord2 < coord1) the two sides are swapped.
    * Cis contacts (same chromosome) are mirrored: both (X,Y) and (Y,X)
      are stored unless X == Y.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the track already exists, no contact files are provided, or
        duplicates are found when *allow_duplicates* is False.

    See Also
    --------
    gtrack_2d_create : Create a 2D track from a DataFrame.
    gtrack_2d_import : Import a 2D track from a tab-delimited file.
    gtrack_rm : Delete a track.

    Examples
    --------
    >>> import pymisha as pm
    >>> _ = pm.gdb_init_examples()
    >>> # pm.gtrack_2d_import_contacts("hic", "HiC", ["contacts.tsv"])
    """
    _checkroot()
    _validate_track_name(track)
    _ensure_track_absent(track)

    if not contacts:
        raise ValueError(
            "Usage: gtrack_2d_import_contacts(track, description, contacts, "
            "fends=None, allow_duplicates=True)"
        )

    # ------------------------------------------------------------------
    # 1. Load fends mapping (if provided)
    # ------------------------------------------------------------------
    fend_map = None  # fend_id -> (chrom, coord)
    if fends is not None:
        if not os.path.exists(fends):
            raise ValueError(f"Fends file not found: {fends}")
        fdf = pd.read_csv(fends, sep="\t")
        required_fend_cols = {"fend", "chr", "coord"}
        if not required_fend_cols.issubset(fdf.columns):
            raise ValueError(
                "Fends file must contain columns: fend, chr, coord"
            )
        fend_map = {}
        fdf["chr"] = fdf["chr"].astype(str).str.replace(r'\.0$', '', regex=True)
        for _, row in fdf.iterrows():
            fend_map[int(row["fend"])] = (str(row["chr"]), int(row["coord"]))

    # ------------------------------------------------------------------
    # 2. Read contact files and build a list of (chrom1, mid1, chrom2, mid2, value)
    # ------------------------------------------------------------------
    records = []  # list of (chrom1_str, mid1, chrom2_str, mid2, value)

    for cfile in contacts:
        if not os.path.exists(cfile):
            raise ValueError(f"Contacts file not found: {cfile}")
        df = pd.read_csv(cfile, sep="\t")

        if fend_map is not None:
            # fends-value format: fend1, fend2, count
            required_cols = {"fend1", "fend2", "count"}
            if not required_cols.issubset(df.columns):
                raise ValueError(
                    "Contacts file (fends mode) must contain columns: "
                    "fend1, fend2, count"
                )
            for _, row in df.iterrows():
                f1 = int(row["fend1"])
                f2 = int(row["fend2"])
                if f1 not in fend_map:
                    raise ValueError(f"Unknown fend id: {f1}")
                if f2 not in fend_map:
                    raise ValueError(f"Unknown fend id: {f2}")
                c1, coord1 = fend_map[f1]
                c2, coord2 = fend_map[f2]
                records.append((c1, coord1, c2, coord2, float(row["count"])))
        else:
            # intervals-value format
            if len(df.columns) < 7:
                raise ValueError(
                    "Contacts file must have at least 7 columns: "
                    "chrom1, start1, end1, chrom2, start2, end2, value"
                )
            cols = list(df.columns)
            df = df.rename(columns={
                cols[0]: "chrom1", cols[1]: "start1", cols[2]: "end1",
                cols[3]: "chrom2", cols[4]: "start2", cols[5]: "end2",
            })
            value_col = cols[6]
            # Ensure chrom columns are strings (pandas may parse '1' as int/float)
            df["chrom1"] = df["chrom1"].astype(str).str.replace(r'\.0$', '', regex=True)
            df["chrom2"] = df["chrom2"].astype(str).str.replace(r'\.0$', '', regex=True)
            for _, row in df.iterrows():
                mid1 = int((int(row["start1"]) + int(row["end1"])) // 2)
                mid2 = int((int(row["start2"]) + int(row["end2"])) // 2)
                records.append(
                    (str(row["chrom1"]), mid1, str(row["chrom2"]), mid2,
                     float(row[value_col]))
                )

    if not records:
        raise ValueError("No contacts found in the provided files")

    # ------------------------------------------------------------------
    # 3. Normalize chromosome names
    # ------------------------------------------------------------------
    all_chroms_raw = list(
        {r[0] for r in records} | {r[2] for r in records}
    )
    try:
        norm = _pymisha.pm_normalize_chroms(all_chroms_raw)
        cmap = dict(zip(all_chroms_raw, norm, strict=False))
    except Exception:
        cmap = {c: c for c in all_chroms_raw}

    from .intervals import gintervals_all
    chrom_sizes_df = gintervals_all()
    known = set(chrom_sizes_df["chrom"].astype(str).tolist())

    # Build chrom ordering for canonical comparison
    chrom_order = {str(c): i for i, c in enumerate(chrom_sizes_df["chrom"])}

    # ------------------------------------------------------------------
    # 4. Canonical ordering + cis mirroring
    # ------------------------------------------------------------------
    # key = (chrom1, mid1, chrom2, mid2), value = summed count
    contact_map = {}

    for c1_raw, m1, c2_raw, m2, val in records:
        c1 = cmap.get(c1_raw, c1_raw)
        c2 = cmap.get(c2_raw, c2_raw)
        if c1 not in known or c2 not in known:
            continue

        # Canonical ordering: ensure chrom1 <= chrom2 (by chrom order)
        # For same chrom: ensure coord1 <= coord2
        o1 = chrom_order.get(c1, 0)
        o2 = chrom_order.get(c2, 0)
        if o1 > o2 or (o1 == o2 and m1 > m2):
            c1, c2 = c2, c1
            m1, m2 = m2, m1

        # Add the contact
        key = (c1, m1, c2, m2)
        if key in contact_map:
            if not allow_duplicates:
                raise ValueError(
                    f"Duplicate contact at ({c1}:{m1}, {c2}:{m2})"
                )
            contact_map[key] += val
        else:
            contact_map[key] = val

        # Mirror cis contacts (same chrom, different coordinate)
        if c1 == c2 and m1 != m2:
            mirror_key = (c1, m2, c2, m1)
            if mirror_key in contact_map:
                if not allow_duplicates:
                    raise ValueError(
                        f"Duplicate contact at ({c1}:{m2}, {c2}:{m1})"
                    )
                contact_map[mirror_key] += val
            else:
                contact_map[mirror_key] = val

    if not contact_map:
        raise ValueError("No valid contacts after normalization")

    # ------------------------------------------------------------------
    # 5. Build intervals DataFrame (POINTS: start=mid, end=mid+1) and values
    # ------------------------------------------------------------------
    rows = {
        "chrom1": [], "start1": [], "end1": [],
        "chrom2": [], "start2": [], "end2": [],
    }
    values = []
    for (c1, m1, c2, m2), val in sorted(contact_map.items()):
        rows["chrom1"].append(c1)
        rows["start1"].append(m1)
        rows["end1"].append(m1 + 1)
        rows["chrom2"].append(c2)
        rows["start2"].append(m2)
        rows["end2"].append(m2 + 1)
        values.append(val)

    intervals_df = pd.DataFrame(rows)
    values_arr = np.array(values, dtype=np.float32)

    # ------------------------------------------------------------------
    # 6. Delegate to gtrack_2d_create (which handles quad-tree + attributes)
    #    Note: _ensure_track_absent was already called above, so we bypass it
    #    by calling the internal machinery directly.
    # ------------------------------------------------------------------
    # We already validated, so call gtrack_2d_create directly.
    # But gtrack_2d_create also calls _ensure_track_absent, so we need to
    # build the track ourselves using the same internal logic.
    from ._quadtree import write_2d_track_file

    intervals_norm, chrom_sizes_df2 = _normalize_2d_intervals(intervals_df)
    chrom_size = {
        str(c): int(e)
        for c, e in zip(chrom_sizes_df2["chrom"], chrom_sizes_df2["end"], strict=False)
    }

    # Sort
    intervals_norm["_orig_idx"] = np.arange(len(intervals_norm))
    intervals_norm = intervals_norm.sort_values(
        ["chrom1", "chrom2", "start1", "start2"]
    ).reset_index(drop=True)
    orig_idx = intervals_norm["_orig_idx"].values
    intervals_norm = intervals_norm.drop(columns=["_orig_idx"])

    track_dir = _track_dir_for_create(track)
    created_new = not track_dir.exists()
    track_dir.mkdir(parents=True, exist_ok=True)

    try:
        for (c1, c2), group in intervals_norm.groupby(["chrom1", "chrom2"]):
            cs1 = chrom_size.get(str(c1))
            cs2 = chrom_size.get(str(c2))
            if cs1 is None or cs2 is None:
                continue
            arena = (0, 0, cs1, cs2)
            objs = []
            for _, row in group.iterrows():
                oi = row.name
                objs.append((
                    int(row["start1"]), int(row["start2"]),
                    float(values_arr[orig_idx[oi]])
                ))
            filename = os.path.join(str(track_dir), f"{c1}-{c2}")
            write_2d_track_file(filename, objs, arena, is_points=True)

        _pymisha.pm_dbreload()

        contacts_str = '", "'.join(contacts)
        fends_str = f'"{fends}"' if fends else "NULL"
        _set_created_attrs(
            track,
            description,
            f'gtrack.2d.import_contacts("{track}", description, '
            f'c("{contacts_str}"), {fends_str}, {allow_duplicates})',
        )
    except Exception:
        if created_new and track_dir.exists():
            shutil.rmtree(track_dir, ignore_errors=True)
        _pymisha.pm_dbreload()
        raise
