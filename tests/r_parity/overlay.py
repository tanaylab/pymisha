"""Writable copy-on-symlink overlay of R misha's read-only test DB.

R's regression suite for track-creating tests (``test-gtrack.create.R``,
``test-gtrack.smooth.R``, ...) runs against an *isolated* writable copy of the
test DB built by ``create_isolated_test_db`` (helper-test_db.R): small metadata
is copied, the large ``seq/`` directory is symlinked, and every fixture track is
symlinked into a *fresh writable* ``tracks/`` directory so new tracks can be
created alongside the read-only fixtures.

The wrinkle R's helper glosses over: misha's track namespace is *hierarchical*
(``test.fixedbin`` lives at ``tracks/test/fixedbin.track``), and on the indexed
test DB the group directories (``tracks/test``) are themselves read-only. A flat
"symlink each top-level entry" overlay therefore can't create ``test.tmptrack``
(it would need to write inside the symlinked-read-only ``tracks/test``). So we
mirror the *directory structure* recursively -- a real writable directory at
every group level -- and symlink only the leaf ``*.track`` / ``*.interv``
entries (and any plain files). Existing track data is shared by symlink (no
copy, no disk); new tracks/interval-sets can be created at any namespace level.
"""

from __future__ import annotations

import os
import shutil


def _mirror_entry(s: str, d: str) -> None:
    """Mirror one filesystem entry into the overlay.

    Real directories are recreated (writable) and recursed into; *dotfiles*
    (``.attributes``, ``.meta``, ``.colnames`` -- track metadata that
    ``gtrack.attr.set`` / ``gtrack.array.set_colnames`` / ``gtrack.var`` mutate)
    are copied so writes stay isolated to the overlay; all other files (bulk
    data: ``track.dat``/``track.idx``, per-chrom(-pair) files) are symlinked to
    the read-only source. This keeps disk and setup cheap while preventing any
    metadata write from reaching the shared source DB.
    """
    if os.path.isdir(s) and not os.path.islink(s):
        os.makedirs(d, exist_ok=True)
        for name in sorted(os.listdir(s)):
            _mirror_entry(os.path.join(s, name), os.path.join(d, name))
    elif os.path.basename(s).startswith("."):
        shutil.copy(s, d)
        os.chmod(d, 0o644)
    else:
        os.symlink(s, d)


def _mirror_tracks(src: str, dst: str) -> None:
    """Mirror the tracks tree: writable group/leaf dirs, symlinked bulk data.

    Group directories and ``*.track``/``*.interv`` leaf dirs alike become real
    writable directories (so new tracks/interval-sets *and* metadata edits stay
    in the overlay); only bulk data files are symlinked. R's
    ``create_isolated_test_db`` symlinks whole leaf dirs, which silently lets
    metadata writes (e.g. ``gtrack.array.set_colnames``) fall through to the
    shared source -- we avoid that.
    """
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        # R's create_isolated_test_db skips the scratch ``temp`` dir; we create a
        # fresh writable one (in build_overlay) instead of mirroring it.
        if name == "temp":
            continue
        _mirror_entry(os.path.join(src, name), os.path.join(dst, name))


def build_overlay(source_db: str, dest: str) -> str:
    """Build a writable overlay of ``source_db`` at ``dest`` and return ``dest``.

    Mirrors R's ``create_isolated_test_db``: copy small metadata, symlink the big
    ``seq/`` directory, copy ``intervs/`` and ``pssms/``, and recursively mirror
    ``tracks/`` (real group dirs + symlinked leaves). A fresh writable
    ``tracks/temp`` is created for misha scratch use.
    """
    os.makedirs(dest, exist_ok=True)

    # Small metadata files (copy + ensure writable).
    for fname in ("chrom_sizes.txt", ".ro_attributes"):
        s = os.path.join(source_db, fname)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(dest, fname))
            os.chmod(os.path.join(dest, fname), 0o644)

    # Large read-only sequence: symlink.
    seq_src = os.path.join(source_db, "seq")
    if os.path.exists(seq_src):
        os.symlink(seq_src, os.path.join(dest, "seq"))

    # Small, occasionally-written directories: copy (and make writable).
    for dname in ("intervs", "pssms"):
        s = os.path.join(source_db, dname)
        if os.path.isdir(s):
            d = os.path.join(dest, dname)
            shutil.copytree(s, d)
            for root, _dirs, files in os.walk(d):
                os.chmod(root, 0o755)
                for f in files:
                    os.chmod(os.path.join(root, f), 0o644)

    # Tracks: recursive structure mirror + leaf symlinks.
    _mirror_tracks(os.path.join(source_db, "tracks"), os.path.join(dest, "tracks"))
    # Fresh writable scratch dir.
    os.makedirs(os.path.join(dest, "tracks", "temp"), exist_ok=True)

    return dest
