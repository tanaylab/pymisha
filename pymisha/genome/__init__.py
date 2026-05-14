"""pymisha.genome - genome build and install subpackage.

This package implements the registry-based genome resolution and
backend dispatch used by ``gdb_build_genome`` and ``gdb_install_intervals``.

Public API (v0.1.46):
- ``gdb_build_genome`` - build a genome database from a recipe (``manual``,
  ``local``, ``s3``, ``ucsc-hub``, ``ncbi`` sequence-side backends;
  intervals via ``ucsc``, ``ucsc-hub``, or ``ncbi``).
- ``gdb_install_intervals`` - install annotation sets (``genes``, ``rmsk``,
  ``cgi``, ``cytoband``) into an existing groot from ``ucsc``,
  ``ucsc-hub``, or ``ncbi`` sources.
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path


def gdb_install_intervals(
    groot: str,
    source: str | dict,
    *,
    sets: tuple[str, ...] = ("genes", "rmsk", "cgi", "cytoband"),
    prefix: str = "",
    gene_sets: dict[str, str] | None = None,
    # Default swaps ``bestRefSeq`` for ``refGene`` because
    # ``bestRefSeq.gtf.gz`` is not consistently published by UCSC.
    gtf_priority: tuple[str, ...] = (
        "ncbiRefSeq", "refGene", "ensGene", "augustus", "xenoRefGene",
    ),
    overwrite: bool = False,
    registry: str | None = None,
    target_chroms: list[str] | None = None,
    target_lengths: list[int] | None = None,
    min_coverage: float = 1.0,
    # R 5.6.16+ default. The multi-pass rescue (length-fill, length-override,
    # name-override, synthetic target_chroms) is needed for any non-trivial
    # assembly; single-pass-only is opt-in.
    match_by_length: bool = True,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """Install annotation interval sets into an existing groot.

    Parameters
    ----------
    groot : str
        Path to an existing misha groot (must contain ``chrom_sizes.txt``
        and ``seq/``).
    source : str or dict
        Either a registry key (string) or a normalized recipe dict.
        v0.1.45 supports ``source='ucsc'`` and ``source='ucsc-hub'``;
        ``ncbi`` lands in v0.1.46. ``local`` and ``s3`` are rejected
        because those backends don't provide an asset fetcher.
    sets : tuple of str, default ("genes","rmsk","cgi","cytoband")
        Which annotation sets to install. Unknown values are ignored by
        the fetcher.
    prefix : str, default ""
        Prefix prepended to every installed set name.
    gene_sets : dict[str, str], optional
        Override gene-feature output names. Default is
        ``{"tss":"tss","exons":"exons","utr3":"utr3","utr5":"utr5"}``.
    gtf_priority : tuple of str
        Order in which to try GTF sources under UCSC's ``bigZips/genes/``.
    overwrite : bool, default False
        If True, pre-existing interval sets with the to-be-created names
        are deleted before installation.
    registry : str, optional
        Explicit registry YAML path forwarded to ``_resolve_genome``.
    target_chroms, target_lengths : list, optional
        Override the groot chrom names / lengths used for alias detection
        (otherwise read from ``<groot>/chrom_sizes.txt``).
    min_coverage : float, default 1.0
        Minimum bp-weighted coverage required for an alias column to be
        accepted by ``_resolve_chrom_alias``.
    match_by_length : bool, default True
        When True (R 5.6.16 default), run the four-pass alias rescue:
        length-fill missing canonical cells, length-override misnamed
        cells, name-override across-row collisions, and synthesize rows
        for unmapped ``target_chroms``. Set to False for strict
        single-pass detection.
    force : bool, default False
        If True, install only the available subset when the source can't
        provide every requested set (warns); otherwise raises.
    verbose : bool, default True
        Reserved for future progress output.

    Returns
    -------
    dict with keys ``installed`` (set_name -> row_count), ``skipped``
    (list of unavailable set names), ``failed`` (reserved; currently
    always ``[]``).
    """
    import pandas as pd

    import pymisha as pm

    from ._chrom_alias import _resolve_chrom_alias
    from ._install_sets import (
        _install_cgi,
        _install_cytoband,
        _install_genes,
        _install_rmsk,
    )
    from ._ucsc import _ucsc_fetch_assets
    from .registry import _normalize_recipe, _resolve_genome, _validate_recipe

    # 1. Resolve / normalize / validate the recipe.
    recipe = (
        _resolve_genome(source, registry)
        if isinstance(source, str)
        else _normalize_recipe(source)
    )
    _validate_recipe(recipe)

    # 2. Reject backends that don't supply assets.
    src = recipe["source"]
    if src in {"local", "s3"}:
        raise ValueError(
            f"source {src!r} does not provide assets; use ucsc/ucsc-hub/ncbi/manual"
        )

    # 3. v0.1.46 supports ucsc, ucsc-hub, and ncbi for asset installation.
    if src == "manual":
        raise NotImplementedError(
            "source 'manual' interval-asset installation has no scheduled release"
        )
    if src not in {"ucsc", "ucsc-hub", "ncbi"}:
        raise NotImplementedError(f"unsupported source {src!r}")

    # 4. Initialize the groot so the per-set installers see the right _GROOT.
    pm.gdb_init(str(groot))

    # 5. Fetch the requested assets.
    if src == "ucsc":
        assets = _ucsc_fetch_assets(recipe, sets, gtf_priority=gtf_priority)
    elif src == "ucsc-hub":
        from ._hub import _hub_fetch_assets
        assets = _hub_fetch_assets(recipe, sets, gtf_priority=gtf_priority)
    else:  # ncbi
        from ._ncbi import _ncbi_fetch_assets
        assets = _ncbi_fetch_assets(recipe, sets, gtf_priority=gtf_priority)

    # 6. Detect missing sets vs. the requested set list.
    available = {k for k in sets if assets.get(k) is not None}
    missing = set(sets) - available
    if missing:
        if force:
            warnings.warn(
                f"Skipping sets not provided by source {src!r}: {sorted(missing)}",
                stacklevel=2,
            )
        else:
            raise ValueError(
                f"Source {src!r} does not provide sets {sorted(missing)}; "
                "pass force=True to install only the available subset"
            )

    # 7. Read groot chroms + lengths from chrom_sizes.txt.
    if target_chroms is not None and target_lengths is not None:
        groot_chroms = list(target_chroms)
        groot_lengths = list(target_lengths)
    else:
        chrom_sizes = pd.read_csv(
            Path(groot) / "chrom_sizes.txt",
            sep="\t",
            header=None,
            names=["name", "length"],
            dtype={"name": str, "length": int},
        )
        groot_chroms = chrom_sizes["name"].tolist()
        groot_lengths = chrom_sizes["length"].tolist()

    # 8. Build the chrom-name translator.
    translator = _resolve_chrom_alias(
        assets.get("chrom_alias"),
        groot_chroms,
        groot_lengths,
        min_coverage=min_coverage,
        match_by_length=match_by_length,
    )

    # 9. Per-set installation.
    gene_set_map = dict(gene_sets) if gene_sets is not None else {
        "tss": "tss", "exons": "exons", "utr3": "utr3", "utr5": "utr5",
    }

    def _maybe_remove(names: list[str]) -> None:
        if not overwrite:
            return
        for name in names:
            try:
                if pm.gintervals_exists(name):
                    pm.gintervals_rm(name)
            except Exception:
                # Best-effort: don't let a failed pre-delete block install.
                pass

    installed: dict[str, int] = {}

    # Pre-compute the *possible* names (per the design doc the installer
    # doesn't know them in advance). Per-rmsk-class names will be computed
    # for the standard six classes; subsets that end up empty are still
    # listed for pre-deletion because the previous install may have
    # written them.
    _RMSK_CLASSES = ("SINE", "LINE", "LTR", "DNA", "Simple_repeat", "Low_complexity")

    if "genes" in available:
        gene_names = [f"{prefix}{v}" for v in gene_set_map.values()]
        _maybe_remove(gene_names)
        installed.update(
            _install_genes(
                assets["genes"], translator,
                gene_sets=gene_set_map, prefix=prefix,
            )
        )

    if "rmsk" in available:
        rmsk_names = [f"{prefix}rmsk"] + [f"{prefix}rmsk_{c}" for c in _RMSK_CLASSES]
        _maybe_remove(rmsk_names)
        installed.update(
            _install_rmsk(assets["rmsk"], translator, prefix=prefix)
        )

    if "cgi" in available:
        _maybe_remove([f"{prefix}cgi"])
        installed.update(
            _install_cgi(assets["cgi"], translator, prefix=prefix)
        )

    if "cytoband" in available:
        _maybe_remove([f"{prefix}cytoband"])
        installed.update(
            _install_cytoband(assets["cytoband"], translator, prefix=prefix)
        )

    # 10. Provenance file under <groot>/tracks/.misha_install.json (atomic).
    tracks_dir = Path(groot) / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "source": src,
        "recipe": {k: v for k, v in recipe.items() if not k.startswith("_")},
        "sets_installed": list(installed.keys()),
        "row_counts": installed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gtf_source": assets.get("genes_source"),
        "skipped_sets": sorted(missing),
    }
    import json
    prov_path = tracks_dir / ".misha_install.json"
    tmp_path = tracks_dir / f".misha_install.json.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, sort_keys=False)
    os.replace(tmp_path, prov_path)

    return {"installed": installed, "skipped": sorted(missing), "failed": []}


def gdb_build_genome(
    name: str,
    *,
    path: str | None = None,
    registry: str | None = None,
    sets: tuple[str, ...] = (),
    prefix: str = "",
    gene_sets: dict[str, str] | None = None,
    gtf_priority: tuple[str, ...] = (
        "ncbiRefSeq", "refGene", "ensGene", "augustus", "xenoRefGene",
    ),
    chrom_naming: str | None = None,
    target_chroms: list[str] | None = None,
    target_lengths: list[int] | None = None,
    min_coverage: float = 1.0,
    match_by_length: bool = True,
    format: str | None = None,
    verbose: bool = True,
) -> None:
    """Build a genome database from a recipe.

    Resolves ``name`` against the registry chain (explicit arg, then
    ``$PYMISHA_GENOME_REGISTRY``, then ``./misha.yaml``, then the bundled
    ``recipes.yaml``), materializes the sequence-side of a misha groot at
    ``path`` (or ``./<name>`` if ``path`` is None), binds the resulting
    groot via ``gdb_init``, and (when ``sets`` is non-empty) installs the
    requested annotation interval sets.

    Parameters
    ----------
    name : str
        Genome name (registry key) such as ``"hg38"``. A normalized recipe
        dict may also be passed in place of a name for ad hoc builds.
    path : str, optional
        Target groot directory. Defaults to ``./<name>`` in the current
        working directory. For ``manual`` recipes you almost always want
        to pass an explicit ``path``.
    registry : str, optional
        Highest-priority registry YAML file. Must exist if provided.
    sets : tuple of str, optional
        Interval sets to install after the sequence is in place. v0.1.44
        supports ``("genes","rmsk","cgi","cytoband")`` via the ``ucsc``
        backend. Default ``()`` skips the install step.
    prefix : str, default ""
        Prefix forwarded to ``gdb_install_intervals``.
    gene_sets : dict, optional
        Forwarded to ``gdb_install_intervals``.
    gtf_priority : tuple of str
        Forwarded to ``gdb_install_intervals``.
    chrom_naming : str, optional
        Accepted for forward-compatibility; ignored in v0.1.44.
    target_chroms, target_lengths : list, optional
        Forwarded to ``gdb_install_intervals``.
    min_coverage : float, default 1.0
        Forwarded to ``gdb_install_intervals``.
    match_by_length : bool, default True
        Forwarded to ``gdb_install_intervals``. See its docstring for the
        four-pass rescue algorithm.
    format : str, optional
        Database format hint forwarded to FASTA backends
        (``"indexed"`` or ``"per-chromosome"``).
    verbose : bool, default True
        If True, backends may emit progress output.

    Notes
    -----
    v0.1.44 supports manual / local / s3 sequence-side backends and
    ``ucsc`` for ``gdb_install_intervals``. ``ucsc-hub`` lands in
    v0.1.45 and ``ncbi`` in v0.1.46.
    """
    del chrom_naming  # forward-compat shim; unused in v0.1.44

    from ._build_seq import _build_seq
    from .registry import _normalize_recipe, _resolve_genome, _validate_recipe

    recipe = (
        _resolve_genome(name, registry)
        if isinstance(name, str)
        else _normalize_recipe(name)
    )
    _validate_recipe(recipe)

    groot = path or name
    _build_seq(recipe, groot, format=format, verbose=verbose)

    from pymisha.db import gdb_init
    gdb_init(str(groot))

    if sets:
        gdb_install_intervals(
            str(groot),
            source=recipe,
            sets=sets,
            prefix=prefix,
            gene_sets=gene_sets,
            gtf_priority=gtf_priority,
            target_chroms=target_chroms,
            target_lengths=target_lengths,
            min_coverage=min_coverage,
            match_by_length=match_by_length,
            registry=registry,
            verbose=verbose,
        )


__all__ = ["gdb_build_genome", "gdb_install_intervals"]
