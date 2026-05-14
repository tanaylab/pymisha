"""Registry resolver for genome recipes.

Resolves a genome name (e.g. ``"hg38"``) to a recipe dict by walking a
prioritized chain of YAML registry files:

1. ``explicit`` argument (if provided; must exist)
2. ``$PYMISHA_GENOME_REGISTRY`` environment variable (if set and the file exists)
3. ``./misha.yaml`` in the current working directory (if present)
4. Bundled ``pymisha/genome/recipes.yaml`` (lowest-priority fallback)

The first layer that contains an entry for ``name`` wins.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

_BUNDLED = Path(__file__).parent / "recipes.yaml"

_VALID_SOURCES = {"ucsc", "ucsc-hub", "ncbi", "manual", "local", "s3"}


def _load_yaml(path: str | Path) -> dict:
    """Load a registry YAML file and return its ``genome`` mapping.

    Raises ``ValueError`` for unsupported schema versions.
    """
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if "version" in data and data["version"] != 1:
        raise ValueError(f"Unsupported registry schema version: {data['version']}")
    return data.get("genome", {}) or {}


def _registry_chain(explicit: str | None) -> list[Path]:
    """Build the ordered list of registry files to consult."""
    chain: list[Path] = []
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"registry path does not exist: {p}")
        chain.append(p)
    env = os.environ.get("PYMISHA_GENOME_REGISTRY")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            chain.append(p)
    proj = Path.cwd() / "misha.yaml"
    if proj.exists():
        chain.append(proj)
    chain.append(_BUNDLED)
    return chain


def _resolve_genome(name: str, registry: str | None = None) -> dict:
    """Return the recipe dict for ``name`` from the first matching registry layer."""
    for layer in _registry_chain(registry):
        entries = _load_yaml(layer)
        if name in entries:
            raw = entries[name]
            return _normalize_recipe(raw, layer)
    raise KeyError(f"genome '{name}' not in any registry layer")


def _normalize_recipe(raw, layer=None) -> dict:
    """Normalize a raw recipe entry into a dict with a valid ``source`` field.

    Bare-string entries are shorthand for ``{source: local, path: <string>}``
    (tgdb/misha.yaml compatibility).
    """
    recipe = {"source": "local", "path": raw} if isinstance(raw, str) else dict(raw)
    if "source" not in recipe:
        raise ValueError(f"recipe missing 'source' field: {recipe}")
    src = recipe["source"]
    if src not in _VALID_SOURCES:
        raise ValueError(f"unknown source: {src!r}")
    if layer is not None:
        recipe["_layer"] = str(layer)
    return recipe


def _validate_recipe(recipe: dict) -> None:
    """Validate that a recipe has the per-source required fields.

    Raises ``ValueError`` listing the missing fields if any are absent.
    """
    src = recipe["source"]
    required = {
        "ucsc": {"assembly"},
        "ucsc-hub": {"accession"},
        "ncbi": {"accession"},
        "manual": {"content"},
        "local": {"path"},
        "s3": {"name"},
    }.get(src, set())
    missing = required - set(recipe)
    if missing:
        raise ValueError(f"recipe for source={src!r} missing fields: {sorted(missing)}")
